# Harness traffic benchmarks

This is an HTTP smoke test, not detector evidence. Its default suite has 48 generic-chat genuine requests, 12 per actor, so no genuine account completes a 100-request window and no detector false positive can occur. Report false escalation, detection timing, and benign cost from the shared study instead: held-out benign replay, live benign utility, and each campaign's `campaign-timing.json` ([multi-domain-study.md](multi-domain-study.md#campaign-timing)).

This benchmark compares the unprotected teacher before harness deployment with the same teacher behind the harness afterward. It measures whether genuine users still receive responses while extraction attempts slow down, receive rate limits, and eventually get blocked. Student training and the existing four-slot model-quality comparison are separate experiments.

The benchmark creates a held-out synthetic traffic suite. It does not add anything to the task prompt pools, collection ledgers, training data, or task capability suites. Only explicitly running `benchmark traffic run` sends its requests. The harness is in `packages/harness`, served over HTTP by `apps/harness-api` ([harness.md](harness.md)).

## Freeze the traffic

```bash
devenv shell
uv run distill benchmark traffic freeze \
	--output data/traffic/suite.json \
	--genuine-count 48 \
	--attack-count 144 \
	--duration-seconds 60 \
	--seed 42
```

The default workload has four genuine actors and two extraction actors. Genuine requests cover writing, summarization, translation, learning, planning, coding, structured output, and legitimate batch work. Repeat requests and batch tasks exercise plausible false positives. Extraction actors submit varied tasks to collect responses, with one increasing its request rate and the other maintaining a steady rate. Their ground-truth labels describe the simulated intent and request pattern. They do not imply that an isolated prompt is unsafe or distinguishable from a genuine question.

The freezer records each actor, ground-truth label, scenario, request, category, prompt, and scheduled time. Its SHA-256 covers the full suite and seed. Repeating a freeze with the same parameters produces the same hash. Existing suites cannot be overwritten. Change the path when changing traffic composition, timing, or counts. The examples are authored simulations with repeated templates; results describe this workload, not a measured production-user distribution or all possible extraction strategies. Broaden and review the frozen scenarios before drawing final research conclusions.

## Run before and after deployment

Review `configs/traffic.example.toml`. Both phases declare the same teacher model, immutable model revision, generation settings, timeout, concurrency ceiling, and workload. Targets can have different URLs, credentials, and service revisions. Record the actual baseline and harness version in `service_revision`. Remote weight identity is declared by the operator; returned model IDs are also recorded, but the benchmark cannot attest to server weights.

```bash
# Serve the unprotected teacher separately, then record its actual baseline.
uv run distill benchmark traffic run \
	--config configs/traffic.example.toml \
	--phase before

# Later, deploy the harness, reset its test-account state, and replay the same suite.
uv run distill benchmark traffic run \
	--config configs/traffic.example.toml \
	--phase after
```

Use the same model, model configuration, hardware, authentication principals, and initial account state in both phases. Reset test-account detection history and quotas before each replay; retain state within a replay so accumulation can trigger throttling and blocking. Repeated runs create new IDs. The report uses the latest completed run for each phase within a compatible comparison. A different suite, model declaration, generation setting, timeout, concurrency ceiling, window size, or experiment creates a separate comparison. Endpoints, credentials, and service revisions can change between phases.

Both `teacher_api` and `openai` backends use real HTTP requests. For `teacher_api`, `base_url` is the server root. For `openai`, include the API prefix such as `/v1` in `base_url`; the runner appends `chat/completions`. Both request the same non-thinking generation contract used elsewhere in this project.

Requests follow their frozen offsets independently of earlier responses. A concurrency ceiling limits in-flight requests; excess requests wait locally. Each result records scheduled, actual start, and completion times. The report warns when p95 client scheduling delay exceeds 100 ms. Increase client capacity or reduce the offered load before interpreting a replay with substantial local queueing. Timeouts include HTTP processing after dispatch; end-to-end delay also includes local queueing. The runner does not retry, honor `Retry-After` by rescheduling, or replace failures, so the denominator and offered workload stay fixed. Answering a proof-of-work challenge is part of the harness protocol rather than a retry: the runner solves it in a process pool, resends the same request with the proof, and records both exchanges as one sample with the solving time in `challenge_seconds`. Interrupted sample journals are retained but are not complete reports.

`X-Benchmark-Actor` carries an opaque actor ID. The runner never sends the ground-truth label or scenario. This header is a test correlation ID, not trustworthy authentication. For policies keyed by authenticated users or API keys, configure `actor_key_envs` under both targets, using the actor IDs from the frozen suite and separate environment-provided test credentials. A shared `api_key_env` is also supported, but shared credentials may cause the harness to treat all actors as one principal. The current teacher wrapper does not implement per-actor detection. Credential values and response bodies are excluded from result artifacts.

## Decision telemetry

`apps/harness-api` implements this contract; see [harness.md](harness.md#protected-api). A challenged attempt returns HTTP 428 with `throttle` and a challenge body. The runner answers it, and the final response decides the sample's outcome. The harness omits output token counts unless `HARNESS_TOKENIZER_MODEL` is set, so compare delivered token throughput across phases only with that setting. Request throughput is unaffected.

### Noise ground truth

With `HARNESS_L4_ACTION=deceive`, the harness answers L4 accounts with keyed noise that looks like an ordinary throttled response, so a client cannot tell it apart and a noise response counts as a success. To measure what each class actually received, the runner sends `X-Benchmark-Request: RUN_ID:REQUEST_ID` with every request, and the harness writes that ID into its experimenter-only decision log. Set `harness_log` on the target to that log's path:

```toml
[targets.after]
backend = "teacher_api"
base_url = "http://127.0.0.1:8003"
service_revision = "harness-v1"
harness_log = "outputs/harness/decisions.jsonl"
```

After the replay, the runner reads the log and marks each sample's `noised` field. A missing log file stops the run rather than reporting unknown noise as none. Without `harness_log`, `noised` stays empty and the noise fields below are unavailable. The log must be readable from the machine running the benchmark; copy it over first when the harness runs elsewhere.

The protected endpoint must return `X-Harness-Decision` on every response, including rejected requests. Accepted values are `allow`, `throttle`, and `block`.

| Decision or response | Recorded meaning |
| --- | --- |
| `allow` | The harness allowed the request. A successful completion still requires a valid response body and finish reason. |
| `throttle` with a successful response | The harness delayed or otherwise throttled the request but still returned a complete response. It contributes to both successful and throttled counts. |
| `throttle` with HTTP 429 | The harness rate-limited this attempt. It contributes to throttled and HTTP 429 counts and does not succeed. |
| `block` | A confirmed harness rejection, regardless of HTTP status. A genuine request with this decision is a false positive. |
| HTTP 429 without a decision | A rate-limited attempt with unknown attribution. It is not counted as a confirmed harness block. |
| Other failure without a decision | An unknown decision plus a separate HTTP, timeout, transport, or response-format failure. HTTP 401, 403, and 5xx alone are not proof of harness detection. |

A silent application refusal inside a nominally successful response is not a confirmed harness block unless the endpoint supplies `block`. If the future harness uses another telemetry protocol, adapt this benchmark's response adapter before reporting its false positives. Without complete decision coverage for genuine requests, the false-positive percentage is unavailable; confirmed block counts and their percentage remain observable lower bounds. The existing unprotected teacher has no decision header, so it can establish genuine success and throughput while leaving decision attribution unknown. Do not fill in an assumed zero false-positive rate for missing telemetry.

## Measures and denominators

Every frozen request remains in its class denominator, including failures, invalid responses, rate limits, and confirmed blocks. The expected unprotected genuine-request success may be 100%, but the runner measures it and never hardcodes it.

| Field in each phase's summary | Formula or rule |
| --- | --- |
| `genuine.n`, `distillation.n` | Number of frozen requests with that actor label. |
| `succeeded`, `success_percent` | Count and `100 × successful / n`. Success requires a 2xx response, a nonempty completion, a normal stopping reason, and no confirmed block. Truncated or malformed responses fail. Answer correctness is outside this traffic metric. |
| `success_ci95` | Wilson interval over the class's binary request outcomes. Repeated prompts and correlated actor behavior limit the independent-trial interpretation. |
| `genuine.false_positive_count` | Number of genuine requests with a confirmed `block` decision. |
| `genuine.false_positive_percent` | `100 × confirmed genuine blocks / genuine.n`, available only with complete genuine decision coverage. |
| `confirmed_block_percent` | `100 × confirmed blocks / n`, including when decision coverage is incomplete. In that case it is a lower bound. |
| `throttled`, `rate_limited` | Explicit `throttle` decision count and HTTP 429 outcome count. They can overlap; neither is added to the confirmed block count. |
| `decision_coverage` | Fraction of requests with a recognized harness decision. |
| `statuses` | Disjoint outcome counts for successful, blocked, rate-limited, HTTP error, timeout, transport error, and invalid responses. |
| `latency_p50`, `latency_p90`, `latency_p95`, `latency_p99` | Dispatch-to-completion latency among successful requests, in seconds. Includes server delays, generation, and network time. |
| `schedule_delay_p95`, `end_to_end_p50` to `end_to_end_p99` | p95 scheduled-to-dispatch delay, and p50/p90/p95/p99 scheduled-to-completion delay, over all requests. |
| `challenge_seconds` | Distribution of proof-of-work solving time over challenged requests, with p50/p90/p95/p99. |
| `successful_requests_per_second` | Successful requests divided by the greater of the frozen replay duration and last completion offset. Failed and blocked attempts still consume observation time. |
| `noised`, `noised_percent` | Successful responses that were keyed noise, and `100 × noised / n`. Available only when the harness log covers every successful response. For genuine requests these are silent corruptions: false positives that were never shown as blocks. |
| `useful_succeeded`, `useful_success_percent` | Successful responses that carried the teacher's answer: `succeeded − noised`, and `100 × useful_succeeded / n`. For extraction requests this is the supervision the attacker actually obtained. |
| `successful_output_tokens_per_second` | Known output tokens from successful responses divided by the same observation time. Missing counts are not invented; `token_coverage` gives the fraction of successful responses with known counts. |

The comparison reports after-minus-before success in percentage points, p95 successful-latency change, and `100 × (1 − after throughput / before throughput)` for delivered requests per second. Throughput reduction is unavailable when the baseline delivers no responses and can be negative if throughput increases. Interpret attack throughput alongside explicit decisions and operational errors; an unavailable server is not evidence of a working detector. Genuine throttling is recorded separately from false-positive blocking because it can harm users even if their requests eventually succeed.

Each extraction actor also reports time to its first confirmed throttle and block response, successful completions before the first confirmed block, and the final consecutive sequence of blocked requests. Times are seconds from the common replay start. `observed_stopped` requires at least two requests in that final blocked sequence to start after all successful in-flight completions have drained. `observed_stop_seconds` gives the first of those request starts. Initial blocking can precede that observed stop while older responses are still arriving; `successful_completions_after_terminal_block` counts those late deliveries. One rejection, HTTP failures, or a final successful completion cannot establish a stop. This is an observed stop within the finite workload; it does not prove permanent prevention or measure the quality of a student trained on leaked responses.

Completion-time windows show delivered requests per second throughout the replay, including zero-delivery windows. They expose the progression from successful extraction to delay and blocking while genuine requests continue. Run-level class and actor summaries use every request exactly once; window summaries group those same outcomes by completion time.

## Reports and dashboard

```bash
# List compatible comparisons, then use the returned ID to export one.
uv run distill benchmark traffic report
uv run distill benchmark traffic report --comparison COMPARISON_ID --output outputs/traffic-report.json

bun run --cwd apps/dashboard build
uv run distill benchmark dashboard \
	--traffic-results-dir outputs/traffic-benchmarks
```

Open the local dashboard and select **Harness traffic**. It works independently of whether model-quality results exist. The view shows per-class before/after counts, genuine false positives, throttling, error outcomes, delivered-throughput timelines, latency, and per-actor blocking progression. Export JSON preserves the suite, run configuration, per-request outcomes, and calculated summaries for review. The dashboard reads artifacts and never sends inference requests.

To preview with explicitly simulated outcomes:

```bash
uv run distill benchmark traffic demo
uv run distill benchmark dashboard --traffic-results-dir outputs/traffic-demo
```

Demo outcomes include a genuine false positive and a scripted extraction slowdown followed by blocking. They are labeled as simulated throughout and are not measurements of a real harness.
