# Extraction harness

The [coding task](coding-task.md) has its own detector serialization, calibration pools, and sandboxed legitimate-utility scoring. Its fixed service instruction is removed before embedding; the specification remains, while the signature wrapper is excluded. Coding uses L4 refusal. Coding deception is rejected explicitly, and another task's calibration must not be reused. Existing non-coding serialization is unchanged.

`packages/harness` implements the paper's detector and access policy as a library: query embeddings, the window MMD score, benign-only calibration, accumulated account evidence, and the graduated actions of Table I. Policies take the account, request text, and time as arguments and never read the clock. `apps/harness-api` serves the same decision code over HTTP in front of the teacher API, and a campaign simulator can call it directly.

## Install

```bash
uv sync --all-packages              # library, CLI, and tests; no encoder
uv sync --all-packages --extra embed  # adds PyTorch and Transformers for the BGE encoder
```

The `harness` command comes with the package. Replays with precomputed embeddings and the tests do not need the `embed` extra.

## Pipeline

| Step | Module | Paper |
| --- | --- | --- |
| Embed the detector serialization of the current user turn (`serialization.py`: the AG News article, the SQuAD question then passage, or the GSM8K problem, without fixed service instructions) with `BAAI/bge-small-en-v1.5` at revision `5c38ec7`, CLS pooling, unit length, 512-token limit with truncations counted | `encoder.py` | III-B |
| Score a window of `w_a` queries: unbiased squared MMD, kernel = mean of four RBF kernels at 0.5, 1, 2, and 4 times the median-distance bandwidth, averaged over 20 fixed reference windows, negative estimates kept | `mmd.py` | Eq. 1 |
| Turn a score into its upper-tail rank among benign calibration scores | `calibration.py` | Eq. 3 |
| Accumulate evidence per account and map it to a level | `accumulators.py`, `calibration.py` | Eq. 4, III-C |
| Apply the level's action, hold the level for the horizon, and reset after 24 hours or 10,000 attempts | `policy.py` | Table I, III-C, III-D |
| Issue and verify SHA-256 proof-of-work challenges | `pow.py` | Table I, L3 |

A completed window includes the current request and is scored before the victim runs, so it can change that request's action. Windows do not overlap. Until a window completes, the previous level applies, and a new account starts at L0.

## Accumulators

All rules read the same window p-values, so they can be compared with everything else fixed. The calibration artifact holds thresholds for each one.

| Name | Rule | Purpose |
| --- | --- | --- |
| `instantaneous` | Current window only, `-log p` | Primary baseline: no memory |
| `paper` | `r = max(0, 0.9 r - log p - 1)` | The draft's Eq. 4 |
| `paper_no_memory` | Same with decay 0 | Ablation |
| `paper_no_decay` | Same with decay 1 | Ablation |
| `e_process` | `log M += log e`, with `e = 0.5 p^(-0.5)` | Session e-process; theoretical thresholds `log(1/alpha)` hold only if each window's p-value is valid given the session so far |
| `e_cusum` | `log R = max(0, log R) + log e` | E-process with a floor, so a long clean history cannot bank negative evidence against a later attack |

`paper` is the default because it is the rule the draft specifies. The choice of primary rule is still open.

Thresholds are calibrated by default: for each benign policy-calibration horizon, take the highest statistic reached, then choose each level's threshold so that at most its target share of horizons strictly exceed it. Targets are 10%, 5%, 2%, and 1% for L1 to L4. Thresholds are nested, and the artifact records the achieved rate next to each target. When the sample is too small for a target, the threshold becomes the largest observed value and the achieved rate is lower than the target; it is never extrapolated. `--thresholds ville` uses `log(1/alpha)` instead, for `e_process` only.

## Actions

| Level | Action |
| --- | --- |
| L0 | Serve |
| L1 | Serve; the decision record carries the level as an alert |
| L2 | Serve no sooner than 1 second after the account's previous admitted request; excess requests queue |
| L3 | L2 spacing plus a proof-of-work challenge per response: 20 leading zero bits, bound to the account, a SHA-256 digest of the whole request, and a 5-minute expiry. Each challenge is accepted once |
| L4 | Refuse for the rest of the horizon, including the request that triggers L4. With `HARNESS_L4_ACTION=deceive`, L3's spacing and challenge with keyed noise in place of the answer, built without a teacher call (see below) |

Each `Decision` carries the action, the earliest release time, any challenge, and telemetry: the level, whether a window completed, its p-value, the statistic, the deception flag, and witness attribution. `client_view()` returns only what a client should see: the action, release time, and challenge. Risk values stay hidden from the attacker, as the threat model requires.

**Witness attribution.** When a completed window raises an account's level, the decision lists the five requests in that window with the largest MMD witness, `f(x) = mean k(x, window) - mean k(x, references)`, as request digests with their values. These are the queries that most separate the window from benign traffic. They explain an escalation to the experimenter and never feed back into decisions.

`AllowPolicy`, `QuotaPolicy`, and `RatePolicy` share the same interface. They are the undefended service and the two block-only baselines. Calibration fits the quota, in requests per horizon, and the per-minute rate cutoff to the L4 target on the same benign horizons. The rate cutoff needs request times.

## Noise at L4

With `HARNESS_L4_ACTION=deceive`, the protected API does not refuse an account at L4. It keeps answering, with noise in the task's output format, and never calls the teacher for those answers. Each noise response costs about 7 to 25 microseconds of CPU, measured on 5,000 task prompts per format. A flagged extractor keeps spending its budget on data with no value for training, and the service spends nothing to serve it. The default, `refuse`, is the paper's policy; noise is an unreported option.

The approach adapts adaptive misinformation (Kariyappa and Qureshi, CVPR 2020) and keyed response changes with watermarking (DAWN, Szyller et al., ACM MM 2021) to a text API. Probability-perturbation defenses such as prediction poisoning do not apply, because the service returns text rather than class probabilities. The paper's main policy refuses and leaves served answers unchanged, so report noise as a separate arm.

- **Same shape as L3.** An account at L4 keeps L3's one-second spacing and proof-of-work challenge. The decision's hidden `deceive` flag marks responses to replace, and `client_view()` never shows it.
- **Same metadata as a real response.** Noise responses copy the provider, model, and endpoint of the most recent real response, with fresh identifiers in the same format. They finish with `stop` and carry the same usage fields as every other response. Before returning, the server waits for a delay drawn from recent real teacher latencies. Waiting costs no compute, and it keeps an instant reply from revealing the noise.
- **Pseudo-random, keyed by request.** Noise is seeded with HMAC-SHA256 of the secret and the request. It looks random, but repeated and Sybil queries for the same request get the same answer, so re-asking does not expose it, and the attacker cannot predict it. `deception_rate` replaces a keyed share of an L4 account's responses and serves the rest truthfully; the default replaces all of them.
- **Persisted key.** Without `HARNESS_DECEPTION_KEY`, the server reads `HARNESS_DECEPTION_KEY_FILE`, `artifacts/harness/deception.key` by default, or creates it on first start with owner-only permissions. A stable key keeps noise consistent across restarts and makes the watermark test possible.
- **Only at L4,** the level calibrated so that at most 1% of benign horizons reach it. A falsely flagged user receives noise without being told, which is worse than a refusal. Report this as silently corrupted legitimate responses, inside unconditional utility.

| `HARNESS_TASK` | Noise |
| --- | --- |
| `ag_news` | A keyed label among `World`, `Sports`, `Business`, and `Sci/Tech` |
| `squad` | A one- to four-word span from one sentence of the passage |
| `gsm8k` | Two to four arithmetic steps chaining numbers from the problem, never negative, ending in `#### N`. The answer is fixed by the key and request; a small `max_output_tokens` only drops early steps |
| `text`, the default | One to three sentences of words drawn from the request and common function words, within the output limit |

The noise is cheap and therefore crude. An attacker who checks answers against another model, or checks GSM8K arithmetic against the problem, can still detect it. Because noise does not depend on the true answer, about a quarter of AG News noise labels are correct by chance.

### Watermark test

The same key tests whether a suspect model learned the noise:

```bash
uv run harness watermark --task ag_news --probes probes.jsonl
```

Each probe line holds `request` and `suspect_response`. Agreement is the share of probes where the suspect gives the keyed noise answer. The p-value ranks that agreement among the agreements under 999 unrelated control keys, so it does not assume how often a clean model happens to give a particular answer. The key comes from `HARNESS_DECEPTION_KEY` or the server's key file and never appears in output files. `harness replay --l4-action deceive` keys noise the same way, or by a key for that run only, and counts deceived decisions.

The evaluation still needs a verifying attacker that re-asks a share of prompts from clean accounts, where answers are truthful, and a filtering attacker that drops answers its own student disagrees with. Defense-penetrating attacks such as D-DAE (Chen et al., IEEE S&P 2023) motivate both tests.

## Aggregate monitor

All requests also join one stream across accounts. Each complete window of 1,500 requests is scored against 20 aggregate reference windows of the same size and raises an alert when its score exceeds the 95th percentile of benign aggregate windows. Alerts are logged only. They never change an account's level, because an aggregate shift does not identify the accounts responsible. Each alert records how many accounts the window held and the largest account's share, for Sybil analysis.

Calibration builds the benign aggregate windows from the score-calibration requests in arrival order: by time when every request has one, otherwise by taking one request from each account in turn. It skips the monitor and records `aggregate_skipped` when the data cannot support it. The 20 reference windows need 30,000 fitting requests, and the AG News fitting pool holds 21,558. Lower `aggregate_reference_windows` in the configuration to fit smaller pools, and expect few benign aggregate windows and a coarse threshold in either case.

## Calibrate

Calibration reads three disjoint benign files of JSON lines with `account`, `text`, and optional `time` in seconds. Requests of an account must appear in order.

```bash
uv run --extra embed harness calibrate \
	--fitting fitting.jsonl \
	--score-calibration score_calibration.jsonl \
	--policy-calibration policy_calibration.jsonl \
	--task ag_news \
	--output-dir artifacts/harness/ag_news
```

| Input | Role |
| --- | --- |
| Fitting | Draws the 20 reference windows and sets the kernel bandwidth from up to 2,000 other fitting queries |
| Score calibration | Each account's complete, non-overlapping windows form the null score distribution |
| Policy calibration | Each account horizon's highest statistic sets the level thresholds and the baseline cutoffs |

`--task` selects the detector serialization, `detector-v2`, recorded in the artifact with per-role truncation counts. Each request's `text` is the full user turn; calibration, replay, and the API strip the same fixed task instructions before embedding. A turn that does not follow its task's template is embedded whole. The default `text` embeds whole turns for services without a template. The API refuses to start when `HARNESS_TASK` differs from the calibration's task, and task artifacts without the current serialization must be recalibrated.

The output directory holds `calibration.json` and `references.npy`. The JSON records the configuration, encoder identity and truncation count, bandwidth, sorted calibration scores, thresholds and achieved rates per accumulator, baseline cutoffs, counts, and SHA-256 hashes of each input and of the reference array. Loading verifies the reference hash. An existing calibration is never overwritten.

## Replay

```bash
uv run --extra embed harness replay \
	--calibration artifacts/harness/ag_news \
	--requests benign_test.jsonl \
	--output outputs/harness/benign-paper.jsonl \
	--policy mmd --accumulator paper
```

Replay sends requests through one policy in file order, writing one decision per line. Requests without a time are spaced one second apart. On benign traffic, the summary's `reached_rate` per level is the account-level false-escalation rate FPR_ℓ. Replay counts challenges but does not solve them, so it measures detection, not what a solving client receives. `--policy` also accepts `allow`, `quota`, and `rate`.

## Accounts

`harness pool-accounts` turns a prompt pool, such as `data/tasks/ag_news/legitimate/fitting.jsonl`, into 500-request accounts after a seeded shuffle of its rows (`--seed`, default 42). Task pools keep source order, which groups SQuAD questions by article, so slicing them unshuffled would make every account and window single-topic. It exists for plumbing checks only. Those accounts are neither realistic clients nor independent samples. Real calibration needs legitimate accounts built with per-client topic concentration and timing, and enough of them: the paper asks for about 3,000 independent benign calibration accounts and 3,000 test accounts per endpoint for the 1% target.

## Smoke run

On 2026-10-16, the AG News pools from `distill tasks build` were cut into 500-request accounts and calibrated on CPU in about 10 minutes: 6,000 fitting requests, 107 score-calibration windows, and 22 policy-calibration accounts. Replaying 22 held-out benign accounts and one 2,000-request extraction account ran every policy end to end.

The numbers describe the construction, not the detector:

- **Quota.** Every benign account had exactly 500 requests, so the fitted quota was 500. It blocked the extractor at its 501st request by construction.
- **MMD.** The extractor's first window received the smallest possible p-value, 1/108. That account is a contiguous slice of the AG News training file, whose row order appears to carry its own distribution shift. Its label mix was balanced, so labels are not the cause.
- **Sample size.** With 22 calibration accounts, the 2% and 1% targets cannot be met. The L3 and L4 thresholds fell back to the largest benign value, with an achieved rate of 0.

## Serving cost

`harness bench` times the harness's own work unit by unit, for the paper's Table X. It replays up to `--limit` requests from a request file as one account and reports p50 and p95 milliseconds per unit, along with the peak Python heap allocated while admitting them. A window-scoring cost is never reported as a per-request latency, so admission is split into requests that complete a window and requests that do not.

```bash
uv run --extra embed harness bench \
	--calibration artifacts/harness/ag_news \
	--requests benign_test.jsonl \
	--task ag_news --output outputs/harness/bench-ag_news.json
```

`--no-encode` skips the encoder and uses random unit embeddings, which leave the other units' costs unchanged. Victim inference is excluded; measure it with the traffic benchmark at the same load and hardware.

Measured on 2026-10-16 on an AMD Ryzen AI 9 HX 370 with no GPU, using the AG News smoke calibration (100-request windows, 20 reference windows, 107 calibration windows) and 1,000 benign requests:

| Unit | p50 (ms) | p95 (ms) |
| --- | --- | --- |
| Embedding per request, BGE on CPU | 14.3 | 19.0 |
| MMD per completed window | 5.9 | 6.5 |
| Risk update per completed window | 0.006 | 0.013 |
| Noise per request, AG News | 0.010 | 0.013 |
| Noise per request, GSM8K | 0.039 | 0.048 |
| Admission per scoring-trigger request | 6.2 | 6.4 |
| Admission per other request | 0.005 | 0.008 |

Peak policy memory was about 1.2 MB. Embedding dominates the harness's per-request cost on CPU, and a GPU would change that row most. Each figure comes from one run on one machine, without warm-up control, so treat it as indicative.

## Protected API

`apps/harness-api` puts the harness in front of the teacher API and serves the same contract, `POST /v1/teacher/generate`. The task collector, the evaluator, and the traffic benchmark can point at it unchanged.

```bash
uv sync --all-packages --extra embed
HARNESS_CALIBRATION_DIR=artifacts/harness/ag_news \
HARNESS_ACCOUNTS_FILE=accounts.json \
HARNESS_TEACHER_URL=http://127.0.0.1:8000 \
	uv run --extra embed harness-api
```

It listens on port 8003 by default. Account state, open challenges, and the aggregate stream live in the process, so the service always runs as one worker.

**Accounts.** `HARNESS_ACCOUNTS_FILE` is a JSON object mapping the SHA-256 hex digest of each API key to an account ID. Clients send `Authorization: Bearer KEY`, and an unknown key receives HTTP 401. For tests only, `HARNESS_TRUST_ACTOR_HEADER=true` takes the account from `X-Benchmark-Actor`, which any client can set.

**Responses.** Every response carries `X-Harness-Decision`:

| Level and action | HTTP | Decision |
| --- | --- | --- |
| L0, L1 | 200 | `allow` |
| L2 | 200, after the server-side wait | `throttle` |
| L3, and L4 with `HARNESS_L4_ACTION=deceive` (noise) | 428 with a challenge; 200 once answered | `throttle` |
| L4 by default | 403 `blocked` | `block` |
| Wrong, reused, expired, or rebound proof | 403 `challenge_failed` | `throttle` |

A 428 body carries the challenge's `nonce`, `prefix`, `difficulty`, and `expires_at`. The client finds a counter such that SHA-256 of `prefix` followed by the decimal counter has `difficulty` leading zero bits, then resends the identical request with `X-Harness-Proof: NONCE:COUNTER`. The proof is bound to the whole request: changing any field, such as `max_output_tokens`, voids it. Answering a challenge does not count as a new attempt. A failed proof consumes the challenge, so the client must send the request again.

**Consistent metadata.** Output token counts are recounted with `HARNESS_TOKENIZER_MODEL` and `HARNESS_TOKENIZER_REVISION` when set, and omitted from every response otherwise. Input token counts are omitted from every response, because a noise response has none to report. The tests check that a noise response matches an ordinary L3 response in status, headers, fields, usage, model, and identifier format, and that it makes no teacher call.

**Decision log.** `HARNESS_DECISION_LOG`, `outputs/harness/decisions.jsonl` by default, records every decision with level, statistic, p-value, deception flag, and witness, along with redemptions, noise responses, and aggregate alerts. Requests appear as SHA-256 digests, never as text. Risk values never appear in responses. When a client sends `X-Benchmark-Request`, a test correlation ID, the log records it with each event; it never affects a decision. The traffic benchmark uses it to learn which responses were noise.

**Test clock.** With `HARNESS_TEST_CLOCK=true`, the server reads the time from `X-Harness-Time` and returns `X-Harness-Release-At` instead of waiting, so simulated schedules replay without real delays.

Other settings, all in `apps/harness-api/src/harness_api/settings.py`: `HARNESS_POLICY` (`mmd`, `allow`, `quota`, or `rate`), `HARNESS_ACCUMULATOR`, `HARNESS_THRESHOLDS`, `HARNESS_L4_ACTION`, `HARNESS_DECEPTION_RATE`, `HARNESS_DECEPTION_KEY`, `HARNESS_DECEPTION_KEY_FILE`, `HARNESS_TASK`, `HARNESS_POW_DIFFICULTY`, `HARNESS_MIN_INTERVAL_SECONDS`, and `HARNESS_PORT`.

## Not yet implemented

- Realistic legitimate-account construction, which calibration needs before its thresholds mean anything
- Adaptive campaign schedules beyond the fixed-order extractor in `distillation/campaign.py`, including the adaptive tests against deception
