# Task capability evaluation

Coding evaluation uses the shared benchmark runner and dashboard with sandboxed single-completion pass@1. Incorrect, invalid, truncated, and timed-out submissions stay in the denominator; sandbox infrastructure failures abort the run. Existing non-coding suite hashes are preserved. Coding suite hashes include executable tests and entry points. Functional code agreement compares typed output hashes on matching ordered input traces, with valid-pair coverage. Pass/fail agreement is labeled separately. See [coding-task.md](coding-task.md).

Evaluate AG News, SQuAD 1.1, and GSM8K through the suites frozen by `distill tasks build`. Each task has its own prompt contract, output limit, test split, and adapters. See [paper-tasks.md](paper-tasks.md) for dataset pins, source grouping, development selection, and query budgets.

## Run an evaluation

```bash
uv run --extra eval distill tasks build --task ag_news
uv run --extra train distill benchmark run --config configs/eval.ag_news.toml --stage before
uv run --extra train distill benchmark run --config configs/eval.ag_news.toml --stage after
```

Build once; existing builds are immutable. Use `development-suite.json` under a separate experiment name for recipe selection, and `test-suite.json` only after the protocol is frozen. The paper's primary metrics are AG News accuracy, SQuAD token F1, and GSM8K numeric exact match. SQuAD exact match is secondary.

| Task | Final split | Output limit | Evaluation config |
| --- | --- | --- | --- |
| AG News | Official test | 16 tokens | `configs/eval.ag_news.toml` |
| SQuAD 1.1 | Official validation | 64 tokens | `configs/eval.squad.toml` |
| GSM8K | Official test | 256 tokens | `configs/eval.gsm8k.toml` |

Each config defines `teacher`, `student_before`, `teacher_after`, and `student_after`. `--stage before` runs the initial victim and unmodified student; `after` reevaluates the victim and scores the adapted student; `all` runs all four. Victim reevaluation measures serving drift. It is never a trained teacher checkpoint.

Set `student_after.adapter_path` to the exact generated adapter directory, such as `artifacts/tasks/ag_news/s42-b2000/attacker` or `gold`. Give every task, seed, budget, defense condition, and student kind a separate experiment identity and output directory. The four slots compare one adapted student with its base and victim; they do not automatically compare all defense conditions or calculate a study-wide extraction ratio.

Remote identities in configs must match the actual server, including quantization. Local model and tokenizer revisions must be immutable Hub commits. Both student slots must share base identity, precision, quantization, EOS, and template settings. Adapter loading verifies manifest identity and training hashes. List other training files in `training_paths`; adapter evaluation adds the files from its manifest. Missing, changed, or overlapping held-out training files fail before inference.

Requests are sequential, temperature zero, fixed-seed, and non-thinking. Every slot uses the task's same output limit. Evaluation has no hidden retries or replacement samples. An interrupted sample journal is incomplete and cannot produce a completed score report.

## Scores and denominators

A completed run contains every frozen item exactly once. Failed API calls, invalid outputs, and truncations remain in the denominator and contribute zero to task quality.

| Field or metric | Definition |
| --- | --- |
| `n` | All frozen task items in the run or category slice |
| `correct` | Number of correct labels or exact matches |
| `score` | `100 × correct / n`; for SQuAD this is secondary exact match |
| `ci95` | 95% Wilson interval for the binary `score` |
| `f1` | SQuAD mean best-gold token F1, multiplied by 100 |
| `f1_ci95` | Normal-approximation 95% interval for the mean SQuAD token F1 |
| `primary_metric`, `primary_score` | `accuracy`, `token_f1` for SQuAD, or `pass_at_1` for coding, and the primary score over all items |
| `statuses` | Counts of `ok`, `invalid`, `truncated`, and `error`, plus `timeout` for coding; counts sum to `n` |
| `latency_p50` to `latency_p99`, `tokens_per_second`, `token_coverage` | See [timing and provenance](#timing-and-provenance) |
| Category slices | The same fields restricted to the frozen task category |

AG News accepts exactly one of World, Sports, Business, and Sci/Tech, ignoring case, surrounding quotes, and a final period. SQuAD lowercases, removes punctuation and articles, and collapses whitespace. Exact match accepts any normalized gold answer; token F1 takes the best overlap against all gold answers. GSM8K accepts a bare valid number or the number after the last `####`; it normalizes decimals and valid thousands separators without guessing from other numbers in a rationale.

Blank or unparseable responses are invalid. `finish_reason="length"` is truncated. Other nonterminal or missing finish reasons and thinking traces are invalid. API failures are errors and store no provider error body or credentials. A wrong but parseable answer is `ok` with `correct=false`.

The dashboard uses SQuAD F1 for task-score charts and summaries and exact match for the binary outcome breakdown. It does not combine the tasks into an overall model-quality score or report teacher-gap closure.

## Paired comparisons

Comparisons require identical item IDs and compatible experiment name, suite hash, scorer version, generation settings, and simulation flag. The latest completed run fills each slot. Missing phases stay unavailable.

| Comparison | First and second slots |
| --- | --- |
| `student_gain` | Base student, adapted student |
| `teacher_drift` | Initial victim, reevaluated victim |
| `teacher_student_before` | Initial victim, base student |
| `teacher_student_after` | Reevaluated victim, adapted student |

`delta_pp` is the mean binary correctness difference in percentage points. `primary_delta_pp` is the primary-metric difference. `primary_ci95`, also exported as `ci95`, uses 2,000 paired bootstrap draws with seed 42 over source documents: SQuAD article titles resample together and other items resample alone. `resampling_unit` and `clusters` record this. The four binary counts `improved`, `regressed`, `both_correct`, and `both_wrong` sum to the matched item count. SQuAD also records `f1_delta_pp` and `f1_ci95` for paired token-F1 differences.

`answer_agreement` is the percentage of pairs with equal parsed predictions among pairs where both responses have status `ok`. `agreement_numerator` and `agreement_n` are its counts and `agreement_coverage` is `agreement_n / n`. Agreement includes shared wrong answers. Coding reports `functional_agreement`: equal typed outputs on identical ordered calls among valid trace pairs, with its numerator, denominator, coverage, and status, plus `pass_fail_agreement`, the share of items where both models pass or both fail.

These intervals describe test-item sampling, clustered by source document, conditional on the trained models. They do not cover training-seed variability or independent campaigns. Keep per-seed results and paired defended/undefended conditions; do not treat these descriptive item intervals as a completed study analysis.

Before final defense testing, establish positive undefended capability transfer on development data. Evaluate defended students on the same frozen task suite and retain raw base, undefended, and defended scores. Define and validate `R_D` against the manuscript's chosen references; do not substitute broad teacher-gap closure. Matched-volume and matched-prefix controls must use released response tokens to separate supervision quantity from selective withholding.

## Timing and provenance

`latency_seconds` times one awaited generation call, including network, queueing, prefill, and decoding, and excluding model loading and scoring. `latency_p50`, `latency_p90`, `latency_p95`, and `latency_p99` use non-error calls and linear interpolation. Invalid and truncated outputs still contribute observed timing. `tokens_per_second` divides known output-token counts by generation time over those same non-error calls. `token_coverage` is their fraction of all items. Provider token accounting may differ.

These sequential evaluations do not measure concurrent serving throughput, time to first token, GPU peak memory, policy waiting costs, or proof-of-work effort. Measure those separately for paper claims.

Each output directory contains a verified `suite-SHA256.json`, completed run JSON files, and append-only sample journals. A run records model and generation config, source revision, artifact and training hashes, packages, platform, and JJ code revision. A comparison exports the suite, selected runs, per-task scores, category slices, paired metrics, warnings, and responses. Existing non-coding suites retain their hashes. Coding suites include their tests and entry points, and this benchmark runner invokes the sandboxed scorer. Generation latency excludes sandbox scoring time.

Exact training-overlap checks cannot detect paraphrases or pretraining contamination. A declared remote model revision requires independent deployment verification. Dataset-only adapter skip checks do not establish a complete training fingerprint.

## Dashboard and exports

```bash
bun install --cwd apps/dashboard
bun run --cwd apps/dashboard build
uv run distill benchmark dashboard --results-dir outputs/tasks/ag_news
uv run distill benchmark report --results-dir outputs/tasks/ag_news
uv run distill benchmark report --results-dir outputs/tasks/ag_news \
	--comparison COMPARISON_ID --output outputs/tasks/ag_news-report.json
```

Open `http://127.0.0.1:8002`. Select a task's output directory when starting the server. The page has task scores, paired changes, exact-match outcome counts, agreement, speed, answer inspection, provenance, and a separate harness-traffic view. API endpoints are `GET /api/experiments` and `GET /api/reports/COMPARISON_ID`. Vite development proxies `/api` to port 8002.

To preview without models, use `distill benchmark demo`, then serve `outputs/tasks-demo/ag_news`. Pass `--task squad --output-dir outputs/tasks-demo/squad` for a SQuAD simulation. Simulations are labeled, stored separately, and refuse to overwrite existing artifacts. They are not paper results.

## Metrics reference

Every benchmark in the repository, with its artifact and fields. Scores and margins are percentage points. Failed, invalid, truncated, refused, and timed-out outcomes stay in their denominators unless a field says otherwise. A *distribution* below means `n`, `mean`, `max`, `p50`, `p90`, `p95`, and `p99` by linear interpolation, with `pq_supported` false below 1/(1 − q) observations, where the value is an interpolation toward the maximum rather than a tail estimate.

### Task capability

Each evaluation run of one model slot on one frozen suite. See [scores and denominators](#scores-and-denominators) and [paired comparisons](#paired-comparisons).

| Field | Meaning |
| --- | --- |
| `primary_metric`, `primary_score` | `accuracy` (AG News, GSM8K), `token_f1` (SQuAD), or `pass_at_1` (coding), and its score over all items |
| `n`, `correct`, `score`, `ci95` | Items, binary correct count, binary score, and Wilson 95% interval; for SQuAD the binary score is exact match |
| `f1`, `f1_ci95` | SQuAD mean best-gold token F1 and its normal-approximation interval |
| `statuses` | Counts of `ok`, `invalid`, `truncated`, `error`, and for coding `timeout` |
| `latency_p50`, `latency_p90`, `latency_p95`, `latency_p99` | Generation-call latency over non-error calls |
| `tokens_per_second`, `token_coverage` | Output tokens per generation second, and the share of items with known token counts |
| Category slices | The same fields per frozen category |

Paired fields for `student_gain`, `teacher_drift`, `teacher_student_before`, and `teacher_student_after`:

| Field | Meaning |
| --- | --- |
| `delta_pp` | Mean binary correctness difference |
| `primary_delta_pp`, `primary_ci95` (`ci95`) | Primary-metric difference and its 2,000-draw paired bootstrap interval |
| `resampling_unit`, `clusters` | `source_document`: SQuAD article titles stay together, other items resample alone |
| `improved`, `regressed`, `both_correct`, `both_wrong` | Binary outcome pairs; they sum to `n` |
| `answer_agreement`, `agreement_numerator`, `agreement_n`, `agreement_coverage` | Equal parsed predictions among pairs where both responses are `ok`; coverage is `agreement_n / n` |
| `f1_delta_pp`, `f1_ci95` | SQuAD token-F1 difference and interval |
| `functional_agreement`, `functional_agreement_numerator`, `functional_agreement_n`, `functional_agreement_coverage`, `functional_agreement_status` | Coding: equal typed outputs on identical ordered calls among valid trace pairs |
| `pass_fail_agreement` | Coding: share of items where both models pass or both fail |

### Development qualification and defense effect

Shared study, per domain, from `recipe.json` and `results.json`. See [the multi-domain study](multi-domain-study.md#outcomes-and-controls).

| Field | Meaning |
| --- | --- |
| `trials[].gains`, `ranks`, `mean_rank` | Each recipe's development gain per domain, its per-domain rank, and the mean rank used for selection |
| `qualification.DOMAIN`: `base_score`, `score`, `gain_pp`, `gain_ci95_pp`, `minimum_gain_pp`, `passed`, `clusters` | Selected recipe on the untouched qualification half; a domain passes when the interval's lower bound exceeds the margin |
| `informative_domains`, `uninformative_domains`, `transfer_gate_passed` | Domains that qualified, those that did not, and whether all did |
| `base_score`, `undefended_score`, `defended_score`, `teacher_score`, `teacher_after_score` | Raw scores for the paired conditions |
| `undefended_gain`, `defended_gain`, `undefended_gain_ci95_pp`, `defended_gain_ci95_pp` | Gains over the base student and their intervals |
| `gain_removed`, `gain_removed_ci95_pp`, `positive_removal_supported` | Undefended minus defended score, its interval, and whether the interval excludes zero |
| `retained_gain_fraction`, `gain_removed_fraction` and their `_ci95` | Ratios to the undefended gain, reported only when `ratio_valid`; otherwise `ratio_invalid_reason` |
| `controls.matched-volume`, `controls.matched-prefix` | Control students' scores and `target_released_tokens`, `achieved_released_tokens`, `residual_tokens` per domain |

The separate-task report adds `per_seed`, `mean_and_sd`, `seed_sd_sources`, `missing_pairs`, `budget_curves`, `cost_to_target` with `first_observed_budget` or a not-reached status, `target_score`, `defended_minus_undefended`, and `defended_minus_control`.

### Campaign collection and released supervision

`campaign.json` per campaign, overall and per domain, and the `http` field of each result row.

| Field | Meaning |
| --- | --- |
| `queries`, `received`, `blocked`, `failed`, `deadline`, `challenge`, `pending` | Attempt outcomes, all charged to the campaign allowance |
| `released_output_tokens`, `proof_requests` | Received output tokens and proof-redemption submissions |
| `levels.Lk`: `J`, `Q`, `reached`, `censored`, `attribution_complete` | Attempt index at first reach of level k, responses received before it, and censoring |
| `termination_by_account`, `elapsed_seconds`, `hashes`, `challenge_seconds` | Why each account stopped sending, wall time, and proof-of-work work |
| `supervision.DOMAIN`: `released_responses`, `released_output_tokens`, `prepared_examples`, `prepared_output_tokens` | Supervision received and what survived preparation |

### Campaign timing

`campaign-timing.json` per campaign. Times are seconds from campaign start. Full definitions are in [campaign timing](multi-domain-study.md#campaign-timing).

| Field | Meaning |
| --- | --- |
| `levels.Lk`: `attempt_index`, `seconds`, `account`, `domain`, `received_before` | First reach of each level, who triggered it, and responses and tokens received per domain before it |
| `accounts.ACCOUNT`: `attempts`, `received`, `first_level_seconds`, `first_challenge_seconds`, `first_block_seconds`, `stopped`, `stop_seconds`, `final_blocked_attempts` | Per-account escalation and observed stop |
| `accounts_blocked`, `accounts_stopped`, `campaign_stopped`, `campaign_stop_seconds`, `received_after_first_block` | Whole-campaign stopping |
| `delivery_windows` | Received responses per 60-second window |
| `latency_seconds` (overall, `by_action`, `by_domain`), `end_to_end_seconds`, `challenge_solve_seconds`, `challenge_hashes`, `schedule_delay_seconds` | Distributions |

### Legitimate-user cost

Live benign utility per defense, `utility.json`:

| Field | Meaning |
| --- | --- |
| `submitted`, `served`, `completion_rate` | Requests sent, served responses, and their ratio |
| `served_only_quality`, `unconditional_utility` | Mean task score over served responses, and over every submission with zero for refusal, timeout, or abandonment |
| `throughput`, `latency_p50` to `latency_p99`, `latency_seconds`, `service_and_queue_seconds` | Served responses per second and latency distributions from intended arrival and from dispatch |
| `strata.domain`, `strata.domain_profile`, `strata.client_type` | Submitted, served, quality, utility, and latency distribution per stratum |
| `ttft`, `server_peak_memory` | Recorded as unavailable: the endpoint is non-streaming and memory is not sampled |

Held-out benign replay per defense, `benign-test/ARM.json`: `reached` and `reached_rate` for L1 to L4 over all account horizons, `reached_rate_scored` over horizons with a complete window, `unscored_horizons`, `implemented_levels`, and `profiles` with the same rates per domain profile and client type. The separate-task report names these `false_escalations`, `false_escalation_rate`, and `false_escalation_rate_scored`, and adds `served_only_quality_change` and `unconditional_utility_change`.

### Calibration

`calibration.json` and the study's `calibration-adequacy.json`. See [the harness](harness.md#calibrate).

| Field | Meaning |
| --- | --- |
| `counts` | Fitting requests, `score_calibration_windows`, `score_calibration_independent_windows`, `score_calibration_stride`, policy-calibration accounts, horizons, scored and unscored horizons, and encoder truncations per role |
| `thresholds.ACCUMULATOR`, `baselines.quota`, `baselines.rate_per_minute` | `thresholds`, `achieved` rates, `units`, and `upper95` Clopper-Pearson bounds per level |
| Adequacy: `passed`, `reasons`, `empirical_p_value_floor`, `nonoverlapping_score_windows`, `scored_policy_horizons`, `minimum_score_windows`, `minimum_policy_horizons` | Whether the calibration resolves the declared targets |

### Traffic smoke benchmark

Before and after harness deployment on the frozen traffic suite. It cannot produce detector false positives with its default suite. See [harness traffic benchmarks](harness-benchmarks.md#measures-and-denominators).

| Field | Meaning |
| --- | --- |
| `n`, `succeeded`, `success_percent`, `success_ci95` | Per genuine and extraction class |
| `noised`, `noised_percent`, `useful_succeeded`, `useful_success_percent` | Keyed-noise responses, known only with the harness log |
| `blocked`, `confirmed_block_percent`, `block_percent`, `throttled`, `rate_limited`, `decision_coverage`, `statuses` | Harness decisions and outcomes |
| `false_positive_count`, `false_positive_percent` | Confirmed genuine blocks |
| `latency_p50` to `latency_p99`, `schedule_delay_p95`, `end_to_end_p50` to `end_to_end_p99`, `challenge_seconds` | Latency and delay |
| `successful_requests_per_second`, `successful_output_tokens_per_second`, `token_coverage` | Delivered throughput |
| Per actor: `first_throttle_seconds`, `first_block_seconds`, `successes_before_first_block`, `terminal_block_start_seconds`, `terminal_blocked_requests`, `observed_stopped`, `observed_stop_seconds`, `observed_stop_blocked_requests`, `successful_completions_after_terminal_block` | Escalation and stopping |
| Comparison: `success_delta_pp`, `latency_p95_delta_seconds`, `throughput_reduction_percent`, windows | Before versus after |

### Harness overhead

`harness bench` reports `n` and p50, p90, p95, and p99 milliseconds for `embedding_per_request`, `mmd_per_completed_window`, `risk_update_per_completed_window`, `noise_per_request`, `admit_per_scoring_trigger_request`, `admit_per_other_request`, and `admit_per_request`, plus `peak_policy_memory_bytes`. It measures the harness's own units, not matched-load serving overhead. See [the harness](harness.md#serving-cost).

### Planning checks

| Command | Fields |
| --- | --- |
| `distill gateway power` | Per suite: `items`, `clusters`, `score_half_width_pp_at_50`, `minimum_detectable_pp`, `clustered_minimum_detectable_pp_bound`, `resolves_minimum_gain`, `resolves_minimum_gain_clustered_bound`. See [suite resolution](multi-domain-study.md#suite-resolution) |
| `distill tasks pilot` | `statuses`, `length_fraction`, `length_denominator`, `served_model`, and `decision` for the GSM8K and coding output caps |

Not measured anywhere yet: time to first token, server memory under load, and a matched-load serving comparison.

## Software checks

```bash
just test
just check
just dashboard-check
just dashboard-build
```

The tests use fixed fixtures and mocked generation. They check task construction, budget nesting, collection/resume, usability review, scores and denominators, comparisons, provenance, and dashboard behavior. They do not establish real GPU training, capability gains, or defense effectiveness.
