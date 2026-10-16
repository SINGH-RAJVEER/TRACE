# Task capability evaluation

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
| `statuses` | Counts of `ok`, `invalid`, `truncated`, and `error`; counts sum to `n` |
| Category slices | The same fields restricted to the frozen task category |

AG News accepts exactly one of World, Sports, Business, and Sci/Tech, ignoring case, surrounding quotes, and a final period. SQuAD lowercases, removes punctuation and articles, and collapses whitespace. Exact match accepts any normalized gold answer; token F1 takes the best overlap against all gold answers. GSM8K accepts a bare valid number or the number after the last `####`; it normalizes decimals and valid thousands separators without guessing from other numbers in a rationale.

Blank or unparseable responses are invalid. `finish_reason="length"` is truncated. Other nonterminal or missing finish reasons and thinking traces are invalid. API failures are errors and store no provider error body or credentials. A wrong but parseable answer is `ok` with `correct=false`.

The dashboard uses SQuAD F1 for task-score charts and summaries and exact match for the binary outcome breakdown. It does not combine the three tasks into an overall model-quality score or report teacher-gap closure.

## Paired comparisons

Comparisons require identical item IDs and compatible experiment name, suite hash, scorer version, generation settings, and simulation flag. The latest completed run fills each slot. Missing phases stay unavailable.

| Comparison | First and second slots |
| --- | --- |
| `student_gain` | Base student, adapted student |
| `teacher_drift` | Initial victim, reevaluated victim |
| `teacher_student_before` | Initial victim, base student |
| `teacher_student_after` | Reevaluated victim, adapted student |

`delta_pp` is the mean binary correctness difference in percentage points. `ci95` uses 2,000 paired item bootstrap draws with seed 42. The four binary counts `improved`, `regressed`, `both_correct`, and `both_wrong` sum to the matched item count. SQuAD also records `f1_delta_pp` and `f1_ci95` for paired token-F1 differences.

`answer_agreement` is the percentage of pairs with equal parsed predictions among pairs where both predictions exist. `agreement_n` is that denominator; report coverage against the full item count. Agreement includes shared wrong answers and can include parsed content from a response rejected for its finish reason, so inspect statuses as well.

Current intervals describe item sampling, not training-seed variability, independent campaigns, or source-document dependence. The paper's required source-group and campaign resampling remains on the implementation checklist. Keep per-seed results and paired defended/undefended conditions; do not treat these descriptive item intervals as a completed study analysis.

Before final defense testing, establish positive undefended capability transfer on development data. Evaluate defended students on the same frozen task suite and retain raw base, undefended, and defended scores. Define and validate `R_D` against the manuscript's chosen references; do not substitute broad teacher-gap closure. Matched-volume and matched-prefix controls must use released response tokens to separate supervision quantity from selective withholding.

## Timing and provenance

`latency_seconds` times one awaited generation call, including network, queueing, prefill, and decoding, and excluding model loading and scoring. `latency_p50` and `latency_p95` use non-error calls and linear interpolation. Invalid and truncated outputs still contribute observed timing. `tokens_per_second` divides known output-token counts by generation time over those same non-error calls. `token_coverage` is their fraction of all items. Provider token accounting may differ.

These sequential evaluations do not measure concurrent serving throughput, time to first token, GPU peak memory, policy waiting costs, or proof-of-work effort. Measure those separately for paper claims.

Each output directory contains a verified `suite-SHA256.json`, completed run JSON files, and append-only sample journals. A run records model and generation config, source revision, artifact and training hashes, packages, platform, and JJ code revision. A comparison exports the suite, selected runs, per-task scores, category slices, paired metrics, warnings, and responses. Frozen v1 task suites retain empty legacy schema fields solely to preserve existing suite hashes; there is no code-execution scorer.

Exact training-overlap checks cannot detect paraphrases or pretraining contamination. A declared remote model revision requires independent deployment verification. Dataset-only adapter skip checks do not establish a complete training fingerprint; the implementation checklist tracks that gap.

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

## Software checks

```bash
just test
just check
just dashboard-check
just dashboard-build
```

The tests use fixed fixtures and mocked generation. They check task construction, budget nesting, collection/resume, usability review, scores and denominators, comparisons, provenance, and dashboard behavior. They do not establish real GPU training, capability gains, or defense effectiveness.
