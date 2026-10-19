# Paper tasks

The default launchers now use [one shared gateway and student](multi-domain-study.md#run-the-shared-pipeline). This page describes the retained separate-task workflow selected with `--mode tasks`; its budgets and adapter counts apply only to that mode.

The [multi-domain study protocol](multi-domain-study.md) records the requested transition to one shared gateway and one student per campaign. `distill gateway benign` and `distill gateway schedule` now generate its first-phase inputs. The task builders and separate-task execution described below remain the current collection/training path; gateway execution is not implemented yet.

The paper's distillation study (Section IV-B) measures extraction on narrow task APIs. `distill tasks` supports four tasks: AG News classification, SQuAD extractive QA, GSM8K generative math, and Python coding. Each task is a separate service with its own prompt format, output limit, ledger, frozen suites, and evaluation config, so results are reported per task as Table VI requires.

This is the separate-task distillation workflow. The study measures task-specific extraction and legitimate utility; it makes no general-chat capability claim.

## Tasks

| Task | Dataset | Pool split | Final test split | Output limit | Primary metric |
| --- | --- | --- | --- | --- | --- |
| `ag_news` | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) at `eb185aa` | `train` | `test` (7,600) | 16 tokens | Label accuracy |
| `squad` | [rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad) `plain_text` at `7b6d24c` | `train` | `validation` (10,570) | 64 tokens | Token F1, with exact match |
| `gsm8k` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main` at `740312a` | `train` | `test` (1,319) | 256 tokens | Numeric exact match |
| `coding` | Pinned NVIDIA OpenCodeInstruct, complete qualified corpus | `train` | 539 reference-qualified items: 163 from pinned `evalplus/humanevalplus` and 376 from pinned `evalplus/mbppplus` | 512 tokens, provisional | Executable pass@1 |

Revisions are pinned in `TASKS` in `packages/distillation/src/distillation/tasks.py`. The AG News dataset card lists its license as unknown; check its terms before redistributing derived data. SQuAD is CC BY-SA 4.0 and GSM8K is MIT.

Every request for a task uses one fixed serialization, in the attacker pool, the legitimate pools, the development suite, and the test suite alike:

- **AG News:** an instruction naming the four topics, then `Article: <text>`. The service must answer with exactly one of `World`, `Sports`, `Business`, or `Sci/Tech`, ignoring case, surrounding quotes, and a final period. Any other output is scored invalid and stays in the denominator.
- **SQuAD:** an instruction to answer with a short span, then `Passage: <context>` and `Question: <question>`. Contexts longer than 6,000 characters are cut at a word boundary before querying, training, and evaluation alike; the manifest counts how often this happens. No pinned row currently needs it. Answer positions are never used to filter or window passages.
- **GSM8K:** the problem followed by a fixed instruction, ending with `####` and the final number.

Coding builds stream every source row, validate reference solutions in the sandbox, decontaminate against the pinned external suite, and then allocate roles. Collection inputs contain specifications and signatures only. Private preparation files retain reference code and tests. See [coding-task.md](coding-task.md) for pins, qualification artifacts, sandbox limits, and unmeasured study gates.

## Run the whole study

```bash
just distill --mode tasks
just distill --mode tasks --tasks ag_news --seeds 42 --budgets 500,2000
just distill --mode tasks --teacher-url http://TEACHER_IP:8000
```

Each task keeps its ledger in `data/tasks/TASK/ledger.jsonl` and its adapters in `artifacts/tasks/TASK/`. Budgets, seeds, and attacker versus gold-reference training remain separate within each task.

Set `TEACHER_TERMS_REVIEW_ID` in `.env` first. The command then:

1. Builds each task that has no `data/tasks/TASK/manifest.json`, and reuses existing builds.
2. Resolves the default `--budgets full` to each task's attacker count in `manifest.json`. Explicit numeric budgets larger than a task's attacker pool are dropped and logged.
3. Writes the prompts of every seed's largest budget. Starts vLLM and the teacher API on loopback, or uses `--teacher-url`, and collects those prompts into the task ledger at the task's own output limit. Then it stops a locally started teacher. If the ledgers already hold every planned prompt, the teacher is not started.
4. For every task, seed, and budget, writes the budget's ledger subset, reviews it without gold labels, prepares `train.jsonl`, and trains the attacker student. It also trains the gold-label reference on `gold-train.jsonl`, unless you pass `--no-gold`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--tasks` | `ag_news,squad,gsm8k,coding` | Tasks to run; math and coding are included by default |
| `--seeds` | `42,1,20` | Campaign seeds; each orders its own attacker queries and, by default, sets the training seed |
| `--training-seed` | none | Fixed training seed for every campaign seed, so seed spread reflects query order only; reports record which sources the seed SD covers |
| `--budgets` | `full` | Use the whole attacker pool per task; pass comma-separated integers for smaller query budgets B_A |
| `--gold/--no-gold` | `--gold` | Also train the gold-label reference student |
| `--config` | `configs/train.task.toml` | Recipe copied into each run with that run's dataset, output directory, and seed |
| `--teacher-url` | none | Collect from a running teacher API instead of starting vLLM here |
| `--port`, `--upstream-url`, `--concurrency` | as in [run-scripts.md](run-scripts.md#task-study-operation) | Local teacher and collection settings |

Each run directory `data/tasks/TASK/runs/sSEED-bBUDGET/` holds the generated `train.attacker.toml` and `train.gold.toml`. The adapters go to `artifacts/tasks/TASK/sSEED-bBUDGET/attacker` and `.../gold`. The command is resumable: collection resumes by `example_id`, training resumes from its last checkpoint, and an adapter whose `training-manifest.json` matches the current dataset is skipped. Both `just distill --mode tasks` and `just start --mode tasks` default to AG News, SQuAD, GSM8K, and coding with three seeds. The default uses each task's whole attacker pool: 53,893 AG News queries, 40,356 SQuAD queries, and 3,389 GSM8K queries with the pinned builds. All three seeds use the same complete pool in different orders. Coding uses the complete qualified attacker pool, whose size is measured during the build. This plans 12 undefended attacker adapters and 12 gold references, excluding defended students, recipe trials, and matched controls. GSM8K supplies the math capability distillation corpus; its 256-token output cap remains provisional until the implemented cap pilot passes against the actual teacher.

`--budgets 500,2000,8000` selects subsets, skipping 8,000 for GSM8K when it is selected. `start` freezes the resolved numeric plan in `budgets.json`; reporting uses those task-specific values. Existing protocol options remain immutable. Protected campaigns retain one attempt limit and deadline across all policy resets, so full-pool runs can span multiple 10,000-attempt horizons without replenishing their campaign allowance. The resulting reports describe separate task studies, not shared-gateway execution.

`just start --mode tasks` collects each seed's undefended baseline through the same single-account campaign envelope as the defended arms: concurrency 1, one attempt per prompt, the attempt budget, and the 24-hour deadline. Training requires a response for every planned prompt. A baseline that loses prompts to teacher failures or an interruption resumes on its original clock, retrying only prompts without a response, for up to three passes; rerunning with the same `--run-dir` resumes it again. Every retry is recorded in the events and counted as `retried_queries`. A baseline stopped by the deadline, or refused by a policy, stops the study with status `undefended_baseline_incomplete`. A deadline means the envelope cannot carry that budget, and defended campaigns at the same budget would be cut off too, so pass smaller `--budgets`. Protected campaigns still refuse to resume.

Evaluation is not part of this command; see [Evaluate](#evaluate). The sections below describe the individual steps it runs.

## Build

```bash
uv sync --all-packages --extra eval
uv run --extra eval distill tasks build --task ag_news
uv run --extra eval distill tasks build --task squad
uv run --extra eval distill tasks build --task gsm8k
```

Each build writes to `data/tasks/TASK/`, which is gitignored, and refuses to overwrite an existing build. Choose a new `--output-dir` when changing the protocol.

| File | Contents |
| --- | --- |
| `attacker.jsonl` | The attacker-query pool, in the collector's prompt format with split `train` |
| `gold-answers.jsonl` | Gold completions for the attacker pool, used only to train the gold-label reference student |
| `legitimate/fitting.jsonl`, `score_calibration.jsonl`, `policy_calibration.jsonl`, `benign_test.jsonl` | The legitimate-workload pool for the future harness, marked `test` so `distill collect` skips them by default |
| `development-suite.json` | A frozen, gold-scored suite of up to 1,000 development items for selecting the training recipe |
| `test-suite.json` | The frozen official evaluation split, used only for final scoring |
| `manifest.json` | Dataset pins, seed, per-role counts and source groups, filter reasons, truncation counts, and SHA-256 hashes of every file |

### Splits

Each source document is assigned once by a seeded hash of its group. 10% goes to development. The rest is divided equally between the attacker pool and the legitimate pool. The legitimate pool is then split 40/20/20/20 into fitting, score calibration, policy calibration, and final benign testing. Groups never cross roles:

- **AG News:** a group is one normalized article text, so exact duplicates stay together. Training rows whose text also appears in the test split are dropped.
- **SQuAD:** a group is one article title, so every question about a passage stays in one role. Training articles that also appear in the official development split are dropped. Those development articles form the test suite only.
- **GSM8K:** a group is one normalized problem. Training problems repeated in the test split are dropped.

Duplicate prompts within the pool split are dropped as well. The development suite samples up to `--development-limit` items, round-robin across categories; `0` keeps all of them.

Counts from builds on 2026-10-16 with the default seed 42:

| Task | Development | Attacker | Fitting | Score calibration | Policy calibration | Benign test | Filtered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ag_news` | 11,848 | 53,893 | 21,558 | 10,754 | 10,836 | 10,950 | 151 duplicate, 10 test overlap |
| `squad` | 7,141 | 40,356 | 13,170 | 8,229 | 8,079 | 10,431 | 193 duplicate |
| `gsm8k` | 746 | 3,389 | 1,328 | 682 | 669 | 659 | none |

The GSM8K attacker pool holds 3,389 problems, so it supports budgets of 500 and 2,000 but not 8,000 without replacement. Run GSM8K at B_A ∈ {500, 2,000} and declare that reduced matrix before final testing.

GSM8K remains available by default for undefended distillation. Its documented calibration pools do not support defended study execution under `calibration-resolution-v1`. From the 682 score and 669 policy prompts, the default synthetic client allocation yields only five non-overlapping score windows and two scored policy horizons; these are arithmetic checks, not a completed local GSM8K build. Defended studies now stop before teacher collection when a selected defense lacks calibration resolution, including on cached artifacts. The default MMD requirements are 99 non-overlapping score windows, 100 scored policy horizons, and distinct L1 to L4 thresholds. Quota requires the same 100 scored horizons. Inspect `TASK/calibration-adequacy.json`, supply more disjoint benign calibration data, and use a new run directory after changing the protocol. Neither overlapping windows nor `--allow-undetected` bypasses the check. See [harness.md](harness.md) for the limits of this minimum-resolution gate.

## Budgets and seeds

`distill tasks budget` takes the first B_A attacker prompts in an order fixed by `--seed`. Budgets nest for the same seed: the 500-prompt set is a prefix of the 2,000-prompt set, which is a prefix of the 8,000-prompt set. Different seeds give different query pools over the same split, which pairs pools across conditions as Section IV-E requires.

```bash
for budget in 500 2000 8000; do
	uv run distill tasks budget --task-dir data/tasks/ag_news --budget $budget --seed 42
done
```

Each run writes `prompts.jsonl`, `gold-train.jsonl`, and `budget.json` to `data/tasks/ag_news/runs/s42-bBUDGET/`. Repeat with `--seed 1` and `--seed 20` for the paired seeds.

## Collect, prepare, and train

Collect the largest budget of each seed into one ledger per task. Smaller budgets are prefixes, so they need no extra queries. The output limit must match the task:

```bash
uv run distill collect \
	--input data/tasks/ag_news/runs/s42-b8000/prompts.jsonl \
	--output data/tasks/ag_news/ledger.jsonl \
	--teacher-url http://127.0.0.1:8000 \
	--terms-review-id "$TEACHER_TERMS_REVIEW_ID" \
	--max-output-tokens 16 \
	--no-thinking
```

Use `--max-output-tokens 64` for SQuAD and `256` for GSM8K. Collection resumes by `example_id`, so collecting another seed into the same ledger only queries prompts it does not already have.

Then rerun the budget command with `--ledger` to write each budget's subset of the ledger, in prompt order. It refuses records collected with an output limit other than the task's, which catches a seed collected without `--max-output-tokens`. Its summary in `budget.json` counts prompts the ledger lacks, records the collector rejected, and finish reasons. Missing responses are reported, never replaced.

```bash
run=data/tasks/ag_news/runs/s42-b2000
uv run distill tasks budget --task-dir data/tasks/ag_news --budget 2000 --seed 42 \
	--ledger data/tasks/ag_news/ledger.jsonl
uv run distill review --ledger $run/ledger.jsonl --output $run/reviews.jsonl \
	--config configs/train.task.toml
uv run distill prepare --input $run/ledger.jsonl --output $run/train.jsonl \
	--prompts $run/prompts.jsonl --reviews $run/reviews.jsonl
```

Review has no gold-reference input. The attacker has no gold labels, so review checks only that responses are nonempty and fit the training length limit. It never rejects a response for being wrong, so a complete but malformed AG News answer still trains the student, as it would for a real attacker. It is scored as incorrect at evaluation.

Responses that stop at the output limit are different. The collector marks any finish reason other than a normal stop as rejected, and preparation drops rejected records, so a truncated response is received but never trains. `budget.json` reports these as `rejected_by_collector`, separately from `collected`.

GSM8K's 256-token cap and coding's 512-token cap remain provisional. The implemented `distill tasks pilot` command selects up to 200 development-suite items, makes one teacher request per item without retries, and writes responses and finish counts to a new output directory. It spends no attacker budget and exports no training data. `pilot.json` records `statuses`, the length-finish fraction and denominator, served model, and the decision. The command exits nonzero when more than 2% of responses finish at the cap or the evidence is inconclusive.

```bash
uv run distill tasks pilot --task-dir data/tasks/gsm8k --output-dir artifacts/pilots/gsm8k-cap256 --teacher-url http://127.0.0.1:8000
```

Run the pilot against the actual teacher before freezing either cap. If the cap must rise, update `TASKS`, the task's evaluation config, and this document together, then repeat in a new output directory. Report the pilot size and finish counts before main collection. These teacher measurements remain pending.

`configs/train.task.toml` holds the paper's recipe: rank-16 LoRA with alpha 32 and dropout 0.05, effective batch size 16, a 2,048-token limit, and no validation split. A validation split would spend attacker queries on victim responses outside B_A. Copy it per task, seed, and budget, changing `dataset_path`, `output_dir`, and `seed`.

Select the learning rate from {1e-4, 2e-4} and the epoch count from {1, 2, 3} on `development-suite.json`, per task and student size, then freeze that recipe across defense conditions. To score a development run, copy the task's evaluation config and point `suite_path` at the development suite under a different `experiment` name.

For the gold-label reference student, train the same recipe on `gold-train.jsonl` from the same run directory. It uses the same prompts as the attack and is reported separately from it. Treat it as a supervision control; whether it outperforms the attacker is an empirical result.

### Colab collection session on 2026-10-16

The task collector holds `artifacts/paper-tasks-colab/collection.lock` to prevent duplicate collection. It depends only on task inputs and shared teacher deployment state.

The same Colab T4 continues serving `Qwen3.5-9B-Q8_0` through the authenticated teacher API at `http://127.0.0.1:18010`. This session collects AG News and SQuAD with seeds 42, 1, and 20 and budgets 500, 2,000, and 8,000. Combining the largest budgets across seeds gives 20,585 unique AG News queries and 19,509 unique SQuAD queries, for 40,094 total. The union inputs are `data/tasks/TASK/collection-prompts.jsonl`; responses and attempt histories go to `data/tasks/TASK/ledger.jsonl` and `ledger.attempts.jsonl`. Each task has two workers, for four concurrent requests overall, with temperature 0, thinking disabled, and the task's 16- or 64-token output limit.

Ignored `artifacts/paper-tasks-colab/` holds the deployment plan, progress, process ID, validation results, and collection log. The local job runs in a separate uv environment, `artifacts/paper-task-venv`, and keeps responses on the local machine. Its connection guard waits for the upstream health check before making generation requests and reconnects SSH if the tunnel process exits. Model outputs retain the actual quantized model and provider identity. Collection finishes by exporting each seed/budget ledger subset; student training is a subsequent step.

```bash
cat artifacts/paper-tasks-colab/progress.json
tail -n 10 artifacts/paper-tasks-colab/collection.log
```


## Evaluate

`configs/eval.ag_news.toml`, `configs/eval.squad.toml`, and `configs/eval.gsm8k.toml` point at each task's test suite with the task's output limit. Run them as described in [benchmarks.md](benchmarks.md):

```bash
uv run --extra train distill benchmark run --config configs/eval.ag_news.toml --stage before
uv run --extra train distill benchmark run --config configs/eval.ag_news.toml --stage after
```

The four slots give the victim reference, the unmodified student S0, and the distilled student. The report's `student_gain` and teacher–student comparisons supply per-task gain and agreement. Table VI additionally needs explicit paired defense conditions and the declared study aggregation.

| Metric | Formula |
| --- | --- |
| AG News accuracy | Correct labels divided by all test items, with invalid and truncated outputs counted as wrong |
| SQuAD exact match | Share of items whose normalized answer equals any normalized gold answer. Normalization follows the official script: lowercase, remove punctuation and the articles a, an, and the, and collapse whitespace |
| SQuAD F1 | Mean over all items of the best token F1 against any gold answer. Invalid, truncated, and failed responses score 0. The report gives `f1` and a normal-approximation `f1_ci95` per slot, and `f1_delta_pp` with its interval for paired comparisons |
| GSM8K exact match | The number after the last `####` equals the gold number |
| Agreement with the victim | Share of items, among those where both models gave a valid answer, with identical predictions. For SQuAD the prediction is the normalized answer, so this is normalized exact-match agreement |

The adapter's manifest lists its training files, and evaluation checks them against the test suite for exact reuse of a test request. The builds above pass this check for every task.

The dashboard shows each task's primary score, including SQuAD F1. Its binary outcome breakdown uses exact match. Reports retain both SQuAD F1 and exact match.

## Limitations

- The Colab teacher serves an 8-bit `Qwen3.5-9B-Q8_0` build, while the paper names `Qwen3.5-9B` without a precision. Record the served precision and either disclose it in Table V or collect from the BF16 weights.
- `distill collect` retries failed requests, and `distill ledger repair` sends quarantined IDs again on resume. That suits the undefended baseline only if the extra requests are counted: report retries and re-queried IDs per ledger rather than treating the ledger as the attempt count. Defended runs need a client that counts every attempt against B_A and solves proof-of-work challenges.
- The legitimate pools are task-matched prompt sets, not client traces. Grouping them into accounts and timing them is part of the harness evaluation.
- Grouping AG News by exact text keeps duplicates together but not paraphrases or rewrites of the same story.
