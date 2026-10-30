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
| `gsm8k` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main` at `740312a` | `train` | `test` (1,319) | 1,024 tokens | Numeric exact match |
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

Each run directory `data/tasks/TASK/runs/sSEED-bBUDGET/` holds the generated `train.attacker.toml` and `train.gold.toml`. The adapters go to `artifacts/tasks/TASK/sSEED-bBUDGET/attacker` and `.../gold`. The command is resumable: collection resumes by `example_id`, training resumes from its last checkpoint, and an adapter whose `training-manifest.json` matches the current dataset is skipped. Both `just distill --mode tasks` and `just start --mode tasks` default to AG News, SQuAD, GSM8K, and coding with three seeds. The default uses each task's whole attacker pool: 35,944 AG News queries, 10,353 SQuAD queries, and 3,389 GSM8K queries with seed-42 split-v2 builds. All three seeds use the same complete pool in different orders. Legacy full-budget task runs use the complete attacker allocation of the declared coding build, which now defaults to the 50,000-eligible-example subset across all roles at `data/tasks/coding-subset-50000-s42`. Use `--coding-pool-size 0` for exhaustive source qualification. This is separate from the shared gateway default of 6,000 total submissions. This plans 12 undefended attacker adapters and 12 gold references, excluding defended students, recipe trials, and matched controls. GSM8K supplies the math capability distillation corpus; its 1,024-token output cap remains provisional until the implemented cap pilot passes against the actual teacher.

`--budgets 500,2000,8000` selects subsets, skipping 8,000 for GSM8K when it is selected. `start` freezes the resolved numeric plan in `budgets.json`; reporting uses those task-specific values. Existing protocol options remain immutable. Protected campaigns retain one global attempt limit and deadline. Each protected account resets only after its 24-hour time horizon; exceeding 10,000 policy attempts refuses further ordinary requests until then. A full-pool run cannot regain an account allowance by sending refusals. The undefended arm remains unrestricted by this account cap. Time resets never replenish the campaign allowance. The resulting reports describe separate task studies, not shared-gateway execution.

`just start --mode tasks` collects each seed's undefended baseline through the same single-account campaign envelope as the defended arms: concurrency 1, one attempt per prompt, the attempt budget, and the 24-hour deadline. Training requires a response for every planned prompt. A baseline that loses prompts to teacher failures or an interruption resumes on its original clock, retrying only prompts without a response, for up to three passes; rerunning with the same `--run-dir` resumes it again. Every retry is recorded in the events and counted as `retried_queries`. A baseline stopped by the deadline, or refused by a policy, stops the study with status `undefended_baseline_incomplete`. A deadline means the envelope cannot carry that budget, and defended campaigns at the same budget would be cut off too, so pass smaller `--budgets`. Protected campaigns still refuse to resume.

Evaluation is not part of this command; see [Evaluate](#evaluate). The sections below describe the individual steps it runs.

## Build

```bash
uv sync --all-packages --extra eval
uv run --extra eval distill tasks build --task ag_news
uv run --extra eval distill tasks build --task squad
uv run --extra eval distill tasks build --task gsm8k
```

AG News and SQuAD now default to `data/tasks/TASK-split-v2-s42/`; GSM8K keeps `data/tasks/gsm8k/`. These paths are gitignored. Builds refuse to overwrite an existing manifest. Existing `data/tasks/ag_news/` and `data/tasks/squad/` remain untouched; launchers select the new paths automatically. Start studies in a new `--run-dir` because source inventories and calibration rules have changed.

| File | Contents |
| --- | --- |
| `attacker.jsonl` | The attacker-query pool, in the collector's prompt format with split `train` |
| `gold-answers.jsonl` | Gold completions for the attacker pool, used only to train the gold-label reference student |
| `legitimate/fitting.jsonl`, `score_calibration.jsonl`, `policy_calibration.jsonl`, `benign_test.jsonl` | The legitimate-workload pool for the future harness, marked `test` so `distill collect` skips them by default |
| `development-suite.json` | A frozen, gold-scored suite of up to 1,000 development items for selecting the training recipe |
| `test-suite.json` | The frozen official evaluation split, used only for final scoring |
| `manifest.json` | Dataset pins, seed, per-role counts and source groups, filter reasons, truncation counts, and SHA-256 hashes of every file |

### Splits

Each source document is assigned once by a seeded hash of its group. `task-source-split-v2` records absolute role shares in each new non-coding task manifest. AG News uses 10/30/5/12/34/9 percent for development, attacker, fitting, score calibration, policy calibration, and benign testing. SQuAD uses 10/10/6/15/50/9 percent. GSM8K and coding retain 10/45/18/9/9/9 percent. The fitting and attacker shares fund the larger AG News and SQuAD calibration pools. The development and benign-test hash boundaries remain unchanged, and new attacker pools are subsets of the old pools. Groups never cross roles:

- **AG News:** a group is one normalized article text, so exact duplicates stay together. Training rows whose text also appears in the test split are dropped.
- **SQuAD:** a group is one article title, so every question about a passage stays in one role. Training articles that also appear in the official development split are dropped. Those development articles form the test suite only.
- **GSM8K:** a group is one normalized problem. Training problems repeated in the test split are dropped.

Duplicate prompts within the pool split are dropped as well. The development suite samples up to `--development-limit` items, round-robin across categories; `0` keeps all of them.

Counts with seed 42 (AG News and SQuAD split-v2; unchanged GSM8K allocation):

| Task | Development | Attacker | Fitting | Score calibration | Policy calibration | Benign test | Filtered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ag_news` | 11,848 | 35,944 | 6,012 | 14,337 | 40,748 | 10,950 | 151 duplicate, 10 test overlap |
| `squad` | 7,141 | 10,353 | 4,110 | 13,226 | 42,145 | 10,431 | 193 duplicate |
| `gsm8k` | 746 | 3,389 | 1,328 | 682 | 669 | 659 | none |

The GSM8K attacker pool holds 3,389 problems, so it supports budgets of 500 and 2,000 but not 8,000 without replacement. Run GSM8K at B_A ∈ {500, 2,000} and declare that reduced matrix before final testing.

The default separate-task client generator at concentrated share 0.1 yields 128 non-overlapping score windows and 108 scored policy accounts for AG News, and 116 windows and 113 scored accounts for SQuAD. These are workload counts, not independent clients or measured detection results. Source clustering remains necessary, especially for SQuAD article groups. Changed seeds or client mixtures can produce different counts. Passing these counts does not prove distinct fitted thresholds.

`calibration-resolution-v2` keeps the 99-window and 100-scored-horizon minimum for AG News, SQuAD, coding, and the shared gateway. Separate-task GSM8K runs may continue with `status: resolution-limited` and explicit warnings for insufficient counts or collapsed levels in a small calibration. Missing scored counts, missing fits, nonfinite or decreasing thresholds, and empty score pools still stop execution. The report includes each task's full calibration decision; a resolution-limited result cannot support the nominal 1% false-block claim. This exception does not apply to shared gateway calibration, which pools all domains, or bypass the separate detectability gate. `--allow-undetected` only controls that detectability gate. See [harness.md](harness.md).

## Budgets and seeds

`distill tasks budget` takes the first B_A attacker prompts in an order fixed by `--seed`. Budgets nest for the same seed: the 500-prompt set is a prefix of the 2,000-prompt set, which is a prefix of the 8,000-prompt set. Different seeds give different query pools over the same split, which pairs pools across conditions as Section IV-E requires.

```bash
for budget in 500 2000 8000; do
	uv run distill tasks budget --task-dir data/tasks/ag_news-split-v2-s42 --budget $budget --seed 42
done
```

Each run writes `prompts.jsonl`, `gold-train.jsonl`, and `budget.json` to `data/tasks/ag_news-split-v2-s42/runs/s42-bBUDGET/`. Repeat with `--seed 1` and `--seed 20` for the paired seeds.

## Collect, prepare, and train

Collect the largest budget of each seed into one ledger per task. Smaller budgets are prefixes, so they need no extra queries. The output limit must match the task:

```bash
uv run distill collect \
	--input data/tasks/ag_news-split-v2-s42/runs/s42-b8000/prompts.jsonl \
	--output data/tasks/ag_news-split-v2-s42/ledger.jsonl \
	--teacher-url http://127.0.0.1:8000 \
	--terms-review-id "$TEACHER_TERMS_REVIEW_ID" \
	--max-output-tokens 16 \
	--no-thinking
```

Use `--max-output-tokens 64` for SQuAD and `1024` for GSM8K. Collection resumes by `example_id`, so collecting another seed into the same ledger only queries prompts it does not already have.

Then rerun the budget command with `--ledger` to write each budget's subset of the ledger, in prompt order. It refuses records collected with an output limit other than the task's, which catches a seed collected without `--max-output-tokens`. Its summary in `budget.json` counts prompts the ledger lacks, records the collector rejected, and finish reasons. Missing responses are reported, never replaced.

```bash
run=data/tasks/ag_news-split-v2-s42/runs/s42-b2000
uv run distill tasks budget --task-dir data/tasks/ag_news-split-v2-s42 --budget 2000 --seed 42 \
	--ledger data/tasks/ag_news-split-v2-s42/ledger.jsonl
uv run distill review --ledger $run/ledger.jsonl --output $run/reviews.jsonl \
	--config configs/train.task.toml
uv run distill prepare --input $run/ledger.jsonl --output $run/train.jsonl \
	--prompts $run/prompts.jsonl --reviews $run/reviews.jsonl
```

Review has no gold-reference input. The attacker has no gold labels, so review checks only that responses are nonempty and fit the training length limit. It never rejects a response for being wrong, so a complete but malformed AG News answer still trains the student, as it would for a real attacker. It is scored as incorrect at evaluation.

Responses that stop at the output limit are different. The collector marks any finish reason other than a normal stop as rejected, and preparation drops rejected records, so a truncated response is received but never trains. `budget.json` reports these as `rejected_by_collector`, separately from `collected`.

GSM8K's 1,024-token cap and coding's 512-token cap remain provisional. GSM8K was raised from 256 tokens after the bounded study's A2 undefended ledger finished 1,133 of 1,500 GSM8K responses at the cap, and the development-selected student, which learned the teacher's long worked solutions, reached the cap on 261 of 373 qualification items, up from 165 for the base student. The implemented `distill tasks pilot` command selects up to 200 development-suite items, makes one teacher request per item without retries, and writes responses and finish counts to a new output directory. It spends no attacker budget and exports no training data. `pilot.json` records `statuses`, the length-finish fraction and denominator, served model, and the decision. The command exits nonzero when more than 2% of responses finish at the cap or the evidence is inconclusive.

```bash
uv run distill tasks pilot --task-dir data/tasks/gsm8k --output-dir artifacts/pilots/gsm8k-cap1024 --teacher-url http://127.0.0.1:8000
```

Run the pilot against the actual teacher before freezing either cap. If the cap must rise, update `TASKS`, the task's evaluation config, and this document together, then repeat in a new output directory. Report the pilot size and finish counts before main collection. These teacher measurements remain pending.

`configs/train.task.toml` holds the paper's recipe: rank-16 LoRA with alpha 32 and dropout 0.05, effective batch size 16, a 2,048-token limit, and no validation split. A validation split would spend attacker queries on victim responses outside B_A. Copy it per task, seed, and budget, changing `dataset_path`, `output_dir`, and `seed`.

Select the learning rate from {1e-4, 2e-4} and the epoch count from {1, 2, 3} on `development-suite.json`, per task and student size, then freeze that recipe across defense conditions. To score a development run, copy the task's evaluation config and point `suite_path` at the development suite under a different `experiment` name.

For the gold-label reference student, train the same recipe on `gold-train.jsonl` from the same run directory. It uses the same prompts as the attack and is reported separately from it. Treat it as a supervision control; whether it outperforms the attacker is an empirical result.

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
