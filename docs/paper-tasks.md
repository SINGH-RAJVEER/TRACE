# Paper tasks

The paper's distillation study (Section IV-B) measures extraction on narrow task APIs rather than general chat. `distill tasks` builds those tasks: AG News four-class classification and SQuAD 1.1 extractive QA form the core study, and GSM8K adds a generative, multi-step task with an objective answer. Each task is a separate service with its own prompt format, output limit, ledger, frozen suites, and evaluation config, so results are reported per task as Table VI requires.

This workflow is separate from the general prompt corpus in [prompt-corpus.md](prompt-corpus.md). The general corpus remains a full-pipeline demonstration and is not part of the paper's claims.

## Tasks

| Task | Dataset | Pool split | Final test split | Output limit | Primary metric |
| --- | --- | --- | --- | --- | --- |
| `ag_news` | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) at `eb185aa` | `train` | `test` (7,600) | 16 tokens | Label accuracy |
| `squad` | [rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad) `plain_text` at `7b6d24c` | `train` | `validation` (10,570) | 64 tokens | Token F1, with exact match |
| `gsm8k` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main` at `740312a` | `train` | `test` (1,319) | 256 tokens | Numeric exact match |

Revisions are pinned in `TASKS` in `packages/distillation/src/distillation/tasks.py`. The AG News dataset card lists its license as unknown; check its terms before redistributing derived data. SQuAD is CC BY-SA 4.0 and GSM8K is MIT.

Every request for a task uses one fixed serialization, in the attacker pool, the legitimate pools, the development suite, and the test suite alike:

- **AG News:** an instruction naming the four topics, then `Article: <text>`. The service must answer with exactly one of `World`, `Sports`, `Business`, or `Sci/Tech`, ignoring case, surrounding quotes, and a final period. Any other output is scored invalid and stays in the denominator.
- **SQuAD:** an instruction to answer with a short span, then `Passage: <context>` and `Question: <question>`. Contexts longer than 6,000 characters are cut at a word boundary before querying, training, and evaluation alike; the manifest counts how often this happens. No pinned row currently needs it. Answer positions are never used to filter or window passages.
- **GSM8K:** the problem followed by the same instruction the general GSM8K benchmark uses, ending with `####` and the final number.

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

Review runs without `--references` on purpose. The attacker has no gold labels, so review checks only that responses are nonempty and fit the training length limit. It never rejects a response for being wrong, so a complete but malformed AG News answer still trains the student, as it would for a real attacker. It is scored as incorrect at evaluation.

Responses that stop at the output limit are different. The collector marks any finish reason other than a normal stop as rejected, and preparation drops rejected records, so a truncated response is received but never trains. `budget.json` reports these as `rejected_by_collector`, separately from `collected`. This rarely matters at 16 or 64 tokens, but the 256-token GSM8K limit is an unmeasured estimate. Collect a pilot of about 200 GSM8K prompts first and check the share of `length` finishes in `finish_reasons`. If it exceeds a few percent, raise the limit in `TASKS`, `configs/eval.gsm8k.toml`, and this document together, before collecting the main run.

`configs/train.task.toml` holds the paper's recipe: rank-16 LoRA with scaling 32 and dropout 0.05, effective batch size 16, a 2,048-token limit, and no validation split. A validation split would spend attacker queries on victim responses outside B_A. Copy it per task, seed, and budget, changing `dataset_path`, `output_dir`, and `seed`.

Select the learning rate from {1e-4, 2e-4} and the epoch count from {1, 2, 3} on `development-suite.json`, per task and student size, then freeze that recipe across defense conditions. To score a development run, copy the task's evaluation config and point `suite_path` at the development suite under a different `experiment` name.

For the gold-label reference student, train the same recipe on `gold-train.jsonl` from the same run directory. It uses the same prompts as the attack and is reported separately from it.

## Evaluate

`configs/eval.ag_news.toml`, `configs/eval.squad.toml`, and `configs/eval.gsm8k.toml` point at each task's test suite with the task's output limit. Run them as described in [benchmarks.md](benchmarks.md):

```bash
uv run --extra train distill benchmark run --config configs/eval.ag_news.toml --stage before
uv run --extra train distill benchmark run --config configs/eval.ag_news.toml --stage after
```

The four slots give the victim reference, the unmodified student S0, and the distilled student. The report's `student_gain` and teacher–student comparisons supply the gain and agreement columns of Table VI.

| Metric | Formula |
| --- | --- |
| AG News accuracy | Correct labels divided by all test items, with invalid and truncated outputs counted as wrong |
| SQuAD exact match | Share of items whose normalized answer equals any normalized gold answer. Normalization follows the official script: lowercase, remove punctuation and the articles a, an, and the, and collapse whitespace |
| SQuAD F1 | Mean over all items of the best token F1 against any gold answer. Invalid, truncated, and failed responses score 0. The report gives `f1` and a normal-approximation `f1_ci95` per slot, and `f1_delta_pp` with its interval for paired comparisons |
| GSM8K exact match | The number after the last `####` equals the gold number |
| Agreement with the victim | Share of items, among those where both models gave a valid answer, with identical predictions. For SQuAD the prediction is the normalized answer, so this is normalized exact-match agreement |

The adapter's manifest lists its training files, and evaluation checks them against the test suite for exact reuse of a test request. The builds above pass this check for every task.

The dashboard shows each task's exact-match score. It does not yet display SQuAD F1; read `f1` from the exported report.

## Limitations

- The paper's victim is Qwen2.5-3B-Instruct, with Qwen2.5-0.5B-Instruct and 1.5B-Instruct students. The configs still use the repository's Qwen3.5-9B teacher and Qwen3.5-2B-Base student until that switch is made. With Qwen2.5, set `target_modules` to the attention projections as the paper specifies.
- `distill collect` retries failed requests. That suits the undefended baseline. Defended runs need a client that counts every attempt against B_A and solves proof-of-work challenges.
- The legitimate pools are task-matched prompt sets, not client traces. Grouping them into accounts and timing them is part of the harness evaluation.
- Grouping AG News by exact text keeps duplicates together but not paraphrases or rewrites of the same story.
