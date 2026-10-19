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
| `coding` | Pinned NVIDIA OpenCodeInstruct, complete qualified corpus | `train` | 163-item reference-qualified subset of pinned `evalplus/humanevalplus` | 512 tokens, provisional | Executable pass@1 |

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

`--budgets 500,2000,8000` selects subsets, skipping 8,000 for GSM8K when it is selected. `start` freezes the resolved numeric plan in `budgets.json`; reporting uses those task-specific values. Existing protocol options remain immutable. Protected campaigns retain one global attempt limit and deadline. Each protected account resets only after its 24-hour time horizon; exceeding 10,000 policy attempts refuses further ordinary requests until then. A full-pool run cannot regain an account allowance by sending refusals. The undefended arm remains unrestricted by this account cap. Time resets never replenish the campaign allowance. The resulting reports describe separate task studies, not shared-gateway execution.

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

Responses that stop at the output limit are different. The collector marks any finish reason other than a normal stop as rejected, and preparation drops rejected records, so a truncated response is received but never trains. `budget.json` reports these as `rejected_by_collector`, separately from `collected`. This rarely matters at 16 or 64 tokens, but the 256-token GSM8K limit is an unmeasured estimate. Section IV-B fixes the check: collect a pilot of about 200 GSM8K prompts from the development split, so it spends no attacker budget, and count `length` finishes in `finish_reasons`. If more than 2% stop at the limit, raise the limit in `TASKS`, `configs/eval.gsm8k.toml`, and this document together, then repeat the pilot. Report the pilot size and finish-reason counts, and freeze every task's limit before collecting the main run. `distill tasks pilot` now constructs the development-only pilot; it still needs to run against the actual teacher before a cap is frozen.

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

## Implementation TODO

This checklist applies to the task-only extraction and account-policy study. It preserves the original 28 review-item numbers so implementation work can be matched to [paper-revision-todo.md](paper-revision-todo.md). Paper items follow the supplied protocol, hybrid items adapt existing repository behavior, and repo-first items bound optional work. Checked entries record completed repository changes only. Unchecked entries include implementation, integration, or experimental validation still required; they do not assert that every underlying component is absent from the in-progress codebase.

### Agreed scope and completed cleanup

| Scope | Work covered |
| --- | --- |
| Core study | AG News classification, SQuAD extractive QA, and GSM8K generative math, budgeted task-specific attacker adapters, an unmodified student, a same-prompt gold-label reference, and paired undefended/defended conditions |
| Required evidence | Positive undefended transfer on development data, task-matched legitimate utility, campaign attempt/time accounting, released-token controls, per-task capability scores, and paired uncertainty |
| Bounded secondary work | Task-matched schedules, detector ablations, aggregate alerts, and overhead measurements declared in the frozen matrix; retain active extraction or upstream reproduction only with their complete protocol and evidence |
| Excluded | General-corpus collection/training, a general instruction/chat student, broad knowledge benchmarks, correctness-filtered attacker supervision, cross-task model rankings, teacher-gap closure, and general-chat dataset or utility studies |

- [x] Remove the general-corpus launchers, builders, prompt families, configs, correctness filters, dedicated tests/docs, and local prompt/ledger artifacts. `just distill --mode tasks` is the only distillation workflow.
- [x] Preserve task pools, frozen-suite hashes, task ledgers, and shared collection/training/evaluation code. Remove the task collector's dependency on the retired general ledger and resume it without losing its existing rows.
- [x] Update the README, operating guides, and dashboard for task-only evaluation, including SQuAD F1 and removal of broad-model aggregate scores and coding filters.
- [ ] Freeze the actual task/model matrix before final testing. The default AG News, SQuAD, GSM8K, and coding matrix plans 24 undefended attacker/gold training runs across three seeds, using each task's full attacker pool. These counts exclude defended students, development recipe trials, and matched supervision controls and do not establish completed results.
- [ ] Finish model provenance. The paper and default configs now agree on Qwen3.5-9B (non-thinking) and Qwen3.5-2B-Base, and `configs/train.task.toml` pins the student revision. Still record the victim revision and served precision (the Colab teacher serves `Qwen3.5-9B-Q8_0`), the chat template, and the adapter modules that `all-linear` resolves to, in every run manifest and in Table V.
- [ ] Freeze task-specific output limits, development-selected rank-16 training settings, and the gold-unavailable attacker review protocol. Do not restore a general-corpus run or a gold correctness filter as a prerequisite for the task study.

### 1. Exclude service instructions from detector embeddings

- [x] Paper: give each request a separate detector serialization containing article text for AG News, context plus question for SQuAD, and problem text for GSM8K. Keep fixed service instructions in the victim prompt.
- [x] Use the same detector serialization in account construction, fitting, score calibration, policy calibration, and testing. Record its version in artifacts so incompatible inputs cannot be mixed.
- [ ] Validate that changing only fixed service instructions leaves detector input unchanged, while changing task content changes it. Check that victim prompts retain the instructions.

### 2. Enforce client and source-document separation

- [ ] Paper: carry client identities and source-group identities from task pools through account construction and calibration manifests. Preserve source grouping when constructing accounts.
- [ ] Reject overlap between fitting, score calibration, policy calibration, and final benign testing on either identity. Reject missing identities when separation cannot be verified, and include the held-out test manifest in the validation.
- [ ] Record the role assignments and separation check in calibration and evaluation artifacts. Validate that shared clients, shared source documents, and missing identities fail before calibration or final scoring.

### 3. Apply L2 spacing on entry

- [ ] Paper: track the previous admitted request's release time across all policy levels, including L0 and L1.
- [x] When L2 spacing applies, release the next admitted request no earlier than one second after the previous admitted request, including the first request entering L2. Keep other release constraints in effect.
- [x] Validate the L0-to-L2 transition and consecutive L2 requests, including multiple arrivals at the same timestamp. Rejected requests must not count as admitted requests.

### 4. Report only levels a baseline implements

- [ ] Paper: declare the levels implemented by each policy and use that declaration in replay summaries and exported reports.
- [ ] Keep quota and rate baseline block decisions, but report their nonexistent L2 and L3 outcomes as N/A rather than inferring that L4 passed through those levels. Preserve actual L2 and L3 outcomes for policies that implement them.
- [ ] Validate baseline block summaries and downstream report rendering so N/A stays distinct from zero and never enters a level-rate denominator as an observed outcome.

### 5. Require valid outputs for agreement

- [ ] Hybrid: retain the existing task parsers and record response validity explicitly alongside parsed predictions. A parsed prediction alone must not make an invalid, truncated, or failed response eligible for agreement.
- [ ] Compute agreement only over pairs where both responses are valid. Report the agreement numerator, valid-pair denominator, total paired items, and coverage. Report agreement as unavailable when no valid pairs remain.
- [ ] Keep invalid, truncated, and failed responses in task-score denominators as incorrect. Validate that matching parsed predictions from unusable responses cannot inflate agreement, while valid shared wrong answers still count as agreement.

### 6. Use each task's primary capability metric consistently

- [x] Retain AG News label-accuracy, SQuAD token-F1, and GSM8K numeric-exact-match scorers. Use SQuAD F1 in dashboard task scores, student-change charts, category scores, and speed comparisons, while naming exact match as the secondary binary outcome.
- [ ] Make the primary metric and denominator explicit in generic exported capability fields and the study report. Use that metric consistently for defended-versus-undefended comparisons and `R_D`; do not add a cross-task aggregate score or teacher-gap closure.
- [ ] Validate a SQuAD example with partial token overlap where F1 differs from exact match through the complete study report and defense comparisons. Preserve the existing label-accuracy and numeric-exact-match behavior for the other tasks.

### 7. Preserve experiment dimensions and aggregate paired runs

- [ ] Hybrid: record explicit task, model identity, defense condition, attack schedule, query budget, query-order seed, and training seed for every run. Keep query-order and training seeds separate even when they share a default value.
- [ ] Keep immutable run identities and the four-slot view for an individual run. Scope slot selection to its full experiment dimensions so a newer result under another condition cannot silently replace it.
- [ ] Add a separate study report that pairs compatible conditions by their declared dimensions and seeds, then aggregates paired results with sample counts and uncertainty. Record the aggregation method and report missing pairs rather than substituting unrelated runs.
- [ ] Validate two conditions with overlapping slot names and multiple seeds, including an incomplete pair. Confirm individual runs remain inspectable and only compatible pairs enter study comparisons.

### 8. Evaluate the adapters produced by the declared run

- [x] Point checked-in task evaluation configs and the standalone task training default at `artifacts/tasks/TASK/sSEED-bBUDGET/attacker`, matching the launcher's actual output structure. Gold adapters remain separate under `.../gold` and must be selected explicitly.
- [ ] Prefer generating evaluation configs from the run manifest, carrying task, seed, budget, student role, suite, model identity, and exact adapter path together. Record those identities in evaluation output.
- [ ] Validate attacker and gold evaluation config generation against a declared run. Fail clearly when its exact adapter is absent or its manifest disagrees with the selected run.

### 9. Match the complete training fingerprint before skipping

- [ ] Hybrid: replace the dataset-only completion check with a deterministic fingerprint covering training-data hashes, base model and immutable revision, resolved training configuration, and serialization. Include tokenizer and chat-template identity wherever they affect training.
- [ ] Store the fingerprint and its inputs in the run and training manifests. Skip completed training only when the fingerprint matches and the expected adapter is present. Treat legacy manifests without the fingerprint as unverified.
- [ ] Preserve resumability for compatible checkpoints and completed adapters. Refuse reuse of an incompatible checkpoint or adapter and require a distinct run output or explicit retraining.
- [ ] Validate that an unchanged run skips correctly, while changes to data, model revision, training settings, or serialization prevent reuse. Confirm evaluation provenance identifies the adapter actually trained.

### 10. Enforce policy decisions before victim inference

- [ ] Hybrid: connect the policy library to a campaign simulator with explicit, stable account identities. Enforce admission, queueing, challenges, and output-release ordering before allowing victim inference or releasing a response.
- [ ] Add an HTTP wrapper for live service evaluation using the same enforcement path. Bind requests to controlled experiment accounts; document that the experiment does not implement production authentication.
- [ ] Validate that refused requests never invoke the victim, queued requests wait for admission, and challenged requests require successful verification before proceeding. Confirm the simulator and HTTP wrapper enforce the same ordering.

### 11. Enforce campaign attempt and elapsed-time budgets

- [ ] Paper: add a campaign-wide attempted-request counter for `B_A` and a deadline for `T_A`. Charge every retry, filler request, and failed request as an attempt, across every account in the campaign.
- [ ] Include waiting and challenge solving in elapsed campaign time. Prevent new attempts beyond either budget and explicitly account for pending requests and responses that miss the deadline. Keep the ordinary undefended collector available separately.
- [ ] Validate retries, failures, filler traffic, concurrent accounts, and deadline boundaries. Record the budget or deadline that terminated each campaign.

### 12. Solve challenges and collect released responses

- [ ] Paper: integrate the existing solver and verifier into the campaign client so attackers can solve challenges and continue extracting within their budgets.
- [ ] Record hashes attempted, solver hardware, solving time, queue time, verification outcomes, and released outputs. Charge solving and waiting to campaign costs and distinguish measured costs from estimates.
- [ ] Validate successful solving followed by response release, failed verification, and solving that runs past the deadline. A challenge event alone must never count as stopped extraction.

### 13. Implement the declared attack schedule matrix

- [ ] Hybrid: implement pure extraction, within-account benign/extraction mixtures, slow traffic, and fixed-total-budget Sybil campaigns as the primary schedules. Record schedule settings and seeds, and share campaign budgets across Sybil accounts.
- [ ] Add paraphrases, benign-matched adaptation, and reset schedules as secondary experiments. Define what resets, what persists, and what each schedule costs before running it.
- [ ] Freeze the primary and secondary matrix before final testing and revise the paper if schedules are deferred. Validate mixture composition, timing, and shared Sybil budget accounting.

### 14. Keep fixed schedules distinct from active extraction

- [ ] Repo first: keep seeded fixed schedules as the reproducible primary study and identify them explicitly as fixed query selection in manifests and reports.
- [ ] Move active extraction to an optional extension in the paper. If retained as an evaluated claim, implement diversity and student-uncertainty selection, retrain the student, update subsequent choices, and run the complete feedback loop against each compared defense.
- [ ] Validate schedule reproducibility. For any active extension, retain selection and model-update history that demonstrates choices changed using the updated student. Never label fixed replay as adaptive extraction.

### 15. Generate realistic task-matched benign clients

- [ ] Paper: build disclosed client generators covering specialist, repetitive, batch, bursty, and short accounts, with varied lengths, topic concentration, repetition, and arrival times.
- [ ] Preserve short accounts even when they never complete a scoring window. Retain client and source identities through generation, and keep consecutive-row slicing available only for plumbing checks.
- [ ] Validate generated workload distributions and role separation. Include concentrated benign specialists in false-positive and utility results because their behavior can resemble extraction.

### 16. Support rare-event claims with independent samples

- [ ] Hybrid: pursue the paper's target of about 3,000 independent benign calibration clients and 3,000 independent benign test clients per endpoint, with enough independent score-calibration windows, where the available data support those counts.
- [ ] Report actual client, source-document, and window counts, dependence assumptions, uncertainty, and the resolution supported by the sample. Narrow rare-event claims when independent data are insufficient; do not replace independent clients with correlated pseudo-accounts.
- [ ] Validate that reports distinguish the existing 22-account and 107-window smoke run from an adequately sampled study, including targets that cannot be resolved by the available sample.

### 17. Limit robustness evaluation to the declared tasks

- [ ] Hybrid: evaluate task-matched benign robustness and a later-time holdout where source timestamps support one. Record timestamp provenance and split rules; disclose when the available data cannot support a genuine temporal holdout.
- [ ] Keep WildChat/LMSYS general-chat studies outside this implementation matrix and remove corresponding manuscript promises. They are not dependencies of the task extraction study.
- [ ] Validate source/client separation for each retained task holdout. Report only task-matched robustness supported by independent data and the frozen protocol.

### 18. Keep story families in one role

- [ ] Hybrid: retain deterministic exact deduplication and add conservative AG News story-family grouping using available metadata and audited similarity matches. Group derivatives, near duplicates, and paraphrase families together where identified.
- [ ] Freeze and version grouping before evaluation, carry family identities through role separation checks, and record the matching procedure and audit decisions. Disclose residual semantic overlap.
- [ ] Validate known rewritten-story families and review false merges. Regenerate affected task splits and downstream artifacts when family assignments change.

### 19. Bound upstream artifact reproduction

- [ ] Repo first: keep the account-policy experiment as the main contribution and define upstream reproduction as a separately bounded check.
- [ ] If retaining the reproduction claim, add the pinned upstream artifact, a reproduction manifest, and a runner for all fourteen declared pairs and adapted baselines. Preserve upstream settings and disclose adaptation details and deviations.
- [ ] Otherwise remove the promised reproduction table from the paper and describe the detector score as an implementation based on prior work. Validate any retained reproduction claim against complete, inspectable results rather than local plumbing checks.

### 20. Add an alert-only aggregate monitor

- [ ] Paper, secondary priority: after account-level campaigns work, add a separate aggregate monitor over 1,500-request windows and record campaign alert outcomes, especially for Sybil schedules.
- [ ] Define window construction and alert calibration separately from account enforcement. An aggregate alert must not automatically block or escalate individual accounts.
- [ ] Validate an aggregate shift spread across accounts and confirm enabling the monitor leaves account enforcement decisions unchanged.

### 21. Train released-token-matched controls

- [ ] Paper, high priority: automate both matched-volume and matched-prefix controls using released response tokens from the corresponding defended campaign. Match tokens rather than selected prompt counts.
- [ ] Freeze the selection rule for each control, pair task, base model, training recipe, and seeds with the defended run, and record the target token count, achieved count, and residual mismatch.
- [ ] Train and evaluate both controls alongside attacker and gold-reference adapters. Validate partial campaign delivery and token-boundary mismatches so reports can distinguish selective supervision removal from returning fewer tokens.

### 22. Recalibrate ablations and sensitivity experiments

- [ ] Hybrid: prioritize instantaneous versus accumulated evidence, no decay, direct blocking, and a small window-size sweep. Freeze the runnable matrix before final testing.
- [ ] Recalibrate each variant using its own score and policy calibration on the separated benign roles and the declared operating targets. Record thresholds and achieved rates so inherited thresholds do not confound the comparison.
- [ ] Treat overlapping windows, proof-of-work hardware and difficulty, long-input stress, and reference contamination as secondary experiments. Validate variant identities and calibration provenance, and remove unrun variants from the paper's promised matrix.

### 23. Derive escalation and supervision from one event ledger

- [ ] Paper, high priority: create a campaign event ledger linking attempts, account identities, decisions, queueing, challenges, inference, response releases, timestamps, and released token counts with stable identifiers.
- [ ] Define and derive `J_ell`, `Q_ell`, censoring indicators, and campaign totals from that ledger. Distinguish reaching a level from receiving supervision, and account for requests still pending at budget or deadline termination.
- [ ] Add the declared survival analysis with explicit event definitions and censoring rules. Keep existing traffic stop metrics as operational diagnostics, and validate challenged-then-released, refused, pending, and right-censored campaigns.

### 24. Report defense capability and validate reduction metrics

- [ ] Paper for the narrow study: establish positive undefended transfer on development data before freezing the defense experiment. Record the learnability gate and its result without using final test outcomes for recipe selection.
- [ ] Implement defended capability and `R_D` using each task's primary metric, with explicit validity checks for its reference scores and denominator. Retain raw scores and explain why the ratio is unavailable when its interpretation fails.
- [ ] Add budget curves and cost to a declared capability target, reporting targets not reached within observed budgets. Validate positive, zero, and negative undefended transfer cases.

### 25. Score legitimate task quality and unconditional utility

- [ ] Hybrid, high priority: retain released legitimate responses and task references so narrow-task requests can receive task-aware scoring alongside operational traffic metrics.
- [ ] Report served-only quality with its denominator and unconditional utility over every submitted legitimate request. Assign zero utility to refusals, timeouts, and abandonment, including responses that miss the declared client deadline.
- [ ] Validate incomplete and deadline-missed campaigns, including no served responses. Restrict utility claims to the declared task endpoints and their task references; general-chat quality evaluation is outside this scope.

### 26. Respect clustering and separate seed variation

- [ ] Paper: implement paired source-document bootstrap intervals for task scores and client/campaign resampling for traffic outcomes. Preserve the intended pairing and resampling unit in exported results.
- [ ] Report each seed's result, the across-seed mean, and SD separately from sampling uncertainty. Keep existing item-level intervals only as descriptive supplementary statistics.
- [ ] Apply Holm correction only when conducting the declared family of formal tests. Validate grouped documents, repeated requests from the same client, paired campaigns, and missing seed pairs without treating correlated items as independent.

### 27. Measure serving overhead under matched load

- [ ] Hybrid: add focused timing for embeddings, completed-window scoring, scoring-trigger requests, time to first token, overall request latency, throughput, and scoped peak memory.
- [ ] Run one controlled matched-load experiment with and without the defense, recording hardware, load, warmup, concurrency, and measurement boundaries. Distinguish component compute costs from intentional queueing and challenge delays.
- [ ] Validate that scoring costs attach to the requests that trigger them and that peak memory covers the declared serving components. Keep this experiment bounded until defense campaigns work.

### 28. Link protocol, resources, supervision, and results

- [ ] Hybrid: add a unified study manifest reusing existing hashes and manifests through immutable identifiers. Link the declared matrix, protocol and serialization, source grouping, campaign settings, calibration, solver details, released supervision and tokens, model fingerprints, adapters, and evaluation results.
- [ ] Record measured resource costs with their measurement methods and mark unavailable measurements explicitly. Do not infer actual time, memory, hardware utilization, or cost from configured settings.
- [ ] Validate artifact links and identity consistency, report missing artifacts, and make the declared study's result artifacts inspectable. Ensure a result can be traced back to its exact campaign, released training data, calibration, and adapter.

Implement shared serialization, client/source identities, and the campaign event ledger before regenerating calibration artifacts or running defended campaigns. Enforce admission, budgets, and challenge handling before interpreting schedule results. Prioritize released-token controls, defense capability, and legitimate utility once campaigns work. Define experiment dimensions and training fingerprints before generating evaluation configs and study reports. Freeze the task/model matrix and corresponding paper scope before final testing; add only declared task-matched secondary experiments whose prerequisites and independent data support them. The retired general pipeline and general-chat evaluation are outside this checklist. Update related existing tests and documentation as each item lands, and regenerate affected artifacts and reports before using them for paper claims.

### Repository audit findings (2026-10-16)

A comparison of `paper/main.tex` with the merged harness, `just distill --mode tasks`, and `just start --mode tasks` found the gaps below. Each discrepancy has one owner. Where the paper's protocol is sound and the repository diverges, the fix is listed here. Where the repository cannot support a promise, the fix is a manuscript change, listed under [manuscript changes decided against the repository](paper-revision-todo.md#manuscript-changes-decided-against-the-repository-2026-10-16) and only cross-referenced here.

Progress against the numbered items above, none of them complete:

- Item 10: `apps/harness-api` enforces decisions before victim inference over HTTP with account keys, and `harness replay` drives the same policy offline. No campaign simulator calls the policy directly.
- Item 12: `distillation/campaign.py` solves challenges and records per-prompt challenge counts and solve seconds. It does not record hashes attempted, solver hardware, or queue and release wait.
- Item 20: `harness/monitor.py` and its calibration exist, but calibration skips the monitor at the current task-pool sizes. The paper declares the monitor not run; see below.
- Item 23: `campaign.events.jsonl` and the harness decision log exist, but `J_ell` and `Q_ell` are not derived from them.
- Item 27: `harness bench` times the harness's own units; the matched-load victim comparison is still missing.
- Item 9: `_trained` now rejects reuse when the dataset hash or recorded config differs, but tokenizer, template, and model-revision identity are not part of the fingerprint.
- Task ledgers: `just start --mode tasks` and `just distill --mode tasks` now validate every planned ledger before starting the teacher, and `distill ledger repair` recovers damaged rows. See the re-query note under Limitations.

Blocking, before any defended campaign:

- [x] Shuffle the legitimate pools before forming calibration accounts and windows. `harness.calibration.pooled_accounts` now orders each role by a seeded hash before slicing, in `just start --mode tasks` and `harness pool-accounts`, and `just start --mode tasks` refuses a cached calibration built from other accounts. On the SQuAD build, no score- or policy-calibration window is single-article any more; each spans 25 to 46 articles. The original finding: `build_task` writes each role in dataset order, and SQuAD's training split is ordered by article. As a result, the 8,079 SQuAD policy-calibration rows form only 35 article runs, and each 500-row pooled account covers 1 to 5 articles. In score calibration, 39 of the 82 consecutive 100-request windows come from a single article, so the benign null distribution consists largely of single-topic windows. References are drawn with `rng.sample` and budget prompts are ordered by a seeded hash, so both are broad. A shuffled extractor therefore scores below the benign null on SQuAD and should never escalate, whatever the defense is worth. AG News, where every row is its own group, is unaffected. Topic-concentrated specialists should still be built deliberately, through the item 15 generator.
- [ ] Run the detectability check. The attacker and legitimate pools are hash-split halves of one training split (`tasks.py`, `_role`), so even after shuffling, a pure extractor should match the benign references in expectation. The benign client mix also biases the check: with the v1 equal split, about two thirds of AG News score-calibration windows came from repetitive or specialist clients, whose concentrated windows score far above any broad window, so a uniform extractor gets p-values near 1. `just diagnose` now recalibrates for each `--concentrated-shares` value (default 0, 0.1 and 0.4) and records, per client type, how the extractor's peak level and p-values compare with held-out benign accounts. `just start --mode tasks` stops before collection when no MMD defense separates the extractor, unless `--allow-undetected` is passed. The result decides the [RQ2 framing](paper-revision-todo.md#manuscript-changes-decided-against-the-repository-2026-10-16) in the paper.
- [ ] Replace the pooled calibration accounts in `study.py` (shuffled 500-request slices from `pooled_accounts`, no timestamps) with the item 15 client generator. The current construction gives about 22 AG News, 17 SQuAD, and 2 GSM8K policy-calibration accounts, so the 1% L4 threshold is the benign maximum. The pools support at most about 108, 80, and 6 scored accounts of 100 requests, so the paper narrows its client-count claims (item 16). `rate_per_minute` is never fitted because no request has a time; the paper drops the rate-rule row rather than fitting it to synthetic timing.
- [ ] Make legitimate traffic task-matched. `just start --mode tasks` freezes one generic chat suite (`benchmarks/traffic.py`, `GENUINE_PROMPTS`) shared by all tasks; its 48 requests across 4 genuine actors never complete a 100-request window, and responses get no task score. Item 25 needs task-matched benign accounts drawn from `legitimate/benign_test.jsonl` and scored with each task's metric.
- [x] Consume `legitimate/benign_test.jsonl`. It is written by `distill tasks build`, but nothing reads it, so no held-out account FPR exists. Replay it through each frozen policy. Done: `just start --mode tasks` replays it through every defense into `TASK/benign-test/ARM.json`, on the same account construction as calibration.
- [x] Embed only the detector serialization (item 1). `harness.serialization.detector_text` (`detector-v2`) now feeds calibration, replay, the policy, bench, and the API; artifacts record the version and per-role truncation counts, and the API refuses a calibration for another task. Before this, `harness_api.app.current_turn` embedded the whole user turn, including the fixed task instruction from `AG_NEWS_PROMPT`, `SQUAD_PROMPT`, and `GSM8K_EVAL_SUFFIX`. For SQuAD, put the question before the passage in the detector serialization: `SQUAD_PROMPT` places it last, so BGE's 512-token truncation drops the question first on long contexts. Record the encoder's `truncated` count per role in the calibration artifact.
- [x] Default `HARNESS_L4_ACTION` and `harness replay --l4-action` to `refuse`. The paper's L4 refuses, and served answers stay unchanged; only `just start --mode tasks` defaults to `refuse` today. Keep `deceive` as an unreported option. No manuscript change is needed unless a noise arm is added later. Done: both default to `refuse`.
- [x] Apply L2 spacing on entry (item 3). `MMDPolicy._respond` updates `last_release` only at L2 and above, so the first L2 request ignores the previous L0 or L1 admission. Done: L0 and L1 admissions set the last release.

Required for Table VI and the primary contrast:

- [x] Add the missing defended arms to `just start --mode tasks`. `study.py` hardcodes `HARNESS_ACCUMULATOR=paper` and `HARNESS_POLICY=mmd`. Table VI and the primary contrast also need `instantaneous` and `quota` campaigns, each with its own ledger, adapter, and evaluation. Done: `--arms full,instantaneous,quota`, each under `protected/ARM/`.
- [ ] Add the development sweep over learning rate {1e-4, 2e-4} and epochs {1, 2, 3}. Six runs per core task on the undefended attacker set at B_A = 2,000 and seed 42, scored on `development-suite.json`, are cheap next to the main matrix. Freeze the winner in each task's generated config. The paper then names that sweep set, rather than this checklist fixing 2e-4 and 3 epochs without selection.
- [ ] Wire the development transfer gate (item 24) into `just start --mode tasks`, so defended campaigns stop or are flagged when the undefended development gain is not positive.
- [x] Charge failures to the campaign instead of aborting. `ProtectedTeacher.generate` raises on any non-block error, and `_collect_one` retries up to three times without charging retries to B_A (item 11). Done for defended campaigns: failed attempts are charged and never retried; undefended collector retries remain a disclosed re-query.
- [ ] Add the study-level report: `G_D`, `Delta_D`, `R_D` with its validity rule, per-seed results with mean and SD, budget curves, and cost to target. `reporting.paired` bootstraps independent items; replace it with source-document clusters for SQuAD (item 26).
- [x] Add a direct-block mode to `MMDPolicy`, which is one of item 22's priority ablations and needs only an L4-only threshold set. The paper drops the stride row instead of the repository adding rolling windows. Done: `MMDPolicy(actions="direct")`, `HARNESS_ACTIONS`, and `harness replay --actions direct`.

Bookkeeping:

- [x] Add a shared GSM8K/coding cap-pilot command: `distill tasks pilot --task-dir PATH --output-dir PATH --teacher-url URL`. It selects up to 200 development-suite items, records finishes, and exits nonzero above 2% length finishes or on inconclusive evidence. Real teacher pilots remain pending. GSM8K remains in the launcher defaults as the math capability distillation corpus; its cap remains provisional pending this pilot.
- [x] Record the reference-window count per task in the run manifest and calibration report. `study.py` uses `min(20, fitting // 100)`, which gives 20 for AG News and SQuAD and 13 for GSM8K. Keep references disjoint rather than resampling; the paper discloses the per-task count. Done: `TASK/calibration-report.json` records it with truncation counts and thresholds.
- [x] Record the aggregate monitor as skipped in every calibration artifact and report. It needs 30,000 fitting requests, but AG News has 21,558, SQuAD 13,170, and GSM8K 1,328, which is less than one window. The paper declares the monitor not run. Done: the calibration report records `aggregate_skipped`.
- [x] Keep `e_process` and `e_cusum` out of `just start --mode tasks` outputs and every reported table. The paper does not declare them, and no change is planned there. Done: study calibrations fit only `paper`, `instantaneous`, `paper_no_memory`, and `paper_no_decay`.
- [ ] Join noise flags into the protected ledger, in case `deceive` is ever used. Under `HARNESS_L4_ACTION=deceive`, noise responses count as received supervision in `campaign.json`; mark them from the decision log as the traffic runner does.
- [x] Include GSM8K by default as the math capability distillation corpus. Both launchers default to `ag_news,squad,gsm8k,coding`. The output-cap pilot remains a separate pending requirement. All selected tasks default to their full attacker pools. Explicit budgets above a task's pool are skipped, and the resolved plan is frozen for reporting.
- [x] Fix stale text: the `configs/train.task.toml` comments still say the paper uses Qwen2.5 and attention-only LoRA targets, and `docs/harness.md` lists the campaign layer as not implemented, although `campaign.py` and `just start --mode tasks` exist. Done.

## Limitations

- The Colab teacher serves an 8-bit `Qwen3.5-9B-Q8_0` build, while the paper names `Qwen3.5-9B` without a precision. Record the served precision and either disclose it in Table V or collect from the BF16 weights.
- `distill collect` retries failed requests, and `distill ledger repair` sends quarantined IDs again on resume. That suits the undefended baseline only if the extra requests are counted: report retries and re-queried IDs per ledger rather than treating the ledger as the attempt count. Defended runs need a client that counts every attempt against B_A and solves proof-of-work challenges.
- The legitimate pools are task-matched prompt sets, not client traces. Grouping them into accounts and timing them is part of the harness evaluation.
- Grouping AG News by exact text keeps duplicates together but not paraphrases or rewrites of the same story.
