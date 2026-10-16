# Paper revision TODO

Track manuscript changes separately from the [implementation checklist](paper-tasks.md#implementation-todo). Item numbers below refer to the supplied 28-item review. The unchecked entries describe manuscript revisions and evidence still to supply. Completed repository cleanup is recorded separately in the implementation checklist; it does not mark a manuscript revision or experiment complete. Freeze scope and methods before final testing, then fill results from verified artifacts. Repository paths and adapter-directory fixes belong in the implementation checklist unless they change a protocol or provenance claim.

## Agreed task-only scope

The repository now supports one distillation workflow, `just paper-tasks`, with a separate adapter for each task, query budget, seed, and supervision condition. The general-corpus builder, launchers, authored/generated prompt families, broad evaluation suite, correctness-filtered review, and their configs have been removed. This checklist records manuscript work still to do; the manuscript itself is not present in this checkout.

The paper's core is AG News and SQuAD extraction under an account policy, with paired undefended/defended students, a base student, a same-prompt gold-label reference, and released-token controls. GSM8K is a generative extension whose inclusion must be frozen before final testing. Required claims concern per-task extraction, campaign costs, and task-matched legitimate utility. Retain only declared secondary schedules, ablations, alerts, and overhead measurements supported by completed evidence.

General-corpus training, general instruction/chat capability, broad knowledge/coding evaluation, correctness-filtered attacker supervision, cross-task model rankings, teacher-gap closure, and WildChat/LMSYS general-chat or utility studies are outside the paper's scope. The remaining TODO entries must not introduce those studies as prerequisites. See the [completed cleanup and implementation scope](paper-tasks.md#agreed-scope-and-completed-cleanup) for repository status.

- [ ] Rewrite the abstract, introduction, and contribution list around budgeted extraction from task APIs and the access policy's effect on extraction and legitimate utility. Remove promises to produce a general instruction/chat model or demonstrate general model quality.
- [ ] In Section IV-B, state that AG News classification and SQuAD extractive QA are the core endpoints. Retain GSM8K only if the generative extension will be run, and freeze that choice before final testing. Remove the standard mixed corpus, quick/standard/full profiles, broad prompt counts, authored/generated families, and general training recipe.
- [ ] Describe the actual task recipe: rank-16 LoRA, alpha 32, dropout 0.05, 2,048-token training limit, effective batch size 16, and development-selected learning rate and epochs. Explain that the task recipe has no additional victim-response validation split. Remove rank-128, 3,072-token, two-epoch and best-validation-checkpoint claims from the retired recipe.
- [ ] Specify query budgets 500, 2,000, and 8,000 where supported, paired seeds 42, 1, and 20, and nested query sets. GSM8K supports 500 and 2,000 from its current attacker pool. With three seeds and both attacker/gold students, the core AG News/SQuAD matrix is 36 training runs. Retaining GSM8K adds 12, for 48. These counts exclude defended students and matched supervision controls and are not counts of completed experiments.
- [ ] State fixed output caps of 16 tokens for AG News, 64 for SQuAD, and provisionally 256 for GSM8K. Document the development pilot and freeze any cap change before main collection.
- [ ] Correct the attacker model: gold answers are unavailable to attacker review. Wrong and malformed complete answers remain in training; collector failures, truncations, empty responses, and records exceeding training length are excluded and counted. The gold-label reference trains separately on the same prompts. Do not call it a guaranteed upper bound.
- [ ] Replace the broad benchmark list and executable coding pass@1 claims with AG News accuracy, SQuAD token F1 with secondary exact match, and GSM8K numeric exact match. Report each task separately. Remove overall general-model rankings, broad macro scores, and teacher-gap-closure figures.
- [ ] Update Table VI and its caption to identify task, victim/student identities, supervision condition, defense, budget, query/training seeds, and actual adapter. Include base, undefended, defended, and gold-reference scores plus matched supervision controls. Declare `R_D` references and validity rules independently of teacher-gap closure.
- [ ] Resolve model provenance before final runs. The draft names Qwen2.5-3B-Instruct and Qwen2.5-0.5B/1.5B-Instruct, while default configs use Qwen3.5-9B and Qwen3.5-2B-Base, and the current Colab collection uses Qwen3.5-9B-Q8_0. Either align the implementation with the draft or revise the draft to the exact deployed models, revisions, quantization, templates, and adapter targets.
- [ ] State the limits of the claims: narrow task transfer does not establish general-chat extraction resistance. Define legitimate utility on each task, including nonresponses and deadlines. Keep synthetic traffic smoke checks separate from capability evidence.
- [ ] Remove retired command/config references from appendices and reproducibility instructions. Use the task manifests, generated per-run configs, task ledgers, adapters, and per-task evaluation configs. Report only measured results and completed controls.

## Scope and experiment promises

- [ ] Items 10 and 28: describe a controlled account-policy experiment with explicit, stable account identities. Distinguish simulator results from live HTTP service results and state the account-binding assumptions. Limit authentication claims to what the experiment actually implements.
- [ ] Items 13 and 14: declare pure extraction, within-account mixtures, slow traffic, and fixed-total-budget Sybil campaigns as the primary schedule matrix. List paraphrases, benign-matched adaptation, and resets as secondary experiments, retaining only schedules that will actually be run.
- [ ] Item 14: describe seeded query schedules as fixed selection. Move diversity/uncertainty active learning to an optional extension unless the complete select, train, update, and reselect loop is implemented and rerun against every compared defense. Remove any claim that fixed replay demonstrates adaptive extraction.
- [ ] Item 17: limit robustness claims to the declared task endpoints and source/client-separated holdouts. State whether timestamp provenance supports a genuine later-time holdout. Remove WildChat client-proxy and LMSYS general-chat shift-test promises from the study matrix and claim list.
- [ ] Item 19: keep the account-policy study as the main contribution. Either supply a faithful pinned reproduction across fourteen pairs and adapted baselines, or remove the promised reproduction table and describe the score as an implementation based on prior work. Disclose deviations for any retained reproduction.
- [ ] Items 20 and 22: declare the aggregate monitor as secondary and alert-only. Prioritize instantaneous versus accumulated evidence, no decay, direct blocking, and a small window-size sweep. Mark overlap, proof-of-work hardware/difficulty, long-input stress, and reference contamination as secondary, and remove unrun experiments from the promised matrix.

## Detector inputs, splits, and benign workloads

- [ ] Items 1 and 9: specify detector serialization separately from victim instructions and use it consistently across fitting, calibration, and testing. Identify the serialization version in the protocol.
- [ ] Items 2 and 18: describe disjoint clients and source groups across fitting, score calibration, policy calibration, and final benign testing. Document exact deduplication, conservative story-family grouping, similarity audits, when grouping was frozen, and residual semantic overlap.
- [ ] Item 15: disclose benign client generation and distributions of account length, topic concentration, repetition, and timing. Include specialist, repetitive, batch, bursty, and short clients. Identify consecutive-row accounts as plumbing fixtures and retain short accounts in utility denominators.
- [ ] Item 16: replace sample-size aspirations with actual independent client, source-document, and window counts in results. Pursue the stated thousands of clients where independent data support them; otherwise report uncertainty and narrow rare-event claims. Explain why correlated pseudo-accounts do not provide independent evidence, and label the 22-account/107-window run as a smoke check.

## Enforcement and attacker costs

- [ ] Items 3 and 10: specify decision, queue, challenge, victim-inference, and response-release ordering. State that L2's one-second spacing applies immediately on entry relative to the previous admitted request across levels.
- [ ] Items 11 and 13: define campaign-wide `B_A` and `T_A`. Charge retries, filler, and failed attempts to the attempt budget, and share total budgets across Sybil accounts. State how pending requests and late responses are treated at termination.
- [ ] Item 12: state that attackers solve challenges and continue. Report hashes attempted, solver hardware, solving time, queue time, and released responses. Include waiting and solving in campaign costs; receiving a challenge alone is not evidence that extraction stopped.
- [ ] Items 13 and 20: define reset semantics and which state persists. Describe aggregate 1,500-request window construction, calibration, and alert metrics separately from individual account enforcement. Do not use an aggregate shift alone to justify account blocking.

## Supervision controls and outcome definitions

- [ ] Item 21, high priority: add both matched-volume and matched-prefix controls based on released response tokens. Specify their selection rules, paired training conditions, achieved token counts, and residual mismatches. Use the controls to assess selective supervision removal separately from reduced supervision volume.
- [ ] Item 23, high priority: define `J_ell` and `Q_ell` precisely, including their units, origins, level-reach rules, and whether counts concern attempts, admissions, or released outputs. Derive them and campaign totals from the event ledger, separating escalation delay from supervision actually received.
- [ ] Item 23: declare censoring and survival-analysis rules, including campaigns ending before escalation and outputs pending at the deadline. Keep operational traffic stop metrics separately identified.
- [ ] Items 4 and 20: mark nonexistent baseline L2/L3 outcomes as N/A in tables and figures, and distinguish aggregate alerts from account-level outcomes.

## Capability and legitimate utility

- [ ] Items 5 and 6: define primary capability metrics as AG News accuracy, SQuAD token F1, and GSM8K numeric exact match. Retain SQuAD exact match as secondary. State valid-output rules, count unusable outputs as incorrect in task scores, and give valid-pair agreement denominators and coverage.
- [ ] Item 24, high priority: declare a development-only positive-undefended-transfer gate before final defense testing. Define `R_D`, its reference scores, validity checks, and interpretation. Retain raw defended and undefended scores when the ratio is uninterpretable, and omit the retired broad-demonstration teacher-gap-closure metric.
- [ ] Item 24: add budget curves and cost to a predefined capability target. Report when a target is not reached within observed budgets rather than assigning an unsupported finite cost.
- [ ] Item 25, high priority: report legitimate served-only task quality and unconditional utility over all submitted requests to each declared endpoint. Specify task references, deadlines, denominators, and zero utility for refusals, timeouts, and abandonment. Remove general-chat quality or utility claims.

## Paired analysis and uncertainty

- [ ] Items 7 and 26: list defense, schedule, budget, query-order seed, training seed, task, and model as explicit experiment dimensions. Describe pairing and study aggregation, report missing pairs, and retain per-run results.
- [ ] Item 26: use paired source-document bootstrap intervals for task scores and client/campaign resampling for traffic outcomes. State resampling units and dependence assumptions. Report each seed, mean, and SD separately from sampling uncertainty; identify item-level intervals as descriptive supplementary statistics.
- [ ] Item 26: declare the formal-test family before testing if using one, and apply Holm correction only to that family. Otherwise remove an unfulfilled formal-testing promise.
- [ ] Item 22: state that every ablation is recalibrated on separated benign data at the declared operating targets. Report achieved rates and calibration provenance alongside variant comparisons.

## Serving overhead and reproducibility

- [ ] Item 27: describe one bounded matched-load experiment and its hardware, load, warmup, concurrency, and measurement boundaries. Report embedding time, completed-window scoring time, trigger-request latency, time to first token, overall latency, throughput, and scoped peak memory. Distinguish compute overhead from policy-induced waiting and challenge costs.
- [ ] Items 8, 9, and 28: identify the exact trained model and attacker, gold-reference, defended, or matched-control adapter used for every evaluation. The repository's default adapter paths have been corrected, but the manuscript still needs the complete condition-to-artifact mapping. Document fingerprints covering data, immutable model revision, resolved training settings, and serialization, including tokenizer/template identity where relevant. Do not imply dataset hashes alone establish model provenance.
- [ ] Item 28: link the declared protocol and matrix to calibration, campaigns, solver details, released-token supervision, controls, adapters, and results through an immutable study manifest. Publish or otherwise make the referenced artifacts inspectable, and report missing artifacts and unavailable measurements explicitly.
- [ ] Items 16, 19, 24, 27, and 28: audit the abstract, contribution list, tables, figures, and conclusion against completed experiments. Replace configured or estimated resource claims with measured values where claimed, label smoke checks, and remove unsupported reproduction, rare-event, adaptation, robustness, or deployment claims.
