# Multi-domain harness effectiveness study

The study asks which distillation campaigns an account-level harness restricts, whether those restrictions prevent useful capability transfer within a declared resource envelope, and what legitimate users lose. One gateway serves AG News, SQuAD, and GSM8K. Each campaign trains one student on all the responses it actually receives. The same strategy runs without a defense, with instantaneous MMD, with the full policy, and with a simple query quota. The attempted-request and time limits define a fair comparison; budget curves are secondary.

Status: the mixed benign generator and fixed A1/A2/A5 schedule builders are implemented. They produce hashed input artifacts. Gateway calibration, multi-account execution, enforcement-aware replay, combined student training orchestration, and campaign reporting remain future phases. The current `just start` still runs separate task studies, defaulting to AG News, SQuAD, and GSM8K with full attacker pools. GSM8K supplies the math capability distillation corpus; its output-cap pilot remains pending. No gateway defense effectiveness has been measured.

## What the paper can claim

The [manuscript](../paper/main.tex) now uses the title "Evaluating an account-level harness against multi-domain model distillation". Its abstract, threat model, methods, result templates, and conclusion follow this protocol. It reports the input generators as implemented and gateway execution and effectiveness results as pending.

The local compiled copy is `paper/main.pdf`, which is gitignored. Regenerate it from the repository root with `tectonic --outdir paper paper/main.tex`; auxiliary files stay temporary under Tectonic's default output settings.

The manuscript asks:

1. Which campaign strategies lose useful learned capability behind the harness relative to their paired undefended runs?
2. How much usable supervision escapes before intervention, and does the harness prevent transfer within the declared envelope or primarily delay collection?
3. What are the false-escalation, completion, latency, and unconditional task-utility costs for specialist and generalist users?
4. What does accumulated MMD add beyond instantaneous MMD and a simple quota at comparable benign cost?

The revised manuscript proposes a primary envelope of 6,000 total HTTP submissions and 24 hours, a balanced candidate inventory of 2,000 prompts per domain, six accounts for A5, and a 10% generalist account share. These are proposed settings to freeze before final testing. It also specifies one shared development-selected training recipe, per-domain transfer gates, and numeric domain success targets selected on development evidence. The target values and final matrix remain unresolved. Older full-pool per-task budgets describe the existing task launcher, not the proposed gateway study.

Detection, response restriction, and reduced learned capability are separate observations. An alert or a blocked account does not establish that a campaign failed. The undefended reference must demonstrate positive transfer first. Call it a successful undefended extraction baseline, rather than "perfect distillation". Task accuracy and agreement with the victim measure different extraction objectives, so keep both with their explicit denominators. This distinction follows [Jagielski et al., USENIX Security 2020](https://www.usenix.org/conference/usenixsecurity20/presentation/jagielski).

Multiplexing, domain blocks, and account splitting are collection strategies under the same response-supervised learning procedure. Describe them as different distillation campaigns. Claims about different learning algorithms require additional attacker training objectives. Adaptive collection also differs from adapting training on already collected responses. A recent preprint reports that adaptive student training recovers capability missed by passive evaluation in its response-side defense setting. That motivates a stronger attacker extension here, but does not predict this harness's results. See [The Distillation Game, revision 4](https://arxiv.org/abs/2605.22737v4).

## Endpoint, data, and identity

Use one unchanged teacher and one gateway endpoint. Preserve each request's domain-specific prompt, output cap, non-thinking setting, and pinned teacher identity. Use the existing task pools and frozen test suites. Coding is a later domain and requires its own executable evaluation protocol before inclusion.

Use one LoRA adapter on `Qwen3.5-2B-Base` per campaign, defense, and training seed. Start each adapter from the same base checkpoint. Combine received supervision across domains and shuffle training examples with a declared seed so collection order is not silently reused as a training curriculum. Preserve prompt/source identity and record domain counts and supervised tokens before and after preparation. A common 2,048-token limit needs measured per-domain exclusions; concatenating datasets does not establish that every example fits.

For this controlled gateway, domain metadata comes from the declared task interface and validated request serialization. Do not treat a client-supplied domain name or a substring inside arbitrary text as trustworthy. A later gateway mode must validate supported templates and define the handling of unmatched requests. Fixed service instructions are excluded from detector text, while task content remains. Task content itself can identify domains, so stripping instructions does not eliminate distribution differences.

Account identities are stable experiment identities, not proof of a production identity system. Account state spans domains within a policy horizon. A Sybil campaign shares one resource envelope and pools every received response across its accounts.

## Benign traffic and calibration

Domain profile and request style are separate dimensions. Specialists use one domain; generalists use at least two. Either can be repetitive, batch, bursty, or short. The benign CLI supplies `task_references(...)` for every selected domain. AG News sources are ordered by news label and SQuAD by article title. GSM8K references all have category `math`, so its specialists remain domain-only rather than math-subtopic controls. The manifest records `domain_topic_categories` and `specialist_scope` so that limitation is visible. These are synthetic workloads, not measured user populations.

Calibrate and test on source-separated fitting, score-calibration, policy-calibration, and benign-test pools across all domains. Generate each role separately. Record the requested and achieved generalist account fractions, domain request fractions, account lengths, unique prompts, source groups, and scored-account counts. Repeated prompts and shared source documents do not create independent evidence.

Use generalist shares such as 0%, 10%, and 30% as declared sensitivity settings. Freeze the primary share before final testing. Report both matched-distribution testing and an explicit shift test where the held-out share differs from the calibration share. Account share and request share differ when accounts have unequal lengths.

A global mixed reference may make a legitimate specialist look more unusual than a multiplexed attacker whose domain proportions match the reference. Same-distribution extraction therefore remains an essential failure control. Neither A1 nor A2 is guaranteed to trigger the detector, and accumulated evidence cannot recover an intent signal absent from its observations. A2 is a test of ordering and window composition, not an assumed demonstration that memory detects domain switching. These are consequences of the request-distribution score specified in [the manuscript](../paper/main.tex) and [the current policy](../packages/harness/src/harness/policy.py).

Report achieved benign error rates and uncertainty at the available independent sample counts. Do not promise a validated 1% false-block rate from a small synthetic account pool. Calibrate quota and instantaneous baselines on the same benign roles; show their achieved utility costs as well as nominal operating targets.

## Campaign matrix

| ID | Campaign | Account and ordering rule | First-phase status |
|---|---|---|---|
| A1 | Multiplexed | One account; seeded interleaving of the selected domain prompts | Generator implemented |
| A2 | Domain-sequential | One account; one seeded block per domain | Generator implemented |
| A3 | Cover traffic | Multiplexed extraction plus declared cover traffic; every received response can train the student | Deferred |
| A4 | Slow | Multiplexed extraction at a declared synthetic rate | Deferred |
| A5 | Sybil per domain | A declared total of k accounts partitioned across domains; each account specializes | Generator implemented; execution deferred |
| A6 | Sybil random | Multiplexed requests split across k accounts | Deferred |
| A7 | Independent prompt sources | Domain-matched prompts from separately pinned public or synthetic sources | Deferred |
| A8 | Active coverage | Live selection, training updates, and reselection using the attacker's observable information | Deferred |

A1, A2, and A5 select identical prompt inventories when the seed, pools, and allocation match. Only order and account assignment change. A1 and A2 use exactly one account. A5 defaults to six accounts total across the campaign, requires at least one account per domain, and never multiplies the budget.

For three current domains, a balanced 6,000-query inventory contains 2,000 queries per domain. Seeds 42, 1, and 20 select overlapping inventories from the 3,389-prompt GSM8K pool; source IDs must be retained for clustered uncertainty across seeds. An 8,000-query inventory contains 2,667 AG News, 2,667 GSM8K, and 2,666 SQuAD queries under the builder's alphabetical remainder rule. The proposed 8,000 = 4 × 2,000 allocation applies only after a fourth domain exists. Pool capacity is validated before artifacts are written.

`B_A` counts all HTTP submissions, including proof redemption and failures. A schedule with `B_A` ordinary-query candidates is an inventory, not a promise to submit all of them. Proof submissions can displace later candidates. `T_A` includes waiting and solving. The generated timestamps specify intended arrivals; actual submission, admission, and response-completion times belong in the future campaign event ledger. A campaign crossing a policy horizon must retain its spent campaign budget while applying the documented account reset.

The current datasets do not provide natural request timing. A4 cannot claim a measured median benign rate until a source of actual timing exists. Fixed schedules remain fixed selection even if an execution client solves challenges or waits for admission. A8 requires each defense to receive a fresh live feedback loop.

## Outcomes and controls

For domain d, define base score S0,d, paired undefended score SU,d, and defended score SD,d. Report raw scores, defended gain SD,d − S0,d, and gain removed SU,d − SD,d. Define the fraction of gain removed as `(SU,d - SD,d) / (SU,d - S0,d)`. The retained fraction is `(SD,d - S0,d) / (SU,d - S0,d)`. Name these separately and suppress ratio interpretation when the undefended gain is negligible or uncertain. Keep the declared practical-size and uncertainty checks consistent in code and manuscript.

Use development-only runs to establish a learnable shared-student baseline and freeze a meaningful per-domain success threshold before final testing. Retain per-domain transfer gates even though training uses one adapter. A domain with no undefended learning is uninformative for defense effectiveness; retain its scores and label that limitation. Do not erase this check when replacing task-specific orchestration. A gold-label student remains a learning reference, separate from the black-box attacker.

Lead the results with campaign strategy, defense, per-domain student score and gain removed, observed extraction success, released supervision, and benign costs. Report success per domain. If a joint campaign success rule is needed, declare whether success means reaching every domain target or any domain target and show both component outcomes. Three seeds are three observed runs, not a precise probability estimate. Keep missing, failed, and censored runs visible.

`J_ell` remains the campaign attempt index at the first account intervention of level ell, accompanied by the triggering account and domain. Record per-account reaches too. Report released responses and tokens by domain before that event, plus campaign totals through termination. Do not reinterpret "one account blocked" as "campaign stopped", or give an unreached intervention a zero delay.

Train matched-volume and matched-prefix controls from undefended responses, matching within each domain. Preserve response counts and match measured released tokens as closely as whole responses allow, reporting residuals. This avoids attributing a different domain mix to selective removal of better training examples. These post-hoc supervision controls supplement the deployable quota comparison; they do not receive false-positive scores.

Pair benign arrival schedules across arms and report false escalation, completed-by-deadline fractions, p50/p95 latency, and unconditional task utility for each domain profile and request style. Distinguish overhead, throttling delay, and challenge cost. Add a fixed-attempt test with a relaxed deadline or a justified extrapolation boundary to distinguish delay from a lasting loss of accessible supervision. Never describe a censored capability target as permanently unreachable.

## Replay and cached responses

Caching is valid for unchanged victim outputs under pinned prompt identity, model revision and quantization, template, decoding settings, output cap, and relevant random seed. Existing ledgers can reduce collection cost only where that full contract matches. Shared source prompts do not by themselves establish response equivalence.

The current [harness replay](../packages/harness/src/harness/replay.py) records policy decisions and scheduled release times. It does not execute challenge solving, proof submissions, teacher inference, or response completion. Filtering cached responses on `action != refuse` would incorrectly count challenged requests as released and ignore the time budget.

Use existing replay for detection diagnostics first. A later enforcement simulator must track outstanding queries, queues, solver time and hardware, proof expiry, proof attempts, deadline cutoffs, and horizon resets. Validate its decision and release traces against the live HTTP gateway under a bounded set of representative campaigns before using it to derive defended training sets. Measured solver work and serving overhead still need runtime evidence. Adaptive collection and modified-response defenses need their own compatible execution path.

## Repository work and exit criteria

| Concern | Recommendation | Work and exit criterion |
|---|---|---|
| Main claim and result order | Paper | Lead with campaign outcomes and benign costs. Move detector reproduction and budget curves behind these results. Remove predicted attack rankings. |
| Mixed benign inputs | Repo | Implemented in `workloads.mixed_clients`; expose achieved account and request mixes and preserve source provenance. |
| A1/A2/A5 inventories | Repo | Implemented in `schedules.campaign_schedule`; paired selection, explicit account ownership, total budgets, and deterministic artifacts. |
| Shared gateway and calibration | Hybrid | Add a versioned mixed serialization and one calibrated account state across domains. Preserve existing task-mode artifact compatibility checks. Verify request-dependent caps and malformed-template handling through HTTP. |
| Multi-account execution | Repo | Consume the new inventory with one durable budget/deadline and authenticated account map. Charge proofs/failures and join releases by request ID. |
| Reuse of teacher outputs | Hybrid | Validate complete response identities and simulator/live equivalence before deriving defended supervision from cached outputs. |
| Shared student | Repo | Reuse response-token SFT; add combined-data preparation and one adapter identity per campaign/arm/seed. Measure length exclusions, token balance, and per-domain transfer gates. |
| Reporting correctness | Hybrid | The current `study_reporting.defense_effect` writes retained gain under `R_D`, while the paper defines removed gain. Reconcile names, formulas, and ratio-validity rules before publishing. |
| Budget and target identities | Repo | Fixed for task studies: `budgets.json` freezes actual per-task numeric budgets, and reports resolve legacy `full` protocols through frozen task manifests. `policy-horizons.json` records reset settings without replenishing campaign limits. Replace the hardcoded score target of 80 with the declared development-frozen thresholds. |
| Combined evidence | Repo | Join capability scores, per-domain supervision, event totals, matched controls, and profile-specific benign utility into campaign comparisons. Existing separate artifacts alone are not a completed result. |

Phase 1 exits when generators have deterministic output, validate pool capacity and domain identity, preserve source provenance, expose achieved profile composition, and keep A5 allocations inside one total budget. Validate the CLI on existing local pools and test its immutable output behavior. This phase produces no effectiveness claim.

Phase 2 adds the shared gateway, mixed calibration, and live multi-account campaigns. Its exit criterion is correct enforcement and event accounting through the HTTP path, including challenge continuation, pending responses, and deadlines. Phase 3 adds validated cached execution where appropriate, combined student training, per-domain evaluation, and token controls. Phase 4 runs the frozen primary matrix, sensitivity settings, and secondary campaigns, then fills the manuscript from those artifacts.

The proposed 48 primary adapters assume A1/A2/A5/A7 × four arms × three seeds. A7 is currently deferred. A1/A2/A5 alone plan 36 adapter conditions; identical undefended datasets may permit reuse only when the complete training contract matches. Neither count includes development recipe selection, matched controls, specialist references, sensitivity retraining, or A8. Declare those costs before freezing the matrix.

## Run the implemented input builders

Enter `devenv shell` first. Existing task builds are inputs and are not modified. Build any missing domain with `uv run distill tasks build --task DOMAIN`. The local checkout currently has AG News and SQuAD builds; GSM8K must be built before using the gateway generators' three-domain defaults. The separate-task launchers include GSM8K by default and build missing tasks automatically; its output-cap pilot remains pending.

```sh
uv run distill gateway benign --domains ag_news,squad --role fitting --accounts 20 --generalist-share 0.3 --requests-per-account 100 --output artifacts/gateway/fitting-g30
uv run distill gateway schedule --domains ag_news,squad --attack A1 --budget 6000 --seed 42 --output artifacts/gateway/A1-s42
uv run distill gateway schedule --domains ag_news,squad --attack A2 --budget 6000 --seed 42 --output artifacts/gateway/A2-s42
uv run distill gateway schedule --domains ag_news,squad --attack A5 --budget 6000 --accounts 6 --seed 42 --output artifacts/gateway/A5-s42
```

Each command creates `requests.jsonl` and `manifest.json` in a new directory and refuses to overwrite existing inputs. Each request retains its full `PromptRecord`, domain in `prompt.task`, source corpus and group, account, planned time, and frozen output cap. The manifest hashes the prompt pools and output file and records `execution_status="not_run"`. Generate all four benign roles separately from their corresponding task pools. The Python schedule API also accepts an explicit domain allocation that must sum to the total budget. These files are for the future gateway runner; they are not accepted by the current single-task collector or replay CLI.

## Phase 1 validation

`uv run pytest tests/test_gateway.py tests/test_tasks.py -q` passed all 14 tests. The gateway tests cover three domains, paired inventories, specialist ownership under A5, achieved generalist fractions, request styles, provenance, capacity failures, deterministic input ordering, and immutable CLI artifacts. Scoped Ruff and mypy checks passed.

The real CLI was also run against the existing AG News and SQuAD pools. The benign input contained 20 accounts, including six generalists, and 1,680 requests. A1 and A2 each produced one account with 6,000 planned queries; A5 produced six specialist accounts sharing the same 6,000-query inventory. Each inventory contained 3,000 prompts per domain, and prompt identities matched across all three strategies. This validates input generation, not HTTP execution, student learning, or harness effectiveness. Three-domain generation was exercised with test fixtures; the local GSM8K build was absent.

The broader `tests/test_start.py` check had three failures in existing startup/campaign tests. All three reproduced in an isolated package copy with the new gateway registration, schemas, and workload additions removed. They concern startup fixtures, interrupted-campaign behavior, and a missing empty-campaign event file. They remain separate existing work, and this phase does not establish that the full study passes.

## Follow-up correctness checks

Task reports use frozen resolved numeric budgets for `full` mode and explicit subsets. The benign CLI loads category references and discloses GSM8K's single-category limitation. A5 defaults to six total accounts in both the Python API and CLI. The separate-task launchers default to AG News, SQuAD, and GSM8K, retaining GSM8K as the math capability distillation corpus with a provisional cap. A bounded HTTP collector test crosses two policy resets and verifies that the original campaign attempt allowance remains unchanged. Full-pool task runs can span several policy horizons; those results are not gateway outcomes. Shared-student recipe selection and the GSM8K cap pilot remain pending.

Validation after these fixes: `uv run pytest tests/test_gateway.py tests/test_tasks.py tests/test_start.py tests/test_pipeline.py -q` completed with 63 passed and the same three pre-existing startup/campaign failures listed above. Scoped Ruff checks passed. Scoped mypy passed for the six changed implementation modules and two changed test files with `MYPYPATH=packages/harness/src`. These checks used `devenv shell`. They validate input generation, report assembly, launcher budget planning, and bounded collector accounting; no teacher collection, model training, cap pilot, or final effectiveness study ran.
