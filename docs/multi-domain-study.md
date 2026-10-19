# Multi-domain harness effectiveness study

The study asks which distillation campaigns an account-level harness restricts, whether those restrictions prevent useful capability transfer within a declared resource envelope, and what legitimate users lose. One gateway serves AG News, SQuAD, GSM8K, and coding. Each campaign trains one student on all the responses it actually receives. The same strategy runs without a defense, with instantaneous MMD, with the full policy, and with a simple query quota. The attempted-request and time limits define a fair comparison; budget curves are secondary.

Status: shared gateway calibration, live multi-account campaigns, combined student training, per-domain evaluation, development recipe selection, token controls, and reporting are implemented. `just start` selects this path by default. `just distill` collects a mixed undefended campaign and trains a shared student without the protected evaluation cycle. Both retain all four domains and full attacker inventories by default. The GSM8K and coding output caps remain provisional until their development pilots pass. No model effectiveness result follows from implementation tests. The manuscript now records this implementation status.

## Run the shared pipeline

```bash
# Four domains, three seeds, full inventories, A1/A2/A5, all three defenses:
just start
# A smaller total envelope; 1,500 candidate prompts per domain:
just start --budgets 6000 --seeds 42 --run-dir artifacts/gateway-b6000
# Undefended mixed collection and shared training only:
just distill --budgets 6000 --seeds 42
# Retain the previous separate-task workflow explicitly:
just start --mode tasks
just distill --mode tasks
```

Numeric budgets apply to the whole campaign, including all accounts and domains. The balanced allocation must fit each pool or the run fails before collection. `full` allocates every domain's complete attacker pool and uses their sum as the global attempt limit, so domain sizes are unequal. A full inventory is a candidate pool, not a promise that all candidates fit the 86,400-second deadline. Planned arrivals default to one second apart and `--interval-seconds` controls that spacing. Refusals, failed HTTP submissions, and proof redemption all spend the same allowance. Account state resets only after its 24-hour time horizon. The protected policies refuse ordinary requests beyond 10,000 policy attempts until that time; refusals cannot renew access. The undefended arm has no account admission cap. Policy time resets never replenish the campaign allowance. A5 uses six total accounts. Each account has at most one in-flight request, with `--concurrency` limiting the campaign's total concurrency.

`start` fits one mixed benign calibration from disjoint fitting, score-calibration, and policy-calibration roles. Category references order specialists; 10% of accounts are generalists. The default workload requests 20 accounts and 100 requests per account, with short and repetitive styles disclosed in each manifest. `--benign-accounts` and `--requests-per-account` change that load. Insufficient source capacity is an error. Held-out benign replay reports account escalation; live benign traffic separately measures task quality, unconditional utility, completion, latency, and solver work. Synthetic clients do not establish real-user false-positive rates.

Before protected collection, six shared training recipes are compared on an additional balanced 2,000-query, seed-42 development campaign. This collection has its own recorded allowance and is separate from each final campaign's budget. `--development-budget` changes that declared selection inventory. Selection maximizes the equal-weight mean of domain development scores, breaking ties by fewer epochs and then lower learning rate. Every domain must show positive development transfer or final defense testing stops. The selected recipe is frozen across final conditions. Test scores do not select it.

Each final campaign has one combined ledger and one combined attacker adapter, evaluated against every domain's frozen suite at that domain's cap. Gold references are optional through `--no-gold`. Per-domain released-token counts define the matched-volume and matched-prefix undefended controls; each control also trains one combined adapter and records whole-response matching residuals. Empty defended datasets evaluate the unmodified base and record why training was skipped. Managed teacher and harness processes stop before student training and local student inference. An externally managed teacher remains the caller's responsibility.

Outputs default to `artifacts/gateway` for `start` and `artifacts/gateway-distill` for `distill`. `protocol.json`, task and input manifests, calibration hashes, `recipe.json`, `policy-horizons.json`, per-campaign `campaign.events.jsonl`, `ledger.jsonl`, preparation manifests, training fingerprints, per-domain benchmark runs, utility results, and `results.json` preserve provenance. The report names retained gain and removed gain explicitly; it leaves ratios undefined when undefended test gain is nonpositive. It does not invent numeric success targets or independence across overlapping seed inventories. Source identities remain in schedules and ledgers for subsequent clustered analysis. Final target thresholds and uncertainty analysis still require the study's frozen analysis protocol.

Completed campaign files are verified before reuse. Interrupted campaign directories are retained and cannot restart with fresh policy state. Select a new run directory for a new campaign. Training can resume using the existing training fingerprint and checkpoint checks; changing protocol options or source inputs requires a new run directory.

To execute an already generated schedule against a configured shared harness:

```bash
uv run distill gateway collect --input-dir artifacts/gateway/A5-s42 --output artifacts/gateway/A5-s42-collected --teacher-url http://127.0.0.1:8003 --keys-file PATH_TO_ACCOUNT_KEYS --decision-log PATH_TO_DECISIONS
```

The harness requires `HARNESS_TASK=gateway`, `HARNESS_GATEWAY_CAPS` as a JSON domain-to-cap mapping, a matching mixed calibration, and authenticated account keys. `just start` configures these automatically. Only refusal mode is supported for shared execution. Cached enforcement simulation and adaptive attacks remain separate work.

## What the paper can claim

The [manuscript](../paper/main.tex) includes coding as the fourth shared-gateway domain and describes the implemented collection, calibration, shared-training, controls, and reporting paths. It distinguishes the launchers' full-pool defaults from the proposed balanced 6,000-submission matrix, which allocates 1,500 candidate prompts per domain. The corpus and cap measurements, GPU training, final target thresholds, clustered analysis, and effectiveness results remain pending. Local HTTP and artifact tests are identified as implementation checks.

The local compiled copy is `paper/main.pdf`, which is gitignored. Regenerate it from the repository root with `tectonic --outdir paper paper/main.tex`; auxiliary files stay temporary under Tectonic's default output settings.

The manuscript asks:

1. Which campaign strategies lose useful learned capability behind the harness relative to their paired undefended runs?
2. How much usable supervision escapes before intervention, and does the harness prevent transfer within the declared envelope or primarily delay collection?
3. What are the false-escalation, completion, latency, and unconditional task-utility costs for specialist and generalist users?
4. What does accumulated MMD add beyond instantaneous MMD and a simple quota at comparable benign cost?

The revised manuscript proposes a primary envelope of 6,000 total HTTP submissions and 24 hours, a balanced candidate inventory of 1,500 prompts per domain across four domains, six accounts for A5, and a 10% generalist account share. These are proposed settings to freeze before final testing. It also specifies one shared development-selected training recipe, per-domain transfer gates, and numeric domain success targets selected on development evidence. The target values and final matrix remain unresolved. Older full-pool per-task budgets describe the existing task launcher, not the proposed gateway study.

Detection, response restriction, and reduced learned capability are separate observations. An alert or a blocked account does not establish that a campaign failed. The undefended reference must demonstrate positive transfer first. Call it a successful undefended extraction baseline, rather than "perfect distillation". Task accuracy and agreement with the victim measure different extraction objectives, so keep both with their explicit denominators. This distinction follows [Jagielski et al., USENIX Security 2020](https://www.usenix.org/conference/usenixsecurity20/presentation/jagielski).

Multiplexing, domain blocks, and account splitting are collection strategies under the same response-supervised learning procedure. Describe them as different distillation campaigns. Claims about different learning algorithms require additional attacker training objectives. Adaptive collection also differs from adapting training on already collected responses. A recent preprint reports that adaptive student training recovers capability missed by passive evaluation in its response-side defense setting. That motivates a stronger attacker extension here, but does not predict this harness's results. See [The Distillation Game, revision 4](https://arxiv.org/abs/2605.22737v4).

## Endpoint, data, and identity

Use one unchanged teacher and one gateway endpoint. Preserve each request's domain-specific prompt, output cap, non-thinking setting, and pinned teacher identity. Use the existing task pools and frozen test suites. Coding is included in repository defaults through the [coding task builder](coding-task.md), with executable pinned assertions and a provisional 512-token cap. The manuscript remains unchanged.

Use one LoRA adapter on `Qwen3.5-2B-Base` per campaign, defense, and training seed. Start each adapter from the same base checkpoint. Combine received supervision across domains and shuffle training examples with a declared seed so collection order is not silently reused as a training curriculum. Preserve prompt/source identity and record domain counts and supervised tokens before and after preparation. A common 2,048-token limit needs measured per-domain exclusions; concatenating datasets does not establish that every example fits.

For this controlled gateway, domain metadata comes from the declared task interface and validated request serialization. Do not treat a client-supplied domain name or a substring inside arbitrary text as trustworthy. Gateway mode requires one user message matching the declared task template and its frozen cap. Unknown domains, mismatched templates, extra messages, and mismatched caps receive HTTP 422 before policy admission. The `X-Gateway-Domain` header is bound into proof continuation and calibration uses the same domain-prefixed detector content. Fixed service instructions are excluded from detector text, while task content remains. Task content itself can identify domains, so stripping instructions does not eliminate distribution differences.

Account identities are stable experiment identities, not proof of a production identity system. Account state spans domains within a policy horizon. A Sybil campaign shares one resource envelope and pools every received response across its accounts.

## Benign traffic and calibration

Domain profile and request style are separate dimensions. Specialists use one domain; generalists use at least two. Either can be repetitive, batch, bursty, or short. `--concentrated-share` (default 0.1) sets the share of accounts with specialist or repetitive request styles. Repeated prompts score far above broad windows, so this share shapes the benign null; `gateway-clients-v1` used an equal five-way split, which `--concentrated-share 0.4` reproduces. Short accounts keep a fifth of the accounts and batch and bursty styles share the rest. The benign CLI supplies `task_references(...)` for every selected domain. AG News sources are ordered by news label and SQuAD by article title. GSM8K references all have category `math`, so its specialists remain domain-only rather than math-subtopic controls. The manifest records `domain_topic_categories` and `specialist_scope` so that limitation is visible. These are synthetic workloads, not measured user populations.

Calibrate and test on source-separated fitting, score-calibration, policy-calibration, and benign-test pools across all domains. Generate each role separately. Record the requested and achieved generalist account fractions, domain request fractions, account lengths, unique prompts, source groups, and scored-account counts. Repeated prompts and shared source documents do not create independent evidence.

Use generalist shares such as 0%, 10%, and 30% as declared sensitivity settings. Freeze the primary share before final testing. Report both matched-distribution testing and an explicit shift test where the held-out share differs from the calibration share. Account share and request share differ when accounts have unequal lengths.

A global mixed reference may make a legitimate specialist look more unusual than a multiplexed attacker whose domain proportions match the reference. Same-distribution extraction therefore remains an essential failure control. Neither A1 nor A2 is guaranteed to trigger the detector, and accumulated evidence cannot recover an intent signal absent from its observations. A2 is a test of ordering and window composition, not an assumed demonstration that memory detects domain switching. These are consequences of the request-distribution score specified in [the manuscript](../paper/main.tex) and [the current policy](../packages/harness/src/harness/policy.py).

Report achieved benign error rates and uncertainty at the available independent sample counts. Do not promise a validated 1% false-block rate from a small synthetic account pool. Calibrate quota and instantaneous baselines on the same benign roles; show their achieved utility costs as well as nominal operating targets.

## Campaign matrix

| ID | Campaign | Account and ordering rule | First-phase status |
|---|---|---|---|
| A1 | Multiplexed | One account; seeded interleaving of the selected domain prompts | Live execution implemented |
| A2 | Domain-sequential | One account; one seeded block per domain | Live execution implemented |
| A3 | Cover traffic | Multiplexed extraction plus declared cover traffic; every received response can train the student | Deferred |
| A4 | Slow | Multiplexed extraction at a declared synthetic rate | Deferred |
| A5 | Sybil per domain | A declared total of k accounts partitioned across domains; each account specializes | Live execution implemented |
| A6 | Sybil random | Multiplexed requests split across k accounts | Deferred |
| A7 | Independent prompt sources | Domain-matched prompts from separately pinned public or synthetic sources | Deferred |
| A8 | Active coverage | Live selection, training updates, and reselection using the attacker's observable information | Deferred |

A1, A2, and A5 select identical prompt inventories when the seed, pools, and allocation match. Only order and account assignment change. A1 and A2 use exactly one account. A5 defaults to six accounts total across the campaign, requires at least one account per domain, and never multiplies the budget.

The gateway generators default to four domains: AG News, coding, GSM8K, and SQuAD. A balanced 6,000-query inventory allocates 1,500 per domain; 8,000 allocates 2,000 each. Explicit three-domain lists retain their earlier allocations. Seeds 42, 1, and 20 select overlapping inventories from GSM8K's 3,389-prompt pool; retain source IDs for clustered uncertainty across seeds. Coding capacity is measured after full qualification. Pool capacity is validated before artifacts are written.

`B_A` counts all HTTP submissions, including proof redemption and failures. A schedule with `B_A` ordinary-query candidates is an inventory, not a promise to submit all of them. Proof submissions can displace later candidates. `T_A` includes waiting and solving. The generated timestamps specify intended arrivals; actual submission, admission, and response-completion times belong in the campaign event ledger. A campaign crossing a time horizon retains its spent campaign budget. Crossing the account attempt cap only adds refusals; it does not start a fresh policy horizon.

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
| Shared gateway and calibration | Hybrid | Implemented in `serialization.py`, `harness_api.app`, and `gateway_study.py`. Strict templates, per-domain caps, mixed calibration, and domain-bound proofs retain one account state across domains. Native HTTP checks cover enforcement. |
| Multi-account execution | Repo | Implemented in `gateway_campaign.collect_gateway`: durable global HTTP allowance, deadline, authenticated account clients, proof/failure costs, release joins, and interruption refusal. |
| Reuse of teacher outputs | Hybrid | Validate complete response identities and simulator/live equivalence before deriving defended supervision from cached outputs. |
| Shared student | Repo | Implemented combined preparation, one adapter per campaign/arm/seed, a shared six-recipe development sweep, per-domain transfer gates, and per-domain whole-response token controls. Real length exclusions and learned outcomes still need measured runs. |
| Reporting correctness | Hybrid | Shared reports explicitly name `retained_gain_fraction` and `gain_removed_fraction`, with undefined ratios for nonpositive undefended test gain. Legacy task reporting retains its previous naming. Development-frozen success targets remain an analysis requirement. |
| Budget and target identities | Repo | Fixed for task studies: `budgets.json` freezes actual per-task numeric budgets, and reports resolve legacy `full` protocols through frozen task manifests. `policy-horizons.json` records reset settings without replenishing campaign limits. Each task's success target is frozen in `TASK/target.json` after recipe selection and before test evaluation: the base student's development score plus half the selected recipe's development gain. Cost to target reports `target_not_declared` when no target exists. |
| Combined evidence | Repo | Shared `results.json` joins domain scores, event totals, preparation counts, control token residuals and scores, plus utility paths. Utility includes domain/profile/style strata. Final effectiveness, numeric success targets, and clustered uncertainty remain unmeasured. |

Phase 1 exits when generators have deterministic output, validate pool capacity and domain identity, preserve source provenance, expose achieved profile composition, and keep A5 allocations inside one total budget. Validate the CLI on existing local pools and test its immutable output behavior. This phase produces no effectiveness claim.

The live Phase 2 execution and Phase 3 shared-training paths are implemented and covered by bounded integration tests. Cached execution is still deferred. Phase 4 requires real teacher collection, GPU training, frozen numeric targets, clustered analysis, and the final study matrix. The manuscript describes the implemented live path while keeping model outcomes unmeasured.

The proposed 48 primary adapters assume A1/A2/A5/A7 × four arms × three seeds. A7 is currently deferred. A1/A2/A5 alone plan 36 adapter conditions; identical undefended datasets may permit reuse only when the complete training contract matches. Neither count includes development recipe selection, matched controls, specialist references, sensitivity retraining, or A8. Declare those costs before freezing the matrix.

## Run the implemented input builders

Enter `devenv shell` first. Existing task builds are inputs and are not modified. Build any missing domain with `uv run distill tasks build --task DOMAIN`. The generators require existing builds for their selected domains. Both default launchers include GSM8K and coding and build missing tasks automatically. Their implemented cap pilots still need actual teacher measurements.

```sh
uv run distill gateway benign --domains ag_news,squad --role fitting --accounts 20 --generalist-share 0.3 --requests-per-account 100 --output artifacts/gateway/fitting-g30
uv run distill gateway schedule --domains ag_news,squad --attack A1 --budget 6000 --seed 42 --output artifacts/gateway/A1-s42
uv run distill gateway schedule --domains ag_news,squad --attack A2 --budget 6000 --seed 42 --output artifacts/gateway/A2-s42
uv run distill gateway schedule --domains ag_news,squad --attack A5 --budget 6000 --accounts 6 --seed 42 --output artifacts/gateway/A5-s42
```

Each command creates `requests.jsonl` and `manifest.json` in a new directory and refuses to overwrite existing inputs. Each request retains its full `PromptRecord`, domain in `prompt.task`, source corpus and group, account, planned time, and frozen output cap. The manifest hashes the prompt pools and output file and records `execution_status="not_run"`. Generate all four benign roles separately from their corresponding task pools. The Python schedule API also accepts an explicit domain allocation that must sum to the total budget. `distill gateway collect` accepts these schedule directories and verifies the request-file hash before execution. The legacy single-task collector does not accept them.

## Current implementation validation

The shared lifecycle test uses actual localhost teacher and harness HTTP servers, real calibration/policy logic with a small deterministic encoder, combined preparation, adapter provenance checks, and real benchmark scoring. It substitutes model generation and optimizer work. It covers A1 and six-account A5, gold references, development selection, controls, per-domain evaluation of the same adapter, report joins, and stopping managed inference before student work. Additional HTTP tests cover cross-domain quotas, global multi-account budgets, proof spending, domain-bound continuation, and immutable campaign reuse. Full model collection, cap pilots, and GPU training remain unmeasured.

Validation on this checkout: `devenv shell -- uv run pytest -q` completed with 166 passed and four opt-in training-stack tests skipped. Ruff passed, and mypy passed for apps, packages, tests, and the bootstrap scripts. The paper source was unchanged. These checks establish local integration behavior, not measured model effectiveness.

## Historical phase 1 validation

`uv run pytest tests/test_gateway.py tests/test_tasks.py -q` passed all 14 tests. The gateway tests cover three domains, paired inventories, specialist ownership under A5, achieved generalist fractions, request styles, provenance, capacity failures, deterministic input ordering, and immutable CLI artifacts. Scoped Ruff and mypy checks passed.

The real CLI was also run against the existing AG News and SQuAD pools. The benign input contained 20 accounts, including six generalists, and 1,680 requests. A1 and A2 each produced one account with 6,000 planned queries; A5 produced six specialist accounts sharing the same 6,000-query inventory. Each inventory contained 3,000 prompts per domain, and prompt identities matched across all three strategies. This validates input generation, not HTTP execution, student learning, or harness effectiveness. Three-domain generation was exercised with test fixtures; the local GSM8K build was absent.

The broader `tests/test_start.py` check had three failures in existing startup/campaign tests. All three reproduced in an isolated package copy with the new gateway registration, schemas, and workload additions removed. They concern startup fixtures, interrupted-campaign behavior, and a missing empty-campaign event file. They remain separate existing work, and this phase does not establish that the full study passes.

## Historical follow-up correctness checks

Task reports use frozen resolved numeric budgets for `full` mode and explicit subsets. The benign CLI loads category references and discloses GSM8K's single-category limitation. A5 defaults to six total accounts in both the Python API and CLI. The separate-task launchers default to AG News, SQuAD, GSM8K, and coding. Math and coding output caps remain provisional. A bounded HTTP collector test verifies that refusal spam cannot reset account state and that actual time resets leave the campaign allowance unchanged. Full-pool task runs use the same time-only reset rule and account cap, while their reports remain separate from shared-gateway outcomes. Shared-student recipe selection and the GSM8K cap pilot remain pending.

Validation after these fixes: `uv run pytest tests/test_gateway.py tests/test_tasks.py tests/test_start.py tests/test_pipeline.py -q` completed with 63 passed and the same three pre-existing startup/campaign failures listed above. Scoped Ruff checks passed. Scoped mypy passed for the six changed implementation modules and two changed test files with `MYPYPATH=packages/harness/src`. These checks used `devenv shell`. They validate input generation, report assembly, launcher budget planning, and bounded collector accounting; no teacher collection, model training, cap pilot, or final effectiveness study ran.
