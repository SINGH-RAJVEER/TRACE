# Multi-domain harness effectiveness study

The study asks which distillation campaigns an account-level harness restricts, whether those restrictions prevent useful capability transfer within a declared resource envelope, and what legitimate users lose. One gateway serves AG News, SQuAD, GSM8K, and coding. Each campaign trains one student on all the responses it actually receives. The same strategy runs without a defense, with instantaneous MMD, with the full policy, and with a simple query quota. The attempted-request and time limits define a fair comparison; budget curves are secondary.

Status: shared gateway calibration, live multi-account campaigns, combined student training, per-domain evaluation, development recipe selection, token controls, and reporting are implemented. `just start` selects this path by default. `just distill` collects a mixed undefended campaign and trains a shared student without the protected evaluation cycle. Both retain all four domains and default to 6,000 total submissions with a globally shuffled 50,000-eligible-example coding subset across all roles. The full-source extension requires `--coding-pool-size 0`; full-pool campaigns additionally use `--budgets full` and a new run directory. The GSM8K and coding output caps remain provisional until their development pilots pass. No model effectiveness result follows from implementation tests. The manuscript now records this implementation status.

## Run the shared pipeline

```bash
# Four domains, three seeds, 6,000 total submissions, A1/A2/A5/A7, all three defenses:
just start
# Explicit primary envelope; 1,500 candidate prompts per domain:
just start --budgets 6000 --seeds 42 --run-dir artifacts/gateway-b6000
# Undefended mixed collection and shared training only:
just distill --budgets 6000 --seeds 42
# Retain the previous separate-task workflow explicitly:
just start --mode tasks
just distill --mode tasks
```

The bounded coding build lives in `data/tasks/coding-subset-50000-s42`; selection metadata, decided source IDs, qualification counts, and hashes identify its sample. Default shared run roots are `artifacts/gateway-bounded-v1` and `artifacts/gateway-distill-bounded-v1`, keeping earlier full-pool protocols immutable. The primary subset target is 50,000 qualified examples across every role, not 50,000 attacker prompts or teacher calls. Keep the complete 539-item external coding test suite. The target is a planning choice; increase it in a new build if benign role capacity or development transfer is inadequate. Do not reduce calibration thresholds or select a sample using final test results.

Numeric budgets apply to the whole campaign, including all accounts and domains. The balanced allocation must fit each pool or the run fails before collection. `full` allocates every domain's complete attacker pool and uses their sum as the global attempt limit, so domain sizes are unequal. A full inventory is a candidate pool, not a promise that all candidates fit the 86,400-second deadline. Planned arrivals default to one second apart and `--interval-seconds` controls that spacing. Refusals, failed HTTP submissions, and proof redemption all spend the same allowance. Account state resets only after 24 hours; protected accounts refuse ordinary requests beyond 10,000 policy attempts until that time. Refusals cannot renew access, and the undefended arm has no account admission cap. Policy time resets never replenish the campaign allowance. A5 and A6 use six total accounts. A7 draws from the independent pools described [below](#independent-prompt-sources); in `full` mode its per-domain allocation is the smaller of the task pool and the external pool, and its attempt limit is that inventory's own total. A3 needs a numeric budget. Each account has at most one in-flight request, with `--concurrency` limiting the campaign's total concurrency.

`start` fits one mixed benign calibration from disjoint fitting, score-calibration, and policy-calibration roles. Category references order specialists; 10% of accounts are generalists. The default workload requests 125 accounts. Non-short accounts draw a Pareto length with a 100-request minimum, shape 1.5, and a 2,000-request cap; short accounts stop at 20 requests. About 35% of non-short accounts exceed 200 requests and 9% exceed 500, so long-lived legitimate accounts complete several detector windows and the fitted quota pays a real benign cost. With fixed 100-request accounts, every benign account completed at most one window, accumulated and instantaneous evidence calibrated identically, and the quota fitted to exactly the account length. Long accounts first receive their minimum, then extend from the capacity left in their domains; the manifest records requested and achieved lengths, `capacity_limited_accounts`, `complete_100_request_windows`, and `multi_window_accounts`. Specialist accounts are assigned to domains by weighted round-robin over each role pool's size, so GSM8K's roughly 670-prompt role pools hold few specialists instead of a quarter of them; the manifest records `specialist_domain_rule`. `--benign-accounts` and `--requests-per-account` change that load, and `distill gateway benign` also accepts `--length-tail` (0 keeps fixed lengths) and `--max-requests-per-account`. Insufficient source capacity is an error. Held-out benign replay reports account escalation; live benign traffic separately measures task quality, unconditional utility, completion, latency, and solver work. Synthetic clients do not establish real-user false-positive rates. The `calibration-resolution-v2` gate requires 100 scored policy horizons and 99 non-overlapping null windows, and the default 125 accounts yield 100 non-short accounts per role when source capacity permits. Threshold degeneracy and source dependence remain separate checks; passing the count alone does not establish adequate calibration. With heavy-tailed lengths that is roughly 30,000 requests per role before capacity limits, and the benign-test role is replayed live through the teacher for every arm. Choose `--benign-accounts` deliberately for that cost.

Before protected collection, the existing development suite is divided into deterministic source-disjoint selection and qualification halves. SQuAD article titles stay together; other tasks split by item ID. Both suites and their hashes are frozen before evaluating candidates. Six shared training recipes are compared on the selection half using an additional balanced 2,000-query, seed-42 development campaign. This collection has its own recorded allowance, separate from each final campaign's budget. `--development-budget` changes that declared selection inventory. Each domain ranks the six trials by development gain, and selection takes the lowest mean rank, breaking ties by fewer epochs and then lower learning rate. A raw cross-domain mean would add accuracy, F1, and pass@1 on different scales and let the noisiest domain choose. `recipe.json` records each trial's gains, ranks, and mean rank. The winner is frozen before qualification and evaluated once on the untouched half. A domain qualifies when its paired 95% confidence interval for undefended gain has its lower bound above the declared minimum useful gain. The default margin is 2 percentage points for AG News accuracy, SQuAD F1, GSM8K accuracy, and coding pass@1. This is a protocol choice, not an empirically established benchmark threshold. Override individual tasks with repeated options such as `--minimum-gain squad=3 --minimum-gain gsm8k=2`. Domains that fail are recorded in `recipe.json` as `uninformative_domains`, stay in every report, and receive no ratio interpretation. Final defense testing stops only when no domain qualifies. A failure cannot trigger selection of another candidate on the same qualification data. Development suites of 1,000 items resolve paired gains of about 4 points at 20% discordance (see [suite resolution](#suite-resolution)), so a 2-point margin needs development gains well above it. The selected recipe is frozen across final conditions. Final test scores do not select it.

Each final campaign has one combined ledger and one combined attacker adapter, evaluated against every domain's frozen suite at that domain's cap. Gold references are optional through `--no-gold`. Per-domain released-token counts define the matched-volume and matched-prefix undefended controls; each control also trains one combined adapter and records whole-response matching residuals. Empty defended datasets evaluate the unmodified base and record why training was skipped. Managed teacher and harness processes stop before student training and local student inference. An externally managed teacher remains the caller's responsibility.

Outputs default to `artifacts/gateway` for `start` and `artifacts/gateway-distill` for `distill`. `protocol.json`, task and input manifests, calibration hashes, `recipe.json`, `policy-horizons.json`, per-campaign `campaign.events.jsonl`, `ledger.jsonl`, preparation manifests, training fingerprints, per-domain benchmark runs, utility results, and `results.json` preserve provenance. The report names retained gain and removed gain explicitly; ratio interpretation requires qualification and a final undefended-gain interval whose lower bound exceeds the same task margin. It does not invent numeric success targets or independence across overlapping seed inventories. Source identities remain in schedules and ledgers for subsequent clustered analysis. Numeric shared-study success targets still require the study's frozen analysis protocol. Capability uncertainty uses 5,000 paired source bootstrap draws with seed 42, conditional on the trained models. It does not measure training-run variability.

Completed campaign files are verified before reuse. Interrupted campaign directories are retained and cannot restart with fresh policy state. Select a new run directory for a new campaign. Training can resume using the existing training fingerprint and checkpoint checks; changing protocol options or source inputs requires a new run directory. The rank selection, per-domain gate, heavy-tailed workload, and A7 default use `shared-gateway-v3`; the earlier qualification change used `shared-gateway-v2` and `task-study-v5`, with `transfer-v2` analysis metadata. Older runs cannot be resumed under these protocols.

Each arm's report also writes `ARM/campaign-timing.json`, described under [campaign timing](#campaign-timing), and each `results.json` row carries `first_intervention`, `campaign_stopped`, and `campaign_stop_seconds`. Legacy task reports without paired evidence retain raw outcomes but do not receive valid ratios from an old positive-gain flag.

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

Domain profile and request style are separate dimensions. Specialists use one domain; generalists use at least two. Either can be repetitive, batch, bursty, or short. `--concentrated-share` (default 0.1) sets the share of accounts with specialist or repetitive request styles. The detector scores each distinct prompt in a window once, so repetition alone no longer inflates a window's score, but topic-concentrated specialists still shape the benign null; `gateway-clients-v1` used an equal five-way split, which `--concentrated-share 0.4` reproduces. Short accounts keep a fifth of the accounts and batch and bursty styles share the rest. The benign CLI supplies `task_references(...)` for every selected domain. AG News sources are ordered by news label and SQuAD by article title. GSM8K references all have category `math`, so its specialists remain domain-only rather than math-subtopic controls. The manifest records `domain_topic_categories` and `specialist_scope` so that limitation is visible. These are synthetic workloads, not measured user populations.

Calibrate and test on source-separated fitting, score-calibration, policy-calibration, and benign-test pools across all domains. Generate each role separately. Record the requested and achieved generalist account fractions, domain request fractions, account lengths, unique prompts, source groups, and scored-account counts. Repeated prompts and shared source documents do not create independent evidence.

Use generalist shares such as 0%, 10%, and 30% as declared sensitivity settings. Freeze the primary share before final testing. Report both matched-distribution testing and an explicit shift test where the held-out share differs from the calibration share. Account share and request share differ when accounts have unequal lengths.

A global mixed reference may make a legitimate specialist look more unusual than a multiplexed attacker whose domain proportions match the reference. Same-distribution extraction therefore remains an essential failure control. Neither A1 nor A2 is guaranteed to trigger the detector, and accumulated evidence cannot recover an intent signal absent from its observations. A7 is the positive control: its prompts come from separately pinned corpora, so the query distribution does shift. A detector that misses A7 is not working, while one that catches A7 but not A1 shows that same-distribution extraction is the boundary of what a request-distribution score can see. A2 is a test of ordering and window composition, not an assumed demonstration that memory detects domain switching. These are consequences of the request-distribution score specified in [the manuscript](../paper/main.tex) and [the current policy](../packages/harness/src/harness/policy.py).

Report achieved benign error rates and uncertainty at the available independent sample counts. Do not promise a validated 1% false-block rate from a small synthetic account pool. Calibrate quota and instantaneous baselines on the same benign roles and the same scored horizons; show their achieved utility costs as well as nominal operating targets.

## Campaign matrix

| ID | Campaign | Account and ordering rule | First-phase status |
|---|---|---|---|
| A1 | Multiplexed | One account; seeded interleaving of the selected domain prompts | Live execution implemented |
| A2 | Domain-sequential | One account; one seeded block per domain | Live execution implemented |
| A3 | Cover traffic | A7 extraction interleaved with in-distribution cover from the task attacker pools (`--cover-share`, default 0.5 of the budget); every received response can train the student | Implemented; secondary |
| A4 | Slow | Multiplexed extraction at a declared synthetic rate | Deferred |
| A5 | Sybil per domain | A declared total of k accounts partitioned across domains; each account specializes | Live execution implemented |
| A6 | Sybil random | Multiplexed requests split across k accounts by a seeded permutation; every account multiplexes | Implemented; secondary |
| A7 | Independent prompt sources | One account; multiplexed prompts from separately pinned public or synthetic sources through the same templates | Implemented; primary positive control |
| A8 | Active coverage | Live selection, training updates, and reselection using the attacker's observable information | Deferred |

A1, A2, A5, and A6 select identical prompt inventories when the seed, pools, and allocation match. Only order and account assignment change. A1, A2, A3, and A7 use exactly one account. A5 and A6 default to six accounts total across the campaign; A5 requires at least one account per domain. Neither multiplies the budget. A3 and A7 reject task pools, and the other strategies reject external pools. A3 marks each request's `client_type` as `cover` or `extraction`. `just start` runs A1, A2, A5, and A7 by default; add A3 and A6 with `--attacks`. A3 and A7 have no gold-label reference, because independent sources carry no gold answers.

## Independent prompt sources

The task pools split one training corpus between attackers and legitimate users, so a uniform extractor matches the benign references in expectation, and a real attacker lacks the victim's own data. `distill gateway external --task DOMAIN` writes `data/tasks/DOMAIN/external/attacker.jsonl` and a manifest from one pinned source per domain, serialized through the domain's frozen template and cap:

| Domain | Source and revision | License | Eligible prompts |
|---|---|---|---|
| AG News | `SetFit/bbc-news` at `9c31f70`, train and test | Unknown | 2,124 |
| SQuAD | `lucadiliello/newsqa` at `728e529`, train | Unknown | 73,891 |
| GSM8K | `MU-NLPC/Calc-math_qa` at `102358b`, train, answer options removed | Apache-2.0 | 20,843 |
| Coding | `bigcode/self-oss-instruct-sc2-exec-filter-50k` at `356bb06`, single-function responses | ODC-BY | 46,980 |

AG News and SQuAD counts come from builds on 2026-10-20; GSM8K and coding counts come from a dry run, because those task builds were not present locally. The builder drops exact duplicates and exact overlaps with the build's development and test suites, and counts them. It cannot detect paraphrases or pretraining overlap. Coding signatures come from the parsed response function; responses without exactly one top-level function are ineligible. No source supplies gold answers to the attacker. `just start` builds missing external pools when the selected attacks need them and freezes their manifests in `external.json`.

The gateway generators default to four domains: AG News, coding, GSM8K, and SQuAD. A balanced 6,000-query inventory allocates 1,500 per domain; 8,000 allocates 2,000 each. Explicit three-domain lists retain their earlier allocations. Seeds 42, 1, and 20 select overlapping inventories from GSM8K's 3,389-prompt pool; retain source IDs for clustered uncertainty across seeds. Coding capacity is measured after the declared subset or full-source qualification. Pool capacity is validated before artifacts are written.

`B_A` counts all HTTP submissions, including proof redemption and failures. A schedule with `B_A` ordinary-query candidates is an inventory, not a promise to submit all of them. Proof submissions can displace later candidates. `T_A` includes waiting and solving. The generated timestamps specify intended arrivals; actual submission, admission, and response-completion times belong in the campaign event ledger. A campaign crossing a time horizon retains its spent campaign budget. Crossing the account attempt cap only adds refusals; it does not start a fresh policy horizon.

The current datasets do not provide natural request timing. A4 cannot claim a measured median benign rate until a source of actual timing exists. Fixed schedules remain fixed selection even if an execution client solves challenges or waits for admission. A8 requires each defense to receive a fresh live feedback loop.

## Outcomes and controls

For domain d, define base score S0,d, paired undefended score SU,d, and defended score SD,d. Report raw scores, defended gain SD,d − S0,d, and gain removed SU,d − SD,d. Define the fraction of gain removed as `(SU,d - SD,d) / (SU,d - S0,d)`. The retained fraction is `(SD,d - S0,d) / (SU,d - S0,d)`. Both study runners export `retained_gain_fraction` and `gain_removed_fraction` separately. The legacy task field `R_D` remains a deprecated alias of retained gain and must not be interpreted as gain removed. All study scores and margins use percentage points. Require the independent qualification gate and a final paired 95% interval for undefended gain whose lower bound exceeds the frozen task margin. Otherwise return null ratios and an explicit reason, retaining raw scores, absolute gains, gain removed, and available intervals. Jointly resample matched base, undefended, and defended results, keeping SQuAD source articles together. If any bootstrap denominator is nonpositive, suppress the ratio interval and interpretation rather than discarding that draw. Do not clip ratios to zero or one. A valid ratio does not establish defense effectiveness; assess absolute gain removed with its paired interval and the declared practical-effect criteria.

Use development-only runs to establish a learnable shared-student baseline and freeze a meaningful per-domain success threshold before final testing. Retain per-domain transfer gates even though training uses one adapter. A domain with no undefended learning is uninformative for defense effectiveness; retain its scores and label that limitation. Do not erase this check when replacing task-specific orchestration. A gold-label student remains a learning reference, separate from the black-box attacker.

Lead the results with campaign strategy, defense, per-domain student score and gain removed, observed extraction success, released supervision, and benign costs. Report success per domain. If a joint campaign success rule is needed, declare whether success means reaching every domain target or any domain target and show both component outcomes. Three seeds are three observed runs, not a precise probability estimate. Keep missing, failed, and censored runs visible.

`J_ell` remains the campaign attempt index at the first account intervention of level ell, accompanied by the triggering account and domain. Record per-account reaches too. Report released responses and tokens by domain before that event, plus campaign totals through termination. Do not reinterpret "one account blocked" as "campaign stopped", or give an unreached intervention a zero delay.

## Campaign timing

`campaign_timing` reads `campaign.events.jsonl` after the decision log is joined. `distill gateway collect` writes `campaign-timing.json` beside each campaign, and the shared report writes one per arm. It records:

| Field | Meaning |
|---|---|
| `levels.Lk` | First reach of level k: attempt index (`J`), seconds from campaign start (`t`), triggering account and domain, and responses and tokens received per domain strictly before it (`Q`). Unreached levels are censored, never zero. |
| `accounts.ACCOUNT` | Attempts, received responses, first time at each level, first challenge, first block, and an observed stop. |
| `stopped`, `stop_seconds` | An account stops when it ends with at least two blocked attempts that started after its last received response completed. One refusal or failure is not a stop. |
| `campaign_stopped`, `campaign_stop_seconds` | Every account stopped; the time is the last account's stop. It is an observed stop within the campaign, not permanent prevention. |
| `received_after_first_block` | Responses delivered after the first account block, often from other accounts. |
| `delivery_windows` | Received responses per 60-second window. |
| `latency_seconds` | Distributions for received responses overall, by policy action, and by domain. |
| `end_to_end_seconds`, `challenge_solve_seconds`, `challenge_hashes`, `schedule_delay_seconds` | Per-prompt first attempt to final response, proof-of-work work, and scheduled-to-start delay when the schedule is given. |

Every distribution reports `n`, mean, max, p50, p90, p95, and p99 by linear interpolation. A quantile q carries `pq_supported=false` below 1/(1 − q) observations, where it is an interpolation toward the maximum rather than a tail estimate. Benign utility now reports the same distributions overall and per domain, profile, and style, plus `service_and_queue_seconds`; capability runs and the traffic benchmark add p90 and p99. Latency by action separates allowed requests from throttled and challenged ones. Server overhead inside one request is not separable from L2 release spacing without server-side timing.

## Suite resolution

`distill gateway power` reports, before any model runs, the smallest paired gain each frozen suite resolves at two-sided 5% and 80% power, using a normal approximation to a paired binary difference with an assumed 20% discordant share. SQuAD also reports an upper bound treating each article as one unit.

| Suite | Items | 95% half-width at 50% | Minimum detectable paired gain | Resolves 2 points |
|---|---|---|---|---|
| AG News test | 7,600 | 1.1 | 1.4 | Yes |
| SQuAD test | 10,570 | 1.0 | 1.2, item level | Yes at item level; not under the article bound |
| GSM8K test | 1,319 | 2.7 | 3.5 | No |
| Coding test, HumanEval+ only | 163 | 7.7 | 9.8 | No |
| Coding test, HumanEval+ and MBPP+ | 539 | 4.2 | 5.4 | No |
| Each development suite | 1,000 | 3.1 | 4.0 | No |

Declare larger GSM8K and coding margins, or treat their 2-point results as underpowered, before final testing. The discordant share is replaced by the observed value once paired runs exist.

Train matched-volume and matched-prefix controls from undefended responses, matching within each domain. Preserve response counts and match measured released tokens as closely as whole responses allow, reporting residuals. This avoids attributing a different domain mix to selective removal of better training examples. These post-hoc supervision controls supplement the deployable quota comparison; they do not receive false-positive scores.

Pair benign arrival schedules across arms and report false escalation, completed-by-deadline fractions, p50/p90/p95/p99 latency, and unconditional task utility for each domain profile and request style. Distinguish overhead, throttling delay, and challenge cost. Add a fixed-attempt test with a relaxed deadline or a justified extrapolation boundary to distinguish delay from a lasting loss of accessible supervision. Never describe a censored capability target as permanently unreachable.

## Replay and cached responses

Caching is valid for unchanged victim outputs under pinned prompt identity, model revision and quantization, template, decoding settings, output cap, and relevant random seed. Existing ledgers can reduce collection cost only where that full contract matches. Shared source prompts do not by themselves establish response equivalence.

The current [harness replay](../packages/harness/src/harness/replay.py) records policy decisions and scheduled release times. It does not execute challenge solving, proof submissions, teacher inference, or response completion. Filtering cached responses on `action != refuse` would incorrectly count challenged requests as released and ignore the time budget.

Use existing replay for detection diagnostics first. A later enforcement simulator must track outstanding queries, queues, solver time and hardware, proof expiry, proof attempts, deadline cutoffs, and horizon resets. Validate its decision and release traces against the live HTTP gateway under a bounded set of representative campaigns before using it to derive defended training sets. Measured solver work and serving overhead still need runtime evidence. Adaptive collection and modified-response defenses need their own compatible execution path.

## Repository work and exit criteria

| Concern | Recommendation | Work and exit criterion |
|---|---|---|
| Main claim and result order | Paper | Lead with campaign outcomes and benign costs. Move detector reproduction and budget curves behind these results. Remove predicted attack rankings. |
| Mixed benign inputs | Repo | Implemented in `workloads.mixed_clients`; expose achieved account and request mixes and preserve source provenance. |
| A1/A2/A3/A5/A6/A7 inventories | Repo | Implemented in `schedules.campaign_schedule`; paired selection, explicit account ownership, total budgets, independent and cover pools, and deterministic artifacts. A7 is in the default matrix. |
| Shared gateway and calibration | Hybrid | Implemented in `serialization.py`, `harness_api.app`, and `gateway_study.py`. Strict templates, per-domain caps, mixed calibration, and domain-bound proofs retain one account state across domains. Native HTTP checks cover enforcement. |
| Multi-account execution | Repo | Implemented in `gateway_campaign.collect_gateway`: durable global HTTP allowance, deadline, authenticated account clients, proof/failure costs, release joins, and interruption refusal. |
| Reuse of teacher outputs | Hybrid | Validate complete response identities and simulator/live equivalence before deriving defended supervision from cached outputs. |
| Shared student | Repo | Implemented combined preparation, one adapter per campaign/arm/seed, a shared six-recipe development sweep, per-domain transfer gates, and per-domain whole-response token controls. Real length exclusions and learned outcomes still need measured runs. |
| Reporting correctness | Hybrid | Both runners use source-disjoint development qualification, frozen practical gain margins, paired source uncertainty, and explicit retained/removed fractions. Legacy `R_D` remains a deprecated retained-gain alias. New protocol versions require new run directories. Shared-study success targets remain an analysis requirement. |
| Budget and target identities | Repo | Fixed for task studies: `budgets.json` freezes actual per-task numeric budgets, and reports resolve legacy `full` protocols through frozen task manifests. `policy-horizons.json` records reset settings without replenishing campaign limits. Each task's success target is frozen in `TASK/target.json` after recipe selection and before test evaluation: the base student's development score plus half the selected recipe's development gain. Cost to target reports `target_not_declared` when no target exists. |
| Combined evidence | Repo | Shared `results.json` joins domain scores, event totals, preparation counts, control token residuals and scores, plus utility paths. Utility includes domain/profile/style strata. Final effectiveness, numeric success targets, and clustered uncertainty remain unmeasured. |

Phase 1 exits when generators have deterministic output, validate pool capacity and domain identity, preserve source provenance, expose achieved profile composition, and keep A5 allocations inside one total budget. Validate the CLI on existing local pools and test its immutable output behavior. This phase produces no effectiveness claim.

The live Phase 2 execution and Phase 3 shared-training paths are implemented and covered by bounded integration tests. Cached execution is still deferred. Phase 4 requires real teacher collection, GPU training, frozen numeric targets, clustered analysis, and the final study matrix. The manuscript describes the implemented live path while keeping model outcomes unmeasured.

The proposed 48 primary adapters assume A1/A2/A5/A7 × four arms × three seeds, the launcher default. A1/A2/A5 alone plan 36 adapter conditions; identical undefended datasets may permit reuse only when the complete training contract matches. Neither count includes development recipe selection, matched controls, specialist references, sensitivity retraining, or A8. Declare those costs before freezing the matrix.

## Run the implemented input builders

Enter `devenv shell` first. Existing task builds are inputs and are not modified. Build any missing domain with `uv run distill tasks build --task DOMAIN`. The generators require existing builds for their selected domains. Both default launchers include GSM8K and coding and build missing tasks automatically. Their implemented cap pilots still need actual teacher measurements.

```sh
uv run distill gateway benign --domains ag_news,squad --role fitting --accounts 20 --generalist-share 0.3 --requests-per-account 100 --output artifacts/gateway/fitting-g30
uv run distill gateway schedule --domains ag_news,squad --attack A1 --budget 6000 --seed 42 --output artifacts/gateway/A1-s42
uv run distill gateway schedule --domains ag_news,squad --attack A2 --budget 6000 --seed 42 --output artifacts/gateway/A2-s42
uv run distill gateway schedule --domains ag_news,squad --attack A5 --budget 6000 --accounts 6 --seed 42 --output artifacts/gateway/A5-s42
uv run distill gateway external --task ag_news
uv run distill gateway external --task squad
uv run distill gateway schedule --domains ag_news,squad --attack A7 --budget 3000 --seed 42 --output artifacts/gateway/A7-s42
uv run distill gateway schedule --domains ag_news,squad --attack A3 --budget 3000 --cover-share 0.5 --seed 42 --output artifacts/gateway/A3-s42
uv run distill gateway power --domains ag_news,squad,gsm8k,coding
```

Each command creates `requests.jsonl` and `manifest.json` in a new directory and refuses to overwrite existing inputs. Each request retains its full `PromptRecord`, domain in `prompt.task`, source corpus and group, account, planned time, and frozen output cap. The manifest hashes the prompt pools and output file and records `execution_status="not_run"`. Generate all four benign roles separately from their corresponding task pools. The Python schedule API also accepts an explicit domain allocation that must sum to the total budget. `distill gateway collect` accepts these schedule directories and verifies the request-file hash before execution. The legacy single-task collector does not accept them.

## Current implementation validation

The shared lifecycle test uses actual localhost teacher and harness HTTP servers, real calibration/policy logic with a small deterministic encoder, combined preparation, adapter provenance checks, and real benchmark scoring. It substitutes model generation and optimizer work. It covers A1, six-account A5, and A7 from fixture external pools without a gold reference, gold references for the task-pool strategies, development selection, controls, per-domain evaluation of the same adapter, report joins, and stopping managed inference before student work. Additional HTTP tests cover cross-domain quotas, global multi-account budgets, proof spending, domain-bound continuation, and immutable campaign reuse. Full model collection, cap pilots, and GPU training remain unmeasured.

Validation on 2026-10-20: `uv run pytest -q` completed with 222 passed and four opt-in training-stack tests skipped. Ruff and mypy pass for apps, packages, and tests. Earlier validation also passed Ruff and mypy for apps, packages, tests, and the bootstrap scripts. These checks establish local integration behavior, not measured model effectiveness.

## Historical phase 1 validation

`uv run pytest tests/test_gateway.py tests/test_tasks.py -q` passed all 14 tests. The gateway tests cover three domains, paired inventories, specialist ownership under A5, achieved generalist fractions, request styles, provenance, capacity failures, deterministic input ordering, and immutable CLI artifacts. Scoped Ruff and mypy checks passed.

The real CLI was also run against the existing AG News and SQuAD pools. The benign input contained 20 accounts, including six generalists, and 1,680 requests. A1 and A2 each produced one account with 6,000 planned queries; A5 produced six specialist accounts sharing the same 6,000-query inventory. Each inventory contained 3,000 prompts per domain, and prompt identities matched across all three strategies. This validates input generation, not HTTP execution, student learning, or harness effectiveness. Three-domain generation was exercised with test fixtures; the local GSM8K build was absent.

The broader `tests/test_start.py` check initially had three failures involving startup fixtures, interrupted-campaign behavior, and a missing empty-campaign event file. Those failures no longer reproduce in the current suite; see the validation above.

## Historical follow-up correctness checks

Task reports use frozen resolved numeric budgets for `full` mode and explicit subsets. The benign CLI loads category references and discloses GSM8K's single-category limitation. A5 defaults to six total accounts in both the Python API and CLI. The separate-task launchers default to AG News, SQuAD, GSM8K, and coding. Math and coding output caps remain provisional. A bounded HTTP collector test verifies that refusal spam cannot reset account state and that actual time resets leave the campaign allowance unchanged. Full-pool task runs use the same time-only reset rule and account cap, while their reports remain separate from shared-gateway outcomes. Shared-student recipe selection and the GSM8K cap pilot remain pending.

The historical follow-up run of `uv run pytest tests/test_gateway.py tests/test_tasks.py tests/test_start.py tests/test_pipeline.py -q` had 63 passes and the three startup/campaign failures described above. The current full suite passes. Scoped Ruff and mypy checks also passed during that follow-up, with `MYPYPATH=packages/harness/src` for mypy. These checks used `devenv shell`. They validate input generation, report assembly, launcher budget planning, and bounded collector accounting; no teacher collection, model training, cap pilot, or final effectiveness study ran.

AG News and SQuAD inputs now use the [split-v2 source allocations](paper-tasks.md#splits). New default task directories preserve older builds. Rebuild inputs and start a fresh run directory before using these allocations; development and benign-test groups stay fixed. The separate-task GSM8K resolution exception does not relax the mixed gateway calibration gate.
