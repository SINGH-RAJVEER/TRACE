# Benchmarking and result tracking: open fixes

These come from a review of the evaluation methodology on 2026-11-02. They are listed by severity. Each item names the evidence and what counts as done.

## 1. Run the declared campaign matrix

`readings/v3/protocol.json` declares seeds 42, 1 and 20, attacks A1, A2, A5 and A7, and arms full, instantaneous and quota. Only A1 at seed 42 completed. A2 was interrupted. A5, A6 and A7 have schedules but no protected runs.

- [ ] Write a new protocol file for the next study, with the seeds, attacks (including A8 and the adaptive variants) and arms it will actually run, before any of them run.
- [ ] Run each attack and arm at every declared seed on the exploratory partition, using fresh state files and output directories.
- [ ] Record every run, including interrupted and refused ones, in the results index (item 5). `just reading` now does this for every arm of an exported run, so this is done once the runs exist.
- [ ] Use separate real credentials for the multi-account A5 and A6 campaigns and for account rotation. A shared token only tests local bookkeeping.

Done when every declared cell has a completed run or a recorded reason why it has none.

## 2. Measure student capability beyond AG News

Transfer has been measured only for AG News, which is under 1% of released A1 tokens. GSM8K (about 63%) and coding (about 35%) have no student evaluation. The learning-curve tooling has only been validated on CPU fixtures.

- [ ] Qualify the learning-curve runner on the GPU host and record the training-library versions.
- [ ] Run the AG News curve (`distill learning-curve`) and declare a development capability target from it.
- [ ] Extend the learning curve to GSM8K and coding, using objective metrics (numeric answers, sandboxed tests) and held-out source families.
- [ ] Use the curves to decide whether 1,500 prompts per domain is the right budget, and whether the 1,215-request quota still releases enough data to train a useful student.

Done when each domain has a curve with base, gold and undefended-teacher controls.

## 3. Stop reusing the A1 development data

The A1 prompts have been examined by the v4 audit, the shadow checks, the evidence replay and the separation study. The separation verdict changes with the split: at W=50 it survives under 2 of 5 split seeds.

- [ ] Write and hash a preregistration file before anyone queries the confirmatory partition. It should cover the primary comparisons, arms, seeds, decision rules, acceptance targets and multiple-testing correction.
- [ ] Generate fresh exploratory schedules from a new seed for further development work, instead of reusing the v3 A1 schedule.
- [ ] For the separation study, run the next step it names (finer topic cells, larger attack pools, more split seeds) on fresh exploratory schedules, and report the verdict as a distribution over split seeds.
- [ ] Log each development look at a dataset in the results index, so how many times each dataset has been inspected stays visible.

## 4. Strengthen the benign baseline

The multi-domain null has 13 account horizons, so the session, novelty and volume ranks all abstain. The coding null has 12 independent windows and GSM8K has 2, so the per-domain shadow scorer abstains for both. The undefended utility baseline completed only 29% of requests, so the teacher was overloaded and utility comparisons against it are invalid.

- [ ] Rebuild larger source-disjoint role pools. The GSM8K score-calibration pool has only 682 prompts.
- [ ] Obtain and audit source-family provenance (publisher, problem family, problem lineage, document title) for the `gateway-clients-v5` workload, then run `scripts/recalibrate-profiles.sh`.
- [ ] Reach at least 19 independent horizons per profile for upper-tail ranks and 39 for two-sided ranks, including coding and GSM8K specialists.
- [ ] Pass `teacher-preflight` at the declared offered rate before collecting any new utility baseline.
- [ ] Add benign roles that use the A7 external corpora, so that A7 can be balanced by corpus.

## 5. Make results tracking queryable and reproducible

Results are kept as prose tables in `docs/evaluation.md` and `docs/harness.md`. Their supporting reports are in the gitignored `artifacts/` directory, which is missing from this checkout.

- [x] Add a results index with one row per arm. Done: `readings/index.tsv`, written by `just reading` through `packages/distillation/src/distillation/results_index.py`. It is TSV because `readings/**/*.json` and `*.jsonl` are stored in Git LFS. Rows are keyed by run, kind, attack and arm rather than contract hash, so gate-refused and interrupted arms, which may have no contract, still get a row; duplicate contract hashes are refused.
- [ ] Backfill the historical v3 A1 arms into the index, labelled as backfill, with unrecorded fields left empty.
- [ ] Freeze the 2026-11-02 separation study and the shadow-check outputs into a versioned reading, with input hashes and the run script, so the numbers in the docs can be reproduced from the repo.
- [ ] Generate the result tables in the docs from the index, and keep `docs/` for method rather than data.
- [x] Record per-file SHA-256 manifests for each new reading. Done: `just reading` writes `SHA256SUMS`.

## 6. Correct stale budget references

The code default is 6,000 HTTP attempts, split evenly into 1,500 prompts per domain (`gateway_cli.py:163`).

- [x] `docs/evaluation.md:13` called session replay and selective retries planned experiments; they are implemented collection options.
- [x] `docs/evaluation.md:11` says the default budget is 12,000 HTTP attempts. Fixed to 6,000.
- [x] `docs/evaluation.md:15` uses `--budget 12000` in the A8 example. Changed to 6000.
- [x] `docs/evaluation.md:48` says the default needs 3,000 distinct eligible prompts per domain. Fixed to 1,500.
- [ ] `evaluation_protocol.py:8` sets `CHECKPOINTS` up to 12,000. This is above the campaign budget. Decide whether the 6,000 and 12,000 checkpoints should stay, and update `docs/evaluation.md:70` and its tests to match. Open: the checkpoints also apply to the supervised gold controls, which the campaign budget does not limit, so this needs your decision.

---

# Proposed improvements from `research/`

Every proposal from the research report "Build defenses around acquired model capability" (cutoff 2026-10-31) and its three notes: `frontier-practice.md`, `detection-research.md` and `output-architecture.md`. The `research/` directory was removed after this list was completed; it is present in git history at commit `5da3d98d`, including the full annotated source register. Each item cites the report section or note it came from. The research describes designs, not measured improvements. Source IDs used below are listed with links in "Sources cited" at the end.

Items already present in the repository are checked, with a pointer to where. Items that are partly present stay unchecked, and their note says what is missing. The research was written against commit `43464d2e` and refers to paths such as `legacy/packages/distillation/src/distillation/gateway_study.py` that no longer exist in this checkout.

## 7. Claims, framing and citations

Report: "The decision is to evolve TRACE", "Use citation categories honestly", "What remains unknown".

- [ ] Adopt the four priorities in order: durable acquisition accounting and identity-bound entitlements, then calibration and evaluation rebuilt around realistic specialists and source-disjoint roles, then selective content and campaign evidence only where it beats the accounting baselines, then output-side methods in a separate inference-access track with provenance beside it.
- [ ] Frame the project goal as raising the money, time and verified identities needed to obtain a useful student while preserving legitimate task completion. Never claim universal prevention while the threat model includes indistinguishable useful-answer collection.
- [ ] State in the manuscript that no design has a measured gain until a new study produces one.
- [ ] Present the interrupted A2 full log as 1,537 admissions and 1,536 responses, never as a completed 6,000-request campaign.
- [ ] Record that the calibration mixture warning is warning-only and changes no enforcement.
- [ ] Cite peer-reviewed sources only within their evaluated setting. Cite preprints as preprints and pin the inspected version. Distinguish arXiv venue metadata from checked proceedings.
- [ ] Keep the MMD paper's June v1 (used in the online study) and v3 (in the inherited source check) as separate provenance records. Audit the version the manuscript cites.
- [ ] Phrase provider evidence as "Anthropic reports" and keep its outcome unit, population and date. Do not convert harvested-reasoning reduction into capability reduction, or a provider's self-reported false-positive rate into a guarantee for TRACE.
- [ ] Use practitioner essays only for incentives and motivation, and anonymous posts only as usability leads, labelled as unverified.
- [ ] For TRACE results, cite the frozen revision, calibration hash, source-role manifest, workload contract and exact artifact. Keep completed live outcomes, interrupted logs, deterministic offline replay and proposed arms separate.
- [ ] Before any novelty claim, run a second literature search around the chosen narrow objective.
- [ ] Audit the full methods of D25 (Chandrasekaran et al.), D26 (Knockoff Nets) and D27 (ActiveThief). They were checked at abstract level only.
- [ ] Get independent reproduction, or full-text checks, of the 2026 preprints and abstract-only sources: D01, D13, D20 and D21 proofs, D24, O05 (ADS-C).
- [ ] Keep an explicit list of unknowns in the manuscript: live lab state and revision, real legitimate traffic and profile prevalence, available identity signals, acceptable explanation loss, account acquisition costs, output-token teaching value by task, false-link rates, student learning curves, and serving overhead.

## 8. Threat model and attacker declaration

Report: "The threat model determines what a defense can control"; `output-architecture.md` "Frozen study design" step 1.

- [ ] Add the actor and access table (one opaque-API account, many cheap accounts, genuine-session replay or purchased logs, reasoning-extraction operator, provider with sampler or logits, provider with weights, suspect-model investigator) to `docs/`, with observations, feasible controls and residual problems for each.
- [ ] Treat answer-only transfer, rationale extraction, numerical or structural extraction, and provenance as separate problems with separate success criteria. Keep closed-teacher capability emulation and literal weight extraction as distinct endpoints.
- [ ] State the total-variation identifiability limit in the docs and the paper: if attacker and benign observables have the same distribution, any detector's true-positive rate equals its false-positive rate.
- [ ] Declare three independent attacker budgets for every study: online queries, output tokens and money; available distinct prompts and outside data; offline compute.
- [ ] Declare the attacker interface: logits, answer-only modes, prefill, tools, attachments, multi-turn history, reasoning summaries and streaming. Declare account cost and available verified identities separately.
- [ ] Report the identity budget explicitly: k unlinkable identities obtain about k times B units, plus concurrency overshoot.
- [ ] Report residual risk from cheaper identities, outside data and longer deadlines alongside every result.
- [ ] Do not expose probabilities or other numerical channels the product does not need. Test any numerical interface separately, since hard-label and text extraction remain possible without it.

## 9. Acquisition accounting and entitlements

Report: "Reserve before inference and settle every released prefix", "Define component contracts"; `output-architecture.md` "Evolutionary TRACE design"; `frontier-practice.md` inference 1.

- [x] One transactional store for admission, reservation and settlement across restarts. Done: `HARNESS_STATE_DB`, `packages/harness/src/harness/store.py`.
- [x] Reserve request count and maximum output tokens before inference; failed and cancelled requests keep their charge; unsettled reservations count as acquired. Done: `docs/harness.md` "Acquisition accounting".
- [x] Idempotency keys bound to the full canonical request; a different request under the same key conflicts; retries never regenerate. Done: `Idempotency-Key` handling.
- [x] Per-principal concurrency cap and reservation leases; outstanding reservations survive horizon rollover. Done: `HARNESS_IN_FLIGHT_LIMIT`, `HARNESS_RESERVATION_LEASE_SECONDS`.
- [x] Per-principal and organization output-token budgets, rolling request and token budgets, trust tiers, contractual organization budgets and supervision-weighted budgets. Done: `docs/harness.md` "Rolling budgets" and "Weighted and tiered rolling allowances".
- [x] Separate admission (`decision`) and settlement (`release`) events; a rejected proof releases no new label. Done.
- [x] Crash test with a stand-in teacher. Done: `tests/test_crash.py`.
- [ ] Add request-plus-output-token quota as a named baseline arm. `distill gateway collect --policy` accepts only `mmd`, `quota`, `rate`, `allow` and `none` (`gateway_cli.py:461`).
- [ ] Support streaming. Reserve the maximum output cap or issue bounded incremental grants per chunk, meter released prefixes, and on cancellation charge the emitted prefix rather than refunding it. Responses are currently buffered.
- [ ] Add fencing tokens or monotonic reservation versions, so an expired lease cannot grant the same entitlement again while the old worker can still emit. The current rule that an expired worker must withhold covers part of this; check it against the full failure model.
- [ ] Derive and test the unrecorded-exposure bound: at most n times g tokens when n grants are in flight and each worker can emit g tokens after its last durable grant.
- [ ] Measure and report detection-lag overshoot: about R times L extra units at campaign rate R and analysis delay L, plus reserved in-flight work.
- [ ] Give proof submissions a separate abuse limit, since they count as load without releasing new supervision.
- [ ] Keep provider cost (including hidden reasoning tokens) separate from delivered supervision in the ledger.
- [ ] Reconcile the JSON-lines decision log with database commits. A crash after commit can omit the aggregate event.
- [ ] Reconcile `off_path_score` records with the reservation ledger, and build the durable multi-process account store that `off_path` needs.
- [ ] Persist the aggregate monitor so more than one worker can run.
- [ ] Define an operational reconciliation process for pending idempotency keys, which are currently kept indefinitely.
- [ ] Check cross-tenant privacy boundaries for any response-cache reuse.
- [ ] Cover every listed accounting failure case with a test: simultaneous near-limit admission, worker death before and after emission, cancellation, duplicate proof redemption, idempotent reconnect, horizon rollover, clock changes, multiple workers and state recovery. Check which of these `tests/test_crash.py` and `tests/test_acquisition_store.py` already cover before adding any.
- [ ] Define component contracts in `apps/harness-api/src/harness_api/app.py` and `packages/harness/src/harness/policy.py`:
	- a principal resolver returning account, verified organization, tier and versioned campaign memberships;
	- a canonicalizer covering domain, messages, tools, attachments, generation parameters and output limits;
	- an admission service that reserves every applicable scope atomically;
	- a scoring service returning typed evidence, a calibration version and an abstention state;
	- a policy evaluator returning allow, space, challenge, hold, reduce tier or refuse, with a reason;
	- a teacher adapter returning stream events and authoritative usage;
	- a settlement service that records release without rerunning admission.
- [ ] Keep raw observation, derived feature, policy evidence and enforcement action in separate event fields, so a new scoring policy can be replayed without pretending it was deployed.
- [ ] Keep the fake teacher so failure scenarios run without the lab GPU.
- [ ] Evaluate holding a suspicious response until checks finish before release, and measure the latency it adds.
- [ ] Document each future revision's contracts, provenance and measured outcomes in `docs/harness.md` and `docs/evaluation.md`, keeping proposed design separate from qualified measurements.

## 10. Identity and campaign evidence

Report: "Identifiability and Sybil limits", "Persist evidence"; `detection-research.md` inference 5; `frontier-practice.md` inference 2.

- [x] Explicit organization mapping with no inferred links; organization budgets. Done: accounts JSON.
- [x] Human review with unlink and budget restoration that keeps the audit trail. Done: `harness review`.
- [ ] Build campaign edges from verified organization or payment, specific infrastructure matches, and rare-template or timing correlation. Treat office NAT, VPNs and public dataset prompts as weak edges.
- [ ] Store each edge's type, evidence, creation time, expiry, confidence and reviewer disposition. Expire uncertain links.
- [ ] Never propagate denial through the transitive closure of weak edges. Weak edges nominate review only.
- [ ] Measure false merges and false splits, including shared NAT, shared public datasets, shared templates, correlated automation and close creation times.
- [ ] Build a fixed, independent cluster null before linkage contributes non-neutral fusion evidence. Treat ranks on adaptively selected clusters as descriptive unless selective validity is proved.
- [ ] Label any graph policy with perfect attack membership as an oracle baseline, not the deployable method.
- [ ] Link the aggregate monitor's exposure (`packages/harness/src/harness/monitor.py`) to evidence-backed campaign review. Never block every account that contributed to a global alert.
- [ ] Separate permissions for abuse review, dataset export and ownership attribution.
- [ ] Model reseller controls, stronger identity checks, contractual research or batch access and cross-provider intelligence sharing as explicit assumptions and costs.
- [ ] Measure the burden of identity verification on privacy and accessibility.

## 11. Detection as evidence

Report: "Detection should contribute evidence rather than impersonate intent"; `detection-research.md` inferences; `frontier-practice.md` inference 3.

- [x] Domain-conditional MMD and cumulative coverage in shadow mode, abstaining on sparse strata. Done: `harness calibrate --shadow`.
- [x] Session, novelty, volume, public-corpus and cluster evidence offline, with fixed convex fusion. Done: `harness evidence-calibrate` and `evidence-check`.
- [x] Source-disjoint separation study with classifier two-sample tests, learned MMD and volume features, and a stop rule. Done: `harness separation`.
- [ ] Keep MMD, instantaneous MMD, request quota and rate limit as named baselines. Keep MMD in alert mode until it beats quota and rate at matched legitimate utility on unseen sources.
- [ ] Organize features into four families with distinct purposes: acquisition, session, semantic and integrity. Make the policy report which family caused each action. Do not count correlated semantic scores as independent corroboration.
- [ ] Add integrity features, none of which exist yet: explicit reasoning-extraction instructions, context-binding misuse and strong account-link evidence.
- [ ] Add the remaining cheap features: exact and template hashes, input length, mixture entropy, repeat clusters, and released output tokens as evidence.
- [ ] Track domain mixture separately from conditional MMD, since conditioning removes broad-coverage differences by design.
- [ ] Include benign topic-ordered sessions in the null.
- [ ] Keep an "unknown client profile" group in evaluation. Base entitlements on verification and contract as well as declared profiles. Count null selection after seeing a score as selection.
- [ ] Test whether coverage works better as a budget weight or as review evidence. Keep raw request and token accounting beside any novelty-weighted quantity.
- [ ] Choose raw versus novelty-weighted budgets from task learning curves. Deduplication can discard valuable diverse stochastic answers, and naive deduplication misses paraphrases.
- [ ] Validate PRADA and SEAT neighborhood-geometry features on text before using them.
- [ ] If multiple window sizes are used, account for the multiplicity and memory cost.
- [ ] Build an exchange classifier only for explicit prohibited actions with labelled examples. Evaluate paraphrases, other languages, legitimate debugging of reasoning interfaces and real normal sessions.
- [ ] Preserve a random sample of first-stage negatives, and a random audit sample that bypasses nomination, for independent labelling.
- [ ] Evaluate any model judge or self-assessment against independent labels, not teacher agreement. Make its timeout fallback explicit and include it in utility tests. Never let it be the sole gate.
- [ ] In shadow mode, measure incremental campaign recall at fixed legitimate restriction and utility, rather than tuning a request-level ROC. Measure whether alerts arrive before the attacker already has enough data.
- [ ] Use bounded sketches or prototypes for per-account history, with exact nearest-neighbor history only for candidates. Check power and numerical accuracy of any approximation such as random features.
- [ ] Follow up the separation study: finer topic cells, larger attack pools, more split seeds. If separation persists, check that it predicts withheld sources and delayed attacks before promoting it to enforcement.
- [ ] Include an explicitly source-shifted condition to measure what a distribution detector can detect well.

## 12. Sequential evidence and calibration

Report: "Sequential methods help only after validity is justified", "Calibration must include the policy"; `detection-research.md` "How should sequential evidence and calibration be rebuilt?".

- [x] `e_process` and `e_cusum`, with Ville thresholds restricted to the e-process. Done: `accumulators.py`.
- [ ] Make calibration in `packages/harness/src/harness/calibration.py` conditional on source and client profile, with explicit abstention and versioned artifacts, and evaluate it on source-disjoint policy horizons.
- [ ] Never let inspected attacks or appeal outcomes enter a calibration role.
- [ ] Use explicitly empirical account-horizon thresholds in the near term: freeze a horizon, score every policy decision over it, and estimate worst-group restriction on held-out independent accounts with intervals.
- [ ] Calibrate alerting, spacing, challenges and hard restriction separately, including latching, repeated tests and recovery rules.
- [ ] Calibrate the complete frozen fusion rule, including domain routing and missing-feature handling.
- [ ] On recalibration, freeze a new version rather than reinterpreting old logs.
- [ ] Keep a time series of score and client-mixture changes. Do not retrain on all recent traffic, since campaigns can contaminate the reference.
- [ ] On drift, distinguish a changed client population, an upstream model update, an encoder change and a campaign.
- [ ] Set alert targets from reviewer capacity (about M times alpha false alerts for M accounts). Make hard-denial targets stricter than review targets and report them per account and organization. Account for real abuse prevalence when stating positive predictive value.
- [ ] For a formal sequential arm, define the filtration and establish conditional superuniformity before claiming Ville bounds. Use non-overlapping windows, and address shared sources, latent topics, adaptation and drift.
- [ ] Calibrate CUSUM and reset procedures by average run length and repeated testing. Never claim an unreset Ville guarantee for a reset or floored process.
- [ ] Select learned fusion weights on development data only and evaluate the full rule once on locked data.
- [ ] Optional: a separate formal-statistics thesis on conditional calibration under account mixtures and adaptive campaign selection.

## 13. Output protection

Report: "Output protection offers a separate, access-dependent research track"; `output-architecture.md` "Which output defenses offer actual protection?".

- [ ] Offer answer-only or compact-explanation tiers only for tasks with a measured legitimate need. Measure solved tasks and explanation need separately.
- [ ] Do not expose hidden reasoning through the wrapper. Respect upstream bindings.
- [ ] If inference is owned, bind protected context to permitted model and account transitions. Validate preceding messages and tools, define mismatch behavior, handle key rotation and legacy keys, and log rejected transfers, dropped protected items and preserved context separately. Test that legitimate multi-turn context survives.
- [ ] Pin reasoning-interface versions and measure semantic continuity rather than HTTP success.
- [ ] Keep keyed deception (`packages/harness/src/harness/deception.py`) as a consented, labelled research arm with explicit utility measurement. Never degrade code, arithmetic or factual answers with hidden noise in normal production.
- [ ] Compare every output intervention with returning fewer tokens or fewer responses at the same legitimate utility. Charge the attacker for forced retries and the defender for extra inference and denied tasks. Include the cost of any external verifier or teacher the attacker uses.
- [ ] Do not present randomized language, temperature changes, a weaker model or mixed teachers as security without student experiments.
- [ ] Run ADS, ADFP and sampler watermarking only in a sampler-access branch, each behind a policy version with quality tests before release.
- [ ] Measure the costs of each output method on the intended serving stack: ADS's two proxy passes (about 1.86 times dense parameter work for a 7B teacher with two 3B proxies), proxy caches and memory, post-hoc rewriting latency, and sampler watermark overhead.
- [ ] Give output defenses adaptive tests: free repetition cleanup, watermark removal, paraphrasing and backtranslation, and model averaging (WASH) where access permits. Test assistant prefill only where the real API allows it, and test user-message prefixes separately.
- [ ] If resources allow, run a small frozen factorial output panel: no intervention, compact answers, a sampler fingerprint, and ADS.
- [ ] Treat training the teacher against transfer as a separate research program only.

## 14. Provenance and attribution

Report: "Provenance is useful after release"; `output-architecture.md` step 7 and the attribution gate.

- [ ] Run attribution as a separate service after a suspected student appears, with registered keys, approved access and a documented hypothesis test. Never count attribution as prevention.
- [ ] Use separate secret keys and thresholds for text detection and suspect-model attribution.
- [ ] Register negative controls before probing. Count candidate models and keys searched, and adjust for multiple comparisons.
- [ ] Include independent clean models, clean fine-tunes, mixed-source students, key-rotation controls and spoofing attacks.
- [ ] Record the suspect access needed to reach the stated confidence, and preserve the access conditions so the test can be reproduced.

## 15. Privacy and retention

Report: "Persist evidence without turning the service into a surveillance archive"; `output-architecture.md` cost model.

- [ ] By default, store only policy counters, timestamps, output usage, keyed prompt digests, coarse domains and calibration metadata.
- [ ] Keep full exchanges only in a restricted, short-lived investigation store with a stated purpose, access logs and deletion rules.
- [ ] Use keyed rather than plain hashes for prompt digests. Treat embeddings as sensitive data.
- [ ] Protect the state database as sensitive data, since it stores response bodies.

## 16. User recovery

Report: "Persist evidence"; `output-architecture.md` design item 5.

- [x] Refusals carry a limit code, scope, reset time and appeal reference. Done: `HARNESS_REVIEW_ENABLED`.
- [ ] Use low-friction verification or review before irreversible restriction on uncertain detections.
- [ ] Give verified enterprise automation explicit contractual budgets.
- [ ] Measure time to review, successful restoration and challenge-loop abandonment.

## 17. Latency and cost

Report: "Keep the critical path small and measure tail latency"; `output-architecture.md` "Cost model and engineering budgets".

- [x] Bounded encoder microbatching. Done: `HARNESS_ENCODER_BATCH_WAIT_MS`.
- [ ] Keep only authentication, canonicalization, reservation, cheap entitlement and integrity checks, and the teacher call on the synchronous path. Include the delay of expensive scoring in the exposure metric.
- [ ] Measure admission p50/p95/p99, time to first token, completed-response latency, token throughput, total GPU-seconds and correct tasks per dollar.
- [ ] Test the proposed targets: at most 10 ms extra p95 admission latency for cheap accounting, and at most 5% end-to-end overhead for the selected wrapper policy.
- [ ] Measure MMD completion-window tail latency, which per-request averages hide. Dense MMD costs about O(k times w squared times d). Measure long inputs and cold caches.
- [ ] Size embedding storage: 1,536 bytes per 384-dimensional float32 vector, about 1.54 GB per million.
- [ ] Measure the cost of the shadow scorers, which currently run on the request path and have not been measured.

## 18. Provider rebuild (only if inference is owned)

Report: "A provider rebuild adds inference controls and organization operations".

- [ ] Put an account service and durable entitlement ledger before the inference router, so the synchronous release policy enforces budgets even when analysis is unavailable.
- [ ] Feed an asynchronous analysis service from privacy-reduced events. Have a separate enforcement service issue versioned, time-bounded actions with a reason and an expiry or review point.
- [ ] Keep detector evidence separate from account status. A new link can reduce a shared allowance without marking linked accounts as historically malicious.
- [ ] Train and audit any activation probe for distillation specifically, not reuse a jailbreak detector. Use self-assessment only for retrospective review prioritization.
- [ ] Meter each released chunk in the output gateway, with streaming holds where checks run.

## 19. Deployment and rollback

Report: "Deployment needs drift controls and a way to recover".

- [ ] Run shadow observation on representative legitimate clients first, against the allow, quota and rate baselines.
- [ ] Introduce hard controls first where their meaning is clear, such as explicit entitlement exhaustion. Use a small canary for uncertain evidence-triggered spacing or review, and keep a holdout audit stream.
- [ ] Monitor per-client correct-task completion, challenge abandonment, review backlog and restoration time alongside attack indicators.
- [ ] Make rollback stop a bad policy without erasing durable usage or regenerating old samples. Keep prior calibration artifacts and policy versions for replay.
- [ ] Freeze a new study when output channels or account semantics change. Note that a model update can change answer length and token costs.

## 20. Validation protocol additions

Report: "Validation must measure capability, utility and acquisition together"; `output-architecture.md` "Frozen study design"; `detection-research.md` acceptance criteria.

Already present: five source-family roles with near-duplicate screening (`scripts/recalibrate-profiles.sh`), sealed confirmatory partitions, teacher-capacity preflight, matched-budget and matched-utility comparison views (`compare_conditions`), the student-plan checkpoints and controls, and most adaptive attack variants (topic order, bursts, deduplication, answer-only, batching, selective retries, verifier filtering, alternative-teacher mixing, session replay, prefill, rotation, A8).

- [x] Fix the workload topic-truncation bug in a new workload version, with coverage checks. Done: category balancing in `gateway-clients-v3` and `gateway-clients-v5`. A fresh calibration on the fixed workload is still covered by item 4.
- [ ] Add a temporal-drift evaluation period, separate from the locked final test.
- [ ] Balance legitimate and attacker sources so corpus membership cannot win the comparison by itself.
- [ ] Add the missing legitimate client types: multilingual sessions, repetitive translation, and stable long-term users versus new accounts. Replay genuine sessions only with consent or a defensible licence. Hold out organizations as well as source families.
- [ ] Add the missing campaign types: explanation-focused collection, explanation requests in other formats, purchased or replayed genuine sessions as a campaign, low-and-slow specialists, campaigns that cannot be linked, uncertain-link cases, and runs with 1, 5, 20 and 100 identities with churn.
- [ ] Add the missing adaptations: backtranslation and semantic paraphrasing beyond the current phrase substitutions, additional unlabelled data, student-side filtering, and attacker knowledge of the budgets and approximate detector family. Test tools, attachments, multi-turn state and protected-reasoning replay only in permitted controlled conditions.
- [ ] Keep the data-pool budget separate from the teacher-query budget.
- [ ] Declare deadlines for every comparison. Rate limits can slow collection without changing eventual capability. Benchmark honest batch use before selecting rate limits.
- [ ] Add an alternative or public-teacher student control, so gains can be compared with what the attacker could get elsewhere.
- [ ] Extend the baseline panel with conditional MMD in alert mode plus quota, and the selected proposed policy, once development ends.
- [ ] At the final conditions, train at least two student families with three seeds each. Add far-transfer tests and held-out problem lineages. Do not use a style or preference judge as evidence of capability.
- [ ] Freeze the capability target on development curves, for example half the teacher-over-base gain. Report cost and time to that target, with censoring.
- [ ] Report the full denominators for every account and campaign: admitted attempts, completed and withheld responses, emitted prefixes, released tokens, distinct examples, time to each level, release before the first restriction, and totals by deadline across identities, including acquisition after a nominal detection. Separate first alert, first reduced service and first hard refusal; an L1 alert is not a blocked request. Count all accounts and scored accounts separately so warm-up evasion is visible.
- [ ] Report campaign-graph precision, recall and false merges, CPU cost and retained bytes.
- [ ] Report legitimate outcomes per profile, never only a pooled benign average, including explanation availability, recovery burden, review load, ledger throughput and memory. Report organization-level denominators for shared restrictions.
- [ ] Use a hierarchical analysis or separate conditional-variability summary for training seeds.
- [ ] Plan for about 300 independent benign horizons per priority group for a zero-event bound near 1%.
- [ ] Add the two missing acceptance gates to `docs/evaluation.md`: attribution (clean and mixed-source controls meet a declared accusation-level error target) and deployment cost (overhead within a declared latency and compute budget, with sustainable review demand).
- [ ] Add the rejection rules to `docs/evaluation.md`: reject semantic restriction triggers that lose all value under topic reordering or cannot beat budgets for specialists; reject a fingerprint as prevention if student capability is unchanged; reject ADS for deployment if cleanup, retries or permitted inputs erase its matched-utility gain.
- [ ] Have stakeholders choose the final utility and security thresholds before any confirmatory run.
- [ ] Record, for each change, whether it needs recalibration and a new frozen study:

| Stage | Work | Recalibration | New frozen study |
|---|---|---|---|
| Research hygiene | Provenance, admission versus release, manuscript corrections, warning-only status | No | No efficacy claim from this alone |
| Systems foundation | Durable ledger, idempotency, streaming settlement, restart-safe horizons | Not for the accounting invariant; limits still need selection | Yes, before claiming security or utility gains |
| Workload repair | Topic coverage, specialists, automation, genuine sessions | Yes | Yes; never mutate the frozen experiment |
| Conditional detection | Client and domain references, abstention, coverage, frozen fusion | Yes, for every changed score and the full policy | Yes, with fresh source-disjoint confirmation |
| Identity and campaign controls | Verified budgets, uncertain links, review and recovery | Empirical linkage and intervention calibration | Yes, including false merges and splits and rotation costs |
| Output tiers | Compact answers, reasoning exposure, optional sampler experiments | Utility and transfer; detector recalibration if features change | Yes, with adaptive students and matched-output controls |
| Provider-only | Probes, cryptographic context binding, ADS, ADFP, training changes | Method-specific | Separate study with declared access |

## 21. Research contribution

Report: "A defensible contribution needs a sharper claim than an ensemble"; `output-architecture.md` "Originality ledger"; `detection-research.md` candidates A to C; `frontier-practice.md` publication inferences.

- [ ] Main track: utility-constrained, identity-aware acquisition allocation. Choose a policy that maximizes the minimum attacker spend to reach a frozen capability threshold, subject to a lower bound on legitimate correct-task completion in every prespecified client group. The attacker optimizes prompt selection, output requests, account splitting and cleanup, with identity price, outside data, deadline and offline compute as explicit constraints.
- [ ] Measure task learning curves G_c(B) and compare raw request and token budgets with supervision-weighted budgets.
- [ ] Report attack dollars per capability point, transfer against denied specialist utility, and review load.
- [ ] Systems foundation: a restart-safe release bound under a declared crash and concurrency model, paired with an empirical capability envelope and an open benchmark of failures.
- [ ] Optional, needing sampler access and training budget: a factorial study of whether acquisition limits constrain cleanup, retries and mixing enough to give an output method value beyond fewer-output controls.
- [ ] Alternative: specialist-safe composite nulls under uncertain campaign attribution, comparing global, account, verified-organization and inferred-cluster detectors at fixed attacker spend, with worst-group intervention and false merges.
- [ ] Alternative: an online adaptive same-source acquisition benchmark with real enforcement, identity ambiguity and cost-constrained adaptation, distinguished from D01's static lifecycle tables.
- [ ] Fallback: a systematic source-disjoint counterevaluation of current TRACE and public output defenses, presented as an empirical contribution rather than a new algorithm.
- [ ] Title the work around measured acquisition risk under legitimate-utility constraints, not prevention of all distillation. Publish negative results if conditional detectors fail.

## 22. Phases and resources

Report: "Resource planning should narrow the study before buying compute"; `output-architecture.md` "Phases and estimated resources". All figures are estimates.

- [ ] Phase A, CPU only, 1 to 3 days: provenance and split audit, protocol, offline replay, accounting design.
- [ ] Phase B, 2 to 5 days: accounting prototype with a fake teacher and failure injection, then a 100-prompt generation pilot with realistic context lengths to measure token throughput and training speed. At 1,024 output tokens, 6,000 examples are about 6.14 million teacher tokens per condition, roughly 34 hours at 50 tokens per second or 8.5 hours at 200.
- [ ] Phase C, 1 to 3 weeks: a narrowed student matrix. Four conditions times two families times three seeds is 24 fits per acquisition point, or 96 over four points, about 192 to 768 GPU-hours before generation and evaluation.
- [ ] Phase D, 2 or more weeks: held-out simultaneous benign and attack runs, deployment shadow measurements and appeals review.
- [ ] Select policies offline on development data, then train students only for the strongest baselines and the final candidate. Use early checkpoints to decide whether a larger campaign is informative. Never use early final-test student results to select the policy.
- [ ] Benchmark training on the actual hardware rather than assuming that a 24 GB or 80 GB card fits a configuration.

## Sources cited

- D01: Liu et al., "Do Defenses Against LLM Extraction Work Across Attacks? A Lifecycle Benchmark of Black-Box Model Extraction", arXiv preprint, 2026. https://arxiv.org/html/2610.00839v1
- D03: Juuti et al., "PRADA: Protecting against DNN Model Stealing Attacks", EuroS&P 2019. https://arxiv.org/html/1805.02628v3
- D04: Zhang, Chen and Wagner, "SEAT: Similarity Encoder by Adversarial Training for Detecting Model Extraction Attack Queries", AISec 2021. https://surrealyz.github.io/files/pubs/seat-aisec2021.pdf
- D13: "Stealing and Defending the Ends of LLMs", ICLR 2026 OpenReview submission (excerpt only). https://openreview.net/pdf?id=dOS3e1kC5u
- D20: Bhattacharyya and Ramdas, "Change detection with conformal martingales: new optimal constructions, and suboptimality of existing methods", arXiv preprint, 2026. https://arxiv.org/html/2609.27179v1
- D21: "Non-partitioned e-detectors for nonparametric sequential change detection", arXiv preprint, 2026. https://arxiv.org/html/2607.28322v1
- D24: Hasan, "Bounded Behavioral Indistinguishability for Black-Box LLM Distillation", arXiv preprint, 2026 (abstract only). https://arxiv.org/abs/2605.30448
- D25: Chandrasekaran et al., "Exploring Connections Between Active Learning and Model Extraction", USENIX Security 2020. https://www.usenix.org/conference/usenixsecurity20/presentation/chandrasekaran
- D26: Orekondy et al., "Knockoff Nets: Stealing Functionality of Black-Box Models", CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Orekondy_Knockoff_Nets_Stealing_Functionality_of_Black-Box_Models_CVPR_2019_paper.html
- D27: Pal et al., "ActiveThief: Model Extraction Using Active Learning and Unannotated Public Data", AAAI 2020. https://ojs.aaai.org/index.php/AAAI/article/view/5432
- O05: Ullah and Khojasteh, "ADS-C: Antidistillation Sampling for Classification", arXiv preprint, 2026 (abstract only). https://arxiv.org/abs/2607.15467
- ADS: Savani et al., "Antidistillation Sampling", 2025. https://arxiv.org/html/2504.13146v5
- ADS reevaluation: Libon et al., "What Does It Mean to Break a Distillation Defense?", arXiv preprint, 2026. https://arxiv.org/html/2606.25059v2
- ADFP: Xu et al., "Antidistillation Fingerprinting", arXiv preprint, 2026. https://arxiv.org/html/2602.03812v2
- WASH: Wu et al., arXiv preprint, 2026. https://arxiv.org/html/2605.30501v1
