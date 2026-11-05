# TRACE detection product roadmap

Updated 2026-11-05. This is the active implementation and testing backlog for extraction-risk detection at the teacher API. Its scope follows the [research report](<docs/reports/Distillation detection methods and standards.md>) and the [academic](<docs/research_notes/Distillation detection methods and standards/academic.md>), [industry](<docs/research_notes/Distillation detection methods and standards/industry.md>) and [standards](<docs/research_notes/Distillation detection methods and standards/standards.md>) notes.

All checkboxes below describe remaining work. Existing plumbing is a starting point, not completion of these product requirements. Mark an item complete only with linked implementation, relevant checks and the evidence named in its section. Record software completion separately from measured detection performance and live operational qualification.

An initial [experimental campaign detector](docs/campaign.md) implements bounded per-principal state, component evidence and diagnostic API treatment, with CPU regression coverage. The items below remain open until their full completion evidence exists; this partial implementation does not establish calibrated detection or prevention.

## Starting point and release order

TRACE already has domain-conditional MMD, rarefied coverage, session/novelty/corpus signals, offline linkage and fusion, durable accounting, quotas, review and experimental whole-path stopping. The chronological [checkpoint workflow](docs/checkpoints.md) can plan and run GSM8K students from actual acquired prefixes. Most evidence signals remain telemetry-only; shadow history is process-local, and experimental path state grows with the full prefix. No effective general-purpose distillation detector or live student-capability prevention result is established.

The first product release is a gateway with persistent campaign evidence, calibrated alerts, reversible API treatment and an operator investigation workflow. It must address ordinary final-answer collection as well as any separately supported protected-reasoning threat. Identical observable transcripts provide no passive evidence of different downstream uses. Quota, cost and transport failures must never be relabeled as successful extraction detections.

| Stage | Work | Dependencies | Exit evidence |
| --- | --- | --- | --- |
| First implementation | D0 contract, D1 durable monitor, D2 incremental signals, D3 combined scorer, D5 API treatment and D6 incident workflow | Existing store, accounting, signal modules and review | One observable request-to-evidence-to-action-to-restoration path, with CPU integration coverage |
| Qualified detector pilot | D4 calibration, T0 reliability, T1 data, T2 adaptive attacks and T3 detector/utility study | Freeze the actual deployed procedure and obtain independent data | Declared false-alert burden, detection power, delay and resource costs on locked evaluation |
| Capability and product release | T4 chronological students, T5 operating qualification and G1 release gates | Passing pilot, teacher capacity and clean-student gates | Live evidence supports exactly the product claims being made |
| Follow-on detection | C1 coordinated accounts, C2 learned classifier, C3 protected reasoning and C4 statistical extensions | Each section's data, telemetry and calibration prerequisites | Independent evidence before enabling additional treatment |

Keep implementation in the existing harness, store and API where appropriate. Extend the existing evaluation commands instead of creating a parallel experiment stack. Preserve frozen readings and legacy defaults; new detector behavior and artifacts must be explicitly versioned and opt-in. Existing shadow scorers retain their telemetry-only contract.

## D0. Freeze the product and measurement contract

- [ ] D0.1 Define supported endpoints, teacher models, authenticated identity scope, legitimate workflows and attacker capabilities. Separate final-answer collection, protected-reasoning extraction and cross-account coordination in the threat model.
- [ ] D0.2 Specify a compact campaign-monitor interface for admission observations, release settlements and evidence retrieval. Define ordering, deduplication, failure behavior and per-request resource limits before implementation.
- [ ] D0.3 Version the encoder/serialization, references, features, routing, score, missingness rules, monitoring scales, resets, horizon, action thresholds and review behavior as one reproducible detector contract.
- [ ] D0.4 Declare the pilot's false-alert budget, utility tolerance, detection-delay target, memory/storage limits and latency budget before selecting the detector. These are product choices, not industry-certified thresholds.

Completion evidence: a reviewable contract and artifact schema distinguish observed risk, unknown evidence, confirmed incidents, acquisition limits and student capability. Sources: [industry findings](<docs/research_notes/Distillation detection methods and standards/industry.md>) and [standards mappings](<docs/research_notes/Distillation detection methods and standards/standards.md>).

## D1. Persist bounded campaign state

- [ ] D1.1 Add durable campaign state scoped to tenant, authenticated principal and verified organization. Preserve mapping provenance and key rotation/revocation history; never trust client-supplied identity or workflow labels without validation.
- [ ] D1.2 Track both request-count and wall-clock scales, with bounded recent buffers and compact longer-lived sketches. Declare retention, inactivity expiry, resets and any lifetime monitoring limits.
- [ ] D1.3 Persist observation positions, deduplication identities, feature state, score state, calibration binding, alerts and action history atomically. Concurrent workers must not lose, reorder or double-count observations.
- [ ] D1.4 Reconcile query attempts, proof redemptions, retries, reservations, releases, failures and cancellations with the existing ledger. Keep actual released tokens separate from conservative budget charges and unknown usage.
- [ ] D1.5 Define crash recovery, outstanding reservations, stale workers, leases, clock behavior and restart recovery. Neither detector history nor collection allowance may reset accidentally.
- [ ] D1.6 Replace full-prefix recomputation/serialization in the product monitor with bounded work. Leave legacy experimental path artifacts interpretable; do not silently reuse their calibration for a changed scorer.
- [ ] D1.7 Decide synchronous versus queued scoring. If queued scoring is supported, persist jobs and application positions, bound backlog, reject stale results and measure releases while evidence is pending. The existing in-memory off-path mode is not a durable product queue.
- [ ] D1.8 Add state migration, incompatible-contract rejection, retention cleanup, backup/restore and deterministic event replay. Define explicit behavior when the database, encoder or references are unavailable.

Completion evidence: T0 proves durable, ordered, bounded state under retries, worker races and failures; resource growth remains within D0 limits over the supported lifetime. Starting files: [store.py](packages/harness/src/harness/store.py), [shadow.py](packages/harness/src/harness/shadow.py) and [path runtime](packages/harness/src/harness/path_runtime.py).

## D2. Adapt research signals for online use

- [ ] D2.1 Reuse conditional-domain MMD with references matched to supported task, endpoint/model, language and legitimate workflow. Verify identical calibration/serving serialization, embedding normalization and scoring statistics. Unsupported cohorts abstain explicitly.
- [ ] D2.2 Add bounded semantic nearest-neighbor history and template-family repetition for paraphrase sweeps and local exploration. Distinguish idempotent retries from new collection attempts; retain legitimate editing, tutoring and debugging as negative controls.
- [ ] D2.3 Add request-count-matched coverage/novelty summaries for systematic broad collection. Require permutation invariance for features advertised as order-invariant coverage and report ordered novelty separately. Compare rarefied cells and incremental projected log-determinant features; test volume confounding and topic sorting before selecting one.
- [ ] D2.4 Incrementally compute supported session features: independent questions versus follow-ups, rephrasing, template diversity, topic transitions, interarrival variability, sustained activity and answer-format preference. Test languages and prompt transformations before relying on lexical heuristics.
- [ ] D2.5 Use attempts, concurrency, released tokens and exposure velocity as context with explicit missingness. High volume, broad topics, structured output or answer-only requests alone must not establish prohibited training.
- [ ] D2.6 Keep public-corpus resemblance as optional provenance evidence. Freeze index versions, test legitimate benchmark users and remove source/generator labels that would leak evaluation ground truth.
- [ ] D2.7 Version feature extraction and any learned similarity encoder. Cache safely within the tenant/contract scope, bound encode cost and test feature failures, long inputs and missing embeddings without substituting fake evidence.

Completion evidence: deterministic fixtures establish feature semantics and resource bounds; T1/T3 show which features retain detection value after task, topic, source, volume and timing controls. Compare existing shadows and each candidate individually before fusion. Sources: [PRADA](https://arxiv.org/html/1805.02628v3), [SEAT](https://people.eecs.berkeley.edu/~daw/papers/seat-aisec21.pdf), [VarDetect](https://arxiv.org/pdf/2107.05166) and [LLM MMD](https://arxiv.org/html/2606.05725v1). Their published thresholds do not transfer to TRACE.

## D3. Implement a combined campaign scorer

- [ ] D3.1 Start with a transparent frozen combination of supported signals. Retain component values, reference support, contributions and abstention reasons with each score. A raw score is not automatically a probability of distillation.
- [ ] D3.2 Define how missing families, incomplete windows, sparse strata and delayed observations affect scoring. Freeze neutral contributions or abstention behavior; do not silently renormalize weights after evidence disappears.
- [ ] D3.3 Combine short-burst and longer-running evidence without counting shared requests as independent samples. Bind every score to a campaign position and declared look schedule.
- [ ] D3.4 Compare the combination with pooled MMD, conditional MMD, similarity-only, coverage-only, behavior-only and simple resource baselines. Retire features that add cost or confounding without held-out value.
- [ ] D3.5 Keep an immutable version history, replay command and observation-only rollout mode. Changing feature routing, score direction, weights, references or reset rules requires a new contract and fresh qualification.

Completion evidence: request-path integration produces reproducible risk evidence and T3 demonstrates incremental value at matched legitimate-user burden. The current offline fusion is not automatically a sequentially valid online detector. Starting files: [signals.py](packages/harness/src/harness/signals.py), [evidence.py](packages/harness/src/harness/evidence.py) and [policy.py](packages/harness/src/harness/policy.py).

## D4. Calibrate the complete operating procedure

- [ ] D4.1 Freeze independent representation/reference fitting, score calibration, policy development and locked testing roles. Separate source families, accounts, organizations and appropriate time periods; audit exact and near-duplicate leakage.
- [ ] D4.2 Build sufficiently large legitimate cohort references and an independent audit sample for the declared false-alert budget. Report effective independent sample counts, attainable resolution and uncertainty; overlapping windows do not create independent campaigns.
- [ ] D4.3 Calibrate the maximum alert statistic over the actual multiscale monitoring path, horizon, reset policy and account/group searches. Measure probability of ever alerting on a legitimate account/campaign, not only per-window error.
- [ ] D4.4 Bind score construction and threshold construction to the same statistic, reference averaging and missingness. Reproduce the chosen method with negative controls before combining methods; do not borrow a paper's nominal percentile as a deployment guarantee.
- [ ] D4.5 Require support checks and held-out benign audit before activating statistical restrictions for a cohort. Missing/stale/incompatible calibration produces an explicit unqualified state; independent configured resource limits continue.
- [ ] D4.6 Define drift detection, reference refresh approval, rollback and retraining rules. Freeze reference provenance and prevent untrusted live traffic from silently poisoning the null. Every refresh rechecks false alerts and power.
- [ ] D4.7 Allocate and record error budgets across monitoring periods, detector versions and cluster searches, or explicitly limit the claim to the audited horizon. A durable campaign must not silently rearm a finite-horizon statistical guarantee.

Completion evidence: calibration and independent audit artifacts bind the exact deployed procedure and state their assumptions, uncertainty and unsupported strata. Existing finite-H path stopping remains experimental until separately qualified. Sources: [time-uniform testing](https://proceedings.mlr.press/v202/podkopaev23a/podkopaev23a.pdf), [testing by betting](https://arxiv.org/html/2112.09162v6) and [conditional tests](https://arxiv.org/abs/2506.03898).

## D5. Connect evidence to reversible API treatment

- [ ] D5.1 Integrate the new scorer with admission and release handling behind an explicit detector version. Define observe, alert, temporary throttle, verification/review and confirmed revocation states with expiry and restoration rules.
- [ ] D5.2 Require supported calibration and the declared evidence/corroboration conditions for each automated treatment. Weak behavioral evidence alone must not become a permanent accusation or irreversible stop.
- [ ] D5.3 Keep detector, request quota, token entitlement, rate, concurrency, provider failure and transport failure reasons separate in API errors, telemetry and reports. Preserve existing resource controls across scoring abstention and review.
- [ ] D5.4 Make proof redemption, idempotency, credits, appeals and administrator actions obey the new state machine. Verification cannot erase exposure history or refill budgets implicitly.
- [ ] D5.5 Record alert observation, decision application, server release and client receipt separately where available. Reconcile late/in-flight outputs by identity, without summing overlapping delay intervals twice.
- [ ] D5.6 Test a complete campaign through alert, actual API restriction, operator review, expiry/restoration and subsequent legitimate use. With the feature off, retain legacy contracts, policy revisions, state scopes and frozen outputs.

Completion evidence: T0 and T3 show correct API consequences and recovery, not merely nonzero scores. Existing treatment and review are documented in [harness.md](docs/harness.md).

## D6. Ship investigation, observability and operating controls

- [ ] D6.1 Add a tenant-scoped incident feed and authenticated administrative interface with campaign evidence, score history, affected identities, treatments, expiries and review/restoration decisions. Expose unknown evidence honestly.
- [ ] D6.2 Add signed, retryable alert webhooks with event IDs and delivery status. Use a durable outbox or equivalent recovery so database commits and notification failures cannot silently lose incidents; consumers must tolerate duplicate delivery.
- [ ] D6.3 Enforce tenant isolation, administrator roles, least-privilege keys, rotation/revocation and audited configuration changes. Prevent forged identity headers, cross-tenant evidence access and secret leakage in exports or logs.
- [ ] D6.4 Set retention and access policies for prompts, completions, embeddings, identity links and incident evidence. Prefer minimized summaries where sufficient; make deletion, backup and any retained deduplication semantics explicit.
- [ ] D6.5 Export health metrics for scorer/encoder latency, queue delay, state growth, reference freshness, abstention, missing tokens, webhook failures, action rates and restoration burden. Exclude sensitive content from metric labels.
- [ ] D6.6 Provide installation/onboarding, calibration validation, rollback, backup/restore, capacity and incident-response runbooks. Reject unsupported worker/storage combinations at startup rather than presenting an unsafe deployment as supported.
- [ ] D6.7 Document the supported teacher protocol and release behavior. Add an OpenAI-compatible adapter only as an explicitly tested integration; preserve request identity, usage accounting and treatment semantics through it.

Completion evidence: an operator can configure, inspect, act on and recover from a campaign without reading raw experiment directories. Authentication, isolation, notification recovery and operational limits are covered by T0/T5.

## T0. Software correctness and compatibility

- [ ] T0.1 Add focused CPU tests for incremental features, routing, score combinations, missingness, calibration binding, state migrations and deterministic replay through their public interfaces.
- [ ] T0.2 Add multi-worker integration tests for duplicate requests, shared organization state, ordering races, proof redemption, lease expiry, stale scores and atomic observation/release settlement.
- [ ] T0.3 Inject crashes around admission, state commit, dispatch, release settlement, incident creation and webhook delivery. Verify recovery, no double acquisition and correct treatment of unknown delivery.
- [ ] T0.4 Test encoder/reference/database outages, queue saturation, cancellation, corrupt artifacts, clock jumps, retention boundaries, long inputs and unsupported cohorts. Assert the declared outage behavior and independent budget enforcement.
- [ ] T0.5 Test tenant isolation, admin authorization, identity spoofing, API-key rotation, webhook authentication, evidence redaction and cross-tenant cache separation.
- [ ] T0.6 Keep meaningful existing frozen-schedule, calibration, contract, state and quota regression checks. Run repository pytest, Ruff lint/format and strict mypy through `devenv shell -- uv run`; use `bun` for any new TypeScript workspace.

Completion evidence: required checks pass and failure-injection fixtures exercise actual API/state consequences. CPU fixtures establish software behavior only. Do not add repetitive smoke tests or implementation-mirroring assertions.

## T1. Legitimate data, labels and negative controls

- [ ] T1.1 Obtain consented legitimate traces or documented representative customer pilots. Cover broad researchers, benchmark sweeps, batch labeling, tutoring, iterative coding, document QA, retries, shared enterprise keys and nighttime automation.
- [ ] T1.2 Keep synthetic fixtures for development, with generator versions and limitations. Do not certify real-user false-alert rates from simulated actors whose definitions encode the detector's own heuristics.
- [ ] T1.3 Separate observed behavior, operational flags, independently investigated incidents and experimentally known attack labels. Preserve unknown cases; an earlier detector alert is not ground-truth training intent.
- [ ] T1.4 Add balanced controls for domain, topic, corpus/source, prompt length, request count, repetition, timing, language, workflow and endpoint/model. Audit source-family leakage and train/test connected groups.
- [ ] T1.5 Rebuild insufficient role pools into fresh versioned directories. Keep fitting/development/locked roles and frozen readings immutable; never borrow locked test prompts to fix sparse calibration strata.
- [ ] T1.6 Include same-transcript benign/attack pairs and legitimate coordinated evaluation campaigns. Report these as observability limits and confound controls, not mislabeled negatives to remove until accuracy improves.

Completion evidence: frozen inventories, source/identity/time provenance, label basis and adequacy reports support D4. Existing [separation tooling](docs/evaluation.md#development-only-separation-study) is a development diagnostic, not a final confirmation set.

## T2. Fixed and adaptive attack coverage

- [ ] T2.1 Preserve A1-A7 fixed schedules as regression cases and test A8 online selection. Include ordinary final-answer harvesting that never requests protected reasoning.
- [ ] T2.2 Test topic sorting, benign padding, low contamination, paraphrases/query blinding, prompt wrappers, language changes, diverse public inputs, answer-only collection and multiple questions per HTTP request.
- [ ] T2.3 Test slow collection, human-like bursts, long idle gaps, staggered accounts, credential rotation, verified-organization boundaries and campaigns spanning multiple supported monitoring periods.
- [ ] T2.4 Test malicious accounts sharing legitimate proxy/relay traffic and benign accounts sharing templates or infrastructure. Include weak/small colluding populations, complementary task collection and confidence-ranked uncertain links.
- [ ] T2.5 Test calibration/reference poisoning, score-probing, boundary walking, detector-aware query selection and attempts to reset state through restarts, renewals, replays or review. Pin attacker access to public/client-visible information.
- [ ] T2.6 Implement and evaluate the student-updating A8 uncertainty/selection hook. Record student checkpoints, seeds, model/scorer identity and offline compute; do not count the callback interface or a phrase rewrite as a learned adaptive attack.
- [ ] T2.7 Charge every online HTTP attempt, failure, retry, proof and cover request to shared campaign resources. Account for external teacher access and offline selection/training cost explicitly, including any multi-provider threat supported by the experiment.

Completion evidence: locked attacks include strategies that evade alarms while retaining useful learning. Unknown or unsupported adaptations remain named limitations, not assumed successes.

## T3. Detector performance, legitimate utility and baseline comparison

- [ ] T3.1 Report per-cohort and overall precision/recall at stated prevalence, campaign detection probability, account/campaign probability of any false alert over the declared horizon, uncertainty, abstention and incomplete/censored campaigns.
- [ ] T3.2 Report requests/tokens released before first alert and first effective treatment, wall-clock delay, queued/in-flight overshoot and unknown usage. Keep server evidence and client receipt bounds distinct.
- [ ] T3.3 Measure legitimate task success, completion, latency, throttling, proof/verification cost, review load and restoration time by workflow. Customer burden includes false restrictions and operator work.
- [ ] T3.4 Automate allow/candidate/quota/quota_tokens/rate development-grid collection with teacher capacity binding and fresh state. Derive per-stratum utility losses from actual runs and bind selected settings to those artifacts.
- [ ] T3.5 Freeze matched baseline settings on development data using the existing quota matcher, then confirm held-out utility. If no single setting matches all declared strata, report no match instead of interpolating favorable strata.
- [ ] T3.6 Run equal attacker-attempt/time/account/cost comparisons as the primary access-control experiment. Add equal-released-query/token controls to isolate information value without hiding the access prevented.
- [ ] T3.7 Run signal ablations and pooled/conditional/similarity/behavior/coverage comparisons on frozen data. Report matched controls, source imbalance, classifier label limitations and negative or inconclusive findings.
- [ ] T3.8 Audit continuous monitoring, sparse cohorts, drift, version changes and resets over the full deployed path. Treat finite-horizon guarantees as finite; do not extrapolate marginal FPR or synthetic AUC to lifetime product behavior.

Completion evidence: an independent report establishes what the detector adds beyond inexpensive access limits at comparable legitimate-user burden. The [utility and quota workflow](docs/evaluation.md#utility-validity-and-measured-quota-matching) already exists in part; measured-grid orchestration and automatic run binding remain work.

## T4. Capability transfer and chronological intervention testing

- [ ] T4.1 Preregister a useful capability target with a source-disjoint evaluation suite, passing teacher-headroom/clean-student gates and pinned student recipe/seeds. Report if the base already meets the target.
- [ ] T4.2 Reuse checkpoint exports for response counts, first-alert/restriction/refusal bounds and campaign end. Retain late/in-flight responses, tied receipts, acquired-versus-trainable counts, pending-quality opt-in and censored targets.
- [ ] T4.3 Extend checkpoint joins only where required for replay, batching, response selection or other supported gateway variants. Freeze raw received-response provenance and attribution rules; keep ambiguous variants refused until tested end to end.
- [ ] T4.4 Run real GPU students on what every arm actually released, with matched-volume mechanism controls, paired seeds and verified resume artifacts. Report target crossing as first observed at a checkpoint, not an exact unobserved learning time.
- [ ] T4.5 Add task-specific checkpoint eligibility, overlap checks, metrics and runner support for AG News, coding or QA only when required by the product claim. GSM8K support alone cannot qualify cross-task protection.
- [ ] T4.6 Report retained student gain for adaptive campaigns, collection exposure, training/selection cost and legitimate utility together. Failure to reach a target within the experiment is censored evidence, not proof of impossibility.

Completion evidence: actual teacher/GPU results support any claim about reduced learned capability. Existing fixture coverage and checkpoint plumbing alone support no prevention claim. See [checkpoints.md](docs/checkpoints.md) and [restoration boundaries](docs/distillation-defense-plan.md).

## T5. Live operating and performance qualification

- [ ] T5.1 Benchmark encoder and scorer overhead, p50/p95/p99 API latency, sustained throughput, storage growth and memory on target hardware. Exercise maximum supported campaigns, tenants and idle-state retention.
- [ ] T5.2 Measure locking, contention, queue delay and backpressure with concurrent workers. Validate SQLite deployment limits before promising horizontal scale; add another store only against a demonstrated requirement.
- [ ] T5.3 Run a bounded observation pilot on representative authorized traffic, review sampled alerts and misses, freeze changes, then conduct an independent supported-enforcement pilot with rollback available.
- [ ] T5.4 Demonstrate full recovery from process/host failure, backup restoration, configuration rollback, reference expiry, webhook outage and provider timeout without silent allowance resets or lost incident records.
- [ ] T5.5 Keep local fixture checks, replay/development studies, live teacher collection, GPU student outcomes, customer pilots and deployment activation as separate evidence states in the release record.

Completion evidence: measured costs fit D0 budgets, the supported deployment topology is explicit and failures have exercised recovery paths.

## C1. Follow-on cross-account coordination

- [ ] C1.1 Add centered/normalized semantic direction summaries and bounded candidate-link searches. Evaluate random-Fourier/spectral coordination against simpler overlap/timing baselines before choosing the algorithm.
- [ ] C1.2 Corroborate behavior using verified organizations and available identity/infrastructure evidence. Preserve link confidence, provenance, expiry and unlink decisions; shared IP alone never establishes common malicious ownership.
- [ ] C1.3 Calibrate the complete cluster-selection/search procedure with an independent benign cluster null. Adaptively selected components cannot inherit an individual-account false-positive guarantee.
- [ ] C1.4 Test false merges/splits using independently supplied identity truth, including enterprise NATs, shared keys, resellers, proxy mixtures and legitimate coordinated evaluations. Unknown identity truth remains unknown.
- [ ] C1.5 Audit small colluder populations, low-rate/padded campaigns, complementarity, poisoning, real-user relays, cluster churn and alarm delay. Connect tentative findings to review before any punitive group action.

Prerequisites: D1-D4, sufficient independent account populations and a declared linkage data policy. Source: [coordination research](https://arxiv.org/html/2610.13083v1). Vendor deployment descriptions do not supply transferable cluster thresholds.

## C2. Follow-on learned session and similarity models

- [ ] C2.1 Collect independently investigated labels and hard benign examples before fitting an operational campaign classifier. Start with a transparent baseline; compare a learned model only for measured held-out improvement.
- [ ] C2.2 Separate representation/model fitting, hyperparameter selection, score calibration and final source/time/organization-disjoint testing. Evaluate unseen attack strategies, workflow shift, class prevalence and operational versus oracle label sensitivity.
- [ ] C2.3 Test whether predictions survive removal of dataset/source, synthetic actor, identity and automation shortcuts. Calibrate classifier outputs before presenting probabilities and document unsupported populations.
- [ ] C2.4 Evaluate a learned text-similarity encoder only if bounded nearest-neighbor/template features fail on valuable attacks. Include paraphrase robustness, natural editing/retries, latency and adaptation by a white-box attacker.

Prerequisites: T1 labels and D3/T3 baselines. Source: [session-classifier report](https://arxiv.org/html/2609.38239v1). Its generated-session results are not production validation.

## C3. Conditional protected-reasoning and artifact defense

- [ ] C3.1 Inventory whether the teacher exposes hidden reasoning, reusable reasoning artifacts, prefills, tool outputs or streamed protected content. Mark this work not applicable to unsupported channels instead of claiming generic coverage.
- [ ] C3.2 If applicable, implement a narrow elicitation/reconstruction detector, with a cheap first stage and richer second stage only where justified. Distinguish ordinary explanations and debugging from attempts to recover protected internal content.
- [ ] C3.3 Bind protected artifacts to authorized user/workspace/organization/model/context, close replay paths and check every supported release path. Hold protected content before emitting it; test tool, provider-adapter and streaming paths independently.
- [ ] C3.4 Red-team encoding/translation, wrappers, role changes, replay, cross-context/model/account reuse and repeated probing. Measure request-level FPR, residual leaked reasoning tokens and actual student learning separately from final-answer campaigns.
- [ ] C3.5 Evaluate output summarization or restricted channel/logprob exposure only when compatible with the product. Measure legitimate task impact and retained training value; never transplant another provider's hidden-reasoning success rate to TRACE.

Prerequisites: a real protected channel and its supported provenance/release contract. Sources: [Anthropic](https://www.anthropic.com/news/distillation-defenses) and [OpenAI](https://openai.com/index/disrupting-a-coordinated-model-distillation-campaign/).

## C4. Statistical research extensions and deferred methods

- [ ] C4.1 Evaluate a genuine sequential MMD/betting or time-uniform detector only after specifying the null, predictable sampling, overlap, reference updates, drift and account/group multiplicity. Reproduce current corrections and compare delay/power with D4's calibrated finite path.
- [ ] C4.2 Evaluate formal covariate-conditioned testing only if cohort routing leaves a measured gap and sample/compute requirements are feasible. Document assumptions; conditioning references by task is not automatically formal conditional MMD.
- [ ] C4.3 Reproduce selected published traffic methods with matched negative controls and a pinned implementation before adapting them. Keep original task, access assumptions and window sizes explicit; do not import classic image-model thresholds.
- [ ] C4.4 Record a feasibility decision for HODA, FDINet and training-anchor/internal-feature methods. Keep them deferred unless teacher checkpoint/activation access and LLM-specific validation exist. ATOM's graph-model defense likewise requires a new applicability case.
- [ ] C4.5 Keep watermarking, fingerprints and suspect-model distillation inference in a separate attribution backlog. Any future work needs authorized suspect access, independent-model controls, false attribution/spoofing/removal tests and measured response-quality impact; it cannot substitute for active-campaign detection.
- [ ] C4.6 Preserve the indistinguishable-transcript limit in threat models and reports. Do not generalize decision-tree impossibility results to every LLM defense or claim that increasing detector complexity solves unobservable intent.

Prerequisites: a demonstrated gap beyond D1-D4 and independent evaluation capacity. Sources and scope limits are consolidated in the [academic notes](<docs/research_notes/Distillation detection methods and standards/academic.md>).

## S1. Standards mapping and defensible claims

- [ ] S1.1 Maintain a versioned control/evidence map to NIST AI 100-2e2025 extraction terminology, AI RMF measurement/monitoring/response and AI 600-1 extraction red-teaming. Link each claimed control to implementation and observed evidence.
- [ ] S1.2 Map authentication, token-aware budgets, rate/burst/concurrency/size/time limits, hard spending caps, logging and circuit breakers to OWASP API4/API6:2023 and GenAI LLM06:2026. Check gaps against existing controls rather than implementing duplicate accounting.
- [ ] S1.3 Map tested attacks to ATLAS extraction/inference/proxy techniques and mitigations AML.M0004, AML.M0024 and AML.M0035. Record the edition and source checked; older category numbers must retain their year.
- [ ] S1.4 Keep formal ISO management-system conformity separate from research performance. Public scope summaries and framework mapping cannot support an ISO-certified detector or a NIST-approved algorithm claim.
- [ ] S1.5 Maintain threat/data/model documentation, incident response, appeals, retention/access policies and a release evidence register. Label provider claims, prototypes, inaccessible advisory content, synthetic results and unverified assumptions accurately.

Completion evidence: a customer can trace each security claim to its scope, test and remaining limitation. Source versions and exact mappings are in the [standards notes](<docs/research_notes/Distillation detection methods and standards/standards.md>); no reviewed source prescribes an industry-standard extraction detector.

## G1. First-release acceptance

- [ ] G1.1 D0-D6 deliver a working request-to-durable-evidence-to-alert-to-treatment-to-restoration path. New defaults remain opt-in until supported activation is explicitly configured.
- [ ] G1.2 T0 passes required software checks; T1 supplies adequate independent populations; T2/T3 measure detection and customer burden against the declared alternatives and baselines.
- [ ] G1.3 T5 qualifies resource bounds, deployment topology and recovery on the intended host. Publish supported cohorts, retention, horizon, failure behavior and known evasion cases.
- [ ] G1.4 Complete T4 before claiming reduced learned capability or prevention. If only detection/acquisition evidence exists, narrow the release claim accordingly and keep the capability gate open.
- [ ] G1.5 Publish negative/inconclusive results, abstentions, false-alert intervals, delayed releases, review/restoration burden and any unmatched baseline. Do not turn a passed fixture, ignored alert or incomplete campaign into qualification.

## Research-to-work coverage

| Research finding | Implementation or explicit disposition | Validation |
| --- | --- | --- |
| Provider campaign classifiers and fingerprints | D1-D3 first; C2 after credible labels | T1, T2, T3, T5 |
| PRADA/SEAT stateful similarity and query blinding | D2.2; optional learned encoder C2.4; reproduce selectively C4.3 | T1 editing/retry controls; T2 transformations; T3 ablation |
| VarDetect and LLM semantic MMD | D2.1, D3, D4; no borrowed vision/text-window thresholds | T1 matched sources; T3 power, dilution and delay |
| Information-gain/coverage and natural-data active learning | D2.3/D2.5; T2.6 learned selection | T1 legitimate broad users; T2 diverse natural inputs; T4 retained gain |
| Cross-account spectral/centered coordination | C1, separate verified identity from inferred links | T1 benign sweeps; T2 relay/poisoning/small groups; C1 false merges and delay |
| Betting/time-uniform/conditional tests | D4 whole-path calibration; C4.1/C4.2 research alternatives | T3 full-horizon error, overlap, drift, resets and multiplicity |
| Reasoning elicitation, replay and artifact isolation | Conditional C3 with a supported protected channel | C3.4 channel attacks/benign explanations and T4 student transfer |
| Account, token, spending and resource limits | D1/D5/D6 and S1.2; reuse existing controls | T0 races/failures; T3 matched utility and exposure |
| HODA/FDINet/ATOM teacher-internal or graph assumptions | C4.4 feasibility gate; deferred from first release | New LLM applicability evidence required |
| Watermark/fingerprint/suspect-model inference | C4.5 separate attribution work | False attribution, spoofing, removal and quality tests |
| Covert learning and observational limits | D0 scope, T1 same-transcript controls and C4.6 | Report indistinguishable and evading cases explicitly |
| NIST/OWASP/ATLAS/ISO scope and evidence limits | S1 and G1 | Traceable controls, red-team evidence and bounded product claims |
