# Build defenses around acquired model capability

A frontier AI company should build an acquisition-control and account-integrity system with model-aware output protection, then use statistical detection as evidence inside that system. TRACE should evolve in that direction rather than replace MMD with another universal intent detector. The practical objective is to increase the money, time and verified identities needed to obtain a useful student while preserving legitimate task completion, especially for specialists and automated enterprise clients. Ordinary useful answers remain useful training data. If legitimate and extraction traffic have the same distribution over everything the service observes, no detector can distinguish them without also rejecting legitimate use at the same rate. Identity assurance, cumulative entitlements, reasoning-channel protection and measured acquisition budgets change the economics; none establishes universal prevention. **No effectiveness gain for the designs in this report has been measured in TRACE.**

Research cutoff: October 31, 2026. This is a design and evidence review, not an implementation, live-service audit or new extraction experiment. The online research covers provider disclosures, peer-reviewed papers, preprints, official documentation, code, practitioner writing and one explicitly unverified anonymous account. Claim-level source IDs link to originals. The annotated source register distinguishes full-text inspection from abstract-only or partial inspection, and provider claims from independent evidence. Public sources cannot reveal every proprietary frontier defense.

## The decision is to evolve TRACE before rebuilding inference

### A practical decision brief

Keep TRACE's reproducible baselines and experimental machinery. Repair the boundary between admission and delivered supervision, persist enforcement state, and make requests plus released output tokens the first controlled quantities. Keep MMD, instantaneous MMD, request quotas and rate limits as named baselines. Treat conditional MMD, semantic coverage, extraction classifiers and campaign linkage as candidate improvements whose incremental value must survive matched-budget and matched-utility comparisons. Do not begin by deploying a large ensemble whose component errors and costs have not been measured.

The most valuable first experiment is a capability learning curve under acquisition constraints. Today the harness can say that a policy released fewer requests; it cannot yet say that the resulting student lost a specified amount of capability. Choose a useful student target on development data, measure the minimum acquisition cost to reach it, and then compare policies on new source families. The attacker should be allowed topic reordering, account rotation, genuine-session replay and answer-only collection. If the defense wins only against the original request ordering, it has learned an artifact of the campaign.

Make hard entitlement decisions explicit. A declared output budget can restrict a client without claiming that the client intended theft. An uncertain content anomaly should usually trigger shadow analysis, a reduced temporary allowance or verification, with a route for legitimate specialists to recover access. Requests that merely alert at level 1 are not blocked requests. Response prefixes already streamed are acquired material even if the final request fails. A framework that confuses those quantities can appear successful while leaving the attacker enough training data.

A complete provider rebuild is justified when the company owns inference serving, customer identity, billing and output channels. That permits protected reasoning, sampler interventions, internal probes and provider-wide account analysis. An opaque API wrapper has a smaller but still useful job: enforce local entitlements, meter release, inspect the observable exchange and preserve auditable decisions. It cannot inspect upstream activations or stop a customer from calling the teacher directly outside the wrapper.

The recommendation has four priorities. First, establish durable acquisition accounting and identity-bound entitlements. Second, rebuild calibration and evaluation around realistic specialists, automation and source-disjoint roles. Third, add selective content and campaign evidence only when it beats the accounting baselines. Fourth, investigate output-side methods in a separate inference-access track. Provenance experiments belong beside that track, with a different endpoint. Evidence that a model absorbed marked text is not evidence that the model failed to learn.

### What the previous investigation actually established

The inherited audit reconstructed the frozen gateway experiment and reproduced all 135 logged window p-values. It found a configuration/statistical mismatch rather than an MMD implementation error that explained the misses. Pooled references represented broad mixed traffic, while benign account windows were mostly topic-concentrated specialists. Broad extraction therefore looked unusually close to the pooled reference. An upper-tail test rewarded it with high p-values. The leaky accumulator then had no positive evidence to accumulate. The workload generator also truncated topics, including a policy-calibration AG News slice containing only Business. That bug worsens calibration credibility but does not explain away the order-evasion result. See the [archived audit](../readings/v4/analysis/REPORT.md).

The frozen A1 full arm admitted and released 6,000 attacker responses. The quota arm released 1,215. The interrupted A2 full log contains 1,537 admissions and 1,536 responses, so it must not be presented as a completed 6,000-request campaign. Offline topic reordering of the same prompts defeats the original upper-tail, lower-tail and two-sided variants without spending more teacher queries. The exact replay verifies consistency of the checked code and artifact; it does not prove the absence of every bug or verify the unavailable lab service. The audit's later replay results are deterministic confirmation of the same data, not new independent observations.

The current code inspection used `main@origin` at `43464d2e`. The harness and API paths were unchanged from warning commit `46d68412` at that inspection. An external consolidation subsequently moved the audit into `readings/v4` while this research ran, associated with `a698fe54`. The actual report is now under `readings/v4/analysis/REPORT.md`; historical absolute paths inside the archive are provenance. Neither the old audit nor this review establishes the live lab revision or activation state. The calibration warning helps expose a blind spot. It changes no enforcement behavior.

The source audit is instructive. The original MMD paper's reported results involve 1,500-query windows and a different attack/benign construction. Its 100% pure-attacker detection claim does not transfer to TRACE's 100-request heterogeneous account null. The paper itself warns about semantic proximity, contamination and drift. A subsequent lifecycle preprint reports MMD detecting 100% of QEDKS batches and 0% of fixed-pool SeqKD batches, using 20 attack and only four benign test batches. Those are scoped batch results, not production campaign protection or a low false-positive certificate. [D02] [D01]

## Frontier disclosures support layers, with narrower proof than headlines

### Providers increasingly protect reasoning and account integrity

Anthropic's February 23, 2026 disclosure attributes more than 16 million exchanges through roughly 24,000 fraudulent accounts to three labs. Its described response combines classifiers, behavioral fingerprints, coordinated-account detection, account verification and links across infrastructure, payment and request timing. That is strong evidence for a practical layered operational design, but it is an interested provider's account of its own telemetry. The post does not publish the undetected population, a controlled detector comparison or a population false-positive rate. It also describes a proxy service mixing many customers, which makes a simple equation between an IP address and an attacker untenable. [F01]

The October 12 disclosure is more specific about reasoning protection. Anthropic describes summarization and binding of protected context, classifiers, restrictions and billing for blocked attempts. It reports early estimates of a 10 to 100-fold reduction in harvested reasoning tokens and a false-positive figure below 0.1%. It also reports 80.4% versus 71.8% SWE-bench Verified for a pretrained student using full versus summarized transcripts. The outcome is narrower than preventing answer-only distillation, and the source acknowledges deployment changes and incomplete experience against highly persistent circumvention. This report treats it as provider-reported operational evidence, not an independently reproduced effect. [F02]

OpenAI's September 30 report describes 16,000 patterned requests across more than 4,000 users during July 24 and 25, with a related cluster exceeding 15,000 users. It describes disruption, signup/infrastructure controls, account and network enforcement, reasoning protection and streaming holds. Count these as reported attempts and affected identities. The disclosure does not show that every request extracted hidden reasoning, that every user belonged to the named core actor, or that a trained student gained capability. Those distinctions matter when defining TRACE's campaign metrics. [F04]

Google's February 12 threat report describes more than 100,000 reasoning-coercion prompts and says Google recognized and reduced the attack. It also mentions measures intended to degrade student usefulness. No public mechanism or downstream capability denominator in that account identifies a particular poisoning method, MMD detector or success rate. A provider's undisclosed control should remain undisclosed in our analysis, rather than become evidence for a guessed implementation. [F05]

The September Anthropic report includes alleged use of stolen credentials, purchased transcripts, residential proxies, virtual cards and replay of genuine user sessions. One attributed campaign reportedly used over 1,500 accounts for more than 400,000 requests. This is especially important for threat modeling: a useful conversation can become training data after it was produced for an apparently normal session. A classifier that detects explicit "reveal your reasoning" instructions covers a different observable action from that passive acquisition path. Attribution and downstream use remain provider-reported. [F03]

### Internal monitoring economics do not transfer automatically to a wrapper

Constitutional Classifiers++ describes a cheap internal first stage and stronger classification for escalated exchanges. Its reported harmless-query refusal and compute overhead figures concern jailbreak safeguards on specified model populations, not distillation. It supports the engineering principle of selective expensive checks. It does not support copying a 1% overhead claim to a wrapper that must pay for a separate API judge and has no activations or shared cache. [F06]

Anthropic's October self-assessment work similarly supports asynchronous review rather than replacing the admission policy with a model's self-opinion. It describes cached-conversation assessment in production and reports useful discovery coverage, but its labels partly derive from a prompted teacher. On its inline cyber experiment the self-assessment intervention remains weaker than a standalone classifier. Teacher agreement, human-validated recall and capability restriction are separate endpoints. A TRACE judge should be evaluated against independent labels and an audit sample of traffic that never reached the expensive stage. [F07]

Current official interfaces also show why hidden reasoning is a separate channel. Claude's preserved-thinking documentation describes context checks and producing/linked-account constraints, with drop or rejection behaviors depending on mismatch and headers. OpenAI supports encrypted reasoning replay without publishing every cryptographic detail. Neither document permits us to infer a universal global key or an unrestricted account transfer model. A 2026 reasoning-replay preprint found cross-session/model issues in an earlier API snapshot; its current exploit success cannot be assumed after changes. Pin interface versions and measure semantic continuity rather than counting HTTP success alone. [F08] [F09] [F14]

### Operational reports and anonymous complaints require different uses

Official identity-verification documentation establishes that selected access can require government ID, selfie or business information, with retries and review. This raises acquisition cost and creates sensitive-data and accessibility burdens; it does not prove one account per human or immunity to stolen credentials. Official rate-limit documentation establishes organization/project token and request controls and trust-dependent allowance. Capacity limits are useful mechanisms, but their numbers were not chosen to bound student learning. [F15] [F12] [F13]

Anthropic's H1 transparency counts include 11.4 million bans, 398,000 appeals and 42,000 overturns. The approximately 10.6% overturn fraction among appeals is neither a distillation false-positive rate nor the false-ban rate over all customers. Appeal selection and non-appealing users prevent that interpretation. The appeals page also warns of delays without guaranteeing a resolution deadline. A deployed defense therefore needs its own request, account and organization error denominators and recovery measurements. [F16] [F17]

The included anonymous Reddit report alleges account restrictions and a ban around identity verification. It is unverified, includes AI-generated policy interpretation, and has uncertain exact dating. It is a lead for usability tests and policy clarity, not evidence that a particular provider falsely banned a particular person. The personal-capacity antidistillation essay is likewise useful for framing incentives and locating researchers, but advocacy is not an effectiveness evaluation. Both can be cited as the existence of a discussion, with explicit labels; neither should support a scientific performance claim. [F19] [F18]

The practical lesson is a division of labor. Provider reports identify operational threat shapes and deployed components. Peer-reviewed papers establish scoped mechanisms and prior art. Preprints propose new methods and supply current counterexamples. Community accounts suggest failures worth measuring. Combining these evidence types is useful only if their different authority remains visible.

## The threat model determines what a defense can control

### Distillation is several different security problems

Answer-only behavior transfer teaches a student from labels, code, explanations or demonstrations. Rationale extraction seeks richer intermediate supervision, including protected reasoning that the product did not intend to expose. Numerical extraction uses probabilities or other output behavior to recover model structure. Provenance asks whether suspect outputs or weights absorbed a provider's material. These are related but different tasks, with different required access and success criteria. A watermark-positive student can still be highly capable. A reasoning blocker can still serve final answers that train a useful student.

Rationales can improve data efficiency, as Distilling Step-by-Step demonstrated on its NLP tasks. DeepSeek's official R1 repository describes distillation into multiple student models using a curated 800,000-example set. Earlier imitation research found style and preference gains without broad factual transfer in its 2023 setting. These sources together justify measuring capability by task and base student, rather than treating either "all imitation works" or "imitation only copies style" as a timeless rule. [O16] [O17] [D11]

Richer numerical access needs separate review. Stealing Part of a Production Language Model demonstrates structural extraction through numerical API behavior, a different endpoint from teaching a small student task skills. Removing optional logprobs can close an unnecessary channel, but hard-label and text-only extraction remain possible. Grey-box work with a known encoder and private linear head shows a particularly favorable setting for in-distribution extraction; it does not prove that a whole unknown transformer can be recovered as easily. [O19] [D10]

| Actor and access | What the defender observes | Feasible controls | Residual problem |
|---|---|---|---|
| One opaque-API account | Authenticated requests, text, local timing and released output | Local budgets, concurrency limits, conditional features, review | Same-distribution traffic and learning below the budget |
| Many cheap accounts | Per-account events plus uncertain local links | Verified-principal entitlements and cautious campaign grouping | Unlinkable identities scale total allowance |
| Genuine-session replay or purchased logs | Apparently legitimate exchanges or no fresh access | Contractual controls, provenance, aggregate economics | Content can be indistinguishable from legitimate use |
| Reasoning-extraction operator | Requests that target protected channels and context transfer | Reasoning minimization, binding, classifiers, streaming holds | Final-answer and alternative-provider supervision |
| Provider with sampler/logits | Full token distributions, serving context and identity | Protected reasoning, sampler marks, experimental ADS | Utility trade-offs, cleanup and adaptive training |
| Provider with weights/training | Activations, training objectives, inference instrumentation | Internal probes and model-side learning objectives | No public universal utility-preserving antidistillation result |
| Suspect-model investigator | Weights, logits, controlled continuations or chat, depending on case | Registered watermark/fingerprint tests | Access limits, spoofing, mixing and false attribution |

### Identifiability and Sybil limits should be explicit

For a detector observing transcripts and metadata, let P_a and P_b denote the attacker and benign distributions over those observables. The difference between its rejection probabilities is bounded by their total variation distance. If the distributions are identical, its true-positive rate equals its false-positive rate. This follows directly from the definition of total variation, not from a new theorem introduced here. It does not claim that every real attack is indistinguishable. It tells us that adding classifiers cannot manufacture evidence absent from the observations.

Metadata expands the observations, so it can improve separation. It also creates error modes. A university NAT, a standard public benchmark and similar nightly schedules can link innocent users. A campaign graph must keep edge evidence and uncertainty, measure false merges and false splits, and avoid automatic transitive bans through weak edges. Shared billing ownership is stronger than a common IP, but even organization ownership does not imply that every member has the same intent. A whole-organization entitlement is an explicit policy choice; a whole-organization accusation requires stronger evidence.

Per-account warm-up has a direct Sybil vulnerability. PRADA explicitly acknowledges enough clients can bypass its single-client analysis. A 2026 distributed-attack preprint shows a 2,500-query MNIST attack partitioned among 400 identities below a 50-query warm-up, with zero recall in that setting. Its global variant improves recall but has very low precision under heavy benign mixing. Those results do not set LLM effect sizes, but they expose the same structural accounting problem. Douceur's classic Sybil work supplies the identity-assurance context, not a quantitative theorem about paid AI accounts. [D03] [D05] [D22]

An illustrative identity budget is simple. If each independently usable identity can obtain B units before restriction, k unlinkable identities can obtain roughly kB units, plus any concurrency overshoot. Rate limits stretch time but do not cap lifetime acquisition. Challenges consume attacker resources but can be outsourced and burden accessibility. Verification makes identities scarcer, with privacy and exclusion costs. None should be presented as a free solution.

A reasonable objective is therefore conditional and economic. For a declared attacker model, minimize the achievable student gain at a fixed acquisition spend and deadline, subject to a constraint on legitimate correct task completion. Report the residual risk from cheaper identities, outside data and longer deadlines. This is more defensible than an unqualified "prevents distillation" claim.

## Detection should contribute evidence rather than impersonate intent

### Match the null to the user whose access is at stake

MMD tests differences between distributions after a chosen feature map. A characteristic kernel can distinguish embedding distributions in population under its assumptions; it cannot restore distinctions erased by truncation or the encoder, and it cannot label a difference as malicious intent. TRACE's audit showed the consequences of comparing a specialist account with a pooled average user. Conditioning on domain and client profile is a plausible repair, but it changes what the detector can detect. A domain-conditioned score deliberately removes mixture differences, so broad collection needs a separate coverage or entitlement measure. [D14]

Build benign references at the decision unit. A specialist's account horizon needs plausible specialist sessions, including topic order, repetitions, long documents and iterative refinements. A shared enterprise key needs multi-user traffic. Batch evaluation and nighttime automation need their own declared use cases. Fit representations, reference windows, empirical score nulls and policy thresholds on separate source roles. Every change to encoder, bandwidth, tail, window size, deduplication or fusion creates a new statistic requiring calibration. A larger score pool does not cure a wrong population.

For sparse or novel strata, abstain from a calibrated false-positive claim. Fall back to an explicit budget and review rather than borrowing a pooled null and silently calling it conditional. Keep an "unknown client profile" group in evaluation. User-declared profiles can be gamed, so entitlements should reflect verification and contract as well as declarations. Choosing the most favorable null after seeing a suspicious score is itself selection and must enter calibration.

The MMD paper gives a useful baseline; PRADA supplies distance-history features; SEAT learns similarity features resistant to query blinding; feature-space coverage work already combines suspicious-client nomination with verification. Deep-kernel and classifier two-sample methods offer representation selection and separation diagnostics. None makes the generic combination of MMD, novelty and a classifier an original algorithm. Image-domain results require a fresh text study, especially where legitimate specialists and active learners resemble the attacker. [D02] [D03] [D04] [D06] [D15] [D16]

### Build a small feature set with independent purposes

Use four feature families initially. Acquisition features count admitted attempts, delivered responses, released output tokens, distinct canonical prompts, concurrency and budget velocity. Session features describe length, repetition, batch structure and domain transitions. Semantic features measure conditional shift, neighborhood novelty and cumulative coverage. Integrity features capture explicit reasoning-extraction instructions, context-binding misuse and strong account-link evidence. The policy should expose which family caused an action. Ten correlated semantic scores are not ten independent witnesses.

Coverage deserves a longer memory than a single 100-request window. Topic reordering can defeat an instantaneous mixture score while cumulative covered regions still grow. That makes coverage a candidate budget weight or review signal. It does not identify malice: a benchmark maintainer or broad researcher can cover the same regions. Keep raw token/request accounting beside any novelty-weighted quantity so that changes in embedding geometry cannot silently change the hard acquisition bound. Discounting duplicates also needs care because repeated stochastic answers can supply valuable diverse supervision.

A learned exchange classifier is most credible for explicit prohibited actions with labeled examples, such as requests to recover protected reasoning. It is weaker as a detector of unspoken future training intent. Evaluate paraphrases, languages, legitimate debugging of reasoning interfaces and real normal sessions. Preserve a random sample of first-stage negatives for independent labeling. Otherwise apparent second-stage precision can coexist with a large unmeasured blind spot in what the cheap first stage never forwards.

Start in shadow mode. Measure incremental campaign recall at fixed legitimate account restriction and task utility, rather than selecting a threshold that produces an attractive request-level ROC. An alert that arrives after the attacker has enough examples has little preventive value. A false intervention early in a specialist's only useful session can be costly even when the overall request rate appears tiny.

### Sequential methods help only after validity is justified

TRACE already has `e_process` and `e_cusum` in `packages/harness/src/harness/accumulators.py`, and already restricts Ville-style thresholds to the e-process policy. "Add e-values" is not a new feature or research contribution. The open question is whether the incoming evidence has the conditional validity needed for the claimed guarantee. A fixed empirical rank against overlapping, shared-source windows is not automatically conditionally superuniform under all prior observations.

Under conditional superuniformity of p_t, the transformation e_t = eta times p_t to the power eta minus one, with 0 < eta < 1, has conditional expectation at most one. Its sequential product supports a Ville crossing bound. If this premise fails, renaming the accumulator does not repair it. Nonoverlapping windows remove one source of overlap, but shared source documents, latent user topics, adaptation and drift remain. General time-uniform supermartingale theory establishes guarantees under stated assumptions, not for arbitrary empirical scores. [D17]

A fixed convex mixture of valid e-processes remains valid even when sensors are dependent. Arbitrary multiplication of detector outputs does not. Taking a maximum across tails, windows, domains and encoders needs error allocation or calibration. Learned fusion must be selected on development data and evaluated as a whole on locked data. Reusing the final test set to choose a mixture weight invalidates the final comparison even if every individual sensor was familiar.

CUSUM-style resets address late-onset changes after a quiet period, but average run length and probability of ever falsely crossing are different guarantees. Resetting or flooring an e-process does not automatically preserve the unreset Ville guarantee. E-detectors and recent conformal-change methods are relevant prior art, including composite-null approaches, but their assumptions and computational requirements must match the account process. The newest 2026 theoretical sources here had relevant sections checked, not independently audited proofs. [D18] [D20] [D21]

For the practical TRACE arm, an empirical account-horizon guarantee is more honest initially. Freeze a horizon, score all policy decisions over it, and estimate worst-group restriction on held-out independent accounts. Report uncertainty and temporal drift explicitly. Conformal methods beyond exchangeability quantify some departures under assumptions; they do not certify arbitrarily pooled clients or post-selected campaign graphs. A formal sequential arm can be a separate research project with its filtration, source dependence and interventions stated precisely. [D19]

### Calibration must include the policy that acts on scores

Calibrate alerting, temporary spacing, challenges and hard restriction separately. A 5% window alert target can yield many account alerts over a long horizon. Policy latching, repeated tests and recovery rules matter. A content-only benchmark with four benign batches cannot certify a 1% account-horizon restriction target. Neither can 300 synthetic windows from a few duplicated sources.

Score fusion calibration should cover the complete frozen rule, including domain routing and missing-feature handling. If a model judge times out, the fallback must be explicit and included in utility tests. If graph linkage updates a principal's cumulative allowance, test uncertainty and retroactive accounting. If drift triggers recalibration, freeze a new version rather than rewriting the interpretation of old logs. Keep a time series of score and client-mixture changes, but do not retrain blindly on all recent traffic because campaigns can contaminate the reference.

The strongest immediate diagnostic is a separation study on development sources. Try source-disjoint classifier two-sample tests alongside MMD and simple volume features. If separation disappears after balancing corpus, topic and client behavior, stop investing in semantic architecture on that dataset. That negative result narrows the available signal and shifts effort toward acquisition budgets and identity. If separation remains, check whether it predicts withheld sources and delayed attacks before promoting it to enforcement.

## Output protection offers a separate, access-dependent research track

### Restrict unnecessary supervision before adding hidden errors

A product should release the information needed for its legitimate task. That supports answer-only or compact-explanation tiers where they meet customer needs, protection of private reasoning, and removal of unnecessary numerical channels. It does not justify silently destroying useful explanations or corrupting correct code. Measure solved tasks and customer need for explanations separately. A user can need an explanation for teaching, auditing or debugging, and the cost of denying it will not appear in final-answer accuracy alone.

DistillGuard supplies a useful scoped example. In its 2026 preprint, removing chain of thought changes a math student's score from 67.8% to 31.4% while coding is unaffected in the studied setup. The attacker is naive and unfiltered, the model families are limited, and the result does not establish the same trade-off for TRACE. It does establish why task-specific response channels deserve testing rather than a blanket belief that all explanations have equal marginal teaching value. [D12]

TRACE already has deterministic keyed deception shared across accounts for the same canonical request, modeled on existing watermarking and misinformation work. It begins at level 4, cannot reduce acquisition before escalation, and disallows coding deception. The frozen shared-gateway study uses refusal rather than this mode. Keep deceptive-output experiments in a labeled research arm with explicit utility measurement. Do not interpret existing code as evidence that poisoning actually restricted a student.

Prediction Poisoning and Adaptive Misinformation are important antecedents. The former targets probability-vector training in classifiers; the latter selectively returns incorrect outputs on detected OOD inputs. Their image-classification effects do not establish a utility-preserving free-text defense. If the attacker's examples are in distribution, the OOD gate's premise disappears. If a four-label API preserves every correct top label, those labels remain usable supervision. [O03] [D08]

### ADS has a concrete mechanism and concrete adaptive counterevidence

Antidistillation sampling adjusts the teacher's token distribution using a proxy estimate of downstream learning loss. It needs sampler access and two perturbed-proxy forward passes per generated token. The inspected small-model reasoning experiments demonstrate settings where a teacher remains useful while a student learns less. That is stronger mechanistic evidence than generic output noise, but it is not an opaque-API configuration switch or a frontier deployment result. [O01]

The direct reevaluation matters as much as the proposal. Libon and colleagues fix a GSM8K pool and test cleanup, larger online budgets and interface changes. Free repetition removal recovers substantial performance, doubled query budgets recover much of the gain for low perturbation strengths, and a permitted one-token assistant prefill improves the attack. Its matched-utility-budget curve is partly interpolated, not a fully measured independent sweep. The lesson is to state query, data and interface budgets separately and to let the attacker clean data before declaring poisoning successful. [O02]

ADS-C preserves top-1 classifier predictions while changing probabilities. Its abstract reports substantial soft-target student degradation on image tasks and explicitly acknowledges hard-label fallback. It cannot protect TRACE's AG News label channel by modifying unexposed probabilities while leaving the label intact. This source was abstract-verified, so the register does not present its detailed experiments as fully audited. The method belongs in a probability-vector access branch, not a general recommendation for all generative endpoints. [O05]

Any output intervention should be compared with simply returning fewer tokens or fewer responses at the same legitimate task utility and cost. If a shorter answer or quota produces the same student reduction, the more complicated sampler has not yet shown incremental value. If corruption forces retries, charge the attacker for those calls and charge the defender for extra inference and denied useful tasks. If the attacker uses an external verifier or teacher, include its cost and data access in the declared model rather than excluding it after results look unfavorable.

### Provenance is useful after release, with its own failure modes

Watermarking has several endpoints. SynthID-Text marks generated text and has production quality evidence across nearly 20 million Gemini responses. Radioactive watermarking tests whether subsequent models learned from marked text. ADFP optimizes absorption by a proxy student. DAWN places a keyed signature through a small changed-label subset. These are different mechanisms and access models; none turns a positive provenance result into a blocked acquisition event. [O10] [O07] [O06] [O09]

ADFP's closed-weight setting still requires controlled continuation access for supplied contexts, which can be stronger than ordinary chat access to a suspect service. Its prompt reuse limits a fully prompt-disjoint attribution interpretation. GINSW also evaluates suspect-model identification rather than capability suppression. PostMark can operate after generation without teacher logits, but inserts text and requires another model operation. Exact class labels, short answer spans and executable code offer little freedom for such rewriting. These are access and product-fit constraints, not reasons to reject all watermark research. [O06] [O08] [O11]

Adaptive tests are mandatory. Watermark Stealing demonstrates both spoofing and removal for tested schemes, making false accusation a separate risk. Paraphrase studies show scheme- and token-count-dependent robustness; a result requiring hundreds of tokens says little about a one-token label. WASH analyzes model-distribution averaging under stated perturbation assumptions and supplies another mixing adaptation. Its theory does not prove that every watermark is removable through every text-only API. [O12] [O13] [O14] [O15]

Use separate secret keys and decision thresholds for text detection and suspect-model attribution. Register negative controls before probing, count the number of candidate models and keys searched, and include independent clean models, clean fine-tunes, mixed-source students and attempted spoofing. Preserve the access conditions needed to reproduce the test. A legal or contractual ownership claim needs more than an impressive p-value on a conveniently chosen suspect model.

### Ranked method comparison

The ordering below is a practical build priority for TRACE, not a measured effectiveness ranking. "Expected benefit" means a testable design hypothesis unless the row explicitly identifies inherited results. Cost descriptions depend on implementation and traffic.

| Priority and method | Signal and required access | Expected benefit and evidence | Cost and utility risk | Evasion and what it does not solve |
|---|---|---|---|---|
| 1. Durable request plus output-token entitlements | Local identity, admission and release events; opaque wrapper sufficient | Predictable acquisition bounds for linked principals; request-quota reduction is measured in the frozen audit | Transactional state and clear product limits; high-volume specialists need verified allowances | Account rotation, outside logs and learning below the cap; tokens are not information |
| 2. Verified organization budgets and concurrency caps | First-party identity/billing links or explicit organization membership | Raises rotation cost and bounds in-flight exposure; provider operational precedent | Verification, privacy, support and access burden | Stolen identities, resellers and uncertain links; cannot prove intent |
| 3. Protected reasoning and minimum useful output tiers | Wrapper can hide channels it exposes; provider controls binding | Reduces unnecessary rationale release; provider and scoped task evidence | Explanation utility and context compatibility | Final-answer/code transfer and alternate channels |
| 4. Conditional MMD plus cumulative coverage in shadow mode | Prompt embeddings, source-aware calibration, account history | Could recover specific shifts while exposing broad collection; no TRACE gain measured | Encoder/kernel cost and specialist calibration | Same-distribution extraction, topic matching and corpus leakage |
| 5. Explicit extraction/misuse classifiers with selective review | Exchange text; internal probes require provider access | Targets visible prohibited actions; provider-reported practice | Separate API judging can be expensive; missed first-stage traffic | Passive reuse, paraphrases and label ambiguity |
| 6. Campaign evidence and human review | Strong links, timing, rare patterns and uncertainty | Aggregates split acquisition when attribution is reliable | Storage, analyst work and false-merge collateral cost | Unlinkable identities, shared NAT and purchased sessions |
| 7. Sequential calibrated fusion | Valid conditional evidence or empirical horizon calibration | Combines timing and sensors with an explicit error target | Calibration data and assumptions dominate compute | Invalid nulls, selection and drift; already partly implemented |
| 8. Sampler watermark or absorption fingerprint | Sampler access, sometimes proxy weights and suspect continuations | Provenance under declared attacks; scoped research and production text marking | Key management and task-dependent output impact | Does not prevent learning; cleanup, spoofing and mixing |
| 9. ADS or other output perturbation | Teacher sampler plus proxy computation | Promising small-model reasoning mechanism with adaptive counterevidence | Multiple passes, cache memory and quality trade-off | Cleanup, budget expansion, prefill and different students |
| 10. Teacher training against transfer | Weight/training access and extensive validation | Research hypothesis beyond this review's demonstrated scope | Large training expense and possible legitimate skill loss | Public answers still teach; no general guarantee found |

An ensemble should combine mechanisms with distinct jobs. Entitlements enforce a measurable bound, reasoning protection closes a channel, conditional scores nominate anomalies and campaign analysis tests identity relationships. Combining all anomaly detectors into one opaque risk score hides those distinctions. It also makes it hard to discover that the apparent gain came entirely from a request quota.

## An evolutionary TRACE architecture can make acquisition auditable

### Define component contracts before selecting more detectors

The next TRACE should have a durable principal and entitlement record, an admission ledger, a policy evaluator, a metered teacher adapter and a completion log. An immutable policy revision should bind every decision. Keep raw observation, derived feature, policy evidence and enforcement action in separate event fields. That permits replay of a new scoring policy without pretending it was deployed during the historical request.

The principal resolver returns a local account, verified organization if present, entitlement tier and evidence-backed campaign memberships with versioned confidence. The canonicalizer returns a stable request digest covering domain, messages, tools, attachments, relevant generation parameters and output limits. The admission service consumes those objects and atomically reserves all applicable scopes. The scoring service returns typed evidence, calibration version and an abstention state. The policy evaluator returns allow, space, challenge, hold, reduce tier or refuse with a reason. The teacher adapter returns stream events and authoritative usage where available. The settlement service records release and reconciliation without rerunning admission.

This is a proposed interface design. Do not invent a current `reserve_tokens` CLI or claim the repository already implements it. The existing entry points in `apps/harness-api/src/harness_api/app.py` and `packages/harness/src/harness/policy.py` are the places to separate these contracts. Keep the experimental fake teacher so failure scenarios can run without touching the lab GPU or upstream provider.

### Reserve before inference and settle every released prefix

Independent counters in multiple workers cannot enforce a shared hard budget safely. Use one transactional store or an equivalent linearizable admission mechanism. Check remaining entitlement and reserve maximum output allowance in the same transaction. Reserve request count, token allowance and concurrency slot together, including organization or confidently linked campaign scopes. If any required scope cannot reserve, reject before starting the teacher.

A reservation records request binding, idempotency key, principal, horizon, policy revision, reserved amount and lifecycle state. A retry with the same key and same request returns or resumes the same result; the same key with a different request conflicts. A reconnect must not trigger a fresh stochastic generation for free. Proof submissions should redeem the bound challenge once without charging another successful supervision example. They still count as load and can have a separate abuse limit.

For streaming, reserve the maximum declared output cap or issue bounded incremental grants before emitting each chunk. Meter released prefixes locally. Upstream reasoning or hidden-token billing may differ from visible tokens, so keep provider cost and delivered supervision separate. On cancellation or connection loss, charge the emitted prefix conservatively; do not refund all usage merely because the client abandoned the final response. A successful socket write does not prove receipt, so the guarantee should be stated as bytes or tokens the gateway made available, with known uncertainty.

If the process crashes after release but before settlement, a durable reservation must survive. Reconciliation can recover usage from durable stream counters or conservatively retain the reserved amount until investigation. Expiring a lease must not automatically grant the same entitlement again while the old worker can still emit output. Fencing tokens, monotonic reservation versions or an equivalent serving check are needed. A daily horizon rollover should not erase outstanding reservations. These are ordinary distributed-systems requirements with direct security consequences.

A transparent bound helps. If each worker can emit at most g tokens after its last durable grant and n grants can be in flight, unrecorded exposure is at most n times g under the stated failure model. Add any already committed reservation to the horizon total. The exact bound depends on implementation and must be tested; it is not a capability bound. Similar reasoning shows asynchronous detection lag releases roughly R times L additional acquisition units at campaign rate R and end-to-end analysis delay L, plus reserved in-flight work. Report that overshoot rather than treating detection time as instantaneous enforcement.

### Persist evidence without turning the service into a surveillance archive

Default records should contain policy counters, timestamps, output usage, keyed prompt digests, coarse domains and necessary calibration metadata. Retain full exchanges only for a defined investigation purpose, short retention and restricted access. A plain prompt hash can be matched against common public prompts; keyed hashing reduces that exposure but still links behavior. Embeddings also retain semantic information and are not anonymized simply because the original text was discarded.

A campaign edge should store its type, evidence, creation time, expiry, confidence and reviewer disposition. Strong first-party ownership can support a shared contractual budget. Weak semantic or network edges should nominate review, not propagate automatic denial by transitive closure. Test splitting a false merge and restoring budgets without deleting the audit trail. Separate permissions for abuse review, dataset export and ownership attribution so a detector investigation does not silently become a training corpus.

The user-facing decision should identify an understandable category such as output budget exhausted, temporary verification needed or protected reasoning unavailable. Avoid exposing exact detector internals that make evasion trivial, but provide enough information for a legitimate user to correct a mistake. Measure time to review and successful restoration. An unexplained challenge loop can destroy utility without appearing as a hard refusal in current logs.

### Keep the critical path small and measure tail latency

The synchronous path should authenticate, canonicalize, reserve, run cheap entitlement and integrity checks, and invoke the teacher. Expensive semantic scoring can run at window completion or asynchronously, but its intervention delay must enter the exposure metric. A small random audit sample should bypass the normal nomination filter for review, with appropriate privacy controls. Never assume the expensive stage sees a representative sample merely because it has high precision on nominated traffic.

For a wrapper, let C_enc be average embedding cost, C_ledger reservation plus settlement cost, and C_judge the extra-model cost for a fraction q of requests. Added average cost is approximately C_ledger + C_enc + q times C_judge, plus storage and review. End-to-end latency is not simply that sum if work overlaps, but held responses pay the critical dependency chain. Measure admission p50/p95/p99, time to first token, completed-response latency, token throughput and correct tasks per dollar. A proposed initial target is no more than 10 ms extra p95 admission latency for cheap accounting and no more than 5% end-to-end overhead for the selected wrapper policy. These are engineering targets, not results.

Dense MMD grows quadratically with window size in the kernel terms. A naive comparison against k reference windows of size w has roughly O(k times w squared times d) kernel work for embedding width d, though cached self terms and vectorization can reduce it. At float32, one 384-dimensional embedding is 1,536 bytes before index overhead; a million is about 1.54 GB. Bounded sketches or prototypes can help, but every approximation needs numerical and detection-power checks. Large completion-window latency can disappear in an average-per-request number.

ADS has different costs. One teacher pass plus two 3B proxy passes around a 7B dense teacher gives an illustrative parameter-work ratio of 13 divided by 7, about 1.86, before caching, batching and memory effects. It is not a wall-time estimate for a frontier mixture-of-experts model. Proxy caches and weights consume memory. Post-hoc rewriting needs another generation and often a hold before release. Sampler watermarking can be cheaper because it need not invoke another full model. Measure each on the intended serving stack instead of borrowing one paper's overhead.

### Map the work to existing code without changing the frozen study

| Existing location | Current role in the inspected snapshot | Proposed isolated task | Validation boundary |
|---|---|---|---|
| `apps/harness-api/src/harness_api/app.py` | Authentication, admission, proofs, teacher call and admission logging | Separate reserve, release and completion events; durable idempotency | Fake-teacher retries, streaming cancellation and crash injection |
| `packages/harness/src/harness/policy.py` | In-memory account state, level actions and caps | Persist principal horizons and reservation-aware actions behind a store interface | Multiple workers cannot overspend or refresh risk after restart |
| `packages/harness/src/harness/calibration.py` | References, empirical ranks, thresholds and mixture warning | Source/client-conditional calibration with explicit abstention and versioning | New artifact and source-disjoint policy-horizon evaluation |
| `packages/harness/src/harness/accumulators.py` | Leaky, e-process and e-CUSUM options | Keep baselines; document exact validity assumptions of any new fusion | Conditional-null checks or clearly empirical guarantees |
| `packages/harness/src/harness/monitor.py` | Aggregate alerts without account blame | Link aggregate exposure to evidence-backed campaign review | False merges, shared NAT and unknown-link campaigns |
| `packages/harness/src/harness/encoder.py` and `mmd.py` | Embeddings and kernel scoring | Benchmark bounded history or representations only after separation diagnostics | Cost, truncation, power and reproducibility |
| `packages/harness/src/harness/deception.py` | Keyed research intervention | Preserve as a separate output arm; do not deploy hidden errors casually | Matched utility, adaptive cleanup and student transfer |
| `legacy/packages/distillation/src/distillation/gateway_study.py` | Historical shared-gateway orchestration | Use as historical protocol evidence, not a live-current claim | Preserve frozen revisions and artifacts |
| `tests/test_harness.py`, `tests/test_harness_api.py`, `tests/test_gateway.py` | Existing policy/API checks | Add only meaningful accounting and changed-behavior tests during implementation | Do not expand this research task into production changes |
| `docs/harness.md` and `docs/evaluation.md` | Harness and evaluation documentation | Document a future revision's contracts, provenance and measured outcomes | Separate proposed design from qualified measurements |

## A provider rebuild adds inference controls and organization operations

### Separate synchronous release from asynchronous investigation

A provider-owned design places an account/organization service and durable entitlement ledger before its inference router. The synchronous release policy sees verified ownership, allowed output channels, concurrency and a compact risk summary. It enforces a budget even when the analysis system is unavailable. The model-serving tier controls reasoning exposure and validates protected-context bindings. The output gateway meters each released chunk. An asynchronous event stream feeds behavioral analysis, investigation and cross-account evidence. A separate enforcement service issues versioned, time-bounded actions and recovery decisions.

The architecture should keep detector evidence distinct from account status. A newly discovered link can reduce a shared campaign allowance without pretending all linked accounts were historically proven malicious. An investigator can see why an edge exists and whether it came from a first-party relationship or a probabilistic match. A high-confidence protected-reasoning exploit can justify a stronger immediate action than broad semantic coverage. Every action needs a reason, expiry or review point, and a record of the policy version.

Model probes can reduce synchronous classifier cost when serving internals allow them, but a probe should be trained and audited for the actual distillation threat. A jailbreak detector is not automatically a distillation detector. Model self-assessment can prioritize retrospective review using cached context; it should not be the sole arbiter of its own blind spots. If analysis is delayed by minutes or hours, entitlements determine how much can leak before the investigation catches up.

A provider can also test stronger identity and reseller controls, contractually distinct research/batch access, and intelligence sharing with other providers. Those require legal, commercial and privacy decisions beyond TRACE. The research system should model them as explicit assumptions and costs, not claim access to proprietary payment or device signals it cannot collect. Avoid making a scientific comparison depend on an oracle that perfectly links every attacker identity but never merges a benign organization.

### The optional inference branch has distinct interfaces

The reasoning-protection component should bind context to permitted model/account transitions, validate preceding messages and tools, and define behavior on mismatch. Compatibility tests must check whether the model still has necessary context, because silently dropped state can harm legitimate multi-turn tasks. Key rotation and legacy handling belong in that protocol. Logs should distinguish a rejected transfer, dropped protected item and successfully preserved context.

The sampler branch can evaluate text watermarking, ADFP or ADS when its required proxy and suspect access are available. Keep each intervention behind a policy version and test quality before release. Do not expose internal logits or auxiliary outputs merely to support a research method if the product does not need them. Numerical channels can create additional extraction risk. A training-objective branch would require a separate research program on the trade-off between useful knowledge and transferability, with no universal guarantee established by the reviewed public evidence.

An attribution service should operate after a suspected student appears, with registered keys, approved access, controls and a documented hypothesis test. This service can support investigation and contractual enforcement. Its results should not feed back as if the original acquisition had been prevented. Mixing acquisition and attribution metrics encourages misleading security claims and can hide a highly capable stolen model behind a successful watermark detection.

### Deployment needs drift controls and a way to recover

Start with shadow observation on representative legitimate clients. Freeze policy revisions and compare against the existing allow/quota/rate baseline. Introduce hard controls first where their semantics are clear, such as explicit entitlement exhaustion. Use a small canary for uncertain evidence-triggered spacing or review, and preserve a holdout audit stream. Monitor per-client correct-task completion, challenge abandonment, review backlog and restoration time alongside attack indicators.

A rollback should stop a bad restriction policy without erasing durable usage or regenerating old samples. Keep prior calibration artifacts and policy versions so replay remains possible. On drift, distinguish a changed client population, upstream model update, encoder change and a campaign. A model update can change answer length and token costs even if prompt statistics are unchanged. Freeze a new study when output channels or account semantics change materially.

Set operational alert targets based on reviewer capacity. If M accounts are evaluated and the false-alert probability per horizon is alpha, expected false alerts are roughly M times alpha before dependencies and triage selection. Even a 1% alert rate can be unaffordable at provider scale. Hard-denial error targets should be stricter than review alerts and reported at account/organization level, with uncertainty. Real abuse prevalence also matters for positive predictive value; a balanced benchmark cannot supply it.

No source in this review publishes a complete public design that simultaneously demonstrates adaptive ordinary-answer extraction control, specialist-safe false restrictions, reliable campaign linkage, low latency and reduced downstream student capability. That gap is an opportunity for a careful study, not evidence that an unpublished frontier solution cannot exist.

## A defensible contribution needs a sharper claim than an ensemble

### The strongest candidate is utility-constrained acquisition control

The proposed capstone question is: under a declared account and compute budget, how much must an attacker spend to recover a specified fraction of the teacher's task advantage while legitimate specialists retain their useful service? The mechanism would allocate request, output and optional explanation allowances by verified use case, maintain the limits through concurrent streaming and restarts, and use uncertain content or linkage evidence to request verification rather than automatically infer theft. The scientific contribution would need a specified optimization or policy construction and evidence that it improves this objective over strong matched baselines.

This is a candidate contribution, not a novelty certification. Active learning and extraction query complexity already have theoretical and empirical antecedents. Knockoff Nets and ActiveThief already study sample-efficient acquisition from natural public data. Coverage-based candidate selection and verification already exist. TRACE already has request quotas, token-matched controls and sequential options. A weighted token bucket, a graph of accounts or a lifecycle benchmark alone would be implementation of prior ideas. The reviewed classic acquisition papers were checked at abstract/proceedings level, so a selected thesis must audit their complete methods before claiming a new optimization. [D25] [D26] [D27] [D06]

One concrete formulation is to choose a policy pi that maximizes the minimum measured attacker spend needed to reach a frozen capability threshold, subject to a lower bound on legitimate correct-task completion in every prespecified client group. The attacker minimizes that spend over prompt selection, output requests, account splitting and allowed cleanup. Identity price, outside-data budget, deadline and offline training compute are explicit constraints. Student capability is empirically estimated from learning curves; tokens provide an enforcement unit, not an assumed sufficient statistic for information.

This formulation is attractive because a negative result is meaningful. If budgets that preserve legitimate specialists still allow cheap useful students, the project can quantify that trade-off instead of presenting a failed detector as a calibration accident. If conditional semantic features do not improve the trade-off, report that they add cost without demonstrated security value. The paper can be original through a carefully established empirical result or new construction, but neither outcome guarantees publication.

### Three candidate contributions and their falsifiers

| Candidate | Closest prior art | What could be new if demonstrated | What defeats the novelty or effectiveness claim |
|---|---|---|---|
| Utility-constrained, identity-aware acquisition allocation | Quotas, active-learning extraction, coverage defenses, TRACE token-matched controls and query/data/interface analysis | A specified policy with a measurable attacker-cost objective across heterogeneous specialists and uncertain identities, evaluated with fresh source families and actual students | Prior work already solves that objective; gains disappear at matched legitimate utility; policy reduces to a standard weighted quota |
| Restart-safe release guarantees tied to empirical capability envelopes | Durable reservations, idempotency, streaming limits and existing acquisition baselines | A precise release bound under a declared crash/concurrency model, paired with evidence showing when that bound is sufficient or insufficient for task transfer | Durable accounting alone is called a new ML defense; no capability experiment; an existing system already has the same guarantee |
| Adaptive composition of acquisition limits and output interventions | ADS, its reevaluation, ADFP, DAWN and misinformation | A factorial causal result showing whether acquisition limits constrain cleanup/retry/mixing enough to give an output method incremental value beyond fewer-output controls | Same interaction already studied; extra compute or retry budget erases gains; no sampler access; effects are only reduced sample count |

The first is the recommended main track, with the second as the systems foundation. The third is optional and resource-intensive. Its strongest version would answer a narrow unanswered interaction, not merely show that adding a quota to ADS releases less data. The lifecycle preprint is especially close prior art to any claim of a broad extraction-defense benchmark. Its existence means "evaluate defenses across attacks" is insufficient as a new research claim. [D01] [O02]

A separate formal-statistics thesis could study conditional calibration under account mixtures and adaptive campaign selection. That is demanding. It would need a defined stochastic model, a useful guarantee under that model and empirical tests of model mismatch. Repackaging existing e-processes or asserting anytime validity for pooled ranks would weaken the work. New 2026 composite-null and conformal change-detection papers narrow the available novelty further. [D20] [D21]

### Use citation categories honestly in the manuscript

Cite peer-reviewed sources for established mechanisms within their evaluated setting. Cite a preprint as a preprint and pin the inspected version. Some sources report venue status in arXiv metadata; that is different from checking final proceedings and should be stated accordingly. The June MMD v1 used in this online study and the later v3 acceptance metadata in the inherited source check should not be silently combined as if their experiments were identical. Preserve both provenance records and audit the chosen manuscript version.

Cite provider posts for what the provider reports doing and observing. Use wording such as "Anthropic reports" and retain the outcome unit, population and date. Do not convert harvested-reasoning reduction into student-capability reduction or self-reported false positives into a guarantee for TRACE. Official API documentation establishes interface behavior at retrieval time, not a permanent protocol. Practitioner essays can support a statement about incentives or research motivation. Anonymous content can motivate a usability hypothesis, with verification caveats, but cannot establish a performance or misconduct claim.

For TRACE's own results, cite the frozen revision, calibration hash, source-role manifest, workload contract and exact artifact. Separate completed live outcomes, interrupted logs, deterministic offline replay and proposed future arms. The old audit's inspected attack and benign sets are now development evidence. They cannot be relabeled untouched simply by moving them to a new directory. A manuscript should state that the new architecture has no measured gain until a new study produces one.

## Validation must measure capability, utility and acquisition together

### Freeze roles by source family before selecting policies

Use at least five roles. Representation-fitting sources train any learned feature map or classifier. Reference-fitting sources construct benign distributions. Score-calibration sources estimate null behavior. Policy-development sources choose thresholds, window sizes, budgets and output tiers. Final confirmatory sources remain locked until selection ends. A temporal-drift period is a further distinct evaluation, not an opportunity to tune the original final test. Near-duplicate screening precedes allocation so paraphrases or shared documents do not cross roles unnoticed.

Split SQuAD-like data by document/title, coding by repository or problem lineage, math by generator/problem family, and news by publisher, period and topic. Row shuffling is insufficient. Multiple accounts drawn from the same narrow source do not provide independent source evidence. Balance legitimate and attacker sources where possible so corpus membership alone cannot win the detector comparison. Also include an explicitly source-shifted condition to measure the kind of anomaly a distribution detector can detect well.

Reserve new data because the current audit already inspected the former final benign and attack sources. Use A1/A2 and the order-evasion probes for development regression and mechanism understanding, with the historical labels intact. Do not tune a domain-conditional ensemble on these probes and then report their improved detection as a confirmatory success. The frozen generator bug should be fixed only in a new workload version, accompanied by source/topic coverage checks and a fresh calibration.

Legitimate clients should include casual single-task users, narrow topic specialists, broad researchers, multilingual sessions, batch evaluation, repetitive translation or labeling, iterative coding, long-document QA, shared enterprise accounts and nighttime automation. Include stable long-term users and new accounts. Replay genuine sessions with consent or a defensible public license, without assigning every replayed session a stereotyped attacker prompt. Hold out organizations and source families, not just account names.

Campaigns should include the original broad extraction, topic-sequential extraction, independent sources, low-and-slow use, account splitting, account rotation, purchased/replayed genuine sessions, cover traffic and explanation-focused collection. Provide uncertain-link cases, false links from shared NAT and shared public datasets, and campaigns that cannot be linked from available signals. A graph policy with perfect attack membership is an oracle baseline, not the deployable method.

### Compare policies under matched budgets and deadlines

The minimum baseline panel is allow, current full MMD, instantaneous MMD, request quota, rate limit and request-plus-output-token quota. Add conditional MMD in alert mode and the selected policy only after development. If enough resources exist, add a small factorial output panel with no output intervention, compact-answer tier and one inference-access method. Avoid an unmanageable sweep over every detector and sampler combination. Selection consumes a development budget and increases comparison multiplicity.

Run two comparison views. The first fixes attacker query/output-token/money budget and wall-clock deadline, then measures acquisition and student outcomes. The second fixes legitimate correct-task utility or solved-task cost and asks which policy makes attack capability most expensive. Matching only query counts can hide longer responses; matching only output tokens can hide labels and verified solutions with different teaching value. Report both raw units and task-specific learning curves rather than inventing an unvalidated universal information score.

Teacher capacity must support the undefended legitimate workload within its deadlines before the defended utility test begins. The inherited shared-gateway benign arm had substantial timeouts, so it cannot estimate population false positives cleanly. Deliberate overload is a separate robustness experiment with its own controls. Account for spacing, challenges and held responses as real latency and abandonment costs. A request that succeeds after the user's deadline is not a successfully completed legitimate task.

Train students at acquisition checkpoints such as 100, 300, 1,000, 3,000, 6,000 and 12,000 delivered responses, reporting released tokens and distinct examples alongside each. Narrow that schedule after a pilot. Include the untrained base, gold-supervision control and undefended-teacher control. At selected final conditions, use at least two student families and three training seeds. The seeds measure optimization variability conditional on data; they are not independent campaigns or source populations.

Use objective task metrics where available: classification accuracy, execution-based coding success, math correctness and QA F1 or exact match. Include far-transfer tests and held-out problem lineages. A style preference judge is a supplementary measure, not proof of recovered capability. Freeze a capability target using development curves, for example half of the teacher-over-base task gain, and report cost/time to that target rather than selecting a flattering endpoint after seeing final results.

### Give the attacker realistic adaptation opportunities

The frozen adaptive set should include topic reordering, shuffled versus bursty timing, semantic paraphrase, duplicate removal, answer-only collection, few-shot batching, selective retries, verifier filtering, alternative-teacher mixing, additional unlabeled data and account rotation. For output defenses include free repetition cleanup, watermark removal, backtranslation and model averaging where access permits. Charge each extra online call and offline operation consistently. Keep data-pool budget distinct from teacher-query budget: repeating a fixed pool and acquiring new problems are different adaptations.

Test assistant prefill only if the actual API permits it; a user instruction containing a prefix is a different interface. Test tools, attachments, multi-turn state and protected reasoning replay only in allowed controlled conditions. If an intervention is defeated by an interface the deployed product does not expose, record that scope rather than declaring universal failure. If the product does expose it, excluding it because the defense loses is not a defensible evaluation. The ADS counterstudy provides the relevant methodological warning. [O02]

Include attacker awareness of budgets and approximate detector family. Security should not depend on keeping a request cap secret. For graph policies, let the attacker vary identity cost and strong-link availability. For specialist safety, let legitimate users have broad coverage and repeated prompts too. A defense tested only against a conspicuous scripted extractor and a casual benign user will overstate its ability to separate real use.

### Report the right denominators and uncertainty

For every account and campaign, report admitted attempts, completed responses, withheld responses, emitted prefixes, released tokens, distinct examples and time to each intervention level. Separate first alert from first reduced service and first hard refusal. Count all accounts and scored accounts separately so warm-up evasion is visible. Report campaign totals by deadline, including acquisition after a nominal detection and across identities. Refusal logs alone cannot reconstruct the student's training set.

For legitimate clients, report correct completed tasks, deadline completion, explanation availability where relevant, escalation and restriction by profile, and recovery burden. Include p50/p95/p99 latency, teacher and auxiliary compute, memory, ledger throughput, analyst review load and correct tasks per dollar. Do not report a single pooled benign average when a specialist group bears most errors. Shared organization restrictions also need an organization-level denominator.

Use paired comparisons by source family or campaign and cluster intervals at the actual independent unit. Do not bootstrap correlated windows as independent. A source-cluster bootstrap is useful when enough independent families exist; with few clusters show every campaign, uncertainty sensitivity and the limits of intervals. Training seeds enter a hierarchical analysis or a separate conditional-variability summary. Multiple detectors, checkpoints and tasks require prespecified primary hypotheses, adjusted inference or a clear exploratory label.

For a zero-event benign restriction estimate, the one-sided 95% upper bound is approximately 3 divided by the number of independent horizons. Roughly 300 independent horizons with no restrictions are needed to support an upper bound near 1%. Three hundred generated accounts sharing the same few templates do not create equivalent real-world assurance. Real traffic coverage, group diversity and independence matter as much as the nominal count. Rare hard-denial and attribution targets can require far more data than a capstone can collect.

### Acceptance gates are targets, not measured outcomes

| Gate | Proposed acceptance criterion | Stop or reject condition |
|---|---|---|
| Legitimate utility | No more than 1 percentage point loss in correct task completion in each priority client group and no more than 1% benign horizons restricted, with stated uncertainty | A specialist group fails even when the pooled average passes; representative data unavailable for the claim |
| Security economics | At least twice attacker spend or time to a preregistered capability target, with source-clustered uncertainty excluding no improvement | Benefit disappears under the strongest declared adaptive attack or relies on cheaper uncounted outside resources |
| Incremental mechanism | Selected detector/output method beats the strongest matched quota/token/rate baseline at matched legitimate utility | Student loss is fully explained by fewer examples; claim acquisition control only rather than poisoning or detector value |
| Accounting | Crashes, retries, cancellations, duplicate proofs, concurrency and horizon changes stay within the stated release bound | Restart refreshes entitlement, retry creates free samples or reconciliation refunds already emitted output |
| Sequential claims | Conditional validity or an explicitly empirical horizon evaluation supports the exact policy and selection rule | Marginal ranks are sold as anytime valid; resets or graph selection are omitted from the error analysis |
| Attribution | Clean and mixed-source controls meet a predeclared accusation-level error target under available suspect access | Fingerprint can be spoofed, prompt reuse is concealed or provenance is called prevention |
| Deployment cost | Measured overhead meets a predeclared latency/compute budget and review demand is sustainable | Auxiliary calls erase service economics or hard cases disappear into an unbounded review queue |

Stakeholders should choose the final utility and security thresholds before confirmatory runs. The numbers above are proposed starting points, not evidence that they are achievable or appropriate for a commercial service. Reject semantic restriction triggers if topic reordering removes all value or they cannot outperform budgets for specialists. Reject an output defense for deployment if cleanup, retries or permitted inputs erase its matched-utility advantage. Reject a universal prevention claim whenever the threat model still includes indistinguishable useful-answer collection.

## Resource planning should narrow the study before buying compute

### Start with an isolated, CPU-only design and accounting phase

A first phase can take one to three researcher days for source-role manifests, policy specification, existing-artifact replay and the accounting state machine. It does not require a teacher GPU. Keep old artifacts read-only. The preserved replay tools are under `readings/v4/method-checks`; their README describes portable launch requirements and cached embedding dependencies. Existing reproduction scripts validate historical measurements, not the new proposal.

The following commands inspect the available instructions and run existing focused tests in the repository's configured development environment. They do not create a new benchmark, start a server or contact a teacher. Use the documented environment first; the current shell may not expose `uv` directly. Do not install an ad hoc environment or launch the full test suite while an active shared experiment is unknown.

```bash
jj status
jj log -r @ -n 1
rg --files readings/v4/method-checks tests packages/harness apps/harness-api
cat readings/v4/method-checks/README.md
uv run pytest tests/test_harness.py tests/test_harness_api.py tests/test_gateway.py
```

Those are existing paths and test entry points, not commands for hypothetical features. New reservation, graph, student-sweep and conditional-calibration commands must be introduced by a later implementation and documented with its actual schema. The correct immediate deliverable is a frozen protocol and task map, not a fabricated one-line "run proposed defense" command.

An accounting prototype can take an estimated two to five development days using a fake teacher and deterministic failure injection, followed by a small reserved-compute pilot. Test simultaneous near-limit admission, worker death before and after emission, cancellation, duplicate proof redemption, idempotent reconnect, horizon rollover and state recovery. These are high-value checks because a detector cannot compensate for a quota that resets after a crash. This report has not implemented or run those tests.

### Pilot output and training cost before committing to a matrix

Six thousand examples with an average of 1,024 generated tokens require about 6.14 million teacher tokens per condition. At an illustrative aggregate 50 tokens per second, generation alone takes about 34 hours; at 200 tokens per second, about 8.5 hours. These are arithmetic scenarios, not measured throughput on the lab hardware. Long prompts, retries, holds, proxy passes and evaluation add work. Measure a 100-prompt pilot with realistic context lengths before approving the main sweep.

Four selected policies times two student families times three seeds produces 24 fits at one acquisition point. Four checkpoints produce 96 fits. If a measured fit takes 2 to 8 GPU-hours, the training matrix needs 192 to 768 GPU-hours before generation, development, evaluation and reruns. The wide interval is intentional: sequence lengths, LoRA configuration, student size and hardware dominate. Benchmark on separately reserved resources rather than promising that a particular 24 GB or 80 GB accelerator will fit every setting.

An initial capstone can reduce cost by separating broad acquisition replay from a narrow student panel. Evaluate many policy choices offline on development data, select a small set, then train students only for the strongest baselines and final candidate. Use early checkpoints to estimate whether a larger campaign is informative. Do not use early final-test student results to select the policy and still call later points confirmatory. If resource limits force one student family or few sources, state that limitation and narrow the conclusion.

A provider shadow study needs additional elapsed time for account horizons, drift and review recovery. One to three weeks of reserved training work and two or more weeks of observation are planning estimates, not deadlines or measured requirements. Human annotation, consented legitimate traffic and linkage-error evaluation may be the actual bottlenecks. An impressive GPU budget cannot substitute for independent benign organizations.

### Rank implementation by what each change requires

| Stage | Recommended work | Requires recalibration? | Requires a new frozen study? |
|---|---|---|---|
| Immediate research hygiene | Preserve provenance, distinguish admissions from release, correct manuscript claims, record warning-only status | No for documentation and event interpretation | No new efficacy claim from this alone |
| Isolated systems foundation | Durable admission ledger, idempotency, streaming settlement and restart-safe horizons | Not for the accounting invariant itself; policy limits still need selection | Yes before claiming improved security/utility |
| Workload repair | Fix topic truncation and represent specialists, automation and genuine sessions | Yes, references/nulls/thresholds tied to changed traffic | Yes; do not mutate the frozen experiment |
| Conditional detection | Client/domain references, abstention, cumulative coverage and frozen fusion | Yes for every changed score and complete policy | Yes with fresh source-disjoint confirmation |
| Identity/campaign controls | Verified-principal budgets, uncertainty-aware links, review and recovery | Empirical linkage and intervention calibration | Yes, including false merges/splits and rotation costs |
| Output-tier track | Compact answers, protected reasoning exposure and optional sampler experiments | Utility and transfer calibration; detector recalibration if features change | Yes with adaptive students and matched-output controls |
| Provider-only track | Activation probes, cryptographic context binding, ADS/ADFP and training changes | Method-specific; sampler/training changes affect distributions | Separate study with declared inference/suspect access |

The immediate scientific priority is the learning curve. The audit's qualified AG News development gain shows that hundreds of responses can matter, while some quota arms release more than the 500 responses used in that development measurement. It does not prove the same gain at every smaller budget, prompt subset or training seed. Establish where useful capability emerges before choosing a supposedly protective threshold. Otherwise the system can block most of a 6,000-response campaign after the attacker already collected enough.

## What remains unknown determines the honest claim

The public record does not establish a general utility-preserving way to stop an adaptive learner from useful ordinary answers. It also does not prove that all practical content signals are useless. The right next step is to find their incremental value under a realistic acquisition and utility objective. The strongest recommendation is a durable bounded-acquisition system with explicit identity assumptions, conditional evidence and protected unnecessary channels, followed by a fresh student evaluation.

Unknowns include the live lab state and revision, real legitimate traffic and profile prevalence, available identity signals, acceptable explanation loss, account acquisition costs, output-token teaching value by task, false-link rates, student learning curves and actual serving overhead. Proprietary frontier mechanisms are only partly disclosed. The newest preprints need independent reproduction, and several sources were verified only at abstract level. No architecture in this report should be called a deployed improvement, and no source count is an effectiveness result.

For an original project, choose the utility-constrained acquisition objective and a narrow mechanism that can be falsified. If conditional detectors fail, publish the measured boundary and concentrate on entitlements. If the policy works only with expensive verified identities, state that dependency. If a sampler method wins after realistic cleanup and matched utility, make that scoped interaction the claim. The resulting thesis can be rigorous and useful without promising to make learning from public model outputs impossible.

## Search scope and annotated source register

The research was organized into provider practice, statistical/behavioral detection and output protection/architecture. Searches and primary-source follow-up covered model extraction and distillation, MMD, PRADA, SEAT, semantic coverage, active learning, sequential calibration, e-processes, CUSUM, conditional validity, Sybil identities, reasoning-channel protection, account operations, ADS and adaptive reevaluation, watermarking, radioactive attribution and operational appeals. Sources were retrieved and checked by the research team on October 31, 2026. No active service, account, frozen artifact or GPU experiment was changed.

The register below deduplicates equivalent scholarly records and exact canonical URLs while retaining every research ID as an alias. Verification notes remain attached to each original record, because one researcher inspecting an abstract and another inspecting a full paper are different provenance statements. "Full text checked" means relevant sections were accessible and inspected, not that experiments or proofs were independently reproduced. The report synthesizes the evidence; external links require a network connection, while the document, navigation and source search work offline. The machine-readable companion retains original claim and limitation fields.

<!-- GENERATED SOURCES -->

The register contains **64 deduplicated sources from 67 original research records**. Categories are editorial evidence labels, not quality scores: 16 provider disclosure or official documentation; 15 preprint or mixed preprint record; 4 code, practitioner writing or other primary material; 1 anonymous, unverified account; 27 peer-reviewed or proceedings record; 1 submission with partial verification.

### F01, D23: Detecting and preventing distillation attacks

[Detecting and preventing distillation attacks](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks). Evidence category: Provider disclosure or official documentation.

F01. Anthropic. Date/version label: 2026-02-23. Type: provider threat report. Verification: Full relevant detection and account-network sections fetched. Limitation: Provider-selected cases; no independent confusion matrix.

D23. Anthropic. Date/version label: 2026-02-23. Type: official provider disclosure. Verification: full-text verified. Limitation: Provider self-report, no public ROC, control population or student outcome ablation..

### F02: Strengthening our defenses against distillation attacks

[Strengthening our defenses against distillation attacks](https://www.anthropic.com/news/distillation-defenses). Evidence category: Provider disclosure or official documentation.

F02. Anthropic. Date/version label: 2026-10-12. Type: provider production disclosure. Verification: Full substantive sections including limitations fetched. Limitation: Hidden-reasoning metric; no persistent sophisticated circumvention established; not generic answer-only prevention.

### F03: Detecting and countering misuse of AI: September2026

[Detecting and countering misuse of AI: September2026](https://www.anthropic.com/threat-intelligence-report-september-2026). Evidence category: Provider disclosure or official documentation.

F03. Anthropic. Date/version label: 2026-09-10. Type: provider threat report. Verification: Full relevant illicit-distillation and credential-abuse sections fetched. Limitation: Selected cases and provider attribution; downstream-training allegations not independently verified.

### F04: Disrupting a coordinated model-distillation campaign

[Disrupting a coordinated model-distillation campaign](https://openai.com/index/disrupting-a-coordinated-model-distillation-campaign/). Evidence category: Provider disclosure or official documentation.

F04. OpenAI. Date/version label: 2026-09-30. Type: provider threat report. Verification: Full relevant report and attempt-count footnote fetched. Limitation: Only core cluster attributed; no student-transfer evaluation.

### F05: GTIG AI Threat Tracker: Distillation, Experimentation, and (Continued) Integration of AI for Adversarial Use

[GTIG AI Threat Tracker: Distillation, Experimentation, and (Continued) Integration of AI for Adversarial Use](https://cloud.google.com/blog/topics/threat-intelligence/distillation-experimentation-integration-ai-adversarial-use). Evidence category: Provider disclosure or official documentation.

F05. Google Threat Intelligence Group. Date/version label: 2026-02-12. Type: provider threat report. Verification: Full model-extraction section fetched. Limitation: No mechanism or capability-reduction evaluation disclosed.

### F06: Next-generation Constitutional Classifiers: More efficient protection against universal jailbreaks

[Next-generation Constitutional Classifiers: More efficient protection against universal jailbreaks](https://www.anthropic.com/research/next-generation-constitutional-classifiers). Evidence category: Provider disclosure or official documentation.

F06. Anthropic. Date/version label: 2026-01-09. Type: provider research and production disclosure. Verification: Full architecture and results sections fetched. Limitation: Jailbreak/CBRN results; populations for costs and refusals differ; not distillation evaluation.

### F07: Scaling Misuse Detection with Self-Assessment

[Scaling Misuse Detection with Self-Assessment](https://safeguards.anthropic.com/2026/self-assessment/). Evidence category: Provider disclosure or official documentation.

F07. Yiming Zhang, Chen Xia, Andy Dau, John Wang, James Yu, Bobby Chen, Andreas Terzis, Nicholas Carlini, Erik Jones. Date/version label: 2026-10-16. Type: provider technical production blog. Verification: Full relevant methods/results/limitations/data-handling sections fetched. Limitation: Training/evaluation share teacher blind spots; cyber experiment is24 tasks; production is offline.

### F08: Preserved thinking

[Preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking). Evidence category: Provider disclosure or official documentation.

F08. Anthropic. Date/version label: None. Type: official API documentation. Verification: Full relevant prefix/account/model-binding sections fetched October31. Limitation: Mutable model-specific documentation; not generic extraction prevention.

### F09: Reasoning models

[Reasoning models](https://developers.openai.com/api/docs/guides/reasoning). Evidence category: Provider disclosure or official documentation.

F09. OpenAI. Date/version label: None. Type: official API documentation. Verification: Relevant encrypted reasoning sections fetched October31. Limitation: Interface does not establish cryptographic implementation or identity restrictions.

### F10: Safety best practices

[Safety best practices](https://developers.openai.com/api/docs/guides/safety-best-practices). Evidence category: Provider disclosure or official documentation.

F10. OpenAI. Date/version label: None. Type: official API documentation. Verification: Relevant safety-identifier section fetched October31. Limitation: Client-provided identity is not unique-person verification.

### F11: Safety classifiers

[Safety classifiers](https://developers.openai.com/api/docs/guides/safety-checks). Evidence category: Provider disclosure or official documentation.

F11. OpenAI. Date/version label: None. Type: official API documentation. Verification: Relevant enforcement section fetched October31. Limitation: Stated biology/chemical coverage; not an anti-distillation endpoint; replacement-account controls required.

### F12: Rate limits

[Rate limits](https://developers.openai.com/api/docs/guides/rate-limits). Evidence category: Provider disclosure or official documentation.

F12. OpenAI. Date/version label: None. Type: official API documentation. Verification: Relevant scopes, metrics and spend-control sections fetched October31. Limitation: Capacity limits do not establish extraction-safe budgets.

### F13: Rate limits

[Rate limits](https://platform.claude.com/docs/en/api/rate-limits). Evidence category: Provider disclosure or official documentation.

F13. Anthropic. Date/version label: None. Type: official API documentation. Verification: Relevant bucket/tier/cap sections fetched October31. Limitation: Account history and payment do not guarantee legitimacy.

### F14: Stealing Reasoning Traces from Proprietary LLM APIs

[Stealing Reasoning Traces from Proprietary LLM APIs](https://arxiv.org/html/2608.09867v1). Evidence category: Preprint or mixed preprint record.

F14. Alexander Panfilov, David Schmotz, Ilia Shumailov, Luca Beurer-Kellner, Joachim Schaeffer, Ameya Prabhu, Jonas Geiping, Maksym Andriushchenko. Date/version label: 2026-08-10. Type: research preprint. Verification: Full HTML setup, compatibility, mitigations and limitations checked. Limitation: Historical APIs; incomplete plaintext ground truth; global-key explanation inferred rather than proven.

### F15: Identity verification on Claude

[Identity verification on Claude](https://support.claude.com/en/articles/14328960-identity-verification-on-claude). Evidence category: Provider disclosure or official documentation.

F15. Anthropic. Date/version label: 2026-08-11. Type: official support documentation. Verification: Relevant identity/business/privacy/appeal sections fetched. Limitation: Selected cases; no quantified uniqueness or false-rejection rate.

### F16: Anthropic Transparency Hub: System Trust and Reporting

[Anthropic Transparency Hub: System Trust and Reporting](https://www.anthropic.com/transparency/system-trust-reporting). Evidence category: Provider disclosure or official documentation.

F16. Anthropic. Date/version label: 2026-07-23. Type: official transparency statistics. Verification: Current H1 2026 bans/appeals/overturn table fetched. Limitation: Not distillation-specific; appeal overturn fraction is not classifier FPR or population false-ban estimate.

### F17: Safeguards warnings and appeals

[Safeguards warnings and appeals](https://support.claude.com/en/articles/8241253-safeguards-warnings-and-appeals). Evidence category: Provider disclosure or official documentation.

F17. Anthropic. Date/version label: None. Type: official support documentation. Verification: Appeal process and delayed-response notice fetched October31; marked updated today. Limitation: No service-level resolution deadline; mutable page.

### F18: Antidistillation preserves AI openness, originality, and safety.

[Antidistillation preserves AI openness, originality, and safety.](https://antidistillation.com/blog/unexpected-externalities-of-distillation/). Evidence category: Code, practitioner writing or other primary material.

F18. Asher Trockman and Yash Savani. Date/version label: 2026-02-19. Type: researcher practitioner opinion. Verification: Full relevant sampling/fingerprinting discussion fetched; updated February23. Limitation: Personal-capacity advocacy; not independent production validation.

### F19: Anthropic is now limiting having multiple individual accounts?

[Anthropic is now limiting having multiple individual accounts?](https://www.reddit.com/r/ClaudeCode/comments/1x6fc82/anthropic_is_now_limiting_having_multiple/). Evidence category: Anonymous, unverified account.

F19. Anonymous Reddit user. Date/version label: None. Type: unverified user anecdote. Verification: Full post fetched October31; allegation not independently verified. Limitation: Conflicting timestamp labels; much policy interpretation explicitly AI-generated; no population inference.

### D01: Do Defenses Against LLM Extraction Work Across Attacks? A Lifecycle Benchmark of Black-Box Model Extraction

[Do Defenses Against LLM Extraction Work Across Attacks? A Lifecycle Benchmark of Black-Box Model Extraction](https://arxiv.org/html/2610.00839v1). Evidence category: Preprint or mixed preprint record.

D01. Shuze Liu; Kaixiang Zhao; Runyang Xu; Jingzhi Chen; Nathan Wu; Yu Wang; Yushun Dong. Date/version label: 2026-09-30. Type: preprint. Verification: full-text verified, v1; tables and protocol inspected. Limitation: Adapted detectors, fixed seed, four benign test batches..

### D02: An Embarrassingly Simple Detector for Model Extraction Attacks in Large Language Model API Traffic

[An Embarrassingly Simple Detector for Model Extraction Attacks in Large Language Model API Traffic](https://arxiv.org/html/2606.05725v1). Evidence category: Preprint or mixed preprint record.

D02. Shuze Liu; Qianwen Guo; Yushun Dong. Date/version label: 2026-06-04. Type: preprint. Verification: full-text verified, v1; methods and limitations inspected. Limitation: Semantically close traffic, low contamination, drift and account calibration remain limitations..

### D03: PRADA: Protecting against DNN Model Stealing Attacks

[PRADA: Protecting against DNN Model Stealing Attacks](https://arxiv.org/html/1805.02628v3). Evidence category: Peer-reviewed or proceedings record.

D03. Mika Juuti; Sebastian Szyller; Samuel Marchal; N. Asokan. Date/version label: 2018-11-16 inspected v3; EuroS&P 2019. Type: peer-reviewed paper. Verification: full-text verified; v3; Sybil discussion inspected. Limitation: Authors explicitly acknowledge sufficient Sybil clients bypass it; image setting..

### D04: SEAT: Similarity Encoder by Adversarial Training for Detecting Model Extraction Attack Queries

[SEAT: Similarity Encoder by Adversarial Training for Detecting Model Extraction Attack Queries](https://surrealyz.github.io/files/pubs/seat-aisec2021.pdf). Evidence category: Peer-reviewed or proceedings record.

D04. Zhanyuan Zhang; Yizheng Chen; David Wagner. Date/version label: 2021-11-15. Type: peer-reviewed workshop paper. Verification: full-text author PDF verified. Limitation: Image tasks; query FPR is not account false-block probability..

### D05: AI Model Extraction Attacks: Bypassing Single-Client Assumptions in Defenses

[AI Model Extraction Attacks: Bypassing Single-Client Assumptions in Defenses](https://arxiv.org/html/2606.03381v1). Evidence category: Preprint or mixed preprint record.

D05. Maxime Schwarzer et al.. Date/version label: 2026-06-02; authors state ICMCIS May 2026 presentation. Type: preprint with conference-presentation claim. Verification: full-text verified, v1; Table II and setup inspected. Limitation: MNIST, customized global PRADA, scenario-specific precision..

### D06: Protecting Object Detection Models from Model Extraction Attack via Feature Space Coverage

[Protecting Object Detection Models from Model Extraction Attack via Feature Space Coverage](https://www.ijcai.org/proceedings/2024/0048.pdf). Evidence category: Peer-reviewed or proceedings record.

D06. Zeyu Li; Yuwen Pu; Xuhong Zhang; Yu Li; Jinbao Li; Shouling Ji. Date/version label: 2024. Type: peer-reviewed conference paper. Verification: full-text PDF verified. Limitation: Object detection; not LLM distillation evaluation..

### D07: Extracting Robust Models with Uncertain Examples

[Extracting Robust Models with Uncertain Examples](https://tianweiz07.github.io/Papers/23-iclr-1.pdf). Evidence category: Peer-reviewed or proceedings record.

D07. Guanlin Li; Guowen Xu; Shangwei Guo; Han Qiu; Jiwei Li; Tianwei Zhang. Date/version label: 2023. Type: peer-reviewed conference paper. Verification: full-text author PDF verified. Limitation: Image robustness objectives and hard-label setting do not transfer numerically to LLMs..

### D08, O04: Defending Against Model Stealing Attacks with Adaptive Misinformation

[Defending Against Model Stealing Attacks with Adaptive Misinformation](https://arxiv.org/pdf/1911.07100). Evidence category: Preprint or mixed preprint record.

D08. Sanjay Kariyappa; Moinuddin K. Qureshi. Date/version label: 2019 preprint; CVPR 2020 work. Type: conference work; inspected preprint version. Verification: full-text arXiv PDF verified; CVF direct fetch returned 403. Limitation: In-distribution extraction violates its premise; vision evaluation..

O04. Sanjay Kariyappa; Moinuddin K Qureshi. Date/version label: 2020. Type: peer-reviewed paper. Verification: abstract and PDF beginning verified. Limitation: OOD assumption does not cover indistinguishable legitimate/attack queries.

### D09: Thieves on Sesame Street! Model Extraction of BERT-based APIs

[Thieves on Sesame Street! Model Extraction of BERT-based APIs](https://arxiv.org/pdf/1910.12366). Evidence category: Peer-reviewed or proceedings record.

D09. Kalpesh Krishna; Gaurav Singh Tomar; Ankur P. Parikh; Nicolas Papernot; Mohit Iyyer. Date/version label: 2020-10-12 v3; ICLR 2020. Type: peer-reviewed conference paper. Verification: full-text PDF verified, v3. Limitation: 2020 task-specific BERT and pricing; not modern frontier sample complexity..

### D10: Grey-box Extraction of Natural Language Models

[Grey-box Extraction of Natural Language Models](https://proceedings.mlr.press/v139/zanella-beguelin21a/zanella-beguelin21a.pdf). Evidence category: Peer-reviewed or proceedings record.

D10. Santiago Zanella-Béguelin; Shruti Tople; Andrew Paverd; Boris Köpf. Date/version label: 2021-07. Type: peer-reviewed conference paper. Verification: full-text proceedings PDF verified. Limitation: Restricted grey-box architecture, not full-transformer recovery..

### D11, O18: The False Promise of Imitating Proprietary LLMs

[The False Promise of Imitating Proprietary LLMs](https://arxiv.org/pdf/2305.15717). Evidence category: Preprint or mixed preprint record.

D11. Arnav Gudibande; Eric Wallace; Charlie Snell; Xinyang Geng; Hao Liu; Pieter Abbeel; Sergey Levine; Dawn Song. Date/version label: 2023-05-25 v1. Type: preprint. Verification: full-text PDF verified. Limitation: Older weak bases; does not establish modern reasoning-distillation impossibility..

O18. Arnav Gudibande; Eric Wallace; Charlie Snell; Xinyang Geng; Hao Liu; Pieter Abbeel; Sergey Levine; Dawn Song. Date/version label: 2023-05-25. Type: preprint. Verification: abstract-only verified. Limitation: 2023models and methods; cannot conclude present distillation fails.

### D12: DistillGuard: Evaluating Defenses Against LLM Knowledge Distillation

[DistillGuard: Evaluating Defenses Against LLM Knowledge Distillation](https://arxiv.org/html/2603.07835v1). Evidence category: Preprint or mixed preprint record.

D12. Bo Jiang. Date/version label: 2026-03-08. Type: preprint. Verification: full-text verified, v1; limitations inspected. Limitation: Single same-family teacher/student, naive attacker, fixed scale..

### D13: Stealing and Defending the Ends of LLMs

[Stealing and Defending the Ends of LLMs](https://openreview.net/pdf?id=dOS3e1kC5u). Evidence category: Submission with partial verification.

D13. Anonymous in retrieved primary excerpt; attributed by Liu et al. to Kulkarni, Boenisch and Dziedzic. Date/version label: ICLR 2026 submission; exact date not independently extracted. Type: submission; accepted status unverified. Verification: snippet-only primary indexed excerpt; full retrieval blocked by browser challenge. Limitation: Do not assert original numerical performance or accepted status..

### D14: A Kernel Two-Sample Test

[A Kernel Two-Sample Test](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf). Evidence category: Peer-reviewed or proceedings record.

D14. Arthur Gretton; Karsten M. Borgwardt; Malte J. Rasch; Bernhard Schölkopf; Alexander Smola. Date/version label: 2012-03. Type: peer-reviewed journal paper. Verification: full-text PDF verified. Limitation: Population characteristic-kernel separation is not extraction-intent detection..

### D15: Learning Deep Kernels for Non-Parametric Two-Sample Tests

[Learning Deep Kernels for Non-Parametric Two-Sample Tests](https://proceedings.mlr.press/v119/liu20m/liu20m.pdf). Evidence category: Peer-reviewed or proceedings record.

D15. Feng Liu; Wenkai Xu; Jie Lu; Guangquan Zhang; Arthur Gretton; Danica J. Sutherland. Date/version label: 2020-07. Type: peer-reviewed conference paper. Verification: full-text proceedings PDF verified. Limitation: No extraction evaluation; possible source/generator overfit..

### D16: Revisiting Classifier Two-Sample Tests

[Revisiting Classifier Two-Sample Tests](https://arxiv.org/html/1610.06545v3). Evidence category: Peer-reviewed or proceedings record.

D16. David Lopez-Paz; Maxime Oquab. Date/version label: ICLR 2017; inspected v3. Type: peer-reviewed conference paper. Verification: full-text arXiv HTML verified; OpenReview retrieval blocked. Limitation: Classification can exploit source leakage rather than malicious intent..

### D17: Time-uniform Chernoff bounds via nonnegative supermartingales

[Time-uniform Chernoff bounds via nonnegative supermartingales](https://arxiv.org/html/1808.03204v8). Evidence category: Peer-reviewed or proceedings record.

D17. Steven R. Howard; Aaditya Ramdas; Jon McAuliffe; Jasjeet Sekhon. Date/version label: Probability Surveys 2020; arXiv v8 2025-12-17. Type: peer-reviewed journal paper; later arXiv revision. Verification: full-text v8 verified; canonical metadata verified. Limitation: Does not validate arbitrary empirical-rank products..

### D18: E-detectors: a nonparametric framework for sequential change detection

[E-detectors: a nonparametric framework for sequential change detection](https://arxiv.org/html/2203.03532v4). Evidence category: Peer-reviewed or proceedings record.

D18. Jaehyeok Shin; Aaditya Ramdas; Alessandro Rinaldo. Date/version label: 2023-10-29 v4 inspected; originally 2022-03-07. Type: research paper; journal status not checked. Verification: full-text verified, v4; ARL section inspected. Limitation: ARL control differs from probability of ever falsely intervening..

### D19: Conformal Prediction Beyond Exchangeability

[Conformal Prediction Beyond Exchangeability](https://arxiv.org/html/2202.13415v3). Evidence category: Peer-reviewed or proceedings record.

D19. Rina Foygel Barber; Emmanuel J. Candès; Aaditya Ramdas; Ryan J. Tibshirani. Date/version label: Annals of Statistics 2023; inspected v3. Type: peer-reviewed journal paper. Verification: full-text verified. Limitation: Not a blanket guarantee for pooled dependent traffic windows..

### D20: Change detection with conformal martingales: new optimal constructions, and suboptimality of existing methods

[Change detection with conformal martingales: new optimal constructions, and suboptimality of existing methods](https://arxiv.org/html/2609.27179v1). Evidence category: Preprint or mixed preprint record.

D20. Swapnaneel Bhattacharyya; Aaditya Ramdas. Date/version label: 2026-09-23. Type: preprint. Verification: full text accessible; relevant introduction checked, proofs not audited. Limitation: Independent-observation assumptions; no LLM deployment test..

### D21: Non-partitioned e-detectors for nonparametric sequential change detection

[Non-partitioned e-detectors for nonparametric sequential change detection](https://arxiv.org/html/2607.28322v1). Evidence category: Preprint or mixed preprint record.

D21. Author names not extracted in this scoped pass; consult canonical arXiv metadata. Date/version label: 2026-07-30. Type: preprint. Verification: full text accessible; abstract and method outline checked, proofs not audited. Limitation: Not an off-the-shelf graph/LLM calibration guarantee..

### D22: The Sybil Attack

[The Sybil Attack](https://www.microsoft.com/en-us/research/publication/the-sybil-attack/). Evidence category: Peer-reviewed or proceedings record.

D22. John R. Douceur. Date/version label: 2002-01 publication page; IPTPS 2002. Type: peer-reviewed workshop paper. Verification: abstract-only official publication page; full paper not audited. Limitation: Distributed-system result does not itself quantify paid-API economics..

### D24: Bounded Behavioral Indistinguishability for Black-Box LLM Distillation

[Bounded Behavioral Indistinguishability for Black-Box LLM Distillation](https://arxiv.org/abs/2605.30448). Evidence category: Preprint or mixed preprint record.

D24. Munawar Hasan. Date/version label: 2026-05-28. Type: preprint. Verification: abstract-only verified. Limitation: Full methodology not checked; lead only..

### D25: Exploring Connections Between Active Learning and Model Extraction

[Exploring Connections Between Active Learning and Model Extraction](https://www.usenix.org/conference/usenixsecurity20/presentation/chandrasekaran). Evidence category: Peer-reviewed or proceedings record.

D25. Varun Chandrasekaran; Kamalika Chaudhuri; Irene Giacomelli; Somesh Jha; Songbai Yan. Date/version label: 2020-08. Type: peer-reviewed conference paper. Verification: Abstract-only official proceedings page verified; full paper not inspected.. Limitation: Classic classifiers; not proof of generative frontier LLM performance..

### D26: Knockoff Nets: Stealing Functionality of Black-Box Models

[Knockoff Nets: Stealing Functionality of Black-Box Models](https://openaccess.thecvf.com/content_CVPR_2019/html/Orekondy_Knockoff_Nets_Stealing_Functionality_of_Black-Box_Models_CVPR_2019_paper.html). Evidence category: Peer-reviewed or proceedings record.

D26. Tribhuvanesh Orekondy; Bernt Schiele; Mario Fritz. Date/version label: 2019-06. Type: peer-reviewed conference paper. Verification: Abstract-only official proceedings page verified; PDF fetch returned 403. Image tasks.. Limitation: Classic classifiers; not proof of generative frontier LLM performance..

### D27: ActiveThief: Model Extraction Using Active Learning and Unannotated Public Data

[ActiveThief: Model Extraction Using Active Learning and Unannotated Public Data](https://ojs.aaai.org/index.php/AAAI/article/view/5432). Evidence category: Peer-reviewed or proceedings record.

D27. Soham Pal; Yash Gupta; Aditya Shukla; Aditya Kanade; Shirish Shevade; Vinod Ganapathy. Date/version label: 2020-04-03. Type: peer-reviewed conference paper. Verification: Abstract-only official proceedings page verified; full experimental settings not checked.. Limitation: Classic classifiers; not proof of generative frontier LLM performance..

### O01: Antidistillation Sampling

[Antidistillation Sampling](https://arxiv.org/html/2504.13146v5). Evidence category: Preprint or mixed preprint record.

O01. Yash Savani; Asher Trockman; Zhili Feng; Yixuan Even Xu; Avi Schwarzschild; Alexander Robey; Marc Finzi; J. Zico Kolter. Date/version label: 2025-10-26. Type: preprint. Verification: full-text verified. Limitation: Adaptive cleanup/retries/prefill counterevidence; no frontier guarantee.

### O02: What Does It Mean to Break a Distillation Defense?

[What Does It Mean to Break a Distillation Defense?](https://arxiv.org/html/2606.25059v2). Evidence category: Preprint or mixed preprint record.

O02. Lena Libon; Pura Peetathawatchai; Michael Aerni; Daniel Paleka; Florian Tramèr. Date/version label: 2026-07-13. Type: preprint. Verification: full-text verified. Limitation: GSM8K and chosen small models; matched-utility curve interpolated.

### O03: Prediction Poisoning: Towards Defenses Against DNN Model Stealing Attacks

[Prediction Poisoning: Towards Defenses Against DNN Model Stealing Attacks](https://resources.mpi-inf.mpg.de/d2/orekondy/predpoison/). Evidence category: Peer-reviewed or proceedings record.

O03. Tribhuvanesh Orekondy; Bernt Schiele; Mario Fritz. Date/version label: 2020. Type: peer-reviewed paper author page. Verification: author-page abstract verified in search; direct open returned403. Limitation: Image classifiers and selected attacks; headline85x error is maximum.

### O05: ADS-C: Antidistillation Sampling for Classification

[ADS-C: Antidistillation Sampling for Classification](https://arxiv.org/abs/2607.15467). Evidence category: Preprint or mixed preprint record.

O05. Khawaja Abaid Ullah; Mohammad Javad Khojasteh. Date/version label: 2026-07-16. Type: preprint. Verification: abstract-only verified. Limitation: Cannot remove hard-label supervision; image classification; not independently reproduced.

### O06: Antidistillation Fingerprinting

[Antidistillation Fingerprinting](https://arxiv.org/html/2602.03812v2). Evidence category: Preprint or mixed preprint record.

O06. Yixuan Even Xu; John Kirchenbauer; Yash Savani; Asher Trockman; Alexander Robey; Tom Goldstein; Fei Fang; J. Zico Kolter. Date/version label: 2026-05-15. Type: paper arXiv record states ICML2026. Verification: full-text verified; proceedings status not independently checked. Limitation: Closed access needs controlled continuations; evaluation reused prompts.

### O07: Watermarking Makes Language Models Radioactive

[Watermarking Makes Language Models Radioactive](https://arxiv.org/abs/2402.14904). Evidence category: Peer-reviewed or proceedings record.

O07. Tom Sander; Pierre Fernandez; Alain Durmus; Matthijs Douze; Teddy Furon. Date/version label: 2024. Type: peer-reviewed paper. Verification: abstract and selected PDF text verified. Limitation: 5percent high-confidence example is open-weight setting; no prevention result.

### O08: Protecting Language Generation Models via Invisible Watermarking

[Protecting Language Generation Models via Invisible Watermarking](https://proceedings.mlr.press/v202/zhao23i.html). Evidence category: Peer-reviewed or proceedings record.

O08. Xuandong Zhao; Yu-Xiang Wang; Lei Li. Date/version label: 2023. Type: peer-reviewed paper. Verification: official abstract verified. Limitation: mAP is attribution metric; no capability suppression guarantee.

### O09: DAWN: Dynamic Adversarial Watermarking of Neural Networks

[DAWN: Dynamic Adversarial Watermarking of Neural Networks](https://arxiv.org/abs/1906.00830). Evidence category: Peer-reviewed or proceedings record.

O09. Sebastian Szyller; Buse Gul Atli; Samuel Marchal; N. Asokan. Date/version label: 2021. Type: peer-reviewed paper. Verification: abstract-only verified. Limitation: Classifier study; incorrect labels cost utility.

### O10: Scalable watermarking for identifying large language model outputs

[Scalable watermarking for identifying large language model outputs](https://www.nature.com/articles/s41586-024-08025-4). Evidence category: Peer-reviewed or proceedings record.

O10. Sumanth Dathathri; Abigail See; Sumedh Ghaisas; Po-Sen Huang; Rob McAdam; et al.. Date/version label: 2024-10-23. Type: peer-reviewed production study. Verification: official search full abstract and selected algorithm text verified; later direct open failed. Limitation: Text provenance is distinct from distillation prevention.

### O11: PostMark: A Robust Blackbox Watermark for Large Language Models

[PostMark: A Robust Blackbox Watermark for Large Language Models](https://aclanthology.org/2024.emnlp-main.506/). Evidence category: Peer-reviewed or proceedings record.

O11. Yapei Chang; Kalpesh Krishna; Amir Houmansadr; John Frederick Wieting; Mohit Iyyer. Date/version label: 2024-11. Type: peer-reviewed paper. Verification: official abstract verified. Limitation: Long text method; exact labels/spans/code suitability not established.

### O12: Watermark Stealing in Large Language Models

[Watermark Stealing in Large Language Models](https://proceedings.mlr.press/v235/jovanovic24a.html). Evidence category: Peer-reviewed or proceedings record.

O12. Nikola Jovanović; Robin Staab; Martin Vechev. Date/version label: 2024. Type: peer-reviewed paper. Verification: official abstract verified. Limitation: Scheme-specific experiments; does not prove universal watermark failure.

### O13: On the Reliability of Watermarks for Large Language Models

[On the Reliability of Watermarks for Large Language Models](https://arxiv.org/abs/2306.04634). Evidence category: Peer-reviewed or proceedings record.

O13. John Kirchenbauer; Jonas Geiping; Yuxin Wen; Manli Shu; Khalid Saifullah; Kezhi Kong; Kasun Fernando; Aniruddha Saha; Micah Goldblum; Tom Goldstein. Date/version label: 2024. Type: peer-reviewed paper. Verification: abstract and selected PDF findings verified. Limitation: 800token human-paraphrase average is setting-specific and irrelevant to one-token labels.

### O14: Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense

[Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense](https://arxiv.org/abs/2303.13408). Evidence category: Peer-reviewed or proceedings record.

O14. Kalpesh Krishna; Yixiao Song; Marzena Karpinska; John Wieting; Mohit Iyyer. Date/version label: 2023-10-18. Type: peer-reviewed paper. Verification: abstract-only verified. Limitation: Retrieval needs stored generations; text detection not student attribution.

### O15: Linear Ensembles Wash Away Watermarks: On the Fragility of Distributional Perturbations in LLMs

[Linear Ensembles Wash Away Watermarks: On the Fragility of Distributional Perturbations in LLMs](https://arxiv.org/html/2605.30501v1). Evidence category: Preprint or mixed preprint record.

O15. Zhihao Wu; Gracia Gong; Qinglin Zhu; Yudong Chen; Runcong Zhao. Date/version label: 2026-05-28. Type: preprint. Verification: full-text assumptions and abstract verified. Limitation: Theory assumes independent centered bounded perturbations; numerical-interface requirements.

### O16: Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes

[Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://arxiv.org/abs/2305.02301). Evidence category: Peer-reviewed or proceedings record.

O16. Cheng-Yu Hsieh; Chun-Liang Li; Chih-Kuan Yeh; Hootan Nakhost; Yasuhisa Fujii; Alexander Ratner; Ranjay Krishna; Chen-Yu Lee; Tomas Pfister. Date/version label: 2023-07-05. Type: peer-reviewed paper. Verification: abstract-only verified. Limitation: Small task-specific distillation; not all-task proof.

### O17: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://github.com/deepseek-ai/DeepSeek-R1). Evidence category: Code, practitioner writing or other primary material.

O17. DeepSeek-AI. Date/version label: 2025. Type: official technical report and code. Verification: official README and arXiv abstract verified. Limitation: Provider-reported setup; no controlled anti-distillation comparison.

### O19: Stealing Part of a Production Language Model

[Stealing Part of a Production Language Model](https://arxiv.org/abs/2403.06634). Evidence category: Peer-reviewed or proceedings record.

O19. Nicholas Carlini; Daniel Paleka; Krishnamurthy Dvijotham; Thomas Steinke; Jonathan Hayase; A. Feder Cooper; Katherine Lee; Matthew Jagielski; Milad Nasr; Arthur Conmy; Itay Yona; Eric Wallace; David Rolnick; Florian Tramèr. Date/version label: 2024-03-11. Type: peer-reviewed paper. Verification: abstract and author project page verified. Limitation: Partial parameter extraction differs from capability imitation.

### O20: Radioactive watermark official reproduction code

[Radioactive watermark official reproduction code](https://github.com/facebookresearch/radioactive-watermark). Evidence category: Code, practitioner writing or other primary material.

O20. Meta FAIR researchers. Date/version label: 2024. Type: official code. Verification: README verified. Limitation: Repository archived2025-12-01; inspect dependency compatibility.

### O21: DAWN official code

[DAWN official code](https://github.com/ssg-research/dawn-dynamic-adversarial-watermarking-of-neural-networks). Evidence category: Code, practitioner writing or other primary material.

O21. DAWN authors. Date/version label: 2021. Type: official code. Verification: README verified. Limitation: README says extraction-attack experiments omitted.

