# Standards, taxonomies and operational controls for distillation detection

Research cut-off is 2026-11-05. Public primary sources were checked live. This note distinguishes formal requirements, voluntary guidance, threat taxonomies and proposed TRACE engineering work. Repository inspection was read-only outside this research note.

## Which standards and authoritative references apply?

### Takeaway

There is no extraction detection algorithm, detection threshold or certified effectiveness level prescribed in the public materials reviewed. The useful references define threats, establish risk management processes, recommend access and resource controls, and call for measurement. Formal ISO management-system requirements and an OWASP risk list have different authority and scope.

### Cited findings

| Reference and checked version | Type and supported scope | Relevance and limits |
| --- | --- | --- |
| [NIST AI 100-2e2025, final 2025-03-24](https://csrc.nist.gov/pubs/ai/100/2/e2025/final) | Taxonomy and terminology report intended to inform standards and future practice guides. | Section 2.4.4 assigns model extraction `NISTAML.031`. It distinguishes architecture/parameter extraction, learning-based extraction, active learning and side channels. Section 2.4.5 explicitly states differential privacy does not guarantee protection against model extraction because it protects training data. This is no certification of a detector. [Full report](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2025.pdf). |
| [NIST AI RMF 1.0, AI 100-1, January 2023](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf) | Voluntary, context-dependent risk management framework. | MEASURE 2.1 documents test sets and metrics; 2.3 evaluates deployment-like conditions; 2.4 monitors production; 2.7 evaluates security and resilience; 3.3 includes appeal feedback. MANAGE includes response, recovery and communication. These outcomes do not choose MMD, novelty, perplexity or another detector. [NIST framework page](https://www.nist.gov/itl/ai-risk-management-framework). |
| [NIST AI 600-1, Generative AI Profile, July 2024](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) | Companion voluntary guidance with suggested actions. | MS-2.7-001 covers model theft, weights exposure and extraction threats. MS-2.7-004 asks for security effectiveness metrics; MS-2.7-007 explicitly includes model extraction in AI red-teaming. MS-2.7-009 asks for continued control assessment. These are process and evaluation recommendations, with no required extraction classifier or accuracy target. |
| [OWASP API Security Top 10, API4:2023](https://api-security.owasp.org/editions/2023/en/0xa4-unrestricted-resource-consumption/) | Community security risk guidance. | Recommends rate limits, per-operation throttling, input-size limits, execution/resource bounds and provider spending limits or billing alerts. This protects cost and availability; it does not classify training intent. The [official project page](https://owasp.org/projects/api-security-project) still lists 2023 as latest. |
| [OWASP API6:2023, Unrestricted Access to Sensitive Business Flows](https://api-security.owasp.org/editions/2023/en/0xa6-unrestricted-access-to-sensitive-business-flows/) | Community guidance for automated abuse of legitimate workflows. | Business owners identify harmful excessive use; engineers choose controls. It discusses non-human patterns, fingerprints, human checks and machine-consumed API access. It explicitly recognizes that the same automated behavior can be harmful in one business and encouraged in another. Therefore automation alone cannot establish unauthorized distillation. |
| [OWASP GenAI LLM Top 10 2026, LLM06:2026 Unbounded Consumption](https://raw.githubusercontent.com/GenAI-Security-Project/GenAI-LLM-Top10/main/2026/final/LLM06_UnboundedConsumption.md) | Current community risk guide, not a normative detection standard. | Explicitly includes extraction and distillation theft. Recommends token-aware limits, pre-flight token estimation, hard spending caps per key/user/team/cloud account, authenticated inference, resource monitoring and agent circuit breakers. LLM03:2026 is Excessive Agency, concerning damaging tool actions. It is not the primary category for a plain inference API extraction campaign. [Official PDF](https://genai.owasp.org/download/56857/). |
| [OWASP 2025 LLM10 Unbounded Consumption](https://genai.owasp.org/llmrisk/llm102025-unbounded-consumption/) and [2023 v1.1 archive](https://owasp.github.io/www-project-top-10-for-large-language-model-applications/) | Earlier editions, useful only with explicit years. | The 2025 category includes functional replication, quotas, logging/anomaly detection and reduced logit/logprob exposure. The 2023 category was LLM10 Model Theft; its page now redirects to 2025, while the archived project points to 2026. Do not call the 2025 numbering current. |
| [MITRE ATLAS v2026.09, released 2026-09-15](https://github.com/mitre-atlas/atlas-data/releases/tag/v2026.09) | Living adversary knowledge base and mitigation catalog. | Current names/IDs in the [official data](https://raw.githubusercontent.com/mitre-atlas/atlas-data/main/dist/v6/ATLAS-2026.09.yaml) are Extract AI Model `AML.T0024.002`, AI Model Inference API Access `AML.T0040`, Train Proxy via Replication `AML.T0005.001`, and AI Intellectual Property Theft `AML.T0048.004`. `AML.T0024` is the parent technique, not the extraction sub-technique. Relevant mitigations are Limit AI Service Query Volume and Rate `AML.M0004`, AI Telemetry Logging `AML.M0024`, and AI Red Team `AML.M0035`. M0004 explicitly mentions authenticated identities, burst/concurrency controls, longer quotas, distributed evasion, and limits against low-query/offline attacks. No certified detector follows. |
| [ISO/IEC 27001:2022](https://www.iso.org/standard/27001) | International Standard specifying information security management system requirements. | Public ISO material supports organization-level security risk management and continuous improvement. It does not establish that an extraction detector is accurate or required in a particular algorithmic form. |
| [ISO/IEC 42001:2023](https://www.iso.org/standard/42001) | International Standard specifying AI management system requirements. | Public scope covers establishing, implementing, maintaining and improving an AIMS. Management-system conformity is a different claim from measured extraction prevention. |
| [ISO/IEC 23894:2023](https://www.iso.org/standard/77304.html) | International Standard providing guidance on AI risk management. | It describes integrating AI risk management into organizational activities, customized to context. Its title says guidance, so do not present it as a required extraction detector. |
| [ISO/IEC TR 24028:2020](https://www.iso.org/standard/77608.html) | Technical Report surveying AI trustworthiness, engineering risks and possible mitigations. | ISO explicitly puts specification of trustworthiness levels outside its scope. It cannot support an efficacy threshold or certification claim for TRACE. |

Version provenance matters. The official [legacy OWASP project page](https://owasp.github.io/www-project-top-10-for-large-language-model-applications/) calls 2026 current and says published August 4, while the [2026 resource page](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/) is dated August 3. The downloadable PDF identifies version 2026 but still has a publication-date placeholder. The source directory is `2026/final`; cite the edition and checked date without resolving the one-day discrepancy by guesswork.

### Inferences

- The defensible phrase is "controls mapped to NIST/OWASP/ATLAS guidance, with independently reported experimental performance." Avoid "industry-standard distillation detector," "ISO-certified extraction detection" and "NIST-approved MMD." This follows from the scopes and recommendations above.
- A practical baseline consists of authenticated access, resource accounting, durable quotas, telemetry and incident handling. Research signals must demonstrate value beyond these controls at matched legitimate utility. The choice of model, algorithm, budget and retention period remains a deployment decision.

### Gaps

- Full ISO texts were not available openly in the materials reviewed. No paywalled clauses or Annex A control numbers are asserted here. Scope summaries cannot establish full compliance or prove absence of every possible algorithm reference in full texts.
- A September 8, 2026 NSA/CISA malicious-distillation advisory was identified by the other research stream. The [CISA page](https://www.cisa.gov/news-events/cybersecurity-advisories/aa26-251a) and [official defense.gov PDF](https://media.defense.gov/2026/Sep/08/2003992823/-1/-1/1/CSA_CHINA_BASED_AI_COMPANIES_MALICIOUS_DISTILLATION_AGAINST_US.PDF) could not be fetched here; the PDF returned HTTP 403 using both browsing and curl. This note makes no detailed claims about its recommendations.

## What should those references mean operationally?

### Takeaway

Use the guidance to define an accountable protection program. Treat volume limits as acquisition/cost controls and statistical alerts as evidence requiring validation and review. Neither proves whether the caller trains a student.

### Cited findings

The following mapping is engineering advice derived from the named references, not an assertion that their authors prescribe TRACE's implementation.

| Control area | Concrete implementation and evidence to collect | Source basis |
| --- | --- | --- |
| Authentication and organization identity | Bind requests to verified principals and declared organizations; rotate and revoke keys; retain mapping provenance. Distinguish shared enterprise keys, proxies and actual common ownership. Evaluate account rotation and incorrectly merged organizations. | [ATLAS M0004 and M0024](https://raw.githubusercontent.com/mitre-atlas/atlas-data/main/dist/v6/ATLAS-2026.09.yaml), [API6](https://api-security.owasp.org/editions/2023/en/0xa6-unrestricted-access-to-sensitive-business-flows/). |
| Durable resource budgets | Reserve maximum output tokens before dispatch, settle actual usage, account for failures/retries/concurrency, and enforce per-principal and organization windows across restarts. Track cost/input/reasoning/tool use where applicable. | [OWASP 2026 LLM06](https://raw.githubusercontent.com/GenAI-Security-Project/GenAI-LLM-Top10/main/2026/final/LLM06_UnboundedConsumption.md), [API4](https://api-security.owasp.org/editions/2023/en/0xa4-unrestricted-resource-consumption/). |
| Rate and resource controls | Apply request/token rate, burst, concurrency, queue, request-size and timeout bounds. Evaluate low-and-slow and distributed campaigns separately. | [API4](https://api-security.owasp.org/editions/2023/en/0xa4-unrestricted-resource-consumption/), [ATLAS M0004](https://raw.githubusercontent.com/mitre-atlas/atlas-data/main/dist/v6/ATLAS-2026.09.yaml). |
| Logging and monitoring | Correlate authentication, request identity, reservation, policy revision, response release, known token use and incident/review decisions. Minimize sensitive prompt retention and control log access. | [ATLAS M0024](https://raw.githubusercontent.com/mitre-atlas/atlas-data/main/dist/v6/ATLAS-2026.09.yaml), [2025 LLM10](https://genai.owasp.org/llmrisk/llm102025-unbounded-consumption/). |
| Anomaly detection | Establish legitimate profile baselines; combine content/session/cost/linkage evidence cautiously; measure false alerts, misses, abstentions and drift. Never infer malicious intent from automation, breadth or a corpus match alone. | [API6](https://api-security.owasp.org/editions/2023/en/0xa6-unrestricted-access-to-sensitive-business-flows/), [AI RMF MEASURE](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf). |
| Response and recovery | Define alert triage, verification, temporary limits, revocation, appeals, restoration and evidence retention. Track incident-to-action time and restoration burden. Keep proven identity links distinct from tentative clusters. | [AI RMF MANAGE and MEASURE 3.3](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf). |
| Red-team validation | Run authorized adaptive extraction and identity evasion, train actual students, assess benign utility, and compare with quota/token/rate baselines. Freeze policy and evaluation data before final assessment. | [AI 600-1 MS-2.7-007](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf), [ATLAS M0035](https://raw.githubusercontent.com/mitre-atlas/atlas-data/main/dist/v6/ATLAS-2026.09.yaml). |

### Inferences

- The first operating target should be a measurable release bound with a known identity scope. A separate research target should be increased attacker cost or time to a prespecified student capability at measured legitimate utility.
- Persist low-cost identity, counters and releases first. Keep expensive semantic scoring bounded and measure its latency before enabling it in comparison arms.
- Put statistical alerts into a proportionate review workflow until independent benign and adaptive-attack evaluations justify an enforcing threshold. A low p-value concerns the selected null distribution; it is not a probability of maliciousness.

### Gaps

- The reviewed sources provide no universal safe token budget, training-intent ground truth, extraction detection threshold or guarantee that request monitoring distinguishes a benign-distributed extractor.

## How does TRACE currently map to these controls, and what remains unqualified?

### Takeaway

TRACE already has many relevant accounting and telemetry mechanisms, with explicit limitations. Its current docs describe development tools and a future evaluation protocol, so neither implementation breadth nor external framework mapping establishes real deployment efficacy.

### Cited findings

- [docs/harness.md](../../harness.md) documents API-key digests mapped to principals and explicit organizations; persistent SQLite admission, reservations, idempotency and aggregate state; per-principal/organization token and rolling budgets; in-flight limits; and append-only human review decisions. State defaults to memory. Organization mapping is supplied by the operator and does not discover Sybils.
- [docs/harness.md](../../harness.md) distinguishes attempt caps, quotas, rate, detector and entitlement reasons. It says budgeting bounds release only under accurate usage, intact state and gateway routing; budget refusals are not detections. It records that rolling/organization controls lack real-teacher/student capability evidence.
- [signals.py](../../../packages/harness/src/harness/signals.py) computes opt-in horizon summaries including count, lexical follow-ups/rephrasing, template diversity, answer-only share, interarrival CV, topic transitions, nearest-neighbor novelty, corpus-match share and optional log-determinant coverage. Its own module says novelty measures geometry rather than intent and synthetic nulls support only generator-specific claims. Calibration uses score-calibration account horizons, minimum null counts, rank floors and abstention. Coverage ranks use request-count bands.
- [shadow.py](../../../packages/harness/src/harness/shadow.py) implements conditional-domain MMD and rarefied geometric coverage beside policy. Its module explicitly says neither identifies extraction and benign-distributed extraction cannot be separated by those scores. ShadowMonitor retains horizon state in process memory; the docs state restart resets it and inline shadow work can change timing.
- [docs/harness.md](../../harness.md) describes additional offline linkage/fusion evidence. Connected components are analysis hypotheses; false merges/splits require supplied truth. Missing families stay neutral; the convex fusion is shadow evidence, not a sequential e-process. The separate experimental path calibration binds a finite horizon and exact look schedule, marks decisions `qualified: false`, and lacks independent benign and teacher/student effectiveness evidence.
- [docs/evaluation.md](../../evaluation.md) labels acceptance targets as proposed. It specifies source-family disjoint roles, held-out legitimate profiles, adaptive A1-A8 campaigns, actual release accounting, baseline matching, teacher capacity checks and actual student evaluation. It separates gate/replay/software evidence from live effectiveness, and warns that inspected development artifacts cannot become locked final tests.

### Inferences

1. Keep the existing budget and detector distinction in every result table. Name the principal/organization mapping and state assumptions alongside any release bound.
2. Run a fresh campaign panel with persisted state and true account rotation. Include crash/retry/concurrency, long windows and post-alert in-flight release accounting. Verify actual output release from the ledger rather than treating decisions as delivery.
3. Assess legitimate automation, broad researchers, labeling jobs and shared organizations before promoting session, corpus or novelty ranks into enforcement. Use independent source families and organizations; count review/restoration cost.
4. Train matched-volume and matched-prefix students on released teacher outputs. Without this step, measured reductions in output establish acquisition control only.
5. Freeze the exact operational scorer, profile/stratum selection, missingness, look schedule, reset policy, review behavior and horizon. Calibrate and audit that complete procedure on independent accounts; marginal ranks alone do not establish sequential guarantees.
6. Profile the encoder/shadow/path state on the deployment hardware. Current process-local monitoring and path-prefix growth need explicit restart and resource-bound decisions before operational use.

### Gaps

- This was a targeted read of the requested docs and signal modules, not a complete implementation audit. Runtime configuration, deployed state, real-user telemetry, protected GPU/student runs and independence of proposed data were not verified. No code, refs or frozen readings were changed.
