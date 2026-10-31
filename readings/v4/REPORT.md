# TRACE harness: why the MMD defense does not restrict the extraction campaigns

Audit date 2026-10-30; recovery and independent replay verification 2026-10-31. Code: `main@e53503f` (pulled at the start of this audit). Data: the committed run record `readings/v3-2026-10-29-gateway-replay` (calibration, reference embeddings, benign workloads, attack schedules, A1 decision logs, interrupted A2 log) and `readings/v2-2026-10-21-gateway-bounded-v1` (undefended ledgers). The work lives in the jj workspace `mmd-audit` at `/home/rajveer/Projects/personal/capstone-mmd-audit`; all scripts and outputs are in its `outputs/mmd-audit/` (gitignored).

This committed archive preserves the report and results originally generated under `outputs/mmd-audit/`, plus the independent recovery results under `verified-20261031/`. Descriptions of local/uncommitted state below refer to the recovery snapshot before the archive commits. Use [README.md](README.md) for the portable launcher instructions; original absolute commands are retained as provenance. Embedding caches and virtual environments are not committed.

## What I could not see

`/home/hpclab/capstone4` is not on this machine, and no SSH host or linked environment for it exists here. I could not identify the active process, its configuration, its live decision logs, or its GPU state, and I did not try to reach it. Every result below is pinned to the committed readings and to `main@e53503f`. The live A2 figures in the request (about 1,700 decisions at L0; first 17 windows p 0.35–0.92) are reported to me, not observed. They are consistent with an offline replay of the frozen schedule: windows 1–15 are AG News at p 0.896–0.925, and windows 16–17 are the first GSM8K windows at p 0.422 and 0.352. That is consistency, not verification. Nothing in the readings directory, the protocol files, or any service was modified.

## Conclusion

The defense fails for a combination of reasons:

- **No implementation defect found in the checked detector, calibration or serving paths.** Exact replay checks consistency; it cannot exclude a defect shared by calibration and serving.
- **One demonstrated bug in the experiment's workload generator (F4).** It skews the calibration roles and makes the thresholds conservative, but it does not cause the A1 or A6 miss.
- **A statistical limitation, amplified by configuration and experimental design.** This is the main cause.

| Claim | Confidence | Evidence |
|---|---|---|
| No implementation defect found that explains the observed arms | High | A from-scratch rebuild reproduces the frozen artifact: sigma within 1.2e-7, reference vectors within 3e-7, all 598 null scores within 3e-8, every threshold and the quota identical. All 60 logged A1 window p-values (full and instantaneous) and all 15 interrupted-A2 p-values reproduce bit for bit. `packages/harness` and `apps/harness-api` are identical between the run's revisions (`3599d6e`, `93c9f1a`) and `e53503f`, so this reproduces the run's own code. |
| A workload-generator bug truncates topics in the calibration roles | High | `policy_calibration` AG News is 100% Business, and its SQuAD covers 56 of 215 titles (F4). |
| As configured, the one-sided upper-tail MMD against pooled references, with a per-account null, detects topic concentration, not extraction | High | Benign specialists are single-domain and topic-ordered, and they set the null's upper tail. Every attack samples its domains broadly, so it scores at or below the benign median. A1's median window score (0.0043) is below the null's 1st percentile (0.0049). A uniform shuffle of held-out benign prompts gets p 0.935–0.995, as A1 does (0.947–1.0). |
| A1 and A6 (same-distribution, multiplexed, Sybil) hit an inherent limit of content-only per-account detection | High | Benign generalists get p 0.928–0.985. Detectors that also test the lower tail would catch A1, but reordering the same 6,000 prompts by topic needs no additional teacher queries and moves every window into the benign bulk, so all three tested MMD variants release all 6,000 responses for both topic-sorted schedules. After that, the only remaining separators are volume and cross-account linkage. |
| A7 (independent sources) is partly a configuration miss | Medium | Domain-conditional references separate A7's coding and GSM8K streams from benign specialist windows. Its AG News (BBC) and SQuAD (NewsQA) streams overlap with topic-ordered benign specialists. |
| The accumulator cannot help in any of these arms | High | With drift 1, the statistic grows only when p < e^-1 ≈ 0.37. Reaching L4 by the twelfth window, before the 1,215-request quota, needs a sustained p below approximately 0.0975. |

In plain terms, the detector asks whether this account's last 100 requests look less like the average of all users than other users' requests do. Real users, and the synthetic ones here, concentrate on one task and topic. An extraction campaign that samples the whole task distribution therefore looks more average than any user. The test only flags the unusual direction, so it never fires.

## Verified request path and correctness scope

Line numbers in this section refer to audit commit `46d68412`; older finding references above describe the pinned baseline unless stated otherwise.

1. `apps/harness-api/src/harness_api/app.py:235` resolves a SHA-256 bearer-key digest through the configured account map. `gateway_study.py:294-309` creates that map and disables trusted actor headers. Account labels in schedules are experimental principals; this does not establish real-world identity or resistance to purchased accounts.
2. `app.py:321-350` rejects unauthenticated or malformed gateway traffic before policy admission. It validates a single task-template user message and a domain-specific frozen output cap, then canonicalizes domain and turn together. Such rejected requests have no policy-decision event; the log is an admission log, not an exhaustive HTTP access log.
3. `serialization.py:26-64` strips service templates, puts the SQuAD question before the passage, and prefixes the gateway domain. `encoder.py:21-95` pins BGE, takes CLS, normalizes vectors, and truncates at 512 tokens. The completed prefix ablation rebuilds references, bandwidth and null together and does not explain the miss.
4. `app.py:383-412` embeds requests, then serializes policy-state mutation with a lock. State order is admission order after embedding, not necessarily network-arrival order for concurrent requests. `policy.py:267-329` counts attempts, forms consecutive non-overlapping windows, optionally collapses exact repeats within a window, updates risk, latches escalation, clears the buffer, and applies the action before the teacher runs. Refused L4 requests count as attempts but are not added to later score windows.
5. `mmd.py:21-100` implements unbiased squared MMD with diagonal removal, retains negative finite-sample values, and averages four RBF scales and 20 fixed reference comparisons. A single distinct prompt uses a defined self-term of one, so repeat-collapse scoring is a policy statistic rather than the ordinary fixed-sample unbiased estimator in that case. Calibration and replay use the same rule.
6. `calibration.py:241-254` uses `(1 + count(null >= score)) / (m + 1)`, including ties conservatively, and strict threshold exceedance. The rank floor is 1/599. Overlap and shared sources mean this empirical rank is not a demonstrated conditionally valid sequential p-value. The adequacy gate establishes resolution only.
7. `app.py:354-382` handles bound, single-use proof redemption without admitting the request again. `app.py:413-447` logs admissions before upstream completion, then refuses, challenges or calls the teacher. An admission does not prove successful delivery or collected supervision. Requests cancelled, rejected or failing before admission are outside decision-log coverage. Logged full-arm A1 outcomes are also checked against campaign ledgers.
8. `policy.py:95-111` stores state in process memory, renews it on elapsed horizon time, and refuses over the attempt cap. Process restart loses state. Separate arm startup is deliberate in this study; the interrupted A2 artifact must not be stitched into the restart as one account trajectory. No live restart history could be checked remotely.

Exact calibration and logged-window replay, existing policy/API tests, and prefix/reordering probes support the explanation above. They do not prove absence of all possible authentication, concurrency, embedding, logging or statistical defects, nor validate a deployment on the unavailable lab host.

## Artifact provenance

| Artifact | Code | Calibration references (SHA-256) | Schedule (SHA-256 from `contract.json`) | UTC window | Attacker decisions / responses |
|---|---|---|---|---|---|
| A1 full | `3599d6e` plus an uncommitted order change (harness identical to `e53503f`) | `493b2b4d7b03…` | `b83c294cc0e8…` | 10-29 13:27 – 19:07 | 6,000 / 6,000 |
| A1 instantaneous | same | same | same | 10-29 19:59 – 10-30 01:39 | 6,000 / 6,000 |
| A1 quota | same | same | same | 10-30 02:31 – 04:11 | 6,000 / 1,215 (4,785 refused) |
| A2 full (interrupted) | same | same | `1713ae17492d…` | 10-30 05:02 – 05:32 | 1,537 / 1,536 (one request in flight at the stop) |
| Calibration | `4bb6381` (not in local history; reproduced exactly by `e53503f`) | `493b2b4d7b03…` | — | 10-21 15:37 – 16:04 | — |

- Each arm ran on a freshly started harness (`gateway_study.py:280-331`), so no state carries over between arms.
- Benign utility followed each attack campaign on the same harness instance, under separate accounts.
- The decision logs contain only `decision` events: no `redeem` events, since no challenges were issued, and every refusal is logged as a decision.
- Benign decisions (6,793 under full) exceed served benign responses (6,358). That is consistent with requests the harness admitted that finished after the client's 120 s deadline.

## Ranked findings

Severity is about effect on the study's conclusions. "Demonstrated" means reproduced by a check in `outputs/mmd-audit`. "Concern" means plausible but not demonstrated.

### F1. The configured statistic cannot flag broad extraction (critical, demonstrated)

- Code: `packages/harness/src/harness/calibration.py:208-212` (upper-tail p), `calibration.py:422-430` (references are a seeded shuffle of all fitting rows, so a pooled four-domain mixture), `calibration.py:432-472` (the null is per-account windows), `packages/harness/src/harness/serialization.py:37-42` (a `domain\n` prefix sharpens domain clusters).
- Workload: `packages/distillation/src/distillation/workloads.py:190-345`, `generalist_share=0.1` by default. In every role, 112 of 125 accounts are single-domain, and specialists are ordered by topic category within their domain (`workloads.py:236-246`).
- Measured (`results/diagnose-detector.json`, frozen references and null, non-overlapping 100-request windows per account):

| Traffic | Windows | Upper-tail p, min / median / max | Share with lower-tail p ≤ 0.05 |
|---|---:|---|---:|
| Benign test, specialists | 182 | 0.002 / 0.540 / 0.930 | 0.0 |
| Benign test, generalists (9 accounts) | 16 | 0.928 / 0.937 / 0.985 | 0.125 |
| Probe: uniform shuffle of benign test | 100 | 0.935 / 0.950 / 0.995 | 0.40 |
| Probe: random single-domain AG News, SQuAD, coding | 30 | 0.896 – 0.928 | 0.0 |
| Probe: random single-domain GSM8K | 6 | 0.327 / 0.340 / 0.489 | 0.0 |
| A1 (one account, interleaved) | 60 | 0.947 / 0.995 / 1.0 | 0.967 |
| A2 (one account, domain-sequential) | 60 | 0.322 / 0.917 / 0.930 | 0.0 |
| A5 (6 specialist accounts) | 58 | 0.327 / 0.917 / 0.930 | 0.0 |
| A6 (6 Sybil accounts, interleaved) | 60 | 0.950 / 0.997 / 1.0 | 0.983 |
| A7 (one account, external sources) | 60 | 0.935 / 0.954 / 0.988 | 0.65 |

The null has a gap: its 39 lowest scores (at most 0.0121) are all generalist windows, and the next score (0.0217) is already in the specialist bulk. A p-value of about 0.935 therefore means "above every generalist window, below every specialist window", and A1 and the pooled control sit at or below it.

On held-out benign windows, Spearman's correlation between window score and top-domain share is 0.47. The only attack windows that leave the 0.9 band are GSM8K windows (A2 windows 16–30: p 0.322–0.422). GSM8K's reasoning prompts sit far from the pooled reference, but not far enough to accumulate evidence.

The result does not transfer from the source paper. Liu et al. use 1,500-query windows and a 95th-percentile threshold over benign-vs-benign reference batches, and their attacker and normal sets mostly come from different corpora [1]. This harness scores 100-request per-account windows against a null made of heterogeneous single-user windows.

### F2. The accumulator has no evidence to accumulate (high, demonstrated arithmetic)

- `packages/harness/src/harness/accumulators.py:36-37`: r = max(0, 0.9r − ln p − 1).
- `results/arithmetic.json`, using the frozen thresholds (2.83, 4.85, 8.10, 9.53):

| Sustained p | Increment | Windows to L1 / L2 / L3 / L4 |
|---:|---:|---|
| 0.0017 (floor) | 5.39 | 1 / 1 / 2 / 2 |
| 0.02 | 2.91 | 1 / 2 / 4 / 4 |
| 0.05 | 2.00 | 2 / 3 / 5 / 7 |
| 0.10 | 1.30 | 3 / 5 / 10 / 13 |
| 0.20 | 0.61 | 6 / 16 / never / never |
| ≥ 0.37 | ≤ 0 | never |

- Consequence: at p = 0.1, which is already the 10th percentile of benign windows, L4 arrives after 1,300 requests, later than the quota (1,215). Sustained p below approximately 0.0975 can block by request 1,200; the observed attack windows do not provide that evidence. A1's p of 0.95–1.0 gives r = 0. A2's GSM8K p of 0.32–0.42 gives increments of at most 0.14, which decay away.
- L2 (1-second spacing) and L3 (20-bit SHA-256 proof, `apps/harness-api/src/harness_api/settings.py:36-38`) add little cost at this budget. 6,000 × 1 s is 1.7 h, inside the 24 h deadline. Only L4 withholds output.

### F3. The gateway study dropped the existing pre-matrix detectability gate (high, demonstrated by code reading)

- `packages/distillation/src/distillation/study.py:426-536` (`detectability`, `require_signal`) stops a separate-task study when no MMD arm escalates a single extractor beyond held-out benign accounts, and offers `--allow-undetected` to run it as a quota comparison. `gateway_study.py` never calls either.
- v1 (`readings/v1-2026-10-17-start-v2`) had already shown the single-account extractor at L0 under `full` and `instantaneous`.
- Consequence: about 15.5 h of protected A1 campaigns and their utility runs (10-29 13:27 to 10-30 04:59 UTC) ran when an offline replay would have predicted the result in minutes. The new mixture-control warning, applied to the frozen calibration, also flags this before any campaign.

### F4. Calibration roles are topic-truncated by the workload generator (high, demonstrated bug)

- `workloads.py:236-246, 280-291`: each domain pool is sorted by topic category, and accounts consume it through one cursor from the start. When a role's pool is larger than its demand, only the first categories are used.
- Category labels from `data/tasks/*-split-v2-s42/legitimate-references.json`, keyed by example id, cover every used row:

| Role | AG News used / pool | AG News category mix used | SQuAD titles used / pool |
|---|---|---|---|
| fitting | 6,251 / ~6,012 | balanced | 20 / 23 |
| score_calibration | 12,507 / 14,337 | World 12.9% | 43 / 68 |
| policy_calibration | 10,446 / 40,748 | **100% Business** | 56 / 215 |
| benign_test | 10,939 / 10,950 | balanced | 32 / 50 |

The local build differs slightly from the lab build: 5,877 of 6,251 fitting AG News ids are in the local pool. The labels belong to the source rows, so the mix is still exact.

- Effect: `policy_calibration` windows are more anomalous than held-out windows, with median p 0.367 against 0.554, and 26% against 14% at p ≤ 0.10 (`results/role-shift.json`). The frozen thresholds were therefore fitted on unrepresentative, more concentrated traffic, which makes them conservative. Shuffled `policy_calibration` AG News windows score 0.0106 against same-domain references, approaching A7's BBC News (0.0152), while every other role scores about 0. This does not change F1's direction, because the attacks sit below the whole null.
- Fixing it changes the frozen workloads, so it needs new workloads, a new calibration, and a new run directory. `write_requests` would reject a changed inventory on resume.

### F5. The live benign utility arm cannot measure false positives (high, demonstrated)

- `utility/undefended.json`: 17,079 of 24,047 benign requests time out at the 120 s client deadline even with no defense (completion 0.290). All accounts start together at concurrency 8, and the utility run follows the attack campaign rather than overlapping it.
- Only 15 benign accounts per arm complete a scored window. "One benign account escalated to L1" means 1 of 15.
- Use the 100-horizon replay (`full.json`, `quota.json`) for false-positive rates. My offline replay of the frozen full policy on `benign_test` matches it: 12 accounts at L1 or higher, 0 at L4.

### F6. Manuscript text states results that the data contradicts or has not produced (high, demonstrated by reading)

`paper/main.tex` contains result narratives written before the results, with only the numbers left as `\tbd{}`:

- **Line 28** (abstract): "It refused \tbd{rate} of legitimate specialists". Held-out replay shows 0 benign accounts at L4 under `full`. "The quota … never fired on the Sybil split" holds for A6 but not for A5, where the quota blocks 2 of 6 accounts.
- **Line 53**: "the policy blocked it [the domain-sequential campaign] within its first block". Contradicted. The interrupted A2 stayed at L0 for 15 windows, and replay of all 60 A2 windows never leaves L0.
- **Line 54**: Sybil "Every account released \tbd{count} responses before its block". The replay predicts no block for A5 or A6 under `full`.
- **Line 57**: "Legitimate specialists carried most of the cost, with \tbd{rate} refused under the full policy". Specialists carry every alert (12 of 12 L1 reaches), but none are refused.
- **Line 272**: "A detector that misses A7 is not working." On this reading, the configured detector is not working: offline, A7 stays at L0, with p 0.935–0.988. Live A7 is still pending.

Completed results are only those in `readings/README.md` and the v3 README. Everything under `\tbd{}` or `\pending` is a placeholder. The v4 Kaggle student has no held-out evaluation.

### F7. Concerns, not demonstrated

- **Truncation.** 1,733 of 101,024 distinct detector texts exceed 512 BGE tokens; calibration logged 143 of its 69,171. Most are external A7 and long SQuAD or coding prompts. SQuAD serialization puts the question first, which limits the effect. Not quantified.
- **Dependence.** The null uses overlapping windows (stride 25; 598 scores, 199 independent), and source documents are shared within a role. Clopper-Pearson bounds over accounts below are therefore optimistic.
- **Restarts.** Harness state lives in memory (`policy.py:95-106`). A restart mid-campaign would silently reset accounts. The v3 README handles this by archiving the interrupted A2 and rerunning it. I could not check the live rerun for restarts.
- **Citation concern resolved on 2026-10-31.** `liu2026lifecycle`, arXiv:2610.00839, resolves to the stated Liu et al. lifecycle benchmark. See `SOURCES-VERIFIED.md`.

## A1 and A2 in plain language

- **Calibration.** The harness learned "normal" from 125 synthetic benign accounts, about 90% of which stick to one task and read through one topic at a time. Each 100-request window from those accounts was compared with 20 random mixed batches of benign traffic. The most topic-focused windows produced the largest differences, and those set the alarm levels. The resolution gate passed because there were enough windows to resolve a 1% target. It says nothing about whether an attacker looks different.
- **A1.** It asks a balanced, random mix of all four tasks. Each of its windows looks like a slice of everyone's traffic, closer to the reference than nearly any single user's window, so its p-values are 0.95–1.0. The risk statistic subtracts 1 per window and never rises above 0. All 6,000 responses (588,653 output tokens) were released.
- **A2.** It asks 1,500 AG News prompts, then GSM8K, coding and SQuAD. Random AG News, coding or SQuAD windows are broader than a benign specialist's topic-ordered windows, giving p ≈ 0.9. GSM8K windows reach p ≈ 0.33–0.42, still above the 0.37 break-even point, or barely below it. The statistic stays at about 0, so A2 is expected to stay at L0 and release all 6,000 responses.

## Candidate solutions compared

Thresholds were fitted on `policy_calibration` scored horizons at the frozen targets, then evaluated once on held-out `benign_test` (100 scored accounts: 91 specialists, 9 generalists) and the five seed-42 schedules. "Released" counts responses served before the first refusal, as total responses / AG News responses. Each attack is one campaign at one seed, so attack rows have no interval. The two refit MMD variants were defined after inspecting attack outcomes, so their rows are **exploratory, not a selected method**. Attack token totals in the offline table reuse per-prompt v2 undefended generations, not actual v3 protected outputs; for example, offline A1 totals 588,239 tokens versus 588,653 measured live. The token quota uses estimated benign token costs (each domain's mean attacker output tokens: AG News 2.4, SQuAD 7.3, coding 136, GSM8K 246).

| Defense | Benign L4 (of 100) | Benign L1+ | Generalist L1+ (of 9) | Benign requests refused | A1 | A2 | A5 | A6 | A7 |
|---|---:|---:|---:|---:|---|---|---|---|---|
| Full (frozen) | 0 (≤3.0%) | 12 | 0 | 0 | 6000/1500 | 6000/1500 | 6000/1500 | 6000/1500 | 6000/1500 |
| Instantaneous (frozen) | 2 (≤6.2%) | 12 | 0 | 13 | 6000/1500 | 6000/1500 | 6000/1500 | 6000/1500 | 6000/1500 |
| Request quota 1,215 (frozen) | 1 (≤4.7%) | 1 | 0 | 617 | 1215/312 | 1215/1215 | 5430/1500 | 6000/1500 | 1215/312 |
| Full + request quota | 1 (≤4.7%) | 13 | 0 | 617 | 1215/312 | 1215/1215 | 5430/1500 | 6000/1500 | 1215/312 |
| Token quota ≈44.2k (est., refit) | 4 (≤8.9%) | 4 | 0 | 1328 | 449/104 | 1665/1500 | 3835/1500 | 2706/659 | 430/101 |
| Lower-tail MMD (exploratory) | 3 (≤7.6%) | 20 | 9 | 875 | 199/45 | 499/499 | 3895/998 | 1194/301 | 399/92 |
| Two-sided MMD (exploratory) | 0 (≤3.0%) | 12 | 3 | 0 | 299/68 | 4399/1500 | 6000/1500 | 1694/403 | 599/138 |

**Rate baseline.** The calibrated cutoff of 601 requests per minute is set by batch clients at 0.1 s spacing. Live A1 peaked at 38 requests per minute under `full` and `instantaneous`, so a rate rule never fires on these campaigns. The quota arm's peak of 2,855 is fast refusals, not served traffic.

**Latency and compute.** Live `service_and_queue_seconds`, p50 / p95: undefended 1.59 / 20.1 s, full 1.39 / 23.3 s, instantaneous 1.40 / 23.6 s, quota 1.35 / 22.3 s. These are load-limited (F5), so they cannot attribute overhead to the detector. The detector's per-request cost is one CPU BGE encode, about 55 texts/s when batched and slower one at a time, plus a negligible 20-reference MMD per completed window. This is not separately measured here.

Upper bounds are one-sided 95% Clopper-Pearson over accounts. They are optimistic because sources are shared. With 9 scored generalists, the generalist L4 bound for every defense is at least 28%, so generalist cost is unresolved.

### Order-only evasion (`results/evasion.json`)

| A1 prompts, one account | Upper-tail p, median | Lower-tail p, median | Full | Two-sided | Lower tail |
|---|---:|---:|---|---|---|
| As logged (interleaved) | 0.995 | 0.007 | 6000 released | 299 | 199 |
| Domain-blocked, topic-sorted (k = 8) | 0.318 | 0.684 | 6000 (L1) | 6000 | 6000 (L2) |
| Domain-blocked, topic-sorted (k = 30) | 0.329 | 0.673 | 6000 (L1) | 6000 | 6000 (L1) |
| Two domains per window | 0.935 | 0.067 | 6000 | 499 | 199 |

These schedules evade the tested fixed-window MMD policies. An order-invariant statistic over all of an account's distinct prompts would not be fooled by order. But what it detects reduces to broad coverage at high volume. Benign generalists share that profile at lower volume, and A6 removes the volume by splitting across accounts.

### Larger windows and domain-conditional references (`results/probes.json`, `results/probe-specialists.json`)

- **Liu-style pooled i.i.d. null at 100, 500 and 1,000 requests.** A1, A7 and a benign 25/25/25/25 control all get p = 0.005 (the floor). This configuration detects domain-mix differences and would flag any generalist whose mix differs from the pool. Window size does not help.
- **Domain-conditional references, compared with benign specialist account windows** (median scores):
  - Coding: A7 0.0204 vs specialists 0.0010 (95th percentile 0.0037). Separable.
  - GSM8K: A7 0.0232 vs specialists 0.0002. Separable, but only 3 specialist windows.
  - AG News: A7 0.0153 vs specialists 0.0122; 34% of specialist windows exceed A7's median.
  - SQuAD: A7 0.0277 vs specialists 0.0777. Not separable.
  - A1 sits near 0 in every domain.

### Literature

- **Established.** Per-client query-distribution detectors are defeated by splitting queries across clients. PRADA's authors state this Sybil limitation [3]. Detection that relies on out-of-distribution queries assumes attackers send them; Kariyappa and Qureshi build on exactly that observation [4]. Liu et al. list semantically close queries, mixed low-fraction traffic, and drift as limitations, leave account-aware monitoring to future work, and evaluate mostly different-source pairs [1]. Their reported Model-Leeching/SQuAD case uses template-wrapped prompts; I have not established an exhaustive classification of every experimental pair.
- **Industry practice, as described by the provider.** Anthropic describes detecting large distillation campaigns run through tens of thousands of coordinated accounts. It relies on classifiers and behavioral fingerprinting over account clusters, identity verification, and output-side countermeasures, not per-account content tests alone [5]. The original provider post was verified on 2026-10-31. Its output safeguards were described as under development; this is provider evidence, not a TRACE validation.
- **Research-stage output defenses.** Antidistillation sampling modifies sampling to degrade students while preserving teacher utility [6]. It requires control of sampling and proxy-model gradient/forward computation, approximately doubling generation cost in the reported setup. It is untested here and cannot be applied as a generic wrapper around an opaque teacher API.
- **Sequential testing.** Ville-style e-process thresholds give anytime-valid false-alarm control under valid conditional p-values [7]. They cannot create a signal that the window score lacks.

## Recommendations, in priority order

### Without recalibration or a new study

1. **Report the frozen MMD arms as non-detecting for every campaign in this study, and treat the study as a quota comparison.** For A1 this is demonstrated live. For A2 it is demonstrated on the interrupted run and by exact replay. For A5, A6 and A7 it is an offline prediction from the exact-replay code path. The live arms add confirmation only.
   - Evidence for: exact reproduction plus the per-window distributions.
   - Evidence against: live A5, A6 and A7 are not observed.
   - Acceptance: the live decision logs match the offline per-window p-values in `results/compare.json` (`attack_window_p_upper`) to within float tolerance.
2. **Correct the manuscript** at lines 28, 53, 54, 57 and 272 before any number is filled in. Rewrite the A7 control sentence so it states the observed outcome.
3. **Use the new calibration warning** (implemented below) so that a future calibration states the blind spot at calibration time.
4. **Measure benign cost by replay, not the live utility arm**, or redesign the utility arm with arrival staggering and per-account pacing so that most accounts complete their windows.

### With a fresh calibration in a new run directory

5. **Fix workload topic truncation (F4).** Draw specialist topic blocks from a category-balanced order, so each role covers its pool's mix, and record per-role category coverage in the manifest. Acceptance: every role's category mix within ±3 percentage points of its pool.
6. **Run the detectability gate for the gateway** (port `require_signal`) on the development campaign and on a same-distribution A1 replay before any protected collection.
7. **Optional: add a domain-conditional MMD arm as an A7-type (foreign-source) detector.** Calibrate it on per-account specialist windows per domain, and select its settings on the development campaign only.
   - Evidence for: coding and GSM8K separation above.
   - Evidence against: AG News and SQuAD overlap; GSM8K has a tiny null; there is no development-set selection yet.
   - Acceptance: on held-out benign data with at least 150 scored accounts per profile, L4 upper95 ≤ 2% for both specialists and generalists, and A7 blocked before 500 responses in each separable domain.

### New research or data required

8. **When benign and attack accounts have the same joint distribution over all observable content, any content-only detector has equal rejection probability for both.** That is an identifiability limit, not a claim that the finite A1/A6 samples are literally identical to benign samples. Here, replay and evasion show that the available score does not reliably separate them. The measurements support a layered alternative:
   - per-identity volume budgets: request or capability-weighted, per horizon, with no reset loophole across days;
   - identity assurance that raises the cost of each additional account;
   - cross-account linkage (payment, device and network fingerprints, timing correlation) aggregated at the cluster level;
   - aggregate monitoring of released supervision per capability;
   - post hoc attribution (watermarking);
   - research on output-side degradation [6].

   None of these is measured here except the request and token quotas. The request quota releases all of the six-account A6 campaign; the estimated token quota restricts it to 2,706 requests. Both remain vulnerable to further account rotation. At this budget an evenly partitioned campaign needs at least ceil(6,000 / 1,215) = 5 accounts under the request quota; balancing the recorded total token cost would require roughly 14 under the estimated token quota. This is an acquisition-budget calculation, not a validated adaptive campaign.
9. **Choose budgets from learning curves.** For AG News, the only domain with qualified transfer (+20.0 pp from 500 development responses, 95% CI 16.4–23.8), the request quota still releases 312 (A1), 1,215 (A2) or 1,500 (A5, A6) AG News responses. The A2, A5 and A6 totals exceed the 500 responses used in the development gain measurement; A1 releases fewer. This does not prove equivalent transfer because prompt selection and training vary. Whether 312 responses already give most of the gain is unknown until a learning curve exists.

## Code changes

Commit `46d68412fad5074abf6fb2c7d1162bab15682b73`, change `zxvoruwz` ("harness: warn when pooled benign traffic ranks below the account null"), on top of `e53503f`, is the parent of the isolated `mmd-audit` workspace. A fresh `jj git fetch` on 2026-10-31 confirms it is already an ancestor of `main@origin` at `43464d2e`. This establishes publication, not deployment on the lab machine:

- `packages/harness/src/harness/calibration.py`:
  - new `MixtureControl` model and `mixture_control()`, which shuffles score-calibration requests across accounts and ranks up to 50 pooled windows in the per-account null;
  - an optional `Calibration.mixture_control` field, defaulting to `None` so legacy artifacts load unchanged;
  - computed during `calibrate()`;
  - `adequacy()` returns a `detectability_warnings` list when at least 90% of the pooled windows rank below the null median. The warning never changes `passed` or `status`.
- `packages/distillation/src/distillation/study.py`: `require_calibration` logs detectability warnings.
- `tests/test_harness.py`, three cases:
  - exchangeable accounts give no warning;
  - topic-clustered accounts warn, without changing the gate verdict;
  - a legacy artifact without the field loads and reports no warning.
- `docs/harness.md`: describes the control and its reading on the frozen calibration.
- Verification:
  - `uv run pytest`: 253 passed, 5 skipped. ruff, ruff format and mypy are clean.
  - Applied to the frozen `gateway-bounded-v1` calibration, the control places 100% of 50 pooled windows below the null median (p 0.935 to 0.950, median 0.935) and emits the warning, while the gate stays `adequate` (`results/mixture-control-frozen.json`).
- Safety: frozen calibrations are reused without recalibration, and `calibration-adequacy.json` is written with `save_json`, not `freeze`. A lab run that pulls this change resumes unchanged, apart from one extra key in that file and one log line. The warning is committed and present in remote main; live lab activation is unverified.

## Reproducing the analysis

Requirements: about 4 GB of free disk; CPU only; about 35 min of BGE embedding per variant on 12 threads. The cached BGE weights must be present, because `run.sh` sets `HF_HUB_OFFLINE=1`.

```
cd outputs/mmd-audit
direnv exec ../.. uv venv .venv
direnv exec ../.. uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match "torch>=2.14.1,<3" "transformers>=5.19.0,<6" scipy scikit-learn -e ../../packages/harness
./run.sh embed.py detector          # ~35 min, cache/detector.npz (101,024 distinct texts)
./run.sh reproduce.py               # calibration rebuild + exact A1/A2 replay -> results/reproduce.json
./run.sh diagnose.py detector       # per-window distributions by profile -> results/diagnose-detector.json
./run.sh arithmetic.py              # accumulator windows-to-level -> results/arithmetic.json
./run.sh volume.py                  # request/token quota release and benign cost -> results/volume.json
./run.sh compare.py                 # defense comparison -> results/compare.json
./run.sh evasion.py                 # order-only evasion -> results/evasion.json
./run.sh probes.py                  # window size and domain-conditional -> results/probes.json
./run.sh probe_specialists.py       # A7 vs specialist windows -> results/probe-specialists.json
./run.sh embed.py noprefix && ./run.sh diagnose.py noprefix   # domain-prefix ablation
```

Data splits:
- Benign roles are source-disjoint by construction (`validate_separation`).
- Selection: `policy_calibration` for thresholds and limits, plus the development campaign for any new variant (not yet done).
- Evaluation: `benign_test` and the attack schedules, used once.

Remaining validation:
1. **Development-only selection** for any new detector, using the development ledger in `readings/v2-…/ledgers/development` (2,000 prompts; about 1 min to embed), then evaluation on newly reserved source-disjoint benign accounts and attacks. The present `benign_test` and attack results have already been inspected and cannot now count as untouched confirmatory tests.
2. **AG News learning curve.** n ∈ {50, 100, 200, 300, 500, 1000, 1500} using the frozen recipe (lr 1e-4, 3 epochs) and 3 training seeds: development-ledger responses for n ≤ 500, and the A1/s42 undefended AG News responses above that, scored on the 1,000-item development suite. The v2 sweep trained six students on 1,607 examples and evaluated four domains in about 18 h on the RTX 4000 Ada. Twenty-one single-domain runs should take an estimated 4–7 GPU hours. This is a projection, not measured.
3. **Larger benign populations.** To resolve a 2% L4 bound per profile, at least 150 scored accounts per profile are needed (zero events gives an upper95 of about 2%), and generalists in particular.
4. **Wait for the live A5, A6 and A7 arms** and compare their logs with the offline predictions.

## Still unknown or blocked

- The lab machine's live state, its code revision, and whether the A2 rerun was ever restarted.
- Live A5, A6 and A7 outcomes, which are offline predictions only.
- Downstream student capability for any defended or quota-limited dataset (no GPU here).
- Real benign traffic. Every benign result comes from one synthetic generator with a disclosed topic-truncation bug.
- Benign token costs, which are estimates.
- (Resolved) The no-domain-prefix ablation, with references, sigma and null rebuilt from the same rows (`results/diagnose-noprefix.json`), leaves every conclusion unchanged: A1 0.950–1.0, A6 0.952–1.0, A7 0.935–0.997, benign specialists median 0.539, uniform-mixture probe 0.935–0.997. GSM8K windows become less detectable (A2 minimum p 0.427 instead of 0.322). The domain prefix is not the cause.

## Recovery verification on 2026-10-31

The isolated audit checkout remains pinned to the warning commit `46d68412`; it was not rebased onto the newer main. `jj git fetch` reported no changes and confirmed `46d68412` in the ancestry of `main@origin` at `43464d2e`. No local Python, harness, teacher or GPU experiment process was observed at recovery. The remote lab process remains unverified; the remote environment-link API is unavailable in this tool session. No service was started, stopped or contacted.

Fresh outputs are in `verified-20261031/`, with read-only reuse of the original embedding cache and no embedding or GPU job. The calibration rebuild and all 135 logged-window comparisons match the previous `reproduce.json` exactly. The volume, policy-comparison and order-evasion results were rerun from the frozen role files and match all three earlier JSON outputs exactly. See `verification-manifest.json` for input hashes and checks. Tests were rerun at the audit commit: 253 passed, 5 skipped in 45.81 seconds; the focused harness/gateway subset passed 37 tests. Ruff passes on all three changed Python files. No new Python implementation was added during recovery. The audit summary added to `docs/harness.md` is local and uncommitted; the earlier warning implementation is already published. The report, scripts, cached embeddings and diagnostic results live in ignored `outputs/` and are local artifacts.

For another CPU-only replay using the existing cache, run from the fresh output directory, substituting the workspace path as needed:

```bash
cd /home/rajveer/Projects/personal/capstone-mmd-audit/outputs/mmd-audit-verified-20261031
direnv exec ../.. env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 /home/rajveer/Projects/personal/capstone-mmd-audit/outputs/mmd-audit/.venv/bin/python reproduce.py
direnv exec ../.. env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 /home/rajveer/Projects/personal/capstone-mmd-audit/outputs/mmd-audit/.venv/bin/python volume.py
direnv exec ../.. env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 /home/rajveer/Projects/personal/capstone-mmd-audit/outputs/mmd-audit/.venv/bin/python compare.py
direnv exec ../.. env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 /home/rajveer/Projects/personal/capstone-mmd-audit/outputs/mmd-audit/.venv/bin/python evasion.py
```

This repeats deterministic diagnostics; it does not provide a second independent test population. At recovery, the original paper citations were checked against primary sources in `SOURCES-VERIFIED.md`. The lifecycle citation is valid. Antidistillation sampling needs teacher sampling control and proxy computation; it is not an available opaque-API configuration fix. The narrow HTML acquisition comparison was rendered inline, but visual preview could not run because this host lacks the browser's shared libraries.

## Sources

1. Liu, Guo, Dong. An Embarrassingly Simple Detector for Model Extraction Attacks in Large Language Model API Traffic. arXiv:2606.05725 (EMNLP 2026). https://arxiv.org/abs/2606.05725 (method details from the HTML version, https://arxiv.org/html/2606.05725)
2. Gretton et al. A Kernel Two-Sample Test. JMLR 13, 2012. https://www.jmlr.org/papers/v13/gretton12a.html
3. Juuti, Szyller, Marchal, Asokan. PRADA: Protecting Against DNN Model Stealing Attacks. EuroS&P 2019. https://arxiv.org/abs/1805.02628
4. Kariyappa, Qureshi. Defending Against Model Stealing Attacks With Adaptive Misinformation. CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/Kariyappa_Defending_Against_Model_Stealing_Attacks_With_Adaptive_Misinformation_CVPR_2020_paper.html
5. Anthropic. Detecting and preventing distillation attacks. 2026-02-23. https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks (original verified on 2026-10-31)
6. Savani et al. Antidistillation Sampling. arXiv:2504.13146. https://arxiv.org/abs/2504.13146
7. Ramdas, Grünwald, Vovk, Shafer. Game-Theoretic Statistics and Safe Anytime-Valid Inference. Statistical Science 38(4), 2023. https://projecteuclid.org/journals/statistical-science/volume-38/issue-4/Game-Theoretic-Statistics-and-Safe-Anytime-Valid-Inference/10.1214/23-STS894.full
