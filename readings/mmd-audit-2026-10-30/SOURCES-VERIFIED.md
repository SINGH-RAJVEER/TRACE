# Primary-source verification

Verified on 2026-10-31 against primary publisher, author-preprint, and provider pages. This note checks the literature claims in `REPORT.md`; it does not revalidate local measurements or change code. No requested citation was misidentified. The lifecycle citation that the report could not verify now resolves correctly.

## Liu, Guo and Dong MMD detector

[arXiv record, v3](https://arxiv.org/abs/2606.05725v3) and [full text](https://arxiv.org/html/2606.05725v3).

The title, authors and identifier match source 1. The record states acceptance to EMNLP 2026. Section 4.1 specifies 1,500-query windows, 1,000 benign calibration samples, a 95th-percentile threshold, and 20 reference repeats. Sections 3.3 and 3.4 define an upper-tail test against benign references. These support the report's protocol comparison.

The limitations section explicitly discusses semantically close queries, low attacker fractions, stronger adaptive distribution matching, contamination, drift, and the offline query-text-only scope. Account-aware monitoring remains future work. These support qualified transfer limitations, not a universal impossibility theorem.

Appendix C Table 6 supports the broad observation that attack and normal traffic often use different sources. The statement that Model-Leeching/SQuAD is the *only* same-source pair needs care: source families and actual query construction differ. Cite the complete source table rather than imply that the paper isolates corpus identity as a controlled causal factor.

## PRADA

[Full text, v5](https://arxiv.org/html/1805.02628v5), section V-C.

The authors explicitly state that distributing queries among enough clients circumvents PRADA because it analyzes one client's queries. This directly supports the report's attribution of a Sybil limitation to PRADA. It establishes that limitation for PRADA, not a measured failure of every detector with account features. Generalizing to TRACE requires TRACE's own account-state logic and campaign measurements.

## Adaptive misinformation

[CVPR 2020 paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kariyappa_Defending_Against_Model_Stealing_Attacks_With_Adaptive_Misinformation_CVPR_2020_paper.pdf) and [author preprint](https://arxiv.org/abs/1911.07100).

The paper proposes returning misinformation selectively on OOD inputs and motivates that defense from OOD queries in the attacks studied. It supports the report's statement that this defense depends on an OOD separation assumption. Its claims concern the attacks and image-classification setting it evaluates, not proof that all contemporary LLM extraction is OOD or that an OOD detector can identify same-distribution distillation.

## Anthropic's February account of campaigns

[Original post, 2026-02-23](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks).

The original is accessible. Replace the report's secondary-summary caveat with primary-source verification. Anthropic reports over 16 million exchanges through approximately 24,000 fraudulent accounts. Its response section describes classifiers, behavioral fingerprints, coordinated-account detection, stronger account verification, and development of product/API/model countermeasures to reduce output usefulness for distillation. It also describes IP, metadata, infrastructure and payment/timing links.

These are provider-reported observations and defenses. The post does not provide an independently reproducible benchmark, benign false-positive rate, or proof that the layered system prevents arbitrary same-distribution extraction. In particular, the February post describes output countermeasures as being developed; avoid claiming demonstrated effectiveness from that post alone.

## Newer Anthropic evidence

[Original post, 2026-10-12](https://www.anthropic.com/news/distillation-defenses).

This newer primary account reports early estimates of 90 to 99 percent reduction in harvested data. It describes reasoning summarization and protections against moving protected reasoning between contexts, models and accounts. A reported SWE-bench Verified experiment gives 80.4 percent for a student trained on full reasoning transcripts versus 71.8 percent on summarized transcripts.

This is relevant evidence for limiting released supervision, but it is a provider report about a different model, threat and output format. It does not validate a TRACE intervention or establish the same utility trade-off for classification labels or short answers. Report those numbers only with that attribution and scope.

## Antidistillation sampling

[arXiv record, v5](https://arxiv.org/abs/2504.13146v5) and [full text](https://arxiv.org/html/2504.13146v5), sections 3 and 4.

The title, identifier and Savani attribution match source 6. The method adjusts the teacher's next-token distribution using a proxy student's loss gradient. It measures utility versus distillability on reasoning tasks, including GSM8K, MATH and MMLU, and tests different model families. The report's description as a research-stage output defense is supported.

Practical limits deserve emphasis. This requires teacher sampling access and proxy-model computation; it is not implementable as a generic prompt wrapper over an opaque teacher API. The paper's setup uses two proxy forward passes per teacher token and estimates about twice the sampling compute when the proxy is half the teacher's size. Teacher utility is a measured trade-off rather than a universal no-cost guarantee. TRACE has not measured it.

## MMD foundation and sequential validity

[Gretton et al., JMLR 2012](https://www.jmlr.org/papers/v13/gretton12a.html) correctly identifies source 2 and supports MMD as a two-sample distribution test. It does not identify malicious intent when the observable distributions coincide.

[Ramdas et al., author preprint](https://arxiv.org/abs/2210.01948) correctly identifies source 7, with a 2023 revised version. The abstract supports anytime-valid e-processes and confidence sequences under their defining probabilistic assumptions. TRACE's empirical ranks and leaky accumulator are not thereby anytime-valid. A proposed p-to-e sequential construction needs conditional validity or another justified null construction, not merely marginal uniformity or a larger calibration set.

## Previously unverified lifecycle citation

[arXiv record, v2](https://arxiv.org/abs/2610.00839v2).

The identifier resolves to *Do Defenses Against LLM Extraction Work Across Attacks? A Lifecycle Benchmark of Black-Box Model Extraction*, by Shuze Liu and colleagues, revised 2026-10-07. It describes six extraction attacks, ten defenses, and two adaptive attacks using paraphrasing/back-translation of protected responses. Remove the report's F7 claim that this reference remains unverified. This check establishes bibliographic identity and abstract scope only; no particular result table from that paper was audited here.

## Wording to preserve scientific scope

The claim that identical observable traffic distributions cannot support a distinguishing test is a statistical inference: if benign and malicious observations have the same law, every decision rule has equal flag probability under both labels. State the equality assumption explicitly. Finite TRACE measurements show failures of the tested rules and schedules; they cannot prove that every possible account or content signal has identical laws. Separately, PRADA documents its specific Sybil weakness, and Anthropic reports signals beyond per-account text. Neither source removes the need for held-out TRACE validation.
