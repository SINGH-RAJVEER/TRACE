# Papers reviewed for distillation resistance

Research conducted 2026-11-02 to 2026-11-03. All 18 PDFs are saved in [`papers/`](./papers). Their PDF signatures, page counts and first-page text were checked; SHA-256 and byte sizes are recorded in [PAPER_MANIFEST.json](./PAPER_MANIFEST.json). A saved paper is evidence reviewed, not a locally reproduced result. The recommendation and experiment are in [ANTI_DISTILLATION_RESEARCH.md](./ANTI_DISTILLATION_RESEARCH.md).

Dates below identify the saved revision or proceedings record. “Venue not established” describes this review, not a claim that no publication exists. For unversioned download URLs, the inspected PDF revision and recorded hash identify the actual file.

## 1. PRADA: Protecting against DNN Model Stealing Attacks

[Local PDF](./papers/PAPER_DETECT_PRADA_1805.02628v5.pdf) · [Primary record](https://arxiv.org/abs/1805.02628) · [Download source](https://arxiv.org/pdf/1805.02628v5)

Saved version: v5. Date: v5, 2019-03-31; first submitted 2018-05-07. Status: IEEE EuroS&P 2019. 17 pages; 1,437,686 bytes.

Inspected: Methods Section V-A (pp. 10-12), mitigation and Sybil discussion (pp. 13-14).

Caveat: DNN classifier extraction setting; does not establish protection for general LLM traffic.

## 2. Stateful Detection of Model Extraction Attacks

[Local PDF](./papers/PAPER_DETECT_VarDetect_2107.05166v1.pdf) · [Primary record](https://arxiv.org/abs/2107.05166) · [Download source](https://arxiv.org/pdf/2107.05166v1)

Saved version: v1. Date: v1, 2021-07-12. Status: arXiv preprint; no peer-reviewed venue verified. 16 pages; 1,512,246 bytes.

Inspected: Threat model, VAE/MMD algorithm, Section 7.3 adaptive attacks (pp. 8-10), including explicit 5% dilution miss.

Caveat: Its reported dilution miss is specific to the evaluated classifier setting.

## 3. An Embarrassingly Simple Detector for Model Extraction Attacks in Large Language Model API Traffic

[Local PDF](./papers/PAPER_DETECT_LLM_MMD_2606.05725v3.pdf) · [Primary record](https://arxiv.org/abs/2606.05725) · [Download source](https://arxiv.org/pdf/2606.05725v3)

Saved version: v3. Date: v3, 2026-08-29; first submitted 2026-06-04. Status: EMNLP 2026 Main; publication record verified. 19 pages; 522,974 bytes.

Inspected: Methods Sections 2-3; experiment settings 4.1; adaptive and contamination analyses 4.5-4.6; limitations; original-protocol comparison 4.7; relevant full PDF text and arXiv HTML inspected.

Caveat: Reported error rates are offline window results, not online pre-transfer stopping guarantees.

[Publication record](https://aclanthology.org/2026.emnlp-main.748/).

## 4. Nonparametric Two-Sample Testing by Betting

[Local PDF](./papers/PAPER_DETECT_Sequential_Two_Sample_Betting_2112.09162.pdf) · [Primary record](https://arxiv.org/abs/2112.09162) · [Download source](https://arxiv.org/pdf/2112.09162)

Saved version: v6. Date: v6, 2025-08-22; first submitted 2021-12-16. Status: IEEE Transactions on Information Theory 2023 work; downloaded later arXiv v6 correction. 59 pages; 844,138 bytes.

Inspected: Sections 1.1, 2.1, conditional-expectation assumptions, pair symmetry Remark, Section 5.1. Latest revision fixes theorem proof; theorem applicability not assumed for raw conversational traffic.

Caveat: Conditional fair-bet assumptions require justification; raw dependent chat windows do not inherit the theorem.

## 5. Retrain or not retrain: conformal test martingales for change-point detection

[Local PDF](./papers/PAPER_DETECT_Conformal_Test_Martingales_Vovk_2021.pdf) · [Primary record](https://proceedings.mlr.press/v152/vovk21b.html) · [Download source](https://proceedings.mlr.press/v152/vovk21b/vovk21b.pdf)

Saved version: PMLR 152 proceedings. Date: PMLR 152:191-210, 2021-09-08 to 2021-09-10 conference. Status: COPA 2021, PMLR proceedings. 20 pages; 1,081,922 bytes.

Inspected: Section 2 definitions, randomized sequential rank p-values, exchangeability assumptions, Proposition 1, Ville inequality (pp. 2-4), calibration example (p. 5).

Caveat: Exchangeability applies to the ranked units; arbitrary overlapping chat windows are not automatically valid units.

## 6. Antidistillation Sampling

[Local PDF](./papers/PAPER_DEFENSE_Antidistillation_Sampling_2504.13146v5.pdf) · [Primary record](https://arxiv.org/abs/2504.13146v5) · [Download source](https://arxiv.org/pdf/2504.13146v5)

Saved version: 2504.13146v5. Date: 2025-10-26. Status: Downloaded arXiv preprint; peer-reviewed venue not established in this review. 22 pages; 2,331,238 bytes.

Inspected: Algorithm 1; Section 4.1; proxy-gradient and decoding-cost discussion.

Caveat: Needs proxy computation and local logits; adaptive learners can exploit preserved reasoning.

## 7. DOGe: Defensive Output Generation for LLM Protection Against Knowledge Distillation

[Local PDF](./papers/PAPER_DEFENSE_DOGe_2505.19504v2.pdf) · [Primary record](https://arxiv.org/abs/2505.19504v2) · [Download source](https://arxiv.org/pdf/2505.19504v2)

Saved version: 2505.19504v2. Date: 2025-10-19. Status: Downloaded arXiv preprint; peer-reviewed venue not established in this review. 22 pages; 2,806,801 bytes.

Inspected: Section 3 and Figure 2; defensive final-head training objective.

Caveat: Requires teacher-head training and a proxy; not a gateway-only output filter.

## 8. Protecting Language Models Against Unauthorized Distillation through Trace Rewriting

[Local PDF](./papers/PAPER_DEFENSE_Trace_Rewriting_2602.15143v2.pdf) · [Primary record](https://arxiv.org/abs/2602.15143v2) · [Download source](https://arxiv.org/pdf/2602.15143v2)

Saved version: 2602.15143v2. Date: 2026-04-16. Status: Downloaded arXiv preprint; peer-reviewed venue not established in this review. 18 pages; 817,703 bytes.

Inspected: Sections 5.1, 6.4; Appendices A.2, B, D.2, E.3, especially Table 5.

Caveat: Small 7B self-rewriter result is math-specific; main setup uses a much larger rewriter; human explanation utility unproven.

## 9. Lossless Anti-Distillation Sampling

[Local PDF](./papers/PAPER_DEFENSE_Lossless_Anti_Distillation_Sampling_2605.18829v2.pdf) · [Primary record](https://arxiv.org/abs/2605.18829v2) · [Download source](https://arxiv.org/pdf/2605.18829v2)

Saved version: 2605.18829v2. Date: 2026-09-24. Status: Downloaded arXiv preprint; peer-reviewed venue not established in this review. 29 pages; 669,484 bytes.

Inspected: Algorithm 2; Theorem 2; Section 4.2/Table 2; Appendix E.

Caveat: Targets aligned redundant sampling; theorem is an upper bound, and Table 2 disagrees with a nearby numerical prose claim.

## 10. DistillGuard: Evaluating Defenses Against LLM Knowledge Distillation

[Local PDF](./papers/PAPER_DEFENSE_DistillGuard_2603.07835v1.pdf) · [Primary record](https://arxiv.org/abs/2603.07835v1) · [Download source](https://arxiv.org/pdf/2603.07835v1)

Saved version: 2603.07835v1. Date: 2026-03-08. Status: Downloaded arXiv preprint; peer-reviewed venue not established in this review. 14 pages; 348,927 bytes.

Inspected: Abstract, threat model and defense comparison experiments.

Caveat: Reported effects depend on task and naive-attacker recipe; removing reasoning does not reliably protect code.

## 11. Watermark Basins: Black-Box Detection of Distilled LLMs

[Local PDF](./papers/PAPER_DEFENSE_Watermark_Basins_2610.37266v1.pdf) · [Primary record](https://arxiv.org/abs/2610.37266v1) · [Download source](https://arxiv.org/pdf/2610.37266v1)

Saved version: 2610.37266v1. Date: 2026-10-27. Status: Downloaded arXiv preprint; peer-reviewed venue not established in this review. 32 pages; 1,943,935 bytes.

Inspected: Section 3; Appendix B.4 on reference-bank and audit costs.

Caveat: Post-training attribution requiring suspect outputs, not prevention of collection.

## 12. Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?

[Local PDF](./papers/PAPER_DEFENSE_Watermark_Removal_2502.11598v2.pdf) · [Primary record](https://arxiv.org/abs/2502.11598v2) · [Download source](https://arxiv.org/pdf/2502.11598v2)

Saved version: 2502.11598v2. Date: 2025-05-24. Status: Downloaded arXiv preprint; peer-reviewed venue not established in this review. 24 pages; 2,090,867 bytes.

Inspected: Methods and experiments on preprocessing and inference-time neutralization.

Caveat: Watermark evidence and actual capability transfer are separate outcomes.

## 13. On the Learnability of Antidistillation Reasoning Traces

[Local PDF](./papers/PAPER_DEFENSE_Learnability_Antidistillation_Traces_EMNLP_2026.pdf) · [Primary record](https://aclanthology.org/2026.findings-emnlp.867/) · [Download source](https://aclanthology.org/2026.findings-emnlp.867.pdf)

Saved version: 2026.findings-emnlp.867.pdf. Date: 2026-10. Status: Findings of EMNLP 2026, publisher PDF. 24 pages; 2,039,526 bytes.

Inspected: Sections 3–5; Appendix Tables 4–6, including OCR reconstruction.

Caveat: Unicode defenses impair machine-readable utility and face rendering/OCR recovery.

## 14. Let Them Steal: Trapping Large Language Model Extraction Attacks with Knowledge Honeypot

[Local PDF](./papers/PAPER_DEFENSE_Knowledge_Honeypot_2606.15810v1.pdf) · [Primary record](https://arxiv.org/abs/2606.15810v1) · [Download source](https://arxiv.org/pdf/2606.15810v1)

Saved version: 2606.15810v1. Date: 2026-06-14. Status: Downloaded arXiv v1; EMNLP 2026 Main acceptance reported by authors department. 16 pages; 731,944 bytes.

Inspected: Sections 2.2–3.3, 4.6; Appendix C.

Caveat: Applies to response-guided entity expansion; fixed-corpus collection need not follow the bait.

[Acceptance status source](https://www.cs.fsu.edu/yushunrailabs/).

## 15. The False Promise of Imitating Proprietary LLMs

[Local PDF](./papers/PAPER_EVAL_2023_False_Promise_2305.15717v1.pdf) · [Primary record](https://arxiv.org/abs/2305.15717v1) · [Download source](https://arxiv.org/pdf/2305.15717v1)

Saved version: 2305.15717v1. Date: 2023-05-25. Status: Downloaded arXiv preprint; peer-reviewed venue not established in this review. 15 pages; 1,051,380 bytes.

Inspected: Full text, especially Sections 2–4.

Caveat: Older model/data findings; not proof that current task-specific distillation fails.

## 16. QLoRA: Efficient Finetuning of Quantized LLMs

[Local PDF](./papers/PAPER_EVAL_2023_QLoRA_2305.14314v1.pdf) · [Primary record](https://arxiv.org/abs/2305.14314v1) · [Download source](https://arxiv.org/pdf/2305.14314v1)

Saved version: 2305.14314v1. Date: 2023-05-23. Status: Downloaded arXiv preprint; peer-reviewed venue not established in this review. 26 pages; 1,065,470 bytes.

Inspected: Full text; introduction, method and limitations.

Caveat: Memory-saving technique; reported hardware results do not certify this GPU or training recipe.

## 17. s1: Simple test-time scaling

[Local PDF](./papers/PAPER_EVAL_2025_s1_2501.19393v3.pdf) · [Primary record](https://arxiv.org/abs/2501.19393v3) · [Download source](https://arxiv.org/pdf/2501.19393v3)

Saved version: 2501.19393v3. Date: 2025-03-01; first submitted 2025-01-31. Status: Downloaded arXiv preprint; peer-reviewed venue not established in this review. 46 pages; 1,480,533 bytes.

Inspected: Full text, especially Sections 1–3.

Caveat: Sample efficiency does not imply the 32B and multi-H100 training recipe fits this machine.

## 18. Do Defenses Against LLM Extraction Work Across Attacks? A Lifecycle Benchmark of Black-Box Model Extraction

[Local PDF](./papers/PAPER_EVAL_2026_Lifecycle_Extraction_Benchmark_2610.00839v2.pdf) · [Primary record](https://arxiv.org/abs/2610.00839v2) · [Download source](https://arxiv.org/pdf/2610.00839v2)

Saved version: 2610.00839v2. Date: 2026-10-07; first submitted 2026-09-30. Status: Downloaded arXiv preprint; peer-reviewed venue not established in this review. 24 pages; 608,189 bytes.

Inspected: Sections 2–4; Appendices B, D, E, especially E.4.

Caveat: Preprint, narrow evaluated models/tasks and 20 attack batches per method; no reproduction here.

## Additional source and access boundary

[LIMO v3](https://arxiv.org/abs/2502.03387v3) was reviewed at abstract/version-history level and is not one of the downloaded full-text papers. An attempted IACR 2022/1039 download returned HTTP 403; no theorem from it supports the report. The bibliography is a focused review through 2026-11-03, not a guarantee of exhaustive field coverage.
