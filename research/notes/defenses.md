# Defenses against distillation beyond query limits

Reviewed 2026-11-03. These notes recover the interrupted defense research and verify the saved PDFs. `defense_manifest.json` records nine PDFs, their exact URLs, versions, SHA-256 values, byte sizes, and page counts. No inference, student training, or model download was performed. All proposed experiments below remain untested in TRACE.

## Which output defenses have credible evidence and a plausible small-hardware implementation?

### Takeaway

The most useful small experiment is targeted reasoning-trace rewriting, paired with answer-only and ordinary-paraphrasing controls. It changes training usefulness instead of imposing an acquisition cap. It is a research candidate, not an established general defense. Lossless Anti-Distillation Sampling is a separate inexpensive candidate specifically for redundant cross-account sampling.

### Cited findings

- **Trace rewriting, April 2026 revision.** The OPT instruction asks the rewriter to express deductions in unusually formal, dense technical language. This is deliberate pedagogical obfuscation, not ordinary paraphrasing. Appendix A.2 contains the exact instruction. Appendix E.3, Table 5 uses DeepSeek-R1-Distill-Qwen-7B as both teacher and rewriter, with Llama-3.2-3B student: teacher accuracy 88.8→88.1%, student 57.7→30.2% on GSM8K. The 27.5-point student difference is not a relative percentage. Appendix B evaluates 150 MATH traces using an LLM judge, not humans; overall quality 4.01→3.83/5. Section 6.4 tests Parrot preprocessing and KPOD, not every adaptive learner. Appendix D.2 uses temperature 0.6, GSM8K generation maximum 1,024 tokens, and four-epoch rank-128 LoRA. Main experiments use a 120B rewriter. Prompt optimization itself uses DeepSeek-V3.1 and repeated proxy training, but the published instruction can be reused. Hardware reported is A100/H100, not the user's GPU. [Paper, §§5.1, 6.4, Appendices A.2/B/D.2/E.3](https://arxiv.org/abs/2602.15143v2).
- **LADS, September 2026 revision.** Algorithm 2 increments the account's count for semantic bucket `i=h(query)`, then derives randomness from a secret keyed function of `(i,count)`. Accounts share randomness when both bucket and count align; each account receives fresh draws along its own sequence. Same seed does not imply identical text for different prompts. Experiments create nine paraphrases per source prompt and use same-bucket seeds with probability 0.7. Table 2 gives Llama-3.1-8B GSM8K 73.39% clean versus 62.72% LADS, a 10.67-point reduction. Nearby prose instead claims an 18.6-point reduction; do not silently repeat that inconsistent claim. The theorem provides uniform-convergence upper bounds under regularity and concentrated-query assumptions. A weaker upper bound is not a lower bound proving every trained student performs worse. [Paper, Algorithm 2, Theorem 2, §4.2/Table 2, Appendix E](https://arxiv.org/abs/2605.18829v2).
- **ADS.** Antidistillation Sampling modifies teacher logits according to a proxy student's estimated downstream-loss change. It leaves teacher weights fixed. A holdout-derived proxy gradient is precomputed, then two proxy forward passes accompany each teacher decoding pass. The paper estimates approximately twice the generation computation when proxy size is half teacher size. It needs local sampling control and a proxy, not merely an API gateway. [Paper, Algorithm 1 and §4.1](https://arxiv.org/abs/2504.13146v5).
- **DOGe.** Train only the teacher's final linear head using a task-preserving supervised loss minus a weighted divergence from a proxy student. This changes teacher parameters and requires defensive training. It is less intrusive than full-model training but still a materially different implementation from request detection. [Paper, §3/Figure 2](https://arxiv.org/abs/2505.19504v2).

### Inferences

- A sequential 7B experiment is plausible: cache clean outputs; reuse the same loaded 7B checkpoint with a rewrite prompt; unload it before student training. This is an engineering proposal, not evidence that quantization or this GPU reproduces Table 5. Quantization, shorter contexts, smaller LoRA ranks, and reduced sample counts each change the experiment and need explicit reporting.
- For a first rewrite prototype, keep the extracted final answer as a separate immutable field and rewrite only its explanation. Check answer equality and task validity before release. This differs from the paper's complete procedure, so first reproduce a small paper-style condition before interpreting the stricter variant.
- Human usefulness cannot be inferred from final-answer accuracy alone. Dense formal explanations may frustrate ordinary users, break educational use, or degrade code readability. A human-rated explanation task is required if explanations are part of the service promise.
- LADS cannot stop a single-account attacker because preserving that account's original sampling law is its design goal. It offers no mechanism against deterministic greedy decoding. Broad unique queries, different bucket occupancy, counter desynchronization, and valid answer-only learning are mandatory challenge conditions. It should be described as reducing redundant sampling gains, not detecting intent.
- Keeping all else fixed, output defenses can be evaluated on both matched query counts and matched *released* tokens. Rewriting may make responses longer. Reporting only queries can misattribute increased exposure or reduced training-token budgets to defense quality.

### Gaps

- No verified throughput or peak-memory measurement exists here for sequential quantized 7B rewriting on the user's card.
- No universal explanation establishes why dense rewriting must inhibit a capable adaptive student. The evidence is empirical and tied to training recipes. Test simplification, answer extraction, selective token weighting, and student-generated explanation reconstruction.
- LADS preservation requires correctly scoped pseudorandom streams, deterministic bucket assignment, atomic counters, and a serving stack where seeds behave as assumed. Batching, retries, replicas, or client-specified seeds may defeat an implementation even when the mathematics is sound.

## What do the newest attacks and defenses change about the recommendation?

### Takeaway

Do not use output oddness as a proxy for low learnability. Recent work shows that learners can ignore disruptive tokens, and that human-readable perturbations can be repaired. Watermarking detects provenance after training; it does not stop acquisition.

### Cited findings

- **Learnability of antidistillation traces, EMNLP Findings October 2026.** The paper argues ADS leaves useful reasoning intact while a sparse high-loss tail destabilizes ordinary training; its offline-RL learner reweights useful reasoning. Its BLADE defense uses proxy hidden-state dependence to select informative spans, then changes Unicode using invisible characters, homoglyphs, and bidirectional controls, with an online bandit selecting operations. Appendix Table 6 demonstrates an important limitation: Qwen2.5-1.5B GSM8K accuracy is 72.48% after clean distillation, 7.50% after BLADE, and 55.31% after OCR reconstruction. MMLU is 54.30%, 1.40%, and 52.40%, respectively. Standard Unicode normalization is weaker than this rendering-and-reconstruction attack. [Paper, §§3–5 and Appendix Tables 4–6](https://aclanthology.org/2026.findings-emnlp.867/).
- **DistillGuard, March 2026.** In its Qwen3-14B→Qwen2.5-7B-Instruct, naive-attacker setting, ordinary paraphrasing barely impairs distillation. Removing chain-of-thought reduces math performance from 67.8% to 31.4%, while code generation remains largely unaffected. These results argue for task-specific controls, not a general output-sanitization guarantee. [Paper, abstract and defense comparisons](https://arxiv.org/abs/2603.07835v1).
- **Watermark removal, May 2025 revision.** Tests targeted preprocessing and inference-time watermark neutralization. Students can retain useful transferred behavior while suppressing inherited statistical-watermark evidence. Watermark detectability and successful capability prevention are different outcomes. [Paper, methods and experiments](https://arxiv.org/abs/2502.11598v2).
- **Watermark Basins, October 27, 2026.** Uses a clean proxy's gradients to compare suspect generations with key-specific reference directions after projecting away ordinary task directions. It requires access to suspect-model outputs. Appendix B.4 estimates about 450 GPU-hours to build the full 1,000-key bank and about 1.6 GPU-hours for an audit after sampling, on the paper's setup. The bank is reusable, but that cost and post-training role make it unsuitable as the first preventive TRACE intervention. [Paper, §3 and Appendix B.4](https://arxiv.org/abs/2610.37266v1).
- **Knowledge Honeypot, June 2026 preprint.** Builds a graph of correct but downstream-irrelevant facts and injects references that lure response-guided entity-expansion attackers into the graph. The suspicion score includes query length, terms such as “define”, and session progression `(t+1)/B`; it is not strong independent evidence of distillation. Section 4.6 considers relevance filtering, template filtering, LLM screening, and obfuscation. The claimed 6.2% average agreement reduction concerns tested domain-extraction pipelines, not arbitrary collection. Low-transferability filtering itself uses probe training. [Paper, §§2.2–3.3, 4.6 and Appendix C](https://arxiv.org/abs/2606.15810v1). The department lists this work as accepted to EMNLP 2026 Main. [FSU announcement](https://www.cs.fsu.edu/yushunrailabs/).

### Inferences

- BLADE is a useful stress test but a poor first API defense. Unicode manipulation can damage copying, parsing, accessibility, math processing, and code despite looking normal. Rendering-preserving is not utility-preserving for a machine-readable service.
- A classifier or likelihood score trained only against naive SFT collectors will miss adaptive collectors that learn from a filtered subset. Include robust-loss or loss-clipping attacks in addition to ordinary SFT.
- Knowledge Honeypot is attractive only when the attacker's next queries depend on the defender's previous suggestions. A fixed externally supplied corpus has no such dependence. Treat it as an optional active-extraction experiment; it does not repair detection against corpus labeling or laundering through benign requests.
- Recent findings do not establish a general method for stopping all distillation while retaining all legitimate API utility. That is a limitation of the observable problem, not a reason to return to query caps.

### Gaps

- Neither the BLADE result nor the 7B trace-rewriting result establishes general protection across code, multilingual use, strong alternative student families, and answer-only supervision.
- A smaller watermark reference bank trades computation against the rank-test p-value floor and statistical power. It cannot inherit the published full-bank detection numbers without re-evaluation.

## What should TRACE actually test first?

### Takeaway

Use detection for reversible enforcement, but separately measure whether response changes suppress real student improvement. A short study can establish a bounded, credible result if it uses real distillation, hard benign lookalikes, and adaptive preprocessing.

### Cited findings

- The paper comparisons above motivate distinct tests for acquisition monitoring, response learnability, and later attribution. Their observables and objectives differ. [Trace rewriting](https://arxiv.org/abs/2602.15143v2), [LADS](https://arxiv.org/abs/2605.18829v2), [Watermark Basins](https://arxiv.org/abs/2610.37266v1).

### Inferences

This is the researcher's proposed experiment, not an existing published guarantee:

1. Pick one task where clean distillation demonstrably improves a 1–3B student. Freeze a held-out test set, clean teacher cache, training recipe, and no-distillation checkpoint. Pilot first; abort the effectiveness comparison if clean distillation produces no gain.
2. Compare clean traces, answer-only outputs, ordinary paraphrases, and the published dense-formal rewrite instruction. Keep the response-policy decision independent of training labels. Measure final-answer correctness, explanation usefulness, tokens released, latency, and student gain.
3. Challenge the best defense with a cheap attacker that extracts answers, simplifies explanations, filters high-loss tokens, or generates replacement explanations. Use identical clean/defended data splits and training budgets. Preserve a separate budget for stronger adaptive attacks.
4. Run LADS only as a separate multi-account ablation. Compare one account, redundant paraphrase collection across accounts, broad unique queries, and counter-desynchronized collection. Do not pool the result with rewriting and then attribute the combination to one mechanism.
5. For enforcement, block or restrict *future releases* only after the detector's calibrated evidence threshold. Track already released responses and tokens. Requests arriving concurrently with a detection decision must have an explicit admission ordering.
6. State success as a reduction in measured student capability gain at an explicit benign-utility cost under the tested threat model. A blocked synthetic campaign or lowered nominal token count alone does not prove prevention of distillation.

An appropriate objective is to minimize the strongest measured attacker's gain, subject to a benign-service constraint:

\[
\min_{\pi}\;\max_{a\in\mathcal A_{\mathrm{tested}}}\left[U(S_{a,\pi})-U(S_{\mathrm{base}})\right]\quad\text{subject to}\quad U_B(\pi)\geq U_B(\mathrm{clean})-\epsilon.
\]

Here `pi` includes admission and response transformations; `a` includes collection, preprocessing, and learning; `U_B` must include user-visible quality and service availability. This is a study objective. It does not certify robustness outside the tested attacker set.

### Gaps

- The final report must supply the repository-specific integration points and a measured hardware inventory from the feasibility researcher. These notes do not infer them from the old project name or from GPU marketing names.
- No runtime claim should be made until a small pilot records the actual model revision, precision, context length, memory use, token throughput, and clean student learning curve.
