# Detecting distillation from observable API behavior

Research continued on 2026-11-03. This note combines verified source findings with an original proposed design. The design has not been implemented or shown effective. Five PDFs in `research/papers` named `PAPER_DETECT_*.pdf` were inspected through their extracted full text. Their versions, hashes, sizes and inspected sections are recorded in `detection_downloads.json`. No paper under `readings/` was changed. Mathematical derivations below are our own synthesis unless explicitly attributed.

## What can a detector actually establish, and what does recent evidence show?

### Takeaway

Observable collection behavior can support a calibrated extraction-risk decision. It cannot establish downstream training intent when a legitimate user and an extractor generate identical observable transcripts. The practical target is to stop statistically distinguishable harvesting early enough to reduce learned capability, while measuring the harm to legitimate automation.

### Cited findings

- Liu, Guo and Dong's *An Embarrassingly Simple Detector for Model Extraction Attacks in Large Language Model API Traffic* uses semantic query embeddings and benign-calibrated MMD. Its default offline window contains 1,500 queries. The reported aggregate results include 0.3% benign FPR and 100% pure-attacker TPR, across fourteen attacker-normal pairs and three seeds. These are window-level empirical results. Its limitations explicitly include difficult 5% attack mixtures, semantically matched queries, reference contamination, drift and incomplete adaptive evaluation. It observes query text and does not demonstrate online interruption before useful student learning. The saved version is arXiv v3 dated 2026-08-29; the live ACL page now establishes publication at EMNLP 2026, pages 17320–17338. [Paper](https://arxiv.org/abs/2606.05725v3), [published primary record](https://aclanthology.org/2026.emnlp-main.748/).
- Juuti et al.'s *PRADA*, IEEE EuroS&P 2019, studies sequential query-distance distributions and deviations from the pattern of benign requests. It is foundational evidence for stateful detection, but its evaluated classifier extraction setting does not establish broad LLM protection. Its mitigation discussion addresses attacker adaptation and distributed identities. Saved version is arXiv v5 dated 2019-03-31. [Paper](https://arxiv.org/abs/1805.02628v5).
- Pal et al.'s *Stateful Detection of Model Extraction Attacks*, 2021, introduces VarDetect using learned latent representations and MMD. Section 7.3.1 explicitly reports an F-MNIST extraction attack mixed at 5% not being detected at the chosen threshold, while 15% is detected in that example. This illustrates dilution sensitivity; it is not a universal 5% detection boundary. Saved version is arXiv v1. No peer-reviewed venue was established during this pass. [Paper](https://arxiv.org/abs/2107.05166v1).
- Shekhar and Ramdas's *Nonparametric Two-Sample Testing by Betting* provides a sequential testing framework with time-uniform type-I control. Its required fair-bet condition is conditional on previous observations. The standard two-sample theorem uses IID streams, with a stated extension to independent exchangeable pairs. The downloaded arXiv revision is v6 dated 2025-08-22 and includes corrections beyond the original 2023 journal work. [Paper](https://arxiv.org/abs/2112.09162).
- Vovk et al.'s *Retrain or not retrain*, COPA 2021, constructs conformal test martingales from randomized ranks under exchangeability and invokes Ville's inequality for sequential false-alarm control. The validity assumption concerns the sequence being ranked. Arbitrary dependent chat requests do not automatically satisfy it. Saved proceedings PDF has 20 pages. [Paper and proceedings record](https://proceedings.mlr.press/v152/vovk21b.html).

### Inferences

Let the complete defender-visible transcript up to a fixed horizon be \(O\), including prompts, account metadata, timing, available response history and any observed challenge outcomes. Let \(P_B\) and \(P_A\) be its distributions for benign and attacking campaigns. For any possibly randomized decision \(\delta(O)\in[0,1]\), the defining variational property of total variation gives

\[
\operatorname{TPR}-\operatorname{FPR}=\mathbb E_{P_A}\delta-\mathbb E_{P_B}\delta\leq\operatorname{TV}(P_A,P_B).
\]

Thus, at false-positive target \(\alpha\), \(\operatorname{TPR}\leq\alpha+\operatorname{TV}(P_A,P_B)\). If the extractor reuses exactly the observable traffic of a legitimate user and only trains afterward, \(P_A=P_B\), so \(\operatorname{TPR}\leq\alpha\). This is a direct mathematical derivation, not a claim that a particular paper proved the exact TRACE setting. Adding a stronger embedding model cannot recover an unobserved variable. Applying a feature map can only discard transcript information, not evade this bound.

The same distinction applies to challenges. An interactive challenge might create a difference if a real attacker behaves differently, but a client capable of producing the legitimate interaction can imitate that too. Passing proof of work measures willingness to spend computation. It does not establish that the acquired answer will not train a model.

A second useful derivation explains dilution. For \(P_\rho=(1-\rho)P_B+\rho P_A\), kernel mean embeddings obey

\[
\operatorname{MMD}(P_\rho,P_B)=\rho\operatorname{MMD}(P_A,P_B),\qquad\operatorname{MMD}^2(P_\rho,P_B)=\rho^2\operatorname{MMD}^2(P_A,P_B).
\]

Consequently, 5% mixture scales population squared MMD to 0.0025 of the pure-attack value for a fixed benign/attack pair and kernel. This does not specify finite-sample detection power, but it explains why broader windows cannot magically remove a vanishing signal. Account aggregation helps only when it makes the attack fraction or behavioral difference larger; merging unrelated legitimate users can make it smaller.

The defensible scientific claim is narrower than detecting intent: a declared, source-disjoint attacker family produces distinguishable behavior; the detector interrupts its acquisition; and training on the acquired data yields materially less capability than under a matched-cost baseline. The detector is useful only if all three hold.

### Gaps

No reviewed source establishes universal prevention of LLM distillation while preserving identical useful answers for identical legitimate requests. The previously attempted IACR 2022/1039 PDF download failed with HTTP 403. It is not saved or full-text reviewed, and no theorem from it is used here. The numerical results above are paper reports, not reproduced TRACE results.

## Which lightweight signals could distinguish harvesting beyond request volume?

### Takeaway

Use a small feature model that tests sequence structure and coverage against the correct legitimate task population, then calibrate the complete decision path. Treat every feature below as a hypothesis requiring matched benign controls. A classifier whose only useful signal is query count or source identity does not answer this project's question.

### Cited findings

The recent MMD paper establishes a useful text-only distribution baseline and explicitly leaves timestamps, account information, response information and accumulation across windows for later deployment work. It does not validate the proposed behavioral features below. [Limitations in the full paper](https://arxiv.org/pdf/2606.05725v3). The repository audit separately records that TRACE already contains conditional MMD, cheap sequence features and source-disjoint logistic separation; the proposed work should extend these rather than recreate them. [Repository audit](repository_and_evaluation.md).

### Inferences

Freeze the encoder, domain mapping, normalization, window schedule and feature-model parameters before calibration. Use a ridge-logistic model or a hand-specified score as the first candidate. A larger neural classifier is unnecessary until the small model demonstrates source-disjoint discrimination. Include these candidate signal groups:

1. Domain-conditioned distribution and coverage. Compare math queries against legitimate math workloads, coding against coding, and bulk generation against legitimate bulk generation. Include both unusually narrow and unusually broad coverage where appropriate. Do not make broad coverage itself malicious. A corpus-matched benign batch is an essential negative control.
2. Sequence dependence. Measure semantic relation between a new query and the previous task/answer, local edits, repeated sampling of the same task, systematic variation of one problem dimension, and topic transition structure. Teacher-output-conditioned follow-ups are observable for legitimate agents and can be imitated. They should contribute evidence only when their frequency differs from the matched null.
3. Template reuse with changing semantic content. Mask literal identifiers, numbers and boilerplate, then measure joint template/content variation. A fixed instruction wrapped around thousands of diverse questions can indicate harvesting; translation services, CI and batch labeling can have the same pattern. Include these applications in the benign reference.
4. Requested supervision. Detect repeated requests for worked solutions, multiple alternative answers, exhaustive examples, labels or token probabilities, if those fields are actually exposed. The signal is behavioral over a campaign. One request for an explanation is normal usage. Do not assume answer-only responses are non-distillable.
5. Cross-account overlap only with justified linkage. Shared unusual prompt sequences can supplement authenticated organization-level aggregation. A shared IP or popular public prompt does not justify treating unrelated accounts as one extractor. Score the actual grouping procedure during calibration; arbitrary post hoc clustering changes the decision family.

A cheap geometric coverage feature can use a fixed projection of normalized embeddings \(v_t\in\mathbb R^{64}\), \(\|v_t\|\leq1\), and a ridge matrix

\[
A_t=\lambda I+\sum_{i<t}v_iv_i^T,\qquad g_t=\log(1+v_t^TA_t^{-1}v_t).
\]

The matrix determinant lemma makes \(g_t\) the increment in log determinant after adding the query. It measures novelty relative to the already observed span. With a Sherman–Morrison inverse update it needs \(O(64^2)\) work and state per update, plus encoding. It is only a coverage proxy, not measured mutual information about teacher weights or expected student improvement. Compare its distribution against benign campaigns at matched prefix length and domain; using cumulative log determinant alone can silently recreate a volume threshold.

A first implementation can use fixed windows of 32, 64 and 128 requests with predetermined evaluation points, a 64-dimensional geometric feature, and the existing text encoder on CPU or cached embeddings. These are proposed pilot sizes, not researched optimal settings. The finite set of windows and evaluation points must be included in calibration. Measure CPU latency before promising overhead; the expensive part may remain tokenization and encoding. Truncated text needs a declared handling policy because prefix-only inspection can miss instructions or content later in the prompt.

Let \(f_\theta(H_t,c)\) be the frozen evidence score for observed prefix \(H_t\) and declared task context \(c\). Train it with source-disjoint splits that group related prompts, accounts and task templates together. Evaluate an ablation excluding count, rate, prompt source IDs and corpus-membership flags. Compare against a quota policy at the same observed legitimate-user harm. The relevant result is earlier restriction of useful attacker acquisition, not simply a larger AUROC.

### Gaps

No reviewed paper establishes that these added features separate TRACE's realistic matched traffic. The same feature model may classify honest evaluation and data-generation customers as extractors. A detector that fails these controls should be rejected before implementing stronger enforcement. The hardware identity remains unverified in the repository audit; the design assumes no extra large model and requires a local timing pilot.

## How should sequential evidence, stopping and false-positive claims be implemented?

### Takeaway

The immediate implementable option is whole-campaign maximum calibration with a frozen scorer. It permits arbitrary dependence within each benign campaign and controls the probability of any score-triggered stop over the declared horizon under exchangeability across campaigns. Use mathematical betting guarantees only when their stronger conditional assumptions are justified for the actual stream.

### Cited findings

For betting, the requirement is \(\mathbb E_0[e_t\mid\mathcal F_{t-1}]\leq1\) with \(e_t\geq0\). Multiplying these factors gives a nonnegative supermartingale, so crossing \(1/\alpha\) has null probability at most \(\alpha\). Independent references alone are insufficient; the monitored sample's conditional null law must match the reference construction. [Shekhar and Ramdas, Section 1.1 and Remark 7](https://arxiv.org/pdf/2112.09162). Conformal sequential ranks and their uniformity require exchangeability of the ranked observations and the specified randomization. [Vovk et al., Section 2, Proposition 1](https://proceedings.mlr.press/v152/vovk21b/vovk21b.pdf).

### Inferences

#### A mathematically valid betting option, with a restrictive assumption

Suppose \(h_t\) is chosen from previous data, takes values in \([-1,1]\), and \((X_t,Y_t)\) has equal conditional marginals under the benign null given \(\mathcal F_{t-1}\). Then

\[
d_t=\frac{h_t(X_t)-h_t(Y_t)}2\in[-1,1],\qquad e_t=1+\eta_t d_t,\quad 0\leq\eta_t\leq1,
\]

with predictable \(\eta_t\), satisfies \(\mathbb E_0[e_t\mid\mathcal F_{t-1}]=1\). Set \(E_t=\prod_{s\leq t}e_s\) and stop at \(E_t\geq1/\alpha\). Under the stated condition, Ville's inequality controls the chance of ever crossing, including repeated looks. Fixed-weight mixtures of independently valid e-processes remain valid; taking their maximum without adjustment does not.

The difficult requirement is the equal conditional marginals. Sampling a fresh historical query of the same topic does not ensure that its law matches a current multi-turn user's next request conditional on their history. Reusing finite references, adapting the score on the current observation, and ranking overlapping windows create further problems. A marginally valid window p-value is not automatically conditionally super-uniform; multiplying transformed p-values need not yield a valid e-process. Apply this construction first to carefully designed independent session units or a validated conditional generator, not to every API turn by assertion. For TRACE's immediate deployment proposal, use the following fallback.

#### Fixed-horizon whole-campaign rank calibration

Define a campaign and its horizon before examining test scores. For example, use authenticated organization-day with a declared maximum observation count \(H\) in the study, or a fixed finite time interval. The horizon defines the statistical unit; it need not deny service when reached. If service continues, start an explicitly accounted new test period rather than claiming the original guarantee extends forever.

Use separate partitions for model fitting, method selection, calibration and final audit. On each of \(n\) untouched benign calibration campaigns compute the maximum over the full declared path:

\[
M_i=\max_{t\in\mathcal T_i,\,t\leq H}f_\theta(H_{i,t},c_i).
\]

Include every planned scale, detector and action-triggering score in this maximum, after any fixed transformations fitted on training data. At runtime maintain \(m_t=\max_{s\leq t}f_\theta(H_s,c)\) and compute

\[
p_t^\star=\frac{1+\sum_{i=1}^{n}\mathbf1\{M_i\geq m_t\}}{n+1}.
\]

If the calibration campaigns and the new benign campaign are exchangeable, conditional on the frozen training/selection pipeline, the final campaign's conservative rank is super-uniform. Since \(m_t\leq M_{\rm new}\), each prefix rank is at least the final rank. Therefore

\[
\Pr_B\{\exists t\leq H:p_t^\star\leq\alpha\}\leq\alpha.
\]

This is a direct finite-sample rank argument. It handles dependence and overlapping windows inside one campaign because calibration observes the maximum of the same complete path. It does not require treating chat turns as IID. It does require exchangeability between full benign campaigns and exact agreement between calibration and runtime scoring, grouping, horizon and missing-data behavior. The guarantee is marginal over calibration and the next campaign, not a conditional promise for every realized calibration set. Audit of the frozen operational threshold is a separate step below.

The smallest attainable conservative rank is \(1/(n+1)\). At least 99 calibration campaigns are needed even to attain a 1% threshold and 999 for 0.1%; those counts alone do not certify a fixed deployed threshold or provide useful power. A rank is not the posterior probability that a client is malicious. If the correct task stratum has too few independent campaigns, abstain from an automatic statistical stop. Unsupported strata can still use ordinary operational protections, but report their outcomes separately.

Proposed algorithm, where indentation uses tabs:

```python
# fit_data, selection_data, calibration_campaigns and audit_campaigns are disjoint
scorer = fit_and_freeze(fit_data, selection_data)
maxima = [full_path_max(scorer, c, frozen_schedule) for c in calibration_campaigns]
policy = freeze_policy(scorer, maxima, horizon, alpha, grouping, frozen_schedule)
certify_frozen_policy(policy, audit_campaigns)

def on_request(request, campaign):
	# Admission state and evidence update must be atomic for parallel requests.
	with admission_transaction(campaign):
		update_observable_history(campaign, request)
		if unsupported_context(campaign) or horizon_expired(campaign):
			return separately_recorded_nonstatistical_handling(request)
		if evaluation_point(campaign):
			score = scorer(campaign.observed_prefix)
			campaign.maximum = max(campaign.maximum, score)
			p = (1 + sum(m >= campaign.maximum for m in maxima)) / (len(maxima) + 1)
			if p <= alpha and policy_has_passed_utility_audit():
				campaign.stopped = True
				return refuse_with_reason("calibrated_behavioral_evidence")
		if campaign.stopped:
			return refuse_with_reason("prior_calibrated_behavioral_evidence")
		return reserve_teacher_admission(request)
```

For a stop based on the incoming request, the scorer must use only observables available before teacher dispatch. A response-based score normally affects later requests unless the gateway buffers the response and makes a release decision before emitting any tokens. Record all in-flight requests and tokens still released after the stop, including concurrency races, instead of claiming zero leakage. Include state after disconnects and restarts; restarting a process must not erase the evidence path.

An alert threshold can be less stringent than a refusal threshold, but both must be predeclared and evaluated. Logging an alert does not change future behavior. A warning, challenge, delay or output alteration can, so benign calibration must represent that same interactive policy, or the original path guarantee may cease to apply. Start with shadow scoring followed by a single terminal stop policy, then validate intermediate actions separately.

Across repeated time periods, allocate a persistent false-alarm budget, such as \(\alpha_j=\alpha_{\rm total}6/(\pi^2j^2)\), and use a union bound. This remains valid without independence between periods if each period's stated marginal error bound holds, but thresholds soon require far more calibration data and lose power. Alternatively report a fixed organization-day error target honestly, with no lifetime claim. Repeatedly resetting \(\alpha\), opening new overlapping campaigns, or running several separate tests at full \(\alpha\) spends additional error budget. Sybil accounts are an evasion problem, and statistical significance does not solve identity linkage.

#### Fixed-policy certification and the scientific success criterion

After freezing the score and operational threshold, run an independent audit of complete benign campaigns. With zero false stops among \(n\) IID campaign-level Bernoulli outcomes, the exact one-sided 95% binomial upper bound is

\[
p_{\rm FP}^{\rm upper}=1-0.05^{1/n}.
\]

This follows by solving \((1-p)^n=0.05\). Thus 299 zero-failure audit campaigns bound the fixed policy's false-stop probability below 1%, and 2,995 bound it below 0.1%. Those are audit sample sizes, distinct from conformal calibration sizes. Do not tune the policy after seeing audit failures or claim these binomial bounds for thousands of overlapping windows from a few customers. If campaigns share an organization or task-source process, independence must be justified or the audit unit enlarged. Exchangeability alone does not imply this IID binomial audit model.

The primary attacker measure should be useful teacher data actually released before and after the stop and resulting student capability, with the exact accounting detailed by the repository/evaluation note. Evaluate release-matched and legitimate-cost-matched quotas, adaptive paraphrasing, benign-padding dilution, legitimate bulk traffic, cross-account splitting and an exact matched-transcript control. The last control must produce identical scores for benign and malicious downstream intent; discrimination there demonstrates leakage in the experiment.

A promising pilot would show lower student gain at the same legitimate task success and latency cost than quota controls. A failed pilot is still informative if it establishes that collection behavior is observationally matched. In that case the research must shift to reducing the transferable information in responses, restricting specific sensitive outputs or post hoc attribution. None of those substitutes should be labeled proven prevention until students trained on the released outputs actually lose the protected capability.

### Gaps

The combined behavioral score has no established TPR, false-stop rate or runtime on this GPU. Whole-campaign conformal validity assumes exchangeable benign campaigns; real drift, contaminated references, hidden task mixture and repeated customers can violate it. A source-disjoint fixed-policy audit is necessary but cannot prove future production stability. The appropriate response to unsupported traffic is an explicit unqualified state, not a fabricated confidence level.
