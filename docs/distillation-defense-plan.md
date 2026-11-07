# Restoring TRACE evaluation and experimental defenses

The restoration uses commit b779c479767ac69ff83baf71e2ef04ad889190f4 in the history of the `archive` branch. It brings back selected code with compatibility and correctness fixes, rather than merging the archive branch. Frozen readings, corpora, legacy workspaces and research artifacts remain outside the restoration. The methods are in [evaluation](evaluation.md) and [harness](harness.md).

## Agreed scope

- Restore acquisition manifests and nested query/token selection, including exact teacher-setting comparison, input/record hashes and exclusive publication.
- Restore GSM8K planning, optional GPU execution, headroom and clean-student gates, and paired transfer metrics. Keep AG News planning available.
- Restore optional utility-capacity binding and comparison validity. Keep unbound collection's v2 contracts and defaults; explicitly bound collection uses v3. Unbound or invalid evidence cannot qualify a legitimate-utility comparison without a recorded diagnostic override.
- Restore the named durable request-plus-token quota and a standalone measured-grid matcher under distill baseline match-quotas.
- Restore log-determinant coverage and whole-path calibration with a durable terminal stop, only as an explicitly selected, unqualified experimental policy.
- Restore relevant existing tests and command integrations. Preserve legacy calibration computations and policy/state behavior when the new settings are unused.

## Required fixes during restoration

Successful fresh collection creates pending-quality responses, while the archived math planner accepted only reviewed responses. Acquisition selection and plan-math now have an explicit include-pending option that never admits rejected rows. The math plan records this choice outside the model recipe; it does not falsely certify response quality or change model identity. The pilot script uses that option so collection can reach training without relabeling its ledger.

Utility binding is optional and must be supplied completely. The original unbound measurement version and contract remain intact, including resume behavior. Historical unbound rows stay unknown; backfill does not hardcode an overload conclusion. Refusing unsupported legitimate-utility comparisons is an intentional tightening of the report, not a change to default collection.

Independent path audits need a numerically stable binomial upper bound at larger counts. That routine is local to path_calibration.audit_upper_bound; the existing calibration.binomial_upper is preserved byte-for-byte.

The first restoration kept the quota matcher under baseline, moved clean-gate verification into baseline_gates and left the archived output-study stack out. That stack and an opt-in gateway output policy have since been added; see [Output defenses](#output-defenses). Poisoning stays excluded.

Manifest publication wraps directory/temp creation errors, flushes data before linking and keeps private mode 0600. The pilot's GPU lock uses append mode and defaults to the runtime or repository artifacts directory. These fixes do not establish model performance or operational qualification.

## What the restored commands establish

A headroom gate checks whether a teacher outperforms the base on the selected problems. A clean gate checks whether a clean student reproducibly learns that task at the declared sizes and seeds. A transfer report computes effects from supplied paired scores, withholding unsupported suppression claims. A quota match selects an observed development setting that matches every declared legitimate-use stratum. A path calibration calibrates a finite-horizon stopping rule under its stated exchangeability assumptions.

None of those commands alone establishes that extraction was prevented. CPU fixtures and syntax checks are development evidence; live teacher and GPU experiments remain separate work. Hashes establish byte consistency, not truthful provenance, authorization or independence.

## Next implementation: intervention before useful learning

1. Freeze a useful capability target using a passing clean experiment and independent development tasks.
2. Export chronological acquired-response prefixes from actual release events, including first alert, restriction, refusal and end of campaign. Include in-flight outputs received after intervention.
3. Train prefix students from the same pinned base at each declared checkpoint and evaluate a locked source-disjoint suite.
4. Compare the earliest checkpoint reaching the target with effective intervention and cumulative attacker cost. A target not reached within the experiment is censored, not impossible.

The ordinary curve subsets are hash-ordered. Historical matched-prefix controls match supervision volume and are not intervention-aligned learning curves. The subsequent [checkpoint workflow](checkpoints.md) adds verified receipt-ordered exports and checkpoint students as separate opt-in tooling. It still requires a preregistered useful-capability target, a valid clean baseline and actual teacher/GPU execution before drawing effectiveness conclusions.

## Next implementation: detector versus quota

1. Collect a legitimate development workload through allow, the candidate and a grid of quota and quota_tokens settings, with a passing capacity preflight and fresh state for each setting.
2. Derive per-stratum task-success losses from those measured runs, then freeze the selected settings using baseline match-quotas. The development-grid derive/match command now derives losses from hash-bound utility rows.
3. Confirm legitimate utility on held-out users and run fixed and adaptive extraction campaigns with comparable attempt, time, money and account resources.
4. Train students on the supervision each arm actually released and compare capability and cost. Use equal-released-token subsets only as an additional mechanism control.

The opt-in development-grid commands plan capacity-bound cells, execute them through a trusted injected runner, verify completed artifacts and derive measured losses for the restored matcher. CPU fixtures cover orchestration; a live adapter, actual measured-grid collection, held-out confirmation and the complete campaign-to-student verdict remain open. See docs/evaluation.md for the adapter contract.

## Next implementation: adaptive retained capability

A8 already selects online using coverage and attacker-visible outcomes and accepts an identified uncertainty hook. Implement a concrete hook that updates a student, scores candidates and records its checkpoints and compute. Test topic sorting, paraphrases, answer-only collection, benign cover, delays and account rotation within declared resources. Keep all defense state hidden from the attacker. Evaluate the resulting students, including attacks that retain capability while avoiding alarms.

No student-updating A8 implementation or adaptive capability result is created by restoration. Do not count a callback interface or a deterministic prompt transform as a completed learned attack.

## Experimental path policy boundary

Path is the first H evaluated admissions of an authenticated principal, with fixed nonoverlapping looks. It neither rearms automatically after H nor links Sybil identities. Without a stop it explicitly abstains after H; a stop remains durable. Unsupported strata and insufficient calibration samples abstain. Every runtime artifact and event remains qualified: false.

The current state retains and serializes all prefix embeddings, and coverage recomputes over that prefix. Large H therefore has substantial memory/storage/latency cost. Keep short experimental horizons and measure cost on the target host. Compact incremental state, realistic benign calibration, an independent audit and actual student effectiveness are prerequisites for deployment. Existing MMD remains available at larger study budgets, with its historical detection failures retained as development evidence.

## Output defenses

A client whose traffic matches a legitimate customer cannot be detected, so only changing the released output can reach it. The research's top prevention candidate is answer-preserving dense rewriting of reasoning. Trace Rewriting (arXiv 2602.15143v2, E.3) reports GSM8K student accuracy falling from 57.7% to 30.2% and teacher accuracy from 88.8% to 88.1%, with a 7B teacher and rewriter. That is the paper's result, not TRACE's.

Two opt-in pieces now exist. [`distill output-study`](evaluation.md#cached-output-defenses-and-matched-quota-comparison) builds output arms from one cached acquisition and joins supplied student evidence. [`HARNESS_OUTPUT_POLICY`](harness.md#opt-in-output-policy) applies `final_answer_only`, `compact_verified`, `paraphrase` or `dense_locked` before the gateway releases a response, for GSM8K's `####` format only.

Neither has run on a real teacher or student, so TRACE has no effectiveness measurement for them. Correct answers remain training data, and final-answer collection is untouched by any rewrite of explanations. An always-on policy changes the product for every client. Rewrites are checked for format and answer equality, not for correct reasoning. The next step is a matched study on actually released text, with adaptive attackers and measured legitimate-user loss.

## Verification and delivery

Run Python checks through devenv shell with uv, repository pytest, Ruff lint/format and strict mypy, plus shell syntax checks. Use fresh temporary artifacts and indexes for fixtures. Verify that the existing changed reading files retain their original hashes and that legacy calibration was not modified. Commit only the restored implementation, its fixes, tests and documentation using path-limited Jujutsu commits. Preserve unrelated working-copy changes. Do not push or activate an experimental defense as part of restoration.
