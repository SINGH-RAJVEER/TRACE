# Teacher and harness evaluation protocol

## Study status and roles

This protocol specifies a future study. Its acceptance targets are proposals, not measured outcomes. Historical v3 A1/A2 campaigns and the v4 reordering probes are development evidence; they have already been inspected and cannot be used as a locked final test. Keep frozen artifacts read-only. Freeze separate source families for representation fitting, reference fitting, score calibration, policy development and locked final test. Split news by publisher, period and topic where available, QA by document, math by problem family, and coding by problem lineage. Screen near duplicates across roles. Record the source manifests, selection rules, versions, seeds and limitations before calibration. The workload and calibration implementation is separate work and must be integrated before claiming these checks ran.

## Baselines and campaign design

The intended baseline panel is allow, full MMD, instantaneous MMD, request quota, rate limit and request-plus-output-token quota. The last baseline requires the separate accounting implementation; declaring it here does not enable it. Use fresh account state per condition and the same frozen schedule, deadline and declared resources. Compare cap enforcement with detector escalation separately. Replay is policy-only and cannot establish delivered answers or solved challenges. L1 alert can serve traffic; report L1, L2, L3 and L4 separately and distinguish first alert, restriction and refusal.

Freeze the original A1-A7 attack schedules using `uv run distill gateway schedule --help`, including broad extraction, topic ordering, independent-source and cover strategies, account splitting and rotation as specified by each schedule. Include adaptive reordering, low-and-slow pacing, answer-only requests, deduplication, replay of legitimate sessions, selective retries and budget-aware account rotation. Add paraphrase, batching, verifier filtering and alternative-teacher mixing when available. Charge online calls, proof attempts, output tokens, money and offline work consistently. Fixed-pool reuse and acquisition of new examples are different budgets. The default 12,000 HTTP-attempt budget includes failures and proof redemption; an account horizon can impose a separate 10,000-attempt cap.

The benign suite should cover casual users, specialists, broad researchers, batch evaluation, repetitive labeling, iterative coding, long-document QA, shared enterprise keys and nighttime automation. Hold out organizations and source families as well as accounts. Include all horizons and scored horizons, new and established accounts, deadlines, correct task completion, challenge abandonment, recovery, latency and organization-level impact. Demonstrate that the undefended teacher has enough capacity for the legitimate workload before measuring utility. Overload is a separate experiment. The workload worker owns implementation of profiles, source manifests and calibration gates.

## Execute and report

Configure the teacher and gateway as described in [harness settings](harness.md). Build pinned task pools with `uv run --extra eval distill tasks build --help`, then freeze schedules and execute with `uv run distill gateway collect --help`. Use fresh output directories and retain contracts, schedules, event logs and response ledgers. Do not infer a delivered answer from admission or a scheduled prompt.

`uv run distill gateway evaluation-report --events RUN/campaign.events.jsonl` summarizes observed releases, failed attempts, distinct examples, timing and known output tokens. Admission and cancellation are unknown in the current event format. Missing usage makes the total released-token count unknown; known tokens and the number of unknown usages are reported separately. Decision attribution can be incomplete, so missing intervention times are not proof of absence. The current policy event omits release time, so delayed serve restrictions cannot be timed by this report. Existing `campaign-timing.json` reports first reaches and per-account timing; its pre-reach token subtotals include an unknown-token count. Completed and interrupted campaigns require separate labels. Report time to intervention and acquisition across all accounts through the deadline, including data released after an alert.

For benign profiles, report escalation, restriction and deadline completion with known and total denominators. Also report correct task completion, p50/p95/p99 latency, challenge abandonment, review and restoration burden, and compute. Missing task outcomes remain unknown. The small authored `distill traffic` suite is a synthetic development check, not representative benign evidence.

## Comparison and uncertainty

The matched-attacker-budget view fixes query attempts, output-token budget, money and deadline, then reports acquisition and student outcomes. The matched-legitimate-utility view fixes the metric, target, population and tolerance and checks observed utility before comparing attack cost. `distillation.evaluation_protocol.compare_conditions` checks supplied declarations and observed utility; researchers must verify them against the actual schedules, invoices and outcomes. Match all priority profiles, not merely a pooled mean. Report raw requests, outputs, tokens and distinct examples rather than inventing a universal information score.

Use paired campaigns or source families as independent units and report each unit when there are few. The protocol module supplies a cluster-mean percentile bootstrap and a one-sided zero-event bound. A 95% zero-event bound is `1 - 0.05 ** (1/n)`, approximately `3/n` for independent units. Clopper-Pearson binomial intervals assume independent trials and can be optimistic when accounts share sources; neither generated accounts nor overlapping windows create independent source families. With few clusters bootstrap intervals are unstable. Training seeds measure optimization variability conditional on the data, not independent campaigns. Prespecify primary comparisons and account for multiple tests, or label them exploratory.

## Student plan

`uv run distill student-plan --family FAMILY --seed 1 --cost-per-run 1.0 --output plan.json` writes a plan, not training. Repeat `--family` and `--seed` for the intended family by seed panel. The plan includes the base, gold and undefended-teacher controls, a defended-teacher arm, and checkpoints of 100, 300, 1,000, 3,000, 6,000 and 12,000 examples for supervised controls. Cost is the supplied per-run amount times planned runs; it excludes other costs unless included in the supplied amount. Attach the actual datasets and conditions before execution. Freeze a capability target on development data, then use objective task metrics and held-out source families. Measure cost and time to target, including censoring when the target is not reached.

## Proposed acceptance targets

- Legitimate utility: at most one percentage point loss in correct task completion in each priority group and at most 1% restricted horizons, with uncertainty and independence stated.
- Security economics: at least twice the attacker spend or time to a prespecified capability target, with source-clustered uncertainty excluding no improvement under the strongest declared adaptive attack.
- Incremental mechanism: outperform the strongest matched quota, token and rate baseline at matched legitimate utility; if fewer examples explain the result, claim acquisition control only.
- Accounting: crash, retry, cancellation, duplicate proof, concurrency and horizon rollover preserve the stated release bound. The separate accounting implementation must provide and pass these gates; current event logs cannot establish them.
- Sequential and operational claims: evaluate the exact selection and reset policy, report calibration limits, and measure latency, compute, review and recovery. Do not infer anytime validity from marginal calibration.

These are proposed thresholds for preregistration, not evidence of achieved performance. A fresh, source-disjoint protected run and actual student evaluation are required for effectiveness claims.
