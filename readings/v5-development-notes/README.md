# v5: development observations moved out of the method docs

**Date moved:** 2026-11-03, from `docs/evaluation.md` and `docs/harness.md` at commit `469c90a7`. The text below is unchanged apart from relative links; each section names the date and inputs it already stated.

These are development observations on frozen v3 roles and schedules, a one-off crash test and the v4 audit's offline estimates. None is a detection, effectiveness or capability result, and the inputs were inspected during development, so none can serve as confirmation. The reports and scripts they cite under `artifacts/` were produced in another checkout and are gitignored; they are not part of this reading, so these numbers cannot yet be reproduced from the repository alone. The method each section relies on remains in the docs.

## Separation study, 2026-11-02

Method: [development-only separation study](../../docs/evaluation.md#development-only-separation-study).

Frozen v3 roles and schedules with the cached detector embeddings from the primary checkout, which was read only. No teacher, no `benign_test`, no threshold tuning; topic cells fitted on the `fitting` role. Commands are in the ignored `artifacts/separation-development/run.sh`, with reports beside it. Split seed 42 unless stated; the attacker side is one principal per schedule however many windows it contributes.

Every run's overall line is `separation survives balancing`, and every run's summary names A7 as the reason. A7 is drawn from external corpora (BBC News, NewsQA, Calc-math_qa, self-oss-instruct). None of its prompts shares a (domain, corpus) pair with the benign roles, so every A7 result is marked `corpus not balanceable`. Its surviving C2ST and learned-MMD AUCs of 0.86–1.00 show corpus provenance, which these controls cannot remove. They are not evidence of separation that survives corpus balancing.

Matched `all` results for A1 and the attack-level verdict. Each cell gives held-out AUC, clustered 95% CI where informative, and permutation p; Bonferroni alpha is 0.0167:

| Run | Matched (schedule / sorted) | Schedule order: C2ST logistic, C2ST MLP, learned MMD | Topic-sorted: C2ST logistic, C2ST MLP, learned MMD | A1 verdict |
|---|---|---|---|---|
| score_calibration, W=100 | 19 / 11 | 0.80 [0.61, 0.94] p .001; 0.83 p .001; 0.89 p .075 | 0.69 [0.42, 0.92] p .063; 0.65 p .122; 0.81 p .343 | disappears, inadequate |
| policy_calibration, W=100 | 18 / 17 | 0.77 p .003; 0.78 p .004; 0.91 p .009 | 0.69 p .032; 0.70 p .024; 0.97 p .084 | disappears, inadequate |
| score_calibration, W=100, `content_sha256` | 21 / 13 | 0.65 p .061; 0.80 [0.66, 0.92] p .002; 0.88 p .053 | 0.71 p .036; 0.68 p .052; 0.73 p .496 | disappears, inadequate |
| score_calibration, W=50, seed 42 | 42 / 42 | 0.58 p .109; 0.63 p .017; 0.75 p .075 | 0.50 p .491; 0.59 p .066; 0.68 p .249 | disappears, adequate |
| seed 43 | 44 / 25 | 0.56 p .176; 0.61 p .047; 0.80 p .023 | 0.50 p .498; 0.72 [0.56, 0.86] p .006; 0.49 p .925 | disappears, adequate |
| seed 44 | 25 / 24 | 0.71 [0.57, 0.84] p .010; 0.77 p .002; 0.60 p .923 | 0.56 p .259; 0.72 [0.57, 0.86] p .002; 0.69 p .956 | survives, adequate |
| seed 45 | 45 / 20 | 0.73 p .001; 0.75 p .001; 0.85 p .002 | 0.72 [0.52, 0.90] p .009; 0.73 p .006; 0.72 p .614 | survives, adequate |
| seed 46 | 50 / 57 | 0.68 p .004; 0.69 p .002; 0.76 p .042 | 0.62 p .010; 0.61 [0.45, 0.77] p .026; 0.80 p .033 | disappears, adequate |

Without balancing, volume features separate every comparison (AUC 0.96–1.00), largely from generator timing. Semantic tests give mixed results. Under `group_id`, the unbalanced A1 test attack windows form a single source group, so their intervals are undefined, because SQuAD article titles connect every interleaved window. On the score-calibration role, the unbalanced A1 C2STs score below 0.5 (0.37 and 0.35). After balancing, A1 retains modest semantic separation, typically AUC 0.6–0.8, and the pooled MMD baseline stays at 0.54–0.84. The fixed rule gives a split-dependent verdict. At W=100 every topic-sorted set is under the 20-window adequacy minimum. At W=50, A1's worst-order verdict is `disappears` for split seeds 42, 43 and 46 and `survives` for 44 and 45; topic-sorted A1 survives in 3 of 5 seeds through the MLP alone. This development evidence therefore does not cleanly trigger the stop rule for A1. The residual signal is weak, unstable across splits and may reflect topic structure finer than the eight cells used for matching. It does not support further semantic architecture investment either. The next development step is finer topic matching with larger attack pools and more split seeds. A7-style campaigns need benign roles that include their corpora before balancing can say anything. These are development observations on synthetic workloads with one principal per attack, not detection confirmation. Window-level permutation p-values can be optimistic when windows share sources; the clustered intervals are the dependence-aware measure.

## Shadow evidence replay

Method: [development-only shadow evidence study](../../docs/evaluation.md#development-only-shadow-evidence-study).

A local development-only replay used the materialized frozen v3 score-calibration and A1/A7 schedules with the existing `artifacts/mmd-audit/cache/detector.npz` cache from the primary checkout. It used no teacher, no `benign_test`, and no threshold tuning. The same-profile multi-domain null contains only 13 account horizons: all new session, novelty and volume ranks abstain for A1, topic-sorted A1 and A7, and the fused evidence is neutral at 1.0. The upper-tail rank floor is 1/14, about 0.0714, and the two-sided floor is 2/14, about 0.1429. Public-corpus matching was unavailable because no registered external index was present; there was no local exported A8 schedule or benign linkage truth. Local ignored reports, raw feature values, input hashes and the replay script are under `artifacts/detection-signals-development/`. These are development observations, not detection confirmation. Real-user session validity, a fixed independent cluster null for non-neutral linkage fusion, A8 replay and an efficiency-loss study remain open.

## Shadow scorers

Method and limits: [shadow scorers](../../docs/harness.md#shadow-scorers).

**Development check, not confirmation.** On the frozen v3 roles with cached embeddings, at target 0.05:

- AG News (102 independent specialist windows) and SQuAD (67) resolve. Coding (12) and GSM8K (2) abstain, so the coding separation the audit measured for A7 is not reported here.
- A1 and topic-sorted A1 have identical coverage: 127 of 128 cells, with rarefied rank p 0.031 at 200 prompts against `all`. That rank is the floor of a 31-horizon null.
- A7 ranks at p 0.094 against `all`. The `multi_domain` null has 13 horizons, so it abstains.
- On held-out `benign_test`, 2 of 13 multi-domain horizons and 1 of 112 single-domain horizons rank at or below 0.05.
- Conditional MMD gives A1 as logged p 1.0 in AG News and SQuAD. Topic-sorted A1 reaches minimum p 0.003 in AG News. 21% of benign AG News specialists also reach a window at or below 0.05.
- A7 is not separated in AG News (minimum p 0.20) or SQuAD (0.97).

These probes were inspected during development. No setting was tuned on them, and they cannot serve as confirmation.

## Budget enforcement

Method and the claims this supports: [what budgets establish](../../docs/harness.md#what-budgets-establish).

**Measured.** `uv run pytest tests/test_crash.py` kills the real server process mid-request and restarts it against a stand-in teacher. A one-off run on one Windows host used recorded A1 requests, a 1,215-request rolling budget, and four requests in flight at the crash: 1,211 responses were released and the four open requests stayed charged, totalling exactly 1,215. The in-memory quota released 2,430 across the same crash. An offline replay of the recorded A1 token counts at 1,215 requests a day released 6,000 responses and 588,239 tokens by day five under a daily-renewing quota, against 1,215 responses and 118,024 tokens under a seven-day budget. A 44,200-token rolling budget released 440 responses. These are one campaign, one seed, and one account. They measure enforcement, not capability. On that host, with no encoder running, the rolling check added about 0.3 ms at p50 to a state-file request of about 8.4 ms, and a refusal took about 1.9 ms. Measure cost on the deployment host.

**Estimated, not measured.** In the [audit's](../v4/analysis/REPORT.md) offline comparison, the six-account A6 campaign stays below a 1,215-request limit and releases all 6,000 responses, while an estimated 44,200-token limit releases 2,706. That token limit reached L4 on 4 of 100 scored benign accounts, against 1 for the request quota, using estimated token costs and synthetic users. No comparison covers a seven-day window, which is stricter than the one-horizon limits the audit fitted.
