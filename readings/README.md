# Run readings

Each directory records every measurement from one study run on the lab machine: one WSL2 host with an RTX 4000 Ada (20 GB). Run outputs live in gitignored `artifacts/` and `data/`, so this directory is the versioned record. Each run has two files:

- `README.md`: the narrative, with the protocol, a timeline, why the run stopped, every result table, and the digests of the source files
- `readings.json`: a machine-readable copy of every small source artifact, plus the derived statistics in the README

Large per-request files stay on the run machine. Each README lists them.

| Run | Date | Run directory | Code | Protocol | Stopped at | What it measured |
|---|---|---|---|---|---|---|
| [v0](v0-2026-10-16-start/) | 2026-10-16 | `artifacts/start` | not recorded | pre-`task-study-v2` (AG News, SQuAD, GSM8K) | after the harness traffic `before` phase | undefended teacher collection, teacher test-suite benchmarks, unprotected harness traffic |
| [v1](v1-2026-10-17-start-v2/) | 2026-10-17 | `artifacts/start-v2` | `866ade6` | `task-study-v2` (AG News, SQuAD) | undefended collection: `collect-protected` rejected `--teacher-url` | MMD detector calibration, benign holdout replay of three defenses, single-account attacker detectability |
| [v2](v2-2026-10-21-gateway-bounded-v1/) | 2026-10-21 → (running) | `artifacts/gateway-bounded-v1` | `4bb6381`, resumed on `3a50a1a` | gateway study (AG News, SQuAD, GSM8K, coding; A1, A2, A5, A7; 3 seeds) | interim: all 12 undefended campaigns and the development sweep done; protected campaigns pending | undefended ledgers (74,000 responses, committed), development transfer gate |
| [v3](v3-2026-10-29-gateway-replay/) | 2026-10-29 → (running) | `artifacts/gateway-bounded-v1` | `3599d6e` plus collection-order change; resumed on `93c9f1a`; A2 attempt 4 on `c102078` | continuation of v2, narrowed on 2026-10-30 to seed 42, `full` harness, A1/A2/A5/A6/A7 | interim: A1 protected campaigns done (`full`, `instantaneous`, `quota`); A6 undefended ledger collected; three A2 attempts interrupted, attempt 4 running from 2026-10-31 07:14 UTC; A5/A6/A7 protected pending | benign policy replay, A1 protected ledgers and decisions, partial A2 protected traces, A6 undefended ledger, live benign utility per defense, development recipe selection with adapters (LFS), calibration, workloads, schedules, evaluation-method checks |
| [v4](v4-2026-10-29-kaggle-training/) | 2026-10-29 | Kaggle `trace-training/undefended` | `1e5b3d8` (notebook) | LoRA student training, FP16, three epochs | training completed | 4,448 training examples, 834 optimizer steps; no validation |
| [v5](v5-2026-10-30-undefended-a2-student/) | 2026-10-30 | `artifacts/undefended-a2-benchmark` | `e53503f` | pre-defense shared student on v2's A2/s42 undefended ledger; development-selected recipe (1e-4, three epochs), BF16 LoRA | training completed; held-out evaluation pending | 4,816 accepted training examples (GSM8K 367 of 1,500), 903 optimizer steps; adapter in LFS |

## Results so far

| Measure | AG News | SQuAD | GSM8K | Run |
|---|---|---|---|---|
| Teacher test score (`Qwen3.5-9B`, fp8, non-thinking) | 85.25% accuracy | EM 72.36%, F1 88.83% | 23.12% (76.6% cut off at 256 tokens) | v0 |
| Teacher agreement with gold on collected attacker prompts | 85.61% | EM 60.12%, F1 81.96% | 23.65% | v0 |
| Undefended responses collected | 20,585 | 19,509 | 3,150 | v0 |
| Attacker (2,000 prompts, one account): highest level under `full` / `instantaneous` / `quota` | L0 / L0 / L4 from attempt 501 | L0 / L0 / L4 from attempt 501 | — | v1 |
| Benign L1 escalation under `full`, all accounts / scored accounts | 10.3% / 35.6% | 9.6% / 50.0% | — | v1 |
| Benign specialist accounts escalated under `full` | 11 of 11 | 7 of 7 | — | v1 |
| Benign accounts blocked by `quota` | 0 | 0 | — | v1 |

Shared gateway, A1 seed 42, 6,000 submissions across four domains (v3): the attacker stayed at L0 under `full` and `instantaneous` and received all 6,000 responses; `quota` refused it from submission 1,216 and released 1,215.

v2's development sweep trained six shared students. AG News is the only domain with qualified transfer: +20.0 pp, from 66.6 to 86.6, 95% CI 16.4–23.8. SQuAD gained +2.8 pp, coding +0.2 pp, and GSM8K −14.5 pp. There are no defense-effect or utility results yet.

The v4 run trained a student adapter, but has no held-out evaluation, so there are no measured v4 distillation-gain, defense-effect, or student utility results. The same holds for v5 until its held-out evaluation finishes.

## Adding a run

Name the directory `vN-YYYY-MM-DD-<run-dir name>`, increasing `N`, and add a row to both tables above. Record every reading the run produced, including partial and failed stages, with the code commit, the full protocol, a UTC timeline, the stop cause, and SHA-256 digests of the source artifacts. Report escalation rates over scored accounts as well as over all accounts, and label simulated or demo outputs (such as `outputs/tasks-demo`, which `just benchmark-demo` creates) as not measured.
