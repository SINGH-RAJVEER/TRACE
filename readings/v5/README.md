# v5: run v5-20261102

Related records: [original undefended A2 student](original-study/),
[earlier snapshot](legacy-20261031/v5-2026-10-30-undefended-a2-student/),
[raw checkpoint metadata](raw-2026-11-02/), and
[development notes](development-notes/). They remain distinct from this run.

**Status:** complete, exit 0. **Code:** `eac63e64ae4e1a2294d161b7b8c6bfcb579ea888` plus the `wait_http` fix recorded in `provenance/20261102T152538Z/git-diff.patch`. **Steps:** 2026-11-02T12:54:20Z to 2026-11-03T15:13:00Z UTC.

This is the first `scripts/start.sh` run on the rebuilt corpus. It uses fresh benign roles (gateway-clients-v3, 125 accounts per role), the sealed exploratory attacker partition, one seed, and three arms per attack. No setting was tuned on its results. A student was not trained on these responses, so the reading measures what each control released, not what a student learned from it.

The complete reading is on `main`, including the LFS-tracked logs, ledgers, decisions, state databases, schedules, calibration and corpus. Narrative, checksums, provenance, timeline, settings and outcomes are stored directly in Git.

## Settings

```
TASKS_DIR=data
SEED=42
BUDGET=6000
PARTITION=exploratory
ATTACKS=A1 A2 A3 A5 A6 A7
ARMS=full instantaneous quota
DOMAINS=ag_news,squad,gsm8k,coding
BENIGN_ACCOUNTS=125
WORKLOAD_VERSION=gateway-clients-v3
A4_INTERVAL=60
CONCURRENCY=8
QUALIFICATION_WORKERS=16
CALIBRATION_DEVICE=cpu
ALLOW_UNDETECTED=0
UTILITY=1
TEACHER_MODEL=Qwen/Qwen3.5-9B
TEACHER_MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
TEACHER_PREFLIGHT_ARGS=
```

Calibration passed the adequacy check (`calibration-resolution-v2`, 100 scored policy horizons, 199 non-overlapping score windows, p-value floor 0.0017). It also raised the detectability warning the v4 audit predicted: all 50 pooled benign-mixture windows score below the median benign account window (median p 0.963), so the upper-tail score cannot flag traffic that resembles the benign mixture. The aggregate detector was skipped because the fitting role has 20,962 requests and needs 30,000.

## Campaign outcomes

Each campaign submits 6,000 requests. "Gate refused" means `distill gateway collect` declined to run the detector arm because the attacker's schedule did not escalate beyond the held-out benign accounts. No responses were collected for those cells.

| Attack | Arm | Served | Blocked | Challenged | Released tokens | First L4 at attempt |
|---|---|---:|---:|---:|---:|---:|
| A1 | full | gate refused | | | | |
| A1 | instantaneous | gate refused | | | | |
| A1 | quota | 1,215 | 4,785 | 0 | 150,442 | 1,216 |
| A2 | full | 1,699 | 4,201 | 100 | 73,774 | 1,800 |
| A2 | instantaneous | 1,599 | 4,401 | 0 | 37,890 | 1,600 |
| A2 | quota | 1,215 | 4,785 | 0 | 2,796 | 1,216 |
| A3 | full | gate refused | | | | |
| A3 | instantaneous | gate refused | | | | |
| A3 | quota | 1,215 | 4,785 | 0 | 201,361 | 1,216 |
| A5 | full | 4,619 | 1,281 | 100 | 284,506 | 2,104 |
| A5 | instantaneous | 4,599 | 1,401 | 0 | 255,064 | 1,022 |
| A5 | quota | 5,430 | 570 | 0 | 642,823 | 3,978 |
| A6 | full | gate refused | | | | |
| A6 | instantaneous | gate refused | | | | |
| A6 | quota | 6,000 | 0 | 0 | 753,916 | never |
| A7 | full | gate refused | | | | |
| A7 | instantaneous | gate refused | | | | |
| A7 | quota | 1,215 | 4,785 | 0 | 235,535 | 1,216 |

Notes on reading the table:

- The detector arms ran only for A2 and A5. For the other four attacks the gate found no separation from benign accounts, so there is no detector result to report for them.
- A2 sends 1,500 AG News prompts, whose answers are one-word labels, before any GSM8K prompt. The quota arm stopped it inside that block (1,215 AG News responses, about 2.3 tokens each). The full arm served all 1,500 AG News prompts and 199 GSM8K prompts, and the instantaneous arm served 99. That is why A2 releases fewer tokens under quota despite similar request counts.
- A5 and A6 spread their requests across several accounts. The per-account quota stopped A5 after 5,430 responses and never stopped A6.
- Per-account first-L4 attempts and the full timeline are in `index.tsv` and in each campaign's `campaign/campaign.events.jsonl`.

## Benign utility

The same 24,047-request `benign_test` schedule was replayed through each arm at concurrency 8 with a 120-second client deadline.

| Arm | Valid | Refused | Timed out | Served-only quality |
|---|---:|---:|---:|---:|
| full | 5,884 | 617 | 17,536 | 0.839 |
| instantaneous | 6,433 | 0 (715 challenged, then served) | 17,601 | 0.844 |
| quota | 6,657 | 0 | 17,378 | 0.846 |

About 73% of requests in every arm timed out. The single-GPU teacher could not keep up with the synthetic arrival rate, so completion rate and unconditional utility measure teacher capacity, not policy cost. The comparable numbers are the refusals and the quality of served responses. The full detector refused 617 benign requests. The quota refused none, because no benign account reached 1,215 requests inside the pass.

## Limitations

- One seed, one exploratory partition, synthetic benign traffic. The confirmatory partition is still sealed.
- The live harness ran its encoder on CUDA while calibration ran on the CPU (`CALIBRATION_DEVICE=cpu`, `HARNESS_ENCODER_DEVICE` unset). Small numeric differences in embeddings may shift scores near thresholds.
- Utility is capacity-bound, as above.
- A4 and A8 were not run.
- No student was trained, so nothing here measures capability transfer.

## Contents

- `provenance/`: code revision, diff, configuration, redacted environment, packages, hardware and corpus manifests, one folder per attempt. The first attempt (20261102T125419Z) stopped because vLLM's `/health` returns an empty body. The second is the run reported here.
- `run.log`, `steps.tsv`, `steps/`, `logs/`: the run timeline, step timings, done-markers and per-step output, including `logs/vllm.log` and `logs/teacher-api.log`.
- `corpus/`: the task corpus the run read as `TASKS_DIR=data`, copied from `data/` unchanged. One folder per domain (`news`, `question_answering`, `math`, `coding-subset-50000-s42`), each with its pools, sealed attacker partitions, evaluation suites and the `external/` pool used by A3 and A7. Its manifests are byte-identical to `provenance/corpus-*.manifest.json` and `provenance/external-*.manifest.json`. To reuse it, copy it back to `data/`.
- `benign/`: the four benign roles and their detector inputs.
- `calibration/`, `adequacy.json`, `schedules/`: calibration and the frozen attack schedules.
- `campaigns/ATTACK/ARM/`: decisions, harness log, settings, state database, outcome, evaluation report, and `campaign/` events, attempts and ledger.
- `utility/ARM/`: benign utility under each arm, with decisions and the state database.
- `SHA256SUMS`, `export.json`: checksums and export record.

Only client credentials (`credentials.json`) are left out. The run had no deception key. Every other file from `artifacts/runs/v5-20261102` is here. Before export, all files were checked against every client key and the upstream API key; none contains one.
