# v7: run v7

**Status:** complete. **Code:** `47db88564e47d2285c546cc509b603748008bc1a`. **Steps:** 2026-11-05T17:45:58Z to 2026-11-07T11:37:46Z UTC.

Exported by `scripts/start.sh` from `artifacts/runs/v7`. Write the results, limitations and stop cause here before committing.

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
CONCURRENCY=2
QUALIFICATION_WORKERS=16
CALIBRATION_DEVICE=cpu
ALLOW_UNDETECTED=0
UTILITY=1
TEACHER_MODEL=Qwen/Qwen3.5-9B-Q4_K_M-GGUF
TEACHER_MODEL_REVISION=sha256:cd76ec205963b3b33350093e6904d9de16c4e666fd104e1f632d25c7f15f2a13
TEACHER_PREFLIGHT_ARGS=
UTILITY_OFFERED_RPS=
UTILITY_MIN_COMPLETION_RATE=
```

## Campaign outcomes

| attack | arm | outcome |
|---|---|---|
| A1 | full | gate refused: attacker not escalated beyond held-out benign accounts |
| A1 | instantaneous | gate refused: attacker not escalated beyond held-out benign accounts |
| A1 | quota | received 1215, blocked 4785, challenge 0, failed 0, deadline 0, released_tokens 161176, 1.67 h |
| A2 | full | received 1699, blocked 4201, challenge 100, failed 0, deadline 0, released_tokens 77408, 1.64 h |
| A2 | instantaneous | received 1599, blocked 4401, challenge 0, failed 0, deadline 0, released_tokens 40355, 1.67 h |
| A2 | quota | received 1215, blocked 4785, challenge 0, failed 0, deadline 0, released_tokens 2784, 1.67 h |
| A3 | full | gate refused: attacker not escalated beyond held-out benign accounts |
| A3 | instantaneous | gate refused: attacker not escalated beyond held-out benign accounts |
| A3 | quota | received 1214, blocked 4785, challenge 0, failed 1, deadline 0, released_tokens 215700, 1.81 h |
| A5 | full | received 4693, blocked 1207, challenge 100, failed 0, deadline 0, released_tokens 321831, 1.94 h |
| A5 | instantaneous | received 4599, blocked 1401, challenge 0, failed 0, deadline 0, released_tokens 286332, 1.67 h |
| A5 | quota | received 5430, blocked 570, challenge 0, failed 0, deadline 0, released_tokens 696685, 4.40 h |
| A6 | full | gate refused: attacker not escalated beyond held-out benign accounts |
| A6 | instantaneous | gate refused: attacker not escalated beyond held-out benign accounts |
| A6 | quota | received 6000, blocked 0, challenge 0, failed 0, deadline 0, released_tokens 806570, 4.24 h |
| A6 | unprotected | received 6000, blocked 0, challenge 0, failed 0, deadline 0, released_tokens 806570, 4.24 h |
| A7 | full | gate refused: attacker not escalated beyond held-out benign accounts |
| A7 | instantaneous | gate refused: attacker not escalated beyond held-out benign accounts |
| A7 | quota | received 1212, blocked 4785, challenge 0, failed 3, deadline 0, released_tokens 247309, 2.34 h |

## Contents

- `provenance/`: code revision, diff, configuration, redacted environment, packages, hardware and corpus manifests, one folder per attempt.
- `run.log`, `steps.tsv`, `logs/`: the run timeline, step timings and per-step output.
- `benign/`, `calibration/`, `adequacy.json`, `schedules/`: workloads, calibration and frozen attack schedules.
- `campaigns/ATTACK/ARM/`: decisions, harness log and settings, outcome, evaluation report, and `campaign/` events, attempts and ledger.
- `utility/ARM/`: benign utility under each arm.
- `SHA256SUMS`, `export.json`: checksums and export record.

Not exported: client credentials, harness state databases, the deception key, vLLM and teacher-API logs, and regenerable detector inputs.

## Unprotected run

`campaigns/A6/unprotected/` is the unprotected A6 run. It was collected through the quota arm, whose quota never fired on A6, so its files match `campaigns/A6/quota/`; `SOURCE.md` there gives the details.

`campaigns/A6/unprotected-v3-archive/` is a second unprotected A6 ledger from the v3 run on the `archive` branch, with a different teacher and corpus build; `SOURCE.md` there records its origin.
