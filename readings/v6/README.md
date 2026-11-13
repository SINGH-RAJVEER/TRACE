# v6: run v6-20261105

**Status:** complete. **Code:** `44b79b71bbac4a51a833ed791d07f0276e252116`. **Steps:** 2026-11-05T13:13:26Z to 2026-11-06T13:04:21Z UTC.

Exported by `scripts/start.sh` from `artifacts/runs/v6-20261105`. Write the results, limitations and stop cause here before committing.

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
UTILITY_OFFERED_RPS=
UTILITY_MIN_COMPLETION_RATE=
```

## Campaign outcomes

| attack | arm | outcome |
|---|---|---|
| A1 | full | gate refused: attacker not escalated beyond held-out benign accounts |
| A1 | instantaneous | gate refused: attacker not escalated beyond held-out benign accounts |
| A1 | quota | received 1215, blocked 4785, challenge 0, failed 0, deadline 0, released_tokens 150442, 1.67 h |
| A2 | full | received 1699, blocked 4201, challenge 100, failed 0, deadline 0, released_tokens 73774, 1.64 h |
| A2 | instantaneous | received 1599, blocked 4401, challenge 0, failed 0, deadline 0, released_tokens 37890, 1.67 h |
| A2 | quota | received 1215, blocked 4785, challenge 0, failed 0, deadline 0, released_tokens 2796, 1.67 h |
| A3 | full | gate refused: attacker not escalated beyond held-out benign accounts |
| A3 | instantaneous | gate refused: attacker not escalated beyond held-out benign accounts |
| A3 | quota | received 1215, blocked 4785, challenge 0, failed 0, deadline 0, released_tokens 201361, 1.89 h |
| A5 | full | received 4619, blocked 1281, challenge 100, failed 0, deadline 0, released_tokens 282232, 1.64 h |
| A5 | instantaneous | received 4599, blocked 1401, challenge 0, failed 0, deadline 0, released_tokens 252516, 1.67 h |
| A5 | quota | received 5430, blocked 570, challenge 0, failed 0, deadline 0, released_tokens 642418, 4.02 h |
| A6 | full | gate refused: attacker not escalated beyond held-out benign accounts |
| A6 | instantaneous | gate refused: attacker not escalated beyond held-out benign accounts |
| A6 | quota | received 6000, blocked 0, challenge 0, failed 0, deadline 0, released_tokens 744962, 1.67 h |
| A7 | full | gate refused: attacker not escalated beyond held-out benign accounts |
| A7 | instantaneous | gate refused: attacker not escalated beyond held-out benign accounts |
| A7 | quota | received 1215, blocked 4785, challenge 0, failed 0, deadline 0, released_tokens 235535, 2.21 h |

## Contents

- `provenance/`: code revision, diff, configuration, redacted environment, packages, hardware and corpus manifests, one folder per attempt.
- `run.log`, `steps.tsv`, `logs/`: the run timeline, step timings and per-step output.
- `benign/`, `calibration/`, `adequacy.json`, `schedules/`: workloads, calibration and frozen attack schedules.
- `campaigns/ATTACK/ARM/`: decisions, harness log and settings, outcome, evaluation report, and `campaign/` events, attempts and ledger.
- `utility/ARM/`: benign utility under each arm.
- `SHA256SUMS`, `export.json`: checksums and export record.

Not exported: client credentials, harness state databases, the deception key, vLLM and teacher-API logs, and regenerable detector inputs.
