# v1: `just start` run `artifacts/start-v2` (2026-10-17)

This file records every measurement from the `just start --concurrency 8 --run-dir artifacts/start-v2` run on 2026-10-17. The run used the code at `866ade6` and stopped at 10:39:59, before any undefended collection, training, or benchmarking. It therefore holds **no student, transfer, or capability results**. What it holds is a complete calibration of the MMD account detector for AG News and SQuAD, a held-out benign replay of all three declared defenses, and a single-account detectability replay of the seed-42 attacker. Every number below was read or derived from the files under `artifacts/start-v2/`, which is gitignored and stays on the run machine. The digests at the end identify the exact files. [`readings.json`](readings.json) holds the same data in machine-readable form. The earlier v0 run is in [`../v0-2026-10-16-start/`](../v0-2026-10-16-start/). Its teacher benchmarks and undefended collection are the teacher-side numbers this run never reached.

## Summary

- **The MMD detector never flagged the attacker.** On both tasks, 2,000 seed-42 attacker prompts from one account produced 20 complete windows and never left L0 under `full` or `instantaneous`. The windows' p-values sat at 0.69 or above on AG News and at exactly 1.0 on SQuAD, so the attacker looked more typical than the benign null windows. The attacker samples uniformly without replacement from the same task pool as the legitimate clients (`schedule: uniform_without_replacement`), so a distribution-shift detector has nothing to separate it on.
- **Only the quota baseline stopped it,** refusing from attempt 501 because the quota is 500 requests per account. Splitting the campaign across accounts would get around that.
- **The detector flagged the legitimate specialist clients instead.** Every scored `specialist` account in the benign holdout escalated to L1 or above: 11 of 11 on AG News and 7 of 7 on SQuAD, under both MMD defenses. Some `repetitive` accounts reached L1. No `batch`, `bursty`, or `short` account escalated. Any benign client whose topic mix is narrower than the pool looks anomalous, and an attacker drawing from the whole pool does not.
- **Escalation rates match the calibration targets only when unscored accounts count.** The reported rates (about 10% / 3–6% / 2–4% / 0–1.3% for L1–L4) divide by all accounts. 70–81% of benign accounts send fewer than 100 requests, so they never complete a window and can never escalate. Over the 45 (AG News) and 36 (SQuAD) scored accounts, `full` escalates 35.6% / 20.0% / 13.3% / 4.4% on AG News and 50.0% / 16.7% / 11.1% / 2.8% on SQuAD. Upstream commit `1b28f44` later added scored-horizon rates to the reports.
- **The quota baseline caused no benign false positives.** No benign account sends more than 500 requests.

## Run record

| Item | Value |
|---|---|
| Command | `just start --concurrency 8 --run-dir artifacts/start-v2` |
| Code | `866ade6` (main, 2026-10-17) |
| Host | WSL2, NVIDIA RTX 4000 Ada Generation (20,475 MiB), 24 CPU threads |
| Protocol | `task-study-v2` |
| Tasks / seeds / budgets | `ag_news`, `squad` / 42, 1, 20 / 500, 2,000, 8,000 |
| Defense arms | `full` (MMD + `paper` decayed CUSUM), `instantaneous` (MMD, no memory), `quota` (block-only) |
| L4 action | `refuse` |
| Gold reference students | enabled |
| Teacher | `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, vLLM, `quantization=fp8`, bf16, `max_seq_len=8192`, served at `http://127.0.0.1:8000` |
| Terms review ID | `self-hosted` |
| Capability targets | 80.0 for both tasks |
| Attempt budget unit | HTTP submissions, including proof redemption |
| Campaign deadline | 86,400 s |
| Attacker schedule | `uniform_without_replacement` |
| Declared controls | matched-volume, matched-prefix |
| Aggregate monitor | not run |
| Calibration root | none (fitted in the run directory) |
| Final `status.json` | `{"status": "running"}`, never updated because the run exited on an error |

### Timeline (UTC, 2026-10-17)

| Time | Event |
|---|---|
| 09:45:46 | `protocol.json` and `tasks.json` frozen; existing task builds and collection ledgers validated |
| 09:51:50 | AG News calibration written |
| 09:54 | Benign workloads written for both tasks |
| ~09:55–10:24 | SQuAD calibration: 13,170 fitting, 12,249 score-calibration, and 10,443 policy-calibration requests embedded on CPU |
| 10:24:33 | SQuAD calibration and AG News calibration report written |
| 10:32:30 | AG News benign holdout replay written (all three arms) |
| 10:33:47 | AG News detectability written; SQuAD calibration report written |
| 10:37 | SQuAD benign holdout replay written |
| 10:38 | SQuAD detectability written |
| ~10:39 | vLLM and the teacher API started; teacher smoke test passed |
| 10:39:59 | `distill collect-protected` exited with code 2; services stopped; the run exited |

From about 09:55 to 10:24, a second identical `just start` process wrote to the same run directory and competed for CPU. It stopped at 10:24:33 with `artifacts/start-v2/squad/calibration already holds a calibration`. Every calibration, report, replay, and detectability file was written by the surviving run. The duplicate's only other writes were `status.json` and the 09:54 workload files, and the surviving run rewrote the AG News workloads at 10:24 and the SQuAD workloads at 10:33. The SQuAD calibration timing is therefore inflated by the contention. On CPU, SQuAD embedding ran at about 15 requests per second. All detector embedding in the study runs on CPU (`BGEEncoder(device="cpu")`), and the GPU sat idle for the whole stage.

### Teacher serving

Taken from `artifacts/logs/vllm.log`, the one log left from this run:

| Item | Value |
|---|---|
| vLLM | 0.31.0, serving on `127.0.0.1:8001`, with the teacher API in front on `:8000` |
| Non-default arguments | `max_model_len=8192`, `quantization=fp8`, `gpu_memory_utilization=0.9`, `language_model_only=True`, `default_chat_template_kwargs={"enable_thinking": false}` |
| Weights | loaded in 8.09 s. The model took 10.36 GiB and 12.43 s to load. |
| KV cache | 6.57 GiB available, 151,552 tokens, 18.5× concurrency at 8,192 tokens per request |
| Engine start-up | 42.51 s (profiling, KV cache, warm-up), plus CUDA graph capture in 16 s and 21 s |
| Model cache | `Qwen/Qwen3.5-9B` was already cached, and the pre-start download fetched 0 bytes |
| Requests served | `GET /healthz` once, plus the one `POST /v1/teacher/generate` smoke test, which passed |
| Up | 10:38:43 to 10:39:59, then stopped by `SIGTERM` from the launcher after the error below |

### Why the run stopped

`study.py` calls `distill collect-protected ... --teacher-url URL --terms-review-id ID`. In `packages/distillation/src/distillation/cli.py`, `collect_protected_command` declared `teacher_url: str` and `terms_review_id: str` without `typer.Option`, so Typer treated them as positional arguments and rejected the flags:

```
Usage: distill collect-protected [OPTIONS] {teacher_url} {terms_review_id}
No such option: --teacher-url
[start] error: distill collect-protected exited with code 2
```

`tests/test_start.py` replaces `Launcher.distill` with a fixture, so the test suite never parses this command line. The same branch carries the fix: both parameters are now `Annotated[str, typer.Option(...)]`, as in `collect`. The bug was still present upstream at `8302b57`.

### Stages that did not run

Undefended campaigns (8,000 requests × 3 seeds × 2 tasks), the six-candidate recipe sweep and transfer gate, teacher and student baseline benchmarks, baseline and gold training, the 18 protected campaigns through the harness API, protected students, matched-volume and matched-prefix controls, and the study report. No teacher responses were collected into `artifacts/start-v2`.

## Task pools (`tasks.json`)

| | AG News | SQuAD |
|---|---|---|
| Dataset | `fancyzhx/ag_news` @ `eb185aa` | `rajpurkar/squad` `plain_text` @ `7b6d24c` |
| License | unknown | CC BY-SA 4.0 |
| Pool / test split | train (120,000) / test (7,600) | train (87,599) / validation (10,570) |
| Attacker pool | 53,893 | 40,356 |
| Development | 11,848 | 7,141 |
| Fitting | 21,558 | 13,170 |
| Score calibration | 10,754 | 8,229 |
| Policy calibration | 10,836 | 8,079 |
| Benign test | 10,950 | 10,431 |
| Source groups (attacker / dev / fitting / score / policy / benign test) | one per article: 53,893 / 11,848 / 21,558 / 10,754 / 10,836 / 10,950 | one per article title: 213 / 35 / 65 / 44 / 35 / 50 |
| Development / test suite items | 1,000 / 7,600 | 1,000 / 10,570 |
| Filtered | 151 duplicate prompts, 10 test overlaps | 193 duplicate prompts |
| Truncated contexts | 0 | 0 |
| Output limit | 16 tokens | 64 tokens |
| Metric | label accuracy | SQuAD (F1, EM) |

The SQuAD legitimate roles come from only 35–65 source articles each. Any source-clustered uncertainty for SQuAD therefore rests on a few dozen clusters.

## Results by task

In the tables below, "scored" means an account with at least 100 requests, which completes at least one detector window. An escalation count is the number of accounts whose highest level reached that level or above. "all" divides by every account, and "scored" divides by scored accounts only.

### AG News

#### Workloads (`task-clients-v1`, synthetic arrival times)

| Role | Requests | Accounts | Scored (≥100 req) | Short (<100) | Source documents | Unique prompts |
|---|---|---|---|---|---|---|
| fitting | 21,558 | 306 | 89 | 217 | 21,558 | 21,558 |
| score_calibration | 15,056 | 154 | 45 | 109 | 10,754 | 10,754 |
| policy_calibration | 15,170 | 155 | 45 | 110 | 10,836 | 10,836 |
| benign_test | 15,330 | 156 | 45 | 111 | 10,950 | 10,950 |

Client mix per role (accounts / requests / min–max requests per account):

| Role | batch | bursty | repetitive | short | specialist |
|---|---|---|---|---|---|
| fitting | 9 / 4,312 / 312–500 | 15 / 4,311 / 111–300 | 44 / 4,312 / 12–100 | 216 / 4,311 / 11–20 | 22 / 4,312 / 112–200 |
| score_calibration | 5 / 2,151 / 151–500 | 8 / 2,151 / 51–300 | 22 / 6,453 / 153–300 | 108 / 2,150 / 10–20 | 11 / 2,151 / 151–200 |
| policy_calibration | 5 / 2,167 / 167–500 | 8 / 2,167 / 67–300 | 22 / 6,501 / 201–300 | 109 / 2,167 / 7–20 | 11 / 2,168 / 168–200 |
| benign_test | 5 / 2,190 / 190–500 | 8 / 2,190 / 90–300 | 22 / 6,570 / 270–300 | 110 / 2,190 / 10–20 | 11 / 2,190 / 190–200 |

#### Calibration

- σ (RBF bandwidth): 1.032004; window 100, 20 reference windows
- Encoder: `BAAI/bge-small-en-v1.5` @ `5c38ec7c405e`, cls pooling, normalized=True, max 512 tokens; 51,784 texts encoded, 0 truncated
- Truncated by role: fitting 0, score calibration 0, policy calibration 0
- Score-calibration windows: 127; policy-calibration accounts/horizons: 155/155, of which 110 unscored
- Null MMD window scores (n=127): min -0.0004504, median 0.006463, mean 0.005587, p90 0.01226, p95 0.01445, p99 0.01723, max 0.01734
- Ville (e-value) thresholds: 2.3026, 2.9957, 3.9120, 4.6052  (= ln 10, ln 20, ln 50, ln 100)

Accumulator thresholds for L1–L4 (achieved policy-calibration escalation rate in brackets; targets 10% / 5% / 2% / 1%):

| Accumulator | L1 | L2 | L3 | L4 | Units |
|---|---|---|---|---|---|
| `paper` | 0.6740 (9.68%) | 1.7785 (4.52%) | 2.9285 (1.94%) | 4.4618 (0.65%) | 155 |
| `instantaneous` | 1.6332 (9.68%) | 2.1440 (4.52%) | 2.9061 (1.29%) | 3.4657 (0.65%) | 155 |
| `paper_no_memory` | 0.6332 (9.68%) | 1.1440 (4.52%) | 1.9061 (1.29%) | 2.4657 (0.65%) | 155 |
| `paper_no_decay` | 0.6740 (9.68%) | 1.8692 (4.52%) | 3.1191 (1.94%) | 4.7083 (0.65%) | 155 |
| baseline `quota` | 500 (0.00%) | — | — | — | 155 |
| baseline `rate_per_minute` | 500 (0.00%) | — | — | — | 155 |

#### Benign holdout replay (`benign_test` role)

| Defense | Requests | Accounts | Scored | Windows | L1 | L2 | L3 | L4 | Actions |
|---|---|---|---|---|---|---|---|---|---|
| `full` | 15,330 | 156 | 45 | 128 | 16 (10.3% all / 35.6% scored) | 9 (5.8% all / 20.0% scored) | 6 (3.8% all / 13.3% scored) | 2 (1.3% all / 4.4% scored) | challenge 204, refuse 2, serve 15,124 |
| `instantaneous` | 15,330 | 156 | 45 | 126 | 15 (9.6% all / 33.3% scored) | 8 (5.1% all / 17.8% scored) | 5 (3.2% all / 11.1% scored) | 2 (1.3% all / 4.4% scored) | challenge 193, refuse 202, serve 14,935 |
| `quota` | 15,330 | 156 | 45 | 0 | N/A | N/A | N/A | 0 (0.0% all / 0.0% scored) | serve 15,330 |

Highest level reached per account, by client type (level: accounts):

| Defense | batch | bursty | repetitive | short | specialist |
|---|---|---|---|---|---|
| `full` | L0: 5 | L0: 8 | L0: 17, L1: 5 | L0: 110 | L1: 2, L2: 3, L3: 4, L4: 2 |
| `instantaneous` | L0: 5 | L0: 8 | L0: 18, L1: 4 | L0: 110 | L1: 3, L2: 3, L3: 3, L4: 2 |
| `quota` | L0: 5 | L0: 8 | L0: 22 | L0: 110 | L0: 11 |

Window statistics (benign):

| Defense | Windows | Min p-value | Median p-value | Max statistic | Mean statistic | Windows with statistic > 0 | First L4 at attempt |
|---|---|---|---|---|---|---|---|
| `full` | 128 | 0.007812 | 0.4766 | 7.319 | 0.5078 | 59 | 200 |
| `instantaneous` | 126 | 0.007812 | 0.4961 | 4.852 | 0.9593 | 125 | 100 |
| `quota` | 0 | — | — | — | — | 0 | — |

#### Detectability (seed 42, budget 2,000, one `extractor` account)

| Defense | Requests | Windows | Min p-value | Median p-value | Max statistic | Mean statistic | Highest level | Actions | First L4 at attempt |
|---|---|---|---|---|---|---|---|---|---|
| `full` | 2,000 | 20 | 0.6875 | 0.8047 | 0 | 0 | L0 | serve 2,000 | — |
| `instantaneous` | 2,000 | 20 | 0.6875 | 0.8047 | 0.3747 | 0.2235 | L0 | serve 2,000 | — |
| `quota` | 2,000 | 0 | — | — | — | — | L4 | refuse 1,500, serve 500 | 501 |

### SQuAD

#### Workloads (`task-clients-v1`, synthetic arrival times)

| Role | Requests | Accounts | Scored (≥100 req) | Short (<100) | Source documents | Unique prompts |
|---|---|---|---|---|---|---|
| fitting | 13,170 | 175 | 55 | 120 | 65 | 13,170 |
| score_calibration | 12,249 | 105 | 38 | 67 | 44 | 8,229 |
| policy_calibration | 10,443 | 152 | 28 | 124 | 35 | 8,079 |
| benign_test | 14,011 | 188 | 36 | 152 | 50 | 10,431 |

Client mix per role (accounts / requests / min–max requests per account):

| Role | batch | bursty | repetitive | short | specialist |
|---|---|---|---|---|---|
| fitting | 6 / 2,608 / 108–500 | 11 / 3,255 / 255–300 | 28 / 2,788 / 88–100 | 119 / 2,364 / 4–20 | 11 / 2,155 / 155–200 |
| score_calibration | 4 / 1,569 / 69–500 | 6 / 1,645 / 145–300 | 21 / 6,030 / 30–300 | 65 / 1,290 / 10–20 | 9 / 1,715 / 115–200 |
| policy_calibration | 3 / 1,341 / 341–500 | 5 / 1,474 / 274–300 | 12 / 3,546 / 246–300 | 123 / 2,451 / 11–20 | 9 / 1,631 / 31–200 |
| benign_test | 5 / 2,082 / 82–500 | 8 / 2,172 / 72–300 | 18 / 5,370 / 270–300 | 150 / 2,988 / 8–20 | 7 / 1,399 / 199–200 |

#### Calibration

- σ (RBF bandwidth): 1.050700; window 100, 20 reference windows
- Encoder: `BAAI/bge-small-en-v1.5` @ `5c38ec7c405e`, cls pooling, normalized=True, max 512 tokens; 35,862 texts encoded, 54 truncated
- Truncated by role: fitting 18, score calibration 22, policy calibration 14
- Score-calibration windows: 108; policy-calibration accounts/horizons: 152/152, of which 124 unscored
- Null MMD window scores (n=108): min 0.01074, median 0.01613, mean 0.02344, p90 0.05179, p95 0.07183, p99 0.08719, max 0.1046
- Ville (e-value) thresholds: 2.3026, 2.9957, 3.9120, 4.6052  (= ln 10, ln 20, ln 50, ln 100)

Accumulator thresholds for L1–L4 (achieved policy-calibration escalation rate in brackets; targets 10% / 5% / 2% / 1%):

| Accumulator | L1 | L2 | L3 | L4 | Units |
|---|---|---|---|---|---|
| `paper` | 0.8010 (9.87%) | 2.0046 (4.61%) | 3.6171 (1.97%) | 4.8163 (0.66%) | 152 |
| `instantaneous` | 1.6003 (9.87%) | 2.2064 (4.61%) | 3.0819 (1.97%) | 4.6913 (0.00%) | 152 |
| `paper_no_memory` | 0.6003 (9.87%) | 1.2064 (4.61%) | 2.0819 (1.97%) | 3.6913 (0.00%) | 152 |
| `paper_no_decay` | 0.8259 (9.87%) | 2.1252 (4.61%) | 3.9170 (1.97%) | 5.1855 (0.66%) | 152 |
| baseline `quota` | 500 (0.00%) | — | — | — | 152 |
| baseline `rate_per_minute` | 500 (0.00%) | — | — | — | 152 |

#### Benign holdout replay (`benign_test` role)

| Defense | Requests | Accounts | Scored | Windows | L1 | L2 | L3 | L4 | Actions |
|---|---|---|---|---|---|---|---|---|---|
| `full` | 14,011 | 188 | 36 | 107 | 18 (9.6% all / 50.0% scored) | 6 (3.2% all / 16.7% scored) | 4 (2.1% all / 11.1% scored) | 1 (0.5% all / 2.8% scored) | challenge 3, refuse 1, serve 14,007 |
| `instantaneous` | 14,011 | 188 | 36 | 107 | 20 (10.6% all / 55.6% scored) | 7 (3.7% all / 19.4% scored) | 4 (2.1% all / 11.1% scored) | 0 (0.0% all / 0.0% scored) | challenge 304, serve 13,707 |
| `quota` | 14,011 | 188 | 36 | 0 | N/A | N/A | N/A | 0 (0.0% all / 0.0% scored) | serve 14,011 |

Highest level reached per account, by client type (level: accounts):

| Defense | batch | bursty | repetitive | short | specialist |
|---|---|---|---|---|---|
| `full` | L0: 5 | L0: 8 | L0: 7, L1: 11 | L0: 150 | L1: 1, L2: 2, L3: 3, L4: 1 |
| `instantaneous` | L0: 5 | L0: 8 | L0: 5, L1: 13 | L0: 150 | L2: 3, L3: 4 |
| `quota` | L0: 5 | L0: 8 | L0: 18 | L0: 150 | L0: 7 |

Window statistics (benign):

| Defense | Windows | Min p-value | Median p-value | Max statistic | Mean statistic | Windows with statistic > 0 | First L4 at attempt |
|---|---|---|---|---|---|---|---|
| `full` | 107 | 0.01835 | 0.3303 | 5.697 | 0.6785 | 56 | 200 |
| `instantaneous` | 107 | 0.01835 | 0.3303 | 3.998 | 1.051 | 97 | — |
| `quota` | 0 | — | — | — | — | 0 | — |

#### Detectability (seed 42, budget 2,000, one `extractor` account)

| Defense | Requests | Windows | Min p-value | Median p-value | Max statistic | Mean statistic | Highest level | Actions | First L4 at attempt |
|---|---|---|---|---|---|---|---|---|---|
| `full` | 2,000 | 20 | 1 | 1 | 0 | 0 | L0 | serve 2,000 | — |
| `instantaneous` | 2,000 | 20 | 1 | 1 | -0 | 0 | L0 | serve 2,000 | — |
| `quota` | 2,000 | 0 | — | — | — | — | L4 | refuse 1,500, serve 500 | 501 |

## Other observations from this session

- **A `just paper-tasks --tasks ag_news --seeds 42 --budgets 2000` run overlapped this one** from 09:34 to about 10:00 UTC, when it was stopped by hand. It spent that time downloading `Qwen/Qwen3.5-2B-Base`. The 4.55 GB weight file finished at 09:57 and is fully cached, but the job was stopped before the first optimizer step. It wrote `artifacts/tasks/ag_news/s42-b2000/attacker/training-run.json` (a training fingerprint, 09:34) but no adapter. `data/tasks/ag_news/runs/s42-b2000/` was last written at 10:32 by this run's detectability step, which prepares the seed-42 budget-2,000 prompts.
- **`data/tasks/ag_news/legitimate-references.json` (54.5 MB) was written at 09:45**, as this run started. It holds the pooled legitimate references used to build the workloads.
- **The study's CPU-bound embedding stage took about 52 minutes** (09:45 to 10:38), with contention for its first half. A GPU encoder would take a few minutes.
- **Test suite at `866ade6`:** 6 failed, 98 passed, and 4 skipped (`test_deception`, `test_harness_api`, three in `test_start`, `test_tasks`), as that commit's message said. After pulling `8302b57` and applying this branch's fix: 240 passed and 5 skipped.

## Notes on reading the results

- **Escalation counts are cumulative.** For example, AG News `full` reaches L1 for 16 accounts: 7 that peak at L1, 3 at L2, 4 at L3, and 2 at L4.
- **`full` and `instantaneous` act differently at L4.** On AG News, both defenses take 2 benign accounts to L4. `full` refuses 2 requests and `instantaneous` refuses 202, because under `instantaneous` an account that reaches L4 at its first window, at attempt 100, stays refused for the rest of its horizon.
- **The `paper` accumulator's attacker statistic stays at 0.** The decayed CUSUM subtracts a drift of 1.0 from each window's evidence and clamps at zero. Evidence that never exceeds the drift leaves the statistic at 0. Under `instantaneous`, AG News attacker windows score up to 0.37, which is below the L1 threshold of 1.63.
- **SQuAD attacker p-values are 1.0 in every window.** Each attacker window is closer to the fitted references than every null calibration window. The `instantaneous` statistic printed as `-0` is a signed zero from that computation.
- **The calibration targets are 10% / 5% / 2% / 1% of policy-calibration accounts.** The achieved column is that rate as calibrated, with unscored accounts in the denominator. L4 lands below 1% because each calibration unit is one of about 150 accounts, which allows only discrete steps.
- **The detectability check is a development diagnostic, not a power estimate** (`purpose` in `detectability.json`). It replays one account with one seed at budget 2,000, with request times one second apart.

## Artifact provenance

| File | SHA-256 |
|---|---|
| `protocol.json` | `c0657ac1cd326a30bcbd609ef0b021735275719ece327dc6e5dfb4b4d1b7491d` |
| `tasks.json` | `945d906874c1ca21288699915af8e342f9c5074f7f749930835a4bb0f25ae9cf` |
| `ag_news/calibration/calibration.json` | `6a105e153c32fdfe42567edf8ff03c2708fd890b041ce8771dae816e256fab90` |
| `squad/calibration/calibration.json` | `3f61f9f3e1fca2bc4f88ea1afff79e342e70f679802094a40b5cd7e1387016ce` |

| Calibration input | AG News | SQuAD |
|---|---|---|
| fitting | `92b171f2e130a1a7f2ac1dbc9adb6e17c8d6872f0602c4950c9aaaab86e95436` | `605a84e7e60be9c88de9c6354259c16b22d47d41b2ac3ce42f1c00ce761d6da9` |
| score calibration | `ef3ae587b290688e9dab830713431eaec3b74bdade459171f54dbaf96987e962` | `0530881a48be03a83b6b16f086c1864e131349ac2fb211ef34ed0c5bb693cf7e` |
| policy calibration | `ba6b872ca4318535544ff6b9c8b1c7978fd7a3a5a4289d74acd59b93ea8348c7` | `3ec96c61bf5cb384d8bbedb9e62d630454f139052e5fb21f08edf420eef44973` |
| references | `9abfeacac4d61e9ed2680c98251e26890a59704c50cc916055e1569a0d05f61b` | `fe8e5e6eb33b78e063988aa5e63b5d823f072423dcb28c21752c19dcb65efcb8` |

Files in `artifacts/start-v2/`, none of them committed:

- `protocol.json`, `tasks.json`, `status.json`
- `<task>/calibration/{calibration.json, references.npy}` and `<task>/calibration-report.json`
- `<task>/workloads/{fitting, score_calibration, policy_calibration, benign_test}.json`, which hold every synthetic request with its account and arrival time
- `<task>/benign-test/{full, instantaneous, quota}.json`, plus `.decisions.jsonl` with one policy decision per request: 15,330 for AG News and 14,011 for SQuAD
- `<task>/detectability.json` and `<task>/detectability/{full, instantaneous, quota}.jsonl`, with 2,000 decisions each

## Machine-readable copy

[`readings.json`](readings.json) holds every small artifact verbatim:

- `protocol.json`, including the full training recipe TOML, and `tasks.json`
- both full `calibration.json` files, with every threshold and all 127 and 108 null window scores
- both calibration reports
- every workload summary, including per-account request counts
- all six benign replay summaries and both `detectability.json` files

It also has the derived statistics used above: action and level counts, window p-value and statistic summaries, peak levels by client type, and the first L4 attempt.

## Training recipe frozen in the protocol

The recipe was frozen but never used, because no training ran. Each run copies it with that run's dataset, output directory, and seed.

| Setting | Value |
|---|---|
| Student | `Qwen/Qwen3.5-2B-Base` @ `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |
| Adapter | LoRA rank 16, alpha 32, dropout 0.05, `all-linear`, no QLoRA |
| Optimization | learning rate 2e-4, 3 epochs, batch 4 × 4 accumulation (effective 16), warmup 0.03, NLL loss |
| Precision and memory | bf16, sdpa attention, gradient checkpointing |
| Sequence | max length 2,048, EOS `<|im_end|>`, thinking disabled |
| Checkpointing | save every 100 steps, keep 2, no best-checkpoint selection |
| Seed | 42 |

The recipe sweep would have selected the learning rate from {1e-4, 2e-4} and the epochs from {1, 2, 3} on the development suite.
