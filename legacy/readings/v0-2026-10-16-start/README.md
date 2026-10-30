# v0: `just start` run `artifacts/start` (2026-10-16)

This is the first `just start` run on this machine. It used the protocol that existed before `task-study-v2`, wrote its results to the default `artifacts/start`, and stopped at about 17:06 UTC, right after its last harness traffic run, as it began the student stage. It produced three sets of measurements:

- undefended teacher responses for the AG News, SQuAD, and GSM8K attacker pools
- teacher benchmarks on all three official test suites
- the unprotected `before` phase of the harness traffic suite, for every task and seed

It trained no student and never started the harness. Every number below was read or derived from `artifacts/start/` and `data/tasks/<task>/`, both gitignored on the run machine. [`readings.json`](readings.json) holds the same data in machine-readable form. The v1 run in [`../v1-2026-10-17-start-v2/`](../v1-2026-10-17-start-v2/) could not reuse this directory because its protocol changed.

## Summary

- **The teacher, `Qwen3.5-9B` in non-thinking mode with fp8, scored 85.25% on AG News, EM 72.36% / F1 88.83% on SQuAD, and 23.12% on GSM8K.** AG News clears the study's capability target of 80.0, mainly on Sports (97.5%) and Business (92.2%). Sci/Tech is weak at 63.2%.
- **GSM8K's 256-token output limit invalidates its teacher score.** The limit cut off 1,011 of 1,319 test responses (76.6%), and 1,004 had no parsable final answer. In the attacker-pool collection, 2,396 of 3,150 responses (76.1%) hit the limit, and only 24.5% end in a `####` answer. A teacher trained on that would learn truncated reasoning, so the GSM8K cap needs piloting before any GSM8K result means anything. Upstream later called the GSM8K and coding caps "provisional".
- **Teacher responses agree with the dataset gold labels about as often as the teacher scores on the test sets:** 85.6% label match on AG News, 60.1% EM and 82.0% F1 on SQuAD, and 23.7% numeric match on GSM8K. The SQuAD EM is lower than on the test set because the pool stores one gold answer per question, where the validation set has several.
- **Harness traffic reached the `before` phase only.** Against the unprotected teacher, 84–91% of the 192 requests per run succeeded, and the rest hit the 1,024-token client cap (`invalid`). Coding requests suffered most, at 58.6% success. The genuine actor `u01` succeeded on only 63.9%, because 8 of its 12 prompts are coding. No harness decisions were recorded (`decision: unknown`), because the `after` phase against the harness never ran.
- **The GSM8K attempt log was damaged.** `data/tasks/gsm8k/ledger.attempts.jsonl` contains a 926-byte block of NUL bytes after line 1,932. The 1,932 attempts before it (10:19:25–10:55:45 UTC) have no ledger records. The collection then restarted at 11:36:50 and recollected all 3,150 prompts. This is consistent with an unclean stop around 10:56, when `teacher-api.log` also ends. The ledger itself has no NUL bytes, and every one of its 3,150 records comes from the second pass.

## Run record

| Item | Value |
|---|---|
| Command | `just start` (defaults), run directory `artifacts/start` |
| Code | not recorded: every benchmark file says `code_revision: unknown`. `main` that day ran from `fe2b3d0` to `7b72210`. |
| Host | WSL2 (`Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43`), NVIDIA RTX 4000 Ada Generation (20,475 MiB) |
| Packages | `distillation-core 0.1.0`, `httpx 0.28.1`, `torch 2.14.1`, `transformers 5.19.0`, `peft 0.21.2`, `datasets 5.0.1` |
| Protocol | pre-`task-study-v2`: no `protocol_version`, `arms`, capability targets, campaign budgets, or controls |
| Tasks / seeds / budgets | `ag_news`, `squad`, `gsm8k` / 42, 1, 20 / 500, 2,000, 8,000 (GSM8K skipped 8,000, which exceeds its 3,389-prompt attacker pool) |
| Gold reference students | enabled, never trained |
| L4 action | `refuse` |
| Teacher | `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, vLLM 0.31.0 with `quantization=fp8`, bf16, `max_model_len=8192`, through the teacher API at `http://127.0.0.1:8000` |
| Terms review ID | `self-hosted` |
| Training recipe | the same settings as v1. Only two comments differ: v0's says the paper uses Qwen2.5 models, and that attention-only adapters would wait for that switch. |
| Final `status.json` | `{"status": "running"}`, never updated because the run was interrupted |

| File | SHA-256 |
|---|---|
| `artifacts/start/protocol.json` | `30911f63dc0a720efd3bf04cc21477eba160fc1871893c582ac303c7bc686e85` |
| `artifacts/start/tasks.json` | `2019a244f6b77ff22003061ffef9030fed67f8449a48e3eec12ff390e8e84014` |
| `artifacts/start/traffic.json` | `538ae037f7e43e7a246145fc2f00fb3e4558401f1258b44f890350fa7390b592` |

Task pools match the v1 build for AG News and SQuAD (see the v1 file). GSM8K's roles: 3,389 attacker, 746 development, 1,328 fitting, 682 score calibration, 669 policy calibration, 659 benign test, and 1,319 test items, all at a 256-token output limit, scored by numeric exact match.

### Timeline (UTC, 2026-10-16)

| Time | Event |
|---|---|
| 09:45:00 – 09:56:41 | AG News collection: 20,585 responses at 29.4 per second |
| 09:56:41 – 10:19:25 | SQuAD collection: 19,509 responses at 14.3 per second |
| 10:19:25 – 10:55:45 | First GSM8K collection pass: 1,932 attempts, then an unclean stop that left the NUL block and no ledger records |
| 10:56 | `teacher-api.log` ends |
| 11:06 | `protocol.json`, `tasks.json`, and `traffic.json` frozen |
| 11:35 | `status.json` written (`running`) |
| 11:36:50 – 12:35:11 | Second GSM8K collection pass: all 3,150 records, at 0.39 per second because most responses run to 256 tokens |
| 12:35:20 – 12:51:15 | Teacher benchmark, AG News test (7,600 items, 15.9 min) |
| 12:51:19 – 13:42:47 | Teacher benchmark, SQuAD validation (10,570 items, 51.5 min) |
| 13:42:58 – 16:39:37 | Teacher benchmark, GSM8K test (1,319 items, 176.6 min, about 8 s per item at the 256-token cap) |
| 16:39:37 – 17:05:59 | Traffic `before` phase: 9 runs of about 3 minutes each (AG News, SQuAD, GSM8K × seeds 42, 1, 20) |
| 17:05:59 | The last traffic run completes; vLLM shuts down with an `EngineDeadError` and a leaked-semaphore warning. |
| 17:06 | The student's `config.json` and `model.safetensors.index.json` are fetched into the Hugging Face cache: the base-student stage begins. No weights, benchmark, or adapter follows. Whether the run was stopped by hand or failed here is not recorded. |

The vLLM and `start-teacher` logs from this run were overwritten by the v1 run on 2026-10-17. Only `artifacts/logs/teacher-api.log` (3.0 MB, ending 10:56) survives from v0.

### Stages that did not run

Detector calibration (no calibration was written), student baselines, baseline and gold training, the harness API, protected campaigns, the traffic `after` phase, protected students, and reports.
## Teacher benchmarks on the official test suites

Each task has one teacher-slot run. The launcher copied it into every `<seed>-<budget>/benchmarks/{attacker,gold,protected}/` directory: 27 copies each for AG News and SQuAD, and 18 for GSM8K, which ran only budgets 500 and 2,000. No student slot ran.

| Task | Run ID | Items | Correct | Primary score | Status | Finish reason | No parsable answer | Started → completed (UTC) | Duration |
|---|---|---|---|---|---|---|---|---|---|
| ag_news | `583b02a874fe43aeb5b73ea0c0f2f5ae` | 7,600 | 6,479 | accuracy 85.25% | ok 7,600 | stop 7,600 | 0 | 12:35:20 → 12:51:15 | 15.9 min |
| squad | `e2627ac3eb11479c96d69e15dee71fe9` | 10,570 | 7,648 | EM 72.36%, F1 88.83% | ok 10,563, truncated 7 | stop 10,563, length 7 | 0 | 12:51:19 → 13:42:47 | 51.5 min |
| gsm8k | `5185f05fd3214c70b0129f02bbd0d555` | 1,319 | 305 | accuracy 23.12% | ok 308, truncated 1,011 | stop 308, length 1,011 | 1,004 | 13:42:58 → 16:39:37 | 176.6 min |

| Task | Suite SHA-256 | Scorer | Max output tokens | Temperature | Seed | Thinking |
|---|---|---|---|---|---|---|
| ag_news | `94fda5ffd693bf2e14d1a63fd2d90946d53f162ccf22c7534e4c2696f58e63c3` | `generated-answer-v1` | 16 | 0 | 42 | False |
| squad | `47ce56291afc870452182d437a8391bc2d6346a5c83a12ebc4f282260a14932d` | `generated-answer-v1` | 64 | 0 | 42 | False |
| gsm8k | `3a4318e57f476b4891c5c42754690734e7647bc31a496baf6bff79abd334c1df` | `generated-answer-v1` | 256 | 0 | 42 | False |

Latency and token use per item:

| Task | Latency mean / p50 / p90 / p95 / p99 / max (s) | Input tokens mean / p50 / p99 / max | Output tokens mean / p50 / p95 / p99 / max |
|---|---|---|---|
| ag_news | 0.125 / 0.113 / 0.176 / 0.179 / 0.206 / 0.223 | 94.4 / 93 / 152 / 284 | 2.34 / 2 / 4 / 4 / 4 |
| squad | 0.292 / 0.239 / 0.480 / 0.634 / 1.041 / 2.288 | 215.7 / 201 / 492 / 831 | 7.19 / 5 / 18 / 30 / 64 |
| gsm8k | 8.035 / 8.312 / 8.437 / 8.512 / 8.609 / 8.659 | 90.2 / 86 / 156 / 217 | 246.36 / 256 / 256 / 256 / 256 |

AG News accuracy by gold topic:

| Topic | Items | Correct | Accuracy |
|---|---|---|---|
| Business | 1,900 | 1,752 | 92.21% |
| Sci/Tech | 1,900 | 1,201 | 63.21% |
| Sports | 1,900 | 1,853 | 97.53% |
| World | 1,900 | 1,673 | 88.05% |

SQuAD by validation article (all 48), sorted by F1:

| Article | Items | EM | F1 |
|---|---|---|---|
| Martin Luther | 479 | 55.95% | 81.99% |
| Islamism | 200 | 58.50% | 82.32% |
| Victoria (Australia) | 124 | 61.29% | 83.21% |
| Private school | 113 | 61.95% | 83.91% |
| Civil disobedience | 197 | 63.45% | 83.92% |
| Economic inequality | 297 | 62.96% | 84.46% |
| Jacksonville, Florida | 96 | 61.46% | 84.75% |
| Normans | 112 | 67.86% | 85.20% |
| Warsaw | 247 | 64.37% | 85.23% |
| Rhine | 291 | 64.26% | 85.65% |
| Intergovernmental Panel on Climate Change | 104 | 64.42% | 85.73% |
| Force | 206 | 74.76% | 85.83% |
| Teacher | 268 | 69.03% | 85.87% |
| Black Death | 108 | 61.11% | 85.98% |
| Packet switching | 106 | 60.38% | 86.01% |
| Amazon rainforest | 183 | 64.48% | 86.14% |
| Oxygen | 247 | 74.49% | 87.72% |
| Sky (United Kingdom) | 108 | 72.22% | 88.11% |
| Imperialism | 187 | 68.98% | 88.26% |
| French and Indian War | 172 | 62.21% | 88.28% |
| Kenya | 231 | 68.83% | 88.46% |
| Yuan dynasty | 228 | 69.74% | 88.60% |
| Scottish Parliament | 189 | 68.25% | 88.69% |
| 1973 oil crisis | 106 | 71.70% | 88.70% |
| Pharmacy | 129 | 76.74% | 88.86% |
| Computational complexity theory | 197 | 77.16% | 88.87% |
| Chloroplast | 295 | 66.44% | 88.88% |
| Southern California | 180 | 73.89% | 89.13% |
| Steam engine | 215 | 75.35% | 89.16% |
| Nikola Tesla | 511 | 72.02% | 89.23% |
| Genghis Khan | 245 | 73.06% | 89.39% |
| Apollo program | 242 | 76.45% | 90.41% |
| Super Bowl 50 | 810 | 79.38% | 90.86% |
| Ctenophora | 182 | 74.18% | 90.93% |
| Prime number | 153 | 79.08% | 91.05% |
| Geology | 116 | 79.31% | 91.30% |
| European Union law | 243 | 74.07% | 91.51% |
| Newcastle upon Tyne | 264 | 79.17% | 91.72% |
| Huguenot | 217 | 82.03% | 91.77% |
| Victoria and Albert Museum | 295 | 78.98% | 91.89% |
| Construction | 98 | 81.63% | 92.43% |
| United Methodist Church | 191 | 80.10% | 92.43% |
| Harvard University | 112 | 78.57% | 92.76% |
| Doctor Who | 319 | 81.82% | 92.77% |
| Fresno, California | 136 | 80.88% | 93.14% |
| Immune system | 238 | 86.13% | 93.30% |
| University of Chicago | 176 | 81.25% | 93.60% |
| American Broadcasting Company | 407 | 84.52% | 95.09% |

Environment recorded in every run: packages `distillation-core 0.1.0`, `httpx 0.28.1`, `torch 2.14.1`, `transformers 5.19.0`, `peft 0.21.2`, `datasets 5.0.1`; platform `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43`; `code_revision` `unknown`; `demo` False.

## Harness traffic suite: before phase only

Protocol `synthetic-traffic-v1`, seed 42, 60 s schedule, 192 requests, SHA-256 `27f2da14fec9848ac8d74b75c9d552f22f824f6cde5020d6ab2a12ef55220da1`. All nine task × seed runs replayed the same general-chat suite. Its categories are not the task prompts:

| Actor | Label | Scenario | Requests |
|---|---|---|---|
| u03 | genuine | ordinary_repeat_user | 12 |
| u01 | genuine | ordinary_repeat_user | 12 |
| u02 | genuine | ordinary_repeat_user | 12 |
| u04 | genuine | ordinary_repeat_user | 12 |
| u00 | distillation | ramping_extraction | 108 |
| u05 | distillation | steady_extraction | 36 |

Requests by category: coding 36, writing 32, math 24, knowledge 24, reasoning 24, structured 24, learning 8, planning 4, summarization 4, translation 4, batch_work 4, data 4.

Client settings: max 1024 output tokens, temperature 0, thinking False, timeout 120.0 s, 32 in flight, 10.0 s windows. The `before` target is the unprotected teacher API (`service_revision` `unprotected`). The `after` target, never reached, was the harness on :8003 (`mmd-paper-refuse`).

| Task / seed | Run ID | Started (UTC) | Wall time | Requests | Succeeded | Invalid (hit 1,024-token cap) | Latency p50 / p95 / max (s) | Output tokens mean / p50 / max | Queue delay p95 (s) |
|---|---|---|---|---|---|---|---|---|---|
| ag_news/s1 | `a3191931` | 16:42:34 | 182.5 s | 192 | 164 (85.42%) | 28 | 19.79 / 48.14 / 48.34 | 542.9 / 433 / 1024 | 78.203 |
| ag_news/s20 | `6d3f8636` | 16:45:37 | 175.9 s | 192 | 163 (84.90%) | 29 | 18.75 / 46.27 / 46.37 | 537.2 / 414 / 1024 | 71.672 |
| ag_news/s42 | `6277251d` | 16:39:37 | 176.6 s | 192 | 167 (86.98%) | 25 | 19.68 / 46.33 / 46.73 | 536.6 / 453 / 1024 | 74.011 |
| gsm8k/s1 | `12cfd1f9` | 17:00:10 | 176.0 s | 192 | 168 (87.50%) | 24 | 19.36 / 46.30 / 46.41 | 539.5 / 427 / 1024 | 73.202 |
| gsm8k/s20 | `b5b10bc6` | 17:03:07 | 172.2 s | 192 | 166 (86.46%) | 26 | 23.55 / 46.33 / 46.43 | 540.8 / 521 / 1024 | 74.903 |
| gsm8k/s42 | `b0b17c3b` | 16:57:13 | 177.1 s | 192 | 166 (86.46%) | 26 | 18.42 / 46.30 / 46.41 | 543.6 / 409 / 1024 | 74.348 |
| squad/s1 | `2648e602` | 16:51:25 | 171.0 s | 192 | 173 (90.10%) | 19 | 18.58 / 45.93 / 46.43 | 533.6 / 409 / 1024 | 71.090 |
| squad/s20 | `6e2c335a` | 16:54:16 | 176.6 s | 192 | 162 (84.38%) | 30 | 19.01 / 46.41 / 46.52 | 538.1 / 423 / 1024 | 74.080 |
| squad/s42 | `a4a06fbd` | 16:48:33 | 171.9 s | 192 | 175 (91.15%) | 17 | 18.89 / 46.12 / 46.44 | 529.4 / 427 / 1024 | 70.259 |

Per actor, summed over all nine runs:

| Actor | Label | Requests | Succeeded | Invalid |
|---|---|---|---|---|
| u00 | distillation | 972 | 837 (86.11%) | 135 |
| u01 | genuine | 108 | 69 (63.89%) | 39 |
| u02 | genuine | 108 | 108 (100.00%) | 0 |
| u03 | genuine | 108 | 105 (97.22%) | 3 |
| u04 | genuine | 108 | 104 (96.30%) | 4 |
| u05 | distillation | 324 | 281 (86.73%) | 43 |

Per category, summed over all nine runs:

| Category | Requests | Succeeded | Invalid |
|---|---|---|---|
| coding | 324 | 190 (58.64%) | 134 |
| knowledge | 216 | 164 (75.93%) | 52 |
| reasoning | 216 | 185 (85.65%) | 31 |
| planning | 36 | 32 (88.89%) | 4 |
| learning | 72 | 69 (95.83%) | 3 |
| batch_work | 36 | 36 (100.00%) | 0 |
| data | 36 | 36 (100.00%) | 0 |
| math | 216 | 216 (100.00%) | 0 |
| structured | 216 | 216 (100.00%) | 0 |
| summarization | 36 | 36 (100.00%) | 0 |
| translation | 36 | 36 (100.00%) | 0 |
| writing | 288 | 288 (100.00%) | 0 |

## Undefended teacher collection (`data/tasks/<task>/ledger.jsonl`)

| Task | Records | Attempts | Attempt status | First → last attempt (UTC) | Span | Records/s | Finish reason | Quality status |
|---|---|---|---|---|---|---|---|---|
| ag_news | 20,585 | 20,585 | succeeded 20,585 | 09:45:00 → 09:56:41 | 11.7 min | 29.38 | stop 20,585 | pending 20,585 |
| squad | 19,509 | 19,509 | succeeded 19,509 | 09:56:41 → 10:19:25 | 22.7 min | 14.31 | stop 19,491, length 18 | pending 19,491, rejected 18 |
| gsm8k | 3,150 | 5,082 | succeeded 5,082 | 10:19:25 → 12:35:11 | 135.8 min | 0.39 | stop 754, length 2,396 | pending 754, rejected 2,396 |

| Task | Output limit | Input tokens mean / p50 / p99 / max | Output tokens mean / p50 / p95 / p99 / max | Agreement with the gold answer |
|---|---|---|---|---|
| ag_news | 16 | 94.9 / 93 / 156 / 342 | 2.33 / 2 / 4 / 4 / 4 | label match with gold 85.61%, valid label 100.00% |
| squad | 64 | 213.6 / 201 / 431 / 1028 | 7.42 / 6 / 18 / 31 / 64 | exact match with gold 60.12%, f1 with gold 81.96% |
| gsm8k | 256 | 88.4 / 85 / 158 / 241 | 246.65 / 256 / 256 / 256 / 256 | numeric match with gold 23.65%, has final answer 24.51% |

The teacher for every record was `self-hosted-vllm Qwen/Qwen3.5-9B` with temperature=0.0, seed=None, enable_thinking=False. Agreement with gold was computed for this record with SQuAD-style normalization (lowercase, strip punctuation and articles); it is not the repository scorer. AG News compares labels, SQuAD compares the response with the single stored gold answer, and GSM8K compares the number after the last `####`.
