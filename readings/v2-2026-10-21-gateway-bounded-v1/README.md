# v2: gateway study `artifacts/gateway-bounded-v1` (started 2026-10-21): undefended ledgers

This is an **interim** record of the shared gateway study, which is still running. It holds the complete undefended ledgers: the teacher's responses to every attacker campaign, collected directly from the teacher with no harness in the path. The protected campaigns, students, and final results will be added when the run finishes.

Every ledger, campaign summary, and execution contract is committed under [`ledgers/`](ledgers/). [`ledgers/manifest.json`](ledgers/manifest.json) lists each file's SHA-256, size, record count, token count, and generation window. Each campaign directory, including `development/campaign/`, also holds its per-attempt record (`attempts.jsonl`) and event log (`campaign.events.jsonl`), added on 2026-10-31 and stored in Git LFS; the manifest predates them and does not list them.

## Run so far

| Item | Value |
|---|---|
| Command | `just start --concurrency 8` (gateway mode, default run directory) |
| Code | `4bb6381` (protocol frozen), resumed 2026-10-29 on `3a50a1a` (offline vLLM restart fix, PR #9) |
| Teacher | `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, vLLM 0.31.0, fp8, non-thinking, temperature 0, seed per campaign |
| Endpoint | teacher API `http://127.0.0.1:8000` over vLLM `:8001`; the harness (`:8003`) was not involved |
| Tasks and output caps | AG News 16, SQuAD 64, GSM8K 256, coding 512 tokens |
| Attacks / seeds / budget | A1, A2, A5, A7 / 42, 1, 20 / 6,000 submissions per campaign |
| Calibration adequacy | passed (`calibration-resolution-v2`): 199 non-overlapping score windows (99 required), 100 of 125 policy horizons scored (100 required), empirical p-value floor 0.0017 |
| Coding pool | 50,000 eligible of 235,024 `nvidia/OpenCodeInstruct` rows qualified (21.3%). Excluded: 130,665 references failing their own tests, 30,615 malformed or unsupported, 20,426 duplicates, 2,043 tests a null stub passes, 782 timeouts, 493 test overlaps |

### Timeline (UTC)

| When | Event |
|---|---|
| 10-21 13:27 | First start stops in the coding build: the bubblewrap sandbox could not see the uv-managed interpreter. Fixed in `4bb6381`. |
| 10-21 13:29 – 15:30 | Task pools (coding qualification about 2 h) and external A7 sources |
| 10-21 15:37 – 16:04 | Detector calibration on CPU; adequacy gate passed |
| 10-21 16:05 – 10-24 04:52 | 12 undefended campaigns |
| 10-24 04:52 – 06:40 | Development campaign (2,000 submissions) |
| 10-24 07:30 – 10-25 01:42 | Six-candidate recipe sweep (learning rate 1e-4 / 2e-4 × 1 / 2 / 3 epochs) |
| 10-25 09:59 | vLLM restart fails: a TLS-intercepting network returned a self-signed certificate while vLLM listed Hub files. The run stopped. |
| 10-29 09:21 | Resumed with offline vLLM startup; every completed campaign and the frozen recipe were reused |
| 10-29 09:36 – | Teacher test-suite evaluation before the protected campaigns (A1 seed 42 first) |

## Undefended campaigns

All 13 campaigns received every submission, with 0 failed, 0 blocked, 0 challenged, 0 deadline stops, and none pending.

| Ledger | Records | Size | Output tokens | Duration | Generated (UTC) | SHA-256 |
|---|---|---|---|---|---|---|
| [`A1/s1-b6000/undefended`](ledgers/A1/s1-b6000/undefended/ledger.jsonl) | 6,000 | 12.6 MB | 589,192 | 5.40 h | 10-22 12:36 – 10-22 18:00 | `071f530cc3ca751d…` |
| [`A1/s20-b6000/undefended`](ledgers/A1/s20-b6000/undefended/ledger.jsonl) | 6,000 | 12.7 MB | 596,545 | 5.45 h | 10-23 08:38 – 10-23 14:05 | `8ce2039762487241…` |
| [`A1/s42-b6000/undefended`](ledgers/A1/s42-b6000/undefended/ledger.jsonl) | 6,000 | 12.7 MB | 588,239 | 5.39 h | 10-21 16:05 – 10-21 21:28 | `5661b50f2bf27877…` |
| [`A2/s1-b6000/undefended`](ledgers/A2/s1-b6000/undefended/ledger.jsonl) | 6,000 | 12.6 MB | 589,192 | 5.38 h | 10-22 18:00 – 10-22 23:23 | `e2cbc64385291a0b…` |
| [`A2/s20-b6000/undefended`](ledgers/A2/s20-b6000/undefended/ledger.jsonl) | 6,000 | 12.7 MB | 596,545 | 5.45 h | 10-23 14:05 – 10-23 19:32 | `0129d5c2a346f1e3…` |
| [`A2/s42-b6000/undefended`](ledgers/A2/s42-b6000/undefended/ledger.jsonl) | 6,000 | 12.7 MB | 588,309 | 5.74 h | 10-21 21:28 – 10-22 03:12 | `55a778db0121e830…` |
| [`A5/s1-b6000/undefended`](ledgers/A5/s1-b6000/undefended/ledger.jsonl) | 6,000 | 12.7 MB | 590,433 | 3.40 h | 10-22 23:23 – 10-23 02:47 | `8cfd47a246ae461b…` |
| [`A5/s20-b6000/undefended`](ledgers/A5/s20-b6000/undefended/ledger.jsonl) | 6,000 | 12.7 MB | 595,051 | 3.49 h | 10-23 19:32 – 10-23 23:02 | `7c38442578ed50bb…` |
| [`A5/s42-b6000/undefended`](ledgers/A5/s42-b6000/undefended/ledger.jsonl) | 6,000 | 12.7 MB | 587,246 | 3.46 h | 10-22 03:12 – 10-22 06:40 | `4f03e7228c8da209…` |
| [`A7/s1-b6000/undefended`](ledgers/A7/s1-b6000/undefended/ledger.jsonl) | 6,000 | 18.8 MB | 635,518 | 5.85 h | 10-23 02:47 – 10-23 08:38 | `3855627a53b18e6e…` |
| [`A7/s20-b6000/undefended`](ledgers/A7/s20-b6000/undefended/ledger.jsonl) | 6,000 | 18.7 MB | 633,631 | 5.84 h | 10-23 23:02 – 10-24 04:52 | `fd41f0d763fe79f7…` |
| [`A7/s42-b6000/undefended`](ledgers/A7/s42-b6000/undefended/ledger.jsonl) | 6,000 | 18.7 MB | 632,739 | 5.94 h | 10-22 06:40 – 10-22 12:36 | `975e4a3d08509f7c…` |
| [`development/campaign`](ledgers/development/campaign/ledger.jsonl) | 2,000 | 4.2 MB | 196,300 | 1.79 h | 10-24 04:52 – 10-24 06:40 | `7cdf77bbb61d7438…` |
| **Total** | **74,000** | **174.4 MB** | **7,418,940** | | | |

| Task | Records | Teacher output tokens | Share of tokens |
|---|---|---|---|
| gsm8k | 18,500 | 4,598,378 | 62.0% |
| coding | 18,500 | 2,631,988 | 35.5% |
| squad | 18,500 | 145,121 | 2.0% |
| ag_news | 18,500 | 43,453 | 0.6% |

Finish reasons: 58,296 `stop` and 15,704 `length` (cut off at the task cap, almost all GSM8K and coding).

### Observations

- **The teacher is deterministic across campaigns.** For each seed, A1 and A2 query the same 6,000 prompts in different orders. All 6,000 completions are byte-identical between them, at all three seeds.
- **A5 runs about 37% faster** (3.4–3.5 h against 5.4–5.9 h), and **A7 ledgers are about 48% larger** (18.7 MB against 12.7 MB), because A7's external news sources have longer prompts.
- **Development sweep training data:** the usability review accepted 1,607 of the 2,000 development responses and rejected all 393 `length` truncations. That left only 124 of the 500 GSM8K examples, which is consistent with the −14.5 pp GSM8K development result. GSM8K's 256-token cap is too short for this teacher; v0 found the same.
- **Throughput** is dominated by GSM8K (about 8 s per request at the cap) and coding (about 4 s); SQuAD and AG News take under 0.3 s.

## Development transfer gate (from `recipe.json`)

Selected recipe: learning rate 1e-4, 3 epochs. `transfer_gate_passed` is false overall, but AG News qualified, so the study continues. Domains without qualified transfer stay in every report labelled uninformative.

| Domain | Base student | Distilled | Gain (pp) | 95% CI | Minimum | Qualified |
|---|---|---|---|---|---|---|
| AG News | 66.6 | 86.6 | +20.0 | 16.4 – 23.8 | 2.0 | yes |
| SQuAD | 75.7 | 78.6 | +2.8 | 0.8 – 5.0 | 2.0 | no (lower bound below the margin) |
| Coding | 57.2 | 57.4 | +0.2 | −3.8 – 4.0 | 2.0 | no |
| GSM8K | 40.5 | 26.0 | −14.5 | −20.4 – −8.6 | 2.0 | no |

## Sources and licenses

Each ledger record keeps its source prompt. The repository is public, and the licenses of the prompt sources differ:

| Corpus | License | Records |
|---|---|---|
| `nvidia/OpenCodeInstruct` | cc-by-4.0 | 14,000 |
| `openai/gsm8k` | mit | 14,000 |
| `fancyzhx/ag_news` | unknown | 14,000 |
| `rajpurkar/squad` | cc-by-sa-4.0 | 14,000 |
| `bigcode/self-oss-instruct-sc2-exec-filter-50k` | odc-by | 4,500 |
| `MU-NLPC/Calc-math_qa` | apache-2.0 | 4,500 |
| `SetFit/bbc-news` | unknown | 4,500 |
| `lucadiliello/newsqa` | unknown | 4,500 |

AG News, BBC News, and NewsQA have no recorded license, and NewsQA's CNN stories carry their own redistribution terms. SQuAD is CC BY-SA 4.0 and OpenCodeInstruct CC BY 4.0, both requiring attribution, with share-alike for SQuAD. These ledgers were published at the repository owner's direction. Teacher completions come from `Qwen/Qwen3.5-9B`. The `contract.json` files store only a SHA-256 of the teacher credential (here the hash of an empty token), never the credential itself.

