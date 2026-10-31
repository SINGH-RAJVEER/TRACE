# v3: harness-protected campaigns on the bounded gateway study (from 2026-10-29)

This is the continuation of the shared gateway study documented in [v2](../v2-2026-10-21-gateway-bounded-v1/). It reuses v2's undefended 6,000-submission ledgers and adds the harness-protected campaigns. It is interim: the [protected campaigns](#protected-campaigns) below are complete for A1 only, and student training and evaluation are pending.

## Scope change on 2026-10-30

The run started with v2's frozen protocol: seeds 42, 1 and 20, the `full`, `instantaneous` and `quota` defenses, and attacks A1, A2, A5 and A7. That original protocol is kept as [protocol.json](protocol.json). On 2026-10-30 the study was narrowed to what the paper needs and resumed in the same run directory, `artifacts/gateway-bounded-v1`, under [protocol-v3-continuation.json](protocol-v3-continuation.json):

- seed 42 only, 6,000 submissions per campaign, the `full` harness only;
- attacks A1, A2, A5, A6 and A7. A6 had no v2 undefended ledger, so the run collects one.

Everything else is unchanged: the calibration, benign workloads, development campaign, selected recipe (learning rate 1e-4, three epochs) and undefended ledgers are reused after their contract and hash checks. The A1 `instantaneous` and `quota` campaigns finished before the change and are recorded below as extra results. The A2 `full` campaign was interrupted at 1,536 responses when the old runner was stopped; an interrupted protected campaign cannot resume without resetting harness state, so it is archived under `harness/A2/s42-b6000/full.interrupted-2026-10-30/` and A2 is collected again. Two later A2 attempts were also interrupted; see [A2 protected attempts](#a2-protected-attempts). The resumed run uses code from PR #11 and PR #12 (`93c9f1a`), which only change launcher defaults and share the bare-teacher benchmark across campaigns.

A full-attacker-pool variant was started on 2026-10-30 in `artifacts/gateway-full-s42-v1` and abandoned after 559 A1 undefended responses, because no full-pool undefended baseline exists. Its protocol and partial ledger are archived under `archive/gateway-full-s42-v1/`.

## Protected campaigns

A1 sends one account's multiplexed requests across all four domains: 1,500 prompts each from AG News, SQuAD, GSM8K and coding, in the same order as the v2 undefended A1/s42 campaign.

| Defense | Responses received | Refused | Attacker's highest level | Scored windows (min p) | Elapsed |
|---|---:|---:|---|---:|---:|
| Undefended (v2) | 6,000 | 0 | — | — | 5.4 h |
| `full` | 6,000 | 0 | L0 | 60 (0.947) | 5.7 h |
| `instantaneous` | 6,000 | 0 | L0 | 60 (0.947) | 5.7 h |
| `quota` | 1,215 | 4,785 | L4 from submission 1,216 | — | 1.7 h |

Under `full` and `instantaneous` the attacker was never escalated: no detector window came close to rejecting the benign reference (smallest p = 0.947), so the full ledger was released. Only `quota` limited extraction, by refusing every request after the 1,215th. Released output tokens: 588,653 under `full` and `instantaneous`, 118,872 under `quota`.

Benign utility on the held-out benign workload (24,047 requests), measured live through each defense:

| Defense | Served | Completion rate | Unconditional utility | Served-only quality | Benign accounts escalated |
|---|---:|---:|---:|---:|---|
| Undefended | 6,968 | 0.290 | 0.254 | 0.877 | — |
| `full` | 6,358 | 0.264 | 0.232 | 0.877 | 1 (to L1) |
| `instantaneous` | 6,346 | 0.264 | 0.232 | 0.878 | 1 (to L2) |
| `quota` | 6,589 | 0.274 | 0.240 | 0.877 | 0 |

Bare-teacher test scores from the first teacher slot (`Qwen3.5-9B`, fp8, non-thinking): AG News 85.38% accuracy on 7,600 items, SQuAD 88.82 F1 on 10,570 items. GSM8K and coding were interrupted and rerun in the resumed study. The interrupted GSM8K teacher run's per-item samples are kept under [teacher/gsm8k.interrupted/](teacher/gsm8k.interrupted/): 1,039 of 1,319 test items, 239 correct, 799 stopped at the 256-token cap. They are partial and are not a teacher score.

## A2 protected attempts

A protected campaign cannot resume after an interruption, because its harness state would restart. Every interrupted A2 `full` attempt is therefore archived and A2 starts again from submission 1. All three archived attempts ran on the same frozen schedule, so their decisions agree on the shared prefix. None escalated the attacker beyond L0.

| Attempt | Directory | UTC window | Submissions decided | Responses released | GSM8K responses at 256-token cap | Scored windows (min p) | Stop cause |
|---|---|---|---:|---:|---:|---:|---|
| 1 | `full.interrupted-2026-10-30` | 10-30 05:02–05:32 | 1,537 | 1,536 (AG News 1,500, GSM8K 36) | 28 | 15 (0.896) | old runner stopped for the scope change above |
| 2 | `full.interrupted-2026-10-30-2` | 10-30 08:45–10:58 | 2,223 | 2,222 (AG News 1,500, GSM8K 722) | 532 | 22 (0.327) | interrupted (SIGINT) to free the GPU for the undefended A2 student ([v5](../v5-2026-10-30-undefended-a2-student/)) |
| 3 | `full.interrupted-2026-10-30-3` | 10-30 12:15–15:05 | 2,481 | 2,480 (AG News 1,500, GSM8K 980) | 724 | 24 (0.327) | killed with the agent session that launched it (see below) |

Attempt 3 was started with `setsid nohup` from an agent shell. That shell ran inside the agent's systemd scope, and when the agent session was stopped at 15:05:46 UTC systemd sent SIGTERM and, 10 seconds later, SIGKILL to the study, vLLM, the teacher API and the gateway, so the runner wrote no exit code. Attempt 4 started on 2026-10-31 at 07:14 UTC as its own systemd user service (`systemd-run --user`) from commit `c102078`, with the same arguments: `--budgets 6000 --seeds 42 --arms full --attacks A1,A2,A5,A6,A7 --concurrency 8 --run-dir artifacts/gateway-bounded-v1`. Partial attempts do not measure defense effectiveness; they are kept for the decision traces.

## A6 undefended ledger

A6 had no v2 undefended ledger, so the resumed run collected one on 2026-10-30 from 07:02 to 08:42 UTC: 6,000 submissions split across six accounts by a seeded permutation, 6,000 responses, zero failures, 588,079 released output tokens, 6,008 s. Non-stop finishes: GSM8K 1,117 and coding 50 at the output cap, SQuAD 1. It is under [ledgers/A6/s42-b6000/undefended/](ledgers/A6/s42-b6000/undefended/) with its contract, campaign summary, attempts and events.

## Development recipe selection

Before any protected collection, the run trained six shared students (Qwen3.5-2B-Base, LoRA) on the 2,000-submission development campaign (1,607 usable responses) and scored each on source-disjoint development suites. Scores are percentages; GSM8K and coding use the selection suite.

| Recipe | AG News | SQuAD | GSM8K | Coding | Mean rank |
|---|---|---|---|---|---:|
| Base student | 66.0 | 77.0 | 39.1 | 56.2 | — |
| lr 1e-4, 1 epoch | 78.8 | 78.7 | 29.8 | 58.2 | 5.0 |
| lr 1e-4, 2 epochs | 84.4 | 80.5 | 26.3 | 59.8 | 3.25 |
| **lr 1e-4, 3 epochs (selected)** | 84.2 | 80.4 | 27.9 | 63.6 | **2.625** |
| lr 2e-4, 1 epoch | 78.6 | 79.3 | 33.0 | 59.0 | 4.25 |
| lr 2e-4, 2 epochs | 85.0 | 80.1 | 27.9 | 63.2 | 3.125 |
| lr 2e-4, 3 epochs | 83.2 | 80.2 | 32.2 | 63.4 | 2.75 |

On the separate qualification split, the selected recipe gained +20.0 pp on AG News (passed the 2 pp minimum), +2.8 pp on SQuAD, −14.5 pp on GSM8K and +0.2 pp on coding (not qualified). Only AG News is informative for transfer, so the overall transfer gate did not pass; the study continues with AG News as the informative domain. Per-trial scores, training manifests, benchmark samples and the six adapters (`adapter_model.safetensors`, stored in Git LFS) are under `development/`. The development ledger itself is in [v2](../v2-2026-10-21-gateway-bounded-v1/ledgers/development/). Each trial directory also holds its prompt inventory (`prompts.jsonl`) and per-response usability reviews (`reviews.jsonl`), and [development-suites/](development-suites/) holds the frozen source-disjoint selection and qualification suites with their split records; these are stored in Git LFS.

## Calibration, workloads and schedules

- `calibration/`: the frozen harness calibration (`calibration.json`, reference embeddings `references.npy`) used by every protected campaign; its adequacy check is [calibration-adequacy.json](calibration-adequacy.json).
- `benign/`: the four synthetic benign workloads (fitting, score calibration, policy calibration, held-out test), with manifests.
- `schedules/`: the frozen seed-42 inventories and request schedules for A1, A2, A5, A6 and A7.
- `run/`: `recipe.json`, `policy-horizons.json`, `tasks.json` and `external.json`.

## Evaluation method checks (2026-10-30)

Final evaluations in this run use two speed-ups merged on 2026-10-30, checked first on real test items. Scripts and raw outputs are under `method-checks/`.

Local students (PR #15) merge the LoRA adapter into the weights and greedy-decode batches of 16. Hugging Face generation of this hybrid model was CPU-bound (one core at 100%, GPU at 5–30%). On 32 test items per domain:

| Comparison | AG News | SQuAD | GSM8K |
|---|---|---|---|
| Base student, batch 1 vs 16: identical text / correct | 32/32, 15 vs 15 | 31/32, 27 vs 28 | 17/32, 10 vs 10 |
| Selected adapter, attached batch 1 vs merged batch 16 | 32/32, 26 vs 26 | 32/32, 22 vs 22 | 22/32, 6 vs 6 |
| Speed-up (base / adapter) | 4.8× / 9× | 4.7× / 8× | 11× / 23× |

The served teacher (PR #16) receives 16 concurrent requests. Against 48 stored one-at-a-time outputs per domain: AG News 48/48 identical (37 → 37 correct), SQuAD 46/48 (41 → 40), GSM8K 5/48 (9 → 12), with GSM8K 23× faster. The GSM8K differences are not caused by concurrency: the same four GSM8K requests sent twice one at a time gave 0/4 identical outputs while a collection shared the server, because vLLM batches each request with whatever else is in flight. Teacher GSM8K scores therefore carry run-to-run variance of a few points.

## Held-out benign policy replay

The replay ran on 2026-10-29 after the 13:11 UTC restart of `just start --concurrency 8`, using the existing frozen calibration and workload under `artifacts/gateway-bounded-v1`. The checkout was based on commit `3599d6e`, with an uncommitted change to run protected collection before teacher evaluations; that change does not alter the replay. The [protocol](protocol.json) identifies the study as `shared-gateway-v4-bounded`. The [calibration adequacy check](calibration-adequacy.json) passed its minimum resolution gate: 199 non-overlapping score windows and 100 scored policy-calibration horizons. This does not establish population error control.

Each defense replayed 24,047 requests across 125 account horizons; 100 horizons had a complete scoring window and 25 did not. The table shows cumulative level reaches, with percentages over all horizons and, in parentheses, scored horizons. Quota implements only L4.

| Defense | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|
| Full | 12: 9.6% (12%) | 3: 2.4% (3%) | 0 | 0 |
| Instantaneous | 12: 9.6% (12%) | 8: 6.4% (8%) | 2: 1.6% (2%) | 2: 1.6% (2%) |
| Quota | — | — | — | 1: 0.8% (1%) |

The JSON summaries retain profile breakdowns and the JSONL files retain per-request replay decisions. The summary `output` fields refer to their original local paths. The quota profile-level scored denominators in the source output are zero; use the overall scored rate as recorded rather than inferring profile rates.

## Source SHA-256

Per-request campaign files (`attempts.jsonl`, `campaign.events.jsonl`), account maps and benchmark samples sit beside the files below. Account credentials are not published; contracts record only their SHA-256.

| File | SHA-256 |
|---|---|
| [harness/A1/s42-b6000/full/campaign/ledger.jsonl](harness/A1/s42-b6000/full/campaign/ledger.jsonl) | `8a4636bfd377b4919584687d34f6daf0f6f8110eca984e5a7ce51f57ab22f289` |
| [harness/A1/s42-b6000/instantaneous/campaign/ledger.jsonl](harness/A1/s42-b6000/instantaneous/campaign/ledger.jsonl) | `4b4a52e4ddf46a8cefb3e742af1343612906f1f1770525d6261d313fc9a1f245` |
| [harness/A1/s42-b6000/quota/campaign/ledger.jsonl](harness/A1/s42-b6000/quota/campaign/ledger.jsonl) | `7265145e7e30dfdd43f0b13ff59285d14aafc2c81f9093d75421b43cc838270b` |
| [harness/A1/s42-b6000/full/decisions.jsonl](harness/A1/s42-b6000/full/decisions.jsonl) | `97e048d361dbed6358264d82fd772458aff398f25bd6398a1418385a987c8f7f` |
| [harness/A1/s42-b6000/instantaneous/decisions.jsonl](harness/A1/s42-b6000/instantaneous/decisions.jsonl) | `e45205b673715a52a3544c2b0d9edca94abade1989f03fdcf4481373558a14b4` |
| [harness/A1/s42-b6000/quota/decisions.jsonl](harness/A1/s42-b6000/quota/decisions.jsonl) | `f5dbe719baac066eead5e89a7adac5e345deb51c18620d8d49ee3f51ccf6aa9e` |
| [harness/A1/s42-b6000/full/utility.json](harness/A1/s42-b6000/full/utility.json) | `40698b46446454e533d20664d3c8b1aae0d63e8e61247ee7231139d1a178f780` |
| [harness/A1/s42-b6000/instantaneous/utility.json](harness/A1/s42-b6000/instantaneous/utility.json) | `d52c331458f129e2af63fb451f6a96b7ab7d7c17cce6bb20d91ab5a5ebc8dd60` |
| [harness/A1/s42-b6000/quota/utility.json](harness/A1/s42-b6000/quota/utility.json) | `b21bd0d563c2ac36d27276ec63d8b9d237cf751a641e73c00839075cb18b606a` |
| [harness/A2/s42-b6000/full.interrupted-2026-10-30/campaign/ledger.jsonl](harness/A2/s42-b6000/full.interrupted-2026-10-30/campaign/ledger.jsonl) | `dd0cc5353ee8601e0a5aee42c647fab3adb0a07e8efc8855c4489fc3f2041d54` |
| [harness/A2/s42-b6000/full.interrupted-2026-10-30-2/campaign/ledger.jsonl](harness/A2/s42-b6000/full.interrupted-2026-10-30-2/campaign/ledger.jsonl) | `46c0bef8cc42a3d6119964f0c71af8b3551ceaa7dc115ada7c4d2b323a3ce784` |
| [harness/A2/s42-b6000/full.interrupted-2026-10-30-2/decisions.jsonl](harness/A2/s42-b6000/full.interrupted-2026-10-30-2/decisions.jsonl) | `40920f9420930cc3b901ea899ae857a5007dd447ce97e7d2d89c15928c589002` |
| [harness/A2/s42-b6000/full.interrupted-2026-10-30-3/campaign/ledger.jsonl](harness/A2/s42-b6000/full.interrupted-2026-10-30-3/campaign/ledger.jsonl) | `e713ae17f13a6afefa7f80d572d3161618651271c9e3eb22be54ccd2abdf7888` |
| [harness/A2/s42-b6000/full.interrupted-2026-10-30-3/decisions.jsonl](harness/A2/s42-b6000/full.interrupted-2026-10-30-3/decisions.jsonl) | `cac339f90f9dcaea90db3612f4827fa4fee778acf4fee0f950888af8bf78be9b` |
| [ledgers/A6/s42-b6000/undefended/ledger.jsonl](ledgers/A6/s42-b6000/undefended/ledger.jsonl) | `df28ce3ec2c052d9d7b2586293b355d3f793612cff7bd506b7d8c2fe8cabbd0c` |
| [utility/undefended.json](utility/undefended.json) | `54d7782baf056ee11163aa3457c3a68041d349e8826901e2fffeec95d314b6fe` |
| [teacher/ag_news/df2e1b0189634222a64c97d8df8bf931.json](teacher/ag_news/df2e1b0189634222a64c97d8df8bf931.json) | `ce1ba11db23a0e9b382590829bcef3728d0952cbe78226f61ffcb0a5874545e2` |
| [teacher/squad/4705f0a99a4b46d1848ca84426117fcb.json](teacher/squad/4705f0a99a4b46d1848ca84426117fcb.json) | `a7aab6541a78e892a4284dce1d01485f69dbf17ca11bc7a103c7f099bdc33105` |
| [protocol-v3-continuation.json](protocol-v3-continuation.json) | `6bee1668d83bb9c977e52164f222fd309768b509925099965d8890b14635daf4` |
| [archive/gateway-full-s42-v1/A1-undefended/ledger.jsonl](archive/gateway-full-s42-v1/A1-undefended/ledger.jsonl) | `fdb934b55fc0fb7fe7ecfbf7ffec1e16fc344f05c4e463fcc681f028b3cd8f79` |
| [full.json](full.json) | `a59c04259aa242cdb9c1237b705b27afdb09ea4feddad2286c3041452e01eaca` |
| [full.jsonl](full.jsonl) | `573885cb5640d04042395457b808b6fbcb7ff8bfc0080dccb7bb16e3480dbc94` |
| [instantaneous.json](instantaneous.json) | `a7654bed5c53d4c15c132cf644f1be5ad816eb0d5a54552751f96e6a7272909b` |
| [instantaneous.jsonl](instantaneous.jsonl) | `19d3147f0664fe4ee2892cb75ae12a7e775c15447bbec44ff6f60ed502a963c7` |
| [quota.json](quota.json) | `9179a26c3610ac14c2e7088eb5eaf4373af8895a4469fb0f461fc46883223fd4` |
| [quota.jsonl](quota.jsonl) | `04ceeca63f44cb2ab0a2990df2d948920b93b0bcb1dabdb73ba584e7ab2b1b74` |
| [calibration-adequacy.json](calibration-adequacy.json) | `1aa5ea9752763b9e18773d77eb65d7cfc48eb59713be3bf2207263c47f0028e1` |
| [protocol.json](protocol.json) | `c11ad1cd68ac2c75d93d016a6586ed21ab02b400fbc099327f54d2ac0b586b64` |
