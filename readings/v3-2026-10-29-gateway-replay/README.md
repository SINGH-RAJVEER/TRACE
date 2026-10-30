# v3: harness-protected campaigns on the bounded gateway study (from 2026-10-29)

This is the continuation of the shared gateway study documented in [v2](../v2-2026-10-21-gateway-bounded-v1/). It reuses v2's undefended 6,000-submission ledgers and adds the harness-protected campaigns. It is interim: the [protected campaigns](#protected-campaigns) below are complete for A1 only, and student training and evaluation are pending.

## Scope change on 2026-10-30

The run started with v2's frozen protocol: seeds 42, 1 and 20, the `full`, `instantaneous` and `quota` defenses, and attacks A1, A2, A5 and A7. That original protocol is kept as [protocol.json](protocol.json). On 2026-10-30 the study was narrowed to what the paper needs and resumed in the same run directory, `artifacts/gateway-bounded-v1`, under [protocol-v3-continuation.json](protocol-v3-continuation.json):

- seed 42 only, 6,000 submissions per campaign, the `full` harness only;
- attacks A1, A2, A5, A6 and A7. A6 had no v2 undefended ledger, so the run collects one.

Everything else is unchanged: the calibration, benign workloads, development campaign, selected recipe (learning rate 1e-4, three epochs) and undefended ledgers are reused after their contract and hash checks. The A1 `instantaneous` and `quota` campaigns finished before the change and are recorded below as extra results. The A2 `full` campaign was interrupted at 1,536 responses when the old runner was stopped; an interrupted protected campaign cannot resume without resetting harness state, so it is archived under `harness/A2/s42-b6000/full.interrupted-2026-10-30/` and A2 is collected again. The resumed run uses code from PR #11 and PR #12 (`93c9f1a`), which only change launcher defaults and share the bare-teacher benchmark across campaigns.

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

Bare-teacher test scores from the first teacher slot (`Qwen3.5-9B`, fp8, non-thinking): AG News 85.38% accuracy on 7,600 items, SQuAD 88.82 F1 on 10,570 items. GSM8K and coding were interrupted and rerun in the resumed study.

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
