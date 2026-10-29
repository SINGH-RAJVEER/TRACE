# v3: held-out benign policy replay (2026-10-29)

This is an interim reading from the continuation of the shared gateway study documented in [v2](../v2-2026-10-21-gateway-bounded-v1/). It is a CPU simulation of policy decisions on held-out synthetic benign traffic, not a defended collection, live utility measurement, or evidence of real-user false-positive rates. The protected campaigns and final study results remain pending.

The replay ran on 2026-10-29 after the 13:11 UTC restart of `just start --concurrency 8`, using the existing frozen calibration and workload under `artifacts/gateway-bounded-v1`. The checkout was based on commit `3599d6e`, with an uncommitted change to run protected collection before teacher evaluations; that change does not alter the replay. The [protocol](protocol.json) identifies the study as `shared-gateway-v4-bounded`. The [calibration adequacy check](calibration-adequacy.json) passed its minimum resolution gate: 199 non-overlapping score windows and 100 scored policy-calibration horizons. This does not establish population error control.

Each defense replayed 24,047 requests across 125 account horizons; 100 horizons had a complete scoring window and 25 did not. The table shows cumulative level reaches, with percentages over all horizons and, in parentheses, scored horizons. Quota implements only L4.

| Defense | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|
| Full | 12: 9.6% (12%) | 3: 2.4% (3%) | 0 | 0 |
| Instantaneous | 12: 9.6% (12%) | 8: 6.4% (8%) | 2: 1.6% (2%) | 2: 1.6% (2%) |
| Quota | — | — | — | 1: 0.8% (1%) |

The JSON summaries retain profile breakdowns and the JSONL files retain per-request replay decisions. The summary `output` fields refer to their original local paths. The quota profile-level scored denominators in the source output are zero; use the overall scored rate as recorded rather than inferring profile rates.

## Source SHA-256

| File | SHA-256 |
|---|---|
| [full.json](full.json) | `a59c04259aa242cdb9c1237b705b27afdb09ea4feddad2286c3041452e01eaca` |
| [full.jsonl](full.jsonl) | `573885cb5640d04042395457b808b6fbcb7ff8bfc0080dccb7bb16e3480dbc94` |
| [instantaneous.json](instantaneous.json) | `a7654bed5c53d4c15c132cf644f1be5ad816eb0d5a54552751f96e6a7272909b` |
| [instantaneous.jsonl](instantaneous.jsonl) | `19d3147f0664fe4ee2892cb75ae12a7e775c15447bbec44ff6f60ed502a963c7` |
| [quota.json](quota.json) | `9179a26c3610ac14c2e7088eb5eaf4373af8895a4469fb0f461fc46883223fd4` |
| [quota.jsonl](quota.jsonl) | `04ceeca63f44cb2ab0a2990df2d948920b93b0bcb1dabdb73ba584e7ab2b1b74` |
| [calibration-adequacy.json](calibration-adequacy.json) | `1aa5ea9752763b9e18773d77eb65d7cfc48eb59713be3bf2207263c47f0028e1` |
| [protocol.json](protocol.json) | `c11ad1cd68ac2c75d93d016a6586ed21ab02b400fbc099327f54d2ac0b586b64` |
