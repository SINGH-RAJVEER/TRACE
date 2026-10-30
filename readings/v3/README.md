# v3: bounded gateway harness observations

**Date:** 2026-10-29 to 2026-10-30.

This is an interim historical record. The original protocol and calibration are retained here for provenance; use the current evaluation workflow for new runs.

A1 used 6,000 submissions across AG News, SQuAD, GSM8K, and coding. Under the full MMD policy and instantaneous MMD, all 6,000 responses were received and the account stayed at L0. The quota policy received 1,215 responses and refused 4,785; L4 began at submission 1,216. The full and instantaneous arms each released 588,653 output tokens; quota released 118,872. These results show that the MMD policies did not restrict this A1 campaign, while the quota did. They cover one campaign and one seed, not general population effectiveness.

Held-out benign policy replay covered 125 account horizons, of which 100 completed a scoring window. The full policy reached L1 on 12 horizons and L2 on three; instantaneous reached L1 on 12, L2 on eight, L3 on two and L4 on two. Quota reached L4 on one. Rates over scored accounts and all accounts have different denominators. These are replay decisions, not live delivered-answer outcomes.

The interrupted A2 campaign is preserved in `harness/`. The [original narrative](../../legacy/readings/v3-2026-10-29-gateway-replay/README.md) records restart and revision details. `calibration/`, `benign/`, `schedules/`, per-request decisions and campaign events preserve the detector and workload evidence. The archived observations do not establish current performance.
