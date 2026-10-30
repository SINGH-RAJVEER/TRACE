# v1: detector calibration and replay

This historical run used code `866ade6` and the `task-study-v2` protocol. It calibrated the MMD detector and replayed held-out benign traffic and a 2,000-request attacker for AG News and SQuAD. Under full and instantaneous MMD, the attacker stayed at L0; quota refused requests after 500. The full detector escalated 16 of 45 scored AG News horizons and 18 of 36 scored SQuAD horizons to L1. Specialist accounts were affected; the synthetic workload does not establish real-user false-positive rates.

The run stopped before protected collection. The machine-readable [readings.json](readings.json) and the [original narrative](../../legacy/readings/v1-2026-10-17-start-v2-README.md) preserve the details and provenance.
