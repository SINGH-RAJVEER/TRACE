# Unprotected A6 run

This directory is a byte-for-byte copy of `../quota/`, the v7 `A6/quota` arm, relabeled as the unprotected A6 run. It is not a separate collection.

The A6 campaign ran with the quota policy in the request path, but the quota never intervened: all 6,000 requests were served, with no alert, spacing, challenge or refusal decision (every row in `decisions.jsonl` is level 0, `serve`). The attacker therefore received exactly what an unprotected teacher would have released for this schedule: 6,000 replies and 806,570 output tokens, including 1,500 GSM8K replies, of which 1,473 finished and 1,402 are correct against the gold answers.

The files still record the quota arm (`harness.env` has `HARNESS_POLICY=quota`), because they were not re-collected. Report this run as the v7 A6 quota arm with no interventions, used as the unprotected reference.

Its GSM8K prompts include every GSM8K prompt the A5 arms received (199 for `full`, 99 for `instantaneous`, 1,215 for `quota`), so it is the unprotected upper bound for the A5 student comparison.

`../unprotected-v3-archive/` holds a second, independently collected unprotected ledger from the archive, from a different teacher and corpus build.
