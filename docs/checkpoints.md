# Chronological acquisition checkpoints

The checkpoint workflow measures when an attacker acquires useful task capability relative to observed intervention. It exports what a completed gateway campaign actually delivered, plans students from those chronological prefixes and evaluates them against a locked task suite. It is experiment tooling, with CPU fixture coverage; a live teacher campaign and GPU student experiment are still required to measure prevention.

## Export contract

`distill checkpoints export` reads `contract.json`, `campaign.json`, `campaign.events.jsonl` and `ledger.jsonl` from a completed gateway campaign. It verifies the campaign contract and artifact digests, joins attempts to outcomes and retained responses, and publishes a new directory containing `manifest.json` and checkpoint ledgers. Existing output directories are refused. Reloading an export verifies both the original sources and exported ledgers, so keep those sources available and immutable.

Response checkpoints use client receipt times. A requested count includes every response tied at the cutoff, so its actual count can exceed the requested count. Count zero exports an empty ledger. An unreached count has no ledger and remains censored. The end checkpoint contains all completed successful receipts, including late in-flight responses, and uses the last outcome time, including failures and refusals. It does not discard unsuccessful HTTP attempts from attacker cost.

First alert, first restriction and first refusal are separate observations. Challenge or block outcomes establish an observed restriction; policy-attributed events can also identify alerts, L2-or-higher spacing policies or refusals. L2 attribution identifies a restricted policy level without measuring the extra delay it actually imposed. The interval runs from the earliest qualifying attempt start to the earliest qualifying outcome receipt. Copied policy event timestamps are attempt starts, not server decision timestamps. Overlapping attempts can therefore widen the interval. These bounds cover available attributed events; earlier unobserved enforcement remains possible. Missing alert attribution stays unknown.

A campaign collected with an experiment binding has a `gateway-execution-v3` contract and exports a `campaign-checkpoints-v2` manifest. Earlier campaigns still export `campaign-checkpoints-v1` with the same rules and checkpoint IDs, and both versions load. In v2 the three general observations widen. `first_alert` also accepts campaign evidence with status `alert`, including observe mode, without assigning it a detector level. `first_restriction` also accepts attributed 429 outcomes, joined campaign throttles and decisions that carry an entitlement code. `first_refusal` also accepts a 429 whose harness decision is `block`; a temporary campaign throttle is not a refusal. A row logged as level 4 by the early entitlement path has no detector decision behind it, so it sets neither a level nor a refusal action, and its logged values stay in `logged_level` and `logged_action`.

The v2 manifest adds three attributed observations: `first_campaign_alert`, `first_campaign_restriction` and `first_entitlement_restriction`. Each reached observation lists `causes`: every cause (`campaign`, `entitlement`, `rate`, `policy` or `unknown`) of a qualifying submission that started inside its interval, so more than one cause can appear. An attributed observation with exactly the bounds of a general one records `checkpoints_from` and exports no second ledger, so no duplicate student is planned. The bracketing is unchanged: earliest qualifying attempt start to earliest qualifying client outcome. Server times are kept in the events and are not used as bounds, because the server and client clocks are not reconciled. Missing attribution stays unknown.

The API logs only the root ID that a query and its proof share. The collector therefore places a row by what it records: an ordinary decision or an admission throttle belongs to the query, and a proof-stage throttle belongs to the proof. Events keep `root_id` and `submission_id` apart. A submission with more than one candidate row, or a row with no single submission, gets no attribution and an `attribution_ambiguous` event instead. The summary reports these as `ambiguous_policy_rows` with `attribution_complete` false, and export refuses the campaign.

Before exporting a v3 campaign, the exporter checks the frozen `campaign-config.json` against the contract's file hash and canonical digest, recomputes the policy binding from `decisions.snapshot.jsonl` and compares it with `campaign.json`. It refuses unless the binding status is `bound`. Both files join the manifest's verified inputs. `manifest.json` also records the experiment declaration and the binding.

Each reached checkpoint records query and proof attempts started, attempts completed, completed outcome counts and known released output tokens. Unknown token counts remain explicit. Costs include all accounts represented by the campaign. A lower-bound ledger contains receipts strictly before that bound; an observed-bound ledger includes receipts at its bound. Cost counters include attempts and outcomes at their cutoff, so they are separate from the lower-bound ledger's conservative membership rule. The late-in-flight count is relative to client observation, not exact server enforcement. In v2, cost also counts `restricted` outcomes and `restriction_wait_seconds`, the retry-after waiting completed by the cutoff.

This first version requires one-to-one attribution between successful receipts and retained ledger records. Batching, session replay, response selection, incomplete attempts, missing retained receipts and ambiguous policy attribution are refused. It cannot reconstruct responses that the collector discarded. Hash verification establishes byte consistency, not trustworthy provenance or authorization.

## Student experiment

The task runner exposes this flow as `just checkpoint-export CAMPAIGN OUTPUT`, `just checkpoint-plan EXPORT SUITE OUTPUT --target-accuracy VALUE`, `just checkpoint-execute OUTPUT` and `just checkpoint-summary OUTPUT`. Extra export/plan flags are forwarded as separate arguments, including paths with spaces and repeated seeds. GPU execution is a separate explicit command; `just start` only serves the protected API. Use `just test-checkpoints` for CPU fixtures.

The initial student implementation supports GSM8K. For example, using fresh output directories:

```bash
devenv shell -- uv run distill checkpoints export \
	--campaign artifacts/protected/run \
	--output artifacts/protected/checkpoints \
	--response-count 0 --response-count 100 --response-count 300 --response-count 1000
devenv shell -- uv run distill checkpoints plan \
	--export artifacts/protected/checkpoints \
	--suite data/tasks/gsm8k/development.json \
	--output artifacts/protected/checkpoint-students \
	--target-accuracy 0.50 --seed 42 --seed 43 --seed 44 --include-pending
devenv shell -- uv run distill checkpoints execute \
	--output artifacts/protected/checkpoint-students
devenv shell -- uv run distill checkpoints summary \
	--output artifacts/protected/checkpoint-students
```

Replace the illustrative target and suite path with the preregistered target and actual frozen development suite. Planning and summary require no GPU dependencies. Default execution uses the existing pinned math student runner and requires its GPU environment.

Choose a useful accuracy target before running the protected experiment, supported by a passing clean-baseline experiment and task headroom. Freeze an independent, source-disjoint evaluation suite and the same pinned student recipe and seeds across compared arms. Each checkpoint student starts from that base and uses the eligible responses actually present at its receipt cutoff. It must not substitute a hash-selected subset of the end ledger.

Accepted, terminal training records are eligible. An explicit pending option also admits unreviewed pending records; it never admits rejected records. Acquired response counts and eligible training counts are reported separately. A reached checkpoint with no eligible examples uses the untrained base. Unreached checkpoints remain censored rather than becoming zero-accuracy students.

Plans bind the verified export, original campaign artifacts, evaluation suite, training membership, model recipe, seeds and declared accuracy target. Execution verifies those bindings before using saved results and refuses missing sources. Keep these files at their recorded paths when executing or resuming. Interrupted cells restart from the base; completed cells resume only with verified score artifacts. The summary reports checkpoint accuracy, paired base gain and the first observed checkpoint whose mean accuracy across the declared training seeds reaches the target, alongside the number of individual seeds reaching it. Base evaluation uses one deterministic greedy evaluation shared across the training seeds. If the base already meets the target, that target cannot demonstrate acquired capability. The summary does not infer the exact first learning time between checkpoints or issue a general prevention verdict.

Every nonempty checkpoint trains separately for every declared seed, including checkpoints with identical eligible training sets. Inspect the plan's cells before scheduling GPU work. Execution does not measure training compute as attacker collection cost.

## Interpretation

Compare lower-bound, observed-bound and end students to assess timing uncertainty and late receipts. A student reaching the target before the intervention lower bound is evidence of useful acquisition before the attributed intervention. A student first reaching it at the observed bound leaves ordering uncertain. A student reaching it only at end shows why post-intervention in-flight responses matter. Failure to reach the target within the observed checkpoints is a censored experiment, not proof that distillation is impossible.

This workflow does not automate legitimate-user utility measurement, quota-grid selection or a student-updating adaptive attacker. Those remain separate parts of the prevention evaluation. Compare detector and quota arms at measured, matched legitimate utility, with comparable attacker attempts, time, money and account resources.
