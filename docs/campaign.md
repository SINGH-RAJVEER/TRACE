# Experimental persistent campaign detection

The campaign detector is an opt-in first implementation of the [detection roadmap](../TODO.md). It adds durable per-principal evidence and explicit experimental API throttling. It has no independently measured false-alert rate, no calibrated probability of distillation and no demonstrated teacher/student prevention result. Every evidence record carries `qualified: false`.

## Configuration

### Task runner flow

`just start` and its alias `just harness` launch the product API. They no longer run the full study. Enter `devenv shell`, use `just setup` for core dependencies and `just setup-eval` for encoder/evaluation dependencies. The teacher model server and `just teacher` adapter are managed separately.

```bash
just campaign-config artifacts/campaign/config.json --namespace my-deployment
just validate --config artifacts/campaign/config.json --state-db artifacts/campaign/state.sqlite3
just start --config artifacts/campaign/config.json --state-db artifacts/campaign/state.sqlite3
```

Before validation/startup, set `HARNESS_ACCOUNTS_FILE` to your authenticated account mapping and `HARNESS_CALIBRATION_DIR` to an existing calibration matching `HARNESS_TASK`. Existing calibration remains necessary even for base policy `allow`. `validate` checks settings and prerequisite file paths, not upstream readiness, calibration adequacy or detection effectiveness. Actual startup loads and verifies API artifacts normally. No state database is created by validation. Product startup rejects test clocks and client-supplied benchmark identities.

The generator creates a new JSON exclusively and defaults to observation mode; existing files are never replaced. `--mode throttle` writes a diagnostic configuration but does not enable its enforcement override. Startup requires the existing explicit `HARNESS_CAMPAIGN_ALLOW_UNQUALIFIED_ENFORCEMENT=true` declaration to use that mode. Set `HARNESS_CAMPAIGN_CONFIG` and `HARNESS_STATE_DB` in `.env` to omit the corresponding start/validate flags. Product startup defaults the base policy to `allow` even if `.env` selects MMD; use `--base-policy mmd`, `quota`, `quota_tokens` or `rate` deliberately, with that policy's additional prerequisites.

| Command | Purpose |
| --- | --- |
| `just setup`, `just setup-eval` | Frozen core dependencies, or encoder/evaluation extras |
| `just teacher` | Teacher adapter only, using the configured upstream |
| `just campaign-config OUTPUT --namespace NAME [--mode observe\|throttle]` | New versioned campaign configuration |
| `just validate [--config FILE] [--state-db FILE] [--base-policy POLICY]` | Offline launch-setting and prerequisite-path validation |
| `just start ...`, `just harness ...` | Campaign-protected API in the foreground |
| `just harness-legacy` | Configured base-policy API with campaign settings explicitly disabled, including dotenv values |
| `just study [RUN_ID]` | Retained corpus/calibration/teacher/attack/utility study; may launch GPU work and export readings |
| `just baseline-gates [RUN_ID]` | Declared teacher headroom and clean-student GPU workflow |
| `just calibrate ...`, `just recalibrate SOURCE OUTPUT [PROFILES]` | Existing calibration and source-disjoint profile preparation |
| `just distill ...`, `just detector ...` | Pass-through corpus, collection, evaluation, replay, evidence and review CLIs |
| `just checkpoint-export CAMPAIGN OUTPUT ...` | Exclusive receipt-ordered export; repeat `--response-count N` |
| `just checkpoint-plan EXPORT SUITE OUTPUT --target-accuracy VALUE ...` | GSM8K student plan; accepts repeated `--seed` and explicit `--include-pending` |
| `just checkpoint-execute OUTPUT`, `just checkpoint-summary OUTPUT` | Explicit GPU execution, then CPU reporting |
| `just test [PYTEST_ARGS...]`, `just test-campaign`, `just test-checkpoints` | Full/custom tests or focused feature suites |
| `just check`, `just format` | Lint/format/type/shell checks, or explicit Python formatting |
| `just reading RUN_ID [NAME]`, `just backfill` | Explicit research export or historical index maintenance |
| `just lock` | Explicit dependency-lock maintenance |

`just study` preserves the previous `just start RUN_ID` workflow under `scripts/study.sh`; existing run paths and resume markers stay the same. Its harness processes use the legacy launcher so a product `.env` cannot silently enable the new detector in historical arms. It is not an automated campaign-detector qualification study. The new start command never invokes study preparation, GPU training or reading export. To inspect declarations without starting those workflows, use `just study --help` or `just baseline-gates --help`.

Set `HARNESS_CAMPAIGN_CONFIG` to a JSON file and `HARNESS_STATE_DB` to a persistent state file. For an observation-only fixture configuration:

```json
{
	"version": "campaign-v2",
	"namespace": "local-development",
	"mode": "observe",
	"window_size": 128,
	"min_samples": 16,
	"max_text_chars": 4096,
	"window_seconds": 86400.0,
	"cooldown_seconds": 300.0,
	"threshold": 0.8,
	"min_variant_fraction": 0.5
}
```

These values illustrate the schema; they are not validated operating thresholds. `mode: throttle` additionally requires `HARNESS_CAMPAIGN_ALLOW_UNQUALIFIED_ENFORCEMENT=true`. This is a diagnostic opt-in, not a qualification override. Observation mode never requests campaign throttling. Statistical calibration, customer pilot evidence and supported production activation remain roadmap work.

The canonical configuration digest binds durable state and the API policy revision. An incompatible configuration requires a new state file, so existing `campaign-v1` configurations and their state must be regenerated. Leaving the campaign setting unset retains legacy policy revisions and does not create campaign SQLite tables or metadata. Existing shadow scorers remain separate and telemetry-only.

## Evidence and state

With `session` omitted or null, the frozen v2 rule combines normalized template repetition, lexical overlap and a heuristic fraction of independent questions. Numeric slots are normalized and English follow-up prefixes reduce the score. Breadth, timing and optional semantic similarity are supporting telemetry, not independent evidence of downstream training. Semantic similarity is absent when usable embeddings are unavailable; no additional teacher call or mandatory encoder is introduced.

Templates are hashed over the full normalized prompt before the write transaction opens, so questions that differ only after the scored text bound are separate templates. An attacker can use the same property to vary a suffix, and any wording change already evades the exact template match.

Exact repeats still raise alerts, but they start a cooldown only when distinct prompts make up more than `min_variant_fraction` of the most common template. A request with `temperature > 0` counts as distinct even when its text repeats, because it can return a new output. This stops a deterministic health check from being throttled, and it stops resampling from using the same exemption. The default of 0.5 is an unvalidated development value, and setting it to 0 restores throttling on all alerts.

State keeps bounded count/time windows of hashed lexical features, template/domain fingerprints, optional embeddings and a bounded request-receipt cache. It does not keep raw prompt text. The cache is not a lifetime deduplication ledger; the API's durable idempotency mechanism remains authoritative. Scored text is bounded by `max_text_chars` and embedding width by 2,048. Prompts longer than that are only hashed whole, before the write transaction. State has a count cap of at most 512 rows per principal, but total principal cardinality is not globally bounded. Retention cleanup, fleet storage quotas and lifecycle administration remain work. Hashed features and embeddings can still reveal sensitive information and require appropriate access controls.

Updates share the admission transaction and rollback with it. Evidence is keyed by deployment namespace and authenticated principal; changing organization metadata does not reset that history. This slice performs no inferred account linking or organization-wide campaign scoring. Count/time window expiry is explicit, and process restarts do not empty the state. Incompatible dimensions are refused. Observations that arrive out of order after encoder work use the last committed campaign time and record both received time and the ordering adjustment; their evidence is retained.

The score uses bounded recent history rather than full-prefix replay or all historical query pairs. Optional semantic comparisons still cost work proportional to window size and embedding width. Target-host latency, memory, contention and large-principal-population measurements have not been performed.

## Opt-in semantic/session experiment

Add a `session` object to a campaign-v2 JSON to select the separate `semantic-session-v1` combination. Omitted/null `session` preserves the historical v2 score, evidence shape, digest and state binding. The generator still defaults to v2; edit the new configuration explicitly before validating or starting it. Enabling the extension, changing its threshold or encoder representation changes the digest and requires a fresh state database. Observation mode and `qualified: false` remain the defaults; diagnostic enforcement still requires the existing override.

```json
"session": {
	"version": "semantic-session-v1",
	"semantic_threshold": 0.9,
	"encoder_identity": null
}
```

This is a JSON member to insert in the full configuration above. With `encoder_identity: null`, the session/lexical path operates without embeddings or any encoder dependency. To enable semantics, replace null with the exact stable `Encoder.identity()` representation for the serving encoder, excluding the mutable `encoded` and `truncated` counters. Use the configured encoder's real identity rather than copying a model name alone. Startup checks equality before opening campaign state and constructs the existing optional BGE encoder when needed, including for base policy `allow`. Its optional embedding dependencies must be installed for that configuration. No encoder or teacher is called by offline launch-setting validation. Identity, semantic threshold, input serialization and combination version are bound through the configuration digest; same-width vectors from another representation cannot silently reuse serving state. Direct core callers must enforce that representation themselves.

The semantic input is the bounded raw current user turn, independent of the base policy's task serialization. The API reuses an embedding only for identical input, otherwise performs a separate bounded encoder call. It adds no teacher call. A whole-prompt identity is still computed outside the transaction. If that identity differs from the scored prefix identity, the row is marked truncated and cannot participate in semantic comparisons. The full-template lexical route remains available. Encoder failures retain the API's existing `503 encoder_error`/overload behavior before admission; partial embedding behavior below applies to core callers and retained windows rather than silently treating serving encoder failure as success.

For each current request, normalized cosine similarity at least `semantic_threshold` selects prior family members in the retained count/time window. The current row anchors the family; there is no transitive clustering or full historical pair replay. Let `R` be matching prior rows divided by all prior retained rows, `A` the window answer-only share, `T` v2 template repetition, `L` v2 lexical overlap, `C` the window contextual-language share, and `W` the window editing/tutoring/debugging-language share. After `min_samples`, the frozen score is:

```text
lexical_contribution = T * L * (1 - max(C, W))
semantic_contribution = R * A * (1 - max(C, W))
score = max(lexical_contribution, semantic_contribution)
```

There is no weight renormalization and no addition of correlated routes. Missing vectors contribute zero to `R` while retaining the full denominator. A missing current vector disables the semantic contribution. Evidence reports available/missing comparison counts, missingness, truncated count, selected route, both contributions, workflow multiplier and combination version. Warmup has a null score and explicit reason. The alert threshold remains an uncalibrated operator declaration, not a probability or measured false-alert guarantee. Semantic similarity alone cannot alert through the semantic route: answer-format preference must corroborate it. Breadth, topic switches, volume and answer-only preference alone cannot supply collection evidence.

The extension shares the offline signal definitions for answer-only and contextual language without importing the optional offline scorer. It also reports hashed word-trigram nearest rephrasing similarity, full-template diversity and directed domain switch rate. These are bounded online analogues, not identical whole-session offline statistics. Rephrasing, diversity and topic switching are context only in this first combination; offline topic entropy, interarrival statistics, novelty and calibrated ranks are not integrated. Contextual language and explicit editing/tutoring/debugging language reduce the window score. A single current follow-up prefix no longer vetoes a whole campaign, although enough contextual language can reduce it. These English heuristics remain spoofable and can misclassify legitimate work.

The selected route's family must exceed `min_variant_fraction` distinct collection variants before diagnostic enforcement. The semantic route measures variants within its current-anchor family; the lexical route uses the modal full-template family. Deterministic exact repeats remain visible but do not accumulate distinct variants. Resampling requests do, while retries retain the existing receipt/idempotency handling. Transactions, monotone commit time, cooldown deadlines, expiry reset and durable restart behavior are shared with v2. Receipts remain bounded, not lifetime deduplication.

Work is bounded by the configured text length, window size and embedding width. Each new row adds hashed trigrams and a few flags without storing raw text; comparisons cost O(window size times embedding width plus window size times bounded trigram count). Hashing the whole input still costs O(full input length) outside the transaction. There is no global principal cap. No target-host latency or storage qualification was performed.

Deterministic CPU controls include paraphrase sweeps, semantic similarity without answer-only preference, editing, tutoring, distinct-topic answer-only batches, long shared prefixes, partial/missing vectors, repeats/resampling and retry/restart behavior. A legitimate answer-only labeling or evaluation batch with the same observable semantic pattern can still score exactly like collection; the system cannot infer downstream intent from indistinguishable observations. These controls do not establish measured detection gain, real-user accuracy, multilingual robustness or statistical calibration. Multiscale scoring and conditional MMD remain separate follow-on experiments.

Verification on 2026-11-06: the repository suite completed with 398 passed and one failure in the unchanged `test_cli_passes_online_contract_to_collection`, whose live gateway health check reported `mmd` while the fixture declared `none`. That failure also reproduced in isolation; the test and CLI module match the implementation base byte-for-byte. The focused campaign/API/offline-signal suites passed 51 tests. `just check` passed Ruff lint/format, strict mypy over 131 source files and shell syntax checks. No teacher generation, GPU job or deployment was performed.

## Temporary treatment

With diagnostic throttling enabled, sufficient supported rule evidence starts a fixed cooldown. Blocked probing cannot extend that deadline. After cooldown expiry, a new request begins a fresh evidence window, so stale evidence cannot immediately rearm treatment. The evaluated-attempt total remains cumulative. Expiry also clears the request-receipt cache, so a request refused during the cooldown is scored as new evidence when retried and cannot be replayed unobserved. Within a cooldown, a retained retry consults the current cooldown. Repeated collection can trigger again after accumulating new evidence.

The API returns a separately attributed `campaign_throttled` error with retry timing before dispatching the teacher. Request/token quotas, proof accounting and entitlement limits remain independent. Rolling/supervision/tier refusals precede campaign evaluation; an active campaign throttle can precede later token/in-flight entitlement checks, so error precedence is explicit rather than evidence that those budgets were evaluated. The observation scope is evaluated admission attempts; it does not count early authentication, budget or encoder failures as scored requests. Decision logs carry campaign evidence only when enabled. Proof redemption also respects an active cooldown and leaves the challenge/reservation unconsumed. Challenge TTL and reservation leases continue to elapse; admitted in-flight requests may still finish.

Observe-mode scoring errors are logged without invented evidence and permit ordinary policy processing. In diagnostic throttle mode, a scoring error with no active cooldown returns `503 campaign_unavailable`; a recorded cooldown still blocks. This prevents deliberately invalid observations from bypassing the experimental restriction. The current API truncates text to the configured character budget before scoring, so suffix content beyond that budget is outside detection scope.

When review is enabled, one appeal is reused per principal/cooldown. An authenticated operator may grant the existing bounded one-request allowance for a fresh admission; it does not reset evidence or other entitlements. Held proofs must wait for cooldown expiry even after a grant, because their earlier reservations cannot account for use of a new reservation-counted allowance. Converting `campaign_throttled` into a verification response through `HARNESS_REVIEW_VERIFY_CODES` is unsupported and rejected at startup. Idempotent throttle responses replay while active with updated remaining retry timing; after their recorded cooldown expires, the same key can proceed without reviving the old refusal. Retained core retries persist the time/expiry watermark without incrementing observation counts, so later reordered retries cannot revive expired state.

## Evidence limits and next work

CPU fixtures cover scoring behavior, bounded state, transaction rollback, configuration binding, restart recovery and concurrency. API fixtures cover teacher-call suppression with a local fixture teacher, observe-mode compatibility, idempotency, independent principals, proof redemption, budget attribution and restoration. Fixtures do not establish real-user accuracy or reduced learned student capability.

Verification on 2026-11-05 passed the full 386-test repository suite and `just check` (Ruff lint/format and strict mypy over 130 files) through `devenv shell`. Independent spec and standards reviews identified a held-proof grant bypass, expired-retry cooldown revival and stale retry timing. Later review identified replay of refused requests after expiry, false alerts from truncated prompts and exact-repeat throttling. Regression tests cover the repaired cases. No live teacher, GPU capability experiment, real-user false-alert audit or deployment activation was performed.

The heuristic can confuse legitimate templated evaluation or labeling with collection and can miss varied prompts or disguised follow-ups. Lexical normalization and follow-up rules need multilingual and adversarial evaluation. Existing conditional MMD, calibrated coverage/session evidence and finite-path calibration are not automatically integrated or qualified by this new rule. Do not present the product roadmap's full D0-D6 scope as completed.

Before supported enforcement, implement independently calibrated whole-campaign decision thresholds, representative legitimate cohorts, adaptive final-answer collection tests, measured quota comparisons, incident investigation and live checkpoint student experiments. Cross-account corroboration, learned classifiers, protected-reasoning channels, durable asynchronous queues and tenant administration remain separate work. See the [research report](<reports/Distillation detection methods and standards.md>) for method applicability and the [evaluation protocol](evaluation.md) for evidence boundaries.
