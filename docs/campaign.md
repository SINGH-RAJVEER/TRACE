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
	"version": "campaign-v1",
	"namespace": "local-development",
	"mode": "observe",
	"window_size": 128,
	"min_samples": 16,
	"max_text_chars": 4096,
	"window_seconds": 86400.0,
	"cooldown_seconds": 300.0,
	"threshold": 0.8
}
```

These values illustrate the schema; they are not validated operating thresholds. `mode: throttle` additionally requires `HARNESS_CAMPAIGN_ALLOW_UNQUALIFIED_ENFORCEMENT=true`. This is a diagnostic opt-in, not a qualification override. Observation mode never requests campaign throttling. Statistical calibration, customer pilot evidence and supported production activation remain roadmap work.

The canonical configuration digest binds durable state and the API policy revision. An incompatible configuration requires a new state file. Leaving the campaign setting unset retains legacy policy revisions and does not create campaign SQLite tables or metadata. Existing shadow scorers remain separate and telemetry-only.

## Evidence and state

The first rule combines normalized template repetition, lexical overlap and a heuristic fraction of independent questions. Numeric slots are normalized and English follow-up prefixes reduce the score. Breadth, timing and optional semantic similarity are supporting telemetry, not independent evidence of downstream training. Semantic similarity is absent when usable embeddings are unavailable; no additional teacher call or mandatory encoder is introduced.

State keeps bounded count/time windows of hashed lexical features, template/domain fingerprints, optional embeddings and a bounded request-receipt cache. It does not keep raw prompt text. The cache is not a lifetime deduplication ledger; the API's durable idempotency mechanism remains authoritative. Bound each prompt by `max_text_chars` and embedding width by 2,048. State has a count cap of at most 512 rows per principal, but total principal cardinality is not globally bounded. Retention cleanup, fleet storage quotas and lifecycle administration remain work. Hashed features and embeddings can still reveal sensitive information and require appropriate access controls.

Updates share the admission transaction and rollback with it. Evidence is keyed by deployment namespace and authenticated principal; changing organization metadata does not reset that history. This slice performs no inferred account linking or organization-wide campaign scoring. Count/time window expiry is explicit, and process restarts do not empty the state. Incompatible dimensions are refused. Observations that arrive out of order after encoder work use the last committed campaign time and record both received time and the ordering adjustment; their evidence is retained.

The score uses bounded recent history rather than full-prefix replay or all historical query pairs. Optional semantic comparisons still cost work proportional to window size and embedding width. Target-host latency, memory, contention and large-principal-population measurements have not been performed.

## Temporary treatment

With diagnostic throttling enabled, sufficient supported rule evidence starts a fixed cooldown. Blocked probing cannot extend that deadline. After cooldown expiry, a new request begins a fresh evidence window, so stale evidence cannot immediately rearm treatment. The evaluated-attempt total remains cumulative. Expiry also clears the request-receipt cache, so a request refused during the cooldown is scored as new evidence when retried and cannot be replayed unobserved. Within a cooldown, a retained retry consults the current cooldown. Repeated collection can trigger again after accumulating new evidence.

The API returns a separately attributed `campaign_throttled` error with retry timing before dispatching the teacher. Request/token quotas, proof accounting and entitlement limits remain independent. Rolling/supervision/tier refusals precede campaign evaluation; an active campaign throttle can precede later token/in-flight entitlement checks, so error precedence is explicit rather than evidence that those budgets were evaluated. The observation scope is evaluated admission attempts; it does not count early authentication, budget or encoder failures as scored requests. Decision logs carry campaign evidence only when enabled. Proof redemption also respects an active cooldown and leaves the challenge/reservation unconsumed. Challenge TTL and reservation leases continue to elapse; admitted in-flight requests may still finish.

Observe-mode scoring errors are logged without invented evidence and permit ordinary policy processing. In diagnostic throttle mode, a scoring error with no active cooldown returns `503 campaign_unavailable`; a recorded cooldown still blocks. This prevents deliberately invalid observations from bypassing the experimental restriction. The current API truncates text to the configured character budget before scoring, so suffix content beyond that budget is outside detection scope.

When review is enabled, one appeal is reused per principal/cooldown. An authenticated operator may grant the existing bounded one-request allowance for a fresh admission; it does not reset evidence or other entitlements. Held proofs must wait for cooldown expiry even after a grant, because their earlier reservations cannot account for use of a new reservation-counted allowance. Converting `campaign_throttled` into a verification response through `HARNESS_REVIEW_VERIFY_CODES` is unsupported and rejected at startup. Idempotent throttle responses replay while active with updated remaining retry timing; after their recorded cooldown expires, the same key can proceed without reviving the old refusal. Retained core retries persist the time/expiry watermark without incrementing observation counts, so later reordered retries cannot revive expired state.

## Evidence limits and next work

CPU fixtures cover scoring behavior, bounded state, transaction rollback, configuration binding, restart recovery and concurrency. API fixtures cover teacher-call suppression with a local fixture teacher, observe-mode compatibility, idempotency, independent principals, proof redemption, budget attribution and restoration. Fixtures do not establish real-user accuracy or reduced learned student capability.

Verification on 2026-11-05 passed the full 372-test repository suite, 25 focused campaign tests, Ruff lint/format and strict mypy over 128 files through `devenv shell -- uv run`. Independent Codex spec and standards reviews identified a held-proof grant bypass, expired-retry cooldown revival and stale retry timing. Regression tests cover the repaired cases. No live teacher, GPU capability experiment, real-user false-alert audit or deployment activation was performed.

The heuristic can confuse legitimate templated evaluation or labeling with collection and can miss varied prompts or disguised follow-ups. Lexical normalization and follow-up rules need multilingual and adversarial evaluation. Existing conditional MMD, calibrated coverage/session evidence and finite-path calibration are not automatically integrated or qualified by this new rule. Do not present the product roadmap's full D0-D6 scope as completed.

Before supported enforcement, implement independently calibrated whole-campaign decision thresholds, representative legitimate cohorts, adaptive final-answer collection tests, measured quota comparisons, incident investigation and live checkpoint student experiments. Cross-account corroboration, learned classifiers, protected-reasoning channels, durable asynchronous queues and tenant administration remain separate work. See the [research report](<reports/Distillation detection methods and standards.md>) for method applicability and the [evaluation protocol](evaluation.md) for evidence boundaries.
