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

Set the `session` member in a campaign-v2 JSON to select the separate `semantic-session-v2` combination. Replace an existing `"session": null` member rather than adding a second member with the same name; duplicate JSON keys are rejected. Omitted/null `session` preserves the historical v2 score, evidence shape, digest and state binding. The generator still defaults to v2; edit the new configuration explicitly before validating or starting it. The launcher reports the selected session version or that it is disabled. Enabling the extension, changing its threshold or encoder representation changes the digest and requires a fresh state database. The revised combination uses a new version so an earlier `semantic-session-v1` configuration cannot silently reuse a different scoring procedure. Observation mode and `qualified: false` remain the defaults; diagnostic enforcement still requires the existing override.

```json
"session": {
	"version": "semantic-session-v2",
	"semantic_threshold": 0.9,
	"encoder_identity": null
}
```

This is a JSON member to set in the full configuration above. With `encoder_identity: null`, the lexical and rephrasing paths operate without embeddings or any encoder dependency. To enable semantics, replace null with the exact stable `Encoder.identity()` representation for the serving encoder, excluding the mutable `encoded` and `truncated` counters. For the default pinned BGE encoder, use:

```json
"encoder_identity": {
	"model": "BAAI/bge-small-en-v1.5",
	"revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
	"max_tokens": 512,
	"pooling": "cls",
	"normalized": true
}
```

Use the configured encoder's actual identity if it differs from this default. Startup checks equality before opening campaign state and constructs the existing optional BGE encoder when needed, including for base policy `allow`. Its optional embedding dependencies must be installed for that configuration. No encoder or teacher is called by offline launch-setting validation. Identity, semantic threshold, input serialization and combination version are bound through the configuration digest; same-width vectors from another representation cannot silently reuse serving state. Direct core callers must enforce that representation themselves.

The semantic input is the bounded raw current user turn, independent of the base policy's task serialization. The API reuses an embedding only for identical input, otherwise performs a separate bounded encoder call. It adds no teacher call. Whole-prompt identity and capped text features are computed outside the transaction. Character-truncated rows cannot participate in semantic or rephrasing comparisons. BGE's optional `input_complete(text)` method checks actual tokenizer length, including special tokens, outside the transaction; there is no byte-count approximation. Token-truncated rows and custom encoders without this completeness capability supply no semantic evidence. The API skips campaign encoding for those inputs, although the base policy can still require its own encoding. Direct core callers must supply a truthful `semantic_complete` receipt alongside their bound embeddings; omitted completeness is unknown. The existing generic Encoder protocol and calibration identity are unchanged. Long prompts that exceed the declared character or token limits can therefore evade the semantic route, and callers can exploit that limitation with padding. The full-template lexical route remains available. Encoder/tokenizer failures retain the API's existing `503 encoder_error`/overload behavior before admission; partial embedding behavior below applies to core callers and retained windows rather than silently treating serving encoder failure as success.

For each current request, normalized cosine similarity at least `semantic_threshold` selects prior semantic family members in the retained count/time window. Word-trigram Jaccard overlap of at least 0.6 selects lexical rephrasing family members. Each family is anchored on the current row; there is no transitive clustering or full historical pair replay. Let `R` and `P` be the respective matching prior-row fractions, each divided by all prior retained rows, `A` the window answer-format share, `T` v2 template repetition, `L` v2 lexical overlap, `F` v2 follow-up-prefix share, `C` online contextual-language share, and `W` narrow editing-workflow share. After `min_samples`, the frozen score is:

```text
lexical_contribution = T * L * (1 - F)
semantic_contribution = R * A * (1 - max(C, W))
rephrasing_contribution = P * A * (1 - max(C, W))
score = max(lexical_contribution, semantic_contribution, rephrasing_contribution)
```

There is no weight renormalization and no addition of correlated routes. Missing vectors contribute zero to `R` while retaining the full denominator. A missing current vector disables the semantic contribution. Evidence reports available/missing comparison counts, missingness, truncated count, selected route, all three contributions, workflow multiplier and combination version. Warmup has a null score and explicit reason. The alert threshold remains an uncalibrated operator declaration, not a probability or measured false-alert guarantee. Semantic similarity or rephrasing alone cannot alert through the added routes: answer-format preference must corroborate them. Breadth, topic switches, volume and answer-only preference alone cannot supply collection evidence.

The online answer-format helper recognizes TRACE's AG News, GSM8K and SQuAD instructions as well as forms such as `Output only the result` and `Respond with just the number`. Offline answer-only and contextual-language definitions remain unchanged. The online contextual helper includes v2's anchored follow-up prefixes; generic words such as `that`, `again` and `instead` do not reduce the score. Editing controls require a leading editing command with local context, such as `Rewrite this` or `Debug my`. Merely asking to explain a concept is not an editing control. These controls apply only to the added routes, so they cannot erase v2's lexical contribution. Sustained follow-up sweeps retain the v2 discount. A single current follow-up prefix no longer vetoes a whole campaign. These English heuristics remain spoofable and can misclassify legitimate work.

Rephrasing contributes to the score without an encoder. Full-template diversity and directed domain switch rate remain telemetry; offline topic entropy, interarrival statistics, novelty and calibrated ranks are not integrated. These online statistics are bounded analogues, not identical whole-session offline statistics.

The selected route's family must exceed `min_variant_fraction` distinct collection variants before diagnostic enforcement. Semantic and rephrasing routes measure variants within their current-anchor families; the lexical route uses the modal full-template family. Deterministic exact repeats remain visible but do not accumulate distinct variants. Resampling requests do, while retries retain the existing receipt/idempotency handling. Transactions, monotone commit time, cooldown deadlines, expiry reset and durable restart behavior are shared with v2. Receipts remain bounded, not lifetime deduplication.

Work is bounded by the configured text length, window size and embedding width. Each row retains at most 254 distinct hashed word trigrams from at most 256 words, prepared before the admission transaction. Inputs exceeding that word cap cannot contribute to the new rephrasing family, so a shared prefix is not treated as a complete prompt. Comparisons cost O(window size times embedding width plus window size times bounded trigram count). Hashing the whole input still costs O(full input length) outside the transaction. There is no global principal cap. No target-host latency or storage qualification was performed.

Deterministic CPU controls include paraphrase sweeps, semantic similarity without answer-only preference, editing, tutoring, distinct-topic answer-only batches, long shared prefixes, partial/missing vectors, repeats/resampling and retry/restart behavior. A legitimate answer-only labeling or evaluation batch with the same observable semantic pattern can still score exactly like collection; the system cannot infer downstream intent from indistinguishable observations. These controls do not establish measured detection gain, real-user accuracy, multilingual robustness or statistical calibration. Multiscale scoring and conditional MMD remain separate follow-on experiments.

Verification on 2026-11-06 after review fixes: all 422 repository tests passed through `devenv shell` and frozen `uv` in a private network namespace with loopback enabled. This keeps the unchanged CLI fixture from contacting the existing study gateway on port 8003; the initial 400-pass/one-failure run queried that service, which serves `mmd` while the fixture declares `none`. Independent review reproduced that environmental failure on both base and implementation, and the earlier 401-test suite passed with the port isolated. The service was left running. Regression coverage now includes legacy follow-up sweeps, retained lexical detections, actual task answer-format instructions, generic-word evasion, encoder-free rephrasing, capped text state, tokenizer completeness and launcher visibility. Ruff lint/format, strict mypy over 131 source files and shell syntax checks passed. No teacher generation, GPU job or deployment was performed.

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

## Opt-in finite coverage monitor and lifetime record

WP2 adds `coverage` and `lifetime` members to campaign configuration. Both default to null and are omitted from the canonical digest when null, preserving historical v2 and semantic-session-v2 digests, evidence and state binding. Each enabled member changes the digest. Use a new state database for a changed configuration. Every result remains `qualified: false`.

`coverage.monitor` declares version `campaign-coverage-v1`, a `CoverageSpec` under `spec`, the stable `encoder_identity`, one supported `domain`, the serving `max_text_chars`, increasing exact request prefixes in `looks`, and EWMA gain scales in `scales`. The projection defaults to 64 dimensions. Every complete embedding is normalized and projected to norm at most one, then updates a ridge inverse with Sherman-Morrison. Each request costs O(d² + dD), where d is projection width and D is embedding width; the persisted sketch costs O(d²) and retains no prefix embeddings. Existing recent-window semantic rows remain separately bounded. Internal admission-store snapshots use quadratic shape, finiteness, symmetry and diagonal checks; external `restore()` also checks positive definiteness. Admission storage is trusted, and administrative state edits are unsupported.

For example, a monitor JSON can declare:

```json
{
	"version": "campaign-coverage-v1",
	"spec": {"seed": 42, "input_dim": 384, "dim": 64, "ridge": 1.0},
	"encoder_identity": {"model": "replace-with-actual-stable-identity"},
	"domain": "text",
	"max_text_chars": 4096,
	"looks": [16, 32, 64, 128],
	"scales": [8, 32]
}
```

Coverage is a geometric proxy, not mutual information or predicted student learning. The score is a maximum of EWMA marginal gains at declared looks, never cumulative coverage as a volume threshold. Calibration and runtime use the identical domain, prefix schedule, scales, projection, encoder and character bound. Other domains abstain and do not advance the supported domain's prefix. This version supports one declared domain per configuration; it does not pool domain nulls or search an undeclared collection of domains.

A missing, character-truncated, token-truncated or unknown-completeness embedding makes the entire domain prefix ineligible for ranks. Later valid vectors cannot repair the missing prefix. Evidence explicitly reports `missing_embedding_prefix`; it supplies neither a substituted vector nor a zero-gain observation. Serving reuses the existing bounded encoder path outside the admission transaction, including tokenizer completeness checks. There is no extra teacher call.

Build a new calibration with `uv run harness campaign-coverage-calibrate --monitor MONITOR.json --campaigns BENIGN.jsonl --output COVERAGE.json --embeddings CACHE.npz --alpha 0.01`. The cache uses the existing CLI's SHA-256 text keys and vectors, and its exact identity must match the monitor. Without `--embeddings`, the command uses an encoder on CPU by default and requires tokenizer completeness. Each account in the request file is one independent benign campaign, in file order, containing exactly the last declared prefix length, all in the declared domain. The caller must establish benign provenance, independence and that domain assignment; the command cannot establish these from text. Input is the raw user turn, matching serving serialization. Output is created exclusively.

Set `coverage` to an object containing `monitor`, the complete output artifact under `calibration`, and the printed `calibration_sha256`. For observation without a null, set only `monitor`. The digest binds the artifact contents, its monitor and alpha; mismatches refuse configuration. Existing path-policy artifacts and legacy calibration formats are unchanged.

Calibration stores one maximum per benign campaign over every declared scale and look. Runtime retains only the sketch, EWMA values, prefix position and running maximum m. Its conservative tie-inclusive rank is `(1 + count(M_i >= m)) / (n + 1)`. It abstains if `1/(n+1) > alpha`, for an unsupported domain, for an incomplete prefix, before the first look, or after the finite horizon. A calibration artifact is required for coverage to start treatment, and API throttle mode still requires `HARNESS_CAMPAIGN_ALLOW_UNQUALIFIED_ENFORCEMENT=true`. Alpha and example settings are development values, not validated thresholds. Rank validity assumes exchangeable complete benign campaigns under the declared monitoring contract; this implementation has no independent benign audit.

The finite coverage record persists across recent-window expiry, cooldown expiry and process restarts, atomically with campaign admission. It never renews its horizon or starts an additional statistical trial after cooldown. A previous coverage crossing cannot immediately rearm treatment by default. State stops growing at the horizon. There is no inactivity reset or automatic lifetime renewal.

The separate `lifetime` member enables `campaign-lifetime-v1`, a durable cooldown count and `cumulative_admitted_attempts`, meaning unique observations that pass campaign admission, not completed teacher responses. The record survives recent-window and cooldown expiry. It does not count durable retries or blocked probes. Set `escalation_factor` above 1 to explicitly allow a retained calibrated crossing to rearm after cooldown and multiply cooldown duration by that factor for each prior cooldown, capped at `cooldown_cap_seconds`. Factor 1 preserves the fresh-window rule and does not rearm stale evidence. All cooldown starts count toward escalation, including heuristic starts. Escalation changes treatment duration, not the interpretation or qualification of the rank. Account deletion, retention administration, fleet cardinality limits, lifetime error budgeting beyond the finite horizon and independent operational validation remain open.
