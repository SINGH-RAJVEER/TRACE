# Extraction harness

`packages/harness` implements the paper's detector and access policy as a library: query embeddings, the window MMD score, benign-only calibration, accumulated account evidence, and the graduated actions of Table I. Policies take the account, request text, and time as arguments and never read the clock. A campaign simulator and a future HTTP server can therefore drive identical decision code. No server exists yet.

## Install

```bash
uv sync --all-packages              # library, CLI, and tests; no encoder
uv sync --all-packages --extra embed  # adds PyTorch and Transformers for the BGE encoder
```

The `harness` command comes with the package. Replays with precomputed embeddings and the tests do not need the `embed` extra.

## Pipeline

| Step | Module | Paper |
| --- | --- | --- |
| Embed the current user turn with `BAAI/bge-small-en-v1.5` at revision `5c38ec7`, CLS pooling, unit length, 512-token limit with truncations counted | `encoder.py` | III-B |
| Score a window of `w_a` queries: unbiased squared MMD, kernel = mean of four RBF kernels at 0.5, 1, 2, and 4 times the median-distance bandwidth, averaged over 20 fixed reference windows, negative estimates kept | `mmd.py` | Eq. 1 |
| Turn a score into its upper-tail rank among benign calibration scores | `calibration.py` | Eq. 3 |
| Accumulate evidence per account and map it to a level | `accumulators.py`, `calibration.py` | Eq. 4, III-C |
| Apply the level's action, hold the level for the horizon, and reset after 24 hours or 10,000 attempts | `policy.py` | Table I, III-C, III-D |
| Issue and verify SHA-256 proof-of-work challenges | `pow.py` | Table I, L3 |

A completed window includes the current request and is scored before the victim runs, so it can change that request's action. Windows do not overlap. Until a window completes, the previous level applies, and a new account starts at L0.

## Accumulators

All rules read the same window p-values, so they can be compared with everything else fixed. The calibration artifact holds thresholds for each one.

| Name | Rule | Purpose |
| --- | --- | --- |
| `instantaneous` | Current window only, `-log p` | Primary baseline: no memory |
| `paper` | `r = max(0, 0.9 r - log p - 1)` | The draft's Eq. 4 |
| `paper_no_memory` | Same with decay 0 | Ablation |
| `paper_no_decay` | Same with decay 1 | Ablation |
| `e_process` | `log M += log e`, with `e = 0.5 p^(-0.5)` | Session e-process; theoretical thresholds `log(1/alpha)` hold only if each window's p-value is valid given the session so far |
| `e_cusum` | `log R = max(0, log R) + log e` | E-process with a floor, so a long clean history cannot bank negative evidence against a later attack |

`paper` is the default because it is the rule the draft specifies. The choice of primary rule is still open.

Thresholds are calibrated by default: for each benign policy-calibration horizon, take the highest statistic reached, then choose each level's threshold so that at most its target share of horizons strictly exceed it. Targets are 10%, 5%, 2%, and 1% for L1 to L4. Thresholds are nested, and the artifact records the achieved rate next to each target. When the sample is too small for a target, the threshold becomes the largest observed value and the achieved rate is lower than the target; it is never extrapolated. `--thresholds ville` uses `log(1/alpha)` instead, for `e_process` only.

## Actions

| Level | Action |
| --- | --- |
| L0 | Serve |
| L1 | Serve; the decision record carries the level as an alert |
| L2 | Serve no sooner than 1 second after the account's previous admitted request; excess requests queue |
| L3 | L2 spacing plus a proof-of-work challenge per response: 20 leading zero bits, bound to the account, a SHA-256 digest of the request, and a 5-minute expiry. Each challenge is accepted once |
| L4 | Refuse for the rest of the horizon. The request that triggers L4 receives no output |

Each `Decision` carries the action, the earliest release time, any challenge, and telemetry: the level, whether a window completed, its p-value, the statistic, and the deception flag. `client_view()` returns only what a client should see: the action, release time, and challenge. Risk values stay hidden from the attacker, as the threat model requires.

`AllowPolicy`, `QuotaPolicy`, and `RatePolicy` share the same interface. They are the undefended service and the two block-only baselines. Calibration fits the quota, in requests per horizon, and the per-minute rate cutoff to the L4 target on the same benign horizons. The rate cutoff needs request times.

## Deception at L4

A separately reported arm replaces refusal at L4 with plausible wrong answers. The paper's main policy still refuses and leaves served answers unchanged; this arm adapts adaptive misinformation (Kariyappa and Qureshi, CVPR 2020), keyed response changes and watermarking (DAWN, Szyller et al., ACM MM 2021), and trace rewriting (Ma et al., ACL 2026) to a text API. Probability-perturbation defenses such as prediction poisoning do not apply, because the service returns text rather than class probabilities.

```python
MMDPolicy(calibration, scorer, l4_action="deceive", deception_rate=1.0, deception_key=secret)
```

- **Indistinguishable from L3.** An account at L4 keeps L3's one-second spacing and proof-of-work challenge. The decision's hidden `deceive` flag marks responses to replace. `client_view()` never shows it, so the switch from L3 to L4 gives the client no signal.
- **Keyed by request.** Every wrong answer and the choice of which requests to deceive come from HMAC-SHA256 of the secret and the request. Repeated and Sybil queries for the same request get the same answer, and the attacker cannot predict it. `deception_rate` deceives a keyed share of an L4 account's requests and serves the rest truthfully.
- **Applied only at L4,** the level calibrated so that at most 1% of benign horizons reach it. A falsely flagged user receives wrong answers without being told, which is worse than a refusal. Report this as silently corrupted legitimate responses, inside unconditional utility.

| Task | Deceiver | Wrong answer |
| --- | --- | --- |
| AG News | `LabelDeceiver` | A keyed choice among the three other labels |
| SQuAD | `SpanDeceiver` | A span from the same passage that shares no token with the answer: a number for a number, a capitalized name for a name (ignoring a leading article), otherwise a word window of the answer's length within one sentence |
| GSM8K | `NumericTraceDeceiver` | The final number shifted by a keyed offset. With a `rewriter` callable, a model call rewrites the solution to reach it, as in Ma et al. Without one, the final line and the last occurrence of the original number are replaced |

`released_text()` returns what the client receives: nothing for a refusal, the victim's response, or the deceptive response. When a deceiver cannot form an answer, for example a SQuAD request without a passage, it falls back to refusal.

A check on 2,000 attacker-pool examples per task, using the gold answers as stand-in victim responses, formed a wrong answer every time. Known weaknesses:

- **SQuAD:** answers to descriptive questions get word windows that read as fragments, such as "through 3 statues and the". A noun-phrase chunker or a model rewrite would make them more plausible.
- **GSM8K:** without a rewriter, the edited line keeps its arithmetic, so "8*2=26" is visibly wrong to any attacker who checks the arithmetic. Use the rewriter for experiments.

### Watermark test

The same key tests whether a suspect model learned the deceptive answers:

```bash
HARNESS_DECEPTION_KEY=<hex> uv run harness watermark --task ag_news --probes probes.jsonl
```

Each probe line holds `request`, `victim_response`, and `suspect_response`. Agreement is the share of probes where the suspect gives the keyed wrong answer. The p-value ranks that agreement among the agreements under 999 unrelated control keys, so it does not assume how often a clean model happens to give a particular wrong answer. The key comes only from the environment and never appears in output files. `harness replay --l4-action deceive --deception-rate Q` reads it the same way and counts deceived decisions.

The evaluation still needs a verifying attacker that re-asks a share of prompts from clean accounts, where answers are truthful, and a filtering attacker that drops answers its own student disagrees with. Defense-penetrating attacks such as D-DAE (Chen et al., IEEE S&P 2023) motivate both tests.

## Calibrate

Calibration reads three disjoint benign files of JSON lines with `account`, `text`, and optional `time` in seconds. Requests of an account must appear in order.

```bash
uv run --extra embed harness calibrate \
	--fitting fitting.jsonl \
	--score-calibration score_calibration.jsonl \
	--policy-calibration policy_calibration.jsonl \
	--output-dir artifacts/harness/ag_news
```

| Input | Role |
| --- | --- |
| Fitting | Draws the 20 reference windows and sets the kernel bandwidth from up to 2,000 other fitting queries |
| Score calibration | Each account's complete, non-overlapping windows form the null score distribution |
| Policy calibration | Each account horizon's highest statistic sets the level thresholds and the baseline cutoffs |

The output directory holds `calibration.json` and `references.npy`. The JSON records the configuration, encoder identity and truncation count, bandwidth, sorted calibration scores, thresholds and achieved rates per accumulator, baseline cutoffs, counts, and SHA-256 hashes of each input and of the reference array. Loading verifies the reference hash. An existing calibration is never overwritten.

## Replay

```bash
uv run --extra embed harness replay \
	--calibration artifacts/harness/ag_news \
	--requests benign_test.jsonl \
	--output outputs/harness/benign-paper.jsonl \
	--policy mmd --accumulator paper
```

Replay sends requests through one policy in file order, writing one decision per line. Requests without a time are spaced one second apart. On benign traffic, the summary's `reached_rate` per level is the account-level false-escalation rate FPR_ℓ. Replay counts challenges but does not solve them, so it measures detection, not what a solving client receives. `--policy` also accepts `allow`, `quota`, and `rate`.

## Accounts

`harness pool-accounts` turns a prompt pool, such as `data/tasks/ag_news/legitimate/fitting.jsonl`, into accounts of consecutive requests. It exists for plumbing checks only. Those accounts are neither realistic clients nor independent samples. Real calibration needs legitimate accounts built with per-client topic concentration and timing, and enough of them: the paper asks for about 3,000 independent benign calibration accounts and 3,000 test accounts per endpoint for the 1% target.

## Smoke run

On 2026-10-16, the AG News pools from `distill tasks build` were cut into 500-request accounts and calibrated on CPU in about 10 minutes: 6,000 fitting requests, 107 score-calibration windows, and 22 policy-calibration accounts. Replaying 22 held-out benign accounts and one 2,000-request extraction account ran every policy end to end.

The numbers describe the construction, not the detector:

- **Quota.** Every benign account had exactly 500 requests, so the fitted quota was 500. It blocked the extractor at its 501st request by construction.
- **MMD.** The extractor's first window received the smallest possible p-value, 1/108. That account is a contiguous slice of the AG News training file, whose row order appears to carry its own distribution shift. Its label mix was balanced, so labels are not the cause.
- **Sample size.** With 22 calibration accounts, the 2% and 1% targets cannot be met. The L3 and L4 thresholds fell back to the largest benign value, with an achieved rate of 0.

## Not yet implemented

- The HTTP wrapper and its decision and challenge headers
- The aggregate monitor over 1,500-request windows, which raises alerts only
- Per-query witness attribution for flagged windows
- Realistic legitimate-account construction
- The campaign layer that drives attacks through these policies
- The adaptive tests against deception, and a model-backed GSM8K rewriter in the server
