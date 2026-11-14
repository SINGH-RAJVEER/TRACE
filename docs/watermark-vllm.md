# KGW watermark on vLLM

`packages/watermark` is the maintained `trace-watermark` package. It holds the keyed
`kgw-v1` generation and detection implementation, its vLLM logits processor, reference
vectors and tests. It replaces the former experiment; there is no serving import from
`experiments/`.

KGW changes the green list with the preceding tokens. Unlike the existing `unigram-v1`
watermark, it cannot be expressed as a static `logit_bias` map. vLLM invokes
`trace_watermark.vllm:KgwLogitsProcessor` on the next-token logits before sampling.
The processor is not argmax-invariant, so it also runs for greedy decoding.

**Integration is not an effectiveness qualification.** CPU tensor tests exercise generation,
batch lifecycle, key separation and detection. They do not establish live Qwen quality,
GPU throughput, student inheritance or adversarial robustness. Do not infer those from
moving the package out of `experiments/`.

## Install and serve

vLLM is the only supported inference runtime. This adapter targets **vLLM 0.28.0's
`vllm.v1.sample.logits_processor` batch interface**, not the different MRV2 interface
in newer documentation. The adapter refuses another vLLM version rather than guessing
at compatibility.

Keep the Python 3.12 serving environment separate from the Python 3.14 app workspace:

```bash
uv venv --python "$VLLM_PYTHON" .venv-vllm
uv pip install --python .venv-vllm/bin/python \
    -r packages/watermark/requirements-vllm.txt ./packages/watermark
just setup
```

Set these in `.env` for the shared launcher and teacher API:

```dotenv
TEACHER_WATERMARK_SCHEME=kgw-v1
TEACHER_WATERMARK_KEY_FILE=artifacts/teacher.watermark.key
TEACHER_WATERMARK_CONTEXT_WIDTH=2
TEACHER_WATERMARK_GAMMA=0.25
TEACHER_WATERMARK_DELTA=2.0
TEACHER_WATERMARK_EPOCH=0
```

Also set `TEACHER_WATERMARK_VOCAB_SIZE` to the model's exact vocabulary size, **not**
`TEACHER_WATERMARK_ELIGIBLE_VOCAB` or an approximate tokenizer length. The engine checks it
against `model_config.get_vocab_size()` and the logits tensor width. A mismatch fails
instead of silently changing green lists. Pin the model and tokenizer revisions.

```bash
just teacher-vllm
# In another shell:
just teacher
```

The shared vLLM launcher provisions the private master key atomically if missing,
loads the KGW logits processor and acknowledgement middleware, and disables asynchronous
scheduling so the processor sees complete output-token history. It does not pass key
material in command arguments. Workers only read the existing key, and fail if it is
missing or malformed. For remote/multi-worker serving, provision the same key and
configuration privately on every process before starting them.

Speculative decoding is unsupported and rejected. Pooling is not a generation path.
Direct vLLM deployments must load both the processor and
`trace_watermark.middleware.KgwMiddleware`, disable asynchronous scheduling, and use the
pinned version. Do not expose the upstream vLLM port to untrusted clients; only the
authenticated TRACE gateway should choose per-principal watermark keys.

`unigram-v1` remains the default to avoid silently changing the marking of an existing
deployment. Selecting KGW is explicit. Use a **fresh harness state database and fresh
evaluation artifacts** when changing scheme, key epoch or profile. Do not pool schemes
in one detection report; their derived keys and green lists are not interchangeable.

## Request and release contract

1. The gateway chooses its existing per-principal public key ID.
2. The teacher derives a domain-separated KGW key from the shared master, and sends only
   the public key ID, profile and a nonce-bound authentication tag in `vllm_xargs`.
3. vLLM validates the profile and authentication tag at admission. The engine uses the
   current prompt/output history supplied by vLLM, not a guessed sample counter.
   Repeated `apply` calls do not advance history; removal, replacement, compaction and
   swaps discard or move the corresponding key state.
4. Middleware acknowledges a successful response only when the pinned engine configuration
   includes the KGW processor, the right vocabulary, synchronous scheduling and a
   non-speculative generation runner.
   The acknowledgement is authenticated with the shared master and bound to the teacher's
   fresh nonce and exact profile.
5. A missing, replayed or mismatched acknowledgement withholds the answer. There is no
   fallback to unigram or unmarked generation. Unkeyed requests remain unchanged.

The acknowledgement proves agreement with the trusted server's **configuration**, not
independent measurement of a statistical signal or a per-token execution audit. An answer
shorter than the context window may have no biased tokens. Application still requires a
correct engine and adapter; end-to-end detection is the separate empirical check.
Neither the derived secret nor master key is sent over HTTP. Acknowledgements, configuration
and token histories are not added to the gateway's client response.

## Detect copied text or suspect model output

Use JSONL with one `text` field per answer and the exact teacher tokenizer:

```bash
uv run --frozen distill watermark detect-kgw \
    --texts suspect.jsonl --key-file artifacts/teacher.watermark.key \
    --principal alice --principal bob \
    --tokenizer Qwen/Qwen3.5-9B \
    --tokenizer-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
    --vocab-size "$TEACHER_WATERMARK_VOCAB_SIZE" \
    --output artifacts/kgw-detection.json
```

Match gamma, context width and epoch to generation. Without the exact templated prompt
or withheld reasoning, the detector skips the first `context_width` tokens of each answer.
Contexts never span answer boundaries. Each distinct context contributes its first
scorable token only once across the **whole pool**, so repeated answers do not inflate
evidence. Candidate keys are corrected jointly with Holm-Bonferroni; short samples abstain.
The report records the scheme, full profile digest and pinned tokenizer, never secrets.
Use `distill watermark detect` only for historical/explicit `unigram-v1` outputs.

## Checks and remaining limits

```bash
uv run --frozen pytest packages/watermark/tests tests/test_kgw_watermark.py
uv run --frozen trace-watermark vectors \
    --fixture packages/watermark/fixtures/reference.json --output artifacts/kgw-vectors.json
uv run --frozen trace-watermark bench --quick --output artifacts/kgw-bench.json
```

The engine tensor tests need torch. They can run in a separate Python 3.12 CPU-torch
environment with `trace-watermark`, numpy and pytest installed:

```bash
python -m pytest packages/watermark/tests/test_vllm_adapter.py
```

Those tests use a small pinned-interface shim instead of loading vLLM/CUDA, and are
skipped in the minimal app environment. They are not a substitute for a live vLLM run.
On a qualified GPU host, run protected and unprotected requests through the teacher API,
test the resulting answers with `detect-kgw`, and measure quality and throughput before
making deployment or inheritance claims.

The algorithm retains `kgw-hmac-splitmix` version 1 and its reference vectors. Its HMAC
context seeding is cryptographic, but splitmix64 token expansion is only a statistical
mixer: many observed memberships under one context may reveal that context's seed.
Re-tokenization of copied text can differ from generated token IDs. Paraphrasing, editing,
short answers and repetitive contexts reduce detection power. A positive test does not
prove training or intent. Corrections across windows/profiles remain the caller's job.

## Pinned upstream interface sources

These are primary sources for the integration, rather than assumptions from the prototype:

- [v0.28.0 custom processor documentation](https://github.com/vllm-project/vllm/blob/v0.28.0/docs/features/custom_logitsprocs.md):
  loading by FQCN, admission validation, greedy behavior and removed/added/moved ordering.
- [v0.28.0 batch interface](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/sample/logits_processor/interface.py):
  `BatchUpdate`, directionality, live output lists and method signatures.
- [v0.28.0 processor loading](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/sample/logits_processor/__init__.py):
  explicit custom processor loading and speculative-decoding restrictions.
- [v0.28.0 chat request protocol](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/entrypoints/openai/chat_completion/protocol.py):
  `vllm_xargs` maps into `SamplingParams.extra_args`.
- [v0.28.0 HTTP server](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/entrypoints/openai/api_server.py):
  middleware loading and `app.state.vllm_config`.
