# Decoder watermark probe

`experiments/watermark` is a CPU-only prototype of a keyed soft KGW decoder watermark: the bias applied to logits during generation, and the statistical test that looks for it in the output. It implements the custom-processor part of the [watermark decoder report](<reports/Watermark decoder implementation.md>).

It has never touched a model. Nothing here shows that a watermark survives on Qwen3.5-9B, what it costs in answer quality, or whether a student inherits it. TRACE's teacher still returns unwatermarked output, and no engine loads this code.

This is unrelated to `harness watermark`, which tests whether a suspect model learned the [keyed noise answers](harness.md) served to accounts at L4. That command works on answers. This probe works on token choices during generation.

## What exists

| Module | Purpose |
| --- | --- |
| `core.py` | Keys, profiles and the green-list definition shared by both sides (`kgw-hmac-splitmix` version 1) |
| `bias.py` | `apply_bias` for one row, and `KgwBatchProcessor` for batches with recycled slots, reordered rows and per-request keys and status |
| `detect.py` | Binomial test on one event per context, abstention on short samples and Holm correction across candidate keys |
| `schemas.py` | `manifest.json` and `generations.jsonl` rows from the report, refusing credential-shaped field names |
| `vectors.py`, `fixtures/reference.json` | Reference vectors that fail when the algorithm changes by accident |

The project is separate from the Python 3.14 app environment and depends only on numpy, so it can later load into a serving environment on Python 3.12.

```bash
cd experiments/watermark
uv run --frozen pytest -q
uv run --frozen watermark-probe vectors --fixture fixtures/reference.json --output artifacts/vectors.json
uv run --frozen watermark-probe bench --output artifacts/bench.json
```

`replay`, `calibrate` and `evaluate` are registered but exit with status 2. They need a serving stage and data that do not exist yet.

## Synthetic results

These runs generate from a seeded table of random bigram logits, with no real language involved. They check that the modules agree with each other and with theory. They are not an estimate of performance on a real model, and they qualify no threshold.

Each cell is 30 sequences at gamma 0.25, context width 2, alpha 1e-3 and a minimum of 32 unique contexts, with the prompt available to the detector. `sharpness` scales the logits, so a higher value means lower entropy.

| Delta | Sharpness | Detected at 64 tokens | Detected at 256 tokens | Mean green rate at 256 tokens |
| --- | --- | --- | --- | --- |
| 0.5 | 1 | 2/30 | 23/30 | 0.358 |
| 0.5 | 4 | 1/30 | 16/30 | 0.337 |
| 1 | 1 | 20/30 | 30/30 | 0.472 |
| 1 | 4 | 10/30 | 30/30 | 0.416 |
| 2 | 1 | 30/30 | 30/30 | 0.712 |
| 2 | 4 | 30/30 | 30/30 | 0.605 |

The wrong key was never detected in these runs. That is an observation, not a false-positive estimate: the cells reuse seeds, and each 64-token output is a prefix of its 256-token output, so the runs are not independent trials. With flat logits the green rate is gamma e^delta / (gamma e^delta + 1 - gamma), about 0.355, 0.475 and 0.711, and the rates at sharpness 1 land close to those values. Lower entropy reduces both the green rate and detection, as the report predicts for constrained answers.

## CPU benchmarks

`watermark-probe bench` times each path on synthetic inputs and records the CPU, thread settings and load alongside the results. These figures come from one run on a Ryzen AI 9 HX 370, at nice 19 and load average about 2, with a study running on the same machine. They measure only this arithmetic, including array allocation, and leave out any model, tokenizer or engine. Bias and batch logits are float32.

| Path | Case | p50 | p95 |
| --- | --- | --- | --- |
| Seed derivation | Context width 4 | 2.6 us | 2.6 us |
| Mask expansion | 5,000 tokens | 13.7 us | 14.1 us |
| Mask expansion | 50,000 tokens | 153 us | 162 us |
| Mask expansion | 200,000 tokens | 897 us | 933 us |
| Single-row bias | 5,000 tokens | 36.6 us | 40.5 us |
| Single-row bias | 50,000 tokens | 404 us | 424 us |
| Single-row bias | 200,000 tokens | 2.2 ms | 2.5 ms |
| Batch of 16, per step | 50,000 tokens | 6.9 ms | 7.1 ms |
| Batch of 16, per step | 200,000 tokens | 28.1 ms | 29.0 ms |
| Detection, one key | 1,024 unique contexts | 4.3 ms | 4.5 ms |
| Detection, eight keys | 1,024 unique contexts | 30.0 ms | 30.1 ms |

The batch path works through rows one at a time, so its cost grows with batch size, and at 200,000 tokens it stays near 400 to 570 tokens per second of bias overhead alone. Detection scales with unique contexts and candidate keys, because each context needs one HMAC per key.

## What the test assumes

Tokens that share a context share a seed, so their green memberships are dependent. A binomial test over every unique (context, token) pair would therefore claim a null distribution it does not have. Instead, each unique context contributes only its first scorable token in text order, a rule that does not depend on the tested key. Each seed is then used once.

If HMAC-SHA256 behaves as a PRF and the text is independent of the tested key, the green count follows Binomial(`unique_contexts`, `green_probability`), and the p-value is exact. `green_probability` is the rounded threshold over 2^64, which can differ slightly from gamma. Both conditions are assumptions: the text's independence from the key fails, for example, if the model was trained on output watermarked with that key.

## Known limits

- The token expansion uses a splitmix64 mixer, which is not a cryptographic PRF. An adversary who sees many memberships under one context might recover that context's seed. Only the context step uses HMAC.
- `key_id` is a label and does not enter the seed. Tenant separation depends entirely on secrets being different, and rotation needs a new secret or epoch.
- Generation and detection must agree exactly on gamma, context width, epoch and `vocab_size`. `vocab_size` must be the engine's logits width, and a mismatch changes every green list without raising an error. Tokenizer and template identity are not in the seed, so a manifest has to record them.
- Detection sees token IDs. Scoring copied text requires re-tokenizing with the same tokenizer, and nothing tests that round trip yet.
- Scoring one event per context discards later tokens under a repeated context, which costs some power on repetitive output.
- Nothing enforces apply, sample, then `record_token`. `apply` can repeat for a retry, using the same history and counting another row, so a caller must compare row counts with `tokens_recorded`.
- Evidence records reject anything nested deeper than 32 levels, counting the record itself as level 0.
- The schema guard reads field names only. A secret inside a string, such as an `error` message, passes. Whatever writes evidence should copy known parameters rather than whole request or config objects.
- `bias.py` rejects NaN and positive-infinity logits, and rejects a delta that overflows the logits dtype, instead of producing an infinity.
- Holm correction covers candidate keys only. Corrections across windows and profiles are left to the caller.
- `Detection` differs from the report's `detections.jsonl` contract, and no schema for that file exists. It uses `z_score` rather than `z_score_or_native_statistic`, has no `unsupported` decision, and leaves `request_id` and `detector_version` to the caller.
- Very strong p-values underflow to 0.0 in `raw_p_value`. `binomial_log_upper_tail` keeps the size.

## What blocks real evaluation

- The report's first step assumes vLLM with a full-precision model. The local teacher is now a Q4 GGUF on `llama-server`, and this machine has an 8 GB GPU that a running study already uses.
- `KgwBatchProcessor` follows the report's description of the vLLM MRV2 interface but has no adapter, and that interface is not checked against a real engine. `llama-server` would need its own integration.
- The teacher adapter sends no watermark controls and defaults to temperature zero, and the endpoint has no trusted tenant identity to select a key.
