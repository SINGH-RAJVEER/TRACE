# v5: undefended A2 student, before the defense stage

This run trains one shared student on every usable response from the completed undefended A2 campaign of the bounded gateway study ([v2](../v2-2026-10-21-gateway-bounded-v1/), seed 42, 6,000 submissions). It is the pre-defense reference: what an attacker learns from the A2 ledger when no harness intervenes. Training finished on 2026-10-30. Held-out evaluation against the base student is pending, so this run has no measured distillation gain yet.

## Inputs

- Ledger: `artifacts/gateway-bounded-v1/A2/s42-b6000/undefended/ledger.jsonl`, SHA-256 `55a778db0121e830aa8114f59b613464c573368073d27b421988ce03cdef9a18`. It was collected from 2026-10-21 21:28:40 to 2026-10-22 03:12:52 UTC (20,652 s) from `Qwen/Qwen3.5-9B`, non-thinking, temperature 0, seed 42.
- Schedule SHA-256: `1713ae17492d5224a4090a0dca736063d7329194d8a307e0cf7afd38273a01a3`.
- 6,000 responses, 1,500 per domain, zero failures, 588,309 released output tokens. Each domain used a fixed 1,500-prompt share of its attacker pool, not the whole pool: AG News 35,944, coding 22,368, GSM8K 3,389, SQuAD 10,353.

Usability review (`task-usability-v1`, no gold labels) accepted 4,816 responses and rejected 1,184, all for `finish_reason: length`:

| Domain | Output cap (tokens) | Accepted | Rejected |
|---|---:|---:|---:|
| AG News | 16 | 1,500 | 0 |
| SQuAD | 64 | 1,499 | 1 |
| Coding | 512 | 1,450 | 50 |
| GSM8K | 256 | 367 | 1,133 |

## Training

- Base model `Qwen/Qwen3.5-2B-Base`, revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`, BF16 LoRA (rank 16, alpha 32, dropout 0.05, all linear modules), 2,048-token limit, effective batch 16, seed 42.
- Recipe: learning rate 1e-4, three epochs, the recipe selected on development data in [v3](../v3-2026-10-29-gateway-replay/).
- Dataset: 4,816 rows, SHA-256 `ec0db543bbd84f33a40e40ca95f05cd0f17054c6c9bde1ba41bb02452447899c`. Prompts `8a0f9779…8d67`, reviews `c1579138…72f5`.
- 903 optimizer steps in 3,303 s on one RTX 4000 Ada (20 GB, WSL2, driver 596.51, CUDA 13.0, Python 3.14.8). Final mean training loss 0.1218; last logged step loss 0.0534, mean token accuracy 0.9876. Validation was disabled, so these are training-set figures, not held-out results.
- Code: checkout `e53503f`; `c64c4f0` added only a document. The training manifest does not record a code revision.
- A first launch failed with a CUDA allocator error; the successful run set `PYTORCH_ALLOC_CONF=expandable_segments:False` and started from scratch, not from a checkpoint.
- Final adapter `adapter/adapter_model.safetensors`, SHA-256 `237fd050aa66ad17f9065c0928d42e01241f8d87723f9d75ac3f13fa7321f7ed`. It requires the pinned base model.

## Files

`adapter/` holds the final adapter, tokenizer files, chat template, and the training manifest. Intermediate checkpoints and optimizer state are not included. `train.toml`, `provenance.json`, `train.manifest.json` and `reviews.summary.json` are the run's configuration and data summaries; `training.log` is the training output. Weights, `training_args.bin`, `tokenizer.json`, and the log are stored with Git LFS. The training rows, prompts and per-response reviews stay on the run machine under `artifacts/undefended-a2-benchmark/`; the ledger they derive from is committed with v2.

## Limitations

- No held-out evaluation yet. The adapter must be scored against the base student on each domain's held-out suite before any capability gain is claimed.
- GSM8K is barely represented: 367 of 4,816 training rows, and only responses short enough to finish within 256 tokens. The 256-token cap also applies at evaluation, where the v3 development student ran out of room on 261 of 373 qualification items. The GSM8K result from this adapter is not expected to be informative. The code cap has since been raised to 1,024 tokens (PR #18), which does not apply to this ledger.
- In v3's development qualification, only AG News cleared the 2-point transfer margin; SQuAD (+2.8, CI 0.8–5.0), coding (+0.2) and GSM8K (−14.5) did not. This run has three times as much data per domain, but whether those domains now qualify is unmeasured.
- One seed and one campaign order (A2, domain-sequential). It does not show seed variability.
