# Distillation pipeline

## Scope

This repository implements a black-box teacher to open-weight student experiment. The teacher is a hosted chat model available through an OpenAI-compatible `POST /chat/completions` endpoint. The student is a Hugging Face causal language model trained with LoRA or QLoRA.

The current method is sequence-level distillation. The collector saves the teacher's generated text, and TRL trains the student with cross-entropy on completion tokens. It does not use teacher logits.

The protection wrapper discussed in the paper is outside this implementation.

## Development environment

Run `devenv shell` to enter the pinned development environment, or allow `.envrc` with `direnv allow` to load it automatically. The shell provides Python 3.14, uv, Bun, Bash, just, a C/C++ compiler, and zlib. On Linux, it exposes the compiler's C++ runtime and `/run/opengl-driver/lib` so binary Python wheels can find native libraries and the host NVIDIA driver.

Shell entry runs `uv sync --all-packages` to install the API and collection dependencies plus pytest, pytest-asyncio, Ruff, and mypy from `uv.lock`. Python dependencies belong in the workspace `pyproject.toml` files; system tools and native libraries belong in `devenv.nix`.

Use these commands inside the shell:

```bash
just check
just test
uv sync --all-packages --extra eval
uv run --extra eval python -c 'import datasets, huggingface_hub'
just setup-train
uv run --extra train python -c 'import torch; print(torch.cuda.is_available())'
```

The `eval` extra installs datasets and huggingface-hub. The `train` extra installs PyTorch, Transformers, datasets, Accelerate, PEFT, TRL, and bitsandbytes on Linux. These extras stay optional because the API and pipeline tests do not need the large training stack. Select the extra on subsequent `uv run` commands as shown above, since uv synchronizes the environment for each invocation.

QLoRA also requires a compatible NVIDIA GPU and a working host driver. The shell provides the compiler and library paths; the CUDA runtime dependencies come from the locked training wheels. The vLLM teacher described below runs as a separate upstream service with its own environment, either on this machine or another host.


## Adopted model pairing

- Student: `Qwen/Qwen3-4B-Base` at revision `906bfd4b4dc7f14ee4320094d8b41684abff8539` (Apache 2.0, verified 2026-09-05).
- Teacher: `Qwen/Qwen3-14B` instruct at revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, run in non-thinking mode.

Both use the Qwen3 chat format. Their pinned tokenizer configurations differ, including the default EOS token, so training explicitly sets `<|im_end|>` and passes `enable_thinking=false`. Measure peak memory at the intended context length and batch size before choosing hardware. QLoRA reduces frozen-weight memory, but activations, optimizer state, and KV cache still consume memory. The 14B teacher's BF16 weights alone are roughly 28 GB; the launcher uses the original checkpoint without a quantization override, so a 24 GB card needs a separately configured supported quantization recipe. The 4B student and 14B teacher leave a useful capacity comparison, but improvement must be measured on held-out tasks.

Self-hosting the teacher with vLLM keeps provider terms out of the picture entirely:

```bash
vllm serve Qwen/Qwen3-14B --revision 40c069824f4251a91eefaf281ebe4c544efd3e18 \
	--host 127.0.0.1 --port 8001 \
	--default-chat-template-kwargs '{"enable_thinking": false}'
```

Point `TEACHER_UPSTREAM_BASE_URL` at `http://127.0.0.1:8001/v1`. After starting it, confirm responses contain no thinking traces before the main collection run, and keep transfer prompts inside the student's 32K native context.

## Authorization gate

Do not collect data until the hosted provider has explicitly authorized using its output to train the planned student. Output ownership does not necessarily grant model-training rights.

Record the approval, contract amendment, or provider permission in `TEACHER_TERMS_REVIEW_ID`. The collector separately requires `--terms-review-id` and writes it into every ledger record. This is an audit reference, not an enforcement mechanism.

The prompt source also requires a corpus name, source ID, and license. The example fixture uses text authored for this repository under Apache-2.0.

## Teacher inference API

Configure `.env` from `.env.example`, then run:

```bash
uv sync --all-packages
uv run teacher-api
```

The service exposes:

- `GET /healthz`, which reports `ready` or `unconfigured`.
- `POST /v1/teacher/generate`, which validates a provider-neutral request and calls the configured hosted teacher.
- `/docs`, FastAPI's generated OpenAPI interface.

Set `TEACHER_API_TOKEN` to require a bearer token on generation requests. Bind to `127.0.0.1` by default. Add network TLS and an authenticated gateway before exposing the service outside a trusted machine.

Example request:

```json
{
	"messages": [
		{"role": "system", "content": "Answer accurately and concisely."},
		{"role": "user", "content": "Why is the sky blue?"}
	],
	"temperature": 0,
	"max_output_tokens": 256,
	"seed": 42,
	"enable_thinking": false
}
```

The upstream adapter sends `max_output_tokens` as the widely supported `max_tokens` chat-completions field. Add a provider adapter if the chosen authorized provider uses a different protocol. API keys remain in the API process and are never returned to collectors.

`enable_thinking` is optional. Set it to `false` for the Qwen/vLLM collection recipe; the upstream request then contains `chat_template_kwargs`. Omit it for providers that do not support that extension. The collector's `--no-thinking` option forwards the setting, records it in provenance, and rejects responses containing thought markers. This check rejects marker-bearing output; it does not establish task correctness.

## Prompt input

The collector accepts JSONL. Each row has a stable ID, a split assigned before teacher collection, task metadata, source rights, and structured messages:

```json
{
	"example_id": "support-0001",
	"split": "train",
	"task": "support-answer",
	"source": {
		"corpus": "licensed-support-v1",
		"source_id": "row-1",
		"license": "internal-approved"
	},
	"prompt": [
		{"role": "user", "content": "How do I reset the device?"}
	]
}
```

Use grouped train, validation, and test splits. Related templates, documents, users, or generated variants belong in the same split. Do not collect training responses for the frozen test split.

Set `source.group_id` when multiple source rows come from one document, conversation, or problem family. Preparation uses `(corpus, group_id)` as the split-separation key, falling back to `(corpus, source_id)` when the group is omitted. Declare groups before collection; exact prompt equality cannot detect paraphrases or semantic overlap.

## Collection and preparation

Run the teacher API first. Then collect responses:

```bash
uv run distill collect \
	--input data/examples/prompts.jsonl \
	--output data/teacher/ledger.jsonl \
	--teacher-url http://127.0.0.1:8000 \
	--terms-review-id "$TEACHER_TERMS_REVIEW_ID" \
	--concurrency 4 \
	--max-attempts 3 \
	--no-thinking
```

Collection is resumable by `example_id`. It rejects duplicate input IDs, retries teacher failures, and appends each successful response from its worker, so a later failure does not discard completed results. The ledger records teacher identity, the actual upstream endpoint, request parameters, provider request ID, token usage, source rights, approval reference, finish reason, and basic checks. A sibling `ledger.attempts.jsonl` file records every successful and failed call, including prompts whose retries all failed. Test prompts are skipped unless `--include-test` is set for a controlled teacher evaluation.

Resume verifies the prompt, split, task, source metadata, generation parameters, and approval reference for IDs already in the ledger. A mismatch requires a new ledger. The collector cannot detect a provider changing the underlying model behind the same endpoint; pin the served model and keep its deployment records.

Collection also exposes `--top-p` and `--top-k`; both are omitted from upstream requests unless supplied and are recorded for resume checks. `top_k` and the thinking switch need an upstream implementation that supports those extensions, such as vLLM. For a reviewed Qwen non-thinking sampling pilot, compare the greedy baseline with `--temperature 0.7 --top-p 0.8 --top-k 20 --no-thinking`, using separate ledgers and the same prompt/task contract. These are Qwen's documented inference recommendations, not a proven distillation optimum. Accept samples by task correctness and coverage before choosing a collection policy.

The basic checks reject blank content and nonterminal finish reasons. Passing responses remain `pending`; the collector does not call them training-ready. Store review decisions in a separate JSONL file so the raw ledger stays unchanged:

```json
{"example_id":"support-0001","status":"accepted","checks":["schema","grounding","human_review"],"reviewer":"reviewer-1"}
```

Task validators and human review must produce these decisions before a paper run. For example, compile generated code, validate structured output against a schema, check citations against supplied text, and audit every task slice.

Project accepted training rows into TRL's conversational prompt-completion schema:

```bash
uv run distill prepare \
	--input data/teacher/ledger.jsonl \
	--reviews data/reviews/reviews.jsonl \
	--prompts data/prompts/all-splits.jsonl \
	--output data/processed/train.jsonl \
	--split train
```

Preparation keeps reviewed and accepted rows from the requested split. Explicit review rejection always wins over `--accept-pending`. It drops duplicate prompts and source groups found in another split, and refuses to prepare the frozen test split. Pass the full original prompt inventory with `--prompts`, including validation and test rows that have no teacher response, to close the ledger-only contamination gap. Without `--prompts`, checks cover only rows present in that ledger. Separate collection files require the same full inventory.

A sibling `train.manifest.json` records input and output hashes, skip-reason counts, accepted task/source counts, and whether pending rows were allowed. Raw ledger files and prepared datasets remain private artifacts. Review near-duplicates and task correctness separately; the automated admission checks do not detect semantic overlap.

## Student training

Install the optional training stack:

```bash
uv sync --all-packages --extra train
```

Review `configs/train.example.toml`, which uses the pinned Qwen3-4B-Base checkpoint. Confirm that the base model license allows fine-tuning and the intended distribution. The two included prompts exercise plumbing; they are insufficient for instruction tuning a base model or measuring distillation quality.

For an experiment, include reviewed validation prompts in the original inventory, prepare them with `--split validation --prompts data/prompts/all-splits.jsonl`, and enable `validation_dataset_path` in the configuration. Training rejects empty datasets and exact train/validation prompt overlap, including datasets prepared outside this collector. The example leaves validation optional for the two-row plumbing fixture; a research run needs validation and a separately frozen test set.

Training input must contain only conversational `prompt` and `completion` columns, with one assistant completion. Unknown columns are rejected. This prevents externally supplied `input_ids`, `labels`, or per-row template overrides from bypassing the tokenizer and completion-loss checks.

Start training:

```bash
uv run --extra train distill train --config configs/train.example.toml
```

The trainer:

- Loads conversational JSONL with `datasets`.
- Uses completion-only loss through TRL `SFTTrainer`.
- Passes the same template arguments to the prompt, full conversation, and TRL tokenization. Empty Qwen thought wrappers are part of the prefilling prompt and are excluded from the loss.
- Applies LoRA to all linear layers.
- Uses NF4, double quantization, and BF16 compute when `qlora = true`.
- Seeds before model and adapter construction. Uses explicit cross-entropy, disables packing, and sets non-reentrant gradient checkpointing.
- Evaluates and saves every epoch when validation is supplied, loads the checkpoint with lowest validation loss, and evaluates the exported adapter at the end. `eval_strategy="steps"` is available; best-checkpoint selection requires `save_steps` to be a multiple of `eval_steps`.
- Saves the adapter, tokenizer, template hash, resolved base revision when available, both dataset hashes, preparation manifests, resolved training arguments, checkpoint choice, final validation metrics, package versions, uv lock hash, JJ revision, and available GPU/CUDA/driver metadata.

The Qwen example sets `eos_token = "<|im_end|>"` to match its chat template rather than the pretrained tokenizer's `<|endoftext|>` token. The loader explicitly aligns the model and generation configuration with the tokenizer's EOS and padding IDs before constructing the trainer, keeping training artifacts consistent with the chat contract.

The updated lock uses TRL 1.15.0, Transformers 5.19.0, PEFT 0.21.2, Accelerate 1.15.0, PyTorch 2.14.1, and bitsandbytes 0.50.2. TRL 1.15's `loss_type="nll"` uses a fused LM head to compute per-token log probabilities without constructing full sequence-by-vocabulary logits. It preserves ordinary cross-entropy while reducing one source of peak memory. `chunked_nll` is now a deprecated alias and is absent from this project's config. PEFT's `target_modules="all-linear"` excludes the output head; explicit output-head LoRA adapters are incompatible with this fused path for both NLL and DFT. No end-to-end speed or memory improvement has been measured here. The fused scoring kernel requires supported GPU/Triton execution; CPU checks and trainer construction do not prove that runtime path.

`attn_implementation="sdpa"` uses PyTorch's native attention backend. You can select `flash_attention_2` on compatible BF16/FP16 training hardware with the required backend installed. `activation_offloading=true` optionally exchanges GPU activation memory for CPU transfers. Keep offloading disabled until you measure its cost. Packing stays disabled to preserve the checked example boundaries. `loss_type="dft"` enables TRL's Dynamic Fine-Tuning objective for a separate ablation; its probability-weighted validation loss is not ordinary NLL and must not be compared directly with NLL runs or converted to perplexity. Compare generated-answer quality on the same frozen suite before adopting it.

The [dated research audit](research/teacher-student-distillation.md#modern-training-and-distillation-options-checked-in-this-audit) records the primary sources and the reasons for adopting these defaults. Liger, black-box on-policy training, data-selection methods, and newer Qwen checkpoints need separate compatibility and quality experiments.

`selective_gradient_checkpointing=true` enables TRL 1.15's selective activation checkpointing when `gradient_checkpointing=true`. It saves attention outputs instead of recomputing them during backward, exchanging additional activation memory for less recomputation. The default remains off until you measure the tradeoff on the training machine. The current recipe uses non-reentrant checkpointing and does not configure DeepSpeed ZeRO-3.

Set `qlora = false` for BF16 or FP16 LoRA when memory permits. Change `precision` to match the GPU. Four-bit training requires CUDA and bitsandbytes. CPU execution is suitable for API and pipeline tests, not the example QLoRA run.

If the base tokenizer has no chat template, set `chat_template_path` to a reviewed local Jinja template. Treat that template as part of the trained artifact. Before loading model weights, the trainer tokenizes every record and checks that prompt tokens are an exact prefix of the full sequence, that supervised completion tokens include the EOS token, and that no sequence exceeds `max_length`. Filter unsuitable records or change the reviewed template/limit rather than silently truncating teacher answers. The pinned Transformers API needs `return_dict=false` for these token-list checks. The configuration keeps `warmup_ratio`, which the locked Transformers version receives as fractional `warmup_steps`.

## Checks before running on training hardware

The default tests use fake teachers and small fixtures. They make no provider calls and perform no optimization steps:

```bash
uv run pytest
uv run ruff check .
uv run mypy apps packages tests
```

After preparing your actual data, check the configured tokenizer and complete tokenized targets without loading model weights or invoking a trainer:

```bash
uv run --extra train distill train --config configs/train.example.toml --preflight-only
```

Preflight prints data hashes, record and supervised-token counts, maximum sequence length, template identity, EOS, loss/attention settings, and whether validation is enabled. It may download tokenizer files. It makes no teacher calls, loads no model weights, and creates no adapter. Passing preflight checks data and serialization; it does not prove GPU capacity, numerical stability, or task quality.

The token report also includes minimum, median, and nearest-rank 95th-percentile lengths, total tokens, and the supervised fraction. Use these values to compare data budgets and choose an appropriate length limit. `just preflight-train` runs the same safe command, and `just test-training-setup` runs the opt-in construction tests below.

An opt-in test constructs TRL and PEFT against a tiny random local Qwen3 model. It checks actual completion labels, empty-thought-prefix masking, fused-loss configuration, and validation/warmup settings. The test replaces the optimizer-running entry point with an inspection function, so it never trains. Its QLoRA variant skips when CUDA is unavailable:

```bash
DISTILL_TEST_TRAINING_STACK=1 uv run --extra train pytest tests/test_training_stack.py
```

Keep tests in a separate uv environment if another process is synchronizing this checkout's environment:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/distillation-checks uv run --extra train pytest
```


## Evaluation

Freeze evaluation prompts before collection. Compare the base student, trained adapter, and authorized teacher under the same task contract. Use executable task metrics where possible. Report quality by task and source, safety failures, truncation, latency, throughput, and peak memory.

Do not use training loss as the main result. Sequence-level distillation aims to transfer behavior on held-out prompts. Add DPO only after SFT if a measured failure has reliable preference pairs. The current code intentionally stops at the SFT baseline.

## Reproducibility limits

Provider models can change behind aliases. Pin a dated snapshot where available and preserve provider request IDs. Hosted generation may remain nondeterministic with a fixed seed. The training manifest records available runtime information and data/template identity; it cannot recover missing teacher deployment history, a model alias's former weights, or undeclared source groups. Archive the full prompt inventory, review decisions, frozen evaluation suite, and teacher deployment configuration with the experiment. Local filesystem model loads may have no resolved Hub revision. Use immutable Hub commits for paper runs.
