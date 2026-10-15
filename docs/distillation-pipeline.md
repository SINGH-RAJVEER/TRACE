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

The `eval` extra installs datasets and huggingface-hub. The `train` extra installs PyTorch, Transformers, datasets, Accelerate, PEFT, TRL, and, on Linux, bitsandbytes and flash-linear-attention. These extras stay optional because the API and pipeline tests do not need the large training stack. Select the extra on subsequent `uv run` commands as shown above, since uv synchronizes the environment for each invocation.

QLoRA also requires a compatible NVIDIA GPU and a working host driver. The shell provides the compiler and library paths; the CUDA runtime dependencies come from the locked training wheels. The vLLM teacher described below runs as a separate upstream service with its own environment, either on this machine or another host.

## Adopted model pairing

- Student: `Qwen/Qwen3.5-2B-Base` at revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` (Apache 2.0, verified 2026-10-15).
- Teacher: `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (Apache 2.0, verified 2026-10-15), run in non-thinking mode.

On a 20 GB GPU, the FP8 teacher should leave room for many concurrent requests, so the whole run can use one such GPU, with collection and training taking turns. These fits are estimates until a quick-corpus run measures them.

Both use the Qwen3.5 chat format and tokenizer. Their pinned tokenizer configurations still differ: the teacher's default EOS token is `<|im_end|>` and the base student's is `<|endoftext|>`, so training explicitly sets `<|im_end|>` and passes `enable_thinking=false`.

Qwen3.5 checkpoints are multimodal. Each holds a vision encoder and a multi-token-prediction (MTP) head next to the text model. vLLM serves the teacher with `--language-model-only`, which skips the vision encoder, and `AutoModelForCausalLM` loads only the student's text model, about 1.88B parameters, so LoRA never touches the vision or MTP weights.

Three of every four layers use Gated DeltaNet linear attention, which keeps a fixed-size state per sequence instead of a growing KV cache. Only the full-attention layers, 8 of the teacher's 32, store KV entries, about 32 KB per token in BF16. The teacher's BF16 checkpoint is 19.3 GB including the vision encoder and MTP head. Its text weights alone are about 18 GB, so a GPU under 24 GB serves it with `--quantization fp8`, which brings the weights to roughly 11 GB (an estimate; the embedding and output layers stay in BF16).

Measure peak memory at the intended context length and batch size before choosing hardware. QLoRA reduces frozen-weight memory, but activations, optimizer state, and KV cache still consume memory. The 2B student and 9B teacher leave a useful capacity comparison, but improvement must be measured on held-out tasks.

Self-hosting the teacher with vLLM keeps provider terms out of the picture entirely:

```bash
vllm serve Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
	--host 127.0.0.1 --port 8001 --language-model-only \
	--default-chat-template-kwargs '{"enable_thinking": false}'
```

Point `TEACHER_UPSTREAM_BASE_URL` at `http://127.0.0.1:8001/v1`. After starting it, confirm responses contain no thinking traces before the main collection run, and keep transfer prompts within the training `max_length` (the student's native context is 262K tokens).

Qwen3.5 needs a vLLM release that supports the `Qwen3_5ForConditionalGeneration` architecture; pin `VLLM_SPEC` to such a release for a paper run. Do not add the `--reasoning-parser qwen3` flag from the model card. It moves `<think>` text out of the response content, where the smoke test and the collector's thought-marker check look for it.

`scripts/pipeline.py` does all of this for you, on two machines or one, and runs the non-thinking check automatically. See [run-scripts.md](run-scripts.md).

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

`distill corpus build` produces a prompt set in this format from pinned public datasets and project-authored prompts, assigns grouped splits, removes held-out benchmark overlap, and writes an answer key for automatic review. See [prompt-corpus.md](prompt-corpus.md).

## Collection and preparation

Run the teacher API first. Then collect responses:

```bash
uv run distill collect \
	--input data/prompts/all-splits.jsonl \
	--output data/teacher/ledger.jsonl \
	--teacher-url http://127.0.0.1:8000 \
	--terms-review-id "$TEACHER_TERMS_REVIEW_ID" \
	--concurrency 4 \
	--max-attempts 3 \
	--no-thinking
```

Collection is resumable by `example_id`. It rejects duplicate input IDs, retries teacher failures, and appends each successful response from its worker, so a later failure does not discard completed results. The ledger records teacher identity, the actual upstream endpoint, request parameters, provider request ID, token usage, source rights, approval reference, finish reason, and basic checks. A sibling `ledger.attempts.jsonl` file records every successful and failed call, including prompts whose retries all failed. Test prompts are skipped unless `--include-test` is set for a controlled teacher evaluation.

Resume verifies the prompt, split, task, source metadata, generation parameters, and approval reference for IDs already in the ledger. A mismatch requires a new ledger. The collector cannot detect a provider changing the underlying model behind the same endpoint; pin the served model and keep its deployment records.

Collection also exposes `--top-p` and `--top-k`; both are omitted from upstream requests unless supplied and are recorded for resume checks. `top_k` and the thinking switch need an upstream implementation that supports those extensions, such as vLLM. For a reviewed Qwen non-thinking sampling pilot, compare the greedy baseline with `--temperature 0.7 --top-p 0.8 --top-k 20 --no-thinking`, using separate ledgers and the same prompt/task contract. These follow Qwen3.5's documented non-thinking recommendations, except `presence_penalty=1.5`, which the collector does not expose. They are not a proven distillation optimum. Accept samples by task correctness and coverage before choosing a collection policy.

The basic checks reject blank content and nonterminal finish reasons. Passing responses remain `pending`; the collector does not call them training-ready. Store review decisions in a separate JSONL file so the raw ledger stays unchanged:

```json
{"example_id":"support-0001","status":"accepted","checks":["schema","grounding","human_review"],"reviewer":"reviewer-1"}
```

Task validators and human review must produce these decisions before a paper run. For example, compile generated code, validate structured output against a schema, check citations against supplied text, and audit every task slice.

`distill review` produces these decisions from the answer key that `distill corpus build` writes:

```bash
uv run --extra train distill review \
	--ledger data/teacher/ledger.jsonl \
	--references data/prompts/references.jsonl \
	--output data/teacher/ledger.reviews.jsonl \
	--config configs/train.distill.toml \
	--overrides data/reviews/human.jsonl
```

It writes one decision for every ledger record, because preparation skips records without one. It compares numeric answers after `####`, multiple-choice letters, exact answers, and JSON objects; parses Python code, and runs its tests in Docker when `--code-image` is set; and checks word limits, bullet counts, and required or forbidden phrases. Prompts without a key are accepted unless the collector rejected them. `--config` rejects records longer than the training `max_length` using the student tokenizer, so preflight does not fail later. `--overrides` decisions replace the automatic ones. The `.summary.json` beside the output counts decisions by status, reference kind, rejection reason, and task. Automatic checks confirm format and final answers, not reasoning quality; sample the accepted open-ended responses by hand before a paper run.

Project accepted training rows into TRL's conversational prompt-completion schema:

```bash
uv run distill prepare \
	--input data/teacher/ledger.jsonl \
	--reviews data/reviews/reviews.jsonl \
	--prompts data/prompts/all-splits.jsonl \
	--output data/processed/train.jsonl \
	--split train
```

Preparation keeps reviewed and accepted rows from the requested split. Explicit review rejection always wins over `--accept-pending`. It drops duplicate prompts and source groups found in another split, and refuses to prepare the frozen test split. Pass the full original prompt inventory with `--prompts`, including validation and test rows that have no teacher response, to close the ledger-only contamination gap. The supplied launchers pass their original prompt inventory automatically. Without `--prompts`, checks cover only rows present in that ledger. Separate collection files require the same full inventory.

A sibling `train.manifest.json` records input and output hashes, skip-reason counts, accepted task/source counts, and whether pending rows were allowed. Raw ledger files and prepared datasets remain private artifacts. Review near-duplicates and task correctness separately; the automated admission checks do not detect semantic overlap.

## Student training

Install the optional training stack:

```bash
uv sync --all-packages --extra train
```

Review `configs/train.example.toml`, which uses the pinned Qwen3.5-2B-Base checkpoint. Confirm that the base model license allows fine-tuning and the intended distribution. It trains without a validation split, so use it to check the setup on the `quick` corpus; `configs/train.distill.toml` is the configuration for a measured run.

For a full run, use `configs/train.distill.toml` with the standard corpus. It raises the LoRA rank from 16 to 128, which trains about 135M adapter parameters instead of 17M (1,051,200 per rank across Qwen3.5-2B's 24 text layers, 18 linear-attention and 6 full-attention), raises `max_length` to 3,072, enables the validation split, trains for 2 epochs with batches of 4 and accumulation of 8, and evaluates every 500 steps so the best checkpoint is kept. The adapted text model stays near 2.0B parameters against the 9B teacher.

For an experiment, include reviewed validation prompts in the original inventory, prepare them with `--split validation --prompts data/prompts/all-splits.jsonl`, and enable `validation_dataset_path` in the configuration. `scripts/pipeline.py` prepares that path automatically when configured. Training rejects empty datasets and exact train/validation prompt overlap, including datasets prepared outside this collector. The example config leaves validation optional; a research run needs validation and a separately frozen test set.

Training input must contain only conversational `prompt` and `completion` columns, with one assistant completion. Unknown columns are rejected. This prevents externally supplied `input_ids`, `labels`, or per-row template overrides from bypassing the tokenizer and completion-loss checks.

Start training:

```bash
uv run --extra train distill train --config configs/train.example.toml
```

The trainer:

- Loads conversational JSONL with `datasets`.
- Uses completion-only loss through TRL `SFTTrainer`.
- Passes the same template arguments to the prompt, full conversation, and TRL tokenization. Empty Qwen thought wrappers are part of the prefilling prompt and are excluded from the loss.
- Applies LoRA to all linear layers of the text model, including the linear-attention projections. The vision encoder and MTP head are not loaded.
- Uses NF4, double quantization, and BF16 compute when `qlora = true`.
- Seeds before model and adapter construction. Uses explicit cross-entropy, disables packing, and sets non-reentrant gradient checkpointing.
- Evaluates and saves every epoch when validation is supplied, loads the checkpoint with lowest validation loss, and evaluates the exported adapter at the end. `eval_strategy="steps"` is available; best-checkpoint selection requires `save_steps` to be a multiple of `eval_steps`.
- Saves the adapter, tokenizer, template hash, resolved base revision when available, both dataset hashes, preparation manifests, resolved training arguments, the checkpoint it resumed from, checkpoint choice, final validation metrics, package versions, uv lock hash, JJ revision, and available GPU/CUDA/driver metadata.

The Qwen example sets `eos_token = "<|im_end|>"` to match its chat template rather than the pretrained tokenizer's `<|endoftext|>` token. The loader explicitly aligns the model and generation configuration with the tokenizer's EOS and padding IDs before constructing the trainer, keeping training artifacts consistent with the chat contract.

The updated lock uses TRL 1.15.0, Transformers 5.19.0, PEFT 0.21.2, Accelerate 1.15.0, PyTorch 2.14.1, and bitsandbytes 0.50.2. TRL 1.15's `loss_type="nll"` uses a fused LM head to compute per-token log probabilities without constructing full sequence-by-vocabulary logits. It preserves ordinary cross-entropy while reducing one source of peak memory. `chunked_nll` is now a deprecated alias and is absent from this project's config. PEFT's `target_modules="all-linear"` excludes the output head; explicit output-head LoRA adapters are incompatible with this fused path for both NLL and DFT. No end-to-end speed or memory improvement has been measured here. The fused scoring kernel requires supported GPU/Triton execution; CPU checks and trainer construction do not prove that runtime path.

flash-linear-attention 0.5 provides Triton kernels for the Gated DeltaNet layers, which Transformers uses when the package is installed. Without it, Transformers falls back to a slower PyTorch implementation. `causal-conv1d`, which speeds up the short convolution in the same layers, is not installed because it builds with `nvcc`; Transformers runs that step in PyTorch.

`attn_implementation="sdpa"` uses PyTorch's native attention backend. You can select `flash_attention_2` on compatible BF16/FP16 training hardware with the required backend installed. `activation_offloading` exchanges GPU activation memory for CPU transfers; see GPU memory below for when it turns on. Packing stays disabled to preserve the checked example boundaries. `loss_type="dft"` enables TRL's Dynamic Fine-Tuning objective for a separate ablation; its probability-weighted validation loss is not ordinary NLL and must not be compared directly with NLL runs or converted to perplexity. Compare generated-answer quality on the same frozen suite before adopting it.

Liger, black-box on-policy training, and data-selection methods need separate compatibility and quality experiments.

`selective_gradient_checkpointing=true` enables TRL 1.15's selective activation checkpointing when `gradient_checkpointing=true`. It saves attention outputs instead of recomputing them during backward, exchanging additional activation memory for less recomputation. The default remains off until you measure the tradeoff on the training machine. The current recipe uses non-reentrant checkpointing and does not configure DeepSpeed ZeRO-3.

Set `qlora = false` for BF16 or FP16 LoRA when memory permits. Change `precision` to match the GPU. Four-bit training requires CUDA and bitsandbytes. CPU execution is suitable for API and pipeline tests, not the example QLoRA run.

If the base tokenizer has no chat template, set `chat_template_path` to a reviewed local Jinja template. Treat that template as part of the trained artifact. Before loading model weights, the trainer tokenizes every record and checks that prompt tokens are an exact prefix of the full sequence, that supervised completion tokens include the EOS token, and that no sequence exceeds `max_length`. Filter unsuitable records or change the reviewed template/limit rather than silently truncating teacher answers. The pinned Transformers API needs `return_dict=false` for these token-list checks. The configuration keeps `warmup_ratio`, which the locked Transformers version receives as fractional `warmup_steps`.

### Checkpoints and resuming

Training writes `checkpoint-N` directories into `output_dir` on the save schedule and always saves the final step. It keeps the newest `save_total_limit` checkpoints, and never removes the best one while validation selects it. Each holds the adapter, optimizer, scheduler, and RNG state. `distill train` resumes on its own: if `output_dir` already holds a complete checkpoint (one with `trainer_state.json`, which the trainer writes last), it continues from the newest one, including the position in the data order. Rerunning a finished run resumes at its last step, so it only repeats the final evaluation and export.

The first run writes `training-run.json` into `output_dir`, holding the training config, both dataset hashes, and the chat template hash. A resume must match it, so a changed learning rate, dataset, or schedule is refused with a message to use a new `output_dir` or remove the `checkpoint-*` directories. Nothing is deleted automatically. Settings that only change memory use or logging may differ on resume: `activation_offloading`, `gradient_checkpointing`, `selective_gradient_checkpointing`, `attn_implementation`, `logging_steps`, `save_total_limit`, and the paths themselves. A collection rerun that recovers previously failed prompts changes the prepared dataset, so resume is refused after it.

The first Ctrl+C or SIGTERM lets the current optimizer step finish, saves a checkpoint, and exits with status 130 without writing the adapter or `training-manifest.json`, which belong only to finished runs. Rerun the same command to continue. A second signal stops at once, losing work since the last saved checkpoint. A crash, power loss, or `kill -9` loses the work since the last scheduled save, so choose `save_steps` with that in mind; with validation and `eval_strategy="epoch"`, saves happen only at epoch ends. On the pinned versions, an interrupted and resumed CPU FP32 run of a tiny fixture produced the same adapter weights as an uninterrupted one. That check replaced TRL's GPU loss kernel with plain PyTorch; GPU kernels in BF16 are not bit-deterministic, so expect close rather than identical weights there.

### GPU memory

PyTorch does not spill GPU memory into system memory on its own. The trainer uses three narrower mechanisms:

- `optimizer = "auto"` selects bitsandbytes' paged AdamW (`paged_adamw_32bit`) when `qlora = true` on native Linux, and the Transformers default, `adamw_torch_fused`, otherwise. Paged state lives in CUDA unified memory, which the driver pages to system memory when the GPU runs short instead of failing. [NVIDIA's WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) states that WSL 2 lacks full managed-memory support and warns of reduced performance and high system memory usage, so `auto` detects WSL 2 and uses `adamw_torch_fused` there. It covers only optimizer state, and bitsandbytes pages only parameters with at least 100,000 elements: the rank-128 adapter matrices of `train.distill.toml` qualify, while most rank-16 matrices stay in ordinary GPU memory. QLoRA runs before this change used `adamw_torch_fused`; compare them with that in mind (the manifest's `training_args` records the optimizer). `adamw_torch`, `adamw_torch_fused`, `paged_adamw_32bit`, and `paged_adamw_8bit` can be set explicitly; paged optimizers require CUDA.
- `activation_offloading = "auto"` starts without offloading. On a CUDA out-of-memory error, it frees the failed attempt, resumes from the last checkpoint (or the first step) with TRL's activation offloading, which keeps saved activations in system memory, and records that in `training-run.json` so later resumes start with it. `true` offloads from the start, and `false` never does; with `false`, an out-of-memory error stops with the last checkpoint and the settings to change. If memory runs out even with offloading, lower `per_device_batch_size` or `max_length` and train into a new `output_dir`. Only `torch.OutOfMemoryError` triggers the retry; allocation failures that cuBLAS or bitsandbytes raise as other errors stop training, since a failed CUDA context is not safe to reuse in the same process. Offloading also does not reduce evaluation memory. bitsandbytes never frees paged buffers, so the failed attempt's optimizer state stays allocated, unused, until the process exits.
- `distill train` sets `PYTORCH_ALLOC_CONF=expandable_segments:True` unless `PYTORCH_ALLOC_CONF` or `PYTORCH_CUDA_ALLOC_CONF` is already set. Growable allocator segments reduce out-of-memory errors caused by fragmentation; they do not use system memory.

The paged optimizer and real activation offloading have not been measured on the training GPU. Out-of-memory recovery was checked on CPU with a simulated error.

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

An opt-in test constructs TRL and PEFT against a tiny random local Qwen3.5 checkpoint saved in the same multimodal layout as the real ones, with an extra MTP tensor. It checks that only the text model loads, with the saved weights and a tied output head, that LoRA covers the linear-attention and full-attention projections but not the convolution, plus actual completion labels, empty-thought-prefix masking, fused-loss configuration, and validation/warmup settings. The test replaces the optimizer-running entry point with an inspection function, so it never trains. Its QLoRA variant skips when CUDA is unavailable:

```bash
DISTILL_TEST_TRAINING_STACK=1 uv run --extra train pytest tests/test_training_stack.py
```

Keep tests in a separate uv environment if another process is synchronizing this checkout's environment:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/distillation-checks uv run --extra train pytest
```

## Evaluation

Freeze evaluation prompts before collection. The benchmark tools compare four slots on the identical frozen suite: teacher and base student before training, then teacher and distilled student after training. The initial teacher stays the fixed reference for gap closure; the second evaluation measures teacher drift. Supported public tasks are GSM8K, ARC-Challenge, MMLU, the authors' binary TruthfulQA task, HumanEval, and sanitized MBPP. Coding scores use executable pass@1 in isolated containers. General tasks use numeric exact match or generated-choice accuracy.

Run `uv run --extra eval distill benchmark freeze --output data/benchmarks/suite.json`, then use `configs/eval.example.toml` with `distill benchmark run --stage before` and later `--stage after`. These commands evaluate existing models and do not perform distillation. The read-only dashboard displays quality scores with Wilson intervals, paired gains and regressions, teacher drift, student–teacher agreement, subject slices, latency, usage, and request/test outcomes. See [benchmarks.md](benchmarks.md) for the complete metric definitions, denominator rules, dataset provenance, coding limits, and dashboard commands. Peak GPU memory and domain-specific safety failures require separate instrumentation; this setup does not record them.

Do not use training loss as the main result. Sequence-level distillation aims to transfer behavior on held-out prompts. Add DPO only after SFT if a measured failure has reliable preference pairs. The current code intentionally stops at the SFT baseline.

## Reproducibility limits

Provider models can change behind aliases. Pin a dated snapshot where available and preserve provider request IDs. Hosted generation may remain nondeterministic with a fixed seed. The training manifest records available runtime information and data/template identity; it cannot recover missing teacher deployment history, a model alias's former weights, or undeclared source groups. Archive the full prompt inventory, review decisions, frozen evaluation suite, and teacher deployment configuration with the experiment. Local filesystem model loads may have no resolved Hub revision. Use immutable Hub commits for paper runs.
