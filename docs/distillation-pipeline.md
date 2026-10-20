# Distillation pipeline

## Scope

This repository implements the paper's task-specific teacher-to-student extraction experiment. The teacher is a hosted chat model available through an OpenAI-compatible `POST /chat/completions` endpoint. The student is a Hugging Face causal language model trained with LoRA on a BF16 base, or optionally with QLoRA.

The current method is sequence-level distillation. The collector saves the teacher's generated text, and TRL trains the student with cross-entropy on completion tokens. It does not use teacher logits.

The protection wrapper discussed in the paper lives in `packages/harness`; see [harness.md](harness.md).

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

Training on a GPU, and QLoRA in particular, requires a compatible NVIDIA GPU and a working host driver. The shell provides the compiler and library paths; the CUDA runtime dependencies come from the locked training wheels. The vLLM teacher described below runs as a separate upstream service with its own environment, either on this machine or another host.

## Adopted model pairing

- Student: `Qwen/Qwen3.5-2B-Base` at revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` (Apache 2.0, verified 2026-10-15).
- Teacher: `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (Apache 2.0, verified 2026-10-15), run in non-thinking mode.

On a 20 GB GPU, the FP8 teacher should leave room for many concurrent requests, so the whole run can use one such GPU, with collection and training taking turns. These fits are estimates until a task pilot measures them.

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

vLLM's FlashInfer top-k/top-p sampler compiles a CUDA kernel with `nvcc` the first time it runs. On hosts that have only the NVIDIA driver and no CUDA toolkit (no `nvcc` on `PATH` or under `CUDA_HOME`), the launcher sets `VLLM_USE_FLASHINFER_SAMPLER=0` so vLLM falls back to its PyTorch sampler. Set the variable yourself to override this.

`scripts/pipeline.py` does all of this for you, on two machines or one, and runs the non-thinking check automatically. See [run-scripts.md](run-scripts.md).

## Authorization gate

Do not collect data until the hosted provider has explicitly authorized using its output to train the planned student. Output ownership does not necessarily grant model-training rights.

Record the approval, contract amendment, or provider permission in `TEACHER_TERMS_REVIEW_ID`. The collector separately requires `--terms-review-id` and writes it into every ledger record. This is an audit reference, not an enforcement mechanism.

The prompt source also requires a corpus name, source ID, and license. The task builder records pinned dataset provenance and grouped source identity.

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

## Task input and splits

`distill tasks build --task TASK` writes the attacker pool, gold completions, legitimate workload pools, development suite, final test suite, and a hashed manifest. See [paper-tasks.md](paper-tasks.md#splits) for group assignment, filtering, and dataset pins. Each collector row has a stable example ID, `split`, `task`, source dataset and group identity, license, and structured messages.

Only attacker-pool prompts enter attacker collection. Gold answers stay in separate files for reference training and evaluation. Frozen development and test suites do not consume attacker queries. Select the recipe on development data and reserve the test suite for final scoring.

## Collection and preparation

The task launcher runs these steps for each seed and supported budget. To run them manually:

```bash
uv run --extra eval distill tasks build --task ag_news
uv run distill tasks budget --task-dir data/tasks/ag_news-split-v2-s42 --seed 42 --budget 2000
uv run distill collect \
	--input data/tasks/ag_news-split-v2-s42/runs/s42-b2000/prompts.jsonl \
	--output data/tasks/ag_news-split-v2-s42/ledger.jsonl \
	--teacher-url http://127.0.0.1:8000 \
	--terms-review-id "$TEACHER_TERMS_REVIEW_ID" \
	--max-output-tokens 16 --no-thinking
uv run distill tasks budget --task-dir data/tasks/ag_news-split-v2-s42 --seed 42 --budget 2000 \
	--ledger data/tasks/ag_news-split-v2-s42/ledger.jsonl
uv run --extra train distill review \
	--ledger data/tasks/ag_news-split-v2-s42/runs/s42-b2000/ledger.jsonl \
	--output data/tasks/ag_news-split-v2-s42/runs/s42-b2000/reviews.jsonl \
	--config configs/train.task.toml
uv run distill prepare \
	--input data/tasks/ag_news-split-v2-s42/runs/s42-b2000/ledger.jsonl \
	--output data/tasks/ag_news-split-v2-s42/runs/s42-b2000/train.jsonl \
	--prompts data/tasks/ag_news-split-v2-s42/runs/s42-b2000/prompts.jsonl \
	--reviews data/tasks/ag_news-split-v2-s42/runs/s42-b2000/reviews.jsonl
```

Use 64 output tokens for SQuAD and 256 for GSM8K, identically in collection and evaluation. Pilot GSM8K truncation before main collection. Budget extraction refuses ledger rows collected at another task output limit.

Collection resumes by `example_id`, rejects duplicate input IDs, and verifies prompt, split, task, source metadata, parameters, and approval identity before appending. Each finished response persists immediately. A sibling attempts ledger records successful and failed calls. The raw ledger retains the actual served model, provider, endpoint, generation settings, request IDs, usage, finish reasons, and collection checks. Resume cannot detect a provider replacing weights behind an unchanged endpoint, so archive deployment identity separately.

### Recover a damaged collection ledger

`just start` and `just distill` validate every planned task ledger before starting the teacher. A `JSONLValidationError` containing NUL bytes, such as `data/tasks/gsm8k/ledger.jsonl:1933: invalid record`, means an existing ledger row cannot be parsed. The teacher can still be healthy. The bytes alone do not identify how the file was damaged.

Stop collection before repairing a ledger. Without `--apply`, this command previews recovery counts and changes no files:

```bash
uv run distill ledger repair --input data/tasks/gsm8k/ledger.jsonl
uv run distill ledger repair --input data/tasks/gsm8k/ledger.jsonl --apply
VLLM_USE_FLASHINFER_SAMPLER=0 just start --concurrency 8
```

`--apply` writes a byte-for-byte `ledger.jsonl.recovery-ID.bak`, quarantines invalid rows in a sibling `.rejected.bin`, and writes a `.report.json` with counts, hashes, original line numbers, and byte offsets. It then atomically replaces the ledger with all schema-valid rows in their original order, including valid rows after the damaged section. It strips NUL padding only outside an otherwise complete record. NUL bytes inside generated text or other JSON fields cause that row to be quarantined. Clean rows retain their original bytes, and a missing final newline is restored before collection can append another response. A clean ledger is left unchanged.

Duplicate example IDs stop recovery without choosing a response or replacing the ledger. Collection still validates the full resume contract after repair. Only missing IDs are queried again, and repairing GSM8K leaves the AG News and SQuAD ledgers unchanged. Existing attempt logs are retained. Damaged responses remain in the backup and quarantine; recovery does not invent replacements.

### Review and preparation

The collector rejects empty responses, nonterminal finish reasons, and thinking traces in non-thinking mode. Other responses remain pending until review. `distill review` accepts nonempty responses that fit the student training length and were not collector-rejected. It has no gold-answer input or correctness filter. Wrong and malformed teacher answers remain in attacker supervision. The review summary records status, rejection reasons, accepted task counts, and whether length was checked.

Preparation writes reviewed, accepted, deduplicated prompt-completion rows and a preparation manifest. It refuses the test split and removes prompts or source groups shared across splits. Pass the original prompt inventory through `--prompts`; grouping and held-out separation also depend on the task build. Explicit rejection wins over the low-level `--accept-pending` bypass, which is only for plumbing checks.

The undefended collector retries transport failures. A defended campaign must charge every attempt, filler request, retry, and challenge to the declared budgets and train only on responses actually released.

## Student training

`configs/train.task.toml` is the task recipe: rank-16 LoRA, alpha 32, dropout 0.05, 2,048-token training limit, effective batch size 16, and no victim-response validation split. The launcher writes one config per task, seed, budget, and student kind, selecting the corresponding training file and adapter directory.

Select learning rate and epoch count on the frozen gold-scored development suite, then freeze the recipe across defense conditions. The attacker and gold-label reference use the same prompts and training recipe. Establish positive undefended transfer on development data before final defense evaluation. A gold reference is a control, not a guaranteed empirical upper bound.

```bash
uv run --extra train distill train \
	--config data/tasks/ag_news-split-v2-s42/runs/s42-b2000/train.attacker.toml --preflight-only
uv run --extra train distill train \
	--config data/tasks/ag_news-split-v2-s42/runs/s42-b2000/train.attacker.toml
```

Training accepts only conversational `prompt` and `completion` columns with one assistant completion. It rejects empty datasets and unsupported per-row tokenization overrides. The shared trainer retains optional validation support, but the paper task recipe selects settings through development task scores rather than extra victim-response queries.

The trainer:

- Loads conversational JSONL with `datasets`.
- Uses completion-only loss through TRL `SFTTrainer`.
- Passes the same template arguments to the prompt, full conversation, and TRL tokenization. Empty Qwen thought wrappers are part of the prefilling prompt and are excluded from the loss.
- Applies LoRA to all linear layers of the text model, including the linear-attention projections. The vision encoder and MTP head are not loaded.
- Loads the base in `precision` by default. With `qlora = true`, it uses NF4, double quantization, and BF16 compute instead.
- Seeds before model and adapter construction. Uses explicit cross-entropy, disables packing, and sets non-reentrant gradient checkpointing.
- Evaluates and saves every epoch when validation is supplied, loads the checkpoint with lowest validation loss, and evaluates the exported adapter at the end. `eval_strategy="steps"` is available; best-checkpoint selection requires `save_steps` to be a multiple of `eval_steps`.
- Saves the adapter, tokenizer, template hash, resolved base revision when available, both dataset hashes, preparation manifests, resolved training arguments, the checkpoint it resumed from, checkpoint choice, final validation metrics, package versions, uv lock hash, JJ revision, and available GPU/CUDA/driver metadata.

The Qwen example sets `eos_token = "<|im_end|>"` to match its chat template rather than the pretrained tokenizer's `<|endoftext|>` token. The loader explicitly aligns the model and generation configuration with the tokenizer's EOS and padding IDs before constructing the trainer, keeping training artifacts consistent with the chat contract.

The updated lock uses TRL 1.15.0, Transformers 5.19.0, PEFT 0.21.2, Accelerate 1.15.0, PyTorch 2.14.1, and bitsandbytes 0.50.2. TRL 1.15's `loss_type="nll"` uses a fused LM head to compute per-token log probabilities without constructing full sequence-by-vocabulary logits. It preserves ordinary cross-entropy while reducing one source of peak memory. `chunked_nll` is now a deprecated alias and is absent from this project's config. PEFT's `target_modules="all-linear"` excludes the output head; explicit output-head LoRA adapters are incompatible with this fused path for both NLL and DFT. No end-to-end speed or memory improvement has been measured here. The fused scoring kernel requires supported GPU/Triton execution; CPU checks and trainer construction do not prove that runtime path.

flash-linear-attention 0.5 provides Triton kernels for the Gated DeltaNet layers, which Transformers uses when the package is installed. Without it, Transformers falls back to a slower PyTorch implementation. `causal-conv1d`, which speeds up the short convolution in the same layers, is not installed because it builds with `nvcc`; Transformers runs that step in PyTorch.

`attn_implementation="sdpa"` uses PyTorch's native attention backend. You can select `flash_attention_2` on compatible BF16/FP16 training hardware with the required backend installed. `activation_offloading` exchanges GPU activation memory for CPU transfers; see GPU memory below for when it turns on. Packing stays disabled to preserve the checked example boundaries. `loss_type="dft"` enables TRL's Dynamic Fine-Tuning objective for a separate ablation; its probability-weighted validation loss is not ordinary NLL and must not be compared directly with NLL runs or converted to perplexity. Compare generated-answer quality on the same frozen suite before adopting it.

Liger, black-box on-policy training, and data-selection methods need separate compatibility and quality experiments.

`selective_gradient_checkpointing=true` enables TRL 1.15's selective activation checkpointing when `gradient_checkpointing=true`. It saves attention outputs instead of recomputing them during backward, exchanging additional activation memory for less recomputation. The default remains off until you measure the tradeoff on the training machine. The current recipe uses non-reentrant checkpointing and does not configure DeepSpeed ZeRO-3.

The task config sets `qlora = false`. A 4-bit base costs small students more accuracy than it saves memory: the 2B student's text weights take about 3.8 GB in BF16, while activations dominate peak memory. Training on a quantized base also lets the adapter partly learn to correct quantization error, which mixes that recovery into the measured distillation gain. Set `qlora = true` only when the BF16 base does not fit; four-bit training requires CUDA and bitsandbytes. Change `precision` to match the GPU. CPU execution is suitable for API and pipeline tests, not the example training runs.

Evaluate an adapter on the same base it was trained on. Set `load_in_4bit` in both student slots of the evaluation config to the adapter's `qlora` value. The evaluator loads four-bit models with the same NF4 and double-quantization settings as training, and rejects an adapter whose `qlora` setting differs from `load_in_4bit`.

If the base tokenizer has no chat template, set `chat_template_path` to a reviewed local Jinja template. Treat that template as part of the trained artifact. Before loading model weights, the trainer tokenizes every record and checks that prompt tokens are an exact prefix of the full sequence, that supervised completion tokens include the EOS token, and that no sequence exceeds `max_length`. Filter unsuitable records or change the reviewed template/limit rather than silently truncating teacher answers. The pinned Transformers API needs `return_dict=false` for these token-list checks. The configuration keeps `warmup_ratio`, which the locked Transformers version receives as fractional `warmup_steps`.

### Checkpoints and resuming

Training writes `checkpoint-N` directories into `output_dir` on the save schedule and always saves the final step. It keeps the newest `save_total_limit` checkpoints, and never removes the best one while validation selects it. Each holds the adapter, optimizer, scheduler, and RNG state. `distill train` resumes on its own: if `output_dir` already holds a complete checkpoint (one with `trainer_state.json`, which the trainer writes last), it continues from the newest one, including the position in the data order. Rerunning a finished run resumes at its last step, so it only repeats the final evaluation and export.

The first run writes `training-run.json` into `output_dir`, holding the training config, both dataset hashes, and the chat template hash. A resume must match it, so a changed learning rate, dataset, or schedule is refused with a message to use a new `output_dir` or remove the `checkpoint-*` directories. Nothing is deleted automatically. Settings that only change memory use or logging may differ on resume: `activation_offloading`, `gradient_checkpointing`, `selective_gradient_checkpointing`, `attn_implementation`, `logging_steps`, `save_total_limit`, and the paths themselves. A collection rerun that recovers previously failed prompts changes the prepared dataset, so resume is refused after it.

The first Ctrl+C or SIGTERM lets the current optimizer step finish, saves a checkpoint, and exits with status 130 without writing the adapter or `training-manifest.json`, which belong only to finished runs. Rerun the same command to continue. A second signal stops at once, losing work since the last saved checkpoint. A crash, power loss, or `kill -9` loses the work since the last scheduled save, so choose `save_steps` with that in mind; with validation and `eval_strategy="epoch"`, saves happen only at epoch ends. On the pinned versions, an interrupted and resumed CPU FP32 run of a tiny fixture produced the same adapter weights as an uninterrupted one. That check replaced TRL's GPU loss kernel with plain PyTorch; GPU kernels in BF16 are not bit-deterministic, so expect close rather than identical weights there.

### GPU memory

PyTorch does not spill GPU memory into system memory on its own. The trainer uses three narrower mechanisms:

- `optimizer = "auto"` selects bitsandbytes' paged AdamW (`paged_adamw_32bit`) when `qlora = true` on native Linux, and the Transformers default, `adamw_torch_fused`, otherwise. Paged state lives in CUDA unified memory, which the driver pages to system memory when the GPU runs short instead of failing. [NVIDIA's WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) states that WSL 2 lacks full managed-memory support and warns of reduced performance and high system memory usage, so `auto` detects WSL 2 and uses `adamw_torch_fused` there. It covers only optimizer state, and bitsandbytes pages only parameters with at least 100,000 elements: most rank-16 task adapter matrices stay in ordinary GPU memory. QLoRA runs before this change used `adamw_torch_fused`; compare them with that in mind (the manifest's `training_args` records the optimizer). `adamw_torch`, `adamw_torch_fused`, `paged_adamw_32bit`, and `paged_adamw_8bit` can be set explicitly; paged optimizers require CUDA.
- `activation_offloading = "auto"` starts without offloading. On a CUDA out-of-memory error, it frees the failed attempt, resumes from the last checkpoint (or the first step) with TRL's activation offloading, which keeps saved activations in system memory, and records that in `training-run.json` so later resumes start with it. `true` offloads from the start, and `false` never does; with `false`, an out-of-memory error stops with the last checkpoint and the settings to change. If memory runs out even with offloading, lower `per_device_batch_size` or `max_length` and train into a new `output_dir`. Only `torch.OutOfMemoryError` triggers the retry; allocation failures that cuBLAS or bitsandbytes raise as other errors stop training, since a failed CUDA context is not safe to reuse in the same process. Offloading also does not reduce evaluation memory. bitsandbytes never frees paged buffers, so the failed attempt's optimizer state stays allocated, unused, until the process exits.
- `distill train` sets `PYTORCH_ALLOC_CONF=expandable_segments:True` unless `PYTORCH_ALLOC_CONF` or `PYTORCH_CUDA_ALLOC_CONF` is already set. Growable allocator segments reduce out-of-memory errors caused by fragmentation; they do not use system memory.

The paged optimizer and real activation offloading have not been measured on the training GPU. Out-of-memory recovery was checked on CPU with a simulated error.

## Checks before running on training hardware

The default tests make no external provider calls and perform no optimization steps. The HTTP collection integration test replaces only the external provider transport, then runs the real OpenAI adapter, teacher API, gateway, collector, review, and preparation. It checks retry history, provider request IDs, actual served-model identity, secret-free artifacts, held-out prompt exclusion, thought/truncation rejection, and append-only resume. Separate checks change each part of the collection contract and verify rejection before any request or ledger write. Event-controlled workers verify the concurrency bound and that a completed response reaches disk while another request is still blocked; those checks do not depend on sleeping long enough for a worker to finish.

```bash
uv run pytest
uv run ruff check .
uv run mypy apps packages tests
```

After preparing your actual data, check the configured tokenizer and complete tokenized targets without loading model weights or invoking a trainer:

```bash
uv run --extra train distill train --config configs/train.task.toml --preflight-only
```

Preflight prints data hashes, record and supervised-token counts, maximum sequence length, template identity, EOS, loss/attention settings, and whether validation is enabled. It may download tokenizer files. It makes no teacher calls, loads no model weights, and creates no adapter. Passing preflight checks data and serialization; it does not prove GPU capacity, numerical stability, or task quality.

The token report also includes minimum, median, and nearest-rank 95th-percentile lengths, total tokens, and the supervised fraction. Use these values to compare data budgets and choose an appropriate length limit. `just preflight-train` runs the same safe command. `just test-training-setup` runs only the opt-in construction checks; `just test-training` also runs optimization and checkpoint-resume checks.

The opt-in tests build tiny random Qwen3.5 checkpoints locally, with no model downloads. A construction check uses the real multimodal checkpoint layout with an extra MTP tensor and verifies that only the saved text weights load, the output head stays tied, LoRA covers both attention types, and actual completion labels mask the prompt and empty thinking prefix. Its QLoRA variant requires CUDA; opting in without the required Python dependencies fails rather than silently skipping them.

The CPU runtime check trains for four optimizer steps, verifies nonzero adapter updates, interrupts a separate run with SIGTERM, and resumes its saved optimizer and scheduler for exactly the two remaining steps. The final adapter must match the uninterrupted run tensor-for-tensor. It reloads the saved adapter through PEFT, runs a forward pass, validates the evaluation manifest, and verifies that changed training data is refused without overwriting the adapter. The preflight CLI check deletes the fixture's model weights, then runs real tokenizer/data validation, including overlong and fully masked targets, and verifies that no adapter is created.

The CPU runtime fixture uses full-attention layers and substitutes a differentiable PyTorch reference for TRL's GPU-only scoring kernel. It exercises real LoRA, autograd, optimization, serialization, and resume, but does not validate CUDA/Triton kernels, hybrid-attention execution, BF16 numerics, or GPU memory capacity. The separate construction check retains the hybrid model layout.

```bash
just test-training
```

Keep tests in a separate uv environment if another process is synchronizing this checkout's environment:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/distillation-checks just test-training
```

## Evaluation

Evaluate each task separately with `configs/eval.TASK.toml`. Point `student_after.adapter_path` at the exact `attacker` or `gold` adapter and use distinct experiment names and output directories for different seeds, budgets, conditions, and student kinds. The default configs point at the seed-42, budget-2,000 attacker adapter.

Primary capability metrics are AG News label accuracy, SQuAD token F1, and GSM8K numeric exact match. Keep SQuAD exact match secondary. Run the victim and unmodified student before training and reevaluate the victim plus adapted student afterward. Victim reevaluation measures drift; the teacher is not trained. These four slots do not by themselves aggregate a paired defended-versus-undefended study or implement `R_D`. See [benchmarks.md](benchmarks.md).

## Reproducibility limits

Freeze dataset groups, prompt serialization, model and tokenizer revisions, training settings, query schedules, and the experiment matrix before final testing. Archive task manifests, ledgers, attempts, reviews, generated configs, adapters, training manifests, calibration, and campaign events. Record quantization and actual provider identity as part of the deployed victim.

Training manifests and passing preflight establish software and data checks, not measured capability. Provider aliases can change, generation can vary despite seeds, and exact-overlap checks do not detect paraphrases or pretraining contamination. Report actual completed experiments and their validation boundaries.
