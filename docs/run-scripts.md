# Run scripts

The `scripts/` directory runs the whole pipeline from one command per machine. Each script ships as a bash version for Linux and macOS and a PowerShell 7.2+ version for Windows, Linux, and macOS. Both versions take the same options and behave the same way.

| Script | Scenario | What it does |
| --- | --- | --- |
| `teacher-host` | Two machines, teacher side | Starts vLLM with Qwen3.5-9B on loopback and the teacher API on the network, then runs until interrupted |
| `student-train` | Two machines, student side | Checks the remote teacher API, collects responses, prepares the dataset, and trains the Qwen3.5-2B adapter |
| `distill-local` | One machine | Starts vLLM and the teacher API on loopback, collects, stops both to free GPU memory, then prepares and trains |

Run them from any directory. Relative paths in options resolve against the repository root.

The same scripts are available as `just` recipes. Bare names run the bash version and the `-ps` suffix runs the PowerShell version. Options after the recipe name go straight to the script:

```bash
just teacher-host
just student-train --teacher-url http://10.0.0.5:8000 --reviews data/reviews/reviews.jsonl
just distill-local --reviews data/reviews/reviews.jsonl
```

```powershell
just teacher-host-ps -UpstreamUrl http://127.0.0.1:8001/v1
just student-train-ps -TeacherUrl http://10.0.0.5:8000 -Reviews data/reviews/reviews.jsonl
just distill-local-ps -Reviews data/reviews/reviews.jsonl
```

The `-ps` recipes call `pwsh`, so PowerShell 7.2+ must be on `PATH`. On Windows, `just` runs recipes through `pwsh` instead of bash. The bash recipes there still need `bash` on `PATH`, for example from Git for Windows. `just` passes options through unquoted, so option values cannot contain spaces.

## Runtime selection

Each script picks how it installs and runs Python before doing anything else:

1. **devenv** (bash only). On NixOS with `devenv` installed, the script re-runs itself inside `devenv shell`, which provides Python 3.14, Python 3.12 for vLLM, uv, a compiler, and the library paths that binary wheels and the NVIDIA driver need. If devenv is missing, the script says so and continues with the next option.
2. **uv**. With `uv` on `PATH`, the script runs `uv sync --all-packages --inexact` (adding `--extra train` for training) and uses `.venv`. uv downloads Python 3.14 if needed, except inside devenv, which provides it.
3. **pip**. Without uv, the script creates `.venv-pip` with a Python 3.14 interpreter (`python3.14`, `python3`, or `py -3.14` on Windows) and installs the workspace packages in editable mode. pip does not read `uv.lock`, so dependency versions can differ from the locked set.

Before installing anything, the script lists every missing tool it can detect, then stops if any of them is required:

- Required: uv for the uv runtime. For the pip runtime: Python 3.14 with `venv` and `ensurepip` (the `python3.14-venv` package on Debian and Ubuntu), plus a Python 3.10 to 3.13 interpreter for vLLM when vLLM is not already installed. `curl` for the bash scripts.
- Warnings when a GPU is needed: `nvidia-smi` (the NVIDIA driver), and a C compiler (`cc` or `gcc`) with the C library headers for the Triton kernels that vLLM and training compile at runtime. On Debian and Ubuntu, including WSL, `build-essential` provides both; a bare `gcc` without `libc6-dev` makes vLLM fail at startup with `fatal error: stdlib.h: No such file or directory`.
- On Windows, the PowerShell scripts stop here if they would need to start vLLM (see Platform notes).

Set `DISTILL_RUNTIME=devenv|uv|pip` to force a runtime.

vLLM gets its own environment because its wheels target older Python releases than this workspace. If `vllm` is on `PATH`, or `VLLM_BIN` points at an executable, the script uses that. Otherwise it creates `.venv-vllm` from `VLLM_PYTHON` (default `3.12`) and installs `VLLM_SPEC` (default `vllm`). The first install downloads several gigabytes. Pin `VLLM_SPEC` to an exact version for a paper run.

## Teacher machine

Set an approval reference and an API token in `.env`, then start the host:

```bash
scripts/teacher-host.sh
```

```powershell
./scripts/teacher-host.ps1
```

The script binds the teacher API to `0.0.0.0` on `TEACHER_PORT` (default 8000) and keeps vLLM on `127.0.0.1:VLLM_PORT` (default 8001), so only the authenticated teacher API is reachable from the network. It refuses to bind beyond loopback without `TEACHER_API_TOKEN`. `.env`'s `TEACHER_HOST` does not apply here; pass `--host` (`-BindHost`) to change the address. The API has no TLS, so expose it only on a trusted network or behind a TLS gateway.

Startup waits for vLLM to load the model, then for `/healthz` to report `ready`, then sends one small generation through the API. That check fails if the token is wrong or if the response contains a `<think>` trace. vLLM starts with `--default-chat-template-kwargs '{"enable_thinking": false}'`, so Qwen3.5 answers in non-thinking mode without changing the API's request format, and with `--language-model-only`, so the checkpoint's vision encoder is never loaded. When ready, the script prints the student command with this machine's address and then watches both processes. Ctrl+C stops both.

| Option (bash / PowerShell) | Default | Meaning |
| --- | --- | --- |
| `--host` / `-BindHost` | `0.0.0.0` | Teacher API bind address |
| `--port` / `-Port` | `TEACHER_PORT` or 8000 | Teacher API port |
| `--upstream-url` / `-UpstreamUrl` | none | Use an existing OpenAI-compatible server instead of starting vLLM |

## Student machine

Use the same `TEACHER_API_TOKEN` as the teacher machine, then point the script at the teacher:

```bash
scripts/student-train.sh --teacher-url http://10.0.0.5:8000 --reviews data/reviews/reviews.jsonl
```

```powershell
./scripts/student-train.ps1 -TeacherUrl http://10.0.0.5:8000 -Reviews data/reviews/reviews.jsonl
```

The script installs the training stack, waits up to 60 seconds for the teacher to report `ready`, runs the same smoke test, and then runs `distill collect`, `distill review` (when `--references` is given), `distill prepare`, and `distill train`. Preparation writes to the `dataset_path` in the training config, so the two always match. Collection resumes by `example_id`, so rerunning after a failure only fetches missing prompts.

## Single machine

```bash
scripts/distill-local.sh --reviews data/reviews/reviews.jsonl
```

```powershell
./scripts/distill-local.ps1 -Reviews data/reviews/reviews.jsonl
```

Both servers bind to loopback and log to `artifacts/logs/` (`vllm.log`, `teacher-api.log`, and `.err.log` files for PowerShell). After collection, the script stops the teacher API and vLLM before training, because vLLM reserves most of the GPU for the teacher and leaves too little for training alongside it. The machine therefore needs enough GPU memory for the teacher during collection (see the model pairing in [distillation-pipeline.md](distillation-pipeline.md)). Lower `VLLM_MAX_MODEL_LEN` or add `--quantization fp8` to `VLLM_EXTRA_ARGS` if vLLM runs out of memory. `--upstream-url` (`-UpstreamUrl`) skips vLLM and uses an existing server instead.

## Full run with the prompt corpus

The example prompts only test the plumbing. For a real run, pass `--corpus` with a profile from [prompt-corpus.md](prompt-corpus.md):

```bash
scripts/distill-local.sh --corpus standard --concurrency 8
```

```powershell
./scripts/distill-local.ps1 -Corpus standard -Concurrency 8 -UpstreamUrl http://127.0.0.1:8001/v1
```

The prompt files are not in the repository. `--corpus` builds them into `data/prompts/` after installing the Python environment, downloading the pinned public datasets and the held-out benchmark questions, which takes a few minutes. A later run with the same profile reuses the existing files. Unless you pass them yourself, `--corpus` also sets `--prompts data/prompts/all-splits.jsonl`, `--references data/prompts/references.jsonl`, and `--config configs/train.distill.toml`. The build is deterministic and a prompt keeps the same ID and text in every profile, so moving from `quick` to `standard` reuses the responses already in the ledger. To build the corpus on its own, run `just corpus standard`.

With `--references`, the script runs `distill review` after collection. It checks every teacher response against the answer key, rejects responses that would exceed the training config's `max_length`, and writes `data/teacher/ledger.reviews.jsonl` plus a `.summary.json` with counts by status and rejection reason. Preparation then uses those decisions. A `--reviews` file passed at the same time holds human decisions that override the automatic ones. Set `REVIEW_CODE_IMAGE=python:3.12-slim` to run the Python tests in the same isolated Docker container the benchmarks use; without it, code is only parsed.

The same steps work on two machines: pass `--corpus` to `student-train`.

### GPUs with 20 to 24 GB

The teacher's BF16 text weights take about 18 GB, so a 20 GB card such as the RTX 4000 Ada must quantize them. Ada and newer GPUs support FP8, which brings the weights to roughly 11 GB; the embedding and output layers stay in BF16. Add this line to `.env`:

```dotenv
VLLM_EXTRA_ARGS="--quantization fp8"
```

The scripts already pass `--language-model-only`, so the vision encoder takes no GPU memory. Only one layer in four keeps a KV cache, about 32 KB per token in BF16, so the default 8,192-token context and `--concurrency 16` should fit in what remains. Keep the KV cache in BF16: the one run with `--kv-cache-dtype fp8` on an RTX 4000 Ada failed while FlashInfer compiled a kernel at startup, which needs `nvcc` from a CUDA toolkit. vLLM still downloads the BF16 checkpoint and quantizes it while loading, so the machine, or the WSL virtual machine, needs enough system memory to read it.

A GPU that also drives a display already has about 1 GB in use. vLLM refuses to start when free memory is below `VLLM_GPU_MEMORY_UTILIZATION` times the total; lower the value if the log reports that.

For training, the 2B student's 4-bit text model takes about 2 to 3 GB, because its 248K-token embedding stays unquantized, and the rank-128 adapter with its optimizer state about 2 GB, so batches of 4 at the config's 3,072-token limit should fit in 20 GB with room to spare. These figures are estimates, not measurements. If training runs out of memory, halve `per_device_batch_size` and double `gradient_accumulation_steps`.

## Pipeline options

`student-train` and `distill-local` share these options:

| Option (bash / PowerShell) | Default | Meaning |
| --- | --- | --- |
| `--terms-review-id` / `-TermsReviewId` | `TEACHER_TERMS_REVIEW_ID` | Approval reference written into every ledger record; required |
| `--corpus` / `-Corpus` | none | Build or reuse the `quick`, `standard`, or `full` prompt corpus and use its prompts, answer key, and `configs/train.distill.toml` |
| `--references` / `-References` | none | Answer key from `distill corpus build`; reviews the ledger automatically before preparation |
| `--reviews` / `-Reviews` | none | Review decisions JSONL. With `--references`, they override the automatic decisions. One of the two is required unless accepting pending records |
| `--accept-pending` / `-AcceptPending` | off | Train on unreviewed records, for plumbing tests only |
| `--prompts` / `-Prompts` | `data/examples/prompts.jsonl` | Prompt JSONL |
| `--ledger` / `-Ledger` | `data/teacher/ledger.jsonl` | Teacher ledger JSONL |
| `--config` / `-Config` | `configs/train.example.toml` | Training config |
| `--concurrency` / `-Concurrency` | 4 | Parallel teacher requests |
| `--max-output-tokens` / `-MaxOutputTokens` | 1024 | Teacher output limit |
| `--seed` / `-Seed` | none | Teacher sampling seed |

`student-train` also takes `--teacher-url` (`-TeacherUrl`, required) and `--token` (`-Token`, default `TEACHER_API_TOKEN`). `distill-local` also takes `--port` and `--upstream-url`.

## Environment variables

The scripts read `.env` from the repository root. Variables already set in the shell take precedence.

| Variable | Default | Meaning |
| --- | --- | --- |
| `TEACHER_UPSTREAM_MODEL` | `Qwen/Qwen3.5-9B` | Model vLLM serves, and the name the teacher API requests |
| `TEACHER_MODEL_REVISION` | pinned commit | Hugging Face revision vLLM loads |
| `VLLM_PORT` | 8001 | vLLM loopback port |
| `VLLM_MAX_MODEL_LEN` | 8192 | Context length; lower it to save KV cache memory |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.90 | Fraction of GPU memory vLLM may use |
| `VLLM_EXTRA_ARGS` | empty | Extra `vllm serve` flags, split on whitespace |
| `VLLM_STARTUP_TIMEOUT` | 1800 | Seconds to wait for vLLM, including the first model download |
| `VLLM_SPEC` | `vllm` | Package spec installed into `.venv-vllm` |
| `VLLM_PYTHON` | `3.12` (devenv sets a Nix path) | Interpreter for `.venv-vllm` |
| `VLLM_BIN` | none | Existing vLLM executable to use instead |
| `DISTILL_RUNTIME` | automatic | Force `devenv`, `uv`, or `pip` |
| `REVIEW_CODE_IMAGE` | none | Docker image for running Python tests during review, for example `python:3.12-slim` |

## Platform notes

- vLLM does not run natively on Windows. On Windows, start an OpenAI-compatible server separately, for example vLLM under WSL2, and pass `-UpstreamUrl`. The PowerShell scripts stop with this message instead of trying to install vLLM.
- QLoRA needs bitsandbytes, which the training extra installs only on Linux. On Windows, set `qlora = false` in the training config or train under WSL2.
- The PowerShell scripts need PowerShell 7.2 or newer (`pwsh`), not Windows PowerShell 5.1.
