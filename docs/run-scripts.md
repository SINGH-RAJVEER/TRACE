# Run scripts

`scripts/pipeline.py` runs the whole pipeline from one command per machine, on Linux, macOS, and Windows. It takes a scenario name and that scenario's options:

| Scenario | Machine | What it does |
| --- | --- | --- |
| `teacher-host` | Two machines, teacher side | Starts vLLM with Qwen3.5-9B on loopback and the teacher API on the network, then runs until interrupted |
| `student-train` | Two machines, student side | Checks the monorepo teacher API, collects responses, prepares the dataset, and trains the Qwen3.5-2B adapter |
| `distill-local` | One machine | Starts vLLM and the teacher API on loopback, collects, stops both to free GPU memory, then prepares and trains |

```bash
python3 scripts/pipeline.py distill-local --corpus standard --concurrency 8
python3 scripts/pipeline.py student-train --help
```

On Windows, start it with `py -3 scripts\pipeline.py`. Any Python 3.9 or newer can start the script, because it uses only the standard library until the workspace environment exists. Run it from any directory. Relative paths in options resolve against the repository root.

The same scenarios are `just` recipes, which pick `python3` or `py -3` for the platform. Options after the recipe name go straight to the scenario:

```bash
just teacher-host
just student-train --teacher-url http://10.0.0.5:8000 --corpus quick --concurrency 8
just distill-local --reviews data/reviews/reviews.jsonl
```

`just` passes options through unquoted, so option values cannot contain spaces.

The script does its work in two stages. First it loads `.env`, selects a runtime, installs the workspace packages, and installs vLLM when the scenario starts it. Then it hands the options to `distill run <scenario>` inside the workspace environment, which starts the services and runs each step. `distill run` also works on its own from the repository root once the environment exists, for example `uv run distill run student-train --help`.

## Progress

Each long step reports what it is doing while it runs:

| Step | What you see |
| --- | --- |
| Installing packages | uv's or pip's own download and install progress |
| Model download | `distill download-model` fetches the teacher weights into the Hugging Face cache before vLLM starts, with file and byte progress. vLLM then loads from the cache |
| vLLM and teacher API startup | A spinner with the elapsed time and the newest line of `artifacts/logs/vllm.log`, such as checkpoint shard loading |
| Corpus build | One line per dataset family, between the dataset downloads' own progress bars |
| Collection | A bar with prompts done out of the total (resumed records count as done), rate, time remaining, and counts of new, rejected, and retried requests |
| Review | A bar over the ledger records, then one over the Docker code tests when `REVIEW_CODE_IMAGE` is set |
| Training | The trainer's step progress bar, preceded by the student weight download |

Progress goes to stderr, so the result lines on stdout stay machine readable. When stderr is not a terminal, for example when the output is redirected to a log file, the bars become a plain line every 30 seconds.

## Runtime selection

The script picks how it installs and runs Python before doing anything else:

1. **devenv**. On NixOS with `devenv` installed, the script re-runs itself inside `devenv shell`, which provides Python 3.14, Python 3.12 for vLLM, uv, a compiler, and the library paths that binary wheels and the NVIDIA driver need. If devenv is missing, the script says so and continues with the next option. NixOS has no system `python3` by default, so start the script from the devenv shell (the repository's `.envrc` loads it with direnv) or run `devenv shell -- python3 scripts/pipeline.py ...`.
2. **uv**. With `uv` on `PATH`, the script runs `uv sync --all-packages --inexact` (adding `--extra train` for training) and uses `.venv`. uv downloads Python 3.14 if needed, except inside devenv, which provides it.
3. **pip**. Without uv, the script creates `.venv-pip` with a Python 3.14 interpreter (`python3.14`, `python3`, or `py -3.14` on Windows) and installs the workspace packages in editable mode. pip does not read `uv.lock`, so dependency versions can differ from the locked set.

Before installing anything, the script lists every missing tool it can detect, then stops if any of them is required:

- Required: uv for the uv runtime. For the pip runtime: Python 3.14 with `venv` and `ensurepip` (the `python3.14-venv` package on Debian and Ubuntu), plus a Python 3.10 to 3.13 interpreter for vLLM when vLLM is not already installed.
- Warnings when a GPU is needed: `nvidia-smi` (the NVIDIA driver), and a C compiler (`cc` or `gcc`) with the C library headers for the Triton kernels that vLLM and training compile at runtime. On Debian and Ubuntu, including WSL, `build-essential` provides both; a bare `gcc` without `libc6-dev` makes vLLM fail at startup with `fatal error: stdlib.h: No such file or directory`.
- On Windows, the script stops here if it would need to start vLLM (see Platform notes).

Set `DISTILL_RUNTIME=devenv|uv|pip` to force a runtime.

vLLM gets its own environment because its wheels target older Python releases than this workspace. If `vllm` is on `PATH`, or `VLLM_BIN` points at an executable, the script uses that. Otherwise it creates `.venv-vllm` from `VLLM_PYTHON` (default `3.12`) and installs `VLLM_SPEC` (default `vllm`). The first install downloads several gigabytes. Pin `VLLM_SPEC` to an exact version for a paper run.

## Teacher machine

Set an approval reference and an API token in `.env`, then start the host:

```bash
just teacher-host
```

The script binds the teacher API to `0.0.0.0` on `TEACHER_PORT` (default 8000) and keeps vLLM on `127.0.0.1:VLLM_PORT` (default 8001), so only the authenticated teacher API is reachable from the network. It refuses to bind beyond loopback without `TEACHER_API_TOKEN`. `.env`'s `TEACHER_HOST` does not apply here; pass `--host` to change the address. The API has no TLS, so expose it only on a trusted network or behind a TLS gateway.

Startup downloads the model into the Hugging Face cache, waits for vLLM to load it, then for `/healthz` to report `ready`, then sends one small generation through the API. That check fails if the token is wrong or if the response contains a `<think>` trace. vLLM starts with `--default-chat-template-kwargs '{"enable_thinking": false}'`, so Qwen3.5 answers in non-thinking mode without changing the API's request format, and with `--language-model-only`, so the checkpoint's vision encoder is never loaded. When ready, the script prints the student command with this machine's address and then watches both processes. Ctrl+C stops both.

| Option | Default | Meaning |
| --- | --- | --- |
| `--host` | `0.0.0.0` | Teacher API bind address |
| `--port` | `TEACHER_PORT` or 8000 | Teacher API port |
| `--upstream-url` | none | Use an existing OpenAI-compatible server instead of starting vLLM |

## Student machine

Start the teacher with `just teacher-host` on the teacher machine. Set the same `TEACHER_API_TOKEN` and an approval reference in the student machine's `.env`, then use the teacher API's server root:

```dotenv
TEACHER_API_TOKEN=the-same-token-as-the-teacher
TEACHER_TERMS_REVIEW_ID=your-review-reference
```

```bash
just student-train --teacher-url http://10.0.0.5:8000 --corpus quick --concurrency 8
```

The script installs the training stack, waits up to 60 seconds for `/healthz` to report `ready`, sends one smoke generation, and collects through `/v1/teacher/generate`. The teacher API handles the upstream model URL, API key, model ID, and provider identity. Use the server root for `--teacher-url`.

After collection, the script runs `distill review`, `distill prepare`, and `distill train`. Preparation writes to the `dataset_path` in the training config. Collection resumes by `example_id`, and training resumes from the newest checkpoint in the config's `output_dir` (see [Checkpoints and resuming](distillation-pipeline.md#checkpoints-and-resuming)). Ctrl+C during training saves a checkpoint before the script exits. `--token` overrides `TEACHER_API_TOKEN`; the launcher passes the credential to the collector in an environment variable, so it never appears in the process list.

## Single machine

```bash
just distill-local --reviews data/reviews/reviews.jsonl
```

Both servers bind to loopback and log to `artifacts/logs/` (`vllm.log` and `teacher-api.log`). Each runs in its own process group, so stopping it also stops vLLM's engine processes. After collection, the script stops the teacher API and vLLM before training, because vLLM reserves most of the GPU for the teacher and leaves too little for training alongside it. The machine therefore needs enough GPU memory for the teacher during collection (see the model pairing in [distillation-pipeline.md](distillation-pipeline.md)). Lower `VLLM_MAX_MODEL_LEN` or add `--quantization fp8` to `VLLM_EXTRA_ARGS` if vLLM runs out of memory. `--upstream-url` skips vLLM and uses an existing server instead.

## Full run with the prompt corpus

A real run starts from the prompt corpus. Pass `--corpus` with a profile from [prompt-corpus.md](prompt-corpus.md):

```bash
just distill-local --corpus standard --concurrency 8
```

The prompt files are not in the repository. `--corpus` builds into `data/prompts/`, or beside an explicit `--prompts` path, after installing the Python environment. It downloads the pinned public datasets and the held-out benchmark questions, which takes a few minutes. A later run with the same profile reuses the existing files. Unless you pass them yourself, `--corpus` also selects the adjacent `references.jsonl` and `--config configs/train.distill.toml`. The build is deterministic and a prompt keeps the same ID and text in every profile, so moving from `quick` to `standard` reuses the responses already in the ledger. To build the corpus on its own, run `just corpus standard`.

With `--references`, the script runs `distill review` after collection. It checks every teacher response against the answer key, rejects responses that would exceed the training config's `max_length`, and writes `data/teacher/ledger.reviews.jsonl` plus a `.summary.json` with counts by status and rejection reason. Preparation then uses those decisions. A `--reviews` file passed at the same time holds human decisions that override the automatic ones. Set `REVIEW_CODE_IMAGE=python:3.12-slim` to run the Python tests in the same isolated Docker container the benchmarks use; without it, code is only parsed.

The same steps work on two machines: pass `--corpus` to `student-train`.

### GPUs with 20 to 24 GB

The teacher's BF16 text weights take about 18 GB, so a 20 GB card such as the RTX 4000 Ada must quantize them. Ada and newer GPUs support FP8, which brings the weights to roughly 11 GB; the embedding and output layers stay in BF16. Add this line to `.env`:

```dotenv
VLLM_EXTRA_ARGS="--quantization fp8"
```

The script already passes `--language-model-only`, so the vision encoder takes no GPU memory. Only one layer in four keeps a KV cache, about 32 KB per token in BF16, so the default 8,192-token context and `--concurrency 16` should fit in what remains. Keep the KV cache in BF16: the one run with `--kv-cache-dtype fp8` on an RTX 4000 Ada failed while FlashInfer compiled a kernel at startup, which needs `nvcc` from a CUDA toolkit. vLLM still downloads the BF16 checkpoint and quantizes it while loading, so the machine, or the WSL virtual machine, needs enough system memory to read it.

A GPU that also drives a display already has about 1 GB in use. vLLM refuses to start when free memory is below `VLLM_GPU_MEMORY_UTILIZATION` times the total; lower the value if the log reports that.

For training, the 2B student's 4-bit text model takes about 2 to 3 GB, because its 248K-token embedding stays unquantized, and the rank-128 adapter with its optimizer state about 2 GB, so batches of 4 at the config's 3,072-token limit should fit in 20 GB with room to spare. These figures are estimates, not measurements. If training runs out of memory, halve `per_device_batch_size` and double `gradient_accumulation_steps`.

## Pipeline options

`student-train` and `distill-local` share these options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--terms-review-id` | `TEACHER_TERMS_REVIEW_ID` | Approval reference written into every ledger record; required |
| `--corpus` | none | Build or reuse the `quick`, `standard`, or `full` prompt corpus and use its prompts, answer key, and `configs/train.distill.toml` |
| `--references` | none | Answer key from `distill corpus build`; reviews the ledger automatically before preparation |
| `--reviews` | none | Review decisions JSONL. With `--references`, they override the automatic decisions. One of the two is required unless accepting pending records |
| `--accept-pending` | off | Train on unreviewed records, for plumbing tests only |
| `--prompts` | `data/prompts/all-splits.jsonl` | Prompt JSONL, written by `distill corpus build` |
| `--ledger` | `data/teacher/ledger.jsonl` | Teacher ledger JSONL |
| `--config` | `configs/train.example.toml` | Training config |
| `--concurrency` | 4 | Parallel teacher requests |
| `--max-output-tokens` | 1024 | Teacher output limit |
| `--seed` | none | Teacher sampling seed |

`student-train` also takes `--teacher-url` (required, the teacher API server root) and `--token` (default `TEACHER_API_TOKEN`). `distill-local` also takes `--port` and `--upstream-url`.

## Environment variables

The script reads `.env` from the repository root. Variables already set in the shell take precedence.

| Variable | Default | Meaning |
| --- | --- | --- |
| `TEACHER_UPSTREAM_MODEL` | `Qwen/Qwen3.5-9B` | Model vLLM serves, and the name the teacher API requests |
| `TEACHER_UPSTREAM_API_KEY` | empty | Upstream bearer token used by the teacher API |
| `TEACHER_API_TOKEN` | empty | Bearer token shared by the teacher API and student launcher |
| `TEACHER_PROVIDER_NAME` | `self-hosted-vllm` for local vLLM | Provider identity recorded by the teacher API |
| `TEACHER_MODEL_REVISION` | pinned commit | Hugging Face revision vLLM loads |
| `VLLM_PORT` | 8001 | vLLM loopback port |
| `VLLM_MAX_MODEL_LEN` | 8192 | Context length; lower it to save KV cache memory |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.90 | Fraction of GPU memory vLLM may use |
| `VLLM_EXTRA_ARGS` | empty | Extra `vllm serve` flags, split like a shell command line, so quotes group words |
| `VLLM_STARTUP_TIMEOUT` | 1800 | Seconds to wait for vLLM to load the downloaded model and start serving |
| `VLLM_SPEC` | `vllm` | Package spec installed into `.venv-vllm` |
| `VLLM_PYTHON` | `3.12` (devenv sets a Nix path) | Interpreter for `.venv-vllm` |
| `VLLM_BIN` | none | Existing vLLM executable to use instead |
| `HF_HUB_OFFLINE` | unset | `1` skips the model download and uses the Hugging Face cache as is |
| `DISTILL_RUNTIME` | automatic | Force `devenv`, `uv`, or `pip` |
| `REVIEW_CODE_IMAGE` | none | Docker image for running Python tests during review, for example `python:3.12-slim` |

## Platform notes

- vLLM does not run natively on Windows. On Windows, start an OpenAI-compatible server separately, for example vLLM under WSL2, and pass `--upstream-url`. The script stops with this message instead of trying to install vLLM.
- QLoRA needs bitsandbytes, which the training extra installs only on Linux. On Windows, set `qlora = false` in the training config or train under WSL2.
- On Windows, the `just` recipes run through PowerShell 7 (`pwsh`), as the justfile's `windows-shell` sets. Running `py -3 scripts\pipeline.py` directly needs only Python.

## Colab over SSH

Google's [Colab CLI](https://github.com/googlecolab/google-colab-cli) implements [SSH over a WebSocket proxy](https://github.com/googlecolab/google-colab-cli/blob/main/docs/06_ssh_access.md). `SESSION` is a session name, not a hostname or credential; use `colab sessions` to find an existing session or create a named T4 session with `colab new -s teacher-t4 --gpu T4`. Check `colab ssh --help` after installing with `uv tool install google-colab-cli`, because older published versions can lack the SSH command. The CLI accepts `--auth oauth2` for browser authorization or `--auth adc` for existing Application Default Credentials; `colab auth` instead authenticates services inside the VM. SSH requires an Ed25519 or ECDSA key. See the CLI's [authentication documentation](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md).

The [Colab FAQ](https://research.google.com/colaboratory/faq.html#disallowed-activities) disallows SSH on managed runtimes running free of charge without a positive compute-unit balance, and says a paid plan with a positive balance removes that restriction. GPU availability and runtime lifetime remain limited.

After starting the teacher API on the remote machine with `python3 scripts/pipeline.py teacher-host --host 127.0.0.1`, forward it to local port 18000:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18000:127.0.0.1:8000 -o ProxyCommand="colab ssh --proxy-mode -s %h" root@teacher-t4
```

Keep that connection open and point the local student launcher at `http://127.0.0.1:18000`, using the same `TEACHER_API_TOKEN`. If SSH cannot find `colab`, use its absolute executable path in `ProxyCommand`.

The current `Qwen/Qwen3.5-9B` teacher needs quantization on a 16 GB T4: the [upstream vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-9B) budgets 22 GB for BF16. T4 meets [vLLM's general GPU requirement](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/), but a compatible quantized checkpoint, FP16 activations, and working Qwen3.5 hybrid-attention kernels still need a real generation check. [vLLM's quantization table](https://docs.vllm.ai/en/latest/features/quantization/#supported-hardware) lists AWQ, GPTQ, and bitsandbytes for Turing; the native FP8 W8A8 path described for Ada above is unsupported there. This SSH connection has not been validated with the teacher on a live T4, and the repository's default teacher command is not a verified T4 configuration.

### Selected T4 profile

Use the text-only `Qwen3.5-9B-Q4_K_M.gguf` from [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/3885219b6810b007914f3a7950a8d1b469d598a5), pinned to revision `3885219b6810b007914f3a7950a8d1b469d598a5`. The file is 5,680,522,464 bytes, about 5.29 GiB, with SHA-256 `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`. Hugging Face's [llama.cpp deployment catalog](https://endpoints.huggingface.co/catalog/collection/qwen?inferenceServer=llamacpp) lists this model and quantization for a single T4. The file size excludes runtime buffers and cache; live GPU memory use and generation remain unverified.

Build [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/2f729cfa4ca6f55b8bacf67bb2ba92cb9548e291) with its CUDA backend for T4's compute capability 7.5, using `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75` and a CUDA toolkit that still supports that architecture. Start its [OpenAI-compatible server](https://github.com/ggml-org/llama.cpp/blob/2f729cfa4ca6f55b8bacf67bb2ba92cb9548e291/tools/server/README.md) on the Colab runtime after downloading and checking the pinned GGUF:

```bash
llama-server \
	--model /content/models/Qwen3.5-9B-Q4_K_M.gguf \
	--alias Qwen3.5-9B-Q4_K_M \
	--host 127.0.0.1 --port 8001 \
	--n-gpu-layers 99 --ctx-size 4096 --parallel 1 \
	--jinja --chat-template-kwargs '{"enable_thinking":false}' \
	--reasoning-format none
```

Keep the vision projector unloaded for text collection. The explicit single slot avoids automatic parallelism reserving extra memory. Thinking is disabled in the template; `--reasoning-format none` leaves any unexpected thought text visible to the teacher API's smoke check.

An alternative to installing the repository on Colab is to forward only this model server and run the teacher API locally:

```bash
ssh -N -o ExitOnForwardFailure=yes \
	-L 127.0.0.1:18001:127.0.0.1:8001 \
	-o ProxyCommand="/home/rajveer/.local/bin/colab ssh --proxy-mode -s %h" \
	root@SESSION
```

Replace the executable path with the CLI location on the local machine and `SESSION` with the actual session name. In another local terminal, with `TEACHER_TERMS_REVIEW_ID` configured as usual:

```bash
TEACHER_UPSTREAM_MODEL=Qwen3.5-9B-Q4_K_M \
TEACHER_PROVIDER_NAME=colab-llamacpp-q4_k_m \
	python3 scripts/pipeline.py teacher-host --host 127.0.0.1 \
	--upstream-url http://127.0.0.1:18001/v1
```

This keeps the existing local teacher API at `http://127.0.0.1:8000`, records the quantized model alias and provider, and uses the launcher's generation check. Confirm GPU offload in the remote server log and inspect `nvidia-smi` under an actual request before collecting data. Start collection with concurrency 1 and persist its ledger locally so a Colab runtime expiration does not discard the responses.
