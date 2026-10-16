# Run scripts

`scripts/pipeline.py` runs the whole pipeline from one command per machine, on Linux, macOS, and Windows. It takes a scenario name and that scenario's options:

| Scenario | Machine | What it does |
| --- | --- | --- |
| `teacher-host` | Two machines, teacher side | Starts vLLM with Qwen3.5-9B on loopback and the teacher API on the network, then runs until interrupted |
| `student-train` | Two machines, student side | Builds or reuses the corpus, checks the monorepo teacher API, collects and reviews responses, prepares the splits, and trains the Qwen3.5-2B adapter |
| `distill-local` | One machine | Builds or reuses the corpus, starts vLLM and the teacher API on loopback, collects a ledger, stops both to free GPU memory, then reviews, prepares, and trains |

```bash
python3 scripts/pipeline.py distill-local
python3 scripts/pipeline.py student-train --help
```

On Windows, start it with `py -3 scripts\pipeline.py`. Any Python 3.9 or newer can start the script, because it uses only the standard library until the workspace environment exists. Run it from any directory. Relative paths in options resolve against the repository root.

The same scenarios are `just` recipes, which pick `python3` or `py -3` for the platform. Options after the recipe name go straight to the scenario:

```bash
just teacher-host
just student-train --teacher-url http://10.0.0.5:8000 --corpus quick --concurrency 8
just distill-local
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
just distill-local
```

Copy `.env.example` to `.env` and set `TEACHER_TERMS_REVIEW_ID` before the first run. With no data source options, this command uses `DISTILL_CORPUS`, defaulting to `standard`, and `configs/train.distill.toml`. It installs the training dependencies, checks for the corpus and builds it if needed, starts the teacher, creates or resumes `data/teacher/ledger.jsonl`, stops the teacher, reviews the responses against the answer key, prepares both dataset splits, and trains the student. It needs no prebuilt corpus or manual review file. Set `DISTILL_CONCURRENCY`, `DISTILL_MAX_OUTPUT_TOKENS`, and optionally `DISTILL_SEED` in `.env` to configure collection without extra flags.

Both servers bind to loopback and log to `artifacts/logs/` (`vllm.log` and `teacher-api.log`). Each runs in its own process group, so stopping it also stops vLLM's engine processes. After collection, the script stops the teacher API and vLLM before training, because vLLM reserves most of the GPU for the teacher and leaves too little for training alongside it. The machine therefore needs enough GPU memory for the teacher during collection (see the model pairing in [distillation-pipeline.md](distillation-pipeline.md)). Lower `VLLM_MAX_MODEL_LEN` or add `--quantization fp8` to `VLLM_EXTRA_ARGS` if vLLM runs out of memory. `--upstream-url` skips vLLM and uses an existing server instead.

## Full run with the prompt corpus

A run with no data source options starts from the `standard` prompt corpus. Set `DISTILL_CORPUS` in `.env` or pass `--corpus` with a profile from [prompt-corpus.md](prompt-corpus.md):

```bash
just distill-local --corpus standard --concurrency 8
```

The prompt files are not in the repository. Corpus mode builds into `data/prompts/`, or beside an explicit `--prompts` path, after installing the Python environment. It downloads the pinned public datasets and the held-out benchmark questions, which takes a few minutes. A later run reuses the existing files when the manifest's profile matches and the prompt and answer-key files are present; otherwise it rebuilds them. Unless you pass them yourself, corpus mode also selects the adjacent `references.jsonl` and `--config configs/train.distill.toml`. The build is deterministic and a prompt keeps the same ID and text in every profile, so moving from `quick` to `standard` reuses the responses already in the ledger. To build the corpus on its own, run `just corpus standard`.

An explicit `--corpus` overrides `DISTILL_CORPUS`. Supplying `--references`, `--reviews`, `--accept-pending`, or a custom `--prompts` path disables automatic corpus selection, so existing custom-data workflows keep their own prompts and config. Add `--corpus PROFILE` explicitly to combine corpus building with those options.

With `--references`, the script runs `distill review` after collection. It checks every teacher response against the answer key, rejects responses that would exceed the training config's `max_length`, and writes `data/teacher/ledger.reviews.jsonl` plus a `.summary.json` with counts by status and rejection reason. Preparation then uses those decisions. A `--reviews` file passed at the same time holds human decisions that override the automatic ones. Set `REVIEW_CODE_IMAGE=python:3.12-slim` to run the Python tests in the same isolated Docker container the benchmarks use; without it, code is only parsed.

The same default corpus workflow runs on two machines with `just student-train --teacher-url http://TEACHER_IP:8000`.

### GPUs with 20 to 24 GB

The teacher's BF16 text weights take about 18 GB, so a 20 GB card such as the RTX 4000 Ada must quantize them. Ada and newer GPUs support FP8, which brings the weights to roughly 11 GB; the embedding and output layers stay in BF16. Add this line to `.env`:

```dotenv
VLLM_EXTRA_ARGS="--quantization fp8"
```

The script already passes `--language-model-only`, so the vision encoder takes no GPU memory. Only one layer in four keeps a KV cache, about 32 KB per token in BF16, so the default 8,192-token context and `--concurrency 16` should fit in what remains. Keep the KV cache in BF16: the one run with `--kv-cache-dtype fp8` on an RTX 4000 Ada failed while FlashInfer compiled a kernel at startup, which needs `nvcc` from a CUDA toolkit. vLLM still downloads the BF16 checkpoint and quantizes it while loading, so the machine, or the WSL virtual machine, needs enough system memory to read it.

To keep more KV cache than fits on the GPU, set `VLLM_KV_OFFLOADING_SIZE` to a size in GiB, for example `16`. vLLM then moves KV blocks evicted from GPU memory into that much pinned system RAM and reloads them when a later request shares the prefix, instead of recomputing it. This does not reduce the GPU memory vLLM reserves or let a single request exceed the GPU's KV capacity; it raises the prefix cache hit rate across requests. Leave enough RAM for loading the checkpoint as well. This differs from `--cpu-offload-gb`, which offloads model weights and slows every forward pass. Offloading with Qwen3.5's hybrid attention layers has not been tested in this repository; check `vllm.log` at startup.

A GPU that also drives a display already has about 1 GB in use. vLLM refuses to start when free memory is below `VLLM_GPU_MEMORY_UTILIZATION` times the total; lower the value if the log reports that.

For training, the 2B student's BF16 text model takes about 3.8 GB, and the rank-128 adapter with its optimizer state about 2 GB, so batches of 4 at the config's 3,072-token limit should fit in 20 GB with room to spare. These figures are estimates, not measurements. If training runs out of memory, halve `per_device_batch_size` and double `gradient_accumulation_steps`. Set `qlora = true` only as a last resort, and set `load_in_4bit = true` for both student slots of the evaluation config to match.

## Pipeline options

`student-train` and `distill-local` share these options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--terms-review-id` | `TEACHER_TERMS_REVIEW_ID` | Approval reference written into every ledger record; required |
| `--corpus` | `DISTILL_CORPUS` or `standard`, when no other data source is supplied | Build or reuse the `quick`, `standard`, or `full` prompt corpus and use its prompts, answer key, and `configs/train.distill.toml` |
| `--references` | none | Answer key from `distill corpus build`; reviews the ledger automatically before preparation |
| `--reviews` | none | Review decisions JSONL. With `--references`, they override the automatic decisions. Custom-data runs need reviews or references unless accepting pending records |
| `--accept-pending` | off | Train on unreviewed records, for plumbing tests only |
| `--prompts` | `data/prompts/all-splits.jsonl` | Prompt JSONL, written by `distill corpus build` |
| `--ledger` | `data/teacher/ledger.jsonl` | Teacher ledger JSONL |
| `--config` | `configs/train.distill.toml` in corpus mode; otherwise `configs/train.example.toml` | Training config |
| `--concurrency` | `DISTILL_CONCURRENCY` or 4 | Parallel teacher requests |
| `--max-output-tokens` | `DISTILL_MAX_OUTPUT_TOKENS` or 1024 | Teacher output limit |
| `--seed` | `DISTILL_SEED` or none | Teacher sampling seed |

`student-train` also takes `--teacher-url` (required, the teacher API server root) and `--token` (default `TEACHER_API_TOKEN`). `distill-local` also takes `--port` and `--upstream-url`.

## Environment variables

The script reads `.env` from the repository root. Variables already set in the shell take precedence; command options override their corresponding environment values. `.env.example` includes the project settings, populated defaults, and blank values for optional overrides or credentials without a fixed default. Training hyperparameters stay in `configs/train.*.toml`; they are not environment variables.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DISTILL_CORPUS` | `standard` | Automatic corpus profile when no data source option is supplied |
| `DISTILL_CONCURRENCY` | 4 | Parallel teacher requests |
| `DISTILL_MAX_OUTPUT_TOKENS` | 1024 | Maximum generated tokens per teacher response |
| `DISTILL_SEED` | none | Optional teacher sampling seed, separate from the training config's seed |
| `TEACHER_TERMS_REVIEW_ID` | none | Required approval reference recorded in the ledger |
| `TEACHER_UPSTREAM_BASE_URL` | local vLLM URL in the example; empty in the standalone API | Standalone API's OpenAI-compatible upstream; launchers use their managed vLLM URL or explicit `--upstream-url` |
| `TEACHER_UPSTREAM_MODEL` | `Qwen/Qwen3.5-9B` | Model vLLM serves, and the name the teacher API requests |
| `TEACHER_UPSTREAM_API_KEY` | empty | Upstream bearer token used by the teacher API |
| `TEACHER_API_TOKEN` | empty | Bearer token shared by the teacher API and student launcher |
| `TEACHER_PROVIDER_NAME` | `self-hosted-vllm` for local vLLM | Provider identity recorded by the teacher API |
| `TEACHER_TIMEOUT_SECONDS` | 120 | Timeout for an upstream generation request |
| `TEACHER_HOST` | `127.0.0.1` | Standalone API bind address; launchers choose their own host |
| `TEACHER_PORT` | 8000 | Teacher API port |
| `TEACHER_LOG_LEVEL` | `info` | Teacher API logging level |
| `TEACHER_MODEL_REVISION` | pinned commit | Hugging Face revision vLLM loads |
| `VLLM_PORT` | 8001 | vLLM loopback port |
| `VLLM_MAX_MODEL_LEN` | 8192 | Context length; lower it to save KV cache memory |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.90 | Fraction of GPU memory vLLM may use |
| `VLLM_EXTRA_ARGS` | empty | Extra `vllm serve` flags, split like a shell command line, so quotes group words |
| `VLLM_KV_OFFLOADING_SIZE` | empty | GiB of system RAM for KV cache blocks evicted from the GPU, passed as `--kv-offloading-size`; empty disables offloading |
| `VLLM_KV_OFFLOADING_BACKEND` | `native` | `--kv-offloading-backend`, `native` or `lmcache`; used only when a size is set |
| `VLLM_STARTUP_TIMEOUT` | 1800 | Seconds to wait for vLLM to load the downloaded model and start serving |
| `VLLM_SPEC` | `vllm` | Package spec installed into `.venv-vllm` |
| `VLLM_PYTHON` | `3.12` (devenv sets a Nix path) | Interpreter for `.venv-vllm` |
| `VLLM_BIN` | none | Existing vLLM executable to use instead |
| `HF_HUB_OFFLINE` | disabled, `0` in the example | `1` or `true` skips the model download and uses the Hugging Face cache as is |
| `DISTILL_RUNTIME` | `auto` | Automatically select a runtime, or force `devenv`, `uv`, or `pip` |
| `REVIEW_CODE_IMAGE` | none | Docker image for running Python tests during review, for example `python:3.12-slim` |
| `TOKENIZERS_PARALLELISM` | `false` for training | Tokenizer worker parallelism |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` for training | CUDA allocator settings; the launcher supplies this only when neither allocator variable is set |
| `PYTORCH_CUDA_ALLOC_CONF` | none | Legacy allocator override; clear `PYTORCH_ALLOC_CONF` in `.env` before using it |
| `PYTHONUNBUFFERED` | `1` in devenv and the example | Flush Python process output without buffering |
| `UV_PYTHON` | devenv's Python 3.14 path | Workspace interpreter; generated path is documented but left unset in the example |
| `UV_PYTHON_DOWNLOADS` | `never` in devenv | devenv supplies Python itself; leave unset elsewhere for uv's own interpreter management |
| `LD_LIBRARY_PATH` | generated on Linux in devenv | Nix C++ runtime, zlib, and `/run/opengl-driver/lib`; leave unset on other platforms |
| `TRITON_LIBCUDA_PATH` | `/run/opengl-driver/lib` on Linux in devenv | Host driver path for Triton; leave unset on other platforms |

## Platform notes

- vLLM does not run natively on Windows. On Windows, start an OpenAI-compatible server separately, for example vLLM under WSL2, and pass `--upstream-url`. The script stops with this message instead of trying to install vLLM.
- QLoRA needs bitsandbytes, which the training extra installs only on Linux. The example configs train without it; on Windows, keep `qlora = false` or train under WSL2.
- On Windows, the `just` recipes run through PowerShell 7 (`pwsh`), as the justfile's `windows-shell` sets. Running `py -3 scripts\pipeline.py` directly needs only Python.

## Colab over SSH

Google's [Colab CLI](https://github.com/googlecolab/google-colab-cli) implements [SSH over a WebSocket proxy](https://github.com/googlecolab/google-colab-cli/blob/main/docs/06_ssh_access.md). `SESSION` is a session name, not a hostname or credential; use `colab sessions` to find an existing session or create a named T4 session with `colab new -s teacher-t4 --gpu T4`. Check `colab ssh --help` after installing with `uv tool install google-colab-cli`, because older published versions can lack the SSH command. The CLI accepts `--auth oauth2` for browser authorization or `--auth adc` for existing Application Default Credentials; `colab auth` instead authenticates services inside the VM. SSH requires an Ed25519 or ECDSA key. See the CLI's [authentication documentation](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md).

The [Colab FAQ](https://research.google.com/colaboratory/faq.html#disallowed-activities) disallows SSH on managed runtimes running free of charge without a positive compute-unit balance, and says a paid plan with a positive balance removes that restriction. GPU availability and runtime lifetime remain limited.

After starting the teacher API on the remote machine with `python3 scripts/pipeline.py teacher-host --host 127.0.0.1`, forward it to local port 18000:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18000:127.0.0.1:8000 -o ProxyCommand="colab ssh --proxy-mode -s %h" root@teacher-t4
```

Keep that connection open and point the local student launcher at `http://127.0.0.1:18000`, using the same `TEACHER_API_TOKEN`. If SSH cannot find `colab`, use its absolute executable path in `ProxyCommand`.

The current `Qwen/Qwen3.5-9B` teacher needs quantization on a 16 GB T4: the [upstream vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-9B) budgets 22 GB for BF16. T4 meets [vLLM's general GPU requirement](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/), but a compatible quantized checkpoint, FP16 activations, and working Qwen3.5 hybrid-attention kernels still need a real generation check. [vLLM's quantization table](https://docs.vllm.ai/en/latest/features/quantization/#supported-hardware) lists AWQ, GPTQ, and bitsandbytes for Turing; the native FP8 W8A8 path described for Ada above is unsupported there. The GGUF configuration below passed live inference through SSH on a T4; the repository's default vLLM command remains unverified on that GPU.

### Selected T4 profile

Use the text-only `Qwen3.5-9B-Q4_K_M.gguf` from [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/3885219b6810b007914f3a7950a8d1b469d598a5), pinned to revision `3885219b6810b007914f3a7950a8d1b469d598a5`. The file is 5,680,522,464 bytes, about 5.29 GiB, with SHA-256 `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`. Hugging Face's [llama.cpp deployment catalog](https://endpoints.huggingface.co/catalog/collection/qwen?inferenceServer=llamacpp) lists this model and quantization for a single T4. The file size excludes runtime buffers and cache; the measurements below cover two short generations rather than a full-context stress test.

The verified deployment uses the official [llama.cpp b11669 release](https://github.com/ggml-org/llama.cpp/releases/tag/b11669), commit `620d2225c`, with its Ubuntu x64 CUDA 12.8 binary and matching `cudart` archive. Both release archive digests and the GGUF checksum were checked before startup. This avoids changing Colab's installed CUDA 13 toolkit. For a source build, use the CUDA backend with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75` and a CUDA toolkit that supports T4's compute capability 7.5. Start the [OpenAI-compatible server](https://github.com/ggml-org/llama.cpp/blob/b11669/tools/server/README.md) on the Colab runtime after downloading and checking the pinned GGUF:

```bash
llama-server \
	--model /content/models/Qwen3.5-9B-Q4_K_M.gguf \
	--alias Qwen3.5-9B-Q4_K_M \
	--host 127.0.0.1 --port 8001 \
	--n-gpu-layers 99 --ctx-size 4096 --parallel 1 \
	--jinja --reasoning off --reasoning-format deepseek
```

Keep the vision projector unloaded for text collection. The explicit single slot avoids automatic parallelism reserving extra memory. `--reasoning off` disables thinking in the template. The DeepSeek-format response parser removes the template's empty thinking markers; `--reasoning-format none` instead includes those synthetic markers in the content even when the model generated no reasoning, which makes the launcher's smoke check fail. Check a direct upstream response as well as the teacher API: `reasoning_content` must be absent or empty, and the content must contain the requested answer without thought tags. Collector requests continue to send `enable_thinking=false`.

An alternative to installing the repository on Colab is to forward only this model server and run the teacher API locally:

```bash
ssh -N -o ExitOnForwardFailure=yes \
	-L 127.0.0.1:18001:127.0.0.1:8001 \
	-o ProxyCommand="~/.local/bin/colab ssh --proxy-mode -s %h" \
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

### Verified deployment on 2026-10-16

The existing T4 runtime was registered locally as `colab`, with no new runtime allocation. The model server runs on Colab loopback port 8001. Its SSH tunnel listens locally on port 18011, and the authenticated teacher API listens at `http://127.0.0.1:18010`; the usual local ports were already occupied. Start this API with the preceding environment overrides, `--port 18010`, and `--upstream-url http://127.0.0.1:18011/v1`. The existing `.env` supplies the teacher API token and terms reference. Model responses identify `Qwen3.5-9B-Q4_K_M` and provider `colab-llamacpp-q4_k_m`.

The launcher's generation check passed. A direct upstream request returned `56` for `7 * 8` with no reasoning content, and the authenticated teacher API returned the same answer plus a complete 60-token explanatory response without thought tags. An unauthenticated generation request returned HTTP 401. During those requests, GPU memory reached 5,277 MiB out of 15,360 MiB and utilization reached 98%. These are deployment checks, not benchmark scores or an assessment of quantization quality.

Colab returned HTTP 429 when a second SSH connection was opened while the tunnel was active. Use OpenSSH multiplexing to share one connection between the tunnel and management commands, with the same `ControlPath` for both. A dedicated Ed25519 key is selected with `-i` for both OpenSSH and the Colab proxy. The first cold load and generation took over a minute while the NVIDIA driver populated its CUDA compilation cache; subsequent requests completed normally.

Deployment files and local logs are under the ignored `artifacts/colab-teacher/` directory: `manifest.json` pins the model and release archive hashes; `validation.json`, `upstream-validation.json`, and `gpu-samples.txt` record the successful checks; `api.log` and `tunnel.log` hold local service logs. Colab's files, server PID, and log are under `/content/capstone-teacher/`. These detached processes continue after the launching terminal exits, while the Colab runtime and local machine remain available. No automatic runtime recreation or service restart is configured.
