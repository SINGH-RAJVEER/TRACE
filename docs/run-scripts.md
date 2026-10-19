# Run scripts

Coding is included in both launcher defaults. On Linux, `devenv shell` provides the sandbox tools. `uv run --extra eval --extra coding distill tasks build --task coding` streams and qualifies the complete pinned source and freezes its role pools and external test suite. `uv run --extra coding python -m coding.scoring` remains available for scoring saved responses. See [coding-task.md](coding-task.md).

`scripts/pipeline.py` runs the task study on Linux, macOS, and Windows.

`start` and `distill` default to the shared multi-domain pipeline. Input generation remains available through `distill gateway benign` and `distill gateway schedule`, and `distill gateway collect` executes a frozen schedule. See [the shared pipeline protocol](multi-domain-study.md#run-the-shared-pipeline).

| Scenario | Purpose |
| --- | --- |
| `teacher-host` | Start vLLM on loopback and the authenticated teacher API on the network; serve until interrupted |
| `distill` | Build tasks, collect budgeted responses, stop a locally hosted teacher, review usability, and train attacker and gold-reference adapters |
| `start` | Run the full study and benchmarks, then protected extraction, fresh student training, and another benchmark |

```bash
just distill
just distill --tasks ag_news,squad,gsm8k,coding --seeds 42,1,20 --budgets full
just distill --teacher-url http://TEACHER_IP:8000
python3 scripts/pipeline.py distill --help
```

On Windows, use `py -3 scripts\pipeline.py`. The bootstrap accepts Python 3.9 or newer and installs the Python 3.14 workspace. Relative option paths resolve against the repository root. just forwards options unquoted, so values cannot contain spaces.

The launcher first loads `.env` and installs the workspace. After validating command options, it installs the training stack and, when needed, vLLM. It builds missing task pools and runs collection and preparation, with review, training, and evaluation invoked as foreground subprocesses. `start` requires uv, directly or through devenv, to install the complete workspace. See [paper-tasks.md](paper-tasks.md#run-the-whole-study) for the task matrix and all options.

## Complete pipeline

Set `TEACHER_TERMS_REVIEW_ID` in `.env`, then run:

```bash
just start
just start --budgets 6000 --seeds 42 --run-dir artifacts/gateway-b6000
just start --teacher-url http://TEACHER_IP:8000
just start --upstream-url http://127.0.0.1:18011/v1
just start --help
```

Defaults are AG News, SQuAD, GSM8K, and coding; seeds 42, 1, and 20; full candidate inventories; A1/A2/A5; and `full`, `instantaneous`, and `quota` defenses. A5 has six total accounts. Numeric budgets are total campaign allowances, balanced across domains. Full mode uses all attacker pools with an unequal allocation and one shared allowance equal to their sum. The 24-hour deadline can stop collection before the full inventory is submitted. Proofs and failures consume that same allowance across every policy reset.

The runner builds missing corpora, calibrates one mixed account policy, collects undefended campaigns, selects a single shared recipe on development data, checks positive transfer in every domain, evaluates the teacher, and collects protected campaigns through fresh harness instances. Managed inference services stop before student work. One adapter per campaign and condition trains on all domains' received responses and is evaluated separately on every frozen suite. Optional gold references and per-domain token-matched controls also train shared adapters. Held-out benign replay and live utility remain separate from attacker training.

`artifacts/gateway` contains frozen protocol and source manifests, calibration, development selection, campaign ledgers and HTTP events, combined training datasets and adapters, per-domain evaluations, utility measurements, and `results.json`. `just distill` performs undefended mixed collection and shared training only, under `artifacts/gateway-distill`. It defaults to A1 and accepts `--attacks`. `start` accepts `--attacks`, `--arms`, `--interval-seconds`, `--deadline-seconds`, `--benign-accounts`, `--requests-per-account`, `--concentrated-share`, and `--development-budget`. See [the protocol](multi-domain-study.md#run-the-shared-pipeline) for exact accounting, controls, and validation limits.

GSM8K's 256-token and coding's 512-token caps remain provisional. Run `distill tasks pilot` against the actual teacher before freezing final study evidence. Coding requires Linux sandbox tools. Shared execution supports L4 refusal only. Existing completed campaigns are verified before reuse; interrupted campaigns cannot restart in place with fresh policy state. Use a new run directory to change the protocol. Externally managed teacher services remain the caller's responsibility during student training.

For the previous separate-task study, pass `--mode tasks` to `start` or `distill`. Its defaults and artifacts remain per task, including `artifacts/start`, numeric budget resolution, and policy-horizon reports. A single-domain study must select this mode:

```bash
just start --mode tasks --tasks ag_news --seeds 42 --budgets 500 --no-gold
just distill --mode tasks --tasks ag_news --seeds 42 --budgets 500,2000
```

## Task-study operation

This section describes the legacy `--mode tasks` workflow.

Copy `.env.example` to `.env` and set `TEACHER_TERMS_REVIEW_ID`. A local run starts the teacher only when planned prompts are missing from task ledgers. `--teacher-url` uses an existing repository teacher API and requires its server root, not its upstream OpenAI URL. `--upstream-url` uses an existing OpenAI-compatible model server while starting the repository API locally.

Each task writes `data/tasks/TASK/ledger.jsonl`; budgets and generated training configs live in `data/tasks/TASK/runs/sSEED-bBUDGET/`. Adapters live in `artifacts/tasks/TASK/sSEED-bBUDGET/attacker` and `gold`. The launcher queries the largest supported budget for each seed and reuses responses for smaller nested budgets. Task output limits are fixed at 16 tokens for AG News, 64 for SQuAD, and 256 for GSM8K.

Review checks response usability and training length without consulting gold labels. Gold completions train a separate reference adapter on the same prompts. The training recipe is `configs/train.task.toml`; generated copies set dataset, adapter directory, and training seed for each run.

Locally started services run in separate process groups and log under `artifacts/logs/`. The launcher stops them before training. Ctrl+C stops those services; during training the trainer saves a checkpoint before exiting. A later run resumes collection by example ID and training from checkpoints. Completed adapters must match the dataset hash and explicitly configured training settings before they are reused. Checkpoint resume also checks the trainer's fingerprint, including the chat template.

After calibration and the held-out benign replay, `start` replays one budget's extraction prompts as a single account under every defense and writes `artifacts/start/TASK/detectability.json`. It reports the extractor's peak level and window p-values next to the held-out benign accounts, broken down by client type. If no MMD defense escalates the extractor to a level reached by fewer than 10% of the benign horizons that complete a 100-request window (the first level target), `start` stops before collection with status `detectability_gate_failed`. Pass `--allow-undetected` to continue and report the study as a quota comparison; `detectability-gate.json` records that choice.

Benign clients are synthetic. Specialist and repetitive clients send concentrated windows that score far above broad windows, so their share sets the upper tail of the benign null and with it the extractor's p-values. `--concentrated-share` sets that share of benign source groups (default 0.1; 0.4 reproduces the equal five-way split of `task-clients-v1`). Changing it changes every calibration hash, so use a fresh `--run-dir`. Run the detectability check on its own, before the expensive matrix, across several shares:

```bash
just diagnose --tasks ag_news,squad --concentrated-shares 0,0.1,0.4
```

Each share gets its own calibration under `artifacts/task-diagnostic/concentrated-SHARE`, and `artifacts/task-diagnostic/detectability-sweep.json` tabulates the extractor's peak level against benign escalation rates for every share and defense.

Long steps show progress on stderr. Nonterminal logs receive periodic progress lines. Weight downloads show their own progress; service startup displays elapsed time and the latest log line. `distill` does not run evaluation automatically; `start` includes it.

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

## GPU settings

Set `VLLM_EXTRA_ARGS="--quantization fp8"` when the teacher needs quantization on a compatible GPU. The launcher uses `--language-model-only` and disables thinking. Lower `VLLM_GPU_MEMORY_UTILIZATION` when a display or other workload consumes memory. Set `VLLM_MAX_MODEL_LEN` for the actual serving context and concurrency, and measure the resulting memory use.

`VLLM_KV_OFFLOADING_SIZE` and `VLLM_KV_OFFLOADING_BACKEND` configure optional KV offloading; they do not reduce model-weight memory. Hybrid-attention offloading needs validation on the actual server. For student training, adjust batch size and accumulation together if needed. `qlora=true` requires matching `load_in_4bit=true` in both student evaluation slots. Hardware fits are estimates until measured with the selected task recipe.

## Environment variables

The script reads `.env` from the repository root. Variables already set in the shell take precedence; command options override their corresponding environment values. `.env.example` includes the project settings, populated defaults, and blank values for optional overrides or credentials without a fixed default. Training hyperparameters stay in `configs/train.*.toml`; they are not environment variables.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DISTILL_CONCURRENCY` | 4 | Parallel teacher requests |
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

## Existing Colab task session

The local task session uses the Colab `Qwen3.5-9B-Q8_0` teacher through the authenticated API at `http://127.0.0.1:18010`. This is a quantized deployment with a different identity from the default vLLM config; retain its actual model/provider metadata in all results.

The task collector is `artifacts/paper-tasks-colab/collect.py`; task ledgers remain under `data/tasks/`. The existing `artifacts/colab-teacher/q8/` directory holds shared teacher deployment and SSH connection state. Keep that state while the teacher is in use. Task collection resumes by example ID and exports each seed/budget subset after collection.

```bash
cat artifacts/paper-tasks-colab/progress.json
tail -n 10 artifacts/paper-tasks-colab/collection.log
```

Check the actual process and exit artifact before starting a second collector. The task session's concurrency and selected tasks are recorded in its plan; these are collection settings, not evidence of completed training or evaluation.
