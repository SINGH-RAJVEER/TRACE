# Run scripts

`scripts/pipeline.py` runs the task study on Linux, macOS, and Windows.

For the new gateway study, `uv run distill gateway benign` builds mixed benign clients and `uv run distill gateway schedule` builds fixed A1/A2/A5 inventories from existing task pools. These commands generate hashed inputs only. See the [multi-domain protocol and examples](multi-domain-study.md#run-the-implemented-input-builders) for profiles, budgets, and pending execution phases.

| Scenario | Purpose |
| --- | --- |
| `teacher-host` | Start vLLM on loopback and the authenticated teacher API on the network; serve until interrupted |
| `distill` | Build tasks, collect budgeted responses, stop a locally hosted teacher, review usability, and train attacker and gold-reference adapters |
| `start` | Run the full study and benchmarks, then protected extraction, fresh student training, and another benchmark |

```bash
just distill
just distill --tasks ag_news,squad,gsm8k --seeds 42,1,20 --budgets full
just distill --teacher-url http://TEACHER_IP:8000
python3 scripts/pipeline.py distill --help
```

On Windows, use `py -3 scripts\pipeline.py`. The bootstrap accepts Python 3.9 or newer and installs the Python 3.14 workspace. Relative option paths resolve against the repository root. just forwards options unquoted, so values cannot contain spaces.

The launcher first loads `.env` and installs the workspace. After validating command options, it installs the training stack and, when needed, vLLM. It then invokes the task builder, collector, reviewer, preparation, and trainer as foreground subprocesses. `start` requires uv, directly or through devenv, to install the complete workspace. See [paper-tasks.md](paper-tasks.md#run-the-whole-study) for the task matrix and all options.

## Complete pipeline

Set `TEACHER_TERMS_REVIEW_ID` in `.env`, then run:

```bash
just start
# Smaller matrix, with the same stages:
just start --tasks ag_news --seeds 42 --budgets 500 --no-gold
# Existing teacher API or existing OpenAI-compatible inference server:
just start --teacher-url http://TEACHER_IP:8000
just start --upstream-url http://127.0.0.1:18011/v1
just start --help
```

Defaults match `distill`: AG News, SQuAD, and GSM8K; seeds 42, 1, and 20; `--budgets full`, resolved from each task's manifest. The pinned builds provide 53,893 AG News queries, 40,356 SQuAD queries, and 3,389 GSM8K queries. GSM8K supplies the math capability distillation corpus. Its 256-token cap is provisional and the cap-pilot command remains unimplemented. The pilot remains required before freezing that cap for final study evidence. Missing task builds are generated automatically. Explicit numeric budgets remain available, with values above a task's pool skipped. The defaults plan 9 undefended attacker adapters and 9 gold references, then up to 9 protected adapters for each defense, excluding development recipe trials and matched supervision controls. `--arms` selects the defenses, by default all three: `full`, `instantaneous`, and `quota`. Each defense collects its own campaign through a fresh service, with its own ledger, adapter, evaluation, and traffic replay. `--no-gold` omits the gold references. `--config` selects the shared training recipe. Both attacker conditions start independently from its base model and revision.

These are separate task studies, not multi-domain gateway results. `budgets.json` freezes the resolved numeric budgets per task and the final report uses that plan, including skipped oversized budgets. Reports for older `full` protocols can recover budgets from their frozen `tasks.json`. `policy-horizons.json` records the calibrated attempt and time horizons. Full-pool campaigns can cross several 10,000-attempt policy horizons. Policy state resets at those boundaries, while the collector retains one campaign-wide attempt limit and deadline. Interpret the resulting defense scores as campaigns spanning those resets.

The task list is frozen in `protocol.json`. To add GSM8K to a study directory previously frozen with only AG News and SQuAD, select a new `--run-dir`; the launcher rejects changes to an existing protocol.

The stages run in this order:

1. Download missing task datasets and build disjoint prompt pools and frozen tests. Reuse existing manifests and append missing baseline responses to `data/tasks/TASK/ledger.jsonl`.
2. Download and serve the teacher through vLLM and the teacher API, or connect to the supplied server. Collect each seed's largest budget, evaluate the teacher, and replay the frozen genuine/extraction traffic benchmark against the unprotected endpoint.
3. Stop managed teacher services, evaluate the unmodified student, review and prepare baseline ledgers, train attacker and optional gold-reference adapters, and benchmark the trained students.
4. Calibrate the harness on benign pools with the paper's declared accumulators only, write `TASK/calibration-report.json`, and replay `legitimate/benign_test.jsonl` through every defense for held-out account false-escalation rates. Restart the teacher and reevaluate it to measure drift. Start the authenticated protected API on loopback port 8003 alongside the teacher.
5. Collect the same largest budget through a fresh protected service instance for every task, seed, and defense. Solve proof-of-work challenges, log confirmed refusals, and save received text in a separate ledger. Replay the same traffic benchmark with independent authenticated actor accounts.
6. Stop managed services. Prepare each budget's received responses, review usability without gold answers, run training preflight, and train a fresh protected student. If no usable responses remain, skip training and record the reason.
7. Benchmark trained protected students on the same frozen tests and write capability reports, paired traffic reports, and completion status.

`distill benchmark run --slot SLOT` evaluates one model independently. The launcher separates teacher and student work to free managed teacher GPU memory. Identical teacher/base-student results are shared across budgets and conditions with their original IDs and timestamps. Every adapter gets its own evaluation. Capability reports use the unprotected teacher as the common reference. Traffic reports measure protected endpoint request success, refusals, throttling, latency, and challenge cost. Synthetic traffic never enters student training, and its HTTP success rate does not measure task-answer correctness.

L4 refuses by default. Use `--l4-action deceive` with a fresh `--run-dir` for the separate deception condition. Its ledger includes deceptive text received by the client. A 403 counts as refusal only with `X-Harness-Decision: block`. Failed proofs, server failures, and timeouts are charged to the budget as `failed` attempts, are never retried or replaced, and appear in `campaign.json` and the event log. Authentication errors, a 403 without a harness decision, or 10 consecutive failures stop collection.

Default calibration groups legitimate fitting, score-calibration, and policy-calibration pools into 500-request accounts after a seeded shuffle of each pool, so no account or window inherits the article order of the SQuAD source split. It uses a 100-request window and up to 20 reference windows as the fitting pool permits, recording the count, the truncation counts, and the skipped aggregate monitor in `TASK/calibration-report.json`. With 500-request accounts, the fitted quota is the account length, so the quota arm measures that construction rather than a fitted client limit until realistic accounts replace it. A run directory whose calibration was built from other benign accounts, including any calibrated before this shuffle, is refused rather than reused; select a fresh `--run-dir`. These artificial accounts exercise the pipeline and do not establish real-user false-positive rates. Supply independently prepared calibrations with `--calibration-root PATH`, containing `PATH/TASK/calibration.json` and its reference arrays for every selected task. Calibration and the BGE encoder use CPU.

`--run-dir` defaults to `artifacts/start`. Paths below are relative to it:

| Path | Contents |
| --- | --- |
| `protocol.json`, `budgets.json`, `tasks.json`, `policy-horizons.json` | Frozen options, resolved per-task budgets, task manifests, policy reset settings |
| `baseline-adapters/TASK/sSEED-bBUDGET/attacker` and `gold` | Baseline adapters |
| `TASK/calibration-report.json` | Reference windows, truncation, thresholds, quota, aggregate-monitor status |
| `TASK/benign-test/ARM.json`, `ARM.decisions.jsonl` | Held-out benign account escalation per defense |
| `TASK/sSEED-bBUDGET/protected/ARM/ledger.jsonl` | Received responses from the largest campaign against defense ARM |
| `TASK/sSEED-bBUDGET/protected/ARM/campaign.json` and `.events.jsonl` | Query counts, refusals, proof costs, outcomes |
| `TASK/sSEED-bBUDGET/protected/ARM/train.jsonl`, `adapter/`, `training-status.json` | Protected data, adapter, or skip reason |
| `TASK/sSEED-bBUDGET/benchmarks/`, `report.*.json` | Capability runs and reports; protected runs are `protected-ARM` |
| `TASK/sSEED/ARM/traffic-results/`, `traffic-report.json`, `decisions.jsonl` | Paired traffic results and the harness decision log |
| `TASK/sSEED/credentials.json`, `accounts.json` | Restricted client credentials and account mappings |

Rerunning preserves baseline ledger rows, reuses completed protected campaigns, and skips matching completed adapters. All planned baseline ledgers and their resume contracts are validated before loading the teacher. For damaged files, use the [ledger recovery command](distillation-pipeline.md#recover-a-damaged-collection-ledger) to back up the original, preserve valid responses, and recollect missing IDs. Changed training data, recipes, or frozen protocol options require a fresh `--run-dir`. Benchmarks run again, sharing reference inference within that invocation.

The protected service keeps account evidence in memory. Interrupted campaigns retain their ledgers, events, and spent costs in place. The collector refuses to restart them with a fresh allowance. Preserve those artifacts and explicitly declare a new campaign directory with fresh policy state if another campaign is needed.

The harness runs locally for this experiment and stops on completion or interruption. Only services started here are stopped. External teachers and upstream servers keep running, so use another GPU or stop an external local server before student work when memory is shared. Logs live in `artifacts/logs/`. `status.json` becomes `complete` only after reports are written. Outstanding paper experiments and analyses remain in [paper-tasks.md](paper-tasks.md#implementation-todo).

## Task-study operation

Copy `.env.example` to `.env` and set `TEACHER_TERMS_REVIEW_ID`. A local run starts the teacher only when planned prompts are missing from task ledgers. `--teacher-url` uses an existing repository teacher API and requires its server root, not its upstream OpenAI URL. `--upstream-url` uses an existing OpenAI-compatible model server while starting the repository API locally.

Each task writes `data/tasks/TASK/ledger.jsonl`; budgets and generated training configs live in `data/tasks/TASK/runs/sSEED-bBUDGET/`. Adapters live in `artifacts/tasks/TASK/sSEED-bBUDGET/attacker` and `gold`. The launcher queries the largest supported budget for each seed and reuses responses for smaller nested budgets. Task output limits are fixed at 16 tokens for AG News, 64 for SQuAD, and 256 for GSM8K.

Review checks response usability and training length without consulting gold labels. Gold completions train a separate reference adapter on the same prompts. The training recipe is `configs/train.task.toml`; generated copies set dataset, adapter directory, and training seed for each run.

Locally started services run in separate process groups and log under `artifacts/logs/`. The launcher stops them before training. Ctrl+C stops those services; during training the trainer saves a checkpoint before exiting. A later run resumes collection by example ID and training from checkpoints. Completed adapters must match the dataset hash and explicitly configured training settings before they are reused. Checkpoint resume also checks the trainer's fingerprint, including the chat template.

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
