# Run scripts

Coding is included in both launcher defaults. On Linux, `devenv shell` provides the sandbox tools. `uv run --extra eval --extra coding distill tasks build --task coding` qualifies a globally shuffled 50,000-eligible-example subset of the pinned source and freezes its role pools and unchanged external test suite. Use `--coding-pool-size 0` for the explicit full-source extension. `uv run --extra coding python -m coding.scoring` remains available for scoring saved responses. See [coding-task.md](coding-task.md).

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

Defaults are AG News, SQuAD, GSM8K, and coding; seed 42 (with every prompt sent, seeds only reorder the same inventory); all attacker pools (`--budgets full`), matching the whole-pool undefended collection; A1/A2/A5/A6/A7; and only the `full` harness. Pass `--arms full,instantaneous,quota` for the ablations. A5 has six total accounts. A7 uses independently pinned prompt pools, built on demand by `distill gateway external`; Add A3 cover traffic (in full mode, A7's inventory plus an equal cover share) or A4 slow collection (A1 at a declared `--slow-interval-seconds`) with `--attacks`; A8 active coverage is not implemented. Numeric budgets are total campaign allowances, balanced across domains. Full mode uses all attacker pools with an unequal allocation and one shared allowance equal to their sum. The 24-hour deadline can stop collection before the full inventory is submitted. Proofs and failures consume that same allowance across every policy reset.

`just start`, `just distill`, standalone task builds, and the low-level qualification CLI default to parallel coding qualification, using up to 16 available CPUs. Set `--coding-workers N` on either launcher, `--qualification-workers N` on `distill tasks build`, or `--workers N` on `python -m coding.dataset`. A value of 1 selects sequential execution. Qualification decisions and output rows remain in source order. Up to two missing task corpora build together, with coding scheduled first. Parallelism can reduce sandbox startup and execution time, but speedup depends on CPU, memory, source I/O, and reference runtimes; it is not guaranteed to be faster on every machine.

Coding builds default to `--coding-pool-size 50000`, an eligible target across all roles, with source selection seed 42 and the separate path `data/tasks/coding-subset-50000-s42`. The standalone builder uses `--seed` to choose its source permutation and role-assignment seed. Source indexing may download the full pinned dataset, but only sampled candidates needed for the eligible target are sandboxed. `selection.json` and qualification IDs/hashes disclose the scope. The full 539-item test suite is unchanged. `--coding-pool-size 0 --budgets full --run-dir artifacts/gateway-full-extension-v1` explicitly restores exhaustive qualification and full-pool collection. Existing completed full-source builds are retained; they are not relabeled as subsets. Default shared run roots are `artifacts/gateway-full-v1` and `artifacts/gateway-distill-full-v1`; the earlier bounded 6,000-submission runs stay in `artifacts/gateway-bounded-v1` and need `--budgets 6000 --run-dir` to reproduce. Shared benign workloads default to 125 accounts; calibration and development-transfer gates remain mandatory.

To build missing task pools without teacher collection, calibration, model loading, or training, run `just start --build-only --tasks coding,gsm8k`. This installs only dataset dependencies. Completed task builds are reused. An interrupted coding build is not resumable; inspect or archive its partial directory and remove it explicitly before restarting. To keep a Linux host awake during preparation:

```bash
systemd-inhibit --what=sleep --mode=block --why="Preparing full coding and GSM8K pools" just start --build-only --tasks coding,gsm8k --coding-workers 16
```

The runner builds missing corpora, calibrates one mixed account policy, collects undefended campaigns, selects a single shared recipe on development data, labels domains without qualified development transfer as uninformative and stops only when none qualifies, then collects protected campaigns through fresh harness instances. Both teacher evaluations run after protected collection; they do not bracket deployment in time. Managed inference services stop before student work. One adapter per campaign and condition trains on all domains' received responses and is evaluated separately on every frozen suite. Optional gold references and per-domain token-matched controls also train shared adapters. Held-out benign replay and live utility remain separate from attacker training.

`artifacts/gateway-bounded-v1` contains frozen protocol and source manifests, calibration, development selection, campaign ledgers and HTTP events, combined training datasets and adapters, per-domain evaluations, utility measurements, per-arm `campaign-timing.json`, and `results.json`. Run `uv run distill gateway power` before final testing to see which suites resolve the declared minimum gain. `just distill` performs undefended mixed collection and shared training only, under `artifacts/gateway-distill-bounded-v1`. It defaults to A1 and accepts `--attacks`. `start` accepts `--attacks`, `--arms`, `--interval-seconds`, `--deadline-seconds`, `--benign-accounts`, `--requests-per-account`, `--concentrated-share`, and `--development-budget`. See [the protocol](multi-domain-study.md#run-the-shared-pipeline) for exact accounting, controls, and validation limits.

GSM8K's 1,024-token and coding's 512-token caps remain provisional. Run `distill tasks pilot` against the actual teacher before freezing final study evidence. Coding requires Linux sandbox tools. Shared execution supports L4 refusal only. Existing completed campaigns are verified before reuse; interrupted campaigns cannot restart in place with fresh policy state. Use a new run directory to change the protocol. Externally managed teacher services remain the caller's responsibility during student training.

For the previous separate-task study, pass `--mode tasks` to `start` or `distill`. Its defaults and artifacts remain per task, including `artifacts/start`, numeric budget resolution, and policy-horizon reports. A single-domain study must select this mode:

```bash
just start --mode tasks --tasks ag_news --seeds 42 --budgets 500 --no-gold
just distill --mode tasks --tasks ag_news --seeds 42 --budgets 500,2000
```

## Shared study readiness and recovery

The 2026-10-21 code and fixture audit does not establish unattended readiness on a GPU host. The shared path has working sequencing and explicit calibration and transfer gates. The focused launcher, shared HTTP study, campaign accounting, transfer, preparation, review, training-configuration, external-source, teacher-client, and task checks passed, 128 tests total. A separate all-HTTP-500 probe reproduced a completed campaign with ten failed submissions and zero responses; the study now rejects such campaigns, as described below. No teacher weights, corpus qualification, GPU training, or full study were run for this audit.

The default matrix contains five campaign inventories, from five attacks and one seed, and five protected campaigns under the full harness. The earlier bounded matrix had 12 inventories and 36 protected campaigns. Empty datasets skip training. Both teacher evaluation slots run after protected collection for every inventory. Held-out benign utility runs once undefended and once for each protected campaign, giving six live replays of the benign-test workload by default, plus warmups. The attempt budget is per campaign and excludes those utility calls and the separate development campaign. The 24-hour deadline is also per collection, not a limit on the whole study.

Use a Linux host for the default coding sandbox, with bubblewrap, namespace support, seccomp, and the runtime tools described in [coding-task.md](coding-task.md). Use uv and Python 3.14 for the workspace, a separate supported Python for vLLM, a working NVIDIA driver, and a C compiler with libc headers. The default student recipe uses BF16 and unquantized LoRA; it needs a GPU that supports that precision and measured capacity for the configured batch and context lengths. A remote teacher removes local teacher inference, but student training and evaluation still run on this host. There is no automatic CPU or smaller-model fallback. Disk space, RAM, model access, CUDA operation, sandbox operation, and actual memory capacity must be checked on the target host; the launcher does not enforce all of them before expensive work.

Before final collection, pin `VLLM_SPEC`, set the teacher review reference and any required credentials, verify free ports, build the frozen task pools, inspect benign-role capacity and calibration adequacy, and pilot the provisional GSM8K and coding output caps. `--build-only` checks corpus preparation only and can still download and qualify substantial datasets. Training preflight validates tokenizer serialization and lengths after responses have been collected; it does not prove GPU capacity. Use a fresh `--run-dir` for changed protocol settings. Do not silently change precision, quantization, task scope, budgets, or training recipes after freezing a study.

| Failure or stop | Current behavior | Recovery or fallback |
| --- | --- | --- |
| Missing uv or dependency installation failure | `start` refuses pip; a failed install stops execution | Install uv or repair the selected environment and retry. vLLM is separate from the locked workspace and needs its own pinned version. |
| CUDA compiler absent for FlashInfer sampling | When `nvcc` is absent and the sampler setting is unset, managed vLLM gets `VLLM_USE_FLASHINFER_SAMPLER=0` | The PyTorch sampler avoids that JIT dependency. An explicit sampler setting is respected; this fallback does not replace the NVIDIA driver or the C compiler needed by other kernels. |
| Missing sandbox facilities or interrupted coding qualification | Qualification fails; partial builds are not resumable | Fix the Linux sandbox, archive the partial build, and explicitly remove that partial directory before rebuilding. Never bypass the sandbox for generated code. |
| Inadequate calibration or no qualified development-transfer domain | The study stops; individual unqualified domains remain labelled uninformative when another domain qualifies | Inspect `gateway/calibration-adequacy.json` or `recipe.json`. Revise development inputs or the declared workload in a new run directory. `just distill` is a separate undefended experiment, not a successful protected-study fallback. |
| Teacher startup failure, wrong token, or thinking trace | Startup and authenticated smoke checks stop execution; managed services are cleaned up | Inspect `artifacts/logs/`, correct the server configuration, or explicitly use `--teacher-url` for a repository teacher API or `--upstream-url` for an OpenAI-compatible server. Changing the frozen endpoint needs a new run directory. |
| Transient HTTP failure or proof failure during a campaign | The submission spends allowance; collection moves on without retrying that prompt | Inspect campaign events and failure counts. There is no free retry or restored allowance. A transport failure is not a harness refusal. |
| Ten consecutive service failures for an account | That account stops with `service_failures`. The campaign's `campaign.json` is still written as evidence, but shared campaigns and tasks-mode protected campaigns then stop the study before training, so a transport failure cannot pass as a harness refusal | Fix the service and use a fresh run directory. Fewer than ten consecutive failures still spend allowance without stopping the study, so inspect each campaign's `failed` count before interpreting results. |
| Interrupted shared campaign or protected utility after collection | Shared campaigns cannot resume with fresh policy state; completed campaigns are verified before reuse, and missing protected utility forces a fresh run | Preserve all artifacts and use a new `--run-dir`. Do not delete spent campaign evidence to restart in place. Collection by example ID belongs to legacy tasks mode. |
| Training interruption after all required collections and protected utility completed | The trainer can save a checkpoint; completed adapter reuse checks the training fingerprint and dataset hash | Rerun the identical command and configuration to reuse verified completed campaigns and resume training. Keep endpoints and credentials stable because collection contracts bind them. SIGKILL, power loss, or disk exhaustion can prevent the latest checkpoint from being saved. |
| GPU memory or unsupported precision error | With `activation_offloading="auto"`, the trainer retries a CUDA out-of-memory error once with activation offloading, resuming the last checkpoint when available. Unsupported precision and OOM after offloading stop the subprocess | Measure capacity on the target host. Adjust serving or training settings deliberately, maintain the paired evaluation configuration, and start a new study if frozen settings change. Offloading does not reduce evaluation memory. Externally managed teacher processes must be stopped or isolated by their operator during student work. |
| Nondefault coding pool with A7 and no prebuilt external pool | The external CLI resolves the default 50,000-example coding directory, while the study expects its configured coding directory | This is an unresolved path mismatch for `--coding-pool-size 0` and other nondefault sizes. Repair the external builder's directory selection before running that extension. The default 50,000 path agrees. |

`status.json` records running, failed, interrupted, or complete, and `complete` excludes campaigns stopped by repeated service failures but not isolated transport failures. Calibration gate details remain in the adequacy report even when the outer shared runner replaces the status with generic `failed`. Treat `results.json` as evidence only after checking the per-campaign failure, deadline, response, and termination fields as well as qualification and calibration reports.

## Task-study operation

This section describes the legacy `--mode tasks` workflow.

Copy `.env.example` to `.env` and set `TEACHER_TERMS_REVIEW_ID`. A local run starts the teacher only when planned prompts are missing from task ledgers. `--teacher-url` uses an existing repository teacher API and requires its server root, not its upstream OpenAI URL. `--upstream-url` uses an existing OpenAI-compatible model server while starting the repository API locally.

Each task writes `data/tasks/TASK/ledger.jsonl`; budgets and generated training configs live in `data/tasks/TASK/runs/sSEED-bBUDGET/`. Adapters live in `artifacts/tasks/TASK/sSEED-bBUDGET/attacker` and `gold`. The launcher queries the largest supported budget for each seed and reuses responses for smaller nested budgets. Task output limits are fixed at 16 tokens for AG News, 64 for SQuAD, and 1,024 for GSM8K.

Review checks response usability and training length without consulting gold labels. Gold completions train a separate reference adapter on the same prompts. The training recipe is `configs/train.task.toml`; generated copies set dataset, adapter directory, and training seed for each run.

Locally started services run in separate process groups and log under `artifacts/logs/`. The launcher stops them before training. Ctrl+C stops those services; during training the trainer saves a checkpoint before exiting. A later run resumes collection by example ID and training from checkpoints. Completed adapters must match the dataset hash and explicitly configured training settings before they are reused. Checkpoint resume also checks the trainer's fingerprint, including the chat template.

After calibration and the held-out benign replay, `start` replays one budget's extraction prompts as a single account under every defense and writes `artifacts/start/TASK/detectability.json`. It reports the extractor's peak level and window p-values next to the held-out benign accounts, broken down by client type. If no MMD defense escalates the extractor to a level reached by fewer than 10% of the benign horizons that complete a 100-request window (the first level target), `start` stops before collection with status `detectability_gate_failed`. Pass `--allow-undetected` to continue and report the study as a quota comparison; `detectability-gate.json` records that choice.

Benign clients are synthetic. Specialist and repetitive clients send concentrated windows that score far above broad windows, so their share sets the upper tail of the benign null and with it the extractor's p-values. `--concentrated-share` sets that share of benign source groups (default 0.1; 0.4 reproduces the equal five-way split of `task-clients-v1`). Changing it changes every calibration hash, so use a fresh `--run-dir`. Run the detectability check on its own, before the expensive matrix, across several shares:

```bash
just diagnose --tasks ag_news,squad --concentrated-shares 0,0.1,0.4
```

Each share gets its own calibration under `artifacts/task-diagnostic/concentrated-SHARE`, and `artifacts/task-diagnostic/detectability-sweep.json` tabulates the extractor's peak level against benign escalation rates for every share and defense.

`artifacts/start/study-report.json` joins everything the run produces. Each per-seed row carries the capability effect, both token-matched controls (`controls`: matched tokens, control score, and the paired defended-minus-control difference; a control with no usable examples is marked `no_usable_examples`), and `escalation`, which holds J and Q per level over that budget's prefix of the largest campaign. `benign_utility` gives served-only quality and unconditional utility for each task, seed, and defense, through the bare teacher and then through the defense, with their changes. `benign_false_escalation` gives account-horizon escalation counts and rates from the held-out benign replay, including unscored horizons. These replayed rates check the pipeline on unseen requests. They are not real-user false-positive rates.

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
