# TRACE teacher API campaign protection

TRACE places authenticated campaign monitoring and access controls in front of a teacher API. Its persistent campaign detector is experimental: it can observe collection patterns and apply explicitly enabled diagnostic throttling, but real-user false-alert rates and reduced student capability remain unqualified. The workspace also retains the evaluation workflows for measuring acquisition, detection timing and legitimate-user cost.

The supported domains are AG News, SQuAD, GSM8K, and coding. Campaign schedules cover multiplexing, sequential domains, cover traffic, slow collection, account splitting, and independent sources. The default campaign budget is 6,000 HTTP attempts, split evenly across domains.

## Setup

Enter `devenv shell`, copy `.env.example` to `.env` and run `just setup`. Configure the upstream teacher, authenticated account mapping and calibration matching `HARNESS_TASK`. Run `just setup-eval` when using encoder/calibration or student-evaluation tools.

**vLLM is the only supported teacher inference runtime.** Install it in a separate Python 3.12 environment at `.venv-vllm` (or set `VLLM_BIN`), then run `just teacher-vllm` on suitable hardware. Run `just teacher` separately for TRACE's HTTP adapter; it does not load the model. There is no GGUF/`llama-server` fallback. See [deployment settings](docs/evaluation.md#prepare-and-deploy).

Per-principal output watermarking is enabled by default. On first configured startup, the teacher generates a private key at `artifacts/teacher.watermark.key` and reuses it on restarts. Override the path with `TEACHER_WATERMARK_KEY_FILE`; `watermark.key` and `*.watermark.key` are gitignored. Back up the key securely for historical detection. Set `HARNESS_WATERMARK=false` explicitly to opt out. Existing state databases from unwatermarked runs need replacing, because watermarking changes the policy revision. See [watermark setup and limitations](docs/watermark.md).

## Run

Create a fresh campaign configuration, configure its durable state and validate the launch:

```bash
just campaign-config artifacts/campaign/config.json --namespace my-deployment
just validate --config artifacts/campaign/config.json --state-db artifacts/campaign/state.sqlite3
just start --config artifacts/campaign/config.json --state-db artifacts/campaign/state.sqlite3
```

The account mapping and calibration must already exist; `validate` checks settings and prerequisite paths without starting models or contacting the teacher. `just start` serves the protected API against the configured teacher URL. It defaults to base policy `allow` with campaign observation, and never launches vLLM, trains students or exports readings. Run `just teacher` separately for the teacher adapter. `just harness` aliases `start`; use `just harness-legacy` for the configured base policy without campaign detection.

The retained end-to-end research run is now `just study [RUN_ID]`. It may launch GPU services and export readings, and explicitly disables campaign detection in its historical comparison arms. `just baseline-gates` remains the separate GPU headroom/clean-student workflow. Use `just checkpoint-export`, `checkpoint-plan`, `checkpoint-execute` and `checkpoint-summary` for chronological learning experiments; GPU execution is explicit.

See [campaign setup and the command reference](docs/campaign.md#task-runner-flow), [the evaluation workflow](docs/evaluation.md) and [checkpoint evaluation](docs/checkpoints.md). `just --list`, `just distill --help` and `just detector --help` list supported commands. `just test-campaign` runs the detector fixtures, while `just check` covers lint, formatting, typing and shell syntax.

Existing observations and their limitations are in [readings](readings/v4/README.md). `just dashboard` serves a local, read-only view of the results index and recorded gate, curve, checkpoint and utility artifacts at `http://127.0.0.1:8002`; see [the dashboard guide](docs/dashboard.md). The harness limits what an account can receive; that is not evidence that it prevents distillation. See [what budgets establish](docs/harness.md#what-budgets-establish).
