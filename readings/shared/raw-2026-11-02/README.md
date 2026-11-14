# Raw run files, 2026-11-02

## Current checkout layout

The original inventory below describes the snapshot before reorganization.
Clearly attributable files now live under `v0`, `v1`, `v5` and `v6`; this folder
retains cross-version gateway files, shared logs and simulated demo outputs.
Use the [relocation map](../relocations.tsv) to translate old checkout paths.
`MANIFEST.tsv` is unchanged historical provenance, not a current path inventory.

| Original source | Current folder |
|---|---|
| `artifacts/start`, `artifacts/tasks`, `data/tasks` | [v0 raw files](../../v0/raw-2026-11-02/) |
| `artifacts/start-v2` | [v1 raw files](../../v1/raw-2026-11-02/) |
| `artifacts/undefended-a2-benchmark` | [v5 raw files](../../v5/raw-2026-11-02/) |
| `artifacts/gateway-v6` | [v6 raw files](../../v6/raw-2026-11-02/) |
| v2/v3 continuation, abandoned full-pool attempt, logs and demos | Remain here under their original paths |

## Original snapshot inventory

This directory holds every file from the lab machine's gitignored `artifacts/`, `data/` and `outputs/` directories that was not already committed to a branch of this repository on 2026-11-02. Each file keeps its original path, so `artifacts/start/...` here is the file that was at `artifacts/start/...` on the run machine. All files except this README and `MANIFEST.tsv` are stored in Git LFS.

`MANIFEST.tsv` lists all 1,585 files under those three directories, with each file's SHA-256, size, and where it is stored:

- 432 files are in this directory.
- 1,048 files were already committed with identical contents. The `stored_at` column gives the branch and path of one copy (`main`, `archive/main-before-harness-20261030` or `archive/original-study-20261031`).
- 105 files were not committed, for one of two reasons:
  - 98 intermediate training-checkpoint files (`checkpoint-*/optimizer.pt`, `scheduler.pt`, `rng_state.pth` and the intermediate `adapter_model.safetensors`, about 4.8 GB). They are only needed to resume training. Each checkpoint's `trainer_state.json` and `training_args.bin` are included, and so is every final adapter.
  - 7 `credentials.json` files holding the gateway's per-account API keys. Each run's `accounts.json`, which maps key digests to account names, is committed.

| Source directory | Run | What this directory adds |
|---|---|---|
| `artifacts/start` | v0 | evaluation configs, teacher/gold/protected benchmark results with per-item samples, harness traffic results |
| `artifacts/start-v2` | v1 | calibration references, workloads, benign-test decisions per defense, detectability results |
| `artifacts/gateway-bounded-v1` | v2/v3 | final adapters for the A1 undefended, gold, `full`, matched-volume and matched-prefix students and the A7 undefended student; the development sweep's training checkpoints metadata; A1 and A7 per-step trainer state; `benign-test/full.json` |
| `artifacts/gateway-full-s42-v1` | abandoned full-pool attempt | A1/A2/A5/A6/A7 full-pool request schedules (72,054 per attack), benign-test replay |
| `artifacts/gateway-v6` | v6, not started | A1–A7 and benign schedules, detector inputs, the GSM8K 1,024-token task build, runner log (the runner exited because the v3 run failed) |
| `artifacts/undefended-a2-benchmark` | v5 | training-checkpoint metadata |
| `artifacts/logs` | all | study, harness, teacher API and vLLM logs, launch and watcher scripts, exit codes |
| `artifacts/tasks`, `data/tasks/{ag_news,squad}` | v0 | early task builds, collection ledgers and the AG News/SQuAD `s42-b2000` training run inputs |
| `outputs/tasks-demo` | demo | `just benchmark-demo` output; simulated, not measured |
