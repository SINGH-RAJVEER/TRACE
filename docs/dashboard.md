# Results dashboard

The dashboard is a local, read-only view of the artifacts that studies, gates and student experiments have already written. It reads `readings/index.tsv` and recognized JSON reports under configured roots. It never trains, evaluates, collects, exports or contacts a teacher, so starting it costs no GPU time or teacher calls.

## Start it

```bash
devenv shell
just setup
just dashboard
```

Open `http://127.0.0.1:8002`. On first use the command installs the frontend from `apps/dashboard/bun.lock`; it rebuilds `apps/dashboard/dist` when the frontend sources are newer than the build. `just dashboard --help` prints the settings.

| Setting | Default | Meaning |
|---|---|---|
| `DASHBOARD_ROOTS` | `readings,artifacts` | Comma-separated roots, as `path` or `name=path`. Relative paths resolve from the repository root. |
| `DASHBOARD_INDEX` | `readings/index.tsv` | The results index |
| `DASHBOARD_HOST` | `127.0.0.1` | Bind address. Keep it local: run directories hold account and credential files. |
| `DASHBOARD_PORT` | `8002` | Port. The teacher uses 8000, vLLM 8001 and the harness 8003. |
| `DASHBOARD_FRONTEND` | `apps/dashboard/dist` | Built page served at `/` |

For frontend work, run `just dashboard` (or `uv run --frozen dashboard-api`) for the API, then `bun run dev` in `apps/dashboard`. Vite proxies `/api` to `DASHBOARD_API`, default `http://127.0.0.1:8002`. `just test-dashboard` runs the API tests, the TypeScript check and the frontend data tests.

## What it reads

Runs come from two places. Every `run_id` in the index is a run. A directory under a root also becomes a run when it holds one of these, recognized by content rather than filename:

| Artifact | Recognized by | Shown in |
|---|---|---|
| Study run directory | `steps.tsv`, or `exit` beside `campaigns/` or `utility/` | Provenance, acquisition, utility |
| Headroom and clean-student gates | `"version": "clean-baseline-gate-v1"`; the `gate` field tells them apart | Teacher, gates |
| Teacher capacity preflight | `"version": "teacher-capacity-v1"` | Teacher |
| Problem score files | `"version": "problem-scores-v1"` | Teacher |
| Learning curve | `plan.json` with `math-learning-curve-v1` or `ag-news-learning-curve-v1`, plus `summary.json` | Students |
| Chronological checkpoints | `plan.json` and `summary.json` with `checkpoint-learning-v1` | Students |
| Transfer report | A saved `distill learning-curve transfer` output, which starts with `"evidence"` | Students |

An indexed run joins a directory whose name matches its run ID or its reading, so `v5-20261102` shows `readings/v5`. Artifacts nested inside a run directory belong to that run; a `just baseline-gates` run directory therefore shows its gates, score files and curve together. The walk is bounded in depth and skips `campaigns`, `utility`, `logs`, `steps`, `scores`, `schedules`, `calibration`, `benign`, `provenance` and similar bulky subtrees, which the run adapters read directly.

When the index already holds a run's campaign and utility rows, they are shown as indexed. A run directory that has not been exported yet is read through the same `results_index.campaign_row` and `utility_rows` functions the export uses, and is labelled as read from the directory.

## How values are shown

- An empty index cell or a missing JSON field is "not recorded", drawn with a hatched mark. Zero and `false` are shown as values. Booleans keep their recorded value, so the v3 A2 campaign shows incomplete attribution with 1,537 unattributed queries.
- A section without an artifact says it is unavailable and why. The checked-in readings have no student, gate, checkpoint, transfer or capacity artifacts, so those sections are unavailable for v3, v4 and v5.
- Gate-refused and interrupted campaign arms keep their rows, with the runner's `outcome` text. Failed and development-only gates are listed with their reasons. A gate report whose digest no longer matches is flagged as edited; a learning-curve plan whose files do not match their hashes is flagged.
- Campaign acquisition (requests received, blocked and challenged, released tokens, distinct examples, intervention times and `J`/`Q`) is a separate tab from model accuracy. Neither is presented as evidence that distillation was prevented.
- Legitimate utility shows each arm's pooled completion, served-only quality, unconditional utility and latency, then the same per domain, domain profile and client type. Validity comes from the index or from `utility_validity`; unbound runs read unknown.
- Teacher headroom (task capability against the pinned base) and the capacity preflight (whether the teacher could serve the workload) are separate sections.
- A curve is charted only for an arm with at least two summarized sizes, with its bootstrap interval. Checkpoint charts plot reached checkpoints only; censored ones are listed and never drawn as zero accuracy.
- Study labels (development or confirmatory), seeds, attacker partition, commits, suite hashes, the student base and revision and the teacher model and revision are shown wherever an artifact records them. Teacher identity is copied from the run's configuration. A GGUF or other quantized teacher appears under the name and revision it was configured with and is not merged with other teachers.

## Safety

The API has only GET routes. Each request rescans the roots, so new artifacts appear without a restart. Runs are addressed by opaque IDs; clients never send paths. The walk does not enter symlinked directories, and every file read refuses a path that passes through a symlink, so a link cannot reach outside a root or rename a credential file. `Roots.resolve`, for root-relative paths, rejects absolute paths, `..`, backslashes and symlinks that leave a root. Files named `credentials.json`, `accounts.json`, `environment.env`, `.env`, `*.key` and SQLite state are never opened. Configuration values whose names contain TOKEN, KEY, SECRET, PASSWORD, CREDENTIAL, AUTH or COOKIE are replaced with `<redacted>`, as is the user-info part of URLs. Git LFS pointers are reported as unhydrated rather than malformed. JSON over 64 MB is not read, and the per-request `results` list in `utility.json` is never sent to the browser.

## Limits

The dashboard shows what artifacts record; it does not recompute statistics, pool runs or compare arms. Use `distill gateway comparison-report` for declared comparisons. Run directories are matched by name, so two unrelated directories that share a run's name both attach to it. Transfer reports are only found when saved to a file. Index `gate:` and `learning-curve:` rows appear as their own runs unless a curve directory has the same name.
