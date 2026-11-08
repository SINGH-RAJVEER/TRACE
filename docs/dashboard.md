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

## Layout

**All runs** has two parts:

- **What each run recorded**: one row per run, one column per evidence group (Acquisition, Utility, Teacher, Students, Gates). Each cell is recorded, partial, missing, error or n/a; hover it for the artifacts behind it. N/a marks evidence that cannot exist for that entry, such as campaigns for a lone student plan. The last column counts the run's gaps. Student plans whose names match no run ID or reading are listed separately as not attached.
- **Released tokens by attack and arm**: every attack named by any run against each run's arms. A cell holds the released tokens, or is hatched (refused by the detectability gate), tinted (interrupted) or outlined (no row).

A run page (`#run=<id>`) is one short screen:

- **Headline figures**: arms that ran, the lowest and highest benign completion, the best student accuracy against its target, and how many evidence kinds were recorded. None of them sums acquisition across arms.
- **Missing**: one line per gap, for example refused or interrupted arms, unknown attribution, attacks other runs recorded, a `utility.json` that is an LFS pointer, unknown validity, plans without a summary, an unattached plan or no commit. Artifacts not found under the roots are folded into one "Not recorded" line. Past five lines the rest sit behind a toggle.
- **Released tokens per attack**: grouped bars per arm on a log axis, so small values stay visible. An arm the detectability gate refused is drawn as a gray dashed bar marked with ≈, which is an assumption and not a measurement (see below).
- **Minutes until the policy first intervened**: the earliest alert, restriction or refusal per arm.
- **Benign workload served**: completion, quality when served and overall utility per arm on a 0 to 100% axis, with p95 latency below.
- **Student accuracy**: each checkpoint on a 0 to 100% axis, with the untrained base, the teacher and the declared target marked. A learning curve is added when a plan has one.
- Gate verdicts (headroom, capacity, clean student) appear as badges when recorded.
- **Recorded values and run configuration**, collapsed: the campaign and utility tables, any transfer reports and student index rows, and the provenance (index labels, run directories, teacher identity, settings, README).

A chart or figure with nothing behind it is not drawn; the gap appears in Missing instead.

The page uses a dark theme only. Each arm keeps one color in every run: full is blue, instantaneous orange, quota green and undefended gray.

## How values are shown

- An empty index cell or a missing JSON field is "not recorded", drawn with a hatched mark, including in charts: a bar that was not recorded is a hatched stub, never a zero-length one. An arm that ran and saw no intervention is an outlined stub marked "none", and only when every query was attributed; otherwise it is hatched as unknown. Zero and `false` are shown as values. Booleans keep their recorded value, so the v3 A2 campaign shows incomplete attribution with 1,537 unattributed queries.
- Evidence without an artifact is listed under Missing rather than drawn as an empty section.
- The detectability gate refuses an arm when the detector does not separate the attacker from held-out benign accounts, so the arm would have let practically all of its traffic through. Its released tokens are therefore drawn as an assumed undefended value: the run's own undefended arm for that attack when there is one, otherwise the attempt budget times the tokens per answered response of the same attack in the same run (from the arm with the most answered responses). The estimate is never taken from another run, is shown gray with ≈ in the run chart and the All runs grid, and its basis is in the tooltip. An arm with no basis for an estimate stays hatched. Tables still show these arms as not recorded.
- Gate-refused and interrupted campaign arms keep their rows, with the runner's `outcome` text. Failed and development-only gates are listed with their reasons. A gate report whose digest no longer matches is flagged as edited; a learning-curve plan whose files do not match their hashes is flagged.
- Campaign acquisition (requests received, blocked and challenged, released tokens, distinct examples, intervention times and `J`/`Q`) is charted separately from model accuracy. Neither is presented as evidence that distillation was prevented.
- Legitimate utility shows each arm's pooled completion, served-only quality, unconditional utility and p95 latency. Validity comes from the index or from `utility_validity`; unbound runs read unknown.
- Teacher headroom (task capability against the pinned base) and the capacity preflight (whether the teacher could serve the workload) are separate verdicts.
- A curve is charted only for an arm with at least two summarized sizes, with its bootstrap interval. Censored checkpoints are drawn hatched, never as zero accuracy.
- Study labels (development or confirmatory), seeds, attacker partition, commits, suite hashes, the student base and revision and the teacher model and revision are shown wherever an artifact records them. Teacher identity is copied from the run's configuration. A GGUF or other quantized teacher appears under the name and revision it was configured with and is not merged with other teachers.

## Safety

The API has only GET routes. Each request rescans the roots, so new artifacts appear without a restart. Runs are addressed by opaque IDs; clients never send paths. The walk does not enter symlinked directories, and every file read refuses a path that passes through a symlink, so a link cannot reach outside a root or rename a credential file. `Roots.resolve`, for root-relative paths, rejects absolute paths, `..`, backslashes and symlinks that leave a root. Files named `credentials.json`, `accounts.json`, `environment.env`, `.env`, `*.key` and SQLite state are never opened. Configuration values whose names contain TOKEN, KEY, SECRET, PASSWORD, CREDENTIAL, AUTH or COOKIE are replaced with `<redacted>`, as is the user-info part of URLs. Git LFS pointers are reported as unhydrated rather than malformed. JSON over 64 MB is not read, and the per-request `results` list in `utility.json` is never sent to the browser.

## Limits

The dashboard shows what artifacts record; it does not recompute statistics, pool runs or compare arms. Use `distill gateway comparison-report` for declared comparisons. Run directories are matched by name, so two unrelated directories that share a run's name both attach to it. Transfer reports are only found when saved to a file. Index `gate:` and `learning-curve:` rows appear as their own runs unless a curve directory has the same name.
