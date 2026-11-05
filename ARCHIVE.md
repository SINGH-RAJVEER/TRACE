# Study archive

This branch merges the two earlier archive branches. Both are parents of the merge commit, so their full histories stay reachable. The tag `archive-20261102` marks the merge.

- `archive/original-study-20261031`: one commit made on 2026-10-31 at 06:32 UTC, 49 seconds before `main` removed the legacy study code in `50e30d5a`. It contributed `legacy/`, the original study's apps, configs, docs, artifacts, data, outputs and readings, plus the two `*-preservation-manifest.json.gz` files.
- `archive/main-before-harness-20261030`: `main` as of `c102078c` (2026-10-30, before the harness rework), plus the study data recorded afterwards. It contributed the top-level code, `readings/`, `corpora/` and `readings/raw-2026-11-02/`.

## Where to look

- `readings/`: the current run readings, v0 to v5, including the A7 results and the v3 student scores. Start at `readings/README.md`.
- `readings/raw-2026-11-02/`: every remaining raw file from the run machine. Its `MANIFEST.tsv` maps all 1,585 files under `artifacts/`, `data/` and `outputs/` to where each is stored.
- `legacy/readings/`: an older copy of the same readings, taken before the A2 attempt-4, A7 and raw-file commits. Where the two differ, `readings/` is current.
- `corpora/`: the four seed-42 task builds used by the bounded study.

Large files are in Git LFS, about 10 GB in total. To fetch only part of it, run `git lfs pull --include="readings/**"` after a clone made with `GIT_LFS_SKIP_SMUDGE=1`.

## Research work

The anti-distillation research and the tooling built for its plan were on `main` from `ba1ad344` until they were removed on 2026-11-05. The research head `480481e3` is a parent of this branch's merge, so its full history stays reachable. The merge replaced the top-level code described above with the code at that commit; the earlier top-level code is still at `313dcb8a` and the tag `archive-20261102`.

- `research/`: the report, notes and saved papers. Start at `research/ANTI_DISTILLATION_RESEARCH.md`.
- `TODO.md`: the open fixes and the phased research plan.
- `readings/v3/`, `readings/v4/`, `readings/v5-development-notes/`, `readings/index.tsv`: the reorganized readings that this code, the docs and the report link to. The dated `readings/` directories above are unchanged alongside them.
