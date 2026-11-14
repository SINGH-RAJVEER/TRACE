# Legacy artifacts and raw data

This is a byte-preserving snapshot from archive commit
`443e5d63baaec3dcbb40d8a0018b3f221e527b2d`, retained on `main` before removing the
other branches. Paths beneath this folder keep their original archive layout.

| Path | Preserved material |
|---|---|
| `legacy/artifacts/` | All legacy study artifacts, including cached outputs and recorded runtime environments |
| `legacy/data/` | Original task data, collection ledgers and training inputs |
| `legacy/outputs/` | Benchmark outputs, including simulated/demo outputs |
| `artifacts/` | Distillation workspace backup and restoration record |
| `corpora/` | The four historical task corpus builds |
| `*-preservation-manifest.json.gz` | Both original preservation manifests |

All 34,829 source entries are listed in `MANIFEST.tsv`, with source paths, Git
modes, payload sizes and SHA-256 digests. Symlink digests cover the link target,
not the file it points to. Existing LFS objects remain in LFS; source Git files
are all smaller than 1 MiB and remain directly in Git. The original archive's
`.gitattributes` is retained here to preserve that per-file storage policy and
byte handling. The generated manifest is stored in LFS.

The older readings are already preserved under `readings/`, including
`readings/legacy-20261031/`. Legacy application code, research notes and paper
drafts outside these data paths were not imported by this snapshot.

## Verify

From the repository root:

```sh
git lfs fetch --include="archives/legacy-study-20261031/**" origin main
python3 archives/legacy-study-20261031/verify.py
```

The verifier accepts materialized files or LFS pointers with their payloads in
the local LFS cache. To materialize the archive, run
`git lfs pull --include="archives/legacy-study-20261031/**"`.

These are inert historical records, not a supported runtime or an additional
Python workspace. Do not activate the archived environments or execute their
binaries: recorded symlinks and launch scripts may refer to old machine paths.
Demo outputs are not measurements. No credentials omitted by the original
archive have been reconstructed.
