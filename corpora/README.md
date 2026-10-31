# Task corpus builds

These are the exact local task builds used by the bounded gateway study ([v2](../readings/v2-2026-10-21-gateway-bounded-v1/) to [v5](../readings/v5-2026-10-30-undefended-a2-student/)), built on 2026-10-21 with seed 42. They are copies of `data/tasks/`, which is gitignored; every file except this README is stored in Git LFS. To reuse them, copy each directory to `data/tasks/` (`cp -a corpora/* data/tasks/`); the launchers reuse an existing build instead of rebuilding it.

Each build holds the role pools (`attacker.jsonl`, `legitimate/*.jsonl`), gold answers, legitimate-reference pools, the development and test suites, the independent external pool used by A7 (`external/`), and a `manifest.json` with source revisions, role counts, filters and file hashes. The GSM8K build also keeps its earlier separate-task ledgers and `runs/`. The coding build keeps its sandbox qualification records under `qualified/`, including the deduplication index `seen.sqlite`.

| Build | Source (revision) | License | Attacker | Fitting | Score cal. | Policy cal. | Benign test | Development | Test suite | Cap | External pool (license) | Size |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `ag_news-split-v2-s42` | `fancyzhx/ag_news` (`eb185aa`) | unknown | 35,944 | 6,012 | 14,337 | 40,748 | 10,950 | 11,848 | 7,600 | 16 | `SetFit/bbc-news` (unknown) | 163 MB |
| `squad-split-v2-s42` | `rajpurkar/squad` (`7b6d24c`) | CC-BY-SA-4.0 | 10,353 | 4,110 | 13,226 | 42,145 | 10,431 | 7,141 | 10,570 | 64 | `lucadiliello/newsqa` (unknown) | 555 MB |
| `gsm8k` | `openai/gsm8k` (`740312a`) | MIT | 3,389 | 1,328 | 682 | 669 | 659 | 746 | 1,319 | 256 | `MU-NLPC/Calc-math_qa` (Apache-2.0) | 37 MB |
| `coding-subset-50000-s42` | `nvidia/OpenCodeInstruct` (`8f3ba5b`) | CC-BY-4.0 | 22,368 | 9,186 | 4,522 | 4,500 | 4,520 | 4,904 | 539 | 512 | `bigcode/self-oss-instruct-sc2-exec-filter-50k` (ODC-BY) | 419 MB |

Counts are examples per role from each manifest; output caps are in tokens. The coding test suite is HumanEval+ and MBPP+ from EvalPlus (Apache-2.0). The coding build is a 50,000-example eligible subset: a seed-42 global shuffle of the 5,000,000 source rows was qualified in order until 50,000 passed sandbox checks, which took 235,024 rows; 185,024 were filtered (130,665 reference failures, 30,615 malformed or unsupported, 20,426 duplicates, 2,043 stub-accepting tests, 782 timeouts, 493 test overlaps). GSM8K's cap was later raised to 1,024 tokens in code; these builds keep the 256-token cap used by this study.

Manifest SHA-256:

| Build | `manifest.json` SHA-256 |
|---|---|
| `ag_news-split-v2-s42` | `38b4fc7617a87def54b9afbe25c37d08916d903cc4bc2dfd97969db5600e645e` |
| `squad-split-v2-s42` | `0856374c5c0ac3b360101ecaefb201e452c6c4452fa44d7044e03caffea16e74` |
| `gsm8k` | `496435b36020d273be8dee6921e7127995cbf9832fb4c2eb9da7ed867279e2a0` |
| `coding-subset-50000-s42` | `6423dad21eef86da3ad565d79900397969dc865622a7027a2a2a65c9e24b711a` |

## Licensing

The builds redistribute third-party text under each source's terms; attribution is the source and revision above. SQuAD-derived files are CC-BY-SA-4.0, so derivatives must keep that license. AG News, SetFit/bbc-news, and NewsQA declare no license on the Hugging Face Hub; their text belongs to the original publishers (NewsQA is built from CNN articles and is distributed for research use). They are included for research reproducibility only. Rebuild from the pinned sources with `distill tasks build` instead of reusing these copies where those terms do not allow redistribution.
