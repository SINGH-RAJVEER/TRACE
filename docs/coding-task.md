# Optional coding task

Coding is an optional extension to the study: implement a Python function from a specification and score one response against hidden tests with pass@1. Its inclusion must be frozen before final testing. No teacher–student transfer gate, detectability replay, calibration, training result or coding score is established by the dataset research below.

## Corpus policy

The default is the entire eligible corpus, with no implicit 8,000-example cap, difficulty quota or fixed-size random sample. Eligibility means a usable function contract and tests, successful sandbox validation of the reference, deduplication and decontamination against the frozen evaluation source. Preserve all source rows or their immutable provenance and record every exclusion with a reason. Do not present the eligible corpus as the entire unfiltered upstream dataset.

Allocate the complete eligible pool across the protocol's disjoint roles before constructing nested collection budgets. The largest attacker collection budget should consume the complete attacker allocation; gold comparisons use the corresponding matched examples. Reserved development, calibration and legitimate-user allocations stay disjoint. A user-specified pilot or budget is a labeled subset and never silently replaces the default corpus. An exhaustive corpus scan and reference validation can be expensive; a partial scan must carry an explicit incomplete status and cannot claim a final pool size.

## Dataset investigation

Research checked first-party dataset cards, metadata APIs and papers on 2026-10-18. Counts below are upstream declarations, not locally measured qualified counts. Only metadata was fetched; the multi-gigabyte corpora were not downloaded or validated.

| Candidate | Declared size | License | Function/tests support | Decision |
| --- | --- | --- | --- | --- |
| NVIDIA OpenCodeInstruct | 5,000,000 train rows in pinned metadata | CC BY 4.0 | Assertion-list tests; no dedicated signature field | Recommended pool source, subject to qualification |
| KodCode V1 | 484,097 train rows and 3,335 `use_with_caution` rows in pinned metadata | CC BY-NC 4.0 | Pytest tests and explicit function metadata; includes other task styles | Alternative with a noncommercial restriction; no automatic easy-only subset |

OpenCodeInstruct has the larger advertised pool and permits attribution-based reuse. Its metadata declares 50 Parquet shards and about 6.86 GB compressed. The live viewer's estimated count differs from the card's declared count; neither establishes the actual number of qualifying function problems. The converter must read every pinned shard and report observed counts. Its assertion lists and prior execution statuses are serialized strings; even `average_test_score` is declared as a string in the feature metadata, despite the prose describing it as a float. [NVIDIA dataset card](https://huggingface.co/datasets/nvidia/OpenCodeInstruct), [pinned metadata](https://huggingface.co/api/datasets/nvidia/OpenCodeInstruct/revision/8f3ba5bafe4d6e8db46082cf7ae6741bc370604d).

The OpenCodeInstruct paper describes generated assertions, previously executed solutions and an upstream decontamination pass. It also reports incorrect tests, unexecutable solutions and self-consistency bias. Upstream success flags therefore do not establish reference correctness. The paper's training results cannot establish that this project's 2B student will pass the transfer gate. Difficulty is not a verified property of the complete eligible corpus. [OpenCodeInstruct paper, sections 2.2–2.4 and 4](https://arxiv.org/html/2504.04030v1).

KodCode exposes `question_id`, `question`, `solution`, `test`, `test_info`, `subset`, `style`, `gpt_difficulty` and trial metadata. Its `test_info` supplies function name, declaration and parameters. The card describes a 43K Prefill subset of simple questions, but that rounded figure does not establish a deduplicated easy pool size. Its difficulty labels derive from GPT-4o trial pass rates. Excluding hard rows by default would violate the full-corpus requirement; an easy-only experiment needs a separate declared configuration. The train/caution split and noncommercial restriction make it less suitable than OpenCodeInstruct as the default permissive pool. [KodCode card](https://huggingface.co/datasets/KodCode/KodCode-V1), [pinned metadata](https://huggingface.co/api/datasets/KodCode/KodCode-V1/revision/97d81f0695ddd7c5c02bfe16436db9a17a4a21e5).

## Revision pins and conversion contract

| Source | Immutable revision | Configuration/split |
| --- | --- | --- |
| `nvidia/OpenCodeInstruct` | `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d` | `train` / `train` |
| `KodCode/KodCode-V1` | `97d81f0695ddd7c5c02bfe16436db9a17a4a21e5` | `default` / `train` |
| `evalplus/humanevalplus` | `d32357cf319e50e9c8d8dab5ea876c72b0fd321b` | `default` / `test` |
| `evalplus/mbppplus` | `b2d74c91837c3f2a20c1299ae98133cbe7cfa077` | `default` / `test` |

These SHA values were returned by the respective first-party Hugging Face APIs. Pin the revision in every loader and record downloaded file hashes and converted artifact SHA-256 hashes in the build manifest. A repository revision identifies the upstream snapshot; it does not replace a hash of the locally frozen suite.

For OpenCodeInstruct, preserve `id` as source provenance; map `input` to the problem specification, `output` to the reference response and `domain` to a development-round-robin category. Parse `unit_tests` as bounded literal data, never with `eval`, and require nonempty assertion statements. Infer the signature from the reference AST only when the function targeted by the tests is unambiguous; otherwise retain an exclusion record. Keep tests and reference responses out of model-facing prompts. Verify the reference against the converted tests inside the same sandbox used for scoring. Do not mistake an upstream all-pass score for a locally verified reference.

The qualification manifest must report raw rows read, malformed rows, unsupported contracts/dependencies, failing references, duplicates, test-overlap exclusions, final eligible count, counts by role, maximum attacker budget and every explicit cap. No post-filter count or role arithmetic has been measured yet; the original approximately 18K feasibility threshold is superseded by reporting the complete eligible allocation.

## External evaluation and decontamination

Recommend HumanEval+ as the first frozen external test source. The pinned Hugging Face snapshot declares 164 tasks with `task_id`, `prompt`, `canonical_solution`, `entry_point` and executable `test`. The MBPP+ snapshot declares 378 tasks with `code`, `prompt`, `test_imports`, `test_list` and `test`. Both cards declare Apache 2.0. Keep any derived suite's notices and upstream attribution. [HumanEval+ metadata](https://huggingface.co/api/datasets/evalplus/humanevalplus/revision/d32357cf319e50e9c8d8dab5ea876c72b0fd321b), [MBPP+ metadata](https://huggingface.co/api/datasets/evalplus/mbppplus/revision/b2d74c91837c3f2a20c1299ae98133cbe7cfa077).

The EvalPlus implementation has additional comparison semantics, including float tolerances and MBPP input reconstruction. A generic assertion runner must not claim equivalence to EvalPlus without preserving the pinned source's tests and oracle semantics. Name the actual frozen artifact and scorer. [HumanEval+ loader](https://github.com/evalplus/evalplus/blob/master/evalplus/data/humaneval.py), [MBPP+ loader](https://github.com/evalplus/evalplus/blob/master/evalplus/data/mbpp.py).

Before role assignment, compare every eligible pool item against every frozen evaluation item using normalized specification n-grams and canonicalized reference-code AST fingerprints. Freeze the normalization, thresholds and algorithm version with the manifest; record matched IDs and reasons. Deduplicate within the pool too, with a deterministic survivor rule. This catches detectable overlap, not semantic equivalence or pretraining exposure. Upstream decontamination is insufficient evidence that this project's exact frozen suite is clean. HumanEval and MBPP pretraining contamination remains a disclosed limitation.

## Scoring and agreement contract

One completion is scored once. `ok` carries a passed/failed result; `invalid`, `truncated` and `timeout` count as wrong in the pass@1 denominator. Infrastructure `error` is reported separately with its count and explicit retry policy. Never retry an item merely because its code timed out or failed tests. Record the total requested items, evaluated denominator, successes and each status count.

Functional agreement requires both responses to execute validly and produce matching observable outputs on the same frozen test inputs. Report matching pairs, valid-pair denominator and coverage against all requested pairs. Passing the same tests is weaker pass/fail agreement and must have its own label. Arbitrary assertion suites alone do not provide output traces; functional agreement remains pending until trace collection preserves the relevant test and comparison semantics.

## Remaining study gates

## Implemented package and commands

Phase 0 research and Phase 1 software live in the `coding` JJ workspace at `../capstone-coding`, based on committed `main`. Its `.venv` and `data/tasks/coding` artifacts are separate from the main checkout. No main-checkout ledger or in-progress change was copied. Install with `uv sync --extra coding`; ordinary `uv sync` leaves the optional package out. `uv sync --all-packages` explicitly installs every workspace member, including coding.

Enter `devenv shell` for Linux bubblewrap and util-linux `prlimit`. Score a saved response with a separate assertion file:

```bash
uv run --extra coding python -m coding.scoring --tests tests.py --response response.txt --finish-reason stop
uv run --extra coding pytest tests/test_coding.py
```

The default boundary uses user, PID, mount and network namespaces, clears inherited environment variables and capabilities, disables nested user namespaces and denies process creation and sockets using seccomp. Runtime directories are read-only and home directories are absent. `/tmp` is writable and capped at 16 MiB. Defaults are 2 CPU seconds, a 3-second wall clock, 256 MiB address space, 32 file descriptors, 1 MiB per output/file and disabled core dumps. CPU time and address space are bounded; this is not CPU affinity or a cgroup RSS guarantee. The runtime supports x86-64 and AArch64 syscall tables; runtime checks so far cover this x86-64 host. Missing sandbox facilities raise `SandboxUnavailable` without retry or fallback. Only the scorer's explicit `--unsafe-local-exec` flag bypasses namespaces and seccomp; do not use it on generated code. Resource limits still apply in that mode.

The runner accepts tests with at least one top-level assertion, so an uncalled test function is rejected. It extracts the first Python fence or parses the entire unfenced response. A normal exit before the completion marker fails. Tests run in the same interpreter as the candidate, so a deliberately adversarial candidate can inspect or tamper with judge code. Tests are hidden from the model request, not inaccessible to the running program. Kernel isolation protects the host; it does not make this a tamper-proof correctness judge. A separate, trusted input/output oracle is still required before adversarial code submissions become study inputs.

Prepare a pinned OpenCodeInstruct JSONL export against a separately frozen evaluation JSONL. Each evaluation row has `id`, `specification` and complete reference `solution` strings. Construct that normalization from the chosen pinned evaluation source; the command does not implement or claim the full EvalPlus scoring protocol.

```bash
uv run --extra coding python -m coding.dataset --input data/tasks/coding/source.jsonl --evaluation data/tasks/coding/evaluation.jsonl --output data/tasks/coding/qualified
```

There is no row-limit option or implicit sample. Preparation streams all input rows and uses SQLite for pool deduplication. It filters ambiguous function contracts, invalid assertions and source fields over 128,000 characters, then excludes exact specification overlap, normalized AST overlap and specification 13-gram containment of at least 70% against evaluation. AST matching preserves literals and operators while normalizing bound names and removing function docstrings. These heuristics can over-filter and miss semantic equivalents; inspect exclusion records before freezing. Each surviving reference must pass the sandbox. Unsupported dependencies and failing or timed-out references are excluded, with no retry.

`eligible.jsonl` contains solutions and tests and must stay private to preparation. `exclusions.jsonl` records source IDs, reasons and matched evaluation IDs. `manifest.json` records source revision, input and evaluation hashes, algorithm parameters, observed counts, an eligible-output hash and whether the scan completed. Revision identity is an operator declaration for the supplied export, not remotely attested. The output directory must be new. Interrupted outputs retain `complete=false` and are not resumable; do not mistake them for a completed corpus. Keep the raw pinned export so exclusions remain auditable. No multi-gigabyte corpus has been downloaded or qualified by this implementation task.

Integration with task role allocation, collection, training, the benchmark runner, functional-agreement traces, the harness, dashboard and manuscript remains Phase 2 onward. Existing AG News and SQuAD suite models and serialization were left unchanged. The staging converter exposes a lazy `TaskExample` conversion but does not export gold-blind attacker files yet.

## Remaining study gates

The 512-token output cap is provisional. A development-only pilot must report the number of completed responses and the fraction ending for length; raise the cap and rerun if that fraction exceeds 2%. Freeze the cap before final testing. The sequence-length report must separately show prompt and supervised-response tokens.

Sandbox validation, a teacher-versus-base development benchmark, the B=2,000 seed-42 recipe sweep where supported, the transfer gate and detectability replay are required before adding coding to the declared experiment matrix. Report unavailable GPU/provider results as pending. This document does not authorize or claim a final study freeze.
