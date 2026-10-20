# Coding task

Coding is included by default in `just start`, `just distill`, and the gateway input generators: implement a Python function from a specification and score one response against hidden tests with pass@1. Its inclusion must be frozen before final testing. No teacher–student transfer gate, detectability replay, calibration, training result or coding score is established by the dataset research below.

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

Functional agreement requires both responses to execute validly and produce matching observable outputs on the same frozen test inputs. Report matching pairs, valid-pair denominator and coverage against all requested pairs. Passing the same tests is weaker pass/fail agreement and must have its own label. The evaluator also replays assertions in an independent sandbox, records ordered typed input/output hashes, and reports agreement only for valid pairs with matching input hashes and call counts. It reports numerator, denominator, and coverage. Shared wrong outputs can agree. Trace equality is exact, including floats, and is distinct from the correctness oracle's tolerances. Unsupported values or execution failures yield no valid trace.

## Implemented package and commands

The coding package is a normal dependency of the distillation workspace: plain `uv sync` installs it. No coding extra is required for the default task list or coding evaluation. The `coding` extra remains available for compatibility with older commands. Both launchers default to `ag_news,squad,gsm8k,coding` and use each task's complete attacker allocation for `--budgets full`. Missing coding builds are created through `distill tasks build --task coding`. The builder streams every pinned OpenCodeInstruct row, validates the external references, qualifies candidate references, decontaminates, and exports the existing study files. It keeps preparation memory bounded with SQLite and streams prompt and legitimate-reference exports. The development suite alone has a selectable size limit, default 1,000; deterministic hash selection does not cap the role pool or attacker budget.

```bash
uv run --extra eval --extra coding distill tasks build --task coding
just distill
just start
# Explicitly select only the coding task:
just start --tasks coding --run-dir artifacts/start-coding
```

Enter `devenv shell` for Linux bubblewrap and util-linux `prlimit`. Score a saved response with a separate assertion file:

```bash
uv run --extra coding python -m coding.scoring --tests tests.py --response response.txt --finish-reason stop
uv run --extra coding pytest tests/test_coding.py
```

The default boundary uses user, PID, mount and network namespaces, clears inherited environment variables and capabilities, disables nested user namespaces and denies process creation and sockets using seccomp. Runtime directories are read-only and home directories are absent. `/tmp` is writable and capped at 16 MiB. Defaults are a 15-second CPU soft limit with a 16-second hard limit, a 20-second wall clock, 256 MiB address space, 32 file descriptors, 1 MiB per output/file and disabled core dumps. CPU time and address space are bounded; this is not CPU affinity or a cgroup RSS guarantee. The runtime supports x86-64 and AArch64 syscall tables; runtime checks so far cover this x86-64 host. Missing sandbox facilities raise `SandboxUnavailable` without retry or fallback. Only the scorer's explicit `--unsafe-local-exec` flag bypasses namespaces and seccomp; do not use it on generated code. Resource limits still apply in that mode.

The runner accepts tests with at least one top-level assertion, so an uncalled test function is rejected. It extracts the first Python fence or parses the entire unfenced response. A normal exit before the completion marker fails. Tests run in the same interpreter as the candidate, so a deliberately adversarial candidate can inspect or tamper with judge code. Tests are hidden from the model request, not inaccessible to the running program. Kernel isolation protects the host; it does not make this a tamper-proof correctness judge. A separate, trusted input/output oracle is still required before adversarial code submissions become study inputs.

Prepare a pinned OpenCodeInstruct JSONL export against a separately frozen evaluation JSONL. Each evaluation row has `id`, `specification` and complete reference `solution` strings. Construct that normalization from the chosen pinned evaluation source; the command does not implement or claim the full EvalPlus scoring protocol.

```bash
uv run --extra coding python -m coding.dataset --input data/tasks/coding/source.jsonl --evaluation data/tasks/coding/evaluation.jsonl --output data/tasks/coding/qualified
```

There is no row-limit option or implicit sample. Preparation streams all input rows and uses SQLite for pool deduplication. It filters ambiguous function contracts, invalid assertions and source fields over 128,000 characters, then excludes exact specification overlap, normalized AST overlap and specification 13-gram containment of at least 70% against evaluation. AST matching preserves literals and operators while normalizing bound names and removing function docstrings. These heuristics can over-filter and miss semantic equivalents; inspect exclusion records before freezing. Each surviving reference must pass the sandbox. Unsupported dependencies and failing or timed-out references are excluded, with no retry.

`eligible.jsonl` contains solutions and tests and must stay private to preparation. `exclusions.jsonl` records source IDs, reasons and matched evaluation IDs. `manifest.json` records source revision, input and evaluation hashes, algorithm parameters, observed counts, an eligible-output hash and whether the scan completed. Revision identity is an operator declaration for the supplied export, not remotely attested. The output directory must be new. Interrupted outputs retain `complete=false` and are not resumable; do not mistake them for a completed corpus. Keep the raw pinned export so exclusions remain auditable. No multi-gigabyte corpus has been downloaded or qualified by this implementation task.

The default builder allocates all qualified examples by source-group hash into development, attacker, fitting, score-calibration, policy-calibration, and benign-test roles. It writes `attacker.jsonl`, `gold-answers.jsonl`, `legitimate/*.jsonl`, private `legitimate-references.json`, and frozen development/test suites. Attacker prompts contain the specification and signature, never the reference code or tests. The full-corpus qualification directory remains private preparation input. Build status remains incomplete after interruption; choose a fresh directory instead of silently accepting partial pools.

All coding qualification entry points default to up to 16 available CPUs. Both `just start` and `just distill` accept `--coding-workers N`; standalone `distill tasks build` accepts `--qualification-workers N`, and `python -m coding.dataset` accepts `--workers N`. Set the value to 1 for sequential execution. Each reference still runs in its own isolated process with the same resource limits; eligibility, deduplication, exclusions, and output writes commit in immutable source order. The first eligible occurrence survives even if later reference runs finish sooner. `build-status.json` records the worker count. Native fixture tests compare the sequential and parallel `eligible.jsonl` and `exclusions.jsonl` byte for byte, including failed references before duplicates. Parallel speedup depends on the host and workload. `just start --build-only --tasks coding,gsm8k` prepares both missing pools together and exits before model or study work; see [run scripts](run-scripts.md#complete-pipeline) for the suspend-blocked invocation. No corpus row cap is introduced.

The standard collection, gold-blind usability review, response-token SFT preparation, training configuration, development recipe selection, protected collection, controls, and reporting paths now accept coding. The benchmark runner and benign utility call the same sandboxed scorer. Candidate timeouts and failed assertions score zero; missing isolation or NumPy infrastructure stops evaluation before inference. The dashboard labels coding as pass@1. Code agreement uses independently replayed input/output traces and reports valid-pair coverage; source equality and pass/fail equality are not substitutes.

The external test suite preserves the pinned `evalplus/humanevalplus` assertion source, including its numerical comparisons, and explicitly invokes `check(entry_point)`. Reference solutions must pass before a suite is frozen. This is the pinned dataset's assertion protocol, not a claim to reproduce every EvalPlus library runner behavior. Development and legitimate scoring use the qualified source's own assertions. Existing non-coding frozen suite hashes remain unchanged.

The harness strips the fixed coding service instruction before embedding and embeds the specification without the service instruction or signature wrapper. Calibration, source-account generation, held-out replay, and utility use coding's own pools. Default L4 behavior is refusal. Coding deception is not implemented and is rejected explicitly by both the study launcher and HTTP settings.

## Remaining study gates

Run the development-only pilot against the actual teacher before freezing the cap:

```bash
uv run distill tasks pilot --task-dir data/tasks/coding --output-dir artifacts/coding-cap-pilot --teacher-url http://TEACHER_IP:8000 --limit 200
```

The command reports finish-reason counts, the number of returned responses, the fraction ending for length, and the served model. It exits nonzero when more than 2% end for length or when provider errors, unknown finishes, or model changes make the result inconclusive. It writes only separate pilot artifacts, never collection or training inputs. No teacher pilot has been measured during integration.

The 512-token output cap is provisional. A development-only pilot must report the number of completed responses and the fraction ending for length; raise the cap and rerun if that fraction exceeds 2%. Freeze the cap before final testing. The sequence-length report must separately show prompt and supervised-response tokens.

Sandbox validation, a teacher-versus-base development benchmark, the B=2,000 seed-42 recipe sweep where supported, the transfer gate and detectability replay remain required experimental evidence before making final coding effectiveness claims. Default inclusion does not establish those measurements. Report unavailable GPU/provider results as pending. This document does not authorize or claim a final study freeze.

## Validation boundary

Tests exercise fixture qualification and disjoint exports, full attacker budgets, collection through response-token training preparation, benchmark/report assembly, and authenticated HTTP enforcement. All 164 pinned external rows were inspected in the real sandbox. With the declared reference-qualified protocol, 163 references pass and one malformed upstream wrapper is excluded explicitly. No full OpenCodeInstruct qualification, teacher collection, GPU training, or final effectiveness experiment ran during integration. The manuscript declares coding as an optional extension and leaves its outcomes pending.

Coding detector serialization is recorded as `detector-v2`. Use fresh calibrations when moving from an older serialization version; existing non-coding extraction rules themselves are unchanged.

Final integration checks: the complete Python suite passed with 156 tests and four skipped training-stack tests. Repository Ruff and mypy checks passed, along with four dashboard data tests and TypeScript checking. The default launchers and gateway generators include all four domains. These checks do not establish a measured qualified-pool size, cap-pilot outcome, or trained coding capability.

## Frozen external subset and execution identity

The pinned Hugging Face export has a malformed `HumanEval/32` wrapper, `_poly(*candidate(*inp), inp)`, which tries to expand a scalar return value. The builder declares that one exclusion before model inference and writes its source-row hash and reason to `evaluation-exclusions.json` and the manifest. Every other reference must pass; unexpected reference failures still abort the build. The resulting HumanEval+ part is the 163-item reference-qualified assertion subset of the 164-row export, not the full official EvalPlus benchmark. Report this name and denominator with results.

HumanEval+ alone gives a pass@1 interval near ±7.7 points at 50%, too wide for the study's 2-point gain margin. The frozen test suite therefore also includes MBPP+ at `b2d74c9`. Each MBPP+ task's specification is its one-line description followed by its first plain assertion as an example; the signature comes from the reference function that the pinned harness calls. The harness runs as a script with its `test_imports` and a final `assert callable(ENTRY)`. All 378 references were run in the sandbox on 2026-10-20: 376 pass. `Mbpp/255` exhausts the sandbox memory limit on the plus inputs, and `Mbpp/596` expects `sys.getsizeof` values that depend on the interpreter version; both are declared exclusions, recorded like `HumanEval/32`. The combined suite has 539 items from two pinned sources, listed in the manifest's `evaluation_datasets`, and qualification decontaminates the training corpus against both. A frozen suite may now hold several pinned sources for one benchmark; their selected counts must sum to its items. Fixture builds without MBPP+ rows keep a HumanEval+-only suite. A coding build started before this change has a HumanEval+-only suite and should be rebuilt in a new directory. The intended polynomial oracle is documented in [EvalPlus](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/eval/__init__.py); this implementation excludes the malformed export row rather than silently rewriting its oracle or gold reference.

Reference-only validation found that tasks 15 and 139 exceed the earlier 2-second CPU limit. The declared 15-second CPU soft limit and 20-second wall clock pass all 163 retained references on this host. The benchmark artifact records execution limits, runtime version, Python and architecture, and includes that identity in comparison keys and cached-run reuse. This qualification used no model outputs and does not tune the set to student performance. The native report is `artifacts/coding-external-reference-check.json`; the full pool has not been qualified.

A separate native probe of the first ten OpenCodeInstruct rows converted eight function contracts. Four references passed and four failed their assertions; two rows were unsupported or malformed. This ordered ten-row prefix is not a representative quality estimate and was not used to choose a task subset or corpus cap. The full scan applies the same declared rules to every row and must report its own counts. Probe metadata is in `artifacts/coding-real-source-probe.json`.

Final local validation on 2026-10-19 passed 157 tests with the coding extra, with four opt-in native-training tests skipped. That earlier validation used optional packaging: a plain install excluded coding and passed 134 tests with seven optional tests or modules skipped. The normal-dependency integration below supersedes that packaging behavior. Ruff, both mypy targets, the four Bun data tests and the dashboard build passed. The shared browser rendered a clearly labeled coding fixture with pass@1; existing AG News and SQuAD frozen suites verified at their stored hashes. No corpus-wide qualification or GPU effectiveness result is implied by these checks.

Functional trace version `reference-covered-calls-v2` keeps comparable shared wrong outputs when a test comparison rejects their type. Exceptions raised by the candidate function, unsupported output values, timeouts or missing reference-covered calls still invalidate the trace. This changes agreement coverage only; assertion correctness remains a separate score, and execution identity prevents mixing trace versions in one comparison.

Normal-install validation: plain `uv sync --locked` installs `coding-task` without extras, and the launcher reports `ag_news,squad,gsm8k,coding`. Coding tests now import the package directly instead of silently skipping a missing optional dependency. The full default-environment suite passed with 157 tests and four skipped opt-in training tests; repository Ruff and mypy checks passed.
