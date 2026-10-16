# Task extraction study

This repository studies how an access policy affects extraction from narrow language-model APIs. AG News classification and SQuAD 1.1 extractive question answering form the core study; GSM8K adds a generative math task. Each task has separate query pools, ledgers, adapters, and frozen evaluation suites.

Students learn from teacher-generated text through sequence-level supervised fine-tuning with completion-only loss. The study measures task capability, extraction under query budgets, and legitimate utility. It makes no general-chat or token-distribution distillation claim.

## Repository layout

```text
apps/teacher-api/         FastAPI gateway to an OpenAI-compatible teacher
apps/harness-api/         Protected teacher API: the harness in front of the teacher gateway
apps/dashboard/           Read-only task and traffic results with shadcn controls
packages/distillation/    Task splits, collection, usability review, training, evaluation
packages/harness/         Extraction detector, calibration, and graduated access policy
configs/                  Task training recipe, per-task evaluation, traffic settings
docs/                     Protocol, operations, metrics, and manuscript revisions
scripts/                  Cross-platform task-study and teacher-host launcher
```

## Run the study

Requirements are Python 3.14, uv, and optionally just. NixOS users can enter `devenv shell` for Python, uv, Bun, native libraries, and build tools. Copy the environment file and set `TEACHER_TERMS_REVIEW_ID` before collection. Set `TEACHER_API_TOKEN` when serving beyond loopback.

```bash
cp .env.example .env
just start
```

`just start` builds the task corpora, loads the teacher, collects ledgers, benchmarks the teacher and base student, trains and benchmarks the baseline students, then starts the extraction harness alongside the teacher. It collects a separate protected ledger, trains a fresh student on usable protected responses, and benchmarks that student on the same frozen tests. Capability and before/after traffic reports are written under `artifacts/start`. See [docs/run-scripts.md](docs/run-scripts.md#complete-pipeline) for options, outputs, and resuming.

For collection and training without automatic benchmarks or the protected cycle:

```bash
just paper-tasks
```

The launcher builds missing task splits, collects each seed's largest supported query budget, stops a locally hosted teacher to free GPU memory, reviews response usability without gold answers, and trains attacker and gold-label reference adapters. The defaults are three seeds and budgets 500, 2,000, and 8,000. GSM8K supports only the first two, giving 48 training runs in total. Collection and training resume from existing artifacts. Evaluation runs separately.

For a smaller declared matrix or a remote teacher:

```bash
just paper-tasks --tasks ag_news --seeds 42 --budgets 500,2000
just teacher-host
# On the student machine, with the same API token:
just paper-tasks --teacher-url http://TEACHER_IP:8000
```

Without just, use `python3 scripts/pipeline.py paper-tasks`. Windows uses `py -3 scripts\pipeline.py paper-tasks --teacher-url http://TEACHER_IP:8000` or an existing local server through `--upstream-url`.

See [docs/paper-tasks.md](docs/paper-tasks.md) for splits, budgets, output limits, development selection, and the implementation checklist; [docs/run-scripts.md](docs/run-scripts.md) for launchers and environment settings; and [docs/distillation-pipeline.md](docs/distillation-pipeline.md) for collection and training internals.

## Task evaluation and dashboard

Build the chosen task before evaluation. Set the evaluation config's teacher identity to the actual deployment and its adapter path to the generated `attacker` or `gold` directory for the seed and budget being scored. Give each condition a distinct experiment name and output directory.

```bash
uv run --extra train distill benchmark run --config configs/eval.ag_news.toml --stage before
uv run --extra train distill benchmark run --config configs/eval.ag_news.toml --stage after
bun install --cwd apps/dashboard
bun run --cwd apps/dashboard build
uv run distill benchmark dashboard --results-dir outputs/tasks/ag_news
```

Open `http://127.0.0.1:8002`. Primary metrics are AG News accuracy, SQuAD token F1, and GSM8K numeric exact match. The report also records SQuAD exact match, agreement, failures, latency, and provenance. See [docs/benchmarks.md](docs/benchmarks.md) for denominators and limitations.

For a labeled simulation without model execution:

```bash
uv run distill benchmark demo
uv run distill benchmark dashboard --results-dir outputs/tasks-demo/ag_news
```

## Extraction policy and paper status

`packages/harness` implements query embeddings, window MMD scores, benign-only calibration, accumulated account evidence, and graduated access actions. See [docs/harness.md](docs/harness.md) and [docs/harness-benchmarks.md](docs/harness-benchmarks.md). Synthetic traffic fixtures do not enter task training or capability scores.

Configured experiments and passing software tests do not establish extraction effectiveness or defense utility. Establish positive undefended transfer on development data before final defense testing, then compare paired defended and undefended students with matched supervision controls. Track manuscript changes in [docs/paper-revision-todo.md](docs/paper-revision-todo.md).

The current configs use Qwen3.5-9B and Qwen3.5-2B-Base. The draft's Qwen2.5 victim and student sizes remain an unresolved protocol difference. Resolve and freeze the actual model identities before final experiments.
