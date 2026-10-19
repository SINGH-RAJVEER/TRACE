# Task extraction study

This repository studies how an access policy affects extraction from narrow language-model APIs. The default study covers AG News classification, SQuAD 1.1 extractive question answering, GSM8K generative math, and Python coding. Each domain keeps separate source pools and frozen evaluation suites. Default campaigns combine received supervision into one student adapter behind one shared gateway.

Students learn from teacher-generated text through sequence-level supervised fine-tuning with completion-only loss. The study measures task capability, extraction under query budgets, and legitimate utility. It makes no general-chat or token-distribution distillation claim.

## Repository layout

```text
apps/teacher-api/         FastAPI gateway to an OpenAI-compatible teacher
apps/harness-api/         Protected teacher API: the harness in front of the teacher gateway
apps/dashboard/           Read-only task and traffic results with shadcn controls
packages/distillation/    Task splits, collection, usability review, training, evaluation
packages/coding/          Python sandbox, full-corpus qualification, and executable scoring
packages/harness/         Extraction detector, calibration, and graduated access policy
configs/                  Task training recipe, per-task evaluation, traffic settings
docs/                     Protocol, operations, metrics, and manuscript revisions
scripts/                  Cross-platform task-study and teacher-host launcher
```

## Run the study

[Coding](docs/coding-task.md) is included by default. The task builder streams the complete pinned OpenCodeInstruct source, qualifies references in a Linux sandbox, removes detected duplicates and evaluation overlap, and exports disjoint study pools. There is no 8,000-example cap. The qualified pool size is measured during the build, not assumed from the upstream row count.

Requirements are Python 3.14, uv, and optionally just. Default coding qualification and evaluation require Linux, bubblewrap, and util-linux `prlimit`, provided by `devenv shell`. NixOS users can enter `devenv shell` for Python, uv, Bun, native libraries, and build tools. Copy the environment file and set `TEACHER_TERMS_REVIEW_ID` before collection. Set `TEACHER_API_TOKEN` when serving beyond loopback.

```bash
cp .env.example .env
just start
```

`just start` builds missing corpora, calibrates one mixed account policy, collects undefended and protected campaigns, selects a shared recipe using development data, and trains one combined student per campaign and condition. The same adapter is evaluated on every domain. Benign utility and token-matched controls are recorded separately. Outputs are in `artifacts/gateway`. See [docs/run-scripts.md](docs/run-scripts.md#complete-pipeline) for options and [the shared protocol](docs/multi-domain-study.md#run-the-shared-pipeline) for accounting and validation limits.

For collection and training without automatic benchmarks or the protected cycle:

```bash
just distill
```

Both launchers include all four domains, seeds 42/1/20, and full attacker inventories by default. `full` combines every domain's attacker pool under one campaign allowance and deadline. Numeric budgets are shared totals with balanced domain allocations. A5 uses six accounts under that same allowance. `distill` runs undefended A1 collection and shared training only; `start` also runs A2/A5 and all three defenses. GSM8K's 256-token and coding's 512-token caps remain provisional until their implemented pilots pass against the actual teacher. Use `--mode tasks` for the retained separate-task workflow. Completed campaign artifacts are verified on reuse; interrupted campaigns require a new directory to avoid replenishing spent resources.

For a smaller declared matrix or a remote teacher:

```bash
just distill --tasks ag_news,squad,gsm8k,coding --seeds 42 --budgets 6000
just teacher-host
# On the student machine, with the same API token:
just distill --teacher-url http://TEACHER_IP:8000
```

Without just, use `python3 scripts/pipeline.py distill`. On Windows, select non-coding tasks with `--tasks ag_news,squad,gsm8k`. The bootstrap uses `py -3 scripts\pipeline.py distill --teacher-url http://TEACHER_IP:8000` or an existing local server through `--upstream-url`.

See [docs/paper-tasks.md](docs/paper-tasks.md) for splits, budgets, output limits, and development selection; [docs/run-scripts.md](docs/run-scripts.md) for launchers and environment settings; and [docs/distillation-pipeline.md](docs/distillation-pipeline.md) for collection and training internals.

## Task evaluation and dashboard

Build the chosen task before evaluation. Set the evaluation config's teacher identity to the actual deployment and its adapter path to the generated `attacker` or `gold` directory for the seed and budget being scored. Give each condition a distinct experiment name and output directory.

```bash
uv run --extra train distill benchmark run --config configs/eval.ag_news.toml --stage before
uv run --extra train distill benchmark run --config configs/eval.ag_news.toml --stage after
bun install --cwd apps/dashboard
bun run --cwd apps/dashboard build
uv run distill benchmark dashboard --results-dir outputs/tasks/ag_news
```

Open `http://127.0.0.1:8002`. Primary metrics are AG News accuracy, SQuAD token F1, GSM8K numeric exact match, and coding pass@1. The report also records SQuAD exact match, agreement, failures, latency, and provenance. See [docs/benchmarks.md](docs/benchmarks.md) for denominators and limitations, and its [metrics reference](docs/benchmarks.md#metrics-reference) for every field the capability, campaign, timing, benign-cost, calibration, traffic, and overhead benchmarks record.

For a labeled simulation without model execution:

```bash
uv run distill benchmark demo
uv run distill benchmark dashboard --results-dir outputs/tasks-demo/ag_news
```

## Extraction policy and paper status

`packages/harness` implements query embeddings, window MMD scores, benign-only calibration, accumulated account evidence, and graduated access actions. See [docs/harness.md](docs/harness.md) and [docs/harness-benchmarks.md](docs/harness-benchmarks.md). Synthetic traffic fixtures do not enter task training or capability scores.

Configured experiments and passing software tests do not establish extraction effectiveness or defense utility. Establish positive undefended transfer on development data before final defense testing, then compare paired defended and undefended students with matched supervision controls.

The current configs use Qwen3.5-9B and Qwen3.5-2B-Base. The draft's Qwen2.5 victim and student sizes remain an unresolved protocol difference. Resolve and freeze the actual model identities before final experiments.
