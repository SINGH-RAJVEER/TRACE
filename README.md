# Distillation lab

This repository is a runnable research environment for sequence-level knowledge distillation from an authorized hosted language model into an open-weight causal language model.

The first implementation uses teacher-generated prompt-completion pairs and completion-only supervised fine-tuning. It does not claim token-level knowledge distillation because a text API does not expose the teacher's full token distribution.

## Repository layout

```text
apps/teacher-api/         FastAPI service in front of an OpenAI-compatible hosted model
apps/dashboard/           Read-only benchmark dashboard using React, shadcn controls, and Recharts
packages/distillation/    Collection, preparation, training, and frozen benchmark evaluation
configs/                  Example training and four-slot evaluation configurations
data/authored/            Hand-written prompts with answer keys, part of the prompt corpus
docs/                     Architecture, operating guide, and research notes
scripts/                  Cross-platform Python launcher for two-machine and single-machine runs
```

## Quick start

Requirements are Python 3.14, [uv](https://docs.astral.sh/uv/), and optionally `just`.

```bash
cp .env.example .env
uv sync --all-packages --extra eval
uv run --extra eval distill corpus build --profile quick
uv run teacher-api
```

The corpus build downloads pinned public datasets and writes the prompts and their answer key to `data/prompts/`. The `quick` profile has about 7,000 prompts; see [docs/prompt-corpus.md](docs/prompt-corpus.md) for the larger profiles.

In another terminal:

```bash
uv run distill collect \
	--input data/prompts/all-splits.jsonl \
	--output data/teacher/ledger.jsonl \
	--teacher-url http://127.0.0.1:8000 \
	--terms-review-id YOUR_APPROVAL_REFERENCE \
	--no-thinking

uv run distill review \
	--ledger data/teacher/ledger.jsonl \
	--references data/prompts/references.jsonl \
	--output data/teacher/ledger.reviews.jsonl

uv run distill prepare \
	--input data/teacher/ledger.jsonl \
	--output data/processed/train.jsonl \
	--prompts data/prompts/all-splits.jsonl \
	--reviews data/teacher/ledger.reviews.jsonl

uv sync --all-packages --extra train
uv run --extra train distill train --config configs/train.example.toml
```

To run the entire distillation loop on one machine, copy `.env.example` to `.env`, set `TEACHER_TERMS_REVIEW_ID` to your approval reference, adjust the GPU settings if needed, and run:

```bash
just distill-local
```

It builds the `standard` prompt corpus if it is missing, reuses it when the profile matches, hosts the teacher, collects a resumable ledger, stops the teacher to free GPU memory, reviews the answers, prepares the training and validation splits, and trains the student with `configs/train.distill.toml`. Set `DISTILL_CORPUS=quick` or `full` in `.env` to choose another profile. `.env.example` lists the supported environment settings and their defaults; training hyperparameters live in the TOML configs.

Without `just`, run `python3 scripts/pipeline.py distill-local`. On Windows, use `py -3 scripts\pipeline.py distill-local --upstream-url http://127.0.0.1:8001/v1` with an existing model server. For two machines, run `just teacher-host` and `just student-train --teacher-url http://TEACHER_IP:8000`. See [docs/run-scripts.md](docs/run-scripts.md) for options, GPU requirements, and platform setup, and [docs/prompt-corpus.md](docs/prompt-corpus.md) for the corpus profiles.

The collection command deliberately requires a terms-review reference. Standard commercial API terms may prohibit using outputs to train a competing model. Configure only a provider and account whose agreement permits this use. `--accept-pending` skips review and suits only plumbing tests. Paper runs should use a separate review file as described in the operating guide.

See [docs/distillation-pipeline.md](docs/distillation-pipeline.md) for the API contract, data formats, and training instructions.

## Benchmarks and dashboard

The evaluation setup supports GSM8K, ARC-Challenge, MMLU, TruthfulQA, HumanEval, and MBPP. Run the teacher and base student before training, then reevaluate the teacher and evaluate the distilled student after training. The initial teacher score stays the fixed gap-closure reference; reevaluation measures teacher drift. It records accuracy or executable pass@1, confidence intervals, paired gains and regressions, agreement, latency, throughput, failures, and reproducibility metadata. It does not start distillation.

See [docs/benchmarks.md](docs/benchmarks.md) for every metric's formula, denominator, units, and limitations, plus dataset sources, the evaluation configuration, and commands for later runs.

To preview the dashboard with explicitly labeled simulated results:

```bash
bun install --cwd apps/dashboard
bun run --cwd apps/dashboard build
uv run distill benchmark demo
uv run distill benchmark dashboard --results-dir outputs/benchmark-demo
```

Open `http://127.0.0.1:8002`. Use `uv run distill benchmark dashboard` to view your measured results instead. The preview does not run inference, Docker tests, or training.
