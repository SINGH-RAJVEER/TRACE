# Distillation lab

This repository is a runnable research environment for sequence-level knowledge distillation from an authorized hosted language model into an open-weight causal language model.

The first implementation uses teacher-generated prompt-completion pairs and completion-only supervised fine-tuning. It does not claim token-level knowledge distillation because a text API does not expose the teacher's full token distribution.

## Repository layout

```text
apps/teacher-api/         FastAPI service in front of an OpenAI-compatible hosted model
packages/distillation/    Collection, dataset preparation, and QLoRA/LoRA training tools
configs/                  Example training configuration
data/examples/            Small input fixture for trying the pipeline
docs/                     Architecture, operating guide, and research notes
scripts/                  Bash and PowerShell scripts for two-machine and single-machine runs
```

## Quick start

Requirements are Python 3.14, [uv](https://docs.astral.sh/uv/), and optionally `just`.

```bash
cp .env.example .env
uv sync --all-packages
uv run teacher-api
```

In another terminal:

```bash
uv run distill collect \
	--input data/examples/prompts.jsonl \
	--output data/teacher/ledger.jsonl \
	--teacher-url http://127.0.0.1:8000 \
	--terms-review-id YOUR_APPROVAL_REFERENCE \
	--no-thinking

uv run distill prepare \
	--input data/teacher/ledger.jsonl \
	--output data/processed/train.jsonl \
	--prompts data/examples/prompts.jsonl \
	--accept-pending

uv sync --all-packages --extra train
uv run --extra train distill train --config configs/train.example.toml
```

To run everything with one command per machine, use `scripts/teacher-host` and `scripts/student-train` on two machines, or `scripts/distill-local` on one. Each comes as `.sh` and `.ps1`. See [docs/run-scripts.md](docs/run-scripts.md).

The collection command deliberately requires a terms-review reference. Standard commercial API terms may prohibit using outputs to train a competing model. Configure only a provider and account whose agreement permits this use. `--accept-pending` is suitable only for the included plumbing example. Paper runs should use a separate review file as described in the operating guide.

See [docs/distillation-pipeline.md](docs/distillation-pipeline.md) for the API contract, data formats, and training instructions. The method choice and primary sources are recorded in [docs/research/teacher-student-distillation.md](docs/research/teacher-student-distillation.md).

