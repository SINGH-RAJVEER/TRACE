set dotenv-load := true
set shell := ["bash", "-cu"]
set windows-shell := ["pwsh.exe", "-NoLogo", "-NoProfile", "-Command"]

default:
	@just --list

# Install runtime and development dependencies.
setup:
	uv sync --all-packages

# Install the optional GPU training stack.
setup-train:
	uv sync --all-packages --extra train

# Install the public benchmark import dependencies.
setup-eval:
	uv sync --all-packages --extra eval

# Freeze the default suite without inference or training.
benchmark-freeze:
	uv run --extra eval distill benchmark freeze --output data/benchmarks/suite.json

# Create explicitly simulated preview artifacts without running models.
benchmark-demo:
	uv run distill benchmark demo

# Run the teacher inference API.
dev:
	uv run teacher-api

# Run the test suite.
test:
	uv run pytest

# Run static checks.
check:
	uv run ruff check .
	uv run mypy apps packages tests

# Format Python source using tabs.
format:
	uv run ruff format .
	uv run ruff check --fix .

# Refresh the locked dependency graph.
lock:
	uv lock

# Collect the included example prompts. Set TERMS_REVIEW_ID in .env first.
collect-example:
	uv run distill collect --input data/examples/prompts.jsonl --output data/teacher/ledger.jsonl --teacher-url http://127.0.0.1:8000 --terms-review-id "{{env('TEACHER_TERMS_REVIEW_ID', '')}}"

# Convert accepted training records into TRL prompt-completion data.
prepare-example:
	uv run distill prepare --input data/teacher/ledger.jsonl --output data/processed/train.jsonl --accept-pending

# Train with the example QLoRA configuration.
train-example:
	uv run --extra train distill train --config configs/train.example.toml

# Check data and tokenizer serialization without model weights or training.
preflight-train config="configs/train.example.toml":
	uv run --extra train distill train --config {{quote(config)}} --preflight-only

# Construct a tiny native trainer; its training method is intercepted before optimization.
test-training-setup:
	DISTILL_TEST_TRAINING_STACK=1 uv run --extra train pytest tests/test_training_stack.py

# Two machines, teacher side: serve vLLM and the teacher API with bash. See docs/run-scripts.md.
teacher-host *args:
	bash scripts/teacher-host.sh {{args}}

# Two machines, teacher side: serve vLLM and the teacher API with PowerShell.
teacher-host-ps *args:
	pwsh -NoLogo -NoProfile -File scripts/teacher-host.ps1 {{args}}

# Two machines, student side: collect from a remote teacher and train with bash.
student-train *args:
	bash scripts/student-train.sh {{args}}

# Two machines, student side: collect from a remote teacher and train with PowerShell.
student-train-ps *args:
	pwsh -NoLogo -NoProfile -File scripts/student-train.ps1 {{args}}

# One machine: run the whole distillation with bash.
distill-local *args:
	bash scripts/distill-local.sh {{args}}

# One machine: run the whole distillation with PowerShell.
distill-local-ps *args:
	pwsh -NoLogo -NoProfile -File scripts/distill-local.ps1 {{args}}
