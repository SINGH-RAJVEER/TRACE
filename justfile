set dotenv-load := true
set shell := ["bash", "-cu"]

default:
	@just --list

# Install runtime and development dependencies.
setup:
	uv sync --all-packages

# Install the optional GPU training stack.
setup-train:
	uv sync --all-packages --extra train

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
	uv run distill collect --input data/examples/prompts.jsonl --output data/teacher/ledger.jsonl --teacher-url http://127.0.0.1:8000 --terms-review-id "$TEACHER_TERMS_REVIEW_ID"

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
