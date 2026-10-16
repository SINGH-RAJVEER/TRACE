set dotenv-load := true
set shell := ["bash", "-cu"]
set windows-shell := ["pwsh.exe", "-NoLogo", "-NoProfile", "-Command"]

# Interpreter that starts scripts/pipeline.py, which then prepares the workspace environment.
python := if os_family() == "windows" { "py -3" } else { "python3" }

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

# Validate and build the dashboard.
dashboard-build:
	bun install --cwd apps/dashboard
	bun run --cwd apps/dashboard build

# Serve measured benchmark results locally.
dashboard:
	uv run distill benchmark dashboard

# Create explicitly simulated preview artifacts without running models.
benchmark-demo:
	uv run distill benchmark demo

# Run dashboard data tests and type checking.
dashboard-check:
	bun test --cwd apps/dashboard src
	bun run --cwd apps/dashboard check

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
	uv run mypy --python-version 3.9 scripts

# Format Python source using tabs.
format:
	uv run ruff format .
	uv run ruff check --fix .

# Refresh the locked dependency graph.
lock:
	uv lock

# Collect teacher answers for the prompt corpus. Run `just corpus` and set TEACHER_TERMS_REVIEW_ID first.
collect:
	uv run distill collect --input data/prompts/all-splits.jsonl --output data/teacher/ledger.jsonl --teacher-url http://127.0.0.1:8000 --terms-review-id "{{env('TEACHER_TERMS_REVIEW_ID', '')}}" --no-thinking

# Build the transfer prompt corpus: quick, standard, or full. See docs/prompt-corpus.md.
corpus profile="standard":
	uv run --extra eval distill corpus build --profile {{quote(profile)}} --output-dir data/prompts

# Check the collected answers against the corpus answer key.
review:
	uv run distill review --ledger data/teacher/ledger.jsonl --references data/prompts/references.jsonl --output data/teacher/ledger.reviews.jsonl

# Convert reviewed training records into TRL prompt-completion data.
prepare:
	uv run distill prepare --input data/teacher/ledger.jsonl --output data/processed/train.jsonl --prompts data/prompts/all-splits.jsonl --reviews data/teacher/ledger.reviews.jsonl

# Train with the example LoRA configuration.
train-example:
	uv run --extra train distill train --config configs/train.example.toml

# Check data and tokenizer serialization without model weights or training.
preflight-train config="configs/train.example.toml":
	uv run --extra train distill train --config {{quote(config)}} --preflight-only

# Construct a tiny native trainer; its training method is intercepted before optimization.
test-training-setup:
	DISTILL_TEST_TRAINING_STACK=1 uv run --extra train pytest tests/test_training_stack.py -k native_setup

# Run offline optimizer, interruption/resume, adapter reload, and preflight checks.
test-training:
	DISTILL_TEST_TRAINING_STACK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra train pytest tests/test_training_stack.py

# Two machines, teacher side: serve vLLM and the teacher API. See docs/run-scripts.md.
teacher-host *args:
	{{python}} scripts/pipeline.py teacher-host {{args}}

# Collect from the monorepo teacher API and train the student.
student-train *args:
	{{python}} scripts/pipeline.py student-train {{args}}

# One machine: run the whole distillation.
distill-local *args:
	{{python}} scripts/pipeline.py distill-local {{args}}

# The paper's AG News, SQuAD, and GSM8K study, with its own ledgers and adapters. See docs/paper-tasks.md.
paper-tasks *args:
	{{python}} scripts/pipeline.py paper-tasks {{args}}
