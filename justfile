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

# Install the task dataset import dependencies.
setup-eval:
	uv sync --all-packages --extra eval

# Validate and build the dashboard.
dashboard-build:
	bun install --cwd apps/dashboard
	bun run --cwd apps/dashboard build

# Serve measured benchmark results locally.
dashboard *args:
	uv run distill benchmark dashboard {{args}}

# Create explicitly simulated preview artifacts without running models.
benchmark-demo:
	uv run distill benchmark demo

# Evaluate one task config: stage is before, after, or all.
benchmark config stage="before" *args:
	uv run --extra train distill benchmark run --config {{quote(config)}} --stage {{stage}} {{args}}

# Export a benchmark report as JSON.
benchmark-report *args:
	uv run distill benchmark report {{args}}

# Run dashboard data tests and type checking.
dashboard-check:
	bun install --cwd apps/dashboard --frozen-lockfile
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

# Check data and tokenizer serialization without model weights or training.
preflight-train config="configs/train.task.toml":
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

# Calibrate, replay held-out benign accounts and one extractor across benign client mixes.
diagnose *args:
	uv run distill tasks diagnose {{args}}

# Run collection, training, evaluation, and a second extraction through the harness.
start *args:
	{{python}} scripts/pipeline.py start {{args}}

# Collect across AG News, SQuAD, GSM8K, and coding, then train a shared student.
distill *args:
	{{python}} scripts/pipeline.py distill {{args}}

# Compile the manuscript to paper/main.pdf.
paper:
	tectonic --outdir paper paper/main.tex
