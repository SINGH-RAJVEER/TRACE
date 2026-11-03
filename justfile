set dotenv-load := true

default:
	@just --list

setup:
	uv sync --all-packages --extra eval --extra embed

teacher:
	uv run teacher-api

harness:
	uv run --extra embed harness-api

# Run the whole study end to end, logging under artifacts/runs/RUN_ID; rerun an ID to resume.
start run_id="":
	bash scripts/start.sh {{run_id}}

# Copy a run's evidence into readings/ (next free vN unless a name is given), minus secrets.
reading run_id name="":
	uv run --frozen python scripts/run_tools.py export artifacts/runs/{{run_id}} readings {{name}}

# Rebuild the results index rows of the frozen v3 study and the v4 audit evidence.
backfill:
	uv run --frozen python scripts/run_tools.py backfill readings

test:
	uv run pytest

check:
	uv run ruff check apps packages tests
	uv run mypy apps packages tests

format:
	uv run ruff format apps packages tests

lock:
	uv lock
