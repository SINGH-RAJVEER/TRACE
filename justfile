set dotenv-load := true

default:
	@just --list

setup:
	uv sync --all-packages --extra eval --extra embed

teacher:
	uv run teacher-api

harness:
	uv run --extra embed harness-api

test:
	uv run pytest

check:
	uv run ruff check apps packages tests
	uv run mypy apps packages tests

format:
	uv run ruff format apps packages tests

lock:
	uv lock
