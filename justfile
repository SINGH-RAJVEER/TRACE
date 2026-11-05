set dotenv-load := true
set positional-arguments := true
set shell := ["bash", "-cu"]

default:
	@just --list

# Install the core API and development dependencies from the frozen lock.
setup:
	uv sync --frozen --all-packages

# Add encoder and evaluation dependencies for calibration, studies and GPU student commands.
setup-eval:
	uv sync --frozen --all-packages --extra eval --extra embed

# Start the teacher adapter; the upstream model server must already be available.
teacher:
	uv run --frozen teacher-api

# Start the campaign-protected API. See `just start --help` for launch options.
start *args:
	bash scripts/start.sh "$@"

alias harness := start

# Check launch settings and required paths without loading models or contacting the teacher.
validate *args:
	uv run --frozen python -m harness_api.launcher validate "$@"

# Write a new campaign JSON; requires --namespace NAME. Defaults to observation only.
campaign-config output *args:
	uv run --frozen python -m harness_api.launcher config "$@"

# Explicit base-policy API entry point, with campaign detection disabled even in .env.
harness-legacy:
	uv run --frozen --extra embed python -m harness_api.launcher legacy

# Run the retained corpus/calibration/teacher/attack/utility study; optional RUN_ID resumes.
study *args:
	bash scripts/study.sh "$@"

# GPU host only: teacher headroom and clean-student gates; requires study declarations.
baseline-gates *args:
	bash scripts/baseline-gates.sh "$@"

# Freeze receipt-ordered prefixes from a completed campaign into a NEW output directory.
checkpoint-export campaign output *args:
	uv run --frozen distill checkpoints export --campaign "$1" --output "$2" "${@:3}"

# Plan GSM8K checkpoint students; --target-accuracy must be declared explicitly.
checkpoint-plan export suite output *args:
	uv run --frozen distill checkpoints plan --export "$1" --suite "$2" --output "$3" "${@:4}"

# GPU execution is explicit and uses the frozen student recipe; completed cells can resume.
checkpoint-execute output:
	uv run --frozen --extra eval distill checkpoints execute --output "$1"

checkpoint-summary output:
	uv run --frozen distill checkpoints summary --output "$1"

# Pass through all corpus, schedule, collection, gate and evaluation subcommands.
distill *args:
	uv run --frozen --extra eval --extra embed distill "$@"

# Pass through offline calibration, replay, adequacy, evidence, review and research commands.
detector *args:
	uv run --frozen --extra embed harness "$@"

calibrate *args:
	uv run --frozen --extra embed harness calibrate "$@"

recalibrate source output profiles="300":
	bash scripts/recalibrate-profiles.sh "$@"

# Explicitly publish a retained study run to readings; product startup never exports it.
reading run_id name="":
	uv run --frozen python scripts/run_tools.py export "artifacts/runs/$1" readings "$2"

# Historical v3/v4 index maintenance, not part of the product startup path.
backfill:
	uv run --frozen python scripts/run_tools.py backfill readings

test *args:
	uv run --frozen pytest "$@"

test-campaign:
	uv run --frozen pytest tests/test_campaign.py tests/test_campaign_api.py

test-checkpoints:
	uv run --frozen pytest tests/test_checkpoints.py tests/test_checkpoint_learning.py

check:
	uv run --frozen ruff check apps packages tests
	uv run --frozen ruff format --check apps packages tests
	uv run --frozen mypy apps packages tests
	for script in scripts/*.sh; do bash -n "$script" || exit; done

format:
	uv run --frozen ruff format apps packages tests

# Maintenance command: deliberately updates the dependency lock.
lock:
	uv lock
