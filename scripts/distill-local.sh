#!/usr/bin/env bash
# Scenario 2, single machine: start the teacher, collect, stop the teacher to free GPU
# memory, then prepare the dataset and train the student.
set -euo pipefail

SCRIPT_NAME="distill-local"
# shellcheck source=lib/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/common.sh"
load_dotenv
enter_devenv_if_nixos "${BASH_SOURCE[0]}" "$@"
cd "$REPO_ROOT"

usage() {
	cat << EOF
Usage: scripts/distill-local.sh (--reviews PATH | --accept-pending) [options]

Runs the whole distillation on this machine. The teacher API binds to 127.0.0.1.
Paths are relative to the repository root.

  --port PORT               Teacher API port (default: \${TEACHER_PORT:-8000})
  --upstream-url URL        Use an existing OpenAI-compatible server instead of starting vLLM
$PIPELINE_USAGE
  -h, --help                Show this help

vLLM settings: VLLM_PORT, VLLM_MAX_MODEL_LEN, VLLM_GPU_MEMORY_UTILIZATION, VLLM_EXTRA_ARGS,
VLLM_SPEC, VLLM_PYTHON, VLLM_BIN, VLLM_STARTUP_TIMEOUT. See docs/run-scripts.md.
EOF
}

PORT="${TEACHER_PORT:-8000}"
UPSTREAM_URL=""
while (($#)); do
	case "$1" in
		--port) PORT="${2:?--port needs a value}"; shift 2 ;;
		--upstream-url) UPSTREAM_URL="${2:?--upstream-url needs a value}"; shift 2 ;;
		-h | --help) usage; exit 0 ;;
		*)
			parse_pipeline_option "$@" || { usage >&2; die "unknown option: $1"; }
			shift "$PIPELINE_SHIFT"
			;;
	esac
done

validate_pipeline_options
# The teacher API refuses requests until it has an approval reference.
export TEACHER_TERMS_REVIEW_ID="$TERMS_REVIEW_ID"

if [[ -z "$UPSTREAM_URL" ]]; then
	select_runtime gpu vllm
else
	select_runtime gpu
fi
prepare_app_env train
trap stop_background EXIT
trap 'exit 130' INT TERM
start_teacher_stack 127.0.0.1 "$PORT" "$UPSTREAM_URL" "$LOG_DIR/teacher-api.log"
collect_teacher_data "http://127.0.0.1:$PORT" "${TEACHER_API_TOKEN:-}"
log "stopping the teacher to free GPU memory for training"
stop_background
prepare_and_train
log "done; the adapter and training manifest are in the config's output_dir"
