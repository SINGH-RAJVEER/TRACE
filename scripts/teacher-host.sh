#!/usr/bin/env bash
# Scenario 1, teacher machine: serve Qwen3.5-9B with vLLM on loopback and expose the
# teacher API on the network for a student machine to collect from.
set -euo pipefail

SCRIPT_NAME="teacher-host"
# shellcheck source=lib/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/common.sh"
load_dotenv
enter_devenv_if_nixos "${BASH_SOURCE[0]}" "$@"
cd "$REPO_ROOT"

usage() {
	cat << EOF
Usage: scripts/teacher-host.sh [options]

Starts vLLM and the teacher API, then runs until interrupted.

  --host ADDRESS        Teacher API bind address (default: 0.0.0.0)
  --port PORT           Teacher API port (default: \${TEACHER_PORT:-8000})
  --upstream-url URL    Use an existing OpenAI-compatible server instead of starting vLLM
  -h, --help            Show this help

Requires TEACHER_TERMS_REVIEW_ID, and TEACHER_API_TOKEN when binding beyond loopback.
vLLM settings: VLLM_PORT, VLLM_MAX_MODEL_LEN, VLLM_GPU_MEMORY_UTILIZATION, VLLM_EXTRA_ARGS,
VLLM_SPEC, VLLM_PYTHON, VLLM_BIN, VLLM_STARTUP_TIMEOUT. See docs/run-scripts.md.
EOF
}

HOST="0.0.0.0"
PORT="${TEACHER_PORT:-8000}"
UPSTREAM_URL=""
while (($#)); do
	case "$1" in
		--host) HOST="${2:?--host needs a value}"; shift 2 ;;
		--port) PORT="${2:?--port needs a value}"; shift 2 ;;
		--upstream-url) UPSTREAM_URL="${2:?--upstream-url needs a value}"; shift 2 ;;
		-h | --help) usage; exit 0 ;;
		*) usage >&2; die "unknown option: $1" ;;
	esac
done

[[ -n "${TEACHER_TERMS_REVIEW_ID:-}" ]] \
	|| die "set TEACHER_TERMS_REVIEW_ID in .env; the teacher API refuses requests without it"
if [[ "$HOST" != 127.0.0.1 && "$HOST" != localhost && -z "${TEACHER_API_TOKEN:-}" ]]; then
	die "set TEACHER_API_TOKEN before binding to $HOST (generate one with: openssl rand -hex 32)"
fi

if [[ -z "$UPSTREAM_URL" ]]; then
	select_runtime gpu vllm
else
	select_runtime
fi
prepare_app_env
trap stop_background EXIT
trap 'exit 130' INT TERM
start_teacher_stack "$HOST" "$PORT" "$UPSTREAM_URL" -

address="$(hostname -I 2> /dev/null | awk '{print $1}')"
log "teacher API ready on $HOST:$PORT"
log "on the student machine run:"
log "  scripts/student-train.sh --teacher-url http://${address:-<this-host>}:$PORT --reviews <file>"
log "with the same TEACHER_API_TOKEN set on that machine"
log "press Ctrl+C to stop"
supervise_background
