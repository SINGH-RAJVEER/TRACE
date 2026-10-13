#!/usr/bin/env bash
# Scenario 1, student machine: collect from a remote teacher API, prepare the dataset,
# and train the Qwen3-4B QLoRA adapter.
set -euo pipefail

SCRIPT_NAME="student-train"
# shellcheck source=lib/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/lib/common.sh"
load_dotenv
enter_devenv_if_nixos "${BASH_SOURCE[0]}" "$@"
cd "$REPO_ROOT"

usage() {
	cat << EOF
Usage: scripts/student-train.sh --teacher-url URL (--reviews PATH | --accept-pending) [options]

Runs collection against a teacher API on another machine, then preparation and training.
Paths are relative to the repository root.

  --teacher-url URL         Teacher API base URL, for example http://10.0.0.5:8000
  --token TOKEN             Teacher API bearer token (default: \$TEACHER_API_TOKEN)
$PIPELINE_USAGE
  -h, --help                Show this help
EOF
}

TEACHER_URL=""
TOKEN=""
while (($#)); do
	case "$1" in
		--teacher-url) TEACHER_URL="${2:?--teacher-url needs a value}"; shift 2 ;;
		--token) TOKEN="${2:?--token needs a value}"; shift 2 ;;
		-h | --help) usage; exit 0 ;;
		*)
			parse_pipeline_option "$@" || { usage >&2; die "unknown option: $1"; }
			shift "$PIPELINE_SHIFT"
			;;
	esac
done

[[ -n "$TEACHER_URL" ]] || { usage >&2; die "--teacher-url is required"; }
TEACHER_URL="${TEACHER_URL%/}"
TOKEN="${TOKEN:-${TEACHER_API_TOKEN:-}}"
validate_pipeline_options

select_runtime gpu
prepare_app_env train
wait_for_http "$TEACHER_URL/healthz" '"ready"' 60
smoke_test_teacher "$TEACHER_URL" "$TOKEN"
collect_teacher_data "$TEACHER_URL" "$TOKEN"
prepare_and_train
log "done; the adapter and training manifest are in the config's output_dir"
