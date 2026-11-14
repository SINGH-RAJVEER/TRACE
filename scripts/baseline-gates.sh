#!/usr/bin/env bash
# Phase 0 and Phase 1 on the GPU host: record the machine, pilot throughput, evaluate the base and
# the teacher on the same held-out problems, apply the headroom gate, collect clean responses,
# unload the teacher, train the curve, and apply the clean-student gate. Stops at the first failed
# step or gate. Everything is logged under the run directory; rerunning an ID resumes.
#
# Model work is serialized: every GPU step holds an exclusive lock on GPU_LOCK, and the teacher is
# stopped before training starts, so it is never served while a student trains.
#
# Usage: scripts/baseline-gates.sh [RUN_ID]    (or `just baseline-gates [RUN_ID]`)
#
# Required, with no defaults because each is a study declaration:
#   MARGIN_POINTS           preregistered teacher-base margin in points (proposed: 10)
#   BENIGN_SCHEDULE         frozen benign gateway schedule directory for the teacher pilot
#   TEACHER_PREFLIGHT_ARGS  declared capacity thresholds for `distill gateway teacher-preflight`
#   TEACHER_TERMS_REVIEW_ID review identifier that `distill baseline collect` records
set -uo pipefail

if [ "${1:-}" = --help ]; then
	cat <<'HELP'
Usage: just baseline-gates [RUN_ID]
GPU workflow: teacher headroom, clean collection, student training and the clean gate.
Requires MARGIN_POINTS, BENIGN_SCHEDULE, TEACHER_PREFLIGHT_ARGS and TEACHER_TERMS_REVIEW_ID.
This command is separate from just start and does not qualify the campaign detector.
HELP
	exit 0
fi
if [ "$#" -gt 1 ]; then
	echo "Usage: just baseline-gates [RUN_ID]" >&2
	exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || exit 1

: "${MARGIN_POINTS:?declare the preregistered headroom margin in points}"
: "${BENIGN_SCHEDULE:?name the frozen benign schedule directory}"
: "${TEACHER_PREFLIGHT_ARGS:?declare the teacher capacity thresholds}"
: "${TEACHER_TERMS_REVIEW_ID:?name the teacher terms review}"

RUN_ID=${1:-${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}
RUN=${RUN_ROOT:-artifacts/baseline-gates}/$RUN_ID
TASK_DIR=${TASK_DIR:-data/math}
SUITE=${SUITE:-$TASK_DIR/development-suite.json}
GOLD=${GOLD:-$TASK_DIR/gold-answers.jsonl}
POOL=${POOL:-$TASK_DIR/attacker.jsonl}
COLLECT_COUNT=${COLLECT_COUNT:-1200}
CONCURRENCY=${CONCURRENCY:-4}
INDEX=${INDEX:-readings/index.tsv}
GPU_LOCK=${GPU_LOCK:-${XDG_RUNTIME_DIR:-$ROOT/artifacts}/trace-gpu.lock}
VLLM_BIN=${VLLM_BIN:-.venv-vllm/bin/vllm}
VLLM_PORT=${VLLM_PORT:-8001}
TEACHER_MODEL=${TEACHER_UPSTREAM_MODEL:-Qwen/Qwen3.5-9B}
TEACHER_MODEL_REVISION=${TEACHER_MODEL_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}
TEACHER_URL=http://${TEACHER_HOST:-127.0.0.1}:${TEACHER_PORT:-8000}

UV=(uv run --frozen --extra eval --extra embed --extra train)
LOGS=$RUN/logs
STEPS=$RUN/steps
mkdir -p "$LOGS" "$STEPS"
mkdir -p "$(dirname "$GPU_LOCK")"

exec 9>>"$RUN/.lock"
flock -n 9 || { echo "another baseline-gates run holds $RUN" >&2; exit 1; }
exec > >(tee -a "$RUN/run.log") 2>&1

note() { echo "[$(date -u +'%F %T') UTC] $*"; }
stop() {
	note "stopped: $*"
	stop_teacher
	exit 1
}

# step NAME COMMAND...: run once, log to logs/NAME.log, and mark it done on success.
step() {
	local name=$1
	shift
	if [ -f "$STEPS/$name.done" ]; then
		note "skip $name (done)"
		return 0
	fi
	note "start $name"
	{
		echo "# $(date -u +%FT%TZ) $*"
		"$@"
	} >> "$LOGS/$name.log" 2>&1
	local code=$?
	if [ "$code" -ne 0 ]; then
		note "FAILED $name (exit $code); last lines of $LOGS/$name.log:"
		tail -n 20 "$LOGS/$name.log" | sed 's/^/    /'
		return "$code"
	fi
	touch "$STEPS/$name.done"
	note "done $name"
}
# gpu_step NAME COMMAND...: the same, holding the machine's GPU lock for its duration.
gpu_step() {
	local name=$1
	shift
	step "$name" flock "$GPU_LOCK" "$@"
}

PIDS=()
TEACHER_LOCK_FD=
# Another teacher would answer the health checks and stay loaded while a student trains.
teacher_down() {
	! curl -fsS "http://127.0.0.1:$VLLM_PORT/health" > /dev/null 2>&1 \
		&& ! curl -fsS "$TEACHER_URL/healthz" > /dev/null 2>&1
}
start_teacher() {
	teacher_down || stop "a teacher is already serving at $TEACHER_URL or port $VLLM_PORT; stop it"
	# Holding the lock while serving keeps every other GPU step out until the teacher is unloaded.
	exec 8>>"$GPU_LOCK"
	flock 8
	TEACHER_LOCK_FD=8
	note "starting vLLM ($TEACHER_MODEL@$TEACHER_MODEL_REVISION)"
	[ -x "$VLLM_BIN" ] || stop "no vLLM at $VLLM_BIN"
	VLLM_BIN="$VLLM_BIN" VLLM_PORT="$VLLM_PORT" TEACHER_UPSTREAM_MODEL="$TEACHER_MODEL" \
		TEACHER_MODEL_REVISION="$TEACHER_MODEL_REVISION" bash scripts/teacher-vllm.sh \
		>> "$LOGS/vllm.log" 2>&1 8>&- &
	PIDS+=($!)
	wait_http "http://127.0.0.1:$VLLM_PORT/health" "${VLLM_STARTUP_TIMEOUT:-1800}" $! \
		|| stop "vLLM did not start"
	TEACHER_UPSTREAM_MODEL=$TEACHER_MODEL TEACHER_UPSTREAM_BASE_URL=http://127.0.0.1:$VLLM_PORT/v1 \
		"${UV[@]}" teacher-api \
		>> "$LOGS/teacher-api.log" 2>&1 8>&- &
	PIDS+=($!)
	wait_http "$TEACHER_URL/healthz" 300 $! || stop "teacher API did not start"
}
stop_teacher() {
	for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done
	for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null; done
	PIDS=()
	if [ -n "$TEACHER_LOCK_FD" ]; then
		flock -u 8
		exec 8>&-
		TEACHER_LOCK_FD=
	fi
}
wait_http() { # URL SECONDS PID
	local deadline=$((SECONDS + $2))
	until curl -fsS "$1" > /dev/null 2>&1; do
		kill -0 "$3" 2>/dev/null || return 1
		[ $SECONDS -lt $deadline ] || return 1
		sleep 5
	done
}
trap 'stop interrupted' INT TERM
trap stop_teacher EXIT

# 1. The machine as it is. A failing nvidia-smi stops the run: nothing about the card is assumed.
hardware() {
	nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || return
	uv pip freeze
}
step hardware hardware || stop "could not record the GPU and library versions"
cp -n "$LOGS/hardware.log" "$RUN/hardware.txt"

# 2. Student pilot, before anything else loads a model. A pilot is sizing, not a result.
gpu_step pilot-student "${UV[@]}" distill baseline pilot --output "$RUN/pilot" \
	--suite "$SUITE" --pool "$POOL" --gold "$GOLD" || stop "student pilot failed"

# 3. The pinned base, with no teacher loaded.
gpu_step evaluate-base "${UV[@]}" distill baseline evaluate-base --suite "$SUITE" \
	--output "$RUN/base-scores.json" || stop "base evaluation failed"

# 4. Teacher pilot and evaluation. A capacity failure stops before scores can be misread.
if [ ! -f "$STEPS/stop-teacher.done" ]; then
	start_teacher
fi
# shellcheck disable=SC2086
step pilot-teacher "${UV[@]}" distill gateway teacher-preflight --input-dir "$BENIGN_SCHEDULE" \
	--output "$RUN/teacher-preflight.json" --teacher-url "$TEACHER_URL" \
	--concurrency "$CONCURRENCY" $TEACHER_PREFLIGHT_ARGS || stop "teacher preflight failed"
step evaluate-teacher "${UV[@]}" distill baseline evaluate-teacher --suite "$SUITE" \
	--output "$RUN/teacher-scores.json" --teacher-url "$TEACHER_URL" \
	--teacher-model "$TEACHER_MODEL" --teacher-revision "$TEACHER_MODEL_REVISION" \
	--concurrency "$CONCURRENCY" || stop "teacher evaluation failed"

# 5. Headroom gate. It exits nonzero on failure, so nothing is collected or trained.
step gate-headroom "${UV[@]}" distill baseline headroom --base-scores "$RUN/base-scores.json" \
	--teacher-scores "$RUN/teacher-scores.json" --suite "$SUITE" \
	--margin-points "$MARGIN_POINTS" --output "$RUN/gate-headroom.json" --index "$INDEX" \
	|| stop "headroom gate did not pass; see $RUN/gate-headroom.json"

# 6. One clean collection from a direct teacher, then freeze it.
step collect "${UV[@]}" distill baseline collect --pool "$POOL" --count "$COLLECT_COUNT" \
	--output "$RUN/collection/ledger.jsonl" --teacher-url "$TEACHER_URL" \
	--concurrency "$CONCURRENCY" || stop "collection failed"
step manifest "${UV[@]}" distill acquisition manifest --ledger "$RUN/collection/ledger.jsonl" \
	--response-policy clean --output "$RUN/acquisition-manifest.json" \
	|| stop "acquisition manifest failed"

# 7. Unload the teacher before any student trains.
stop_teacher
touch "$STEPS/stop-teacher.done"
teacher_down || stop "a teacher is still serving; nothing trains until it is unloaded"
note "teacher stopped"

# 8. Train clean and gold students from the pinned base.
step plan "${UV[@]}" distill learning-curve plan-math --output "$RUN/curve" \
	--ledger "$RUN/collection/ledger.jsonl" --suite "$SUITE" --gold "$GOLD" \
	--include-pending \
	|| stop "learning-curve plan failed"
gpu_step train "${UV[@]}" distill learning-curve execute --output "$RUN/curve" \
	|| stop "training failed"
step summary "${UV[@]}" distill learning-curve summary --output "$RUN/curve" \
	|| stop "learning-curve summary failed"

# 9. Clean-student gate. A failure means fix the learning experiment, not move on to defenses.
step gate-clean "${UV[@]}" distill baseline clean --curve "$RUN/curve" \
	--headroom "$RUN/gate-headroom.json" --manifest "$RUN/acquisition-manifest.json" \
	--output "$RUN/gate-clean.json" --index "$INDEX" \
	|| stop "clean-student gate did not pass; see $RUN/gate-clean.json"

note "both gates passed; see $RUN/gate-headroom.json and $RUN/gate-clean.json"
