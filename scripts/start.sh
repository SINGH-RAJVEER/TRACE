#!/usr/bin/env bash
# End-to-end study run: corpus, benign roles, calibration, schedules, teacher, then one fresh
# harness per attack and arm, benign utility per arm, and reports. Everything is logged under
# the run directory. Rerunning with the same RUN_ID resumes: finished steps are skipped,
# unfinished offline steps are redone, and a partial campaign stops the run so it is preserved.
#
# At the end, or when a run that reached the teacher stops, its evidence is copied to
# readings/vN (the next free number, or READING=vN) without credentials, state databases or keys.
#
# Usage: scripts/start.sh [RUN_ID]    (or `just start [RUN_ID]`)
# Settings come from the environment and .env; see the defaults below.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || exit 1

RUN_ID=${1:-${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}
RUN=${RUN_ROOT:-artifacts/runs}/$RUN_ID
TASKS_DIR=${TASKS_DIR:-data}
SEED=${SEED:-42}
BUDGET=${BUDGET:-6000}
PARTITION=${PARTITION:-exploratory}
ATTACKS=${ATTACKS:-A1 A2 A3 A5 A6 A7}
ARMS=${ARMS:-full instantaneous quota}
DOMAINS=${DOMAINS:-ag_news,squad,gsm8k,coding}
BENIGN_ACCOUNTS=${BENIGN_ACCOUNTS:-125}
WORKLOAD_VERSION=${WORKLOAD_VERSION:-gateway-clients-v3}
A4_INTERVAL=${A4_INTERVAL:-60}
CONCURRENCY=${CONCURRENCY:-8}
QUALIFICATION_WORKERS=${QUALIFICATION_WORKERS:-16}
CALIBRATION_DEVICE=${CALIBRATION_DEVICE:-cpu}
ALLOW_UNDETECTED=${ALLOW_UNDETECTED:-0}
UTILITY=${UTILITY:-1}
HARNESS_RUN_PORT=${HARNESS_RUN_PORT:-8003}
VLLM_BIN=${VLLM_BIN:-.venv-vllm/bin/vllm}
VLLM_PORT=${VLLM_PORT:-8001}
TEACHER_MODEL=${TEACHER_UPSTREAM_MODEL:-Qwen/Qwen3.5-9B}
TEACHER_MODEL_REVISION=${TEACHER_MODEL_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}
TEACHER_URL=http://${TEACHER_HOST:-127.0.0.1}:${TEACHER_PORT:-8000}
# Declared capacity thresholds for `distill gateway teacher-preflight`; empty skips the gate.
TEACHER_PREFLIGHT_ARGS=${TEACHER_PREFLIGHT_ARGS:-}

UV=(uv run --frozen --extra eval --extra embed)
TOOLS=(env PYTHONPATH="$ROOT/scripts" "${UV[@]}" python "$ROOT/scripts/run_tools.py")
LOGS=$RUN/logs
STEPS=$RUN/steps
mkdir -p "$LOGS" "$STEPS" "$RUN/provenance"

exec 9>"$RUN/.lock"
flock -n 9 || { echo "another start is already running for $RUN" >&2; exit 1; }
# Every line of this script's own output goes to the console and the run log.
exec > >(tee -a "$RUN/run.log") 2>&1

note() { echo "[$(date -u +'%F %T') UTC] $*"; }
PIDS=()
# Wait only for started services: a bare `wait` would also wait for the run-log tee, which
# cannot exit before this script does.
cleanup() {
	for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done
	for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null; done
	PIDS=()
}
# Runs that reached the teacher are exported to readings/, complete or not; earlier failures are
# offline preparation only and can be exported by hand with `just reading RUN_ID`.
CONTACTED=0
finish() {
	trap - INT TERM
	echo "$1 $(date -u +%FT%TZ) ${2:-}" > "$RUN/exit"
	note "run exited with $1${2:+: $2}"
	cleanup
	if [ "${EXPORT:-1}" = 1 ] && { [ "$1" = 0 ] || [ "$CONTACTED" = 1 ]; }; then
		note "exporting to readings/: $("${TOOLS[@]}" export "$RUN" readings "${READING:-}" 2>&1)"
	fi
	exit "$1"
}
trap 'finish 130 interrupted' INT TERM
trap cleanup EXIT

# step NAME COMMAND...: run once, log its output to logs/NAME.log, time it, mark it done.
step() {
	local name=$1
	shift
	if [ -f "$STEPS/$name.done" ]; then
		note "skip $name (done)"
		return 0
	fi
	note "start $name"
	local started=$SECONDS
	{
		echo "# $(date -u +%FT%TZ) $*"
		"$@"
	} >> "$LOGS/$name.log" 2>&1
	local code=$?
	local took=$((SECONDS - started))
	printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$name" "$code" "$took" >> "$RUN/steps.tsv"
	if [ "$code" -ne 0 ]; then
		note "FAILED $name after ${took}s (exit $code); last lines of $LOGS/$name.log:"
		tail -n 20 "$LOGS/$name.log" | sed 's/^/    /'
		return "$code"
	fi
	touch "$STEPS/$name.done"
	note "done $name in ${took}s"
}
# Offline outputs are deterministic, so an unfinished one is removed and rebuilt.
fresh() { [ -f "$STEPS/$1.done" ] || rm -rf "${@:2}"; }

# An empty PATTERN only requires success: vLLM's /health answers 200 with an empty body.
wait_http() { # URL PATTERN SECONDS [PID]
	local deadline=$((SECONDS + $3))
	until if [ -z "$2" ]; then curl -fsS -o /dev/null "$1" 2>/dev/null
	else curl -fsS "$1" 2>/dev/null | grep -q -- "$2"; fi; do
		if [ -n "${4:-}" ] && ! kill -0 "$4" 2>/dev/null; then return 1; fi
		[ $SECONDS -lt $deadline ] || return 1
		sleep 5
	done
}

# 0. Provenance: code, configuration, environment and hardware at the start of this attempt.
attempt=$RUN/provenance/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$attempt"
{
	echo "run $RUN_ID"
	echo "commit $(git rev-parse HEAD)"
	echo "branch $(git rev-parse --abbrev-ref HEAD)"
	echo "dirty $(git status --porcelain | wc -l) paths"
} > "$attempt/code.txt"
git status --porcelain > "$attempt/git-status.txt"
git diff HEAD > "$attempt/git-diff.patch"
cat > "$attempt/config.env" <<EOF
TASKS_DIR=$TASKS_DIR
SEED=$SEED
BUDGET=$BUDGET
PARTITION=$PARTITION
ATTACKS=$ATTACKS
ARMS=$ARMS
DOMAINS=$DOMAINS
BENIGN_ACCOUNTS=$BENIGN_ACCOUNTS
WORKLOAD_VERSION=$WORKLOAD_VERSION
A4_INTERVAL=$A4_INTERVAL
CONCURRENCY=$CONCURRENCY
QUALIFICATION_WORKERS=$QUALIFICATION_WORKERS
CALIBRATION_DEVICE=$CALIBRATION_DEVICE
ALLOW_UNDETECTED=$ALLOW_UNDETECTED
UTILITY=$UTILITY
TEACHER_MODEL=$TEACHER_MODEL
TEACHER_MODEL_REVISION=$TEACHER_MODEL_REVISION
TEACHER_PREFLIGHT_ARGS=$TEACHER_PREFLIGHT_ARGS
EOF
redact() { sed -E 's/^([A-Z0-9_]*(TOKEN|KEY|SECRET|PASSWORD)[A-Z0-9_]*)=.+/\1=<redacted>/'; }
env | grep -E '^(TEACHER|HARNESS|VLLM|HF|CUDA)_' | sort | redact > "$attempt/environment.env"
"${UV[@]}" python -c 'import platform, sys; print(sys.version); print(platform.platform())' \
	> "$attempt/python.txt" 2>&1
uv pip freeze > "$attempt/packages.txt" 2>&1
sha256sum uv.lock > "$attempt/uv.lock.sha256"
{ nproc; free -g; nvidia-smi 2>&1 || echo "no nvidia-smi"; } > "$attempt/hardware.txt"
note "run $RUN_ID at $(git rev-parse --short HEAD) in $RUN; provenance in $attempt"
note "attacks [$ATTACKS], arms [$ARMS], budget $BUDGET, partition $PARTITION, seed $SEED"

# 1. Environment.
step setup uv sync --frozen --all-packages --extra eval --extra embed || finish 1 "setup failed"

# 2. Corpus: each domain's pools with sealed attacker partitions. Shared across runs in TASKS_DIR.
IFS=, read -ra domains <<< "$DOMAINS"
for task in "${domains[@]}"; do
	directory=$("${UV[@]}" python -c 'import sys; from pathlib import Path
from distillation.tasks import task_directory
print(task_directory(sys.argv[1], seed=int(sys.argv[2]), root=Path(sys.argv[3])))' \
		"$task" "$SEED" "$TASKS_DIR")
	if "${TOOLS[@]}" corpus-ready "$directory"; then
		note "corpus $task ready in $directory"
	else
		rm -f "$STEPS/corpus-$task.done"
		if [ -e "$directory/manifest.json" ]; then
			finish 1 "$directory holds a build without sealed partitions; move it aside to rebuild"
		fi
		args=(distill tasks build --task "$task" --seed "$SEED" --output-dir "$directory")
		[ "$task" != coding ] || args+=(--qualification-workers "$QUALIFICATION_WORKERS")
		step "corpus-$task" "${UV[@]}" "${args[@]}" || finish 1 "corpus build failed for $task"
	fi
	cp "$directory/manifest.json" "$RUN/provenance/corpus-$task.manifest.json"
	# Independent-source pools for A3 and A7, built once per corpus.
	if [[ " $ATTACKS " == *" A3 "* || " $ATTACKS " == *" A7 "* ]]; then
		if [ -f "$directory/external/manifest.json" ]; then
			note "external $task ready in $directory/external"
		else
			rm -rf "$directory/external" "$STEPS/external-$task.done"
			step "external-$task" "${UV[@]}" distill gateway external --task "$task" \
				--tasks-dir "$TASKS_DIR" || finish 1 "external pool failed for $task"
		fi
		cp "$directory/external/manifest.json" "$RUN/provenance/external-$task.manifest.json"
	fi
done

# 3. Benign roles and their detector inputs.
BENIGN=$RUN/benign
for role in fitting score_calibration policy_calibration benign_test; do
	fresh "benign-$role" "$BENIGN/$role"
	step "benign-$role" "${UV[@]}" distill gateway benign --output "$BENIGN/$role" --role "$role" \
		--accounts "$BENIGN_ACCOUNTS" --tasks-dir "$TASKS_DIR" --domains "$DOMAINS" --seed "$SEED" \
		--workload-version "$WORKLOAD_VERSION" || finish 1 "benign $role failed"
	fresh "detector-$role" "$BENIGN/$role.detector.jsonl"
	step "detector-$role" "${UV[@]}" distill gateway detector-input --input-dir "$BENIGN/$role" \
		--output "$BENIGN/$role.detector.jsonl" || finish 1 "detector input failed for $role"
done

# 4. Calibration and its adequacy verdict.
CALIBRATION=$RUN/calibration
fresh calibrate "$CALIBRATION"
step calibrate "${UV[@]}" harness calibrate --fitting "$BENIGN/fitting.detector.jsonl" \
	--score-calibration "$BENIGN/score_calibration.detector.jsonl" \
	--policy-calibration "$BENIGN/policy_calibration.detector.jsonl" \
	--output-dir "$CALIBRATION" --task gateway --device "$CALIBRATION_DEVICE" --seed "$SEED" \
	|| finish 1 "calibration failed"
fresh adequacy "$RUN/adequacy.json"
# An inadequate verdict exits nonzero; it is recorded, and the detectability gate still applies.
step adequacy "${UV[@]}" harness adequacy --calibration "$CALIBRATION" \
	--output "$RUN/adequacy.json" || note "WARNING: calibration adequacy failed; see $RUN/adequacy.json"

# 5. Frozen attack schedules.
for attack in $ATTACKS; do
	args=(--attack "$attack" --budget "$BUDGET" --partition "$PARTITION" --seed "$SEED"
		--tasks-dir "$TASKS_DIR" --domains "$DOMAINS" --campaign-id "$RUN_ID-$attack")
	[ "$attack" != A4 ] || args+=(--interval-seconds "$A4_INTERVAL")
	fresh "schedule-$attack" "$RUN/schedules/$attack"
	step "schedule-$attack" "${UV[@]}" distill gateway schedule --output "$RUN/schedules/$attack" \
		"${args[@]}" || finish 1 "schedule failed for $attack"
done

# 6. Teacher: reuse a healthy one, otherwise start vLLM and the teacher API.
if curl -fsS "$TEACHER_URL/healthz" > /dev/null 2>&1; then
	note "using the teacher API already serving at $TEACHER_URL"
else
	if ! curl -fsS "http://127.0.0.1:$VLLM_PORT/health" > /dev/null 2>&1; then
		[ -x "$VLLM_BIN" ] || finish 1 "no teacher at $TEACHER_URL and no vLLM at $VLLM_BIN"
		note "starting vLLM ($TEACHER_MODEL@$TEACHER_MODEL_REVISION) on port $VLLM_PORT"
		# shellcheck disable=SC2086
		"$VLLM_BIN" serve "$TEACHER_MODEL" --revision "$TEACHER_MODEL_REVISION" \
			--served-model-name "$TEACHER_MODEL" --host 127.0.0.1 --port "$VLLM_PORT" \
			--max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
			--gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
			--default-chat-template-kwargs '{"enable_thinking": false}' --language-model-only \
			${VLLM_EXTRA_ARGS:-} >> "$LOGS/vllm.log" 2>&1 &
		PIDS+=($!)
		wait_http "http://127.0.0.1:$VLLM_PORT/health" "" "${VLLM_STARTUP_TIMEOUT:-1800}" $! \
			|| finish 1 "vLLM did not start; see $LOGS/vllm.log"
	fi
	note "starting the teacher API at $TEACHER_URL"
	TEACHER_UPSTREAM_BASE_URL=http://127.0.0.1:$VLLM_PORT/v1 "${UV[@]}" teacher-api \
		>> "$LOGS/teacher-api.log" 2>&1 &
	PIDS+=($!)
	wait_http "$TEACHER_URL/healthz" "" 300 $! || finish 1 "teacher API did not start"
fi
note "teacher ready"
CONTACTED=1

# 7. Teacher capacity under the benign workload, when thresholds are declared.
if [ -n "$TEACHER_PREFLIGHT_ARGS" ]; then
	fresh preflight "$RUN/teacher-preflight.json"
	# shellcheck disable=SC2086
	step preflight "${UV[@]}" distill gateway teacher-preflight --input-dir "$BENIGN/benign_test" \
		--output "$RUN/teacher-preflight.json" --teacher-url "$TEACHER_URL" \
		--concurrency "$CONCURRENCY" $TEACHER_PREFLIGHT_ARGS || finish 1 "teacher preflight failed"
	step capacity "${UV[@]}" distill gateway require-teacher-capacity \
		--evidence "$RUN/teacher-preflight.json" --input-dir "$BENIGN/benign_test" \
		|| finish 1 "teacher capacity gate failed"
else
	note "WARNING: TEACHER_PREFLIGHT_ARGS unset; skipping the teacher capacity gate"
fi

# 8. One fresh harness per attack and arm, then benign utility per arm.
CAPS=$("${TOOLS[@]}" caps "$BENIGN/benign_test/manifest.json") || finish 1 "no gateway caps"
HARNESS_PID=
start_harness() { # ARM DIR SCHEDULE_REQUESTS
	local arm=$1 dir=$2 policy accumulator=paper
	case $arm in
		full) policy=mmd ;;
		instantaneous) policy=mmd accumulator=instantaneous ;;
		quota | rate | allow) policy=$arm ;;
		*) note "unknown arm $arm"; return 1 ;;
	esac
	"${TOOLS[@]}" accounts "$dir" "$3" >> "$dir/harness.log" 2>&1 || return 1
	local settings=(
		HARNESS_HOST=127.0.0.1 HARNESS_PORT="$HARNESS_RUN_PORT"
		HARNESS_TEACHER_URL="$TEACHER_URL" HARNESS_TEACHER_TOKEN="${TEACHER_API_TOKEN:-}"
		HARNESS_TASK=gateway HARNESS_GATEWAY_CAPS="$CAPS" HARNESS_CALIBRATION_DIR="$CALIBRATION"
		HARNESS_ACCOUNTS_FILE="$dir/accounts.json" HARNESS_TRUST_ACTOR_HEADER=false
		HARNESS_TEST_CLOCK=false HARNESS_POLICY="$policy" HARNESS_ACCUMULATOR="$accumulator"
		HARNESS_POLICY_VERSION="${HARNESS_POLICY_VERSION:-legacy-v1}" HARNESS_L4_ACTION=refuse
		HARNESS_STATE_DB="$dir/state.sqlite3" HARNESS_DECISION_LOG="$dir/decisions.jsonl"
		HARNESS_DECEPTION_KEY_FILE="$RUN/deception.key"
	)
	printf '%s\n' "${settings[@]}" | redact > "$dir/harness.env"
	env | grep '^HARNESS_' | redact | sed 's/^/inherited /' >> "$dir/harness.env"
	env "${settings[@]}" "${UV[@]}" harness-api >> "$dir/harness.log" 2>&1 &
	HARNESS_PID=$!
	PIDS+=("$HARNESS_PID")
	wait_http "http://127.0.0.1:$HARNESS_RUN_PORT/healthz" '"ready"' 600 "$HARNESS_PID"
}
stop_harness() {
	[ -z "$HARNESS_PID" ] || { kill "$HARNESS_PID" 2>/dev/null; wait "$HARNESS_PID" 2>/dev/null; }
	HARNESS_PID=
}
if curl -fsS "http://127.0.0.1:$HARNESS_RUN_PORT/healthz" > /dev/null 2>&1; then
	finish 1 "port $HARNESS_RUN_PORT is already serving; stop that harness or set HARNESS_RUN_PORT"
fi

printf 'attack\tarm\toutcome\n' > "$RUN/summary.tsv"
for attack in $ATTACKS; do
	for arm in $ARMS; do
		dir=$RUN/campaigns/$attack/$arm
		name=campaign-$attack-$arm
		if [ -f "$STEPS/$name.done" ]; then
			printf '%s\t%s\t%s\n' "$attack" "$arm" "$(cat "$dir/outcome")" >> "$RUN/summary.tsv"
			continue
		fi
		[ ! -e "$dir/campaign" ] || finish 2 "partial campaign in $dir; preserve it, then rerun with a new RUN_ID"
		rm -rf "$dir"
		mkdir -p "$dir"
		start_harness "$arm" "$dir" "$RUN/schedules/$attack/requests.jsonl" \
			|| finish 1 "harness did not start for $attack/$arm; see $dir/harness.log"
		policy=$arm
		args=(--input-dir "$RUN/schedules/$attack" --output "$dir/campaign"
			--teacher-url "http://127.0.0.1:$HARNESS_RUN_PORT" --keys-file "$dir/credentials.json"
			--decision-log "$dir/decisions.jsonl" --concurrency "$CONCURRENCY")
		if [ "$arm" = full ] || [ "$arm" = instantaneous ]; then
			policy=mmd
			args+=(--calibration "$CALIBRATION" --benign-test "$BENIGN/benign_test.detector.jsonl")
			[ "$ALLOW_UNDETECTED" != 1 ] || args+=(--allow-undetected)
		fi
		note "campaign $attack/$arm started"
		step "$name" "${UV[@]}" distill gateway collect --policy "$policy" "${args[@]}"
		code=$?
		stop_harness
		if [ "$code" -eq 1 ] && [ "$policy" = mmd ] && [ ! -e "$dir/campaign" ]; then
			# The detectability gate refused the arm before any request was sent.
			echo "gate refused: attacker not escalated beyond held-out benign accounts" > "$dir/outcome"
			cp "$LOGS/$name.log" "$dir/gate.log"
			touch "$STEPS/$name.done"
			note "campaign $attack/$arm: detectability gate refused; no requests sent"
		elif [ "$code" -ne 0 ]; then
			finish 1 "campaign $attack/$arm failed; see $LOGS/$name.log"
		else
			"${UV[@]}" distill gateway evaluation-report \
				--events "$dir/campaign/campaign.events.jsonl" > "$dir/evaluation-report.json" \
				2>> "$LOGS/$name.log"
			"${TOOLS[@]}" summary "$dir/campaign/campaign.json" > "$dir/outcome"
			note "campaign $attack/$arm done: $(cat "$dir/outcome")"
		fi
		printf '%s\t%s\t%s\n' "$attack" "$arm" "$(cat "$dir/outcome")" >> "$RUN/summary.tsv"
	done
done

if [ "$UTILITY" = 1 ]; then
	for arm in $ARMS; do
		dir=$RUN/utility/$arm
		name=utility-$arm
		[ -f "$STEPS/$name.done" ] && { note "skip $name (done)"; continue; }
		[ ! -e "$dir/utility.json" ] || finish 2 "partial utility run in $dir; preserve it, then rerun with a new RUN_ID"
		rm -rf "$dir"
		mkdir -p "$dir"
		start_harness "$arm" "$dir" "$BENIGN/benign_test/requests.jsonl" \
			|| finish 1 "harness did not start for utility/$arm; see $dir/harness.log"
		step "$name" "${TOOLS[@]}" utility "$BENIGN/benign_test" "$TASKS_DIR" "$dir/credentials.json" \
			"$dir/utility.json" "http://127.0.0.1:$HARNESS_RUN_PORT" "$CONCURRENCY"
		code=$?
		stop_harness
		[ "$code" -eq 0 ] || finish 1 "utility failed for $arm; see $LOGS/$name.log"
	done
fi

note "summary:"
column -t -s $'\t' "$RUN/summary.tsv" | sed 's/^/    /'
finish 0
