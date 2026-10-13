# shellcheck shell=bash disable=SC2034
# Shared helpers for the launcher scripts. Source this file; do not execute it.
#
# Runtime selection, in order:
#   1. devenv  NixOS with devenv installed. The calling script re-runs itself inside
#              `devenv shell`, which provides Python, uv, a compiler, and library paths.
#   2. uv      uv on PATH. Missing system tools are reported before anything starts.
#   3. pip     Neither of the above. A Python 3.14 interpreter and pip build .venv-pip.
# Set DISTILL_RUNTIME=devenv|uv|pip to force one.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/artifacts/logs"
DEFAULT_TEACHER_MODEL="Qwen/Qwen3-14B"
DEFAULT_TEACHER_REVISION="40c069824f4251a91eefaf281ebe4c544efd3e18"

RUNTIME=""
APP_BIN=""
VLLM=""
BG_PIDS=()
BG_NAMES=()

log() {
	printf '[%s] %s\n' "${SCRIPT_NAME:-distill}" "$*" >&2
}

die() {
	log "error: $*"
	exit 1
}

is_nixos() {
	[[ -e /etc/NIXOS ]] || grep -qs '^ID=nixos$' /etc/os-release
}

# Re-executes the calling script inside devenv on NixOS. Call before anything else.
enter_devenv_if_nixos() {
	local script="$1"
	shift
	local requested="${DISTILL_RUNTIME:-auto}"
	if [[ -n "${DEVENV_ROOT:-}" ]]; then
		[[ "$requested" == auto ]] && export DISTILL_RUNTIME=devenv
		return 0
	fi
	if [[ "$requested" != auto && "$requested" != devenv ]]; then
		return 0
	fi
	if ! is_nixos && [[ "$requested" != devenv ]]; then
		return 0
	fi
	if ! command -v devenv > /dev/null 2>&1; then
		[[ "$requested" == devenv ]] && die "DISTILL_RUNTIME=devenv but devenv is not installed"
		log "NixOS detected but devenv is not installed; falling back to uv or pip"
		return 0
	fi
	log "NixOS detected; re-running inside devenv shell"
	cd "$REPO_ROOT" || exit 1
	export DISTILL_RUNTIME=devenv
	exec devenv shell -- bash "$script" "$@"
}

# Exports .env values that are not already set in the environment.
load_dotenv() {
	local file="$REPO_ROOT/.env" line key value
	[[ -f "$file" ]] || return 0
	while IFS= read -r line || [[ -n "$line" ]]; do
		line="${line%$'\r'}"
		[[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
		[[ "$line" == *=* ]] || continue
		key="${line%%=*}"
		key="${key#export }"
		key="${key//[[:space:]]/}"
		value="${line#*=}"
		if [[ "$value" =~ ^\"(.*)\"$ || "$value" =~ ^\'(.*)\'$ ]]; then
			value="${BASH_REMATCH[1]}"
		fi
		[[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
		printenv "$key" > /dev/null 2>&1 || export "$key=$value"
	done < "$file"
}

# Prints the first interpreter in the arguments that is Python >= major.minor.
find_python() {
	local minimum="$1" candidate
	shift
	for candidate in "$@"; do
		command -v "$candidate" > /dev/null 2>&1 || continue
		if "$candidate" -c "import sys; sys.exit(sys.version_info[:2] < tuple(map(int, '$minimum'.split('.'))))" \
			2> /dev/null; then
			command -v "$candidate"
			return 0
		fi
	done
	return 1
}

find_vllm_python() {
	local python="${VLLM_PYTHON:-3.12}"
	[[ "$python" =~ ^[0-9.]+$ ]] && python="python$python"
	find_python 3.10 "$python" python3.12 python3.13 python3.11 python3
}

vllm_installed() {
	[[ -n "${VLLM_BIN:-}" || -x "$REPO_ROOT/.venv-vllm/bin/vllm" ]] || command -v vllm > /dev/null 2>&1
}

# Picks the runtime and reports missing tools. Pass "gpu" when the script needs CUDA and
# "vllm" when it starts vLLM.
select_runtime() {
	local needs_gpu="" needs_vllm="" flag missing=() warnings=()
	for flag in "$@"; do
		case "$flag" in
			gpu) needs_gpu=1 ;;
			vllm) needs_vllm=1 ;;
		esac
	done
	RUNTIME="${DISTILL_RUNTIME:-auto}"
	if [[ "$RUNTIME" == auto ]]; then
		if command -v uv > /dev/null 2>&1; then
			RUNTIME=uv
		else
			RUNTIME=pip
		fi
	fi
	case "$RUNTIME" in
		devenv | uv)
			command -v uv > /dev/null 2>&1 || missing+=("uv (https://docs.astral.sh/uv/)")
			;;
		pip)
			warnings+=("using pip, which does not read uv.lock; install uv for locked dependencies")
			local python
			if python="$(find_python 3.14 python3.14 python3 python)"; then
				"$python" -c 'import ensurepip, venv' 2> /dev/null \
					|| missing+=("venv and ensurepip for $python (python3.14-venv on Debian and Ubuntu)")
			else
				missing+=("Python 3.14 (python3.14 on PATH)")
			fi
			if [[ -n "$needs_vllm" ]] && ! vllm_installed && ! find_vllm_python > /dev/null; then
				missing+=("Python 3.10 to 3.13 for vLLM (set VLLM_PYTHON)")
			fi
			;;
		*)
			die "unknown DISTILL_RUNTIME '$RUNTIME'; use devenv, uv, or pip"
			;;
	esac
	command -v curl > /dev/null 2>&1 || missing+=("curl")
	if [[ -n "$needs_gpu" ]]; then
		command -v nvidia-smi > /dev/null 2>&1 \
			|| warnings+=("nvidia-smi not found; vLLM and QLoRA training need an NVIDIA GPU and driver")
		local compiler
		if compiler="$(command -v cc || command -v gcc)"; then
			# A compiler without libc headers passes the check above but fails inside Triton.
			printf '#include <stdlib.h>\n' | "$compiler" -x c -fsyntax-only - > /dev/null 2>&1 \
				|| warnings+=("$compiler cannot find the C library headers; install build-essential (Debian and Ubuntu)")
		else
			warnings+=("no C compiler (cc or gcc); Triton kernels in vLLM and training compile with one")
		fi
	fi

	log "runtime: $RUNTIME"
	local item
	for item in "${warnings[@]+"${warnings[@]}"}"; do
		log "warning: $item"
	done
	if ((${#missing[@]})); then
		for item in "${missing[@]}"; do
			log "missing: $item"
		done
		die "install the missing tools above and run again"
	fi
}

# Installs the workspace packages and sets APP_BIN. Pass "train" for the training stack.
prepare_app_env() {
	local extra="${1:-}"
	cd "$REPO_ROOT" || exit 1
	if [[ "$RUNTIME" == pip ]]; then
		local python
		python="$(find_python 3.14 python3.14 python3 python)"
		APP_BIN="$REPO_ROOT/.venv-pip/bin"
		[[ -x "$APP_BIN/python" ]] || "$python" -m venv "$REPO_ROOT/.venv-pip"
		local distillation="./packages/distillation"
		[[ "$extra" == train ]] && distillation="./packages/distillation[train]"
		log "installing workspace packages with pip"
		"$APP_BIN/python" -m pip install --quiet --upgrade pip
		"$APP_BIN/python" -m pip install --quiet -e "$distillation" -e ./apps/teacher-api
	else
		local args=(sync --all-packages --inexact)
		[[ "$extra" == train ]] && args+=(--extra train)
		log "installing workspace packages with uv"
		uv "${args[@]}"
		APP_BIN="$REPO_ROOT/.venv/bin"
	fi
}

# Locates vLLM or installs it into .venv-vllm, separate from the Python 3.14 workspace.
prepare_vllm() {
	if [[ -n "${VLLM_BIN:-}" ]]; then
		VLLM="$VLLM_BIN"
	elif command -v vllm > /dev/null 2>&1; then
		VLLM="$(command -v vllm)"
	else
		local venv="$REPO_ROOT/.venv-vllm" spec="${VLLM_SPEC:-vllm}"
		local python="${VLLM_PYTHON:-3.12}"
		VLLM="$venv/bin/vllm"
		if [[ ! -x "$VLLM" ]]; then
			log "installing $spec into .venv-vllm (several GB)"
			if [[ "$RUNTIME" == pip ]]; then
				python="$(find_vllm_python)" \
					|| die "vLLM needs a Python interpreter between 3.10 and 3.13; set VLLM_PYTHON"
				"$python" -m venv "$venv"
				"$venv/bin/python" -m pip install --quiet --upgrade pip
				"$venv/bin/python" -m pip install "$spec"
			else
				uv venv --python "$python" "$venv"
				uv pip install --python "$venv/bin/python" "$spec"
			fi
		fi
	fi
	[[ -x "$VLLM" ]] || die "vLLM executable not found at $VLLM"
}

start_background() {
	local name="$1" logfile="$2"
	shift 2
	if [[ "$logfile" == - ]]; then
		"$@" &
	else
		mkdir -p "$LOG_DIR"
		"$@" > "$logfile" 2>&1 &
	fi
	BG_PIDS+=("$!")
	BG_NAMES+=("$name")
	log "started $name (pid $!)"
}

# Stops background services, newest first, with SIGTERM then SIGKILL after 30 seconds.
stop_background() {
	local index pid waited
	for ((index = ${#BG_PIDS[@]} - 1; index >= 0; index--)); do
		pid="${BG_PIDS[index]}"
		kill -0 "$pid" 2> /dev/null || continue
		log "stopping ${BG_NAMES[index]} (pid $pid)"
		kill -TERM "$pid" 2> /dev/null || true
		waited=0
		while kill -0 "$pid" 2> /dev/null && ((waited < 30)); do
			sleep 1
			waited=$((waited + 1))
		done
		if kill -0 "$pid" 2> /dev/null; then
			pkill -KILL -P "$pid" 2> /dev/null || true
			kill -KILL "$pid" 2> /dev/null || true
		fi
		wait "$pid" 2> /dev/null || true
	done
	BG_PIDS=()
	BG_NAMES=()
}

# Waits until a URL returns 2xx and its body contains the expected text.
wait_for_http() {
	local url="$1" expected="$2" timeout="$3" pid="${4:-}" logfile="${5:-}"
	local started=$SECONDS body
	log "waiting for $url"
	while ((SECONDS - started < timeout)); do
		if body="$(curl -fsS --max-time 5 "$url" 2> /dev/null)" && [[ "$body" == *"$expected"* ]]; then
			return 0
		fi
		if [[ -n "$pid" ]] && ! kill -0 "$pid" 2> /dev/null; then
			if [[ -n "$logfile" && -f "$logfile" ]]; then
				tail -n 40 "$logfile" >&2
				# Child processes such as the vLLM engine core log the root cause well above
				# the tail, so repeat the error lines from the whole log.
				log "error lines from $logfile:"
				grep -E ' ERROR |(Error|Exception)(:|$)' "$logfile" | tail -n 20 >&2 || true
			fi
			die "process for $url exited during startup"
		fi
		sleep 3
	done
	die "timed out after ${timeout}s waiting for $url"
}

# Sends one small generation through the teacher API to check auth and non-thinking mode.
smoke_test_teacher() {
	local teacher_url="${1%/}" token="$2" response status
	local auth=()
	[[ -n "$token" ]] && auth=(-H "Authorization: Bearer $token")
	response="$(mktemp)"
	status="$(curl -sS --max-time 300 -o "$response" -w '%{http_code}' \
		-H 'Content-Type: application/json' "${auth[@]+"${auth[@]}"}" \
		-d '{"messages":[{"role":"user","content":"Reply with the single word: ready"}],"temperature":0,"max_output_tokens":32}' \
		"$teacher_url/v1/teacher/generate")" || status=000
	if [[ "$status" != 200 ]]; then
		cat "$response" >&2
		rm -f "$response"
		die "teacher smoke test failed with HTTP $status"
	fi
	if grep -q '<think>' "$response"; then
		rm -f "$response"
		die "teacher response contains a <think> trace; disable Qwen3 thinking mode upstream"
	fi
	rm -f "$response"
	log "teacher smoke test passed"
}

# Starts vLLM (unless an upstream URL is given) and the teacher API, then waits for both.
# The teacher API logs to api_log, or to the terminal when it is "-".
start_teacher_stack() {
	local host="$1" port="$2" upstream_url="$3" api_log="$4"
	export TEACHER_UPSTREAM_MODEL="${TEACHER_UPSTREAM_MODEL:-$DEFAULT_TEACHER_MODEL}"
	if [[ -z "$upstream_url" ]]; then
		prepare_vllm
		local vllm_port="${VLLM_PORT:-8001}" vllm_log="$LOG_DIR/vllm.log" extra=()
		read -r -a extra <<< "${VLLM_EXTRA_ARGS:-}"
		start_background vllm "$vllm_log" "$VLLM" serve "$TEACHER_UPSTREAM_MODEL" \
			--revision "${TEACHER_MODEL_REVISION:-$DEFAULT_TEACHER_REVISION}" \
			--served-model-name "$TEACHER_UPSTREAM_MODEL" \
			--host 127.0.0.1 \
			--port "$vllm_port" \
			--max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
			--gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
			--default-chat-template-kwargs '{"enable_thinking": false}' \
			"${extra[@]+"${extra[@]}"}"
		log "vLLM log: $vllm_log"
		wait_for_http "http://127.0.0.1:$vllm_port/health" "" "${VLLM_STARTUP_TIMEOUT:-1800}" \
			"${BG_PIDS[${#BG_PIDS[@]} - 1]}" "$vllm_log"
		upstream_url="http://127.0.0.1:$vllm_port/v1"
		export TEACHER_PROVIDER_NAME="${TEACHER_PROVIDER_NAME:-self-hosted-vllm}"
	fi
	export TEACHER_UPSTREAM_BASE_URL="$upstream_url"
	export TEACHER_HOST="$host"
	export TEACHER_PORT="$port"
	start_background teacher-api "$api_log" "$APP_BIN/teacher-api"
	wait_for_http "http://127.0.0.1:$port/healthz" '"ready"' 120 "${BG_PIDS[${#BG_PIDS[@]} - 1]}" \
		"$api_log"
	smoke_test_teacher "http://127.0.0.1:$port" "${TEACHER_API_TOKEN:-}"
}

# Blocks until a background service exits, then fails.
supervise_background() {
	local index
	while true; do
		for index in "${!BG_PIDS[@]}"; do
			if ! kill -0 "${BG_PIDS[index]}" 2> /dev/null; then
				die "${BG_NAMES[index]} exited unexpectedly"
			fi
		done
		sleep 5
	done
}

# Pipeline options shared by student-train.sh and distill-local.sh.
DEFAULT_PROMPTS="data/examples/prompts.jsonl"
DEFAULT_CONFIG="configs/train.example.toml"
PROMPTS_PATH="$DEFAULT_PROMPTS"
CORPUS_PROFILE=""
LEDGER_PATH="data/teacher/ledger.jsonl"
REVIEWS_PATH=""
REFERENCES_PATH=""
ACCEPT_PENDING=0
TRAIN_CONFIG="$DEFAULT_CONFIG"
TERMS_REVIEW_ID=""
CONCURRENCY=4
MAX_OUTPUT_TOKENS=1024
SEED=""
PIPELINE_SHIFT=0

PIPELINE_USAGE="  --terms-review-id ID      Approval reference written into every ledger record
                            (default: \$TEACHER_TERMS_REVIEW_ID)
  --corpus PROFILE          Build the prompt corpus (quick, standard, or full) into
                            data/prompts unless it already exists, and use its prompts,
                            answer key, and configs/train.distill.toml
  --references PATH         Answer key written by distill corpus build; reviews the ledger
                            automatically before preparation
  --reviews PATH            Review decisions JSONL; with --references, they override
                            the automatic decisions
  --accept-pending          Train on unreviewed records (plumbing tests only)
  --prompts PATH            Prompt JSONL (default: $PROMPTS_PATH)
  --ledger PATH             Teacher ledger JSONL (default: $LEDGER_PATH)
  --config PATH             Training TOML (default: $TRAIN_CONFIG)
  --concurrency N           Parallel teacher requests (default: $CONCURRENCY)
  --max-output-tokens N     Teacher output limit (default: $MAX_OUTPUT_TOKENS)
  --seed N                  Teacher sampling seed"

# Parses one pipeline option. Sets PIPELINE_SHIFT to the number of arguments consumed.
parse_pipeline_option() {
	PIPELINE_SHIFT=2
	case "$1" in
		--terms-review-id) TERMS_REVIEW_ID="${2:?--terms-review-id needs a value}" ;;
		--corpus) CORPUS_PROFILE="${2:?--corpus needs a value}" ;;
		--reviews) REVIEWS_PATH="${2:?--reviews needs a value}" ;;
		--references) REFERENCES_PATH="${2:?--references needs a value}" ;;
		--prompts) PROMPTS_PATH="${2:?--prompts needs a value}" ;;
		--ledger) LEDGER_PATH="${2:?--ledger needs a value}" ;;
		--config) TRAIN_CONFIG="${2:?--config needs a value}" ;;
		--concurrency) CONCURRENCY="${2:?--concurrency needs a value}" ;;
		--max-output-tokens) MAX_OUTPUT_TOKENS="${2:?--max-output-tokens needs a value}" ;;
		--seed) SEED="${2:?--seed needs a value}" ;;
		--accept-pending)
			ACCEPT_PENDING=1
			PIPELINE_SHIFT=1
			;;
		*) return 1 ;;
	esac
}

validate_pipeline_options() {
	TERMS_REVIEW_ID="${TERMS_REVIEW_ID:-${TEACHER_TERMS_REVIEW_ID:-}}"
	[[ -n "$TERMS_REVIEW_ID" ]] \
		|| die "set --terms-review-id or TEACHER_TERMS_REVIEW_ID (see docs/distillation-pipeline.md)"
	local build_corpus=0
	if [[ -n "$CORPUS_PROFILE" ]]; then
		[[ "$CORPUS_PROFILE" =~ ^(quick|standard|full)$ ]] || die "--corpus must be quick, standard, or full"
		[[ "$PROMPTS_PATH" == "$DEFAULT_PROMPTS" ]] && PROMPTS_PATH="data/prompts/all-splits.jsonl"
		[[ -n "$REFERENCES_PATH" ]] || REFERENCES_PATH="$(dirname -- "$PROMPTS_PATH")/references.jsonl"
		[[ "$TRAIN_CONFIG" == "$DEFAULT_CONFIG" ]] && TRAIN_CONFIG="configs/train.distill.toml"
		build_corpus=1
	fi
	if [[ -z "$REVIEWS_PATH" && -z "$REFERENCES_PATH" && "$ACCEPT_PENDING" == 0 ]]; then
		die "pass --references PATH or --reviews PATH, or --accept-pending for a plumbing test"
	fi
	[[ -z "$REVIEWS_PATH" || -f "$REVIEWS_PATH" ]] || die "reviews file not found: $REVIEWS_PATH"
	if ((build_corpus == 0)); then
		[[ -z "$REFERENCES_PATH" || -f "$REFERENCES_PATH" ]] \
			|| die "references file not found: $REFERENCES_PATH"
		[[ -f "$PROMPTS_PATH" ]] || die "prompts file not found: $PROMPTS_PATH"
	fi
	[[ -f "$TRAIN_CONFIG" ]] || die "training config not found: $TRAIN_CONFIG"
}

# Builds the prompt corpus for --corpus unless the same profile is already in place. The build
# is deterministic, so a prompt keeps its example_id and text across rebuilds and profiles.
build_corpus_if_needed() {
	[[ -n "$CORPUS_PROFILE" ]] || return 0
	local directory manifest current=""
	directory="$(dirname -- "$PROMPTS_PATH")"
	manifest="$directory/manifest.json"
	if [[ -f "$manifest" && -f "$PROMPTS_PATH" && -f "$REFERENCES_PATH" ]]; then
		current="$("$APP_BIN/python" -c \
			'import json, sys; print(json.load(open(sys.argv[1])).get("profile", ""))' "$manifest")"
	fi
	if [[ "$current" == "$CORPUS_PROFILE" ]]; then
		log "using the existing $CORPUS_PROFILE prompt corpus in $directory"
		return 0
	fi
	log "building the $CORPUS_PROFILE prompt corpus into $directory (downloads public datasets)"
	"$APP_BIN/distill" corpus build --profile "$CORPUS_PROFILE" --output-dir "$directory"
	[[ -f "$PROMPTS_PATH" && -f "$REFERENCES_PATH" ]] \
		|| die "corpus build did not write $PROMPTS_PATH and $REFERENCES_PATH"
}

collect_teacher_data() {
	local teacher_url="$1" token="$2"
	local args=(collect --input "$PROMPTS_PATH" --output "$LEDGER_PATH" --teacher-url "$teacher_url"
		--terms-review-id "$TERMS_REVIEW_ID" --concurrency "$CONCURRENCY"
		--max-output-tokens "$MAX_OUTPUT_TOKENS" --no-thinking)
	[[ -n "$SEED" ]] && args+=(--seed "$SEED")
	log "collecting teacher responses into $LEDGER_PATH"
	TEACHER_API_TOKEN="$token" "$APP_BIN/distill" "${args[@]}"
}

# Validates the ledger against the answer key. Sets REVIEWS_PATH to the automatic decisions.
review_teacher_data() {
	[[ -n "$REFERENCES_PATH" ]] || return 0
	local output="${LEDGER_PATH%.jsonl}.reviews.jsonl"
	local args=(review --ledger "$LEDGER_PATH" --references "$REFERENCES_PATH" --output "$output"
		--config "$TRAIN_CONFIG")
	[[ -n "$REVIEWS_PATH" ]] && args+=(--overrides "$REVIEWS_PATH")
	[[ -n "${REVIEW_CODE_IMAGE:-}" ]] && args+=(--code-image "$REVIEW_CODE_IMAGE")
	log "reviewing teacher responses into $output"
	"$APP_BIN/distill" "${args[@]}"
	REVIEWS_PATH="$output"
}

prepare_and_train() {
	local dataset validation_dataset
	review_teacher_data
	dataset="$("$APP_BIN/python" -c \
		'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["dataset_path"])' \
		"$TRAIN_CONFIG")"
	validation_dataset="$("$APP_BIN/python" -c \
		'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb")).get("validation_dataset_path", ""))' \
		"$TRAIN_CONFIG")"
	local args=(prepare --input "$LEDGER_PATH" --output "$dataset" --split train --prompts "$PROMPTS_PATH")
	[[ -n "$REVIEWS_PATH" ]] && args+=(--reviews "$REVIEWS_PATH")
	[[ "$ACCEPT_PENDING" == 1 ]] && args+=(--accept-pending)
	log "preparing training data into $dataset"
	"$APP_BIN/distill" "${args[@]}"
	if [[ -n "$validation_dataset" ]]; then
		args=(prepare --input "$LEDGER_PATH" --output "$validation_dataset" --split validation
			--prompts "$PROMPTS_PATH")
		[[ -n "$REVIEWS_PATH" ]] && args+=(--reviews "$REVIEWS_PATH")
		[[ "$ACCEPT_PENDING" == 1 ]] && args+=(--accept-pending)
		log "preparing validation data into $validation_dataset"
		"$APP_BIN/distill" "${args[@]}"
	fi
	log "training the student with $TRAIN_CONFIG"
	"$APP_BIN/distill" train --config "$TRAIN_CONFIG"
}
