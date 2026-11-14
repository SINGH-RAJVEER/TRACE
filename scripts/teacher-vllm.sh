#!/usr/bin/env bash
# The sole supported teacher inference runtime. Keep the app and vLLM environments separate.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

VLLM_BIN=${VLLM_BIN:-.venv-vllm/bin/vllm}
MODEL=${TEACHER_UPSTREAM_MODEL:-Qwen/Qwen3.5-9B}
REVISION=${TEACHER_MODEL_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}
[ -x "$VLLM_BIN" ] || {
	echo "no vLLM executable at $VLLM_BIN; install vLLM in .venv-vllm or set VLLM_BIN" >&2
	exit 1
}

# Retained study configuration uses whitespace-separated flags. For values containing spaces
# (for example JSON chat-template kwargs), pass explicit arguments to just teacher-vllm instead.
read -r -a extra_args <<< "${VLLM_EXTRA_ARGS:-}"
watermark_args=()
case "${TEACHER_WATERMARK_SCHEME:-unigram-v1}" in
	unigram-v1) ;;
	kgw-v1)
		: "${TEACHER_WATERMARK_VOCAB_SIZE:?KGW requires the engine model vocabulary size}"
		# Workers read the same key file as the teacher; never send secrets in command arguments.
		uv run --frozen python -c 'import os; from pathlib import Path; from trace_watermark.keys import load_or_create_master_secret; load_or_create_master_secret(Path(os.getenv("TEACHER_WATERMARK_KEY_FILE", "artifacts/teacher.watermark.key")))'
		# Host-side context inspection requires synchronous output-token history.
		watermark_args=(--logits-processors trace_watermark.vllm:KgwLogitsProcessor
			--middleware trace_watermark.middleware.KgwMiddleware --no-async-scheduling)
		;;
	*) echo "unsupported TEACHER_WATERMARK_SCHEME" >&2; exit 1 ;;
esac
exec "$VLLM_BIN" serve "$MODEL" --revision "$REVISION" \
	--served-model-name "$MODEL" --host 127.0.0.1 --port "${VLLM_PORT:-8001}" \
	--max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
	--gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
	--reasoning-parser qwen3 --language-model-only \
	"${extra_args[@]}" "$@" "${watermark_args[@]}"
