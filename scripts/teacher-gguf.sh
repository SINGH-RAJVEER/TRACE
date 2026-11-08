#!/usr/bin/env bash
# Serve a quantized GGUF teacher through llama-server on the port the study checks for vLLM.
# Settings come from the environment; extra arguments are passed to llama-server.
set -euo pipefail

MODEL=${TEACHER_GGUF:-$HOME/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf}
[ -f "$MODEL" ] || { echo "no GGUF teacher at $MODEL; set TEACHER_GGUF" >&2; exit 1; }

# llama-server divides the context between slots, so each slot gets CONTEXT / SLOTS tokens.
# Reasoning is on by default and returned separately so the teacher API can withhold it. Keep
# the reasoning budget equal to TEACHER_REASONING_BUDGET_TOKENS: the adapter adds that allowance
# to each answer cap. Each slot needs prompt + budget + answer cap; raise the context to match.
exec llama-server --model "$MODEL" \
	--reasoning "${TEACHER_GGUF_REASONING:-on}" --reasoning-format deepseek \
	--reasoning-budget "${TEACHER_REASONING_BUDGET_TOKENS:-2048}" \
	--alias "${TEACHER_UPSTREAM_MODEL:-Qwen/Qwen3.5-9B}" \
	--host 127.0.0.1 --port "${VLLM_PORT:-8001}" \
	--ctx-size "${TEACHER_GGUF_CONTEXT:-8192}" --parallel "${TEACHER_GGUF_SLOTS:-2}" \
	--device "${TEACHER_GGUF_DEVICE:-Vulkan0}" --n-gpu-layers "${TEACHER_GGUF_GPU_LAYERS:-99}" --jinja "$@"
