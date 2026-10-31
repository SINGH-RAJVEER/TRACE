#!/usr/bin/env bash
set -euo pipefail
cd /home/rajveer/Projects/personal/capstone
exec devenv shell -- env \
	TEACHER_UPSTREAM_MODEL=Qwen3.5-9B-Q4_K_M \
	TEACHER_PROVIDER_NAME=colab-llamacpp-q4_k_m \
	TEACHER_TIMEOUT_SECONDS=300 \
	python3 scripts/pipeline.py teacher-host --host 127.0.0.1 --port 18010 \
	--upstream-url http://127.0.0.1:18011/v1
