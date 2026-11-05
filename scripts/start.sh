#!/usr/bin/env bash
# Product entry point. The upstream teacher is managed separately with `just teacher`.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
exec uv run --frozen python -m harness_api.launcher serve "$@"
