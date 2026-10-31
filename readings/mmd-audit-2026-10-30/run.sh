#!/usr/bin/env bash
# Keep generated embeddings and diagnostic outputs outside the committed reading.
set -euo pipefail
audit_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
audit_root=$(cd -- "$audit_source/../.." && pwd)
audit_output=${MMD_AUDIT_OUTPUT_DIR:-$audit_root/outputs/mmd-audit}
audit_python=${MMD_AUDIT_PYTHON:-$audit_root/outputs/mmd-audit/.venv/bin/python}
if [[ $# -eq 0 || ! -f "$audit_source/$1" || "$1" != *.py || "$1" == */* ]]; then
	printf '%s\n' 'Usage: run.sh SCRIPT.py [arguments]' >&2
	exit 2
fi
audit_script=$1
shift
mkdir -p -- "$audit_output/results" "$audit_output/cache"
export MMD_AUDIT_CACHE_DIR=${MMD_AUDIT_CACHE_DIR:-$audit_output/cache}
cd -- "$audit_output"
exec direnv exec "$audit_root" env HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" "$audit_python" "$audit_source/$audit_script" "$@"
