#!/usr/bin/env bash
# Serve the read-only results dashboard. Installs the frontend from its lockfile on first use,
# rebuilds it when its sources are newer than the build, then starts the API that also serves
# the page. It only reads artifacts: no model, teacher, evaluation or export is started.
set -euo pipefail

if [ "${1:-}" = --help ]; then
	cat <<'HELP'
Usage: just dashboard
Serves the read-only dashboard at http://127.0.0.1:8002 by default.
Settings: DASHBOARD_ROOTS (default readings,artifacts), DASHBOARD_INDEX (readings/index.tsv),
DASHBOARD_HOST (127.0.0.1), DASHBOARD_PORT (8002).
HELP
	exit 0
fi
if [ "$#" -gt 0 ]; then
	echo "Usage: just dashboard (configure with DASHBOARD_* variables; see --help)" >&2
	exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
APP=apps/dashboard

if [ ! -d "$APP/node_modules" ]; then
	(cd "$APP" && bun install --frozen-lockfile)
fi
if [ ! -f "$APP/dist/index.html" ] \
	|| [ -n "$(find "$APP/src" "$APP/index.html" "$APP/package.json" "$APP/bun.lock" -newer "$APP/dist/index.html" -print -quit)" ]; then
	(cd "$APP" && bun run build)
fi
exec uv run --frozen dashboard-api
