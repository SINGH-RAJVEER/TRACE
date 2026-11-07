"""Read-only HTTP API and the built dashboard frontend.

Only GET routes exist. Runs are rescanned on each request, so new artifacts appear without a
restart; nothing is written, trained, evaluated or sent to a teacher.
"""

import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dashboard_api.adapters import detail
from dashboard_api.discovery import Run, build_runs, reading_directories, walk
from dashboard_api.files import Roots
from dashboard_api.index import load_index
from dashboard_api.settings import Settings

RUN_ID = re.compile(r"[0-9a-f]{16}")


def summary(run: Run) -> dict[str, Any]:
	counts: dict[str, int] = {}
	for row in run.rows:
		counts[row["kind"]] = counts.get(row["kind"], 0) + 1
	artifacts: dict[str, int] = {}
	for artifact in run.artifacts:
		artifacts[artifact.kind] = artifacts.get(artifact.kind, 0) + 1
	statuses = sorted({row["status"] for row in run.rows if row["status"]})
	return {
		"id": run.id,
		"label": run.label,
		"kind": run.kind,
		"run_id": run.run_id,
		"readings": sorted(run.readings),
		"studies": sorted({r["study"] for r in run.rows if r["kind"] == "campaign" and r["study"]}),
		"commits": sorted({r["commit"] for r in run.rows if r["commit"]}),
		"statuses": statuses,
		"rows": counts,
		"artifacts": artifacts,
		"places": [{"root": d.root, "path": d.path} for d in run.directories],
	}


def create_app(
	settings: Settings | None = None,
	*,
	roots: Roots | None = None,
	index: Path | None = None,
	frontend: Path | None = None,
) -> FastAPI:
	settings = settings or Settings()
	base = Path.cwd()
	roots = roots or Roots.parse(settings.roots, base)
	index = index or (base / settings.index)
	frontend = frontend or (base / settings.frontend)
	app = FastAPI(title="TRACE results dashboard", docs_url=None, redoc_url=None)

	def scan() -> tuple[dict[str, Any], list[Run]]:
		loaded = load_index(index)
		runs = build_runs(walk(roots), loaded["rows"], reading_directories(roots))
		return loaded, runs

	@app.get("/api/health")
	def health() -> dict[str, Any]:
		return {"status": "ok", "read_only": True}

	@app.get("/api/runs")
	def runs() -> dict[str, Any]:
		loaded, found = scan()
		return {
			"roots": [{"name": r.name, "exists": r.path.is_dir()} for r in roots.roots],
			"index": {
				"state": loaded["state"],
				"path": loaded["path"],
				"error": loaded["error"],
				"rows": len(loaded["rows"]),
			},
			"runs": [summary(run) for run in found],
		}

	@app.get("/api/runs/{run_id}")
	def run(run_id: str) -> dict[str, Any]:
		if not RUN_ID.fullmatch(run_id):
			raise HTTPException(status_code=404, detail="unknown run")
		_, found = scan()
		match = next((r for r in found if r.id == run_id), None)
		if match is None:
			raise HTTPException(status_code=404, detail="unknown run")
		return {"run": summary(match), **detail(match)}

	if (frontend / "index.html").is_file():
		if (frontend / "assets").is_dir():
			app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

		@app.get("/", include_in_schema=False)
		def page() -> FileResponse:
			return FileResponse(frontend / "index.html")

	return app
