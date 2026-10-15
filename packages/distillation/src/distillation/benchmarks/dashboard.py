"""Read-only local dashboard API and optional built frontend."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from distillation.benchmarks.reporting import build_report, experiments
from distillation.benchmarks.traffic_reporting import build_traffic_report, traffic_experiments


def create_app(
	results_dir: Path,
	frontend_dir: Path | None = None,
	*,
	traffic_results_dir: Path = Path("outputs/traffic-benchmarks"),
) -> FastAPI:
	app = FastAPI(title="Distillation benchmarks")

	@app.get("/api/traffic/experiments")
	def list_traffic() -> dict[str, Any]:
		return traffic_experiments(traffic_results_dir)

	@app.get("/api/traffic/reports/{key}")
	def get_traffic_report(key: str) -> dict[str, Any]:
		try:
			return build_traffic_report(traffic_results_dir, key)
		except (ValueError, OSError) as error:
			raise HTTPException(status_code=422, detail=str(error)) from error

	@lru_cache(maxsize=8)
	def cached_report(key: str, fingerprint: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
		return build_report(results_dir, key)

	@app.get("/api/experiments")
	def list_experiments() -> dict[str, Any]:
		return experiments(results_dir)

	@app.get("/api/reports/{key}")
	def get_report(key: str) -> dict[str, Any]:
		fingerprint = tuple(
			(p.name, p.stat().st_mtime_ns, p.stat().st_size)
			for p in sorted(results_dir.glob("*.json"))
		)
		try:
			return cached_report(key, fingerprint)
		except (ValueError, OSError) as error:
			raise HTTPException(status_code=422, detail=str(error)) from error

	if frontend_dir and (frontend_dir / "index.html").is_file():
		app.mount("/assets", StaticFiles(directory=frontend_dir / "assets"), name="assets")

		@app.get("/")
		def index() -> FileResponse:
			return FileResponse(frontend_dir / "index.html")

	return app
