"""Read one run's artifacts into the sections the dashboard shows.

Each adapter copies fields its producer wrote and adds nothing it did not measure. A missing
artifact makes its section unavailable with a reason; a malformed or edited one is reported
with its error. Failed gates, refused campaigns and null values are passed through as such.
"""

import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

from distillation.baseline_gates import CLEAN, HEADROOM, load_gate
from distillation.learning_curve import load_plan
from distillation.results_index import COLUMNS, campaign_row, utility_rows

from dashboard_api.discovery import Artifact, Run
from dashboard_api.files import Loaded, parse_env, read_json, read_text, scan_dir
from dashboard_api.index import cell

TEACHER_KEYS = (
	"TEACHER_MODEL",
	"TEACHER_MODEL_REVISION",
	"TEACHER_UPSTREAM_MODEL",
	"TEACHER_GGUF",
	"TEACHER_GGUF_CONTEXT",
	"TEACHER_GGUF_GPU_LAYERS",
)
STUDY_KEYS = ("SEED", "BUDGET", "PARTITION", "ATTACKS", "ARMS", "DOMAINS", "WORKLOAD_VERSION")
EXPORT_KEYS = ("run_id", "status", "exit", "commit", "exported_at", "files", "excluded")
UTILITY_POOLED = ("latency_p50", "latency_p90", "latency_p99", "throughput", "elapsed_seconds")


def section(items: list[Any], reason: str, **extra: Any) -> dict[str, Any]:
	"""A section that is available only when it holds at least one item."""
	return {
		"state": "available" if items else "unavailable",
		"reason": None if items else reason,
		"items": items,
		**extra,
	}


def _place(artifact: Artifact) -> dict[str, str]:
	return {"root": artifact.root, "path": artifact.path}


def _get(value: Any, *keys: str) -> Any:
	for key in keys:
		if not isinstance(value, dict):
			return None
		value = value.get(key)
	return value


def _typed(row: dict[str, Any]) -> dict[str, Any]:
	"""A row computed from a directory, typed exactly like an index row."""
	return {
		column: cell(column, "" if row.get(column) is None else str(row[column]))
		for column in COLUMNS
	}


# Provenance


def _attempts(directory: Path) -> list[Path]:
	_, dirs = scan_dir(directory / "provenance")
	return [Path(entry.path) for entry in dirs]


def provenance(run: Run) -> dict[str, Any]:
	"""What the run says about itself: index labels, exit, export, settings and README."""
	rows = run.rows
	studies = sorted({r["study"] for r in rows if r["kind"] == "campaign" and r["study"]})
	sources = []
	for place in run.directories:
		directory = place.absolute
		attempts = _attempts(directory)
		latest = attempts[-1] if attempts else None
		config = read_text(latest / "config.env", "config.env") if latest else None
		settings = parse_env(config.value) if config and config.state == "ok" else []
		values = {pair["key"]: pair["value"] for pair in settings}
		export = read_json(directory / "export.json", "export.json")
		readme = read_text(directory / "README.md", "README.md", max_bytes=64 * 1024)
		exit_text = read_text(directory / "exit", "exit", max_bytes=256)
		sums = read_text(directory / "SHA256SUMS", "SHA256SUMS", max_bytes=4 * 1024 * 1024)
		texts = {
			name: read_text(latest / f"{name}.txt", f"{name}.txt", max_bytes=16 * 1024)
			for name in ("code", "hardware", "python")
			if latest is not None
		}
		sources.append(
			{
				**_place(place),
				"exit": exit_text.value.strip() if exit_text.state == "ok" else None,
				"export": {k: _get(export.value, k) for k in EXPORT_KEYS}
				if export.state == "ok"
				else None,
				"export_file": export.public(),
				"readme": readme.value if readme.state == "ok" else None,
				"readme_truncated": readme.error,
				"checksums": len(sums.value.splitlines()) if sums.state == "ok" else None,
				"attempts": [path.name for path in attempts],
				"settings_attempt": latest.name if latest else None,
				"settings": settings,
				"study_settings": {k: values.get(k) for k in STUDY_KEYS},
				"teacher": {k: values.get(k) for k in TEACHER_KEYS},
				"texts": {
					name: text.value if text.state == "ok" else None for name, text in texts.items()
				},
			}
		)
	commits = sorted({r["commit"] for r in rows if r["commit"]})
	statuses: dict[str, int] = {}
	for row in rows:
		statuses[f"{row['kind']}:{row['status'] or 'not recorded'}"] = (
			statuses.get(f"{row['kind']}:{row['status'] or 'not recorded'}", 0) + 1
		)
	return {
		"run_id": run.run_id,
		"readings": sorted(run.readings),
		"commits": commits,
		"commit_recorded": bool(commits),
		"studies": studies,
		"statuses": statuses,
		"seeds": sorted({r["seed"] for r in rows if r["seed"] is not None}),
		"partitions": sorted({r["partition"] for r in rows if r["partition"]}),
		"sources": sources,
	}


# Campaign acquisition


def _outcome(directory: Path, attack: str, arm: str) -> str | None:
	text = read_text(directory / "campaigns" / attack / arm / "outcome", "outcome", max_bytes=2048)
	return text.value.strip() if text.state == "ok" else None


def campaigns(run: Run) -> dict[str, Any]:
	"""Campaign arms from the index, or from run directories the index does not hold yet."""
	studies = [d.absolute for d in run.directories]
	items = []
	indexed = [r for r in run.rows if r["kind"] == "campaign"]
	for row in indexed:
		outcome = next((o for d in studies if (o := _outcome(d, row["attack"], row["arm"]))), None)
		items.append({**row, "source": "index", "outcome": outcome})
	errors = []
	if not indexed:
		for directory in studies:
			_, attacks = scan_dir(directory / "campaigns")
			for attack in attacks:
				_, arms = scan_dir(Path(attack.path))
				for arm in arms:
					try:
						row = campaign_row(
							Path(arm.path),
							directory / "schedules" / attack.name,
							None,
							attack.name,
							arm.name,
						)
					except (OSError, ValueError, KeyError, TypeError) as error:
						errors.append({"attack": attack.name, "arm": arm.name, "error": str(error)})
						continue
					items.append(
						{
							**_typed(row),
							"source": "directory",
							"outcome": _outcome(directory, attack.name, arm.name),
						}
					)
	return section(
		items,
		"no campaign rows in the index and no campaign directories in this run",
		errors=errors,
	)


# Legitimate utility


def _latency(stratum: dict[str, Any]) -> dict[str, Any]:
	latency = stratum.get("latency_seconds") or {}
	return {key: latency.get(key) for key in ("n", "p50", "p95", "p99")}


def _utility_file(directory: Path, arm: str) -> tuple[Loaded, dict[str, Any] | None]:
	loaded = read_json(directory / "utility" / arm / "utility.json", f"utility/{arm}/utility.json")
	if loaded.state != "ok" or not isinstance(loaded.value, dict):
		return loaded, None
	value = loaded.value
	strata = {
		group: {name: _latency(stratum) for name, stratum in (members or {}).items()}
		for group, members in (value.get("strata") or {}).items()
		if isinstance(members, dict)
	}
	return loaded, {
		"pooled": {key: value.get(key) for key in UTILITY_POOLED},
		"measurement_version": value.get("measurement_version"),
		"load": value.get("load"),
		"ttft_reason": value.get("ttft_reason"),
		"baseline": value.get("baseline"),
		"strata_latency": strata,
	}


def utility(run: Run) -> dict[str, Any]:
	"""Benign-user utility per arm: pooled and per stratum, with the validity verdict."""
	studies = [d.absolute for d in run.directories]
	rows = [r for r in run.rows if r["kind"] == "utility"]
	source = "index"
	if not rows:
		source = "directory"
		for directory in studies:
			_, entries = scan_dir(directory / "utility")
			preflight = directory / "teacher-preflight.json"
			for entry in entries:
				computed = utility_rows(
					Path(entry.path),
					None,
					entry.name,
					preflight=preflight if preflight.is_file() else None,
				)
				rows += [_typed(row) for row in computed]
	arms: dict[str, dict[str, Any]] = {}
	for row in rows:
		arm = arms.setdefault(
			row["arm"],
			{
				"arm": row["arm"],
				"policy": row["policy"],
				"accumulator": row["accumulator"],
				"study": row["study"],
				"status": row["status"],
				"validity": row["validity"],
				"validity_reason": row["validity_reason"],
				"preflight_sha256": row["preflight_sha256"],
				"min_completion_rate": row["min_completion_rate"],
				"source": source,
				"pooled": None,
				"strata": [],
				"file": None,
				"detail": None,
			},
		)
		metrics = {
			key: row[key]
			for key in (
				"submitted",
				"served",
				"completion_rate",
				"served_only_quality",
				"unconditional_utility",
				"latency_p95",
			)
		}
		if row["stratum"] == "all":
			arm["pooled"] = metrics
		else:
			group, _, value = (row["stratum"] or "").partition("=")
			arm["strata"].append({"group": group, "value": value, **metrics})
	for arm in arms.values():
		for directory in studies:
			loaded, detail = _utility_file(directory, arm["arm"])
			if loaded.state != "missing":
				arm["file"] = loaded.public()
				arm["detail"] = detail
				break
	return section(list(arms.values()), "no utility rows in the index and no utility.json")


# Teacher capability and capacity


def _gate(artifact: Artifact) -> dict[str, Any]:
	loaded = read_json(artifact.absolute, artifact.path)
	report = loaded.value if isinstance(loaded.value, dict) else {}
	integrity = None
	if loaded.state == "ok":
		try:
			load_gate(artifact.absolute)
			integrity = "intact"
		except ValueError, KeyError, TypeError:
			integrity = "digest does not match: the report was edited or is incomplete"
	return {"place": _place(artifact), "file": loaded.public(), "integrity": integrity, **report}


def headroom(artifact: Artifact) -> dict[str, Any]:
	gate = _gate(artifact)
	return {
		key: gate.get(key)
		for key in (
			"place",
			"file",
			"integrity",
			"gate",
			"task",
			"base",
			"teacher",
			"suite_sha256",
			"items",
			"families",
			"parameters",
			"results",
			"status",
			"passed",
			"reasons",
			"action",
			"inputs_sha256",
		)
	}


def clean(artifact: Artifact) -> dict[str, Any]:
	gate = _gate(artifact)
	return {
		key: gate.get(key)
		for key in (
			"place",
			"file",
			"integrity",
			"gate",
			"task",
			"base",
			"student_revision",
			"suite_sha256",
			"items",
			"families",
			"declared_seeds",
			"parameters",
			"results",
			"gold_diagnostic",
			"status",
			"passed",
			"reasons",
			"action",
			"inputs_sha256",
		)
	}


CAPACITY_KEYS = (
	"schedule_sha256",
	"slo",
	"offered",
	"completed",
	"errors",
	"timeouts",
	"defended_responses",
	"offered_rps",
	"completed_rps",
	"elapsed_seconds",
	"latency_p50_seconds",
	"latency_p95_seconds",
	"latency_p99_seconds",
	"queued_mid",
	"queued_at_end",
	"max_queue",
	"concurrency",
	"timeout_seconds",
	"checks",
	"passed",
)


def capacity(artifact: Artifact) -> dict[str, Any]:
	loaded = read_json(artifact.absolute, artifact.path)
	value = loaded.value if isinstance(loaded.value, dict) else {}
	return {
		"place": _place(artifact),
		"file": loaded.public(),
		**{key: value.get(key) for key in CAPACITY_KEYS},
	}


def scores(artifact: Artifact) -> dict[str, Any]:
	"""A per-problem score file: who was scored on which suite, and the share correct."""
	loaded = read_json(artifact.absolute, artifact.path)
	value = loaded.value if isinstance(loaded.value, dict) else {}
	marks = value.get("scores")
	numbers = (
		[float(v) for v in marks.values() if isinstance(v, int | float)]
		if isinstance(marks, dict)
		else []
	)
	return {
		"place": _place(artifact),
		"file": loaded.public(),
		**{
			key: value.get(key)
			for key in (
				"role",
				"model",
				"revision",
				"task",
				"suite_sha256",
				"evaluation_max_new_tokens",
			)
		},
		"items": len(numbers) if isinstance(marks, dict) else None,
		"accuracy": statistics.fmean(numbers) if numbers else None,
	}


def _of(run: Run, kind: str) -> list[Artifact]:
	return [a for a in run.artifacts if a.kind == kind]


def teacher(run: Run) -> dict[str, Any]:
	gates = [g for g in (headroom(a) for a in _of(run, "gate")) if g["gate"] == HEADROOM]
	return {
		"capacity": section(
			[capacity(a) for a in _of(run, "capacity")],
			"no teacher-capacity-v1 preflight report under the configured roots for this run",
		),
		"headroom": section(
			gates, "no headroom gate report under the configured roots for this run"
		),
		"scores": section(
			[scores(a) for a in _of(run, "scores")],
			"no problem-scores-v1 file under the configured roots for this run",
		),
	}


# Students


def curve(artifact: Artifact) -> dict[str, Any]:
	directory = artifact.absolute
	plan = read_json(directory / "plan.json", f"{artifact.path}/plan.json")
	summary = read_json(directory / "summary.json", f"{artifact.path}/summary.json")
	value = plan.value if isinstance(plan.value, dict) else {}
	integrity = None
	if plan.state == "ok":
		try:
			load_plan(directory)
			integrity = "intact"
		except (ValueError, KeyError, OSError) as error:
			integrity = f"plan files do not match their recorded hashes: {error}"
	recipe = value.get("recipe") or {}
	result = summary.value if isinstance(summary.value, dict) else None
	return {
		"place": _place(artifact),
		"plan_file": plan.public(),
		"summary_file": summary.public(),
		"integrity": integrity,
		"version": value.get("version"),
		"task": "gsm8k" if value.get("version") == "math-learning-curve-v1" else "ag_news",
		"scope": value.get("scope"),
		"sizes": value.get("sizes"),
		"seeds": value.get("seeds"),
		"include_pending": value.get("include_pending"),
		"target_fraction": value.get("target_fraction"),
		"training_ledgers": len(value.get("training_ledgers") or []) or None,
		"cells": len(value.get("cells") or []),
		"recipe": {
			key: recipe.get(key)
			for key in ("base_model", "revision", "tokenizer_model", "tokenizer_revision")
		},
		"summary": None
		if result is None
		else {
			key: result.get(key)
			for key in (
				"results",
				"gold_results",
				"reference_gain_pp",
				"target_gain_pp",
				"smallest_n",
				"bootstrap",
				"warning",
			)
		},
	}


def checkpoints(artifact: Artifact) -> dict[str, Any]:
	directory = artifact.absolute
	plan = read_json(directory / "plan.json", f"{artifact.path}/plan.json")
	summary = read_json(directory / "summary.json", f"{artifact.path}/summary.json")
	value = plan.value if isinstance(plan.value, dict) else {}
	result = summary.value if isinstance(summary.value, dict) else None
	recipe = value.get("recipe") or {}
	return {
		"place": _place(artifact),
		"plan_file": plan.public(),
		"summary_file": summary.public(),
		"task": value.get("task"),
		"seeds": value.get("seeds"),
		"target_accuracy": value.get("target_accuracy"),
		"recipe": {key: recipe.get(key) for key in ("base_model", "revision")},
		"summary": None
		if result is None
		else {
			key: result.get(key)
			for key in (
				"version",
				"scope",
				"target_accuracy",
				"base_accuracy",
				"base_meets_target",
				"seeds",
				"checkpoints",
				"first_reaching_target",
				"target_status",
				"censored_checkpoints",
				"unordered_checkpoints",
				"interventions",
				"limitations",
				"claim_boundary",
			)
		},
	}


def transfer(artifact: Artifact) -> dict[str, Any]:
	loaded = read_json(artifact.absolute, artifact.path)
	value = loaded.value if isinstance(loaded.value, dict) else {}
	if "seeds" in value and "seed_variability" in value:
		seeds = value["seeds"] if isinstance(value["seeds"], dict) else {}
		reports = {label: _transfer_report(report) for label, report in seeds.items()}
		return {
			"place": _place(artifact),
			"file": loaded.public(),
			"evidence": value.get("evidence"),
			"seeds": reports,
			"seed_variability": value.get("seed_variability"),
			"seeds_meeting_target": value.get("seeds_meeting_target"),
		}
	return {
		"place": _place(artifact),
		"file": loaded.public(),
		"evidence": value.get("evidence"),
		"seeds": {"single": _transfer_report(value)} if value else {},
		"seed_variability": None,
		"seeds_meeting_target": None,
	}


def _transfer_report(report: Any) -> dict[str, Any]:
	if not isinstance(report, dict):
		return {}
	return {
		key: report.get(key)
		for key in (
			"evidence",
			"weighting",
			"families",
			"items",
			"raw",
			"headroom",
			"clean_gain",
			"normalized_transfer",
			"suppression",
			"suppression_gate",
			"decision",
			"bootstrap",
		)
	}


def students(run: Run) -> dict[str, Any]:
	rows = [r for r in run.rows if r["kind"] == "student"]
	return {
		"curves": section(
			[curve(a) for a in _of(run, "learning-curve")],
			"no learning-curve plan under the configured roots for this run",
		),
		"checkpoints": section(
			[checkpoints(a) for a in _of(run, "checkpoints")],
			"no checkpoint-learning-v1 plan under the configured roots for this run",
		),
		"transfers": section(
			[transfer(a) for a in _of(run, "transfer")],
			"no saved transfer report (role teacher/base/clean/defended) for this run",
		),
		"index": section(rows, "the results index holds no student rows for this run"),
	}


def gates(run: Run) -> dict[str, Any]:
	reports = [clean(a) if _kind(a) == CLEAN else headroom(a) for a in _of(run, "gate")]
	refused = [
		{key: r[key] for key in ("attack", "arm", "policy", "accumulator", "status")}
		for r in run.rows
		if r["kind"] == "campaign" and r["status"] == "gate-refused"
	]
	return {
		"reports": section(reports, "no clean-baseline-gate-v1 report for this run"),
		"index": section(
			[r for r in run.rows if r["kind"] == "gate"],
			"the results index holds no gate rows for this run",
		),
		"detectability": section(
			refused, "no campaign arm of this run was refused by the detectability gate"
		),
	}


def _kind(artifact: Artifact) -> str | None:
	loaded = read_json(artifact.absolute, artifact.path)
	kind = _get(loaded.value, "gate")
	return kind if isinstance(kind, str) else None


SECTIONS: dict[str, Callable[[Run], dict[str, Any]]] = {
	"provenance": provenance,
	"campaigns": campaigns,
	"utility": utility,
	"teacher": teacher,
	"students": students,
	"gates": gates,
}


def detail(run: Run) -> dict[str, Any]:
	"""Every section; one that fails to read reports its error instead of failing the run."""
	result: dict[str, Any] = {}
	for name, build in SECTIONS.items():
		try:
			result[name] = build(run)
		except (OSError, ValueError, KeyError, TypeError) as error:
			result[name] = {"state": "error", "reason": f"{type(error).__name__}: {error}"}
	return result
