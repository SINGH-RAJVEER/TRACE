"""Development utility grids with isolated state and hash-verified completed runs."""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from harness.calibration import BenignRequest

from distillation.quota_matching import finite_number, match_quotas
from distillation.teacher_capacity import bind_utility_baseline
from distillation.utility import check_arrivals

Runner = Callable[[dict[str, Any], Path, Path, dict[str, Any]], None]


def digest(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
	with path.open("x") as stream:
		stream.write(json.dumps(value, sort_keys=True, indent="\t") + "\n")


def plan_grid(spec: dict[str, Any], output: Path) -> dict[str, Any]:
	"""Freeze a development declaration and capacity evidence in a new directory."""
	if spec.get("split") != "development" or not spec.get("population_id"):
		raise ValueError("a development population_id is required")
	finite_number(spec["tolerance"], "tolerance", maximum=1)
	if not spec.get("strata"):
		raise ValueError("declare all strata as name: {field, value}")
	for selector in spec["strata"].values():
		if selector["field"] not in ("domain", "domain_profile", "client_type"):
			raise ValueError("unsupported stratum field")
	settings = spec["settings"]
	ids = [s["id"] for s in settings]
	if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", i) for i in ids):
		raise ValueError("setting IDs must be unique safe directory names")
	policies = [s["policy"] for s in settings]
	if policies.count("allow") != 1 or policies.count("candidate") != 1:
		raise ValueError("exactly one allow and candidate setting required")
	if not {"quota", "quota_tokens"} <= set(policies) or set(policies) - {
		"allow",
		"candidate",
		"quota",
		"quota_tokens",
		"rate",
	}:
		raise ValueError("grid needs quota and quota_tokens; rate is optional")
	for setting in settings:
		for field in ("request_quota", "output_token_budget"):
			if (
				field in setting
				or (field == "request_quota" and setting["policy"] in ("quota", "quota_tokens"))
				or (field == "output_token_budget" and setting["policy"] == "quota_tokens")
			):
				value = setting.get(field)
				if isinstance(value, bool) or not isinstance(value, int) or value < 1:
					raise ValueError(f"{field} must be a positive integer")
	schedule = Path(spec["schedule"])
	capacity = Path(spec["capacity"])
	binding = bind_utility_baseline(
		capacity,
		schedule_sha256=digest(schedule),
		declared_offered_rps=spec["offered_rps"],
		min_completion_rate=spec["min_completion_rate"],
		concurrency=spec["concurrency"],
		request_deadline=spec["request_deadline"],
	)
	requests = [
		BenignRequest.model_validate_json(line)
		for line in schedule.read_text().splitlines()
		if line.strip()
	]
	check_arrivals(requests, binding)
	counts: Counter[str] = Counter()
	expected = []
	for row in requests:
		if row.example_id is None:
			raise ValueError("every scheduled request requires an example_id")
		expected.append([row.account, counts[row.account], row.example_id, row.client_type])
		counts[row.account] += 1
	if not expected:
		raise ValueError("schedule cannot be empty")
	output.mkdir(parents=True, exist_ok=False)
	(output / "schedule.jsonl").write_bytes(schedule.read_bytes())
	(output / "capacity.json").write_bytes(capacity.read_bytes())
	plan = spec | {
		"schedule": "schedule.jsonl",
		"capacity": "capacity.json",
		"binding": binding,
		"version": "development-grid-v1",
		"expected_requests": sorted(expected),
	}
	write(output / "plan.json", plan)
	write(output / "plan.sha256.json", digest(output / "plan.json"))
	return plan


def load_plan(root: Path) -> dict[str, Any]:
	if digest(root / "plan.json") != json.loads((root / "plan.sha256.json").read_text()):
		raise ValueError("plan hash mismatch")
	plan: dict[str, Any] = json.loads((root / "plan.json").read_text())
	for filename, key in (
		("schedule.jsonl", "schedule_sha256"),
		("capacity.json", "preflight_sha256"),
	):
		if digest(root / filename) != plan["binding"][key]:
			raise ValueError("frozen input hash mismatch")
	return plan


def artifacts(directory: Path) -> dict[str, str]:
	paths = sorted(directory.rglob("*"))
	if any(p.is_symlink() for p in paths):
		raise ValueError("run artifacts cannot be symlinks")
	return {
		str(p.relative_to(directory)): digest(p)
		for p in paths
		if p.is_file() and p != directory / "receipt.json"
	}


def verify(root: Path, setting: dict[str, Any]) -> dict[str, Any]:
	directory = root / setting["id"]
	if directory.is_symlink():
		raise ValueError("run directory cannot be a symlink")
	receipt: dict[str, Any] = json.loads((directory / "receipt.json").read_text())
	if receipt != {
		"plan_sha256": digest(root / "plan.json"),
		"setting": setting,
		"artifacts": artifacts(directory),
	}:
		raise ValueError("completed run artifact hash mismatch")
	return receipt


def execute_grid(root: Path, runner: Runner) -> None:
	"""Runner owns gateway lifecycle and measurement; it must use the supplied fresh DB.

	A partial run is never retried in place. Completed cells resume only after verification.
	"""
	plan = load_plan(root)
	for setting in plan["settings"]:
		directory = root / setting["id"]
		if (directory / "receipt.json").exists():
			verify(root, setting)
			continue
		if directory.exists():
			raise ValueError("refusing to reuse setting state; plan a new grid")
		directory.mkdir()
		runner(setting, directory / "state.sqlite", directory, plan | {"root": str(root.resolve())})
		if not (directory / "state.sqlite").is_file() or not (directory / "utility.json").is_file():
			raise ValueError("runner must produce state.sqlite and utility.json")
		load_plan(root)
		validate_utility(plan, json.loads((directory / "utility.json").read_text()), setting)
		write(
			directory / "receipt.json",
			{
				"plan_sha256": digest(root / "plan.json"),
				"setting": setting,
				"artifacts": artifacts(directory),
			},
		)


def validate_utility(plan: dict[str, Any], run: dict[str, Any], setting: dict[str, Any]) -> None:
	if (
		run.get("measurement_version") != "utility-response-latency-v3"
		or run.get("baseline") != plan["binding"]
	):
		raise ValueError("utility needs the frozen capacity binding")
	if (
		run["load"]["concurrency"] != plan["concurrency"]
		or run["load"]["deadline_seconds"] != plan["request_deadline"]
	):
		raise ValueError("utility load mismatch")
	rows = run["results"]
	if not rows or run["submitted"] != len(rows):
		raise ValueError("utility requires complete submitted rows")
	actual = sorted(
		[r["account"], r["sequence"], str(r["example_id"]), r.get("client_type")] for r in rows
	)
	if actual != plan["expected_requests"]:
		raise ValueError("utility rows do not match the frozen schedule")
	for row in rows:
		finite_number(row["quality"], "task success", maximum=1)
		if not row["received"] and row["quality"] != 0:
			raise ValueError("nonresponses must score zero")
	if (
		setting["policy"] == "allow"
		and sum(bool(r["received"]) for r in rows) / len(rows) < plan["min_completion_rate"]
	):
		raise ValueError("allow completion below capacity floor")


def derive_grid(root: Path) -> dict[str, Any]:
	"""Derive absolute utility losses from all submissions, never served-only quality."""
	plan = load_plan(root)
	runs = {}
	bindings = {}
	identities = None
	for setting in plan["settings"]:
		bindings[setting["id"]] = verify(root, setting)
		run = json.loads((root / setting["id"] / "utility.json").read_text())
		validate_utility(plan, run, setting)
		keys = sorted(
			(
				r["account"],
				r["sequence"],
				str(r["example_id"]),
				r["domain"],
				str(r.get("domain_profile")),
				str(r.get("client_type")),
			)
			for r in run["results"]
		)
		if len(set(keys)) != len(keys) or (identities is not None and keys != identities):
			raise ValueError("runs must measure the same complete population")
		identities = keys
		values = {}
		for name, selector in plan["strata"].items():
			rows = [r for r in run["results"] if r.get(selector["field"]) == selector["value"]]
			if not rows:
				raise ValueError(f"missing declared stratum: {name}")
			values[name] = sum(r["quality"] for r in rows) / len(rows)
		runs[setting["id"]] = values
	allow = next(runs[s["id"]] for s in plan["settings"] if s["policy"] == "allow")
	measured = []
	for setting in plan["settings"]:
		losses = {k: allow[k] - v for k, v in runs[setting["id"]].items()}
		if any(v < 0 for v in losses.values()):
			raise ValueError(
				"negative loss unsupported by quota matcher; report improvement separately"
			)
		measured.append(
			setting
			| {
				"arm": setting["id"],
				"loss_by_stratum": losses,
				"run_artifacts": bindings[setting["id"]],
			}
		)
	development = {k: plan[k] for k in ("split", "population_id", "tolerance")}
	development |= {
		"schedule_sha256": plan["binding"]["schedule_sha256"],
		"candidate": next(s for s in measured if s["policy"] == "candidate"),
		"settings": [s for s in measured if s["policy"] in ("quota", "quota_tokens")],
		"allow_artifacts": next(
			bindings[s["id"]] for s in plan["settings"] if s["policy"] == "allow"
		),
	}
	return {"development": development, "matches": match_quotas(development)}
