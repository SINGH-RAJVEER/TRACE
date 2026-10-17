"""Paired task outcomes, observed budget curves, and inspectable artifact provenance."""

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from distillation.benchmarks.models import SampleResult, save_json
from distillation.benchmarks.reporting import paired


def defense_effect(
	base: float, undefended: float, defended: float, *, development_gate: bool
) -> dict[str, Any]:
	gain = undefended - base
	valid = development_gate and gain > 0
	return {
		"base": base,
		"undefended": undefended,
		"defended": defended,
		"G_U": gain,
		"G_D": defended - base,
		"Delta_D": undefended - defended,
		"R_D": (defended - base) / gain if valid else None,
		"R_D_valid": valid,
		"R_D_invalid_reason": None
		if valid
		else "nonpositive undefended gain or failed development transfer gate",
	}


def study_report(root: Path) -> dict[str, Any]:
	protocol = json.loads((root / "protocol.json").read_text())
	plan_path = root / "budgets.json"
	if plan_path.exists():
		budgets = json.loads(plan_path.read_text())
	else:
		# Legacy studies froze the requested mode and task manifests separately.
		tasks_path = root / "tasks.json"
		manifests = json.loads(tasks_path.read_text()) if tasks_path.exists() else {}
		requested = protocol["budgets"]
		if requested == "full" and not manifests:
			raise ValueError("full budgets require frozen budgets.json or tasks.json")
		budgets = {}
		for task in protocol["tasks"]:
			pool = manifests.get(task, {}).get("roles", {}).get("attacker")
			budgets[task] = (
				[pool]
				if requested == "full"
				else [budget for budget in requested if pool is None or budget <= pool]
			)
	horizons_path = root / "policy-horizons.json"
	horizons = json.loads(horizons_path.read_text()) if horizons_path.exists() else {}
	rows, missing = [], []
	for task in protocol["tasks"]:
		selection_path = root / task / "recipe.json"
		gate = (
			selection_path.exists()
			and json.loads(selection_path.read_text())["transfer_gate_passed"]
		)
		for seed in protocol["seeds"]:
			for budget in budgets[task]:
				directory = root / task / f"s{seed}-b{budget}"
				baseline = directory / "report.attacker.json"
				for arm in protocol["arms"]:
					defended = directory / f"report.protected-{arm}.json"
					identity = {
						"task": task,
						"query_seed": seed,
						"training_seed": seed,
						"budget": budget,
						"defense": arm,
						"schedule": "uniform_without_replacement",
					}
					if not baseline.exists() or not defended.exists():
						missing.append(identity)
						continue
					left, right = json.loads(baseline.read_text()), json.loads(defended.read_text())
					if left["suite"]["sha256"] != right["suite"]["sha256"]:
						raise ValueError("study comparison mixes frozen suites")
					try:
						b = left["scores"]["student_before"][task]["primary_score"]
						u = left["scores"]["student_after"][task]["primary_score"]
						d = right["scores"]["student_after"][task]["primary_score"]
					except KeyError:
						missing.append(identity)
						continue
					for report in (left, right):
						dim = report["runs"]["student_after"]["dimensions"]
						if (
							dim["task"],
							dim["query_seed"],
							dim["training_seed"],
							dim["budget"],
						) != (task, seed, seed, budget):
							raise ValueError("study comparison mixes experiment dimensions")
					models = [r["runs"]["student_after"]["model"] for r in (left, right)]
					if [{k: v for k, v in m.items() if k != "adapter_path"} for m in models][0] != {
						k: v for k, v in models[1].items() if k != "adapter_path"
					}:
						raise ValueError(
							"study comparison mixes student models or inference settings"
						)
					comparison = paired(
						[
							SampleResult.model_validate(s["results"]["student_after"])
							for s in left["samples"]
						],
						[
							SampleResult.model_validate(s["results"]["student_after"])
							for s in right["samples"]
						],
					)
					rows.append(
						{
							**identity,
							**defense_effect(b, u, d, development_gate=gate),
							"defended_minus_undefended": comparison,
							"baseline_report": str(baseline),
							"defended_report": str(defended),
						}
					)
	groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		groups[(row["task"], row["budget"], row["defense"])].append(row)
	aggregates = []
	for (task, budget, defense), members in groups.items():
		values = {}
		for field in ("base", "undefended", "defended", "G_U", "G_D", "Delta_D", "R_D"):
			numbers = [r[field] for r in members if r[field] is not None]
			values[field] = {
				"n": len(numbers),
				"mean": statistics.fmean(numbers) if numbers else None,
				"sd": statistics.stdev(numbers) if len(numbers) > 1 else None,
			}
		aggregates.append(
			{
				"task": task,
				"budget": budget,
				"defense": defense,
				"seeds": [r["query_seed"] for r in members],
				"statistics": values,
			}
		)
	cost = []
	for task in protocol["tasks"]:
		target = protocol.get("capability_targets", {}).get(task, 80.0)
		for seed in protocol["seeds"]:
			for defense in protocol["arms"]:
				curve = sorted(
					[
						r
						for r in rows
						if (r["task"], r["query_seed"], r["defense"]) == (task, seed, defense)
					],
					key=lambda r: r["budget"],
				)
				reached = next((r for r in curve if r["defended"] >= target), None)
				cost.append(
					{
						"task": task,
						"seed": seed,
						"defense": defense,
						"target_score": target,
						"first_observed_budget": reached["budget"] if reached else None,
						"status": "reached_at_observed_budget"
						if reached
						else "not_reached_within_observed_budgets",
						"observed_budgets": [r["budget"] for r in curve],
					}
				)
	return {
		"study_kind": "separate_task",
		"task_budgets": budgets,
		"policy_horizons": horizons,
		"per_seed": rows,
		"mean_and_sd": aggregates,
		"budget_curves": rows,
		"cost_to_target": cost,
		"missing_pairs": missing,
		"units": "scores and gains in percentage points; retention is a ratio",
		"sampling_uncertainty": "paired source-document bootstrap, separate from seed SD",
	}


def study_manifest(root: Path) -> dict[str, Any]:
	"""Hash scientific artifacts only; credentials and account-key maps are excluded."""
	excluded = {
		"credentials.json",
		"accounts.json",
		"deception.key",
		"study-manifest.json",
		"status.json",
	}
	entries = {}
	for path in sorted(root.rglob("*")):
		if (
			not path.is_file()
			or path.name in excluded
			or path.suffix not in {".json", ".jsonl", ".toml", ".npy", ".safetensors"}
		):
			continue
		if "checkpoint-" in str(path):
			continue
		digest = hashlib.sha256()
		with path.open("rb") as source:
			for chunk in iter(lambda: source.read(1024 * 1024), b""):
				digest.update(chunk)
		entries[str(path.relative_to(root))] = {
			"sha256": digest.hexdigest(),
			"bytes": path.stat().st_size,
		}
	return {
		"version": 1,
		"artifacts": entries,
		"credential_files_excluded": True,
		"complete": (root / "study-report.json").exists()
		and not json.loads((root / "study-report.json").read_text())["missing_pairs"],
	}


def write_study_results(root: Path) -> None:
	from distillation.study import freeze

	report = study_report(root)
	save_json(root / "study-report.json", report)
	freeze(root / "study-manifest.json", study_manifest(root))
