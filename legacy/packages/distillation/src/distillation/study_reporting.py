"""Paired task outcomes, observed budget curves, and inspectable artifact provenance."""

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from distillation.benchmarks.models import SampleResult, save_json
from distillation.benchmarks.reporting import paired
from distillation.transfer import ANALYSIS_VERSION, effect_inference


def defense_effect(
	base: float,
	undefended: float,
	defended: float,
	*,
	development_gate: bool,
	inference: dict[str, Any] | None = None,
) -> dict[str, Any]:
	gain = undefended - base
	evidence = inference or {
		"retained_gain_fraction": None,
		"gain_removed_fraction": None,
		"ratio_valid": False,
		"ratio_invalid_reason": "missing_paired_inference",
	}
	if inference and any(
		not math.isclose(value, inference["sample_scores_pp"][key], abs_tol=1e-8)
		for key, value in (("base", base), ("undefended", undefended), ("defended", defended))
	):
		raise ValueError("aggregate capability scores disagree with paired samples")
	valid = development_gate and evidence["ratio_valid"]
	reason = (
		None
		if valid
		else ("qualification_failed" if not development_gate else evidence["ratio_invalid_reason"])
	)
	return {
		"base": base,
		"undefended": undefended,
		"defended": defended,
		"G_U": gain,
		"G_D": defended - base,
		"Delta_D": undefended - defended,
		**evidence,
		"retained_gain_fraction": evidence["retained_gain_fraction"] if valid else None,
		"gain_removed_fraction": evidence["gain_removed_fraction"] if valid else None,
		# Compatibility alias always keeps the historical retained-gain meaning.
		"R_D": evidence["retained_gain_fraction"] if valid else None,
		"R_D_semantics": "deprecated alias of retained_gain_fraction",
		"ratio_valid": valid,
		"retained_gain_fraction_ci95": evidence.get("retained_gain_fraction_ci95")
		if valid
		else None,
		"gain_removed_fraction_ci95": evidence.get("gain_removed_fraction_ci95") if valid else None,
		"ratio_invalid_reason": reason,
		"R_D_valid": valid,
		"R_D_invalid_reason": reason,
	}


CONTROLS = ("matched-volume", "matched-prefix")
UTILITY_FIELDS = (
	"submitted",
	"served",
	"served_only_quality",
	"unconditional_utility",
	"completion_rate",
	"latency_p50",
	"latency_p95",
	"strata",
)


def _load(path: Path) -> Any:
	return json.loads(path.read_text()) if path.exists() else None


def _student_after(report: dict[str, Any]) -> list[SampleResult]:
	return [SampleResult.model_validate(s["results"]["student_after"]) for s in report["samples"]]


def _controls(directory: Path, task: str, arm: str, defended: dict[str, Any]) -> dict[str, Any]:
	"""Undefended responses matched to the defended released tokens, scored on the same suite."""
	results = {}
	for name in CONTROLS:
		tokens = _load(directory / "protected" / arm / name / "control.json")
		report = _load(directory / f"report.{arm}-{name}.json")
		if report and report["suite"]["sha256"] != defended["suite"]["sha256"]:
			raise ValueError("control comparison mixes frozen suites")
		if report:
			status = "reported"
		elif tokens and not tokens["training_examples"]:
			status = "no_usable_examples"
		else:
			status = "missing"
		results[name] = {
			"tokens": tokens,
			"score": report["scores"]["student_after"][task]["primary_score"] if report else None,
			"defended_minus_control": paired(_student_after(report), _student_after(defended))
			if report
			else None,
			"status": status,
		}
	return results


def _utility(root: Path, task: str, seed: int, arm: str) -> dict[str, Any]:
	"""Held-out benign task utility through the bare teacher, then through the defense."""
	directory = root / task / f"s{seed}" / arm
	before, after = (_load(directory / f"utility-{phase}.json") for phase in ("before", "after"))

	def change(field: str) -> float | None:
		if not before or not after or before[field] is None or after[field] is None:
			return None
		return float(after[field]) - float(before[field])

	return {
		"task": task,
		"seed": seed,
		"defense": arm,
		"before": {k: before[k] for k in UTILITY_FIELDS} if before else None,
		"after": {k: after[k] for k in UTILITY_FIELDS} if after else None,
		"unconditional_utility_change": change("unconditional_utility"),
		"served_only_quality_change": change("served_only_quality"),
	}


def _benign_escalation(root: Path, task: str, arm: str) -> dict[str, Any] | None:
	"""Account-horizon false-escalation rates from replaying the held-out benign pool."""
	summary = _load(root / task / "benign-test" / f"{arm}.json")
	if summary is None:
		return None
	return {
		"task": task,
		"defense": arm,
		**{k: summary[k] for k in ("horizons", "unscored_horizons", "implemented_levels")},
		"false_escalations": summary["reached"],
		"false_escalation_rate": summary["reached_rate"],
		# Over scored horizons, the denominator the level targets were fitted on.
		"false_escalation_rate_scored": summary.get("reached_rate_scored"),
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
	fixed_training_seed = protocol.get("training_seed")
	rows, missing = [], []
	for task in protocol["tasks"]:
		selection_path = root / task / "recipe.json"
		selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
		gate = selection.get("analysis_version") == ANALYSIS_VERSION and selection.get(
			"qualification", {}
		).get("passed", False)
		for seed in protocol["seeds"]:
			for budget in budgets[task]:
				directory = root / task / f"s{seed}-b{budget}"
				baseline = directory / "report.attacker.json"
				for arm in protocol["arms"]:
					defended = directory / f"report.protected-{arm}.json"
					identity = {
						"task": task,
						"query_seed": seed,
						"training_seed": seed
						if fixed_training_seed is None
						else fixed_training_seed,
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
						) != (task, seed, identity["training_seed"], budget):
							raise ValueError("study comparison mixes experiment dimensions")
					models = [r["runs"]["student_after"]["model"] for r in (left, right)]
					if [{k: v for k, v in m.items() if k != "adapter_path"} for m in models][0] != {
						k: v for k, v in models[1].items() if k != "adapter_path"
					}:
						raise ValueError(
							"study comparison mixes student models or inference settings"
						)
					comparison = paired(_student_after(left), _student_after(right))
					inference = None
					if selection.get("analysis_version") == ANALYSIS_VERSION and all(
						"student_before" in s["results"] for s in left["samples"]
					):
						inference = effect_inference(
							[
								SampleResult.model_validate(s["results"]["student_before"])
								for s in left["samples"]
							],
							[
								SampleResult.model_validate(s["results"]["student_after"])
								for s in left["samples"]
							],
							[
								SampleResult.model_validate(s["results"]["student_after"])
								for s in right["samples"]
							],
							margin=selection["qualification"]["minimum_gain_pp"],
							qualified=gate,
						)
					rows.append(
						{
							**identity,
							**defense_effect(b, u, d, development_gate=gate, inference=inference),
							"defended_minus_undefended": comparison,
							"controls": _controls(directory, task, arm, right),
							# J and Q over this budget's prefix of the largest campaign.
							"escalation": _load(
								directory / "protected" / arm / "budget-campaign.json"
							),
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
		for field in (
			"base",
			"undefended",
			"defended",
			"G_U",
			"G_D",
			"Delta_D",
			"retained_gain_fraction",
			"gain_removed_fraction",
			"R_D",
		):
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
		target_path = root / task / "target.json"
		target = (
			json.loads(target_path.read_text())["target_score"]
			if target_path.exists()
			# Studies before task-study-v4 declared one target score in the protocol.
			else protocol.get("capability_targets", {}).get(task)
		)
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
				reached = (
					None
					if target is None
					else next((r for r in curve if r["defended"] >= target), None)
				)
				cost.append(
					{
						"task": task,
						"seed": seed,
						"defense": defense,
						"target_score": target,
						"first_observed_budget": reached["budget"] if reached else None,
						"status": "target_not_declared"
						if target is None
						else "reached_at_observed_budget"
						if reached
						else "not_reached_within_observed_budgets",
						"observed_budgets": [r["budget"] for r in curve],
					}
				)
	utility = [
		_utility(root, task, seed, arm)
		for task in protocol["tasks"]
		for seed in protocol["seeds"]
		for arm in protocol["arms"]
	]
	benign = [
		result
		for task in protocol["tasks"]
		for arm in protocol["arms"]
		if (result := _benign_escalation(root, task, arm))
	]
	return {
		"study_kind": "separate_task",
		"calibration_adequacy": {
			task: _load(root / task / "calibration-adequacy.json") for task in protocol["tasks"]
		},
		"task_budgets": budgets,
		"policy_horizons": horizons,
		"per_seed": rows,
		"mean_and_sd": aggregates,
		"budget_curves": rows,
		"cost_to_target": cost,
		"missing_pairs": missing,
		"benign_utility": utility,
		"benign_false_escalation": benign,
		"benign_false_escalation_scope": "held-out benign pool replayed offline with pooled "
		"accounts; checks the pipeline on unseen requests, not real-user rates",
		"units": "scores and gains in percentage points; retention is a ratio",
		"sampling_uncertainty": "paired source-document bootstrap conditional on trained models; "
		"separate from seed SD",
		"ratio_semantics": "R_D is a deprecated retained-gain alias; invalid ratios remain null",
		"seed_sd_sources": "query order and training randomness (paired seeds)"
		if fixed_training_seed is None
		else f"query order only (training seed fixed at {fixed_training_seed})",
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
