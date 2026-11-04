"""Join frozen training sets and student transfer intervals without inventing outcomes."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from distillation.baseline_gates import CLEAN, load_gate
from distillation.evaluation_protocol import compare_conditions
from distillation.output_sets import ADAPTIVE_ARMS, CONTROL_ARMS, verify_artifact
from distillation.quota_matching import BASELINES, finite_number, match_quotas
from distillation.results_index import read_index, utility_validity_check

VIEWS = ("equal_query", "equal_token")


def _read(path: str | Path) -> tuple[dict[str, Any], str]:
	data = Path(path).read_bytes()
	value = json.loads(data)
	if not isinstance(value, dict):
		raise ValueError("report must be a JSON object")
	return value, hashlib.sha256(data).hexdigest()


def read_gate(path: Path | None, manifest_digest: str) -> dict[str, Any]:
	"""An intact, passing clean gate over this manifest. An unrelated pass is insufficient."""
	if path is None:
		raise ValueError("missing clean-baseline gate report")
	gate = load_gate(path)
	if (
		gate.get("gate") != CLEAN
		or gate.get("passed") is not True
		or gate.get("manifest_digest") != manifest_digest
	):
		raise ValueError("clean-baseline gate must pass and bind this acquisition manifest")
	return {"path": str(path.resolve()), "sha256": _read(path)[1], "report": gate}


def comparison_report(
	declaration: dict[str, Any],
	*,
	gate_report: Path | None = None,
) -> dict[str, Any]:
	"""Require both budget views, matched quotas, complete students and >=3 paired seeds.

	Transfer is lower-is-better. At every seed the largest upper bound over the candidate and
	all declared adaptive attacks must be strictly below each matched baseline's lower bound.
	This conservative interval-separation rule does not pool seed uncertainty or turn a
	within-seed task-family bootstrap into a three-seed confidence interval.
	"""
	matching = match_quotas(declaration["development"])
	candidate = matching["development"]["candidate"]["arm"]
	adaptive = declaration.get("adaptive_arms", [])
	controls = list(CONTROL_ARMS)
	arms = [candidate, *BASELINES, *adaptive, *controls]
	if len(set(arms)) != len(arms):
		raise ValueError("declare distinct candidate, baseline, adaptive and control arm labels")
	if any(
		isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
		for seed in declaration["seeds"]
	):
		raise ValueError("training seeds must be nonnegative integers")
	reference = declaration["reference_queries"]
	if isinstance(reference, bool) or not isinstance(reference, int) or reference < 1:
		raise ValueError("reference_queries must be a positive integer")
	seeds = [str(seed) for seed in declaration["seeds"]]
	if not seeds or len(set(seeds)) != len(seeds):
		raise ValueError("declare distinct training seeds")
	manifest_digest = declaration["manifest_digest"]
	issues = []
	if declaration.get("evidence_kind") != "measurement":
		issues.append("fixture or undeclared evidence cannot support a measurement verdict")
	gate = None
	try:
		gate = read_gate(gate_report, manifest_digest)
	except (ValueError, OSError) as error:
		issues.append(str(error))
	if len(seeds) < 3:
		issues.append("at least three paired training seeds required; one seed is screening only")
	if not adaptive:
		issues.append("declare adaptive attackers")
	for baseline in BASELINES:
		if matching["baselines"][baseline]["status"] != "matched":
			issues.append(f"{baseline}: not matched")
	views: dict[str, Any] = {}
	for view in VIEWS:
		entries = declaration.get("views", {}).get(view)
		if not entries:
			issues.append(f"missing {view} view")
			views[view] = {"status": "no result", "arms": {}}
			continue
		results: dict[str, Any] = {}
		for arm in arms:
			entry = entries.get(arm)
			if entry is None:
				results[arm] = {
					"status": "no result",
					"reason": "missing training set and students",
				}
				issues.append(f"{view}/{arm}: missing data")
				continue
			try:
				training, training_sha = _read(entry["training_set"])
				verify_artifact(training)
				if training.get("version") != "student-training-set-v1":
					raise ValueError("not a student training set")
				rows = training["rows"]
				selection = training["selection"]
				if (
					selection["count"] != len(rows)
					or selection["example_ids"] != [r["example_id"] for r in rows]
					or len(set(selection["example_ids"])) != len(rows)
					or selection["output_tokens"] != sum(r["released_tokens"] for r in rows)
				):
					raise ValueError("training selection accounting mismatch")
				if not training["trainable_rows"]:
					raise ValueError("no released training examples")
				if arm == candidate and training["recipe"]["arm"] != declaration["development"][
					"candidate"
				].get("recipe", candidate):
					raise ValueError("candidate recipe mismatch")
				if training["manifest_digest"] != manifest_digest:
					raise ValueError("training set acquisition mismatch")
				if training["selection"]["by"] != (
					"queries" if view == "equal_query" else "tokens"
				):
					raise ValueError("training set budget view mismatch")
				if arm in BASELINES:
					selected = matching["baselines"][arm]["selected"]
					if selected is not None:
						setting = selected["setting"]
						expected = {
							k: setting[k]
							for k in ("request_quota", "output_token_budget")
							if k in setting
						}
						if (
							training["recipe"]["arm"] != arm
							or training.get("baseline_settings") != expected
						):
							raise ValueError(
								"student set does not use selected matched quota setting"
							)
				if arm in adaptive and training["recipe"]["arm"] not in ADAPTIVE_ARMS:
					raise ValueError("adaptive arm must use an adaptive training recipe")
				if arm in controls and training["recipe"]["arm"] != arm:
					raise ValueError("control recipe mismatch")
				intervals = {}
				student_reports = {}
				for seed in seeds:
					student, student_sha = _read(entry["students"][seed])
					if (
						student.get("training_set_digest") != training["digest"]
						or str(student.get("seed")) != seed
					):
						raise ValueError(f"seed {seed}: student/training-set join mismatch")
					if gate is None or student.get("gate_report_sha256") != gate["sha256"]:
						raise ValueError(
							f"seed {seed}: missing or mismatched clean gate dependency"
						)
					transfer = student["transfer"]
					quantity = transfer["normalized_transfer"]["defended"]
					interval = quantity["interval"]
					if (
						quantity["value"] is None
						or not isinstance(interval, list)
						or len(interval) != 2
					):
						raise ValueError(f"seed {seed}: missing student transfer interval")
					# Transfer ratios may be negative or exceed one; never clip them.
					if (
						any(
							isinstance(x, bool)
							or not isinstance(x, int | float)
							or not math.isfinite(x)
							for x in [quantity["value"], *interval]
						)
						or interval[0] > interval[1]
					):
						raise ValueError("invalid student interval")
					if not (
						transfer["headroom"]["usable"] is True
						and transfer["clean_gain"]["usable"] is True
					):
						raise ValueError("undefined transfer denominator")
					intervals[seed] = interval
					student_reports[seed] = {"sha256": student_sha, "transfer": transfer}
				cost = entry.get("short_output_cost")
				if cost is None:
					raise ValueError("record short_output_cost, including zero extra costs")
				for key in ("extra_queries", "money", "seconds"):
					finite_number(cost[key], f"short_output_cost.{key}")
				if not isinstance(cost["extra_queries"], int) or isinstance(
					cost["extra_queries"], bool
				):
					raise ValueError("extra_queries must be an integer")
				if cost["extra_queries"] != max(
					0, training["selection"]["count"] - declaration["reference_queries"]
				):
					raise ValueError(
						"extra_queries does not match the selected distinct query count"
					)
				results[arm] = {
					"status": "result",
					"training_set_sha256": training_sha,
					"training_set_digest": training["digest"],
					"arm_digest": training["arm_digest"],
					"parent_digest": training.get("parent_digest"),
					"selection": training["selection"],
					"validation_counts": {
						"task_valid": sum(r["task_valid"] for r in rows),
						"answer_equal": sum(r["answer_equal"] for r in rows),
						"fallbacks": sum(r["fallback"] is not None for r in rows),
					},
					"rewriter_cost": {
						key: sum(r["rewriter_cost"][key] for r in rows)
						for key in ("queries", "money", "seconds")
					},
					"intervals": intervals,
					"students": student_reports,
					"short_output_cost": cost,
				}
			except (KeyError, ValueError, TypeError, OSError) as error:
				results[arm] = {"status": "no result", "reason": str(error)}
				issues.append(f"{view}/{arm}: {error}")
		available = [
			r
			for arm, r in results.items()
			if arm not in {"clean_truncated", "clean_subsampled"} and r["status"] == "result"
		]
		budgets = {r["selection"]["budget"] for r in available}
		if len(budgets) != 1:
			issues.append(f"{view}: unequal or missing declared budgets")
		# Prefix selection can leave slack. Never call unequal delivered supervision matched.
		# Truncated/subsampled controls intentionally use less data and are reported separately.
		panel = [
			r
			for arm, r in results.items()
			if arm not in {"clean_truncated", "clean_subsampled"} and r["status"] == "result"
		]
		field = "count" if view == "equal_query" else "output_tokens"
		amounts = [r["selection"][field] for r in panel]
		if not amounts or any(x is None for x in amounts) or len(set(amounts)) != 1:
			issues.append(f"{view}: unequal actual {field}")
		if (
			view == "equal_query"
			and len({tuple(r["selection"]["example_ids"]) for r in panel}) != 1
		):
			issues.append(f"{view}: questions are not paired")
		if len({r["selection"]["seed"] for r in panel}) != 1:
			issues.append(f"{view}: subset seeds differ")
		if any(r["selection"]["exhausted"] for r in panel):
			issues.append(f"{view}: cached pool exhausted before budget")
		for attack in adaptive:
			if (
				results[attack]["status"] == "result"
				and results[candidate]["status"] == "result"
				and results[attack]["parent_digest"] != results[candidate]["arm_digest"]
			):
				issues.append(
					f"{view}/{attack}: adaptive data is not derived from candidate outputs"
				)
		clears = {}
		if all(results[a]["status"] == "result" for a in [candidate, *BASELINES, *adaptive]):
			for seed in seeds:
				strongest = max(
					[candidate, *adaptive], key=lambda a: results[a]["intervals"][seed][1]
				)
				upper = results[strongest]["intervals"][seed][1]
				clears[seed] = {
					"strongest_declared_attacker_arm": strongest,
					"worst_candidate_upper": upper,
					"clears": {b: upper < results[b]["intervals"][seed][0] for b in BASELINES},
				}
			if not all(all(row["clears"].values()) for row in clears.values()):
				issues.append(
					f"{view}: intervals do not clear both baselines under all declared attacks"
				)
		views[view] = {"arms": results, "per_seed_comparison": clears}
	# Reuse the established capacity and matched-utility contract, one call per stratum.
	utility = {}
	try:
		capacity, _ = _read(declaration["capacity_evidence"])
		schedule_sha = hashlib.sha256(Path(declaration["benign_schedule"]).read_bytes()).hexdigest()
		dev = matching["development"]
		if schedule_sha != dev["schedule_sha256"]:
			raise ValueError("benign schedule differs from the measured development schedule")
		baseline = declaration["utility_baseline"]
		validity = utility_validity_check(
			read_index(Path(baseline["index"])),
			[{"arm": baseline["arm"], "runs": [baseline["run_id"]]}],
			baseline_arm=baseline["arm"],
			require_baseline=True,
		)
		for stratum, loss in dev["candidate"]["loss_by_stratum"].items():
			conditions = []
			for arm in [candidate, *BASELINES]:
				selected = matching["baselines"].get(arm, {}).get("selected")
				observed = (
					loss
					if arm == candidate
					else selected["setting"]["loss_by_stratum"][stratum]
					if selected
					else None
				)
				conditions.append(
					{
						"arm": arm,
						"utility_metric": "task_success_loss",
						"utility_target": loss,
						"utility_observed": observed,
						"utility_population": f"{dev['population_id']}/{stratum}",
						"utility_tolerance": dev["tolerance"],
					}
				)
			utility[stratum] = compare_conditions(
				conditions,
				"legitimate_utility",
				capacity_evidence=capacity,
				schedule_sha256=schedule_sha,
				utility_validity=validity,
			)
			if not utility[stratum]["comparable"]:
				issues.append(f"{stratum}: capacity or legitimate utility not comparable")
	except (ValueError, KeyError, OSError) as error:
		issues.append(f"capacity evidence: {error}")
	return {
		"version": "quota-student-comparison-v1",
		"manifest_digest": manifest_digest,
		"matching": matching,
		"utility_by_stratum": utility,
		"views": views,
		"gate": gate,
		"seeds": seeds,
		"evidence_kind": declaration.get("evidence_kind", "undeclared"),
		"evidence": "screening only" if len(seeds) == 1 else "declared training seeds",
		"verdict": "candidate beats matched quota baselines" if not issues else "no result",
		"refusal_reasons": issues,
		"rule": "At every paired seed and in both budget views, the worst candidate/adaptive upper "
		"transfer bound must be below both quota lower bounds. Uncertainty remains per seed.",
	}
