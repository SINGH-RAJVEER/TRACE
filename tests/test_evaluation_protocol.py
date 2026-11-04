"""Tests use synthetic records, without a teacher or student training."""

import hashlib
import json
from pathlib import Path

import pytest
from distillation import results_index
from distillation.cli import app
from distillation.evaluation_protocol import (
	benign_profiles,
	clopper_pearson,
	cluster_interval,
	compare_conditions,
	student_plan,
	zero_event_upper,
)
from typer.testing import CliRunner


def test_profiles_intervals_and_matching() -> None:
	profiles = benign_profiles(
		[{"profile": "specialist", "restricted": False}, {"profile": "specialist"}]
	)
	assert profiles["specialist"]["restricted"] == {"yes": 0, "known": 1, "total": 2}
	assert 0.009 < zero_event_upper(300) < 0.011
	assert 0.016 < clopper_pearson(0, 200)["upper"] < 0.019
	assert cluster_interval([("a", 0), ("a", 1), ("b", 1)])["estimate"] == 0.75
	with pytest.raises(ValueError):
		cluster_interval([("a", 0)])
	conditions = [{"query_budget": 10}, {"query_budget": 10}]
	assert not compare_conditions(conditions, "attacker_budget")["comparable"]


def test_student_plan_and_cli(tmp_path) -> None:
	plan = student_plan(["student-a", "student-b"], [1, 2, 3], 2)
	assert len(plan["runs"]) == 114 and plan["estimated_cost"] == 228
	assert not plan["executes_training"]
	result = CliRunner().invoke(
		app,
		[
			"student-plan",
			"--family",
			"s",
			"--seed",
			"1",
			"--cost-per-run",
			"2",
			"--output",
			str(tmp_path / "plan.json"),
		],
	)
	assert result.exit_code == 0, result.output
	assert (tmp_path / "plan.json").exists()


def test_comparisons_require_matching_passing_teacher_capacity(tmp_path) -> None:
	import hashlib
	import json

	conditions = [
		{
			"query_budget": 100,
			"output_token_budget": 1000,
			"money_budget": 1,
			"deadline_seconds": 10,
		},
	] * 2
	checks = dict.fromkeys(
		[
			"offered_throughput",
			"completed_throughput",
			"latency_p95",
			"errors",
			"timeouts",
			"queue_end",
			"queue_growth",
			"undefended",
		],
		True,
	)
	schedule = tmp_path / "requests.jsonl"
	schedule.write_text("frozen fixture\n")
	sha = hashlib.sha256(schedule.read_bytes()).hexdigest()
	evidence = {
		"version": "teacher-capacity-v1",
		"passed": True,
		"schedule_sha256": sha,
		"slo": {},
		"checks": checks,
	}
	assert not compare_conditions(conditions, "attacker_budget")["comparable"]
	assert compare_conditions(
		conditions, "attacker_budget", capacity_evidence=evidence, schedule_sha256=sha
	)["comparable"]
	for invalid in [evidence | {"passed": False}, evidence | {"schedule_sha256": "other"}]:
		assert not compare_conditions(
			conditions, "attacker_budget", capacity_evidence=invalid, schedule_sha256=sha
		)["comparable"]
	conditions_file = tmp_path / "conditions.json"
	conditions_file.write_text(json.dumps(conditions))
	evidence_file = tmp_path / "capacity.json"
	evidence_file.write_text(json.dumps(evidence))
	args = [
		"gateway",
		"comparison-report",
		"--conditions",
		str(conditions_file),
		"--view",
		"attacker_budget",
		"--capacity-evidence",
		str(evidence_file),
		"--benign-schedule",
		str(schedule),
	]
	result = CliRunner().invoke(app, args)
	assert result.exit_code == 0, result.output
	assert json.loads(result.output)["comparable"] is True
	schedule.write_text("different schedule\n")
	result = CliRunner().invoke(app, args)
	assert result.exit_code == 1, result.output
	assert "teacher_capacity" in result.output


def test_utility_comparison_refuses_an_invalid_baseline_unless_overridden(tmp_path: Path) -> None:
	sha = hashlib.sha256(b"schedule\n").hexdigest()
	schedule = tmp_path / "requests.jsonl"
	schedule.write_text("schedule\n")
	checks = dict.fromkeys(
		[
			"offered_throughput",
			"completed_throughput",
			"latency_p95",
			"errors",
			"timeouts",
			"queue_end",
			"queue_growth",
			"undefended",
		],
		True,
	)
	evidence = tmp_path / "capacity.json"
	evidence.write_text(
		json.dumps(
			{
				"version": "teacher-capacity-v1",
				"passed": True,
				"schedule_sha256": sha,
				"slo": {},
				"checks": checks,
			}
		)
	)
	shared = dict.fromkeys(results_index.COLUMNS, "")
	shared.update(status="complete", study="development", stratum="all", run_id="r1", attack="A1")
	shared.update(attempt_budget="6000", deadline_seconds="86400", completion_rate="0.9")
	rows = []
	for arm in ("quota", "full"):
		rows.append({**shared, "kind": "campaign", "arm": arm, "policy": arm})
		rows.append(
			{**shared, "kind": "utility", "attack": "benign_test", "arm": arm, "policy": arm}
			| {"validity": "valid"}
		)
	baseline = {**shared, "kind": "utility", "attack": "benign_test", "arm": "allow"}
	baseline.update(policy="allow", validity="invalid", validity_reason="completed 29%")
	rows.append(baseline)
	index = tmp_path / "index.tsv"
	results_index.update_index(index, rows)
	args = [
		"gateway",
		"comparison-report",
		"--index",
		str(index),
		"--attack",
		"A1",
		"--arm",
		"quota",
		"--arm",
		"full",
		"--view",
		"legitimate_utility",
		"--utility-metric",
		"completion_rate",
		"--utility-target",
		"0.9",
		"--utility-tolerance",
		"0.01",
		"--utility-baseline-arm",
		"allow",
		"--capacity-evidence",
		str(evidence),
		"--benign-schedule",
		str(schedule),
	]
	refused = CliRunner().invoke(app, args)
	assert refused.exit_code == 1, refused.output
	report = json.loads(refused.output)
	assert report["unmatched_or_missing"] == ["utility_baseline_validity"]
	assert report["utility_validity"]["issues"][0]["reason"] == "completed 29%"
	allowed = CliRunner().invoke(app, [*args, "--allow-invalid-utility-baseline", "pilot only"])
	assert allowed.exit_code == 0, allowed.output
	report = json.loads(allowed.output)
	assert report["comparable"] is True
	assert report["utility_validity"]["valid"] is False
	assert report["utility_validity"]["override"] == "pilot only"
	# The attacker-budget view is not a utility comparison; it marks the values and proceeds.
	budget = [*args]
	budget[budget.index("legitimate_utility")] = "attacker_budget"
	budget += ["--output-token-budget", "1", "--money-budget", "1"]
	marked = CliRunner().invoke(app, budget)
	assert marked.exit_code == 0, marked.output
	assert json.loads(marked.output)["utility_validity"]["arms"]["allow"] == "invalid"
