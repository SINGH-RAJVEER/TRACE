"""Tests use synthetic records, without a teacher or student training."""

import pytest
from distillation.cli import app
from distillation.evaluation_protocol import (
	benign_profiles,
	campaign_report,
	clopper_pearson,
	cluster_interval,
	compare_conditions,
	student_plan,
	zero_event_upper,
)
from typer.testing import CliRunner


def test_campaign_unknowns_and_interventions() -> None:
	events = [
		{
			"event": "attempt",
			"id": "1",
			"kind": "query",
			"policy_request_id": "p1",
			"example_id": "x",
			"at_seconds": 0,
		},
		{"event": "outcome", "id": "1", "status": "received", "output_tokens": None},
		{"event": "policy", "policy_request_id": "p1", "level": 1, "action": "serve"},
		{
			"event": "attempt",
			"id": "2",
			"kind": "query",
			"policy_request_id": "p2",
			"at_seconds": 4,
		},
		{"event": "policy", "policy_request_id": "p2", "level": 2, "action": "throttle"},
		{"event": "outcome", "id": "2", "status": "failed"},
	]
	report = campaign_report(events)
	assert report["admitted"] is None and report["cancelled"] is None
	assert report["released_tokens"] is None and report["released_tokens_unknown_count"] == 1
	assert report["distinct_examples"] == 1 and report["failed"] == 1
	assert report["first_alert_seconds"] == 0 and report["first_restriction_seconds"] == 4
	assert report["first_refusal_seconds"] is None


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
