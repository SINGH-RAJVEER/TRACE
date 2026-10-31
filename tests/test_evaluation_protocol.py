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
		{"event": "policy", "policy_request_id": "p2", "level": 3, "action": "challenge"},
		{"event": "outcome", "id": "2", "status": "failed"},
	]
	report = campaign_report(events)
	assert report["admitted"] is None and report["cancelled"] is None
	assert report["released_tokens"] is None and report["released_tokens_unknown_count"] == 1
	assert report["distinct_examples"] == 1 and report["failed"] == 1
	assert report["first_alert_seconds"] == 0 and report["first_restriction_seconds"] == 4
	assert report["first_refusal_seconds"] is None


def test_redeemed_challenge_still_counts_as_restriction() -> None:
	attempt = {"event": "attempt", "kind": "query", "policy_request_id": "p1", "example_id": "x"}
	events = [
		{**attempt, "id": "1", "at_seconds": 2},
		{"event": "outcome", "id": "1", "status": "challenge"},
		{**attempt, "id": "2", "kind": "proof", "at_seconds": 3},
		{"event": "outcome", "id": "2", "status": "received", "output_tokens": 5},
		# The later redemption decision replaces the challenge decision for this prompt.
		{"event": "policy", "policy_request_id": "p1", "level": 3, "action": "serve"},
	]
	report = campaign_report(events)
	assert report["first_restriction_seconds"] == 2 and report["first_refusal_seconds"] is None
	assert report["released"] == 1 and report["released_tokens"] == 5


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
