"""The results index uses a synthetic run directory, without a teacher or harness."""

import csv
import hashlib
import json
from pathlib import Path

import pytest
from distillation.results_index import (
	BASELINE_POLICIES,
	COLUMNS,
	_attributed_fields,
	backfill_rows,
	comparison_from_index,
	read_index,
	run_rows,
	student_rows,
	update_index,
	utility_validity,
	utility_validity_check,
)


def _write(path: Path, value: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(value if isinstance(value, str) else json.dumps(value))


def _run(root: Path, name: str, contract: str) -> Path:
	run = root / name
	_write(run / "calibration" / "calibration.json", {"config": {}})
	_write(
		run / "schedules" / "A1" / "manifest.json",
		{"seed": 42, "attempt_budget": 6000, "attacker_partition": "exploratory"},
	)
	_write(run / "schedules" / "A1" / "requests.jsonl", "{}\n")
	complete = run / "campaigns" / "A1" / "quota"
	_write(complete / "harness.env", "HARNESS_POLICY=quota\nHARNESS_ACCUMULATOR=paper\n")
	_write(complete / "decisions.jsonl", json.dumps({"policy_revision": "rev"}) + "\n")
	_write(complete / "campaign" / "campaign.json", {"contract": contract})
	_write(complete / "campaign" / "ledger.jsonl", f'{{"run": "{name}"}}\n')
	events = [
		{
			"event": "attempt",
			"id": "1",
			"kind": "query",
			"policy_request_id": "p1",
			"example_id": "x",
			"at_seconds": 0,
			"index": 1,
		},
		{"event": "outcome", "id": "1", "status": "received", "output_tokens": None},
	]
	_write(
		complete / "campaign" / "campaign.events.jsonl",
		"".join(json.dumps(e) + "\n" for e in events),
	)
	refused = run / "campaigns" / "A1" / "full"
	_write(refused / "harness.env", "HARNESS_POLICY=mmd\nHARNESS_ACCUMULATOR=paper\n")
	_write(
		refused / "outcome",
		"gate refused: attacker not escalated beyond held-out benign accounts\n",
	)
	utility = {
		"submitted": 4,
		"served": 3,
		"completion_rate": 0.75,
		"strata": {
			"client_type": {
				"specialist": {"submitted": 2, "served": 1, "latency_seconds": {"p95": 9.0}},
				"batch": {"submitted": 2, "served": 2, "latency_seconds": {"p95": 1.0}},
			}
		},
	}
	_write(run / "utility" / "quota" / "harness.env", "HARNESS_POLICY=quota\n")
	_write(run / "utility" / "quota" / "utility.json", utility)
	return run


def test_rows_keep_unknowns_and_replace_a_rerun(tmp_path: Path) -> None:
	index = tmp_path / "readings" / "index.tsv"
	first = run_rows(_run(tmp_path, "run-a", "c1"), "v5", "abc")
	update_index(index, first)
	rows = {row["arm"]: row for row in read_index(index) if row["kind"] == "campaign"}
	utility = {r["stratum"]: r for r in read_index(index) if r["kind"] == "utility"}
	assert utility["all"]["completion_rate"] == "0.75"
	assert utility["client_type=specialist"]["completion_rate"] == "0.5"
	assert utility["client_type=specialist"]["latency_p95"] == "9.0"
	assert rows["quota"]["status"] == "complete"
	assert rows["quota"]["study"] == "development"
	assert rows["quota"]["received"] == "1"
	assert rows["quota"]["released_tokens"] == ""
	assert rows["quota"]["accumulator"] == ""
	assert rows["full"]["status"] == "gate-refused"
	assert rows["full"]["contract_sha256"] == ""

	update_index(index, run_rows(_run(tmp_path, "run-b", "c2"), "v6", "def"))
	update_index(index, first)
	assert len(read_index(index)) == 10

	with pytest.raises(ValueError, match="contract"):
		update_index(index, run_rows(_run(tmp_path, "run-c", "c1"), "v7", "ghi"))


def _row(run: str, arm: str, received: str, study: str = "development") -> dict[str, str]:
	row = dict.fromkeys(COLUMNS, "")
	row.update(
		run_id=run,
		kind="campaign",
		stratum="all",
		attack="A1",
		arm=arm,
		policy=arm,
		status="complete",
		study=study,
		attempt_budget="6000",
		deadline_seconds="86400",
		received=received,
	)
	return row


def test_comparison_pools_runs_and_never_invents_budgets() -> None:
	rows = [_row("r1", "quota", "1215"), _row("r2", "quota", "1213"), _row("r1", "rate", "6000")]
	rows.append({**_row("r3", "rate", ""), "status": "interrupted"})
	for stratum, completion in (("all", "0.9"), ("client_type=specialist", "0.5")):
		utility = {**_row("r1", "quota", ""), "kind": "utility", "attack": "benign_test"}
		rows.append({**utility, "stratum": stratum, "completion_rate": completion})
	conditions, outcomes = comparison_from_index(
		rows, attack="A1", arms=["quota", "rate"], view="attacker_budget", declared={}
	)
	quota, rate = conditions
	assert quota["runs"] == ["r1", "r2"] and quota["query_budget"] == 6000
	assert rate["excluded_runs"] == {"r3": "interrupted"}
	assert quota["output_token_budget"] is None and quota["declared"] == []
	received = outcomes["quota"]["campaign"]["received"]
	assert received["estimate"] == 1214 and received["interval"]["clusters"] == 2
	assert outcomes["rate"]["campaign"]["received"]["interval"] is None
	assert outcomes["quota"]["utility"]["completion_rate"]["estimate"] == 0.9
	specialist = outcomes["quota"]["utility_by_stratum"]["client_type=specialist"]
	assert specialist["completion_rate"]["estimate"] == 0.5
	assert outcomes["quota"]["campaign"]["released_tokens"]["unknown_units"] == 2
	assert quota["commits"] == ["unknown"]
	# Backfilled rows ran other code; they are pooled only when asked for.
	backfilled = [*rows, {**_row("backfill:v3", "quota", "1215")}]
	assert comparison_from_index(
		backfilled, attack="A1", arms=["quota", "rate"], view="attacker_budget", declared={}
	)[0][0]["runs"] == ["r1", "r2"]
	with pytest.raises(ValueError, match="different policy"):
		comparison_from_index(
			[*rows, {**_row("r5", "quota", "1"), "policy": "rate"}],
			attack="A1",
			arms=["quota", "rate"],
			view="attacker_budget",
			declared={},
		)
	with pytest.raises(ValueError, match="reasoning mode"):
		comparison_from_index(
			[*rows, {**_row("r6", "rate", "6000"), "teacher_thinking": "on"}],
			attack="A1",
			arms=["quota", "rate"],
			view="attacker_budget",
			declared={},
		)
	with pytest.raises(ValueError, match="development and confirmatory"):
		comparison_from_index(
			[*rows, _row("r4", "rate", "6000", study="confirmatory")],
			attack="A1",
			arms=["quota", "rate"],
			view="attacker_budget",
			declared={},
		)


def test_student_rows_link_to_the_campaign_that_supplied_their_ledger(tmp_path: Path) -> None:
	index = tmp_path / "index.tsv"
	run = _run(tmp_path, "run-a", "c1")
	update_index(index, run_rows(run, "v5", "abc"))
	ledger = run / "campaigns" / "A1" / "quota" / "campaign" / "ledger.jsonl"
	curve = tmp_path / "curve"
	_write(
		curve / "plan.json",
		{
			"inputs": {
				str(ledger): hashlib.sha256(ledger.read_bytes()).hexdigest(),
				"/elsewhere/development.jsonl": "f" * 64,
				"/elsewhere/suite.json": "e" * 64,
			}
		},
	)
	results = [
		{
			"n": 500,
			"seed_gains_pp": [19.0, 20.0, 21.0],
			"mean_gain_pp": 20.0,
			"ci95_pp": [16.4, 23.8],
		},
		{
			"n": 1500,
			"seed_gains_pp": [24.0, 25.0, 26.0],
			"mean_gain_pp": 25.0,
			"ci95_pp": [21.0, 29.0],
		},
	]
	_write(curve / "summary.json", {"results": results, "target_gain_pp": 21.0})
	rows = student_rows(curve, read_index(index))
	assert [(r["stratum"], r["student_target_reached"]) for r in rows] == [
		("n=500", False),
		("n=1500", True),
	]
	assert rows[0]["training_runs"] == "v5/run-a/A1/quota"
	assert len(rows[0]["training_ledger_sha256"].split(";")) == 2
	update_index(index, rows)
	students = [r for r in read_index(index) if r["kind"] == "student"]
	assert [r["student_gain_lower"] for r in students] == ["21.0", "16.4"]


def test_older_indexes_read_with_new_columns_empty(tmp_path: Path) -> None:
	index = tmp_path / "index.tsv"
	index.write_text("run_id\tkind\tattack\tarm\tstratum\nr1\tcampaign\tA1\tquota\tall\n")
	(row,) = read_index(index)
	assert row["arm"] == "quota" and row["received"] == "" and len(row) == len(COLUMNS)
	index.write_text("run_id\tsurprise\nr1\tx\n")
	with pytest.raises(ValueError, match="surprise"):
		read_index(index)


def _bound(preflight: Path, served: int = 19) -> dict[str, object]:
	preflight.write_text("{}")
	return {
		"submitted": 20,
		"served": served,
		"baseline": {
			"preflight_sha256": hashlib.sha256(b"{}").hexdigest(),
			"min_completion_rate": 0.95,
		},
	}


def test_completion_floor_applies_to_baselines_and_binding_to_every_arm(tmp_path: Path) -> None:
	preflight = tmp_path / "teacher-preflight.json"
	assert "allow" in BASELINE_POLICIES and "none" in BASELINE_POLICIES
	assert utility_validity(_bound(preflight), "none", preflight)["validity"] == "valid"
	low = utility_validity(_bound(preflight, served=18), "none", preflight)
	assert low["validity"] == "invalid" and "completed 18 of 20" in low["validity_reason"]
	assert low["min_completion_rate"] == 0.95
	# A defended arm serving less is the outcome being measured, once its teacher is qualified.
	assert utility_validity(_bound(preflight, served=2), "mmd", preflight)["validity"] == "valid"
	bound = _bound(preflight)
	assert utility_validity(bound, "none", None)["validity"] == "unknown"
	preflight.write_text("changed")
	assert utility_validity(bound, "none", preflight)["validity"] == "invalid"
	assert utility_validity({"served": 20, "submitted": 20}, "none", preflight)["validity"] == (
		"unknown"
	)


def test_run_rows_copy_validity_to_every_stratum(tmp_path: Path) -> None:
	run = _run(tmp_path, "run-a", "c1")
	utility = json.loads((run / "utility" / "quota" / "utility.json").read_text())
	preflight = run / "teacher-preflight.json"
	_write(run / "utility" / "quota" / "utility.json", {**utility, **_bound(preflight, served=3)})
	rows = [r for r in run_rows(run, "v5", "abc") if r["kind"] == "utility"]
	assert {r["validity"] for r in rows} == {"valid"} and len(rows) == 3
	assert {r["preflight_sha256"] for r in rows} == {hashlib.sha256(b"{}").hexdigest()}


def test_older_utility_rows_remain_unknown_without_a_binding(
	tmp_path: Path,
) -> None:
	index = tmp_path / "readings" / "index.tsv"
	index.parent.mkdir()
	index.write_text(
		"run_id\tkind\tattack\tarm\tstratum\tcompletion_rate\n"
		"backfill:v3\tutility\tbenign_test\tundefended\tall\t0.2898\n"
	)
	(old,) = read_index(index)
	assert old["validity"] == "unknown" and old["validity_reason"]
	_write(
		index.parent / "v3" / "utility" / "undefended.json",
		{"submitted": 24047, "served": 6968, "completion_rate": 0.28976587516114277},
	)
	(row,) = backfill_rows(index.parent)
	assert row["validity"] == "unknown" and row["policy"] == "none"
	assert row["validity_reason"]
	# Writing another run leaves the historical row's stored validity cell empty.
	update_index(index, [{"run_id": "gate:x", "kind": "gate", "attack": "phase0-headroom"}])
	with index.open() as stream:
		stored = next(r for r in csv.DictReader(stream, delimiter="\t") if r["kind"] == "utility")
	assert stored["validity"] == "" and stored["validity_reason"] == ""
	rewritten = next(r for r in update_index(index, [row]) if r["kind"] == "utility")
	assert rewritten["validity"] == "unknown" and rewritten["completion_rate"]


def test_validity_check_covers_compared_arms_and_the_baseline_in_each_run() -> None:
	def utility(run: str, arm: str, validity: str, policy: str) -> dict[str, str]:
		row = _row(run, arm, "")
		row.update(kind="utility", validity=validity, validity_reason=validity, policy=policy)
		return row

	conditions = [{"arm": "quota", "runs": ["r1", "r2"]}, {"arm": "full", "runs": ["r1"]}]
	rows = [
		utility("r1", "quota", "valid", "quota"),
		utility("r2", "quota", "valid", "quota"),
		utility("r1", "full", "valid", "mmd"),
		utility("r1", "allow", "valid", "allow"),
		utility("r2", "allow", "valid", "allow"),
	]
	check = utility_validity_check(rows, conditions, baseline_arm="allow", require_baseline=True)
	assert check["valid"] is True and check["issues"] == []
	invalid = [*rows[:-1], utility("r2", "allow", "invalid", "allow")]
	check = utility_validity_check(invalid, conditions, baseline_arm="allow", require_baseline=True)
	assert check["valid"] is False and check["arms"]["allow"] == "invalid"
	assert [(i["arm"], i["run_id"]) for i in check["issues"]] == [("allow", "r2")]
	check = utility_validity_check(
		rows[:-1], conditions, baseline_arm="allow", require_baseline=True
	)
	assert check["issues"][0]["reason"] == "no utility row in the index"
	check = utility_validity_check(rows, conditions, baseline_arm="quota", require_baseline=True)
	assert "not a baseline" in check["issues"][0]["reason"]
	assert not utility_validity_check(rows, conditions, baseline_arm=None, require_baseline=True)[
		"valid"
	]


def test_attributed_columns_are_unknown_without_attribution() -> None:
	assert _attributed_fields({"attempts": 3}, None) == {
		"policy_binding": None,
		"campaign_digest": None,
	}
	fields = _attributed_fields(
		{
			"restricted": 3,
			"restrictions_by_kind": {"campaign": 2, "output": 1},
			"restriction_give_ups": 1,
			"output_receipts": {"received": 4, "released_hash_mismatches": 0},
		},
		{"policy_binding": {"status": "bound", "campaign_digests": ["d"]}},
	)
	assert fields["policy_binding"] == "bound" and fields["campaign_digest"] == "d"
	assert fields["restrictions_campaign"] == 2 and fields["restrictions_entitlement"] == 0
	assert fields["restrictions_output"] == 1 and fields["output_receipts"] == 4
	assert set(fields) <= set(COLUMNS)
