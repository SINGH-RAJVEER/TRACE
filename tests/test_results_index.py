"""The results index uses a synthetic run directory, without a teacher or harness."""

import json
from pathlib import Path

import pytest
from distillation.results_index import read_index, run_rows, update_index


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
	return run


def test_rows_keep_unknowns_and_replace_a_rerun(tmp_path: Path) -> None:
	index = tmp_path / "readings" / "index.tsv"
	first = run_rows(_run(tmp_path, "run-a", "c1"), "v5", "abc")
	update_index(index, first)
	rows = {row["arm"]: row for row in read_index(index)}
	assert rows["quota"]["status"] == "complete"
	assert rows["quota"]["study"] == "development"
	assert rows["quota"]["released"] == "1"
	assert rows["quota"]["released_tokens"] == ""
	assert rows["quota"]["accumulator"] == ""
	assert rows["full"]["status"] == "gate-refused"
	assert rows["full"]["contract_sha256"] == ""

	update_index(index, run_rows(_run(tmp_path, "run-b", "c2"), "v6", "def"))
	update_index(index, first)
	assert len(read_index(index)) == 4

	with pytest.raises(ValueError, match="contract"):
		update_index(index, run_rows(_run(tmp_path, "run-c", "c1"), "v7", "ghi"))
