"""CPU fixture runner exercises grid planning, execution and measured matching."""

import json
from pathlib import Path
from typing import Any

import pytest
from distillation.cli import app
from distillation.development_grid import derive_grid, digest, execute_grid, plan_grid
from typer.testing import CliRunner


def spec(tmp_path: Path) -> dict[str, Any]:
	schedule = tmp_path / "input.jsonl"
	schedule.write_text(
		"\n".join(
			json.dumps(
				{
					"account": "benign",
					"text": "fixture",
					"time": i,
					"example_id": str(i),
					"client_type": c,
				}
			)
			for i, c in enumerate(("specialist", "batch"))
		)
	)
	capacity = tmp_path / "preflight.json"
	capacity.write_text(
		json.dumps(
			{
				"version": "teacher-capacity-v1",
				"passed": True,
				"schedule_sha256": digest(schedule),
				"slo": {},
				"checks": dict.fromkeys(
					(
						"offered_throughput",
						"completed_throughput",
						"latency_p95",
						"errors",
						"timeouts",
						"queue_end",
						"queue_growth",
						"undefended",
					),
					True,
				),
				"offered_rps": 2,
				"completed_rps": 2,
				"concurrency": 1,
				"timeout_seconds": 10,
			}
		)
	)
	return {
		"split": "development",
		"population_id": "fixture",
		"schedule": str(schedule),
		"capacity": str(capacity),
		"offered_rps": 2,
		"min_completion_rate": 1,
		"concurrency": 1,
		"request_deadline": 10,
		"tolerance": 0.01,
		"strata": {c: {"field": "client_type", "value": c} for c in ("specialist", "batch")},
		"settings": [
			{"id": "allow", "policy": "allow"},
			{"id": "candidate", "policy": "candidate"},
			{"id": "quota", "policy": "quota", "request_quota": 1},
			{
				"id": "tokens",
				"policy": "quota_tokens",
				"request_quota": 2,
				"output_token_budget": 10,
			},
		],
	}


def fixture_runner(
	setting: dict[str, Any], state: Path, output: Path, plan: dict[str, Any]
) -> None:
	assert not state.exists()
	state.write_bytes(b"fixture state")
	rows = [
		{
			"account": "benign",
			"sequence": i,
			"example_id": str(i),
			"domain": "math",
			"client_type": c,
			"received": True,
			"quality": 1 if setting["policy"] == "allow" else (0 if i == 0 else 1),
		}
		for i, c in enumerate(("specialist", "batch"))
	]
	(output / "utility.json").write_text(
		json.dumps(
			{
				"measurement_version": "utility-response-latency-v3",
				"baseline": plan["binding"],
				"load": {"concurrency": 1, "deadline_seconds": 10},
				"submitted": 2,
				"results": rows,
			}
		)
	)


def test_grid_cli_plan_derive_resume_and_tamper(tmp_path: Path) -> None:
	declaration = spec(tmp_path)
	input_file = tmp_path / "spec.json"
	input_file.write_text(json.dumps(declaration))
	root = tmp_path / "grid"
	runner = CliRunner()
	assert (
		runner.invoke(
			app, ["development-grid", "plan", "--spec", str(input_file), "--output", str(root)]
		).exit_code
		== 0
	)
	execute_grid(root, fixture_runner)
	calls = []
	execute_grid(root, lambda *args: calls.append(args))
	assert calls == []
	result = derive_grid(root)
	assert result["development"]["candidate"]["loss_by_stratum"] == {"specialist": 1, "batch": 0}
	for baseline in result["matches"]["baselines"].values():
		assert baseline["status"] == "matched"
		assert "utility.json" in baseline["selected"]["setting"]["run_artifacts"]["artifacts"]
	assert (
		runner.invoke(
			app,
			[
				"development-grid",
				"match",
				"--output",
				str(root),
				"--result",
				str(tmp_path / "matches.json"),
			],
		).exit_code
		== 0
	)
	(root / "quota" / "state.sqlite").write_bytes(b"changed")
	with pytest.raises(ValueError, match="hash mismatch"):
		execute_grid(root, fixture_runner)
	with pytest.raises(ValueError, match="hash mismatch"):
		derive_grid(root)


def test_partial_state_and_invalid_capacity_refused(tmp_path: Path) -> None:
	declaration = spec(tmp_path)
	root = tmp_path / "grid"
	plan_grid(declaration, root)
	(root / "allow").mkdir()
	with pytest.raises(ValueError, match="reuse"):
		execute_grid(root, fixture_runner)
	capacity = Path(declaration["capacity"])
	value = json.loads(capacity.read_text()) | {"passed": False}
	capacity.write_text(json.dumps(value))
	with pytest.raises(ValueError, match="passing"):
		plan_grid(declaration, tmp_path / "bad")


def test_no_match_and_missing_stratum(tmp_path: Path) -> None:
	declaration = spec(tmp_path)
	root = tmp_path / "grid"
	plan_grid(declaration, root)

	def different(setting: dict[str, Any], state: Path, output: Path, plan: dict[str, Any]) -> None:
		fixture_runner(setting, state, output, plan)
		if setting["policy"] == "quota":
			path = output / "utility.json"
			run = json.loads(path.read_text())
			run["results"][0]["quality"] = 1
			path.write_text(json.dumps(run))

	execute_grid(root, different)
	assert derive_grid(root)["matches"]["baselines"]["quota"]["status"] == "not matched"
	declaration["strata"]["absent"] = {"field": "client_type", "value": "absent"}
	other = tmp_path / "other"
	plan_grid(declaration, other)
	execute_grid(other, fixture_runner)
	with pytest.raises(ValueError, match="missing declared stratum"):
		derive_grid(other)


@pytest.mark.parametrize("damage", ["binding", "population", "nonresponse", "improvement"])
def test_unusable_measurements_refused(tmp_path: Path, damage: str) -> None:
	root = tmp_path / "grid"
	plan_grid(spec(tmp_path), root)

	def damaged(setting: dict[str, Any], state: Path, output: Path, plan: dict[str, Any]) -> None:
		fixture_runner(setting, state, output, plan)
		path = output / "utility.json"
		run = json.loads(path.read_text())
		if damage == "binding":
			run["baseline"] = {}
		elif damage == "population":
			run["results"].pop()
			run["submitted"] = 1
		elif damage == "nonresponse":
			run["results"][1]["received"] = False
		elif setting["policy"] == "allow":
			run["results"][1]["quality"] = 0
		path.write_text(json.dumps(run))

	with pytest.raises(ValueError):
		execute_grid(root, damaged)
		derive_grid(root)
