"""CPU fixture runner exercises grid planning, execution and measured matching."""

import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from distillation.cli import app
from distillation.development_grid import derive_grid, digest, execute_grid, load_plan, plan_grid
from distillation.development_grid_runner import gateway_command, gateway_environment, run_setting
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


GATEWAY = {
	"port": 18999,
	"tasks_dir": "tasks",
	"max_output_tokens": {"gsm8k": 64},
	"env": {"HARNESS_TASK": "gateway"},
	"candidate": {"policy": "mmd", "env": {"HARNESS_ACCUMULATOR": "paper"}},
}
# A real listener that reports as a gateway does, and starts a child the runner must also stop.
FIXTURE_GATEWAY = (
	sys.executable,
	"-c",
	"""
import http.server, json, os, subprocess, sys

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
open("child.pid", "w").write(str(child.pid))
report = {
	"status": "ready",
	"policy": os.environ.get("FIXTURE_POLICY", os.environ["HARNESS_POLICY"]),
	"instance_id": os.environ["HARNESS_INSTANCE_ID"],
}


class Health(http.server.BaseHTTPRequestHandler):
	def do_GET(self):
		self.send_response(200)
		self.end_headers()
		self.wfile.write(json.dumps(report).encode())

	def log_message(self, *args):
		pass


address = (os.environ["HARNESS_HOST"], int(os.environ["HARNESS_PORT"]))
http.server.HTTPServer(address, Health).serve_forever()
""",
)


def _gateway() -> dict[str, Any]:
	with socket.socket() as sock:
		sock.bind(("127.0.0.1", 0))
		return GATEWAY | {"port": sock.getsockname()[1]}


def _assert_stopped(output: Path) -> None:
	pid = int((output / "child.pid").read_text())
	deadline = time.monotonic() + 5
	while time.monotonic() < deadline:
		try:
			os.kill(pid, 0)
		except ProcessLookupError:
			return
		time.sleep(0.05)
	os.kill(pid, 9)
	pytest.fail("the launcher's child outlived its cell")


def test_gateway_environment_isolates_each_setting(tmp_path: Path) -> None:
	setting = {
		"id": "tokens",
		"policy": "quota_tokens",
		"request_quota": 2,
		"output_token_budget": 9,
	}
	inherited = {"PATH": "/bin", "HARNESS_OUTPUT_POLICY": "{}", "HARNESS_TEACHER_TOKEN": "secret"}
	environment = gateway_environment(setting, GATEWAY, tmp_path / "s", tmp_path, inherited)
	assert environment["HARNESS_POLICY"] == "quota_tokens"
	assert environment["HARNESS_REQUEST_QUOTA"] == "2"
	assert environment["HARNESS_OUTPUT_TOKEN_BUDGET"] == "9"
	assert environment["HARNESS_STATE_DB"] == str(tmp_path / "s")
	assert environment["HARNESS_TEACHER_TOKEN"] == "secret"
	assert environment["PATH"] == "/bin" and "HARNESS_OUTPUT_POLICY" not in environment
	candidate = gateway_environment(
		{"id": "c", "policy": "candidate"}, GATEWAY, tmp_path / "s", tmp_path, {}
	)
	assert candidate["HARNESS_POLICY"] == "mmd" and candidate["HARNESS_ACCUMULATOR"] == "paper"
	with pytest.raises(ValueError, match="may not declare"):
		gateway_environment(
			{"id": "x", "policy": "allow", "env": {"HARNESS_STATE_DB": "/elsewhere"}},
			GATEWAY,
			tmp_path / "s",
			tmp_path,
			{},
		)


def test_only_a_declared_campaign_uses_the_product_launcher(tmp_path: Path) -> None:
	state = tmp_path / "s"
	assert gateway_command({"policy": "candidate"}, GATEWAY, state) == ["legacy"]
	declared = {**GATEWAY, "candidate": {"policy": "allow", "campaign_config": "/c.json"}}
	assert gateway_command({"policy": "quota"}, declared, state) == ["legacy"]
	assert gateway_command({"policy": "candidate"}, declared, state) == [
		"serve",
		"--config",
		"/c.json",
		"--state-db",
		str(state),
		"--base-policy",
		"allow",
	]


def test_the_runner_measures_a_grid_and_always_stops_its_gateway(tmp_path: Path) -> None:
	root = tmp_path / "grid"
	plan_grid(spec(tmp_path) | {"gateway": _gateway()}, root)
	started: list[str] = []

	def runner(setting: dict[str, Any], state: Path, output: Path, plan: dict[str, Any]) -> None:
		def measure(*args: Any) -> None:
			fixture_runner(setting, state, output, plan)
			started.append(setting["id"])

		policy = "mmd" if setting["policy"] == "candidate" else setting["policy"]
		run_setting(
			setting,
			state,
			output,
			plan,
			python=FIXTURE_GATEWAY,
			measure=measure,
		)
		_assert_stopped(output)
		assert json.loads((output / "gateway.json").read_text())["policy"] == policy
		assert (output / "credentials.json").stat().st_mode & 0o777 == 0o600

	execute_grid(root, runner)
	assert started == ["allow", "candidate", "quota", "tokens"]
	assert derive_grid(root)["development"]["schedule_sha256"]


def _refused_cell(tmp_path: Path, message: str, **overrides: Any) -> Path:
	root = tmp_path / "grid"
	plan_grid(spec(tmp_path) | {"gateway": _gateway()}, root)
	output = tmp_path / "cell"
	output.mkdir()
	with pytest.raises(ValueError, match=message):
		run_setting(
			{"id": "allow", "policy": "allow"},
			output / "state.sqlite",
			output,
			load_plan(root) | {"root": str(root)},
			**({"python": FIXTURE_GATEWAY} | overrides),
		)
	return output


def _measured(*args: Any) -> None:
	pytest.fail("measured the wrong gateway")


def test_the_runner_refuses_a_port_already_serving(tmp_path: Path) -> None:
	ready = {"status": "ready", "policy": "allow", "instance_id": "foreign"}
	_refused_cell(tmp_path, "already serving", probe=lambda url: ready, measure=_measured)


def test_the_runner_refuses_a_service_that_took_the_port_after_the_check(tmp_path: Path) -> None:
	replies = iter([None, {"status": "ready", "policy": "allow", "instance_id": "foreign"}])
	output = _refused_cell(
		tmp_path, "another process", probe=lambda url: next(replies), measure=_measured
	)
	assert not (output / "gateway.json").exists()


def test_the_runner_refuses_its_own_gateway_on_the_wrong_policy(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setenv("FIXTURE_POLICY", "quota")
	_assert_stopped(_refused_cell(tmp_path, "gateway serves quota", measure=_measured))


def test_a_failed_measurement_still_stops_the_gateway_and_its_children(tmp_path: Path) -> None:
	def measure(*args: Any) -> None:
		raise ValueError("measurement failed")

	_assert_stopped(_refused_cell(tmp_path, "measurement failed", measure=measure))
