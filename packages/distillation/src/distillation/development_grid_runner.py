"""Run one development-grid setting against a real, isolated gateway process.

The spec's `gateway` member declares the service. Nothing here starts a teacher, so its endpoint
must already be serving. Pass `distillation.development_grid_runner:gateway_runner` to
`distill development-grid execute`. It has only run against fixtures, so it measures nothing yet.
"""

import asyncio
import contextlib
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from harness.calibration import BenignRequest

from distillation.tasks import task_directory
from distillation.utility import measure_utility
from distillation.workloads import task_references

# Each setting's identity, so no declaration can move a run off its own state or policy.
_RESERVED = {
	"HARNESS_ACCOUNTS_FILE",
	"HARNESS_DECISION_LOG",
	"HARNESS_HOST",
	"HARNESS_INSTANCE_ID",
	"HARNESS_OUTPUT_TOKEN_BUDGET",
	"HARNESS_POLICY",
	"HARNESS_PORT",
	"HARNESS_REQUEST_QUOTA",
	"HARNESS_STATE_DB",
	"HARNESS_TEST_CLOCK",
	"HARNESS_TRUST_ACTOR_HEADER",
}
_SECRET_PASSTHROUGH = {"HARNESS_TEACHER_TOKEN"}

Probe = Callable[[str], dict[str, Any] | None]


def _settings(extra: dict[str, Any] | None, source: str) -> dict[str, str]:
	values = extra or {}
	bad = [k for k in values if not k.startswith("HARNESS_") or k in _RESERVED]
	if bad:
		raise ValueError(f"{source} may not declare {sorted(bad)}")
	return {key: str(value) for key, value in values.items()}


def gateway_command(setting: dict[str, Any], gateway: dict[str, Any], state: Path) -> list[str]:
	"""The launcher arguments. Only a declared campaign candidate uses the product launcher."""
	candidate = gateway.get("candidate") or {}
	config = candidate.get("campaign_config") if setting["policy"] == "candidate" else None
	if config is None:
		return ["legacy"]
	policy = candidate.get("policy", "allow")
	return ["serve", "--config", str(config), "--state-db", str(state), "--base-policy", policy]


def declared_policy(setting: dict[str, Any], gateway: dict[str, Any]) -> str:
	if setting["policy"] != "candidate":
		return str(setting["policy"])
	policy = (gateway.get("candidate") or {}).get("policy")
	if not isinstance(policy, str):
		raise ValueError("the gateway declaration needs candidate.policy")
	return policy


def gateway_environment(
	setting: dict[str, Any],
	gateway: dict[str, Any],
	state: Path,
	output: Path,
	inherited: dict[str, str],
) -> dict[str, str]:
	"""Inherited HARNESS_ settings are dropped, so a shell cannot change what a cell measures."""
	candidate = gateway.get("candidate") or {}
	environment = {
		k: v
		for k, v in inherited.items()
		if not k.startswith("HARNESS_") or k in _SECRET_PASSTHROUGH
	}
	environment |= _settings(gateway.get("env"), "gateway.env")
	if setting["policy"] == "candidate":
		environment |= _settings(candidate.get("env"), "gateway.candidate.env")
	environment |= _settings(setting.get("env"), f"setting {setting['id']} env")
	environment |= {
		"HARNESS_HOST": "127.0.0.1",
		"HARNESS_PORT": str(gateway["port"]),
		"HARNESS_POLICY": declared_policy(setting, gateway),
		"HARNESS_STATE_DB": str(state),
		"HARNESS_DECISION_LOG": str(output / "decisions.jsonl"),
		"HARNESS_ACCOUNTS_FILE": str(output / "accounts.json"),
		"HARNESS_TRUST_ACTOR_HEADER": "false",
		"HARNESS_TEST_CLOCK": "false",
	}
	if "request_quota" in setting:
		environment["HARNESS_REQUEST_QUOTA"] = str(setting["request_quota"])
	if "output_token_budget" in setting:
		environment["HARNESS_OUTPUT_TOKEN_BUDGET"] = str(setting["output_token_budget"])
	return environment


def write_credentials(output: Path, requests: list[BenignRequest]) -> dict[str, str]:
	"""Fresh per-cell keys. The credentials file is private and the API sees only key hashes."""
	keys = {name: secrets.token_hex(32) for name in sorted({r.account for r in requests})}
	path = output / "credentials.json"
	with open(path, "x", opener=lambda p, f: os.open(p, f, 0o600)) as stream:
		json.dump(keys, stream)
	digests = {hashlib.sha256(key.encode()).hexdigest(): name for name, key in keys.items()}
	(output / "accounts.json").write_text(json.dumps(digests, indent="\t"))
	return keys


def probe_health(url: str) -> dict[str, Any] | None:
	try:
		response = httpx.get(f"{url}/healthz", timeout=2)
		body = response.json() if response.is_success else None
	except httpx.HTTPError, ValueError:
		return None
	return body if isinstance(body, dict) else None


def wait_ready(
	process: subprocess.Popen[bytes],
	url: str,
	policy: str,
	instance: str,
	probe: Probe,
	timeout: float,
	interval: float = 0.5,
) -> dict[str, Any]:
	"""The health report of the started process, once it serves the declared policy.

	Only the started process knows `instance`, so a service that took the port after the free-port
	check cannot pass as this cell's gateway.
	"""
	deadline = time.monotonic() + timeout
	while time.monotonic() < deadline:
		if process.poll() is not None:
			raise ValueError(f"gateway exited with status {process.returncode}")
		health = probe(url)
		if health is not None and health.get("status") == "ready":
			if health.get("instance_id") != instance:
				raise ValueError(f"another process is serving {url}")
			if health.get("policy") != policy:
				raise ValueError(f"gateway serves {health.get('policy')}, not {policy}")
			return health
		time.sleep(interval)
	raise ValueError("gateway did not become ready")


def _signal_group(process: subprocess.Popen[bytes], signum: int) -> None:
	with contextlib.suppress(ProcessLookupError):
		os.killpg(process.pid, signum)


def stop_gateway(process: subprocess.Popen[bytes], timeout: float = 30) -> None:
	"""Stop the launched process and anything it started, even after the launcher has exited."""
	_signal_group(process, signal.SIGTERM)
	with contextlib.suppress(subprocess.TimeoutExpired):
		process.wait(timeout=timeout)
	_signal_group(process, signal.SIGKILL)
	process.wait()


def _measure(
	output: Path,
	plan: dict[str, Any],
	gateway: dict[str, Any],
	requests: list[BenignRequest],
	url: str,
	keys: dict[str, str],
) -> None:
	caps = gateway["max_output_tokens"]
	references = {
		key: value
		for domain in caps
		for key, value in task_references(
			task_directory(domain, root=Path(gateway["tasks_dir"]))
		).items()
	}
	profiles = gateway.get("account_profiles")
	asyncio.run(
		measure_utility(
			requests=requests,
			references=references,
			url=url,
			keys=keys,
			max_output_tokens=caps,
			account_profiles=json.loads(Path(profiles).read_text()) if profiles else None,
			output=output / "utility.json",
			baseline=plan["binding"],
			concurrency=plan["concurrency"],
			request_deadline=plan["request_deadline"],
			attribute_restrictions=True,
		)
	)


def run_setting(
	setting: dict[str, Any],
	state: Path,
	output: Path,
	plan: dict[str, Any],
	*,
	python: tuple[str, ...] = (sys.executable, "-m", "harness_api.launcher"),
	probe: Probe = probe_health,
	measure: Callable[..., None] = _measure,
) -> dict[str, Any]:
	"""One cell: start a gateway on the fresh state, measure the frozen schedule, always stop."""
	gateway = plan.get("gateway")
	if not isinstance(gateway, dict):
		raise ValueError("the grid spec needs a gateway declaration for this runner")
	state, output = state.resolve(), output.resolve()
	url = f"http://127.0.0.1:{gateway['port']}"
	# A cell must never measure a gateway that was already serving, such as a running study.
	if probe(url) is not None:
		raise ValueError(f"port {gateway['port']} is already serving; declare a free port")
	schedule = Path(plan["root"]) / plan["schedule"]
	requests = [
		BenignRequest.model_validate_json(line)
		for line in schedule.read_text().splitlines()
		if line.strip()
	]
	keys = write_credentials(output, requests)
	instance = secrets.token_hex(16)
	environment = gateway_environment(setting, gateway, state, output, dict(os.environ))
	environment["HARNESS_INSTANCE_ID"] = instance
	with (output / "harness.log").open("ab") as log:
		# The cell's directory holds no dotenv, so a product .env cannot alter this process.
		process = subprocess.Popen(
			[*python, *gateway_command(setting, gateway, state)],
			env=environment,
			cwd=output,
			stdout=log,
			stderr=subprocess.STDOUT,
			start_new_session=True,
		)
		try:
			health = wait_ready(
				process,
				url,
				environment["HARNESS_POLICY"],
				instance,
				probe,
				float(gateway.get("startup_timeout", 600)),
			)
			measure(output, plan, gateway, requests, url, keys)
			(output / "gateway.json").write_text(json.dumps(health, indent="\t", sort_keys=True))
			return health
		finally:
			stop_gateway(process)


def gateway_runner(
	setting: dict[str, Any], state: Path, output: Path, plan: dict[str, Any]
) -> None:
	run_setting(setting, state, output, plan)
