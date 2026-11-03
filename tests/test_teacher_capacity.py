import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from distillation.schemas import (
	GatewayRequest,
	Message,
	PromptRecord,
	PromptSource,
	TeacherGeneration,
)
from distillation.teacher_capacity import (
	CapacitySLO,
	bind_utility_baseline,
	preflight,
	require_capacity_pass,
)


def schedule() -> list[GatewayRequest]:
	return [
		GatewayRequest(
			request_id=str(i),
			account="benign",
			time=i * 0.01,
			max_output_tokens=8,
			prompt=PromptRecord(
				example_id=str(i),
				split="test",
				task="squad",
				source=PromptSource(corpus="fixture", source_id=str(i), **{"license": "test"}),
				prompt=[Message(role="user", content="hello")],
			),
		)
		for i in range(8)
	]


@pytest.mark.asyncio
async def test_capacity_healthy_and_overloaded() -> None:
	async def teacher(request: httpx.Request) -> httpx.Response:
		if request.url.host == "slow":
			await asyncio.sleep(0.04)
		return httpx.Response(
			200,
			json=TeacherGeneration(
				id="one",
				provider="fake",
				model="fake",
				generated_at=datetime.now(UTC),
				message=Message(role="assistant", content="ok"),
				endpoint="fake",
			).model_dump(mode="json"),
		)

	slo = CapacitySLO(50, 30, 0.1, 0, 0, 2, 2)
	kwargs: dict[str, Any] = dict(
		rows=schedule(),
		token="",
		schedule_sha256="fixture",
		slo=slo,
		concurrency=1,
		timeout_seconds=1,
		transport=httpx.MockTransport(teacher),
	)
	healthy = await preflight(url="http://fast", **kwargs)
	overloaded = await preflight(url="http://slow", **kwargs)
	assert healthy["passed"]
	assert not overloaded["passed"]
	assert overloaded["queued_at_end"] > healthy["queued_at_end"]
	require_capacity_pass(healthy, schedule_sha256="fixture")
	with pytest.raises(ValueError):
		require_capacity_pass(overloaded, schedule_sha256="fixture")
	with pytest.raises(ValueError):
		require_capacity_pass(healthy, schedule_sha256="other")


def _evidence(path: Path, **changes: Any) -> Path:
	checks = dict.fromkeys(
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
	)
	path.write_text(
		json.dumps(
			{
				"version": "teacher-capacity-v1",
				"passed": True,
				"schedule_sha256": "s",
				"slo": {},
				"checks": checks,
				"offered_rps": 10.0,
				"completed_rps": 9.9,
				"concurrency": 8,
				"timeout_seconds": 120,
				**changes,
			}
		)
	)
	return path


def test_utility_collection_binds_to_a_passing_preflight_at_the_declared_rate(
	tmp_path: Path,
) -> None:
	declared: dict[str, Any] = {
		"schedule_sha256": "s",
		"declared_offered_rps": 10.0,
		"min_completion_rate": 0.95,
		"concurrency": 8,
		"request_deadline": 120.0,
	}
	evidence = _evidence(tmp_path / "ok.json")
	binding = bind_utility_baseline(evidence, **declared)
	assert binding["preflight_sha256"] == hashlib.sha256(evidence.read_bytes()).hexdigest()
	assert binding["min_completion_rate"] == 0.95
	for match, path, changes in (
		("needs a teacher preflight", tmp_path / "absent.json", {}),
		("passing capacity", _evidence(tmp_path / "failed.json", passed=False), {}),
		("passing capacity", evidence, {"schedule_sha256": "other"}),
		("below the declared", evidence, {"declared_offered_rps": 20.0}),
		("too few", _evidence(tmp_path / "slow.json", completed_rps=3.0), {}),
		("concurrency", evidence, {"concurrency": 4}),
		("deadline", evidence, {"request_deadline": 60.0}),
		("minimum completion", evidence, {"min_completion_rate": 0.0}),
	):
		with pytest.raises(ValueError, match=match):
			bind_utility_baseline(path, **{**declared, **changes})


@pytest.mark.asyncio
async def test_measured_utility_stores_the_binding_and_refuses_other_arrivals(
	tmp_path: Path,
) -> None:
	from distillation.benchmarks.models import BenchmarkItem
	from distillation.utility import measure_utility
	from harness.calibration import BenignRequest

	async def teacher(request: httpx.Request) -> httpx.Response:
		return httpx.Response(
			200,
			json=TeacherGeneration(
				id="one",
				provider="fake",
				model="fake",
				generated_at=datetime.now(UTC),
				message=Message(role="assistant", content="#### 2"),
				finish_reason="stop",
				endpoint="fake",
			).model_dump(mode="json"),
		)

	requests = [
		BenignRequest(
			account="a",
			text="q",
			time=i * 0.01,
			example_id=str(i),
			source_group="g",
			client_type="c",
		)
		for i in range(2)
	]
	references = {
		str(i): BenchmarkItem(
			id=str(i),
			benchmark="gsm8k",
			category="math",
			question="q",
			messages=[Message(role="user", content="q")],
			metric="numeric_exact_match",
			answer="2",
		)
		for i in range(2)
	}
	output = tmp_path / "utility.json"
	binding = {"preflight_sha256": "abc", "min_completion_rate": 0.95, "preflight_offered_rps": 1.0}
	kwargs: dict[str, Any] = dict(
		requests=requests,
		references=references,
		url="http://teacher",
		keys={},
		max_output_tokens=8,
		output=output,
		solver_workers=1,
		concurrency=1,
		transport=httpx.MockTransport(teacher),
	)
	with pytest.raises(ValueError, match="teacher preflight offered"):
		await measure_utility(baseline=binding, **kwargs)
	assert not output.exists()
	binding["preflight_offered_rps"] = 200.0
	result = await measure_utility(baseline=binding, **kwargs)
	assert result["baseline"]["preflight_sha256"] == "abc"
	assert json.loads(output.read_text())["baseline"] == binding
	assert result["completion_rate"] == 1
	assert result["measurement_version"] == "utility-response-latency-v3"
	# No binding keeps the original contract, so existing unbound measurements can resume.
	unbound = await measure_utility(**{**kwargs, "output": tmp_path / "unbound.json"})
	assert unbound["measurement_version"] == "utility-response-latency-v2"
	assert "baseline" not in unbound
	legacy_payload = {
		"measurement_version": "utility-response-latency-v2",
		"horizon_policy": unbound["horizon_policy"],
		"solver": unbound["solver"],
		"requests": [r.model_dump() for r in requests],
		"references": {k: v.model_dump(mode="json") for k, v in references.items()},
		"cap": 8,
		"account_profiles": None,
		"concurrency": 1,
		"deadline": 120,
		"url": "http://teacher",
	}
	assert (
		unbound["contract"]
		== hashlib.sha256(json.dumps(legacy_payload, sort_keys=True).encode()).hexdigest()
	)
	assert await measure_utility(**{**kwargs, "output": tmp_path / "unbound.json"}) == unbound
