import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from distillation.benchmarks.traffic import TrafficConfig, TrafficTarget, freeze_traffic
from distillation.benchmarks.traffic_reporting import summarize_traffic
from distillation.benchmarks.traffic_runner import run_traffic, solve_challenge
from distillation.schemas import (
	Message,
	TeacherGeneration,
	TeacherGenerationRequest,
	TokenUsage,
)
from harness.calibration import BenignRequest, Calibration, HarnessConfig, calibrate
from harness.mmd import Array, WindowScorer
from harness_api.app import create_app
from harness_api.settings import Settings

KEY = "00112233445566778899aabbccddeeff"


class Encoder:
	def encode(self, texts: list[str]) -> Array:
		rows = [
			np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).normal(
				size=4
			)
			for t in texts
		]
		return np.array([row / np.linalg.norm(row) for row in rows])

	def identity(self) -> dict[str, Any]:
		return {"model": "fake"}


class Teacher:
	def __init__(self) -> None:
		self.calls = 0

	async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
		self.calls += 1
		return TeacherGeneration(
			id=f"chatcmpl-{self.calls:08x}",
			provider="fake",
			model="fake-model",
			generated_at=datetime.now(UTC),
			message=Message(role="assistant", content="World"),
			finish_reason="stop",
			usage=TokenUsage(input_tokens=5, output_tokens=1),
			endpoint="fake",
		)

	async def close(self) -> None:
		pass


@pytest.fixture(scope="module")
def artifacts() -> tuple[Calibration, WindowScorer, WindowScorer]:
	requests = [BenignRequest(account=f"a{i // 20}", text=f"q {i}") for i in range(400)]
	calibration, references = calibrate(
		config=HarnessConfig(
			window_size=2,
			reference_windows=2,
			bandwidth_sample=20,
			aggregate_window_size=10,
			aggregate_reference_windows=2,
		),
		encoder=Encoder(),
		fitting=requests[:100],
		score_calibration=requests[100:200],
		policy_calibration=requests[200:],
	)
	assert calibration.aggregate is not None and calibration._aggregate_references is not None
	aggregate = WindowScorer(calibration._aggregate_references, calibration.sigma)
	return calibration, WindowScorer(references, calibration.sigma), aggregate


def _app(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer],
	tmp_path: Path,
	levels: list[float],
	**settings: Any,
) -> tuple[Any, Teacher, Path]:
	calibration, scorer, aggregate = artifacts
	forced = calibration.model_copy(deep=True)
	forced.thresholds["paper"].thresholds = levels
	forced.config.task = settings.get("HARNESS_TASK", "text")
	log = tmp_path / f"decisions-{len(list(tmp_path.iterdir()))}.jsonl"
	teacher = Teacher()
	app = create_app(
		settings=Settings(
			_env_file=None,  # type: ignore[call-arg]
			HARNESS_TRUST_ACTOR_HEADER=True,
			HARNESS_POW_DIFFICULTY=4,
			HARNESS_MIN_INTERVAL_SECONDS=0,
			HARNESS_DECISION_LOG=log,
			HARNESS_DECEPTION_KEY_FILE=tmp_path / "deception.key",
			**settings,
		),
		teacher=teacher,
		encoder=Encoder(),
		calibration=forced,
		scorer=scorer,
		aggregate_scorer=aggregate,
	)
	return app, teacher, log


def _request(text: str) -> dict[str, Any]:
	return TeacherGenerationRequest(
		messages=[Message(role="user", content=text)], max_output_tokens=16
	).model_dump(mode="json")


async def _solved(client: httpx.AsyncClient, actor: str, text: str) -> httpx.Response:
	headers = {"X-Benchmark-Actor": actor}
	first = await client.post("/v1/teacher/generate", json=_request(text), headers=headers)
	assert first.status_code == 428, first.text
	challenge = first.json()["challenge"]
	counter = solve_challenge(challenge["prefix"], challenge["difficulty"])
	proof = {"X-Harness-Proof": f"{challenge['nonce']}:{counter}"}
	return await client.post("/v1/teacher/generate", json=_request(text), headers=headers | proof)


@pytest.mark.asyncio
async def test_protected_collector_solves_real_api_challenges_and_collects_noise(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	from distillation.campaign import ProtectedTeacher

	app, teacher, log = _app(
		artifacts,
		tmp_path,
		[-10, -10, -10, -10],
		HARNESS_TASK="ag_news",
		HARNESS_L4_ACTION="deceive",
	)
	gateway = ProtectedTeacher("http://harness", "", transport=httpx.ASGITransport(app=app))
	gateway.client.headers["X-Benchmark-Actor"] = "extractor"
	try:
		for _ in range(3):
			generation = await gateway.generate(
				TeacherGenerationRequest(
					messages=[Message(role="user", content="Classify this article")],
					max_output_tokens=16,
				)
			)
			assert generation.message.content
	finally:
		await gateway.close()
	assert gateway.challenges == 2
	assert teacher.calls == 1
	assert any(json.loads(line)["event"] == "noise" for line in log.read_text().splitlines())


@pytest.mark.asyncio
async def test_traffic_benchmark_runs_through_the_harness_with_one_sample_per_request(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	suite_path = tmp_path / "traffic.json"
	suite = freeze_traffic(suite_path, genuine_count=12, attack_count=12, duration_seconds=0.05)
	target = TrafficTarget(base_url="http://harness/", service_revision="test-harness")
	config = TrafficConfig(
		experiment="harness",
		suite_path=suite_path,
		output_dir=tmp_path / "results",
		model="fake-model",
		model_revision="fake-v1",
		targets={"before": target, "after": target},
	)
	cases: list[tuple[list[float], str, set[str]]] = [
		([-1.0, -1.0, -1.0, 1e9], "refuse", {"allow", "throttle"}),
		([-1.0] * 4, "deceive", {"allow", "throttle"}),
		([-1.0] * 4, "refuse", {"block"}),
	]
	for levels, action, expected in cases:
		app, _, _ = _app(artifacts, tmp_path, levels, HARNESS_L4_ACTION=action)
		run = await run_traffic(config, "after", transport=httpx.ASGITransport(app=app))
		assert len(run.samples) == len(suite.requests)
		assert {sample.request_id for sample in run.samples} == {r.id for r in suite.requests}
		decisions = {sample.decision for sample in run.samples}
		assert expected <= decisions | {"allow"}
		if "throttle" in expected:
			throttled = [s for s in run.samples if s.decision == "throttle"]
			assert throttled and all(s.succeeded and s.challenge_seconds for s in throttled)
		else:
			assert any(s.status == "blocked" for s in run.samples)


@pytest.mark.asyncio
async def test_challenges_bind_to_the_request_and_cannot_be_reused(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	app, teacher, log = _app(artifacts, tmp_path, [-1, -1, -1, 1e9])
	transport = httpx.ASGITransport(app=app)
	async with httpx.AsyncClient(transport=transport, base_url="http://harness") as client:
		path = "/v1/teacher/generate"
		assert (await client.post(path, json=_request("x"))).status_code == 401
		actor = {"X-Benchmark-Actor": "u1"}
		assert (await client.post(path, json=_request("q 1"), headers=actor)).status_code == 200
		challenged = await client.post(path, json=_request("q 2"), headers=actor)
		challenge = challenged.json()["challenge"]
		counter = solve_challenge(challenge["prefix"], challenge["difficulty"])
		proof = actor | {"X-Harness-Proof": f"{challenge['nonce']}:{counter}"}
		changed = _request("q 2") | {"max_output_tokens": 999}
		assert (await client.post(path, json=changed, headers=proof)).status_code == 403
		again = await client.post(path, json=_request("q 3"), headers=actor)
		retry = again.json()["challenge"]
		answer = solve_challenge(retry["prefix"], retry["difficulty"])
		proof = actor | {"X-Harness-Proof": f"{retry['nonce']}:{answer}"}
		served = await client.post(path, json=_request("q 3"), headers=proof)
		assert served.status_code == 200 and served.headers["x-harness-decision"] == "throttle"
		assert (await client.post(path, json=_request("q 3"), headers=proof)).status_code == 403
		await asyncio.sleep(0.1)
	assert teacher.calls == 2
	events = [json.loads(line) for line in log.read_text().splitlines()]
	assert "q 2" not in log.read_text()
	assert {event["event"] for event in events} >= {"decision", "redeem"}


@pytest.mark.asyncio
async def test_l4_noise_needs_no_teacher_and_looks_like_l3(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	l3, _, _ = _app(artifacts, tmp_path, [-1, -1, -1, 1e9], HARNESS_TASK="ag_news")
	l4, teacher, log = _app(
		artifacts, tmp_path, [-1] * 4, HARNESS_TASK="ag_news", HARNESS_L4_ACTION="deceive"
	)
	responses = []
	for app in (l3, l4):
		transport = httpx.ASGITransport(app=app)
		async with httpx.AsyncClient(transport=transport, base_url="http://harness") as client:
			actor = {"X-Benchmark-Actor": "u1"}
			await client.post("/v1/teacher/generate", json=_request("warm"), headers=actor)
			for i in range(12):
				responses.append(await _solved(client, "u1", f"article {i}"))
			again = await _solved(client, "u1", "article 0")
			await asyncio.sleep(0.1)
	truthful, noisy = responses[:12], responses[12:]
	assert teacher.calls == 1
	assert (tmp_path / "deception.key").exists()
	for honest, false in zip(truthful, noisy, strict=True):
		assert honest.status_code == false.status_code == 200
		assert set(honest.headers) == set(false.headers)
		assert honest.headers["x-harness-decision"] == false.headers["x-harness-decision"]
		first, second = honest.json(), false.json()
		assert set(first) == set(second) and first["usage"] == second["usage"]
		assert second["model"] == first["model"] and second["id"].startswith("chatcmpl-")
		assert len(second["id"]) == len(first["id"]) and second["finish_reason"] == "stop"
	labels = [response.json()["message"]["content"] for response in noisy]
	assert set(labels) <= {"World", "Sports", "Business", "Sci/Tech"} and len(set(labels)) > 1
	assert again.json()["message"]["content"] == labels[0]
	events = [json.loads(line) for line in log.read_text().splitlines()]
	assert sum(event["event"] == "noise" for event in events) == 13
	assert any(event["event"] == "aggregate" for event in events)


@pytest.mark.asyncio
async def test_gsm8k_noise_keeps_the_answer_format_within_the_limit(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	app, teacher, _ = _app(
		artifacts,
		tmp_path,
		[-1] * 4,
		HARNESS_TASK="gsm8k",
		HARNESS_DECEPTION_KEY=KEY,
		HARNESS_L4_ACTION="deceive",
	)
	transport = httpx.ASGITransport(app=app)
	async with httpx.AsyncClient(transport=transport, base_url="http://harness") as client:
		actor = {"X-Benchmark-Actor": "u1"}
		await client.post("/v1/teacher/generate", json=_request("warm"), headers=actor)
		response = await _solved(client, "u1", "Tom has 3 boxes of 12 apples. How many apples?")
	content = response.json()["message"]["content"]
	assert teacher.calls == 1 and content.rsplit("\n", 1)[1].startswith("#### ")
	assert len(content.split()) <= 16


@pytest.mark.asyncio
async def test_traffic_benchmark_separates_noise_from_useful_responses(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	suite_path = tmp_path / "traffic.json"
	suite = freeze_traffic(suite_path, genuine_count=12, attack_count=12, duration_seconds=0.05)
	app, _, log = _app(
		artifacts, tmp_path, [-1.0] * 4, HARNESS_TASK="ag_news", HARNESS_L4_ACTION="deceive"
	)
	attached = TrafficTarget(base_url="http://harness/", service_revision="noise", harness_log=log)
	plain = TrafficTarget(base_url="http://harness/", service_revision="noise")
	results = []
	for target in (attached, plain):
		config = TrafficConfig(
			experiment="noise",
			suite_path=suite_path,
			output_dir=tmp_path / "results",
			model="fake-model",
			model_revision="fake-v1",
			targets={"before": plain, "after": target},
		)
		results.append(await run_traffic(config, "after", transport=httpx.ASGITransport(app=app)))
	with_log, without_log = results
	assert len(with_log.samples) == len(suite.requests)
	assert all(sample.noised is not None for sample in with_log.samples)
	assert all(sample.noised is None for sample in without_log.samples)
	summary = summarize_traffic(with_log.samples, 1.0)
	assert summary["noised"] > 0
	assert summary["useful_succeeded"] == summary["succeeded"] - summary["noised"]
	assert summarize_traffic(without_log.samples, 1.0)["noised"] is None
	noise_events = [json.loads(line) for line in log.read_text().splitlines() if '"noise"' in line]
	assert all(
		event["benchmark_request"].startswith(with_log.id)
		or event["benchmark_request"].startswith(without_log.id)
		for event in noise_events
	)
	missing = TrafficTarget(
		base_url="http://harness/", service_revision="noise", harness_log=tmp_path / "none.jsonl"
	)
	config = TrafficConfig(
		experiment="noise",
		suite_path=suite_path,
		output_dir=tmp_path / "results",
		model="fake-model",
		model_revision="fake-v1",
		targets={"before": plain, "after": missing},
	)
	with pytest.raises(ValueError, match="harness log not found"):
		await run_traffic(config, "after", transport=httpx.ASGITransport(app=app))
