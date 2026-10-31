import asyncio
import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from distillation.benchmarks.models import GenerationConfig
from distillation.benchmarks.traffic import (
	TrafficConfig,
	TrafficSample,
	TrafficTarget,
	freeze_traffic,
)
from distillation.benchmarks.traffic_reporting import summarize_traffic
from distillation.benchmarks.traffic_runner import run_traffic, solve_challenge
from distillation.schemas import (
	Message,
	TeacherGeneration,
	TeacherGenerationRequest,
	TokenUsage,
)
from distillation.teacher import TeacherUpstreamError
from harness.calibration import BenignRequest, Calibration, HarnessConfig, calibrate
from harness.mmd import Array, WindowScorer
from harness.policy import QuotaPolicy, RatePolicy
from harness_api.app import create_app
from harness_api.settings import Settings

KEY = "00112233445566778899aabbccddeeff"


def test_l4_defaults_and_unsupported_tasks() -> None:
	assert Settings(_env_file=None).l4_action == "deceive"  # type: ignore[call-arg]
	assert Settings(_env_file=None, HARNESS_TASK="coding").l4_action == "refuse"  # type: ignore[call-arg]
	with pytest.raises(ValueError, match="refusal only"):
		Settings(_env_file=None, HARNESS_TASK="coding", HARNESS_L4_ACTION="deceive")  # type: ignore[call-arg]


def test_partial_level_coverage_is_not_a_detection_rate() -> None:
	samples = [
		TrafficSample(
			request_id="a",
			scheduled_seconds=0,
			started_seconds=0,
			completed_seconds=0,
			status="ok",
			level=4,
		),
		TrafficSample(
			request_id="b",
			scheduled_seconds=0,
			started_seconds=0,
			completed_seconds=0,
			status="error",
		),
	]
	summary = summarize_traffic(samples, 1.0)
	assert summary["l4_requests"] == 1
	assert summary["level_coverage"] == 0.5
	assert summary["l4_percent"] is None


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
@pytest.mark.parametrize(("level", "status"), [(1, 200), (2, 200), (3, 428), (4, 428)])
async def test_verified_risk_actions_at_each_level(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer],
	tmp_path: Path,
	level: int,
	status: int,
) -> None:
	app, teacher, _ = _app(
		artifacts,
		tmp_path,
		[-1.0] * level + [1e10] * (4 - level),
		HARNESS_POLICY_VERSION="verify-risk-v2",
		HARNESS_L4_ACTION="refuse",
	)
	async with httpx.AsyncClient(
		transport=httpx.ASGITransport(app=app), base_url="http://test"
	) as client:
		headers = {"X-Benchmark-Actor": "alice"}
		assert (
			await client.post("/v1/teacher/generate", json=_request("one"), headers=headers)
		).status_code == 200
		response = await client.post("/v1/teacher/generate", json=_request("two"), headers=headers)
		assert response.status_code == status
		assert teacher.calls == (2 if status == 200 else 1)


@pytest.mark.asyncio
async def test_verified_risk_l4_recovers_one_request_and_respects_cap(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	app, teacher, _ = _app(
		artifacts,
		tmp_path,
		[-1.0] * 4,
		HARNESS_POLICY_VERSION="verify-risk-v2",
		HARNESS_L4_ACTION="refuse",
	)
	transport = httpx.ASGITransport(app=app)
	async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
		headers = {"X-Benchmark-Actor": "alice"}
		assert (
			await client.post("/v1/teacher/generate", json=_request("one"), headers=headers)
		).status_code == 200
		response = await client.post("/v1/teacher/generate", json=_request("two"), headers=headers)
		assert response.status_code == 428
		challenge = response.json()["challenge"]
		proof = (
			f"{challenge['nonce']}:{solve_challenge(challenge['prefix'], challenge['difficulty'])}"
		)
		assert (
			await client.post(
				"/v1/teacher/generate",
				json=_request("two"),
				headers={"X-Benchmark-Actor": "bob", "X-Harness-Proof": proof},
			)
		).status_code == 403
		# A mismatched redemption consumes the pending request, so obtain a fresh one.
		response = await client.post(
			"/v1/teacher/generate", json=_request("three"), headers=headers
		)
		challenge = response.json()["challenge"]
		proof = (
			f"{challenge['nonce']}:{solve_challenge(challenge['prefix'], challenge['difficulty'])}"
		)
		assert (
			await client.post(
				"/v1/teacher/generate",
				json=_request("three"),
				headers=headers | {"X-Harness-Proof": proof},
			)
		).status_code == 200
		assert (
			await client.post(
				"/v1/teacher/generate",
				json=_request("three"),
				headers=headers | {"X-Harness-Proof": proof},
			)
		).status_code == 403
		assert teacher.calls == 2


@pytest.mark.asyncio
async def test_authenticated_coding_endpoint_preserves_code_and_enforces_quota(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	pytest.importorskip("coding", reason="coding endpoint needs the coding extra")
	from coding.dataset import PROMPT
	from distillation.campaign import PolicyRefusal, ProtectedTeacher
	from harness.serialization import detector_text

	calibration, scorer, _ = artifacts
	calibration = calibration.model_copy(deep=True)
	calibration.config.task = "coding"
	calibration.config.aggregate_enabled = False
	calibration.baselines["quota"].thresholds = [1.0]
	accounts = tmp_path / "accounts.json"
	accounts.write_text(json.dumps({hashlib.sha256(KEY.encode()).hexdigest(): "coder"}))

	class CodingTeacher(Teacher):
		async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
			assert request.max_output_tokens == 512
			result = await super().generate(request)
			return result.model_copy(
				update={"message": Message(role="assistant", content="def f(): return 1")}
			)

	app = create_app(
		settings=Settings(
			_env_file=None,  # type: ignore[call-arg]
			HARNESS_TASK="coding",
			HARNESS_POLICY="quota",
			HARNESS_ACCOUNTS_FILE=accounts,
			HARNESS_DECISION_LOG=tmp_path / "decisions.jsonl",
		),
		teacher=CodingTeacher(),
		encoder=Encoder(),
		calibration=calibration,
		scorer=scorer,
	)
	text = PROMPT.format(specification="Return one.", signature="def f():")
	assert detector_text("coding", text) == "Return one."
	request = TeacherGenerationRequest(
		messages=[Message(role="user", content=text)], max_output_tokens=512
	)
	client = ProtectedTeacher("http://coding", KEY, transport=httpx.ASGITransport(app=app))
	try:
		assert (await client.generate(request)).message.content == "def f(): return 1"
		with pytest.raises(PolicyRefusal):
			await client.generate(request)
	finally:
		await client.close()


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
async def test_gateway_traffic_uses_news_template_and_domain(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	suite_path = tmp_path / "gateway-traffic.json"
	suite = freeze_traffic(
		suite_path, genuine_count=4, attack_count=4, duration_seconds=0.05, gateway_news=True
	)
	app, teacher, _ = _app(
		artifacts,
		tmp_path,
		[1e9] * 4,
		HARNESS_TASK="gateway",
		HARNESS_GATEWAY_CAPS={"ag_news": 256, "squad": 256},
	)
	config = TrafficConfig(
		experiment="gateway-news",
		suite_path=suite_path,
		output_dir=tmp_path / "results",
		model="fake-model",
		model_revision="fake-v1",
		generation=GenerationConfig(max_output_tokens=256),
		targets={
			"after": TrafficTarget(
				base_url="http://harness", service_revision="test", gateway_domain="ag_news"
			)
		},
	)
	run = await run_traffic(config, "after", transport=httpx.ASGITransport(app=app))
	assert len(run.samples) == len(suite.requests) == teacher.calls
	assert all(sample.succeeded and sample.http_status == 200 for sample in run.samples)


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
	l3, _, _ = _app(
		artifacts, tmp_path, [-1, -1, -1, 1e9], HARNESS_TASK="ag_news", HARNESS_L4_ACTION="deceive"
	)
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
	assert all(sample.level is not None for sample in with_log.samples)
	assert all(sample.level is None for sample in without_log.samples)
	summary = summarize_traffic(with_log.samples, 1.0)
	assert summary["noised"] > 0
	assert summary["l4_requests"] > 0 and summary["level_coverage"] == 1
	assert summarize_traffic(without_log.samples, 1.0)["l4_percent"] is None
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


@pytest.mark.asyncio
async def test_gateway_campaign_shares_quota_across_domains_and_budget_across_accounts(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	from distillation.collection import CollectionOptions
	from distillation.gateway_campaign import collect_gateway
	from distillation.io import read_jsonl, write_jsonl
	from distillation.schemas import GatewayRequest, LedgerRecord, PromptRecord, PromptSource
	from distillation.tasks import AG_NEWS_PROMPT, GSM8K_EVAL_SUFFIX

	calibration, scorer, aggregate = artifacts
	calibration = calibration.model_copy(deep=True)
	calibration.config.horizon_attempts = 2
	calibration.baselines["quota"].thresholds = [1.0]
	app, teacher, decisions = _app(
		(calibration, scorer, aggregate),
		tmp_path,
		[1e9] * 4,
		HARNESS_TASK="gateway",
		HARNESS_GATEWAY_CAPS={"ag_news": 16, "gsm8k": 256},
		HARNESS_POLICY="quota",
	)

	# Real credential resolution, independent of test actor headers.
	# _app uses actor headers; wrap only auth routing for this fixture.
	class Transport(httpx.ASGITransport):
		async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
			request.headers["X-Benchmark-Actor"] = request.headers["Authorization"].removeprefix(
				"Bearer "
			)
			return await super().handle_async_request(request)

	rows = [
		GatewayRequest(
			request_id=f"q{i}",
			account="shared" if i < 4 else "other",
			time=0,
			max_output_tokens=16 if i % 2 == 0 else 256,
			prompt=PromptRecord(
				example_id=f"p{i}",
				task="ag_news" if i % 2 == 0 else "gsm8k",
				split="train",
				source=PromptSource(
					corpus="fixture", source_id=str(i), group_id=str(i), license="fixture"
				),
				prompt=[
					Message(
						role="user",
						content=AG_NEWS_PROMPT.format(text=f"story {i}")
						if i % 2 == 0
						else f"problem {i}{GSM8K_EVAL_SUFFIX}",
					)
				],
			),
		)
		for i in range(8)
	]
	schedule = tmp_path / "requests.jsonl"
	write_jsonl(schedule, rows)
	output = tmp_path / "campaign"
	kwargs: dict[str, Any] = dict(
		schedule=schedule,
		output=output,
		url="http://gateway",
		keys={"shared": "shared", "other": "other"},
		options=CollectionOptions(terms_review_id="fixture", concurrency=1, enable_thinking=False),
		attempt_budget=5,
		transport=Transport(app=app),
		decision_log=decisions,
	)
	summary = await collect_gateway(**kwargs)
	assert summary["queries"] == 5
	assert summary["blocked"] >= 1
	assert summary["received"] == teacher.calls
	events = [
		json.loads(line) for line in (output / "campaign.events.jsonl").read_text().splitlines()
	]
	attempts = [e for e in events if e["event"] == "attempt"]
	assert [e["index"] for e in attempts] == list(range(1, 6))
	assert {e["domain"] for e in attempts} == {"ag_news", "gsm8k"}
	assert {e["account"] for e in attempts} == {"shared", "other"}
	assert len(read_jsonl(output / "ledger.jsonl", LedgerRecord)) == summary["received"]
	assert all(e["contract"] == summary["contract"] for e in attempts)
	assert await collect_gateway(**kwargs) == summary
	assert teacher.calls == summary["received"]
	with pytest.raises(ValueError, match="changed"):
		await collect_gateway(**(kwargs | {"attempt_budget": 6}))
	(output / "campaign.json").unlink()
	with pytest.raises(ValueError, match="interrupted"):
		await collect_gateway(**kwargs)


@pytest.mark.asyncio
async def test_gateway_challenge_binding_and_proofs_share_the_campaign_budget(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	from distillation.collection import CollectionOptions
	from distillation.gateway_campaign import collect_gateway
	from distillation.io import write_jsonl
	from distillation.schemas import GatewayRequest, PromptRecord, PromptSource
	from distillation.tasks import AG_NEWS_PROMPT

	app, teacher, decisions = _app(
		artifacts,
		tmp_path,
		[-1, -1, -1, 1e9],
		HARNESS_TASK="gateway",
		HARNESS_GATEWAY_CAPS={"ag_news": 16, "gsm8k": 256},
	)

	class Transport(httpx.ASGITransport):
		async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
			request.headers["X-Benchmark-Actor"] = "one"
			return await super().handle_async_request(request)

	rows = [
		GatewayRequest(
			request_id=f"q{i}",
			account="one",
			time=0,
			max_output_tokens=16,
			prompt=PromptRecord(
				example_id=f"p{i}",
				task="ag_news",
				split="train",
				source=PromptSource(
					corpus="test", source_id=str(i), group_id=str(i), license="test"
				),
				prompt=[Message(role="user", content=AG_NEWS_PROMPT.format(text=f"story{i}"))],
			),
		)
		for i in range(5)
	]
	schedule = tmp_path / "requests.jsonl"
	write_jsonl(schedule, rows)
	summary = await collect_gateway(
		schedule=schedule,
		output=tmp_path / "campaign",
		url="http://gateway",
		keys={"one": "key"},
		options=CollectionOptions(terms_review_id="fixture"),
		attempt_budget=3,
		decision_log=decisions,
		transport=Transport(app=app),
	)
	assert summary["queries"] == 3
	assert summary["proof_requests"] == 1
	assert summary["received"] == teacher.calls == 2
	assert summary["hashes"] >= 1
	async with httpx.AsyncClient(transport=Transport(app=app), base_url="http://gateway") as client:
		invalid = await client.post(
			"/v1/teacher/generate", json=_request("bad"), headers={"X-Gateway-Domain": "gsm8k"}
		)
		assert invalid.status_code == 422
		malformed = await client.post(
			"/v1/teacher/generate",
			json=_request("unmatched free text"),
			headers={"X-Gateway-Domain": "ag_news"},
		)
		assert malformed.status_code == 422
		first = await client.post(
			"/v1/teacher/generate",
			json=_request(AG_NEWS_PROMPT.format(text="binding")),
			headers={"X-Gateway-Domain": "ag_news"},
		)
		assert first.status_code == 428
		challenge = first.json()["challenge"]
		proof = solve_challenge(challenge["prefix"], challenge["difficulty"])
		changed = _request(
			"binding\nSolve step by step. End with #### followed by the final numeric answer."
		) | {"max_output_tokens": 256}
		second = await client.post(
			"/v1/teacher/generate",
			json=changed,
			headers={
				"X-Gateway-Domain": "gsm8k",
				"X-Harness-Proof": f"{challenge['nonce']}:{proof}",
			},
		)
		assert second.status_code == 403


@pytest.mark.parametrize("policy", ["quota", "mmd"])
async def test_http_refusal_spam_cannot_renew_an_account(
	policy: str, artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	calibration, scorer, aggregate = artifacts
	calibration = calibration.model_copy(deep=True)
	calibration.config.horizon_attempts = 3
	calibration.config.horizon_seconds = 100
	calibration.baselines["quota"].thresholds = [1]
	app, teacher, log = _app(
		(calibration, scorer, aggregate),
		tmp_path,
		[1e9] * 4,
		HARNESS_POLICY=policy,
		HARNESS_TEST_CLOCK=True,
	)
	headers = {"X-Benchmark-Actor": "same-account", "X-Harness-Time": "10"}
	async with httpx.AsyncClient(
		transport=httpx.ASGITransport(app=app), base_url="http://fixture"
	) as client:
		responses = [
			await client.post("/v1/teacher/generate", json=_request("valid query"), headers=headers)
			for _ in range(10)
		]
		allowed = 1 if policy == "quota" else 3
		assert [r.status_code for r in responses] == [200] * allowed + [403] * (10 - allowed)
		assert teacher.calls == allowed
		headers["X-Harness-Time"] = "109.999"
		assert (
			await client.post(
				"/v1/teacher/generate", json=_request("still blocked"), headers=headers
			)
		).status_code == 403
		headers["X-Harness-Time"] = "110"
		assert (
			await client.post("/v1/teacher/generate", json=_request("time reset"), headers=headers)
		).status_code == 200
		assert teacher.calls == allowed + 1
	decisions = [
		json.loads(line)
		for line in log.read_text().splitlines()
		if json.loads(line)["event"] == "decision"
	]
	assert [e["attempt"] for e in decisions] == list(range(1, 12)) + [1]


def _events(log: Path, kind: str) -> list[dict[str, Any]]:
	events = [json.loads(line) for line in log.read_text().splitlines()]
	return [event for event in events if event["event"] == kind]


@pytest.mark.asyncio
async def test_restart_keeps_account_state_and_redeems_a_proof_once(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	state = tmp_path / "state.sqlite3"
	path = "/v1/teacher/generate"
	actor = {"X-Benchmark-Actor": "u1"}
	before, _, before_log = _app(artifacts, tmp_path, [-1, -1, -1, 1e9], HARNESS_STATE_DB=state)
	async with httpx.AsyncClient(
		transport=httpx.ASGITransport(app=before), base_url="http://harness"
	) as client:
		assert (await client.post(path, json=_request("q 1"), headers=actor)).status_code == 200
		challenged = await client.post(path, json=_request("q 2"), headers=actor)
	assert challenged.status_code == 428
	challenge = challenged.json()["challenge"]
	proof = actor | {
		"X-Harness-Proof": f"{challenge['nonce']}:"
		f"{solve_challenge(challenge['prefix'], challenge['difficulty'])}"
	}
	# A new process on the same state file: the challenge and the account's level survive.
	after, teacher, after_log = _app(artifacts, tmp_path, [-1, -1, -1, 1e9], HARNESS_STATE_DB=state)
	async with httpx.AsyncClient(
		transport=httpx.ASGITransport(app=after), base_url="http://harness"
	) as client:
		served = await client.post(path, json=_request("q 2"), headers=proof)
		assert served.status_code == 200 and served.headers["x-harness-decision"] == "throttle"
		assert (await client.post(path, json=_request("q 2"), headers=proof)).status_code == 403
		assert (await client.post(path, json=_request("q 3"), headers=actor)).status_code == 428
	assert teacher.calls == 1
	assert [e["attempt"] for e in _events(before_log, "decision")] == [1, 2]
	assert [e["attempt"] for e in _events(after_log, "decision")] == [3]
	assert [(e["attempt"], e["status"]) for e in _events(before_log, "release")] == [
		(1, "released")
	]
	assert [(e["attempt"], e["status"]) for e in _events(after_log, "release")] == [(2, "released")]
	assert [e["solved"] for e in _events(after_log, "redeem")] == [True, False]
	with pytest.raises(ValueError, match="another policy or calibration"):
		_app(artifacts, tmp_path, [-1] * 4, HARNESS_STATE_DB=state)
	# Legacy state cannot be resumed under the recoverable verification policy.
	with pytest.raises(ValueError, match="another policy or calibration"):
		_app(
			artifacts,
			tmp_path,
			[-1, -1, -1, 1e9],
			HARNESS_STATE_DB=state,
			HARNESS_POLICY_VERSION="verify-risk-v2",
		)


@pytest.mark.asyncio
async def test_concurrent_near_limit_admissions_release_exactly_the_quota(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	calibration, scorer, aggregate = artifacts
	calibration = calibration.model_copy(deep=True)
	calibration.baselines["quota"].thresholds = [3]
	app, teacher, log = _app(
		(calibration, scorer, aggregate),
		tmp_path,
		[1e9] * 4,
		HARNESS_POLICY="quota",
		HARNESS_STATE_DB=tmp_path / "state.sqlite3",
	)
	async with httpx.AsyncClient(
		transport=httpx.ASGITransport(app=app), base_url="http://harness"
	) as client:
		responses = await asyncio.gather(
			*(
				client.post(
					"/v1/teacher/generate",
					json=_request(f"q {i}"),
					headers={"X-Benchmark-Actor": "u1"},
				)
				for i in range(12)
			)
		)
	assert sorted(r.status_code for r in responses) == [200] * 3 + [403] * 9
	assert teacher.calls == 3
	assert sorted(e["attempt"] for e in _events(log, "decision")) == list(range(1, 13))
	releases = _events(log, "release")
	assert sorted(e["attempt"] for e in releases) == [1, 2, 3]
	assert all(e["status"] == "released" and e["output_tokens"] == 1 for e in releases)


@pytest.mark.asyncio
async def test_cancelled_and_failed_releases_settle_without_refunding_the_admission(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	calibration, scorer, aggregate = artifacts
	calibration = calibration.model_copy(deep=True)
	calibration.baselines["quota"].thresholds = [3]
	state = tmp_path / "state.sqlite3"
	app, teacher, log = _app(
		(calibration, scorer, aggregate),
		tmp_path,
		[1e9] * 4,
		HARNESS_POLICY="quota",
		HARNESS_STATE_DB=state,
	)
	entered = asyncio.Event()
	mode = "hang"
	generate = teacher.generate

	async def controlled(request: TeacherGenerationRequest) -> TeacherGeneration:
		if mode == "hang":
			entered.set()
			await asyncio.Event().wait()
		if mode == "fail":
			raise TeacherUpstreamError("upstream down")
		return await generate(request)

	teacher.generate = controlled  # type: ignore[method-assign]
	headers = {"X-Benchmark-Actor": "u1"}
	async with httpx.AsyncClient(
		transport=httpx.ASGITransport(app=app), base_url="http://harness"
	) as client:

		async def post(text: str) -> httpx.Response:
			return await client.post("/v1/teacher/generate", json=_request(text), headers=headers)

		# The client goes away while the teacher is generating.
		pending = asyncio.create_task(post("q 1"))
		await entered.wait()
		pending.cancel()
		with pytest.raises(asyncio.CancelledError):
			await pending
		mode = "fail"
		assert (await post("q 2")).status_code == 502
		mode = "serve"
		assert (await post("q 3")).status_code == 200
		# Neither the cancelled nor the failed attempt was refunded.
		assert (await post("q 4")).status_code == 403
	assert [(e["attempt"], e["status"]) for e in _events(log, "release")] == [
		(1, "cancelled"),
		(2, "failed"),
		(3, "released"),
	]
	with sqlite3.connect(state) as db:
		rows = db.execute("SELECT attempt, status FROM reservations ORDER BY attempt").fetchall()
	assert rows == [(1, "cancelled"), (2, "failed"), (3, "released")]


def _budget_app(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path, **limits: Any
) -> tuple[Any, Teacher, Path]:
	return _app(
		artifacts,
		tmp_path,
		[1e9] * 4,
		HARNESS_POLICY="allow",
		HARNESS_TEST_CLOCK=True,
		HARNESS_STATE_DB=tmp_path / "state.sqlite3",
		**limits,
	)


def _client(app: Any) -> httpx.AsyncClient:
	return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://harness")


async def _ask(client: httpx.AsyncClient, text: str, **headers: str) -> httpx.Response:
	return await client.post(
		"/v1/teacher/generate",
		json=_request(text),
		headers={"X-Benchmark-Actor": "u1", "X-Harness-Time": "10", **headers},
	)


def test_budget_limits_need_a_durable_store() -> None:
	with pytest.raises(ValueError, match="HARNESS_STATE_DB"):
		Settings(_env_file=None, HARNESS_BUDGET_REQUESTS=1)  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_spent_budget_refuses_before_embedding_and_survives_a_restart(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer],
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	encoded: list[str] = []
	original = Encoder.encode

	def counted(self: Encoder, texts: list[str]) -> Array:
		encoded.extend(texts)
		return original(self, texts)

	monkeypatch.setattr(Encoder, "encode", counted)
	app, teacher, log = _budget_app(artifacts, tmp_path, HARNESS_BUDGET_REQUESTS=2)
	async with _client(app) as client:
		responses = [await _ask(client, f"q {i}") for i in range(3)]
	assert [r.status_code for r in responses] == [200, 200, 403]
	assert responses[2].json()["detail"]["code"] == "budget_exhausted"
	assert responses[2].headers["x-harness-decision"] == "block"
	assert teacher.calls == 2 and len(encoded) == 2

	releases = _events(log, "release")
	assert [(e["status"], e["output_tokens"]) for e in releases] == [("released", 1)] * 2
	decisions = _events(log, "decision")
	assert [e["reason"] for e in decisions] == [None, None, "budget_requests"]
	assert decisions[2]["level"] == 4 and decisions[2]["attempt"] is None
	revisions = {e["policy_revision"] for e in decisions + releases}
	assert revisions == {_events(log, "startup")[0]["policy_revision"]}

	restarted, teacher, _ = _budget_app(artifacts, tmp_path, HARNESS_BUDGET_REQUESTS=2)
	async with _client(restarted) as client:
		assert (await _ask(client, "after restart")).status_code == 403
	assert teacher.calls == 0


@pytest.mark.asyncio
async def test_teacher_failure_counts_as_an_attempt_but_uses_no_budget(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	app, teacher, log = _budget_app(artifacts, tmp_path, HARNESS_BUDGET_REQUESTS=1)
	working = teacher.generate

	async def failing(request: TeacherGenerationRequest) -> TeacherGeneration:
		raise TeacherUpstreamError("down")

	async with _client(app) as client:
		teacher.generate = failing  # type: ignore[method-assign]
		assert (await _ask(client, "q")).status_code == 502
		teacher.generate = working  # type: ignore[method-assign]
		assert (await _ask(client, "q")).status_code == 200
		assert (await _ask(client, "q")).status_code == 403
	assert [(e["attempt"], e["status"]) for e in _events(log, "release")] == [
		(1, "failed"),
		(2, "released"),
	]


@pytest.mark.asyncio
async def test_concurrency_limit_throttles_and_a_key_cannot_regenerate(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer], tmp_path: Path
) -> None:
	app, teacher, _ = _budget_app(artifacts, tmp_path, HARNESS_BUDGET_CONCURRENCY=1)
	working = teacher.generate
	entered, finish = asyncio.Event(), asyncio.Event()

	async def held(request: TeacherGenerationRequest) -> TeacherGeneration:
		entered.set()
		await finish.wait()
		return await working(request)

	teacher.generate = held  # type: ignore[method-assign]
	async with _client(app) as client:
		first = asyncio.create_task(_ask(client, "q", **{"X-Idempotency-Key": "k"}))
		await entered.wait()
		second = await _ask(client, "other")
		assert second.status_code == 429
		assert second.headers["x-harness-decision"] == "throttle"
		finish.set()
		assert (await first).status_code == 200
		retry = await _ask(client, "q", **{"X-Idempotency-Key": "k"})
		assert retry.status_code == 409 and retry.json()["detail"]["code"] == "duplicate_request"
		conflict = await _ask(client, "changed", **{"X-Idempotency-Key": "k"})
		assert conflict.json()["detail"]["code"] == "key_conflict"
	assert teacher.calls == 1


def test_every_restriction_names_its_cause() -> None:
	assert QuotaPolicy(1, 100, 10).admit("a", "q", 0).reason is None
	quota = QuotaPolicy(1, 100, 10)
	quota.admit("a", "q", 0)
	assert quota.admit("a", "q", 1).reason == "quota"
	capped = QuotaPolicy(100, 100, 1)
	capped.admit("a", "q", 0)
	assert capped.admit("a", "q", 1).reason == "attempt_cap"
	rate = RatePolicy(1, 100, 10)
	rate.admit("a", "q", 0)
	assert rate.admit("a", "q", 1).reason == "rate"
