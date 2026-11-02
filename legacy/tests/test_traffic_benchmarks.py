"""Replay the HTTP contract with fixtures, keeping detector outcomes distinct."""

import json
from collections import Counter
from pathlib import Path
from typing import Literal

import httpx
import pytest
from distillation.benchmarks.dashboard import create_app
from distillation.benchmarks.models import save_json
from distillation.benchmarks.traffic import (
	TrafficConfig,
	TrafficSample,
	TrafficTarget,
	freeze_traffic,
	load_traffic_config,
)
from distillation.benchmarks.traffic_demo import create_traffic_demo
from distillation.benchmarks.traffic_reporting import (
	_progression,
	build_traffic_report,
	discover_traffic,
	summarize_traffic,
	traffic_experiments,
)
from distillation.benchmarks.traffic_runner import _completion, run_traffic


def test_traffic_is_frozen_reproducible_and_contains_both_classes(tmp_path: Path) -> None:
	first = freeze_traffic(tmp_path / "one.json")
	second = freeze_traffic(tmp_path / "two.json")
	assert first.sha256 == second.sha256
	assert Counter(a.label for a in first.actors) == {"genuine": 4, "distillation": 2}
	assert len(first.requests) == 192
	assert {r.category for r in first.requests} >= {"coding", "batch_work", "learning"}
	assert all("distillation" not in a.id and "genuine" not in a.id for a in first.actors)
	with pytest.raises(FileExistsError):
		freeze_traffic(tmp_path / "one.json")
	first.requests[0].at_seconds += 1
	with pytest.raises(ValueError, match="hash mismatch"):
		first.verify()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["teacher_api", "openai"])
async def test_replay_counts_false_positives_throttles_and_errors_separately(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: Literal["teacher_api", "openai"]
) -> None:
	suite_path = tmp_path / "traffic.json"
	suite = freeze_traffic(suite_path, genuine_count=12, attack_count=12, duration_seconds=0.06)
	labels = {a.id: a.label for a in suite.actors}
	genuine = [a.id for a in suite.actors if a.label == "genuine"]
	monkeypatch.setenv("TRAFFIC_TEST_TOKEN", "secret-not-in-artifacts")
	config = TrafficConfig(
		experiment="fixture",
		suite_path=suite_path,
		output_dir=tmp_path / "results",
		model="fixture-model",
		model_revision="fixture-v1",
		window_seconds=0.02,
		targets={
			"before": TrafficTarget(
				backend=backend,
				base_url="http://test/before/",
				service_revision="no-harness",
				api_key_env="TRAFFIC_TEST_TOKEN",
			),
			"after": TrafficTarget(
				backend=backend,
				base_url="http://test/after/",
				service_revision="fixture-harness",
				api_key_env="TRAFFIC_TEST_TOKEN",
			),
		},
	)
	counts: Counter[tuple[str, str]] = Counter()
	seen: dict[str, list[tuple[str, object]]] = {"before": [], "after": []}

	async def endpoint(request: httpx.Request) -> httpx.Response:
		phase = "before" if request.url.path.startswith("/before/") else "after"
		actor = request.headers["x-benchmark-actor"]
		assert request.headers["authorization"] == "Bearer secret-not-in-artifacts"
		payload = json.loads(request.content)
		assert "label" not in payload and "scenario" not in payload
		seen[phase].append((actor, payload))
		counts[phase, actor] += 1
		n = counts[phase, actor]
		decision = "allow"
		if phase == "after":
			if (labels[actor] == "distillation" and n >= 3) or (actor == genuine[0] and n == 1):
				return httpx.Response(403, headers={"X-Harness-Decision": "block"})
			if actor == genuine[1] and n == 1:
				# A plain 403 is not evidence of a detector decision.
				return httpx.Response(403, text="secret provider body")
			if actor == genuine[2] and n == 1:
				raise httpx.ReadTimeout("secret timeout")
			if actor == genuine[3] and n == 1:
				return httpx.Response(429, headers={"Retry-After": "5"})
			if labels[actor] == "distillation" and n == 2:
				decision = "throttle"
		body: dict[str, object]
		if backend == "teacher_api":
			assert payload["enable_thinking"] is False
			body = {
				"id": "fixture",
				"model": "fixture-model",
				"provider": "test",
				"generated_at": "2026-10-15T00:00:00Z",
				"endpoint": "http://test",
				"message": {"role": "assistant", "content": "A complete response"},
				"usage": {"output_tokens": 4},
				"finish_reason": "stop",
			}
		else:
			assert payload["chat_template_kwargs"] == {"enable_thinking": False}
			body = {
				"model": "fixture-model",
				"choices": [
					{"message": {"content": "A complete response"}, "finish_reason": "stop"}
				],
				"usage": {"completion_tokens": 4},
			}
		return httpx.Response(200, headers={"X-Harness-Decision": decision}, json=body)

	transport = httpx.MockTransport(endpoint)
	before = await run_traffic(config, "before", transport=transport)
	after = await run_traffic(config, "after", transport=transport)
	assert before.comparison_key() == after.comparison_key()
	assert Counter(json.dumps(row, sort_keys=True) for row in seen["before"]) == Counter(
		json.dumps(row, sort_keys=True) for row in seen["after"]
	)
	report = build_traffic_report(config.output_dir, before.comparison_key())
	baseline, protected = report["scores"]["before"], report["scores"]["after"]
	assert baseline["genuine"]["success_percent"] == 100
	assert baseline["genuine"]["false_positive_percent"] == 0
	assert protected["genuine"]["n"] == 12
	assert protected["genuine"]["succeeded"] == 8
	assert protected["genuine"]["false_positive_count"] == 1
	assert protected["genuine"]["false_positive_percent"] is None
	assert protected["genuine"]["statuses"] == {
		"blocked": 1,
		"http_error": 1,
		"timeout": 1,
		"rate_limited": 1,
		"ok": 8,
	}
	assert protected["distillation"]["throttled"] == 2
	assert protected["distillation"]["succeeded"] == 4
	assert protected["distillation"]["blocked"] == 8
	assert report["comparison"]["distillation"]["throughput_reduction_percent"] > 0
	for score in (baseline, protected):
		for label in ("genuine", "distillation"):
			assert sum(w[label]["n"] for w in score["windows"]) == score[label]["n"]
	assert "secret" not in (config.output_dir / f"{after.id}.json").read_text()
	assert len((config.output_dir / f"{after.id}.samples.jsonl").read_text().splitlines()) == 24
	after.samples.pop()
	save_json(config.output_dir / f"{after.id}.json", after)
	with pytest.raises(ValueError, match="complete frozen suite"):
		build_traffic_report(config.output_dir, before.comparison_key())


def test_traffic_metrics_and_stop_do_not_treat_failures_as_blocks() -> None:
	def sample(i: int, decision: str, succeeded: bool = False) -> TrafficSample:
		return TrafficSample.model_validate(
			{
				"request_id": str(i),
				"scheduled_seconds": i,
				"started_seconds": i,
				"completed_seconds": i + 0.5,
				"decision": decision,
				"succeeded": succeeded,
				"status": "ok" if succeeded else "blocked" if decision == "block" else "error",
			}
		)

	samples = [sample(0, "allow", True), sample(1, "throttle", True), sample(2, "block")]
	assert not _progression(samples)["observed_stopped"]
	samples.append(sample(3, "block"))
	assert _progression(samples)["observed_stopped"]
	assert _progression(samples)["first_throttle_seconds"] == 1.5
	samples[0].completed_seconds = 4
	assert not _progression(samples)["observed_stopped"]
	samples.append(sample(4, "unknown"))
	assert not _progression(samples)["observed_stopped"]
	metrics = summarize_traffic(samples, 5)
	assert metrics["block_percent"] is None
	assert metrics["confirmed_block_percent"] == 40
	assert metrics["successful_output_tokens_per_second"] is None
	assert summarize_traffic([], 5)["success_percent"] is None


@pytest.mark.parametrize(
	("text", "finish", "valid"),
	[("Complete answer", "stop", True), ("", "stop", False), ("Cut off", "length", False)],
)
def test_complete_response_required_for_traffic_success(
	text: str, finish: str, valid: bool
) -> None:
	response = httpx.Response(
		200,
		json={
			"model": "fixture-model",
			"choices": [{"message": {"content": text}, "finish_reason": finish}],
		},
	)
	assert _completion(response, "openai") == (valid, None, "fixture-model")
	with pytest.raises(ValueError):
		_completion(httpx.Response(200, json={"model": "fixture", "choices": []}), "openai")


@pytest.mark.asyncio
async def test_traffic_dashboard_and_comparison_protocol(tmp_path: Path) -> None:
	create_traffic_demo(tmp_path)
	runs, errors = discover_traffic(tmp_path)
	assert not errors and len(runs) == 2
	key = runs[0].comparison_key()
	assert traffic_experiments(tmp_path)["experiments"][0]["phases"] == ["after", "before"]
	report = build_traffic_report(tmp_path, key)
	assert report["scores"]["after"]["genuine"]["false_positive_count"] == 1
	assert report["scores"]["after"]["genuine"]["false_positive_percent"] == 100 / 48
	assert all(
		a["observed_stopped"]
		for a in report["scores"]["after"]["actors"].values()
		if a["label"] == "distillation"
	)
	runs[0].config.generation.max_output_tokens += 1
	assert runs[0].comparison_key() != key
	app = create_app(tmp_path / "quality", traffic_results_dir=tmp_path)
	async with httpx.AsyncClient(
		transport=httpx.ASGITransport(app=app), base_url="http://test"
	) as client:
		assert (await client.get("/api/traffic/experiments")).json()["experiments"]
		assert (await client.get(f"/api/traffic/reports/{key}")).json()["demo"] is True
		assert (await client.get("/api/traffic/reports/missing")).status_code == 422
	config = load_traffic_config(Path("configs/traffic.example.toml"))
	assert set(config.targets) == {"before", "after"}
