"""Scoring and comparison checks use fixtures, never downloaded model weights."""

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from distillation.benchmarks.dashboard import create_app
from distillation.benchmarks.demo import create_demo
from distillation.benchmarks.inference import ConcurrentServed, make_model, validate_adapter
from distillation.benchmarks.models import (
	BenchmarkItem,
	DatasetSource,
	EvaluationConfig,
	EvaluationRun,
	FrozenSuite,
	ModelConfig,
	SampleResult,
	load_config,
	save_json,
)
from distillation.benchmarks.reporting import build_report, experiments, paired, summarize, wilson
from distillation.benchmarks.runner import audit_training, run_evaluation
from distillation.benchmarks.scoring import score_answer
from distillation.schemas import Message, TeacherGeneration, TeacherGenerationRequest, TokenUsage
from distillation.tasks import _select


def item(identifier: str = "q1", metric: str = "label_accuracy") -> BenchmarkItem:
	return BenchmarkItem.model_validate(
		{
			"id": identifier,
			"benchmark": "fixture",
			"category": "general",
			"question": f"Distinct held-out question {identifier}",
			"messages": [{"role": "user", "content": f"Distinct held-out question {identifier}"}],
			"metric": metric,
			"answer": "1200" if metric == "numeric_exact_match" else "A",
			"choices": ["A", "B"],
		}
	)


def suite_path(directory: Path, count: int = 5) -> Path:
	suite = FrozenSuite(
		seed=42,
		created_at=datetime.now(UTC),
		items=[item(str(i)) for i in range(count)],
		sources=[
			DatasetSource(
				benchmark="fixture",
				repository="fixture",
				revision="fixture-v1",
				configuration="fixture",
				split="test",
				license="Apache-2.0",
				url="https://example.invalid",
				available=count,
				selected=count,
			)
		],
	)
	suite.sha256 = suite.digest()
	path = directory / "suite.json"
	save_json(path, suite)
	return path


def sample(identifier: str, correct: bool, **changes: object) -> SampleResult:
	return SampleResult.model_validate(
		{
			"item_id": identifier,
			"benchmark": "fixture",
			"category": "general",
			"response": "A",
			"prediction": "A" if correct else "B",
			"correct": correct,
			"status": "ok",
			"latency_seconds": 2,
			**changes,
		}
	)


@pytest.mark.parametrize(
	("response", "prediction", "correct"),
	[
		("A", "A", True),
		(" B ", "B", False),
		("The answer is A", None, False),
		("A or B", None, False),
		("C", None, False),
		("", None, False),
	],
)
def test_choice_parser_never_guesses(response: str, prediction: str | None, correct: bool) -> None:
	assert score_answer(item(), response) == (prediction, correct)


@pytest.mark.parametrize(
	("response", "correct"),
	[
		("Work: 600 + 600.\n#### 1,200", True),
		("1200.00", True),
		("I computed 1200", False),
		("#### 1200 and 9", False),
		("#### 1,20,0", False),
		("#### 12", False),
		("#### 1200.", True),
		("#### $1,200", True),
		("#### $1200.", True),
		("#### 1200..", False),
		("#### $$1200", False),
	],
)
def test_math_exact_match_requires_a_final_answer(response: str, correct: bool) -> None:
	assert score_answer(item(metric="numeric_exact_match"), response)[1] is correct


def test_statistics_keep_errors_and_unknown_usage() -> None:
	summary = summarize(
		[
			sample("a", True, usage=TokenUsage(output_tokens=8)),
			sample("b", False, status="error", latency_seconds=100),
		]
	)
	assert summary["score"] == 50
	assert summary["n"] == 2
	assert summary["latency_p50"] == 2
	assert summary["tokens_per_second"] == 4
	assert summary["token_coverage"] == 0.5
	assert summarize([sample("a", False, status="error")])["latency_p95"] is None
	assert wilson(0, 10)[1] > 0
	assert wilson(10, 10)[0] < 100


def test_paired_bootstrap_counts_regressions_and_is_reproducible() -> None:
	before = [sample("a", False), sample("b", True), sample("c", True)]
	after = [sample("c", True), sample("a", True), sample("b", False)]
	comparison = paired(before, after)
	assert comparison == paired(before, after)
	assert comparison["delta_pp"] == 0
	assert comparison["improved"] == comparison["regressed"] == 1
	assert comparison["both_correct"] == 1
	with pytest.raises(ValueError, match="same nonempty"):
		paired(before, after[:2])


def test_freeze_detects_mutation_and_training_overlap(tmp_path: Path) -> None:
	path = suite_path(tmp_path)
	suite = FrozenSuite.model_validate_json(path.read_text())
	suite.items[0].answer = "B"
	with pytest.raises(ValueError, match="hash mismatch"):
		suite.verify()
	training = tmp_path / "training.jsonl"
	training.write_text(
		json.dumps({"prompt": [{"content": "Distinct held-out question 0\nAnswer A"}]}) + "\n"
	)
	with pytest.raises(ValueError, match="held-out"):
		audit_training(path, [training])
	training.write_text(json.dumps({"prompt": [{"content": "Unrelated training prompt"}]}) + "\n")
	assert str(training) in audit_training(path, [training])


def test_development_sampling() -> None:
	items = [item(str(i)).model_copy(update={"category": f"subject-{i % 3}"}) for i in range(12)]
	assert len({i.category for i in _select(items, 3, 42)}) == 3
	assert _select(items, 4, 42) == _select(items, 4, 42)
	assert len(_select(items, 0, 42)) == 12


class FakeModel:
	closed = False

	async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
		assert request.enable_thinking is False
		assert request.temperature == 0
		question = request.messages[0].content
		if question.endswith("1"):
			raise RuntimeError("secret provider response")
		response = "<think>private trace</think>A" if question.endswith("3") else "A"
		return TeacherGeneration(
			id="fixture",
			provider="fixture",
			model="fixture-model",
			generated_at=datetime.now(UTC),
			message=Message(role="assistant", content=response),
			finish_reason="length"
			if question.endswith("2")
			else None
			if question.endswith("4")
			else "stop",
			endpoint="https://example.invalid",
			usage=TokenUsage(output_tokens=1),
		)

	async def close(self) -> None:
		self.closed = True


@pytest.mark.asyncio
async def test_runner_saves_all_attempts_and_never_scores_truncated_answers(tmp_path: Path) -> None:
	fake = FakeModel()
	config = EvaluationConfig(
		experiment="test",
		suite_path=suite_path(tmp_path),
		output_dir=tmp_path / "results",
		models={
			"student_before": ModelConfig(
				backend="openai",
				model="fixture-model",
				revision="fixture-v1",
				base_url="https://example.invalid",
			)
		},
	)
	run = await run_evaluation(config, "student_before", factory=lambda _: fake)
	assert fake.closed
	assert [s.status for s in run.samples] == ["ok", "error", "truncated", "invalid", "invalid"]
	assert sum(s.correct for s in run.samples) == 1
	assert "secret" not in (config.output_dir / f"{run.id}.json").read_text()
	assert len((config.output_dir / f"{run.id}.samples.jsonl").read_text().splitlines()) == 5
	key = experiments(config.output_dir)["experiments"][0]["id"]
	assert (
		build_report(config.output_dir, key)["scores"]["student_before"]["fixture"]["score"] == 20
	)


class BatchedModel(FakeModel):
	batch_size = 2
	execution = {"backend": "transformers", "batch_size": 2, "adapter_merged": True}

	def __init__(self) -> None:
		self.batches: list[int] = []

	async def generate_batch(
		self, requests: list[TeacherGenerationRequest]
	) -> list[TeacherGeneration]:
		self.batches.append(len(requests))
		if any(r.messages[0].content.endswith("1") for r in requests):
			raise RuntimeError("secret provider response")
		return [await self.generate(r) for r in requests]


@pytest.mark.asyncio
async def test_batched_runner_keeps_item_order_and_records_its_execution(tmp_path: Path) -> None:
	config = EvaluationConfig(
		experiment="test",
		suite_path=suite_path(tmp_path),
		output_dir=tmp_path / "results",
		models={
			"student_before": ModelConfig(
				backend="openai",
				model="fixture-model",
				revision="fixture-v1",
				base_url="https://example.invalid",
			)
		},
	)
	single = await run_evaluation(config, "student_before", factory=lambda _: FakeModel())
	batched_model = BatchedModel()
	batched = await run_evaluation(config, "student_before", factory=lambda _: batched_model)
	assert batched_model.batches == [2, 2, 1]
	# The batch holding the failing item retries singly, so only that item is an error.
	assert [(s.item_id, s.status, s.correct) for s in batched.samples] == [
		(s.item_id, s.status, s.correct) for s in single.samples
	]
	assert batched.execution == BatchedModel.execution
	assert single.execution == {}
	assert "secret" not in (config.output_dir / f"{batched.id}.json").read_text()


@pytest.mark.asyncio
async def test_served_models_keep_requests_in_flight_without_changing_results(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = EvaluationConfig(
		experiment="test",
		suite_path=suite_path(tmp_path),
		output_dir=tmp_path / "results",
		models={
			"teacher": ModelConfig(
				backend="teacher_api",
				model="fixture-model",
				revision="fixture-v1",
				base_url="https://example.invalid",
			)
		},
	)
	single = await run_evaluation(config, "teacher", factory=lambda _: FakeModel())
	served = ConcurrentServed(FakeModel(), "teacher_api", 3)
	concurrent = await run_evaluation(config, "teacher", factory=lambda _: served)
	# The failing request is an item error while the rest of its batch still scores.
	assert [(s.item_id, s.status, s.correct) for s in concurrent.samples] == [
		(s.item_id, s.status, s.correct) for s in single.samples
	]
	assert concurrent.execution == {"backend": "teacher_api", "concurrency": 3}
	assert served._model.closed
	assert "secret" not in (config.output_dir / f"{concurrent.id}.json").read_text()
	model = make_model(config.models["teacher"])
	assert isinstance(model, ConcurrentServed) and model.batch_size == 16
	monkeypatch.setenv("DISTILL_EVAL_CONCURRENCY", "1")
	assert not isinstance(make_model(config.models["teacher"]), ConcurrentServed)


def test_adapter_identity_and_data_integrity(tmp_path: Path) -> None:
	data = tmp_path / "train.jsonl"
	data.write_text("original")
	import hashlib

	manifest = {
		"config": {"base_model": "base", "base_revision": "revision", "dataset_path": str(data)},
		"dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
	}
	(tmp_path / "training-manifest.json").write_text(json.dumps(manifest))
	config = ModelConfig(backend="local", model="base", revision="revision", adapter_path=tmp_path)
	assert validate_adapter(config) == [data]
	data.write_text("changed")
	with pytest.raises(ValueError, match="differs"):
		validate_adapter(config)
	with pytest.raises(ValueError, match="does not match"):
		validate_adapter(config.model_copy(update={"model": "different"}))
	data.write_text("original")
	with pytest.raises(ValueError, match="qlora"):
		validate_adapter(config.model_copy(update={"load_in_4bit": True}))


def test_comparison_contract_separates_incompatible_runs_and_detects_missing_items(
	tmp_path: Path,
) -> None:
	create_demo(tmp_path)
	listing = experiments(tmp_path)
	assert len(listing["experiments"]) == 1
	key = listing["experiments"][0]["id"]
	report = build_report(tmp_path, key)
	assert report["demo"] and len(report["runs"]) == 4
	assert "student_gain" in report["comparisons"]
	assert "teacher_drift" in report["comparisons"]
	path = tmp_path / "demo-student_after.json"
	run = EvaluationRun.model_validate_json(path.read_text())
	run.generation.max_output_tokens += 1
	save_json(path, run)
	assert len(experiments(tmp_path)["experiments"]) == 2
	assert "student_gain" not in build_report(tmp_path, key)["comparisons"]
	run.generation.max_output_tokens -= 1
	run.samples.pop()
	save_json(path, run)
	with pytest.raises(ValueError, match="complete frozen suite"):
		build_report(tmp_path, key)


@pytest.mark.asyncio
async def test_dashboard_reads_artifacts_and_empty_state(tmp_path: Path) -> None:
	app = create_app(tmp_path)
	async with httpx.AsyncClient(
		transport=httpx.ASGITransport(app=app), base_url="http://test"
	) as client:
		assert (await client.get("/api/experiments")).json()["experiments"] == []
		assert (await client.get("/api/reports/missing")).status_code == 422
		create_demo(tmp_path)
		key = (await client.get("/api/experiments")).json()["experiments"][0]["id"]
		assert (await client.get(f"/api/reports/{key}")).json()["demo"] is True


@pytest.mark.parametrize("task", ["ag_news", "squad", "gsm8k"])
def test_task_config_has_all_four_slots_without_loading_weights(task: str) -> None:
	config = load_config(Path(f"configs/eval.{task}.toml"))
	assert set(config.models) == {"teacher", "teacher_after", "student_before", "student_after"}
	assert config.models["student_before"].adapter_path is None
	assert config.models["student_after"].adapter_path is not None
	before = config.models["student_before"].model_dump(exclude={"adapter_path"})
	after = config.models["student_after"].model_dump(exclude={"adapter_path"})
	assert before == after
