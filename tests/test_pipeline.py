import asyncio
import json
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from distillation.collection import CollectionOptions, CollectionProgress, collect
from distillation.io import read_jsonl, write_jsonl
from distillation.preparation import prepare_sft_dataset
from distillation.review import review_ledger
from distillation.schemas import (
	CollectionAttemptLog,
	LedgerRecord,
	Message,
	PromptRecord,
	PromptSource,
	ReviewRecord,
	SFTRecord,
	TeacherGeneration,
	TeacherGenerationRequest,
	TokenUsage,
)
from distillation.teacher import OpenAICompatibleTeacher, TeacherAPIGateway, TeacherUpstreamError
from teacher_api.app import create_app
from teacher_api.settings import Settings


class FakeTeacher:
	def __init__(self) -> None:
		self.calls = 0
		self.closed = False

	async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
		self.calls += 1
		return TeacherGeneration(
			id=f"generation-{self.calls}",
			provider="licensed-teacher",
			model="teacher-v1",
			generated_at=datetime.now(UTC),
			message=Message(role="assistant", content=f"Answer to {request.messages[-1].content}"),
			finish_reason="stop",
			usage=TokenUsage(input_tokens=4, output_tokens=3),
			endpoint="https://teacher.example/v1/chat/completions",
		)

	async def close(self) -> None:
		self.closed = True


class PartiallyFailingTeacher(FakeTeacher):
	async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
		if request.messages[-1].content.endswith("two"):
			raise TeacherUpstreamError("planned failure")
		await asyncio.sleep(0.01)
		return await super().generate(request)


def prompt(example_id: str) -> PromptRecord:
	return PromptRecord(
		example_id=example_id,
		split="train",
		task="qa",
		source=PromptSource(
			corpus="test",
			source_id=example_id,
			license="Apache-2.0",
		),
		prompt=[Message(role="user", content=f"Question {example_id}")],
	)


@pytest.mark.asyncio
async def test_collection_is_resumable_and_records_provenance(tmp_path: Path) -> None:
	input_path = tmp_path / "prompts.jsonl"
	output_path = tmp_path / "ledger.jsonl"
	write_jsonl(input_path, [prompt("one"), prompt("two")])
	teacher = FakeTeacher()

	collected, skipped = await collect(
		input_path=input_path,
		output_path=output_path,
		teacher=teacher,
		options=CollectionOptions(terms_review_id="approval-2026-08"),
	)
	second_teacher = FakeTeacher()
	second_collected, second_skipped = await collect(
		input_path=input_path,
		output_path=output_path,
		teacher=second_teacher,
		options=CollectionOptions(terms_review_id="approval-2026-08"),
	)

	records = read_jsonl(output_path, LedgerRecord)
	attempts = read_jsonl(tmp_path / "ledger.attempts.jsonl", CollectionAttemptLog)
	assert (collected, skipped) == (2, 0)
	assert (second_collected, second_skipped) == (0, 2)
	assert teacher.closed and second_teacher.closed
	assert {record.example_id for record in records} == {"one", "two"}
	assert records[0].teacher.terms_review_id == "approval-2026-08"
	assert records[0].teacher.endpoint == "https://teacher.example/v1/chat/completions"
	assert records[0].quality.status == "pending"
	assert records[0].attempts[0].status == "succeeded"
	assert len(attempts) == 2


@pytest.mark.asyncio
async def test_prepare_writes_completion_only_records(tmp_path: Path) -> None:
	input_path = tmp_path / "prompts.jsonl"
	ledger_path = tmp_path / "ledger.jsonl"
	reviews_path = tmp_path / "reviews.jsonl"
	prepared_path = tmp_path / "train.jsonl"
	write_jsonl(input_path, [prompt("one")])
	await collect(
		input_path=input_path,
		output_path=ledger_path,
		teacher=FakeTeacher(),
		options=CollectionOptions(terms_review_id="approval-2026-08"),
	)
	write_jsonl(
		reviews_path,
		[
			ReviewRecord(
				example_id="one",
				status="accepted",
				checks=["human_review"],
				reviewer="reviewer-1",
			)
		],
	)

	written, skipped = prepare_sft_dataset(
		input_path=ledger_path,
		output_path=prepared_path,
		reviews_path=reviews_path,
	)
	records = read_jsonl(prepared_path, SFTRecord)

	assert (written, skipped) == (1, 0)
	assert records[0].prompt[0].content == "Question one"
	assert records[0].completion[0].content == "Answer to Question one"


@pytest.mark.asyncio
async def test_collection_skips_test_prompts_without_explicit_opt_in(tmp_path: Path) -> None:
	input_path = tmp_path / "prompts.jsonl"
	output_path = tmp_path / "ledger.jsonl"
	test_prompt = prompt("held-out").model_copy(update={"split": "test"})
	write_jsonl(input_path, [test_prompt])
	teacher = FakeTeacher()

	collected, skipped = await collect(
		input_path=input_path,
		output_path=output_path,
		teacher=teacher,
		options=CollectionOptions(terms_review_id="approval-2026-08"),
	)

	assert (collected, skipped) == (0, 1)
	assert teacher.calls == 0
	assert teacher.closed
	assert not output_path.exists()


def test_prepare_rejects_frozen_test_split(tmp_path: Path) -> None:
	ledger_path = tmp_path / "ledger.jsonl"
	write_jsonl(ledger_path, [])

	with pytest.raises(ValueError, match="frozen test split"):
		prepare_sft_dataset(
			input_path=ledger_path,
			output_path=tmp_path / "test.jsonl",
			split="test",
		)


@pytest.mark.asyncio
async def test_collection_persists_completed_calls_before_later_failure(tmp_path: Path) -> None:
	input_path = tmp_path / "prompts.jsonl"
	output_path = tmp_path / "ledger.jsonl"
	write_jsonl(input_path, [prompt("one"), prompt("two")])
	teacher = PartiallyFailingTeacher()

	with pytest.raises(TeacherUpstreamError, match="planned failure"):
		await collect(
			input_path=input_path,
			output_path=output_path,
			teacher=teacher,
			options=CollectionOptions(
				terms_review_id="approval-2026-08",
				max_attempts=1,
			),
		)

	records = read_jsonl(output_path, LedgerRecord)
	attempts = read_jsonl(tmp_path / "ledger.attempts.jsonl", CollectionAttemptLog)
	assert [record.example_id for record in records] == ["one"]
	assert {attempt.example_id for attempt in attempts} == {"one", "two"}
	assert {attempt.attempt.status for attempt in attempts} == {"succeeded", "failed"}
	assert teacher.closed
	before = output_path.read_bytes()
	recovered = FakeTeacher()
	assert await collect(
		input_path=input_path,
		output_path=output_path,
		teacher=recovered,
		options=CollectionOptions(terms_review_id="approval-2026-08", max_attempts=1),
	) == (1, 1)
	assert recovered.calls == 1 and recovered.closed
	assert output_path.read_bytes().startswith(before)
	assert [r.example_id for r in read_jsonl(output_path, LedgerRecord)] == ["one", "two"]
	attempts = read_jsonl(tmp_path / "ledger.attempts.jsonl", CollectionAttemptLog)
	assert [(a.example_id, a.attempt.status) for a in attempts if a.example_id == "two"] == [
		("two", "failed"),
		("two", "succeeded"),
	]


@pytest.mark.asyncio
async def test_prepare_drops_prompts_present_in_another_split(tmp_path: Path) -> None:
	input_path = tmp_path / "prompts.jsonl"
	ledger_path = tmp_path / "ledger.jsonl"
	reviews_path = tmp_path / "reviews.jsonl"
	train_prompt = prompt("train-one")
	validation_prompt = prompt("validation-one").model_copy(
		update={"split": "validation", "prompt": train_prompt.prompt}
	)
	write_jsonl(input_path, [train_prompt, validation_prompt])
	await collect(
		input_path=input_path,
		output_path=ledger_path,
		teacher=FakeTeacher(),
		options=CollectionOptions(terms_review_id="approval-2026-08"),
	)
	write_jsonl(
		reviews_path,
		[
			ReviewRecord(
				example_id="train-one",
				status="accepted",
				checks=["human_review"],
				reviewer="reviewer-1",
			)
		],
	)

	written, skipped = prepare_sft_dataset(
		input_path=ledger_path,
		output_path=tmp_path / "train.jsonl",
		reviews_path=reviews_path,
	)

	assert (written, skipped) == (0, 2)


@pytest.mark.asyncio
async def test_collection_rejects_duplicate_input_ids(tmp_path: Path) -> None:
	input_path = tmp_path / "prompts.jsonl"
	write_jsonl(input_path, [prompt("duplicate"), prompt("duplicate")])
	teacher = FakeTeacher()

	with pytest.raises(ValueError, match="duplicate example_id"):
		await collect(
			input_path=input_path,
			output_path=tmp_path / "ledger.jsonl",
			teacher=teacher,
			options=CollectionOptions(terms_review_id="approval-2026-08"),
		)

	assert teacher.calls == 0
	assert teacher.closed


@pytest.mark.asyncio
async def test_pending_bypass_never_overrides_explicit_rejection(tmp_path: Path) -> None:
	input_path, ledger_path = tmp_path / "prompts.jsonl", tmp_path / "ledger.jsonl"
	reviews_path = tmp_path / "reviews.jsonl"
	write_jsonl(input_path, [prompt("one")])
	await collect(
		input_path=input_path,
		output_path=ledger_path,
		teacher=FakeTeacher(),
		options=CollectionOptions(terms_review_id="approval"),
	)
	write_jsonl(
		reviews_path,
		[
			ReviewRecord(
				example_id="one", status="rejected", checks=["incorrect_answer"], reviewer="human"
			)
		],
	)
	assert prepare_sft_dataset(
		input_path=ledger_path,
		output_path=tmp_path / "train.jsonl",
		reviews_path=reviews_path,
		accept_pending=True,
	) == (0, 1)


@pytest.mark.asyncio
async def test_prepare_checks_uncollected_test_prompts_and_source_groups(tmp_path: Path) -> None:
	input_path, ledger_path = tmp_path / "prompts.jsonl", tmp_path / "ledger.jsonl"
	train = prompt("one")
	duplicate = prompt("test").model_copy(update={"split": "test", "prompt": train.prompt})
	variant = prompt("variant").model_copy(update={"source": train.source})
	grouped_test = prompt("grouped-test").model_copy(
		update={"split": "test", "source": train.source}
	)
	write_jsonl(input_path, [train, duplicate, variant, grouped_test])
	await collect(
		input_path=input_path,
		output_path=ledger_path,
		teacher=FakeTeacher(),
		options=CollectionOptions(terms_review_id="approval"),
	)
	assert len(read_jsonl(ledger_path, LedgerRecord)) == 2
	assert prepare_sft_dataset(
		input_path=ledger_path,
		output_path=tmp_path / "train.jsonl",
		prompts_path=input_path,
		accept_pending=True,
	) == (0, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
	"field",
	[
		"prompt",
		"split",
		"task",
		"source.corpus",
		"source.source_id",
		"source.license_name",
		"source.group_id",
		"temperature",
		"max_output_tokens",
		"seed",
		"enable_thinking",
		"top_p",
		"top_k",
		"terms_review_id",
	],
)
async def test_resume_rejects_changed_contract_without_touching_the_ledger(
	tmp_path: Path, field: str
) -> None:
	input_path, ledger_path = tmp_path / "prompts.jsonl", tmp_path / "ledger.jsonl"
	original = prompt("one")
	options = CollectionOptions(terms_review_id="approval")
	write_jsonl(input_path, [original])
	await collect(
		input_path=input_path,
		output_path=ledger_path,
		teacher=FakeTeacher(),
		options=options,
	)
	ledger_before = ledger_path.read_bytes()
	attempt_path = tmp_path / "ledger.attempts.jsonl"
	attempts_before = attempt_path.read_bytes()
	changed = original
	if field.startswith("source."):
		changed = original.model_copy(
			update={"source": original.source.model_copy(update={field.split(".")[1]: "different"})}
		)
	elif field in {"prompt", "split", "task"}:
		changes = {
			"prompt": [Message(role="user", content="A different question")],
			"split": "validation",
			"task": "different",
		}
		changed = original.model_copy(update={field: changes[field]})
	else:
		options = {
			"temperature": replace(options, temperature=0.7),
			"max_output_tokens": replace(options, max_output_tokens=64),
			"seed": replace(options, seed=42),
			"enable_thinking": replace(options, enable_thinking=False),
			"top_p": replace(options, top_p=0.8),
			"top_k": replace(options, top_k=20),
			"terms_review_id": replace(options, terms_review_id="different"),
		}[field]
	write_jsonl(input_path, [changed])
	teacher = FakeTeacher()
	with pytest.raises(ValueError, match="resume mismatch"):
		await collect(
			input_path=input_path,
			output_path=ledger_path,
			teacher=teacher,
			options=options,
		)
	assert teacher.calls == 0
	assert teacher.closed
	assert ledger_path.read_bytes() == ledger_before
	assert attempt_path.read_bytes() == attempts_before


@pytest.mark.asyncio
async def test_http_collection_retries_reviews_and_resumes_without_losing_provenance(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Only the external provider is simulated; both HTTP adapters and pipeline stages run."""
	monkeypatch.chdir(tmp_path)
	seen: Counter[str] = Counter()

	async def upstream(request: httpx.Request) -> httpx.Response:
		assert request.url == "https://provider.invalid/v1/chat/completions"
		assert request.headers["authorization"] == "Bearer upstream-secret"
		payload = json.loads(request.content)
		assert payload["model"] == "requested-model"
		assert payload["temperature"] == 0.7 and payload["max_tokens"] == 64
		assert payload["top_p"] == 0.8 and payload["top_k"] == 20 and payload["seed"] == 42
		assert payload["chat_template_kwargs"] == {"enable_thinking": False}
		question = payload["messages"][-1]["content"]
		seen[question] += 1
		if question == "Question retry" and seen[question] == 1:
			return httpx.Response(429, text="private provider diagnostic upstream-secret")
		return httpx.Response(
			200,
			headers={"x-request-id": f"request-{question}"},
			json={
				"id": f"generation-{question}",
				"model": "actual-quantized-model",
				"choices": [
					{
						"message": {
							"content": "<think>hidden</think>#### 56"
							if question == "Question thinking"
							else "#### 56"
						},
						"finish_reason": "length" if question == "Question truncated" else "stop",
					}
				],
				"usage": {"prompt_tokens": 10, "completion_tokens": 4},
			},
		)

	teacher = OpenAICompatibleTeacher(
		base_url="https://provider.invalid/v1",
		api_key="upstream-secret",
		model="requested-model",
		provider="test-provider",
		transport=httpx.MockTransport(upstream),
	)
	app = create_app(settings=Settings(TEACHER_API_TOKEN="gateway-secret"), teacher=teacher)
	input_path, ledger = tmp_path / "prompts.jsonl", tmp_path / "ledger.jsonl"
	prompts = [prompt(name) for name in ("good", "retry", "truncated", "thinking")]
	prompts.append(prompt("held-out").model_copy(update={"split": "test"}))
	write_jsonl(input_path, prompts)
	options = CollectionOptions(
		terms_review_id="approval",
		temperature=0.7,
		top_p=0.8,
		top_k=20,
		seed=42,
		max_output_tokens=64,
		enable_thinking=False,
		concurrency=2,
	)
	progress: list[CollectionProgress] = []
	async with app.router.lifespan_context(app):

		def gateway() -> TeacherAPIGateway:
			return TeacherAPIGateway(
				base_url="http://teacher",
				token="gateway-secret",
				transport=httpx.ASGITransport(app=app),
			)

		assert await collect(
			input_path=input_path,
			output_path=ledger,
			teacher=gateway(),
			options=options,
			progress=progress.append,
		) == (4, 1)
		before = ledger.read_bytes()
		before_calls = seen.copy()
		write_jsonl(input_path, [*prompts, prompt("new")])
		assert await collect(
			input_path=input_path, output_path=ledger, teacher=gateway(), options=options
		) == (1, 5)
		assert seen - before_calls == {"Question new": 1}
		assert ledger.read_bytes().startswith(before)
	assert "Question held-out" not in seen
	assert seen["Question retry"] == 2
	assert (progress[-1].done, progress[-1].total, progress[-1].retries, progress[-1].rejected) == (
		4,
		4,
		1,
		2,
	)
	records = read_jsonl(ledger, LedgerRecord)
	assert len({r.example_id for r in records}) == 5
	assert all(r.teacher.model == "actual-quantized-model" for r in records)
	assert all(
		r.teacher.endpoint == "https://provider.invalid/v1/chat/completions" for r in records
	)
	assert all(r.usage == TokenUsage(input_tokens=10, output_tokens=4) for r in records)
	retried = next(r for r in records if r.example_id == "retry")
	assert [a.status for a in retried.attempts] == ["failed", "succeeded"]
	attempts = read_jsonl(tmp_path / "ledger.attempts.jsonl", CollectionAttemptLog)
	assert len(attempts) == 6
	assert len({a.attempt.attempt_id for a in attempts}) == 6
	assert retried.attempt_id == retried.attempts[-1].attempt_id
	assert retried.teacher.request_id == "request-Question retry"
	for record in records:
		assert [a.attempt for a in attempts if a.example_id == record.example_id] == record.attempts
	assert {
		r.example_id: r.quality.rejection_reason for r in records if r.quality.status == "rejected"
	} == {
		"truncated": "finish_reason:length",
		"thinking": "unexpected_thinking_trace",
	}
	for path in (ledger, tmp_path / "ledger.attempts.jsonl"):
		assert "secret" not in path.read_text() and "private provider" not in path.read_text()
	reviews, prepared = (tmp_path / name for name in ("reviews.jsonl", "train.jsonl"))
	review_ledger(ledger_path=ledger, output_path=reviews)
	assert prepare_sft_dataset(
		input_path=ledger, output_path=prepared, reviews_path=reviews, prompts_path=input_path
	) == (3, 2)
	assert [r.completion[0].content for r in read_jsonl(prepared, SFTRecord)] == ["#### 56"] * 3
	assert ledger.read_bytes().startswith(before)


@pytest.mark.asyncio
async def test_collection_bounds_concurrency_and_flushes_before_other_workers_finish(
	tmp_path: Path,
) -> None:
	input_path, ledger = tmp_path / "prompts.jsonl", tmp_path / "ledger.jsonl"
	write_jsonl(input_path, [prompt(name) for name in ("slow", "fast", "queued")])
	started: list[str] = []
	full, release_fast, release_slow, persisted = (asyncio.Event() for _ in range(4))
	active = peak = 0

	class ControlledTeacher(FakeTeacher):
		async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
			nonlocal active, peak
			active += 1
			peak = max(peak, active)
			question = request.messages[-1].content
			started.append(question)
			if len(started) == 2:
				full.set()
			try:
				await (release_fast if question == "Question fast" else release_slow).wait()
				return await super().generate(request)
			finally:
				active -= 1

	def progress(state: CollectionProgress) -> None:
		if state.collected:
			persisted.set()

	teacher = ControlledTeacher()
	task = asyncio.create_task(
		collect(
			input_path=input_path,
			output_path=ledger,
			teacher=teacher,
			options=CollectionOptions(terms_review_id="approval", concurrency=2),
			progress=progress,
		)
	)
	try:
		await asyncio.wait_for(full.wait(), timeout=5)
		assert started == ["Question slow", "Question fast"] and active == 2
		release_fast.set()
		await asyncio.wait_for(persisted.wait(), timeout=5)
		assert [r.example_id for r in read_jsonl(ledger, LedgerRecord)] == ["fast"]
		assert not task.done()
	finally:
		release_fast.set()
		release_slow.set()
		result = await asyncio.wait_for(task, timeout=5)
	assert result == (3, 0) and peak == 2 and active == 0
	assert teacher.closed
	assert {r.example_id for r in read_jsonl(ledger, LedgerRecord)} == {"slow", "fast", "queued"}
