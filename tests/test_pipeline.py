import asyncio
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread

import pytest
import uvicorn
from distillation import launcher
from distillation.collection import CollectionOptions, collect
from distillation.io import read_jsonl, write_jsonl
from distillation.preparation import prepare_sft_dataset
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
from distillation.teacher import TeacherUpstreamError
from teacher_api.app import create_app
from teacher_api.settings import Settings
from typer.testing import CliRunner


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


def test_student_launcher_collects_through_monorepo_teacher_api(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Exercise readiness, auth, smoke generation, and collection through the real API."""
	monkeypatch.setattr(launcher.Launcher, "install", lambda self, component: None)
	monkeypatch.setattr(
		launcher.Pipeline,
		"prepare_and_train",
		lambda self: (tmp_path / "training-started").touch(),
	)
	monkeypatch.chdir(tmp_path)
	monkeypatch.setenv("TEACHER_API_TOKEN", "wrapper-key")
	monkeypatch.setenv("TEACHER_TERMS_REVIEW_ID", "test-approval")
	(tmp_path / "configs").mkdir()
	(tmp_path / "configs/train.distill.toml").write_text('dataset_path = "unused.jsonl"\n')
	corpus = tmp_path / "data/prompts"
	corpus.mkdir(parents=True)
	write_jsonl(corpus / "all-splits.jsonl", [prompt("one"), prompt("two")])
	(corpus / "references.jsonl").write_text("")
	(corpus / "manifest.json").write_text('{"profile":"quick"}\n')
	teacher = FakeTeacher()
	api = create_app(settings=Settings(TEACHER_API_TOKEN="wrapper-key"), teacher=teacher)
	with socket.socket() as listener:
		listener.bind(("127.0.0.1", 0))
		base_url = f"http://127.0.0.1:{listener.getsockname()[1]}"
		server = uvicorn.Server(uvicorn.Config(api, log_level="error"))
		thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
		thread.start()
		try:
			deadline = time.monotonic() + 5
			while not server.started and thread.is_alive() and time.monotonic() < deadline:
				time.sleep(0.01)
			assert server.started
			result = CliRunner().invoke(
				launcher.run_app,
				[
					"student-train",
					"--teacher-url",
					base_url,
					"--corpus",
					"quick",
					"--concurrency",
					"8",
				],
			)
		finally:
			server.should_exit = True
			thread.join(timeout=5)
	assert result.exit_code == 0, result.output
	assert (tmp_path / "training-started").exists()
	records = read_jsonl(tmp_path / "data/teacher/ledger.jsonl", LedgerRecord)
	assert len(records) == 2
	assert teacher.calls == 3  # One smoke generation and two corpus prompts.
	assert teacher.closed
	assert all(record.teacher.model == "teacher-v1" for record in records)
	assert all(record.teacher.provider == "licensed-teacher" for record in records)


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
async def test_resume_rejects_changed_prompt_before_any_calls(tmp_path: Path) -> None:
	input_path, ledger_path = tmp_path / "prompts.jsonl", tmp_path / "ledger.jsonl"
	write_jsonl(input_path, [prompt("one")])
	await collect(
		input_path=input_path,
		output_path=ledger_path,
		teacher=FakeTeacher(),
		options=CollectionOptions(terms_review_id="approval"),
	)
	changed = prompt("one").model_copy(
		update={"prompt": [Message(role="user", content="A different question")]}
	)
	write_jsonl(input_path, [changed])
	teacher = FakeTeacher()
	with pytest.raises(ValueError, match="resume mismatch"):
		await collect(
			input_path=input_path,
			output_path=ledger_path,
			teacher=teacher,
			options=CollectionOptions(terms_review_id="approval"),
		)
	assert teacher.calls == 0
	assert teacher.closed
