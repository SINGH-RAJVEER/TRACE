"""The opt-in release-boundary output policy, through HTTP with a fixture teacher."""

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from distillation.output_sets import DENSE_INSTRUCTION, INSTRUCTIONS
from distillation.schemas import (
	Message,
	TeacherGeneration,
	TeacherGenerationRequest,
	TokenUsage,
)
from distillation.tasks import AG_NEWS_PROMPT, GSM8K_EVAL_SUFFIX
from distillation.teacher import TeacherUpstreamError
from harness.calibration import Calibration
from harness.mmd import WindowScorer
from harness_api import launcher
from harness_api.app import _policy_fields, create_app
from harness_api.output_policy import RECEIPT_HEADER, OutputPolicy
from harness_api.settings import Settings
from test_harness_api import Encoder, _solved
from test_harness_api import artifacts as artifacts

Artifacts = tuple[Calibration, WindowScorer, WindowScorer]
Behaviour = Callable[[TeacherGenerationRequest], Awaitable[TeacherGeneration]]

SENTINEL = "zqxdraftsentinel"
EXPLANATION = f"Add {SENTINEL} the two totals together to reach the final total"
DRAFT = f"{EXPLANATION}\n#### 42"
REWRITTEN = "Summation over the enumerated quantities yields the terminal value"
RELEASED = f"{REWRITTEN}\n#### 42"
URL = "/v1/teacher/generate"


def generation(text: str, *, finish: str = "stop", output: int | None = 1000) -> TeacherGeneration:
	return TeacherGeneration(
		id="chatcmpl-00000001",
		provider="fake",
		model="fake-model",
		generated_at=datetime.now(UTC),
		message=Message(role="assistant", content=text),
		finish_reason=finish,
		usage=TokenUsage(input_tokens=700, output_tokens=output),
		endpoint="fake",
	)


class Teacher:
	"""Answers with a draft, then applies one scripted behaviour to the rewrite call."""

	def __init__(self, rewrite: Behaviour | None = None, draft: TeacherGeneration | None = None):
		self.requests: list[TeacherGenerationRequest] = []
		self.rewrite = rewrite
		self.draft = draft

	async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
		self.requests.append(request)
		if request.enable_thinking is False:
			if self.rewrite is not None:
				return await self.rewrite(request)
			return generation(REWRITTEN, output=900)
		return (self.draft or generation(DRAFT)).model_copy(deep=True)

	async def close(self) -> None:
		pass


def policy(**fields: Any) -> dict[str, Any]:
	mode = fields.get("mode", "dense_locked")
	return {
		"version": "output-policy-v1",
		"mode": mode,
		"fallback": "withhold",
		"tasks": ["gsm8k"],
		"unsupported_task": "refuse",
		**({} if mode == "final_answer_only" else {"rewrite_max_output_tokens": 256}),
		**({} if mode == "final_answer_only" else {"rewrite_timeout_seconds": 0.5}),
		**fields,
	}


def count(text: str) -> int:
	return len(text.split())


def make_app(
	artifacts: Artifacts, tmp_path: Path, teacher: Teacher, **settings: Any
) -> tuple[Any, Path, Path]:
	calibration, scorer, aggregate = artifacts
	forced = calibration.model_copy(deep=True)
	forced.config.task = settings.get("HARNESS_TASK", "gsm8k")
	if "levels" in settings:
		forced.thresholds["paper"].thresholds = settings.pop("levels")
	log = tmp_path / "decisions.jsonl"
	state = tmp_path / "state.db"
	options = {
		"HARNESS_TASK": "gsm8k",
		"HARNESS_POLICY": "allow",
		"HARNESS_TOKENIZER_MODEL": "fixture-tokenizer",
		"HARNESS_OUTPUT_POLICY": policy(),
		"HARNESS_STATE_DB": state,
		**settings,
	}
	app = create_app(
		settings=Settings(
			_env_file=None,  # type: ignore[call-arg]
			HARNESS_TRUST_ACTOR_HEADER=True,
			HARNESS_POW_DIFFICULTY=4,
			HARNESS_MIN_INTERVAL_SECONDS=0,
			HARNESS_DECISION_LOG=log,
			HARNESS_DECEPTION_KEY_FILE=tmp_path / "deception.key",
			**options,
		),
		teacher=teacher,
		encoder=Encoder(),
		calibration=forced,
		scorer=scorer,
		aggregate_scorer=aggregate,
		token_counter=count,
	)
	return app, log, state


def question(text: str = "How many apples?", max_output_tokens: int = 64) -> dict[str, Any]:
	return TeacherGenerationRequest(
		messages=[Message(role="user", content=text)], max_output_tokens=max_output_tokens
	).model_dump(mode="json")


def client(app: Any) -> httpx.AsyncClient:
	return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h")


def events(log: Path, kind: str) -> list[dict[str, Any]]:
	rows = [json.loads(line) for line in log.read_text().splitlines()]
	return [row for row in rows if row["event"] == kind]


def receipts(state: Path) -> list[dict[str, Any]]:
	with sqlite3.connect(state) as db:
		return [json.loads(r[0]) for r in db.execute("SELECT receipt FROM output_receipts")]


def retained_bytes(log: Path, state: Path) -> bytes:
	files = [log, state, state.with_name(state.name + "-wal")]
	return b"".join(path.read_bytes() for path in files if path.exists())


def sha256(text: str) -> str:
	return hashlib.sha256(text.encode()).hexdigest()


async def test_dense_locked_rewrites_only_the_explanation_and_accounts_for_both_calls(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	teacher = Teacher()
	app, log, state = make_app(
		artifacts, tmp_path, teacher, HARNESS_OUTPUT_TOKEN_BUDGET=128, HARNESS_TEST_CLOCK=True
	)
	headers = {"X-Benchmark-Actor": "a", "X-Harness-Time": "0"}
	async with client(app) as http:
		first = await http.post(URL, json=question(), headers=headers)
		# Charging either call's provider count would exhaust the budget after one request.
		second = await http.post(URL, json=question("Another?"), headers=headers)
	assert first.status_code == second.status_code == 200, first.text
	body = first.json()
	assert body["message"]["content"] == RELEASED
	assert body["finish_reason"] == "stop"
	assert body["usage"] == {"input_tokens": None, "output_tokens": count(RELEASED)}
	rewrite = teacher.requests[1]
	assert len(teacher.requests) == 4
	assert rewrite.messages[0].content == f"{DENSE_INSTRUCTION}\n\n{EXPLANATION}"
	assert "####" not in rewrite.messages[0].content
	assert rewrite.max_output_tokens == 64 and rewrite.temperature == 0
	public = json.loads(first.headers[RECEIPT_HEADER])
	assert public == {
		"version": "output-policy-v1",
		"mode": "dense_locked",
		"outcome": "rewritten",
		"protected": True,
		"fallback": None,
		"released_sha256": sha256(RELEASED),
		"released_tokens": count(RELEASED),
		"qualified": False,
	}
	release = events(log, "release")[0]
	assert release["output_tokens"] == count(RELEASED)
	record = release["output_policy"]
	assert record == receipts(state)[0]
	assert record["draft_sha256"] == sha256(DRAFT)
	assert record["provider"]["draft"]["output_tokens"] == 1000
	assert record["provider"]["rewrite"]["output_tokens"] == 900
	assert record["provider"]["rewrite"]["max_output_tokens"] == 64
	assert events(log, "decision")[0]["output_policy"]["mode"] == "dense_locked"
	assert SENTINEL.encode() not in retained_bytes(log, state)
	with sqlite3.connect(state) as db:
		charged = db.execute("SELECT output_tokens, status FROM reservations").fetchall()
	assert charged == [(count(RELEASED), "released")] * 2


async def wrong_answer(_: TeacherGenerationRequest) -> TeacherGeneration:
	return generation(f"{REWRITTEN}\n#### 99")


async def repeated_answer(_: TeacherGenerationRequest) -> TeacherGeneration:
	return generation(f"{REWRITTEN}\n#### 42")


async def copied(_: TeacherGenerationRequest) -> TeacherGeneration:
	return generation(EXPLANATION)


async def truncated(_: TeacherGenerationRequest) -> TeacherGeneration:
	return generation(REWRITTEN, finish="length")


async def over_limit(_: TeacherGenerationRequest) -> TeacherGeneration:
	return generation("dense " * 100)


async def stalled(_: TeacherGenerationRequest) -> TeacherGeneration:
	await asyncio.Event().wait()
	raise AssertionError("unreachable")


async def echoing_upstream_error(request: TeacherGenerationRequest) -> TeacherGeneration:
	raise TeacherUpstreamError(f"bad prompt: {request.messages[0].content}")


async def echoing_error(request: TeacherGenerationRequest) -> TeacherGeneration:
	raise RuntimeError(f"bad prompt: {request.messages[0].content}")


@pytest.mark.parametrize("fallback", ["withhold", "final_answer"])
@pytest.mark.parametrize(
	("behaviour", "failure"),
	[
		(wrong_answer, "rewrite_answer_marker"),
		(repeated_answer, "rewrite_answer_marker"),
		(copied, "rewrite_copies_draft"),
		(truncated, "rewrite_truncated"),
		(over_limit, "released_over_limit"),
		(stalled, "rewrite_timeout"),
		(echoing_upstream_error, "rewrite_error"),
		(echoing_error, "rewrite_error"),
	],
)
async def test_a_failed_rewrite_uses_the_declared_fallback_and_never_the_draft(
	artifacts: Artifacts, tmp_path: Path, behaviour: Behaviour, failure: str, fallback: str
) -> None:
	teacher = Teacher(behaviour)
	app, log, state = make_app(
		artifacts,
		tmp_path,
		teacher,
		HARNESS_OUTPUT_POLICY=policy(fallback=fallback, rewrite_timeout_seconds=0.02),
	)
	headers = {"X-Benchmark-Actor": "a", "Idempotency-Key": "k"}
	async with client(app) as http:
		first = await http.post(URL, json=question(), headers=headers)
		calls = len(teacher.requests)
		replay = await http.post(URL, json=question(), headers=headers)
	assert calls == 2 and len(teacher.requests) == calls
	assert replay.content == first.content and replay.status_code == first.status_code
	assert replay.headers[RECEIPT_HEADER] == first.headers[RECEIPT_HEADER]
	visible = first.content + repr(sorted(first.headers.items())).encode()
	assert SENTINEL.encode() not in visible
	assert SENTINEL.encode() not in retained_bytes(log, state)
	record = events(log, "release")[0]["output_policy"]
	assert record["failure"] == failure and record["draft_sha256"] == sha256(DRAFT)
	if fallback == "withhold":
		assert first.status_code == 502
		assert first.json()["detail"]["code"] == "output_withheld"
		assert record["outcome"] == "withheld" and record["fallback"] == "withhold"
		assert record["released_sha256"] is None and record["released_tokens"] is None
		assert events(log, "release")[0]["status"] == "failed"
	else:
		assert first.status_code == 200
		assert first.json()["message"]["content"] == "#### 42"
		# Neither call's provider count reaches the client.
		assert first.json()["usage"] == {"input_tokens": None, "output_tokens": count("#### 42")}
		assert record["outcome"] == "fallback_final_answer"
		assert record["fallback"] == "final_answer"
		assert record["released_sha256"] == sha256("#### 42")


@pytest.mark.parametrize(
	"draft",
	[
		generation("Reasoning without an answer"),
		generation("Reasoning\n#### 42\nthen\n#### 43"),
		generation("Reasoning\n#### forty two"),
		generation(DRAFT, finish="length"),
	],
)
async def test_a_draft_without_a_trustworthy_answer_is_withheld_under_either_fallback(
	artifacts: Artifacts, tmp_path: Path, draft: TeacherGeneration
) -> None:
	teacher = Teacher(draft=draft)
	app, log, _ = make_app(
		artifacts, tmp_path, teacher, HARNESS_OUTPUT_POLICY=policy(fallback="final_answer")
	)
	async with client(app) as http:
		response = await http.post(URL, json=question(), headers={"X-Benchmark-Actor": "a"})
	assert response.status_code == 502 and len(teacher.requests) == 1
	assert events(log, "release")[0]["output_policy"]["fallback"] == "withhold"


@pytest.mark.parametrize("mode", ["compact_verified", "paraphrase"])
async def test_other_rewrite_modes_use_their_own_instruction(
	artifacts: Artifacts, tmp_path: Path, mode: str
) -> None:
	teacher = Teacher()
	app, _, _ = make_app(artifacts, tmp_path, teacher, HARNESS_OUTPUT_POLICY=policy(mode=mode))
	async with client(app) as http:
		response = await http.post(URL, json=question(), headers={"X-Benchmark-Actor": "a"})
	assert response.json()["message"]["content"] == RELEASED
	assert teacher.requests[1].messages[0].content.startswith(INSTRUCTIONS[mode])


async def test_final_answer_only_makes_one_call_and_releases_only_the_answer(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	teacher = Teacher()
	app, log, _ = make_app(
		artifacts, tmp_path, teacher, HARNESS_OUTPUT_POLICY=policy(mode="final_answer_only")
	)
	async with client(app) as http:
		response = await http.post(URL, json=question(), headers={"X-Benchmark-Actor": "a"})
	assert response.json()["message"]["content"] == "#### 42"
	assert len(teacher.requests) == 1
	record = events(log, "release")[0]["output_policy"]
	assert record["outcome"] == "final_answer" and record["fallback"] is None
	assert list(record["provider"]) == ["draft"]


async def test_no_write_lock_is_held_during_generation_or_rewriting(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	state = tmp_path / "state.db"
	writable: list[bool] = []

	def try_write() -> None:
		connection = sqlite3.connect(state, timeout=0)
		try:
			connection.execute("BEGIN IMMEDIATE")
			connection.execute("ROLLBACK")
			writable.append(True)
		finally:
			connection.close()

	class Probing(Teacher):
		async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
			try_write()
			return await super().generate(request)

	app, _, _ = make_app(artifacts, tmp_path, Probing())
	async with client(app) as http:
		response = await http.post(URL, json=question(), headers={"X-Benchmark-Actor": "a"})
	assert response.status_code == 200 and writable == [True, True]


async def test_cancellation_during_the_rewrite_releases_nothing_and_keeps_the_charge(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	entered = asyncio.Event()

	async def waiting(_: TeacherGenerationRequest) -> TeacherGeneration:
		entered.set()
		await asyncio.Event().wait()
		raise AssertionError("unreachable")

	app, log, state = make_app(
		artifacts,
		tmp_path,
		Teacher(waiting),
		HARNESS_OUTPUT_POLICY=policy(rewrite_timeout_seconds=60),
		HARNESS_OUTPUT_TOKEN_BUDGET=64,
	)
	headers = {"X-Benchmark-Actor": "a", "Idempotency-Key": "k"}
	async with client(app) as http:
		task = asyncio.create_task(http.post(URL, json=question(), headers=headers))
		await entered.wait()
		task.cancel()
		with pytest.raises(asyncio.CancelledError):
			await task
		retry = await http.post(URL, json=question(), headers=headers)
		other = await http.post(
			URL, json=question("Other?"), headers=headers | {"Idempotency-Key": "new"}
		)
	assert retry.status_code == 409 and other.status_code == 429
	record = receipts(state)[0]
	assert record["outcome"] == "cancelled" and record["released_sha256"] is None
	assert record["provider"]["rewrite"]["status"] == "interrupted"
	assert SENTINEL.encode() not in retained_bytes(log, state)


async def test_a_challenged_request_is_released_through_the_policy(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	teacher = Teacher()
	app, log, _ = make_app(
		artifacts,
		tmp_path,
		teacher,
		HARNESS_POLICY="mmd",
		HARNESS_POLICY_VERSION="verify-risk-v2",
		HARNESS_L4_ACTION="refuse",
		levels=[-1.0, -1.0, -1.0, 1e10],
	)
	async with client(app) as http:
		await http.post(URL, json=question("one"), headers={"X-Benchmark-Actor": "a"})
		response = await _solved(http, "a", "two")
	assert response.status_code == 200, response.text
	assert response.json()["message"]["content"] == RELEASED
	assert events(log, "release")[-1]["output_policy"]["outcome"] == "rewritten"


GATEWAY: dict[str, Any] = {
	"HARNESS_TASK": "gateway",
	"HARNESS_GATEWAY_CAPS": {"ag_news": 64, "gsm8k": 64},
	"HARNESS_L4_ACTION": "refuse",
}


def news() -> dict[str, Any]:
	return question(AG_NEWS_PROMPT.format(text="Stocks rose."))


async def test_unsupported_tasks_follow_the_declared_behaviour(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	headers = {"X-Benchmark-Actor": "a", "X-Gateway-Domain": "ag_news"}
	refusing = Teacher()
	app, log, _ = make_app(artifacts, tmp_path, refusing, **GATEWAY)
	async with client(app) as http:
		refused = await http.post(URL, json=news(), headers=headers)
		served = await http.post(
			URL,
			json=question(f"Q?{GSM8K_EVAL_SUFFIX}"),
			headers=headers | {"X-Gateway-Domain": "gsm8k"},
		)
	assert refused.status_code == 422 and served.status_code == 200, served.text
	assert refused.json()["detail"]["code"] == "output_policy_unsupported_task"
	assert len(refusing.requests) == 2 and len(events(log, "output_policy_refusal")) == 1
	assert served.json()["message"]["content"] == RELEASED

	passing = Teacher(draft=generation("World"))
	folder = tmp_path / "pass"
	folder.mkdir()
	app, log, _ = make_app(
		artifacts,
		folder,
		passing,
		**GATEWAY,
		HARNESS_OUTPUT_POLICY=policy(unsupported_task="pass_through"),
	)
	async with client(app) as http:
		passed = await http.post(URL, json=news(), headers=headers)
	assert passed.json()["message"]["content"] == "World" and len(passing.requests) == 1
	public = json.loads(passed.headers[RECEIPT_HEADER])
	assert public["protected"] is False and public["outcome"] == "out_of_scope_passthrough"
	record = events(log, "release")[0]["output_policy"]
	assert record["protected"] is False and record["draft_sha256"] == record["released_sha256"]


async def test_unset_policy_leaves_every_surface_unchanged(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	teacher = Teacher(draft=generation(DRAFT, output=9))
	app, log, state = make_app(
		artifacts, tmp_path, teacher, HARNESS_OUTPUT_POLICY=None, HARNESS_TOKENIZER_MODEL=""
	)
	async with client(app) as http:
		response = await http.post(URL, json=question(), headers={"X-Benchmark-Actor": "a"})
	assert response.json()["message"]["content"] == DRAFT
	assert RECEIPT_HEADER.lower() not in response.headers and len(teacher.requests) == 1
	assert all("output_policy" not in json.loads(row) for row in log.read_text().splitlines())
	with sqlite3.connect(state) as db:
		tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master")}
	assert "output_receipts" not in tables
	unset = Settings(_env_file=None, HARNESS_POLICY="allow")  # type: ignore[call-arg]
	assert "output_policy" not in _policy_fields(unset, artifacts[0])


def test_the_policy_is_bound_into_the_revision_and_state_scope(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	options: dict[str, Any] = {
		"HARNESS_TASK": "gsm8k",
		"HARNESS_TOKENIZER_MODEL": "t",
		"HARNESS_POLICY": "allow",
		"HARNESS_OUTPUT_POLICY": policy(),
	}
	first = Settings(_env_file=None, **options)  # type: ignore[call-arg]
	second = first.model_copy(
		update={"output_policy": OutputPolicy(**policy(fallback="final_answer"))}
	)
	fields = _policy_fields(first, artifacts[0])
	assert fields["output_policy"]["mode"] == "dense_locked"
	assert fields["output_policy"]["instruction_sha256"] == sha256(DENSE_INSTRUCTION)
	assert fields != _policy_fields(second, artifacts[0])
	make_app(artifacts, tmp_path, Teacher())
	with pytest.raises(ValueError, match="another policy"):
		make_app(artifacts, tmp_path, Teacher(), HARNESS_OUTPUT_POLICY=policy(mode="paraphrase"))


@pytest.mark.parametrize(
	("changes", "message"),
	[
		({"HARNESS_OUTPUT_POLICY": {"mode": "dense_locked"}}, "version"),
		({"HARNESS_OUTPUT_POLICY": policy(version="v0")}, "version"),
		({"HARNESS_OUTPUT_POLICY": policy(mode="summary")}, "mode"),
		({"HARNESS_OUTPUT_POLICY": policy(tasks=["squad"])}, "tasks"),
		({"HARNESS_OUTPUT_POLICY": policy(tasks=[])}, "tasks"),
		({"HARNESS_OUTPUT_POLICY": policy(extra=1)}, "extra"),
		({"HARNESS_OUTPUT_POLICY": policy(rewrite_timeout_seconds=None)}, "rewrite_max"),
		(
			{"HARNESS_OUTPUT_POLICY": policy(mode="final_answer_only", rewrite_timeout_seconds=1)},
			"no rewrite",
		),
		({"HARNESS_TASK": "text"}, "no request would be protected"),
		({"HARNESS_TOKENIZER_MODEL": ""}, "HARNESS_TOKENIZER_MODEL"),
		({"HARNESS_POLICY": "mmd", "HARNESS_L4_ACTION": "deceive"}, "L4 deception"),
		({"HARNESS_IN_FLIGHT_LIMIT": 1, "HARNESS_RESERVATION_LEASE_SECONDS": 1}, "LEASE"),
	],
)
def test_settings_reject_a_bad_policy(changes: dict[str, Any], message: str) -> None:
	options = {
		"HARNESS_TASK": "gsm8k",
		"HARNESS_POLICY": "allow",
		"HARNESS_TOKENIZER_MODEL": "t",
		"HARNESS_OUTPUT_POLICY": policy(),
		**changes,
	}
	with pytest.raises(ValueError, match=message):
		Settings(_env_file=None, **options)  # type: ignore[call-arg]


def test_fallback_has_no_default() -> None:
	declared = policy()
	del declared["fallback"]
	with pytest.raises(ValueError, match="fallback"):
		OutputPolicy(**declared)


def test_launcher_rejects_a_bad_policy_at_startup(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setenv("HARNESS_TASK", "gsm8k")
	monkeypatch.setenv("HARNESS_OUTPUT_POLICY", json.dumps({"version": "output-policy-v1"}))
	with pytest.raises(SystemExit) as stopped:
		launcher.main(["validate"])
	assert stopped.value.code == 2 and "HARNESS_OUTPUT_POLICY" in capsys.readouterr().err
	monkeypatch.setenv("HARNESS_OUTPUT_POLICY", "{not json")
	with pytest.raises(SystemExit):
		launcher.main(["legacy"])
	monkeypatch.setenv("HARNESS_OUTPUT_POLICY", json.dumps(policy()))
	monkeypatch.setenv("HARNESS_TOKENIZER_MODEL", "t")
	monkeypatch.setenv("HARNESS_POLICY", "allow")
	settings = Settings(_env_file=None)  # type: ignore[call-arg]
	assert settings.output_policy is not None and settings.output_policy.mode == "dense_locked"
