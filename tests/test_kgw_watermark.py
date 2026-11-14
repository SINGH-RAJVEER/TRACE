"""Teacher/vLLM HTTP contract, key separation, configuration and offline KGW detection."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from distillation.schemas import Message, TeacherGenerationRequest
from distillation.teacher import OpenAICompatibleTeacher, TeacherWatermarkError
from distillation.watermark import key_id
from fastapi import FastAPI
from teacher_api.app import create_app
from teacher_api.settings import Settings
from trace_watermark.middleware import KgwMiddleware
from trace_watermark.serving import (
	ACK_HEADER,
	CHALLENGE_HEADER,
	PROCESSOR,
	KgwSpec,
	acknowledgement,
	principal_key,
	request_args,
	validate_request,
)

SECRET = bytes(range(32))  # Public fixture.
SPEC = KgwSpec(512)
ALICE = key_id("trace", "alice")


@pytest.fixture
def engine_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	import os

	monkeypatch.chdir(tmp_path)
	for key in list(os.environ):
		if key.startswith("TEACHER_WATERMARK_"):
			monkeypatch.delenv(key)
	path = tmp_path / "fixture.watermark.key"
	path.write_text(SECRET.hex())
	monkeypatch.setenv("TEACHER_WATERMARK_KEY_FILE", str(path))
	monkeypatch.setenv("TEACHER_WATERMARK_VOCAB_SIZE", "512")
	monkeypatch.setattr("trace_watermark.middleware.version", lambda _: "0.28.0")
	return path


def test_kgw_request_contract_separates_keys_and_refuses_mismatches(engine_env: Path) -> None:
	first = principal_key(SECRET, ALICE, SPEC)
	assert first == principal_key(SECRET, ALICE, SPEC)
	assert first.secret != principal_key(SECRET, key_id("trace", "bob"), SPEC).secret
	assert first.secret != principal_key(bytes(32), ALICE, SPEC).secret
	args = request_args(SECRET, ALICE, SPEC, "0" * 32)
	assert validate_request(args) == (SPEC, first)
	assert SECRET.hex() not in json.dumps(args)
	assert first.secret.hex() not in json.dumps(args)
	assert validate_request(None) is None
	for changed in (
		{"trace_watermark_key": "b" * 32},
		{"trace_watermark_spec": "{}"},
		{"trace_watermark_auth": "wrong"},
		{"trace_watermark_extra": "unknown"},
	):
		with pytest.raises(ValueError):
			validate_request(args | changed)


@pytest.mark.asyncio
@pytest.mark.parametrize("ack", ["valid", "missing", "wrong-key", "replayed"])
async def test_teacher_sends_kgw_args_and_requires_authenticated_ack(ack: str) -> None:
	payloads: list[dict[str, Any]] = []

	def handler(request: httpx.Request) -> httpx.Response:
		payload = json.loads(request.content)
		payloads.append(payload)
		nonce = request.headers[CHALLENGE_HEADER]
		assert payload["vllm_xargs"] == request_args(SECRET, ALICE, SPEC, nonce)
		assert "logit_bias" not in payload
		headers = {}
		if ack != "missing":
			headers[ACK_HEADER] = acknowledgement(
				bytes(32) if ack == "wrong-key" else SECRET,
				"1" * 32 if ack == "replayed" else nonce,
				SPEC,
			)
		return httpx.Response(
			200,
			headers=headers,
			json={"id": "kgw", "choices": [{"message": {"content": "visible answer"}}]},
		)

	teacher = OpenAICompatibleTeacher(
		base_url="http://vllm/v1",
		api_key="",
		model="fixture",
		provider="test",
		transport=httpx.MockTransport(handler),
		watermark=SPEC,
		watermark_secret=SECRET,
	)
	try:
		request = TeacherGenerationRequest(
			messages=[Message(role="user", content="Question")],
			watermark_key=ALICE,
		)
		if ack == "valid":
			assert (await teacher.generate(request)).watermarked is True
		else:
			with pytest.raises(TeacherWatermarkError, match="acknowledge"):
				await teacher.generate(request)
	finally:
		await teacher.close()
	assert len(payloads) == 1  # No fallback to unigram or unmarked generation.


@pytest.mark.asyncio
@pytest.mark.parametrize("processor_loaded", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
async def test_vllm_middleware_refuses_missing_processor(
	engine_env: Path,
	processor_loaded: bool,
	async_scheduling: bool,
) -> None:
	app = FastAPI()
	app.state.vllm_config = SimpleNamespace(
		model_config=SimpleNamespace(
			get_vocab_size=lambda: 512,
			runner_type="generate",
			logits_processors=[PROCESSOR] if processor_loaded else [],
		),
		speculative_config=None,
		scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
	)
	app.add_middleware(KgwMiddleware)

	@app.post("/v1/chat/completions")
	async def completion() -> dict[str, Any]:
		return {"id": "kgw", "choices": [{"message": {"content": "answer"}}]}

	teacher = OpenAICompatibleTeacher(
		base_url="http://vllm/v1",
		api_key="",
		model="fixture",
		provider="test",
		transport=httpx.ASGITransport(app=app),
		watermark=SPEC,
		watermark_secret=SECRET,
	)
	try:
		request = TeacherGenerationRequest(
			messages=[Message(role="user", content="Question")],
			watermark_key=ALICE,
		)
		if processor_loaded and not async_scheduling:
			assert (await teacher.generate(request)).watermarked
		else:
			from distillation.teacher import TeacherUpstreamError

			with pytest.raises(TeacherUpstreamError, match="503"):
				await teacher.generate(request)
	finally:
		await teacher.close()


def test_teacher_config_constructs_kgw_and_requires_vocab(
	engine_env: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	captured: list[dict[str, Any]] = []
	monkeypatch.setattr("teacher_api.app.OpenAICompatibleTeacher", lambda **kw: captured.append(kw))
	settings = Settings(
		TEACHER_UPSTREAM_BASE_URL="http://vllm/v1",
		TEACHER_UPSTREAM_MODEL="fixture",
		TEACHER_TERMS_REVIEW_ID="fixture",
		TEACHER_WATERMARK_SCHEME="kgw-v1",
	)
	create_app(settings=settings)
	assert captured[0]["watermark"] == SPEC
	assert captured[0]["watermark_secret"] == SECRET
	settings.watermark_vocab_size = None
	with pytest.raises(ValueError, match="VOCAB_SIZE"):
		create_app(settings=settings)


def test_kgw_detection_cli_uses_serving_keys_and_preserves_existing_reports(
	engine_env: Path,
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	import numpy as np
	from distillation.watermark_cli import app
	from trace_watermark.bias import apply_bias
	from typer.testing import CliRunner

	history = [11, 23]
	generated: list[int] = []
	rng = np.random.default_rng(7)
	for _ in range(256):
		logits = rng.standard_normal(512)
		biased = apply_bias(logits, history, principal_key(SECRET, ALICE, SPEC), SPEC.profile())
		token = int(np.argmax(biased))
		generated.append(token)
		history.append(token)
	texts = tmp_path / "suspect.jsonl"
	texts.write_text(json.dumps({"text": "fixture output"}) + "\n")
	monkeypatch.setattr("distillation.watermark.hf_tokenizer", lambda *_: lambda _: generated)
	report_path = tmp_path / "report.json"
	arguments = [
		"detect-kgw",
		"--texts",
		str(texts),
		"--key-file",
		str(engine_env),
		"--principal",
		"alice",
		"--principal",
		"bob",
		"--tokenizer",
		"fixture",
		"--tokenizer-revision",
		"pinned",
		"--vocab-size",
		"512",
		"--output",
		str(report_path),
	]
	runner = CliRunner()
	result = runner.invoke(app, arguments)
	assert result.exit_code == 0, result.output
	report = json.loads(report_path.read_text())
	assert report["scheme"] == "kgw-v1" and report["spec_digest"] == SPEC.digest
	assert [row["decision"] for row in report["results"]] == ["detected", "not_detected"]
	assert SECRET.hex() not in report_path.read_text()
	original = report_path.read_bytes()
	assert runner.invoke(app, arguments).exit_code != 0
	assert report_path.read_bytes() == original
