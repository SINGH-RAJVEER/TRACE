from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from distillation.schemas import (
	Message,
	TeacherGeneration,
	TeacherGenerationRequest,
	TokenUsage,
)
from distillation.teacher import TeacherUpstreamError
from distillation.watermark import WatermarkSpec, read_master_secret
from teacher_api.app import create_app
from teacher_api.settings import Settings


@pytest.fixture(autouse=True)
def isolated_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)


class FakeTeacher:
	def __init__(self) -> None:
		self.requests: list[TeacherGenerationRequest] = []
		self.closed = False

	async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
		self.requests.append(request)
		return TeacherGeneration(
			id="generation-1",
			provider="fake",
			model="fake-model",
			generated_at=datetime.now(UTC),
			message=Message(role="assistant", content="A response"),
			finish_reason="stop",
			usage=TokenUsage(input_tokens=4, output_tokens=2),
			endpoint="https://teacher.example/v1/chat/completions",
		)

	async def close(self) -> None:
		self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("authorization", [None, "Bearer wrong-key", "local-secret", "Bearer "])
async def test_teacher_api_requires_configured_bearer_token(authorization: str | None) -> None:
	teacher = FakeTeacher()
	app = create_app(
		settings=Settings(
			TEACHER_UPSTREAM_BASE_URL="",
			TEACHER_UPSTREAM_MODEL="",
			TEACHER_TERMS_REVIEW_ID="",
			TEACHER_API_TOKEN="local-secret",
		),
		teacher=teacher,
	)
	transport = httpx.ASGITransport(app=app)
	async with (
		app.router.lifespan_context(app),
		httpx.AsyncClient(transport=transport, base_url="http://test") as client,
	):
		unauthorized = await client.post(
			"/v1/teacher/generate",
			headers={"Authorization": authorization} if authorization is not None else {},
			json={"messages": [{"role": "user", "content": "Question"}]},
		)
		assert unauthorized.status_code == 401
		assert teacher.requests == []
		payload = {
			"messages": [{"role": "user", "content": "Question"}],
			"temperature": 0.7,
			"top_p": 0.8,
			"top_k": 20,
			"seed": 42,
			"max_output_tokens": 64,
			"enable_thinking": False,
		}
		response = await client.post(
			"/v1/teacher/generate",
			headers={"Authorization": "Bearer local-secret"},
			json=payload,
		)

	assert response.status_code == 200
	assert response.json()["message"]["content"] == "A response"
	assert teacher.requests == [TeacherGenerationRequest.model_validate(payload)]
	assert teacher.closed


@pytest.mark.asyncio
async def test_teacher_failure_is_a_safe_502_and_lifespan_closes_the_client() -> None:
	class UnavailableTeacher(FakeTeacher):
		async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
			raise TeacherUpstreamError("teacher returned HTTP 429")

	teacher = UnavailableTeacher()
	app = create_app(settings=Settings(TEACHER_API_TOKEN=""), teacher=teacher)
	async with (
		app.router.lifespan_context(app),
		httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
	):
		response = await client.post(
			"/v1/teacher/generate",
			json={"messages": [{"role": "user", "content": "Question"}]},
		)
	assert response.status_code == 502
	assert response.json() == {
		"detail": {"code": "teacher_upstream_error", "message": "teacher returned HTTP 429"}
	}
	assert teacher.closed


@pytest.mark.asyncio
async def test_unconfigured_teacher_reports_service_unavailable() -> None:
	app = create_app(
		settings=Settings(
			TEACHER_UPSTREAM_BASE_URL="",
			TEACHER_UPSTREAM_MODEL="",
			TEACHER_TERMS_REVIEW_ID="",
			TEACHER_API_TOKEN="",
		)
	)
	transport = httpx.ASGITransport(app=app)
	async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
		health = await client.get("/healthz")
		response = await client.post(
			"/v1/teacher/generate",
			json={"messages": [{"role": "user", "content": "Question"}]},
		)

	assert health.json() == {"status": "unconfigured"}
	assert response.status_code == 503


def test_unconfigured_teacher_does_not_create_a_watermark_key(tmp_path: Path) -> None:
	settings = Settings(
		TEACHER_UPSTREAM_BASE_URL="",
		TEACHER_UPSTREAM_MODEL="",
		TEACHER_TERMS_REVIEW_ID="",
		TEACHER_WATERMARK_KEY_FILE=tmp_path / "unconfigured.watermark.key",
	)
	create_app(settings=settings)
	assert not settings.watermark_key_file.exists()


@pytest.mark.parametrize("custom_path", [False, True])
def test_configured_teacher_creates_and_reuses_its_watermark_key(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, custom_path: bool
) -> None:
	captured: list[dict[str, object]] = []

	def adapter(**kwargs: object) -> FakeTeacher:
		captured.append(kwargs)
		return FakeTeacher()

	monkeypatch.setattr("teacher_api.app.OpenAICompatibleTeacher", adapter)
	settings = Settings(
		TEACHER_UPSTREAM_BASE_URL="http://localhost:8001/v1",
		TEACHER_UPSTREAM_MODEL="test-model",
		TEACHER_TERMS_REVIEW_ID="test-review",
	)
	if custom_path:
		settings.watermark_key_file = tmp_path / "private" / "custom.watermark.key"
	else:
		assert settings.watermark_key_file == Path("artifacts/teacher.watermark.key")
	assert not settings.watermark_key_file.exists()
	create_app(settings=settings)
	first = read_master_secret(settings.watermark_key_file)
	create_app(settings=settings)
	assert len(captured) == 2
	assert all(call["watermark_secret"] == first for call in captured)
	assert all(isinstance(call["watermark"], WatermarkSpec) for call in captured)
	assert read_master_secret(settings.watermark_key_file) == first


def test_injected_teacher_does_not_create_a_watermark_key(tmp_path: Path) -> None:
	settings = Settings(
		TEACHER_UPSTREAM_BASE_URL="http://localhost:8001/v1",
		TEACHER_UPSTREAM_MODEL="test-model",
		TEACHER_TERMS_REVIEW_ID="test-review",
		TEACHER_WATERMARK_KEY_FILE=tmp_path / "injected.watermark.key",
	)
	create_app(settings=settings, teacher=FakeTeacher())
	assert not settings.watermark_key_file.exists()
