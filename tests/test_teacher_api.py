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
