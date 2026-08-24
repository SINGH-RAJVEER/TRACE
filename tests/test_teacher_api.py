from datetime import UTC, datetime

import httpx
import pytest
from distillation.schemas import (
	Message,
	TeacherGeneration,
	TeacherGenerationRequest,
	TokenUsage,
)
from teacher_api.app import create_app
from teacher_api.settings import Settings


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
async def test_teacher_api_requires_configured_bearer_token() -> None:
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
	async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
		unauthorized = await client.post(
			"/v1/teacher/generate",
			json={"messages": [{"role": "user", "content": "Question"}]},
		)
		response = await client.post(
			"/v1/teacher/generate",
			headers={"Authorization": "Bearer local-secret"},
			json={"messages": [{"role": "user", "content": "Question"}]},
		)

	assert unauthorized.status_code == 401
	assert response.status_code == 200
	assert response.json()["message"]["content"] == "A response"
	assert teacher.requests[0].temperature == 0.0


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
