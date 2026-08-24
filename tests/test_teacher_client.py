from datetime import UTC, datetime

import httpx
import pytest
from distillation.schemas import Message, TeacherGenerationRequest
from distillation.teacher import OpenAICompatibleTeacher, TeacherUpstreamError


@pytest.mark.asyncio
async def test_openai_compatible_teacher_maps_request_and_response() -> None:
	async def handler(request: httpx.Request) -> httpx.Response:
		assert request.url == "https://teacher.example/v1/chat/completions"
		assert request.headers["authorization"] == "Bearer secret"
		payload = __import__("json").loads(request.content)
		assert payload == {
			"model": "teacher-snapshot",
			"messages": [{"role": "user", "content": "Hello"}],
			"temperature": 0.0,
			"max_tokens": 64,
			"seed": 7,
		}
		return httpx.Response(
			200,
			headers={"x-request-id": "provider-request"},
			json={
				"id": "completion-1",
				"model": "teacher-snapshot",
				"choices": [
					{
						"message": {"role": "assistant", "content": "Hello back"},
						"finish_reason": "stop",
					}
				],
				"usage": {"prompt_tokens": 3, "completion_tokens": 2},
			},
		)

	teacher = OpenAICompatibleTeacher(
		base_url="https://teacher.example/v1",
		api_key="secret",
		model="teacher-snapshot",
		provider="approved-provider",
		transport=httpx.MockTransport(handler),
	)
	result = await teacher.generate(
		TeacherGenerationRequest(
			messages=[Message(role="user", content="Hello")],
			max_output_tokens=64,
			seed=7,
		)
	)
	await teacher.close()

	assert result.message.content == "Hello back"
	assert result.provider == "approved-provider"
	assert result.request_id == "provider-request"
	assert result.endpoint == "https://teacher.example/v1/chat/completions"
	assert result.usage.input_tokens == 3
	assert result.generated_at <= datetime.now(UTC)


@pytest.mark.asyncio
async def test_openai_compatible_teacher_hides_upstream_body_on_error() -> None:
	async def handler(_: httpx.Request) -> httpx.Response:
		return httpx.Response(401, text="secret provider detail")

	teacher = OpenAICompatibleTeacher(
		base_url="https://teacher.example/v1",
		api_key="secret",
		model="teacher-snapshot",
		provider="approved-provider",
		transport=httpx.MockTransport(handler),
	)
	with pytest.raises(TeacherUpstreamError, match="HTTP 401") as captured:
		await teacher.generate(
			TeacherGenerationRequest(messages=[Message(role="user", content="Hello")])
		)
	await teacher.close()
	assert "secret provider detail" not in str(captured.value)
