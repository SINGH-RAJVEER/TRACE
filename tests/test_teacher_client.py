from datetime import UTC, datetime

import httpx
import pytest
from distillation.schemas import Message, TeacherGenerationRequest
from distillation.teacher import (
	DEFAULT_REASONING_BUDGET_TOKENS,
	OpenAICompatibleTeacher,
	TeacherReasoningError,
	TeacherUpstreamError,
	withhold_reasoning,
)


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
			# Thinking is the default, so the reasoning allowance rides on the answer cap.
			"max_tokens": 64 + DEFAULT_REASONING_BUDGET_TOKENS,
			"chat_template_kwargs": {"enable_thinking": True},
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
						"message": {
							"role": "assistant",
							"content": "Hello back",
							"reasoning_content": "private chain of thought",
						},
						"finish_reason": "stop",
					}
				],
				"usage": {
					"prompt_tokens": 3,
					"completion_tokens": 12,
					"completion_tokens_details": {"reasoning_tokens": 10},
				},
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
	assert result.usage.output_tokens == 2 and result.usage.reasoning_tokens == 10
	assert result.thinking is True
	assert "private chain of thought" not in result.model_dump_json()
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


@pytest.mark.asyncio
async def test_non_thinking_is_explicit_and_blank_responses_are_retryable() -> None:
	async def handler(request: httpx.Request) -> httpx.Response:
		payload = __import__("json").loads(request.content)
		assert payload["chat_template_kwargs"] == {"enable_thinking": False}
		assert payload["top_p"] == 0.8
		assert payload["top_k"] == 20
		return httpx.Response(
			200,
			json={
				"id": "blank",
				"choices": [{"message": {"content": "   "}, "finish_reason": "stop"}],
			},
		)

	teacher = OpenAICompatibleTeacher(
		base_url="https://teacher.example/v1",
		api_key="",
		model="qwen",
		provider="test",
		transport=httpx.MockTransport(handler),
	)
	try:
		with pytest.raises(TeacherUpstreamError, match="invalid response"):
			await teacher.generate(
				TeacherGenerationRequest(
					messages=[Message(role="user", content="Question")],
					enable_thinking=False,
					top_p=0.8,
					top_k=20,
				)
			)
	finally:
		await teacher.close()


def test_inline_reasoning_is_withheld_and_unseparable_traces_fail_closed() -> None:
	assert withhold_reasoning("<think>\nwork</think>\n\n42") == ("42", True)
	# Templates that open the block in the prompt emit only its closing tag.
	assert withhold_reasoning("work</think>42") == ("42", True)
	# An empty block, as non-thinking templates emit, is not reasoning.
	assert withhold_reasoning("<think>\n\n</think>\n\n42") == ("42", False)
	assert withhold_reasoning("plain answer") == ("plain answer", False)
	with pytest.raises(TeacherReasoningError):
		withhold_reasoning("<think>never closed")


@pytest.mark.asyncio
async def test_spent_reasoning_allowance_is_a_reasoning_failure_with_unknown_usage() -> None:
	replies = iter(
		[
			{"content": "", "reasoning_content": "long trace"},
			{"content": "<think>trace</think>answer"},
		]
	)

	async def handler(request: httpx.Request) -> httpx.Response:
		message = next(replies)
		return httpx.Response(
			200,
			json={
				"id": "x",
				"choices": [
					{
						"message": message,
						"finish_reason": "length" if not message["content"] else "stop",
					}
				],
				"usage": {"completion_tokens": 30},
			},
		)

	teacher = OpenAICompatibleTeacher(
		base_url="https://teacher.example/v1",
		api_key="",
		model="qwen",
		provider="test",
		transport=httpx.MockTransport(handler),
	)
	request = TeacherGenerationRequest(messages=[Message(role="user", content="Question")])
	try:
		with pytest.raises(TeacherReasoningError, match="allowance"):
			await teacher.generate(request)
		result = await teacher.generate(request)
	finally:
		await teacher.close()
	assert result.message.content == "answer"
	# The completion count includes reasoning of unknown length, so nothing is charged as output.
	assert result.usage.output_tokens is None and result.usage.reasoning_tokens is None
