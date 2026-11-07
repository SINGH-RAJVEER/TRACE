"""Clients for hosted and locally wrapped teacher inference APIs."""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError

from distillation.schemas import (
	Message,
	TeacherGeneration,
	TeacherGenerationRequest,
	TokenUsage,
)
from distillation.watermark import WatermarkSpec, logit_bias


class Teacher(Protocol):
	"""Small interface used by the API and collector tests."""

	async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration: ...

	async def close(self) -> None: ...


class TeacherUpstreamError(RuntimeError):
	"""A safe error raised when hosted inference fails."""


class TeacherReasoningError(TeacherUpstreamError):
	"""The answer could not be released without its reasoning.

	Raised when a trace cannot be cleanly separated from the answer, or when the teacher spent
	its whole allowance reasoning and produced no answer. Never retried as a blank response.
	"""


class TeacherWatermarkError(TeacherUpstreamError):
	"""A watermark was requested from a teacher without one configured."""


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
# Server-side reasoning allowance added on top of the client's visible-answer cap.
DEFAULT_REASONING_BUDGET_TOKENS = 2048


def withhold_reasoning(content: str) -> tuple[str, bool]:
	"""Return the visible answer and whether nonblank reasoning text was removed.

	Handles inline traces from servers without a reasoning parser, including templates that
	open the block in the prompt and emit only its closing tag. Fails closed: any marker left
	after the last closing tag means the trace cannot be separated, so nothing is released.
	"""
	answer = content
	reasoned = False
	if THINK_CLOSE in content:
		trace, _, answer = content.rpartition(THINK_CLOSE)
		reasoned = bool(trace.replace(THINK_OPEN, "").replace(THINK_CLOSE, "").strip())
		answer = answer.lstrip()
	if THINK_OPEN in answer or THINK_CLOSE in answer:
		raise TeacherReasoningError("teacher reasoning could not be separated from its answer")
	return answer, reasoned


class _UpstreamMessage(BaseModel):
	role: str = "assistant"
	content: str | None = None
	# llama-server and vLLM reasoning parsers return the trace here. It is read only to
	# detect that reasoning happened and is never returned.
	reasoning_content: str | None = None
	reasoning: str | None = None


class _UpstreamChoice(BaseModel):
	message: _UpstreamMessage
	finish_reason: str | None = None


class _UpstreamCompletionDetails(BaseModel):
	reasoning_tokens: int | None = Field(default=None, ge=0)


class _UpstreamUsage(BaseModel):
	prompt_tokens: int | None = Field(default=None, ge=0)
	completion_tokens: int | None = Field(default=None, ge=0)
	completion_tokens_details: _UpstreamCompletionDetails | None = None


class _UpstreamResponse(BaseModel):
	id: str
	model: str | None = None
	choices: list[_UpstreamChoice] = Field(min_length=1)
	usage: _UpstreamUsage | None = None


class OpenAICompatibleTeacher:
	"""Calls a hosted provider implementing the chat-completions protocol."""

	def __init__(
		self,
		*,
		base_url: str,
		api_key: str,
		model: str,
		provider: str,
		timeout_seconds: float = 120.0,
		transport: httpx.AsyncBaseTransport | None = None,
		default_thinking: bool = True,
		reasoning_budget_tokens: int = DEFAULT_REASONING_BUDGET_TOKENS,
		watermark: WatermarkSpec | None = None,
		watermark_secret: bytes | None = None,
	) -> None:
		if (watermark is None) != (watermark_secret is None):
			raise ValueError("a watermark needs both a spec and a secret")
		if reasoning_budget_tokens < 1:
			raise ValueError("the reasoning budget must be at least one token")
		headers = {"Accept": "application/json"}
		if api_key:
			headers["Authorization"] = f"Bearer {api_key}"
		self._client = httpx.AsyncClient(
			base_url=f"{base_url.rstrip('/')}/",
			headers=headers,
			timeout=timeout_seconds,
			transport=transport,
		)
		self._model = model
		self._provider = provider
		self._endpoint = str(self._client.base_url.join("chat/completions"))
		self._default_thinking = default_thinking
		self._reasoning_budget = reasoning_budget_tokens
		self._watermark = watermark
		self._watermark_secret = watermark_secret

	async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
		thinking = (
			self._default_thinking if request.enable_thinking is None else request.enable_thinking
		)
		payload: dict[str, object] = {
			"model": self._model,
			"messages": [message.model_dump() for message in request.messages],
			"temperature": request.temperature,
			# Upstream servers count reasoning inside max_tokens, so the visible-answer cap
			# alone would leave nothing for the answer.
			"max_tokens": request.max_output_tokens + (self._reasoning_budget if thinking else 0),
			"chat_template_kwargs": {"enable_thinking": thinking},
		}
		if request.seed is not None:
			payload["seed"] = request.seed
		if request.top_p is not None:
			payload["top_p"] = request.top_p
		if request.top_k is not None:
			payload["top_k"] = request.top_k
		if request.watermark_key is not None:
			if self._watermark is None or self._watermark_secret is None:
				raise TeacherWatermarkError("this teacher has no watermark configured")
			payload["logit_bias"] = logit_bias(
				self._watermark_secret, request.watermark_key, self._watermark
			)

		try:
			response = await self._client.post("chat/completions", json=payload)
			response.raise_for_status()
			upstream = _UpstreamResponse.model_validate(response.json())
		except httpx.HTTPStatusError as error:
			raise TeacherUpstreamError(
				f"teacher returned HTTP {error.response.status_code}"
			) from error
		except (httpx.HTTPError, ValidationError, ValueError) as error:
			raise TeacherUpstreamError("teacher returned an invalid response") from error

		choice = upstream.choices[0]
		answer, inline = withhold_reasoning(choice.message.content or "")
		separate = choice.message.reasoning_content or choice.message.reasoning or ""
		reasoned = inline or bool(separate.strip())
		if not answer.strip():
			if reasoned and choice.finish_reason == "length":
				raise TeacherReasoningError("teacher exhausted its reasoning allowance")
			raise TeacherUpstreamError("teacher returned an invalid response")
		usage = upstream.usage or _UpstreamUsage()
		details = usage.completion_tokens_details
		reasoning_tokens = details.reasoning_tokens if details else None
		output_tokens = usage.completion_tokens
		if reasoning_tokens is not None and output_tokens is not None:
			output_tokens = max(0, output_tokens - reasoning_tokens)
		elif reasoned:
			# The completion count includes reasoning of unknown length. Leave the visible
			# count unknown rather than charge reasoning as released output.
			output_tokens = None
		return TeacherGeneration(
			id=upstream.id,
			provider=self._provider,
			model=upstream.model or self._model,
			generated_at=datetime.now(UTC),
			message=Message(role="assistant", content=answer),
			finish_reason=choice.finish_reason,
			usage=TokenUsage(
				input_tokens=usage.prompt_tokens,
				output_tokens=output_tokens,
				reasoning_tokens=reasoning_tokens,
			),
			request_id=_request_id(response.headers),
			endpoint=self._endpoint,
			thinking=thinking,
			watermarked=request.watermark_key is not None,
		)

	async def close(self) -> None:
		await self._client.aclose()


class TeacherAPIGateway:
	"""Calls the repository's provider-neutral teacher inference API."""

	def __init__(
		self,
		*,
		base_url: str,
		token: str = "",
		timeout_seconds: float = 120.0,
		transport: httpx.AsyncBaseTransport | None = None,
	) -> None:
		headers = {"Accept": "application/json"}
		if token:
			headers["Authorization"] = f"Bearer {token}"
		self._client = httpx.AsyncClient(
			base_url=f"{base_url.rstrip('/')}/",
			headers=headers,
			timeout=timeout_seconds,
			transport=transport,
		)

	async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
		try:
			response = await self._client.post(
				"v1/teacher/generate",
				json=request.model_dump(mode="json"),
			)
			response.raise_for_status()
			return TeacherGeneration.model_validate(response.json())
		except httpx.HTTPStatusError as error:
			raise TeacherUpstreamError(
				f"teacher API returned HTTP {error.response.status_code}"
			) from error
		except (httpx.HTTPError, ValidationError, ValueError) as error:
			raise TeacherUpstreamError("teacher API returned an invalid response") from error

	async def close(self) -> None:
		await self._client.aclose()


def _request_id(headers: Mapping[str, str]) -> str | None:
	for name in ("x-request-id", "request-id", "openai-request-id"):
		if value := headers.get(name):
			return value
	return None
