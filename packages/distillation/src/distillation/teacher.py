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


class Teacher(Protocol):
	"""Small interface used by the API and collector tests."""

	async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration: ...

	async def close(self) -> None: ...


class TeacherUpstreamError(RuntimeError):
	"""A safe error raised when hosted inference fails."""


class _UpstreamMessage(BaseModel):
	role: str = "assistant"
	content: str


class _UpstreamChoice(BaseModel):
	message: _UpstreamMessage
	finish_reason: str | None = None


class _UpstreamUsage(BaseModel):
	prompt_tokens: int | None = Field(default=None, ge=0)
	completion_tokens: int | None = Field(default=None, ge=0)


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
	) -> None:
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

	async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
		payload: dict[str, object] = {
			"model": self._model,
			"messages": [message.model_dump() for message in request.messages],
			"temperature": request.temperature,
			"max_tokens": request.max_output_tokens,
		}
		if request.seed is not None:
			payload["seed"] = request.seed

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
		usage = upstream.usage or _UpstreamUsage()
		return TeacherGeneration(
			id=upstream.id,
			provider=self._provider,
			model=upstream.model or self._model,
			generated_at=datetime.now(UTC),
			message=Message(role="assistant", content=choice.message.content),
			finish_reason=choice.finish_reason,
			usage=TokenUsage(
				input_tokens=usage.prompt_tokens,
				output_tokens=usage.completion_tokens,
			),
			request_id=_request_id(response.headers),
			endpoint=self._endpoint,
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
