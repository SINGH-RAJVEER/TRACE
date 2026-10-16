"""Shared request, response, and dataset schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Message(BaseModel):
	"""One structured chat message."""

	role: Literal["system", "user", "assistant", "tool"]
	content: str = Field(min_length=1)

	@field_validator("content")
	@classmethod
	def content_must_not_be_blank(cls, content: str) -> str:
		if not content.strip():
			raise ValueError("message content must not be blank")
		return content


class TokenUsage(BaseModel):
	"""Provider token counts when the upstream API returns them."""

	input_tokens: int | None = Field(default=None, ge=0)
	output_tokens: int | None = Field(default=None, ge=0)


class TeacherGenerationRequest(BaseModel):
	"""Stable inference contract exposed by the teacher API."""

	messages: list[Message] = Field(min_length=1)
	temperature: float = Field(default=0.0, ge=0.0, le=2.0)
	top_p: float | None = Field(default=None, gt=0.0, le=1.0)
	top_k: int | None = Field(default=None, ge=1)
	max_output_tokens: int = Field(default=1024, ge=1, le=32768)
	seed: int | None = None
	enable_thinking: bool | None = None

	@field_validator("messages")
	@classmethod
	def require_user_message(cls, messages: list[Message]) -> list[Message]:
		if not any(message.role == "user" for message in messages):
			raise ValueError("messages must contain at least one user message")
		return messages


class TeacherGeneration(BaseModel):
	"""Provider-neutral teacher response returned to collectors."""

	id: str
	provider: str
	model: str
	generated_at: datetime
	message: Message
	finish_reason: str | None = None
	usage: TokenUsage = Field(default_factory=TokenUsage)
	request_id: str | None = None
	endpoint: str


class PromptSource(BaseModel):
	"""Rights and lineage metadata for one source prompt."""

	model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

	corpus: str = Field(min_length=1)
	source_id: str = Field(min_length=1)
	license_name: str = Field(alias="license", min_length=1)
	content_sha256: str | None = None
	group_id: str | None = Field(default=None, min_length=1)


class PromptRecord(BaseModel):
	"""Input record consumed by the collector."""

	example_id: str = Field(min_length=1)
	split: Literal["train", "validation", "test"]
	task: str = Field(min_length=1)
	source: PromptSource
	prompt: list[Message] = Field(min_length=1)


class TeacherMetadata(BaseModel):
	"""Teacher provenance saved beside a generated sequence."""

	provider: str
	model: str
	endpoint: str
	request_id: str | None = None
	generated_at: datetime
	parameters: dict[str, bool | int | float | str | None]
	terms_review_id: str


class QualityRecord(BaseModel):
	"""Automated collection checks. Human review may replace the reviewer."""

	status: Literal["accepted", "rejected", "pending"]
	checks: list[str]
	reviewer: str
	rejection_reason: str | None = None


class CollectionAttempt(BaseModel):
	"""One call made while collecting a teacher response."""

	attempt_id: str
	attempted_at: datetime
	status: Literal["succeeded", "failed"]
	error_code: str | None = None


class CollectionAttemptLog(BaseModel):
	"""Sidecar record retained even when all retries fail."""

	example_id: str
	attempt: CollectionAttempt


class LedgerRecord(BaseModel):
	"""Immutable JSONL record produced by teacher collection."""

	example_id: str
	attempt_id: str
	split: Literal["train", "validation", "test"]
	task: str
	source: PromptSource
	prompt: list[Message]
	teacher: TeacherMetadata
	completion: list[Message]
	finish_reason: str | None = None
	usage: TokenUsage
	quality: QualityRecord
	attempts: list[CollectionAttempt] = Field(min_length=1)


class SFTRecord(BaseModel):
	"""Conversational prompt-completion record accepted by TRL."""

	model_config = ConfigDict(extra="forbid")

	prompt: list[Message] = Field(min_length=1)
	completion: list[Message] = Field(min_length=1, max_length=1)

	@field_validator("completion")
	@classmethod
	def completion_must_be_assistant(cls, messages: list[Message]) -> list[Message]:
		if messages[0].role != "assistant":
			raise ValueError("the teacher completion must be an assistant message")
		return messages


class ReviewRecord(BaseModel):
	"""Human or task-validator decision stored separately from the raw ledger."""

	example_id: str = Field(min_length=1)
	status: Literal["accepted", "rejected"]
	checks: list[str] = Field(min_length=1)
	reviewer: str = Field(min_length=1)
	reason: str | None = None
