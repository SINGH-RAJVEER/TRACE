"""Environment-backed teacher API settings."""

from pathlib import Path

from distillation.teacher import DEFAULT_REASONING_BUDGET_TOKENS
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	model_config = SettingsConfigDict(env_file=".env", extra="ignore")

	upstream_base_url: str = Field(default="", alias="TEACHER_UPSTREAM_BASE_URL")
	upstream_api_key: str = Field(default="", alias="TEACHER_UPSTREAM_API_KEY")
	upstream_model: str = Field(default="", alias="TEACHER_UPSTREAM_MODEL")
	provider_name: str = Field(default="unknown", alias="TEACHER_PROVIDER_NAME")
	terms_review_id: str = Field(default="", alias="TEACHER_TERMS_REVIEW_ID")
	timeout_seconds: float = Field(default=120.0, gt=0, alias="TEACHER_TIMEOUT_SECONDS")
	# Requests that leave enable_thinking unset reason first. Reasoning is never returned.
	enable_thinking: bool = Field(default=True, alias="TEACHER_ENABLE_THINKING")
	reasoning_budget_tokens: int = Field(
		default=DEFAULT_REASONING_BUDGET_TOKENS, ge=1, alias="TEACHER_REASONING_BUDGET_TOKENS"
	)
	# Keyed unigram watermark for requests that carry a watermark key ID. Off without a key
	# file; the secret never leaves this service and the detector.
	watermark_key_file: Path | None = Field(default=None, alias="TEACHER_WATERMARK_KEY_FILE")
	watermark_gamma: float = Field(default=0.25, gt=0, lt=1, alias="TEACHER_WATERMARK_GAMMA")
	watermark_delta: float = Field(default=2.0, gt=0, le=20, alias="TEACHER_WATERMARK_DELTA")
	watermark_eligible_vocab: int = Field(
		default=32768, ge=256, alias="TEACHER_WATERMARK_ELIGIBLE_VOCAB"
	)
	api_token: str = Field(default="", alias="TEACHER_API_TOKEN")
	host: str = Field(default="127.0.0.1", alias="TEACHER_HOST")
	port: int = Field(default=8000, ge=1, le=65535, alias="TEACHER_PORT")
	log_level: str = Field(default="info", alias="TEACHER_LOG_LEVEL")

	@property
	def is_configured(self) -> bool:
		return bool(self.upstream_base_url and self.upstream_model and self.terms_review_id)
