"""Environment-backed teacher API settings."""

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
	api_token: str = Field(default="", alias="TEACHER_API_TOKEN")
	host: str = Field(default="127.0.0.1", alias="TEACHER_HOST")
	port: int = Field(default=8000, ge=1, le=65535, alias="TEACHER_PORT")
	log_level: str = Field(default="info", alias="TEACHER_LOG_LEVEL")

	@property
	def is_configured(self) -> bool:
		return bool(self.upstream_base_url and self.upstream_model and self.terms_review_id)
