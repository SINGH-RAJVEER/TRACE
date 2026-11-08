"""Environment-backed dashboard settings. Relative paths resolve from the working directory."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	model_config = SettingsConfigDict(env_file=".env", extra="ignore")

	roots: str = Field(default="readings,artifacts,paper/figures", alias="DASHBOARD_ROOTS")
	index: str = Field(default="readings/index.tsv", alias="DASHBOARD_INDEX")
	frontend: str = Field(default="apps/dashboard/dist", alias="DASHBOARD_FRONTEND")
	host: str = Field(default="127.0.0.1", alias="DASHBOARD_HOST")
	port: int = Field(default=8002, ge=1, le=65535, alias="DASHBOARD_PORT")
	log_level: str = Field(default="info", alias="DASHBOARD_LOG_LEVEL")
