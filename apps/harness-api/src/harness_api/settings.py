"""Environment-backed settings for the protected teacher API."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	model_config = SettingsConfigDict(env_file=".env", extra="ignore")

	# The frozen artifact from `harness calibrate`.
	calibration_dir: Path = Field(
		default=Path("artifacts/harness"), alias="HARNESS_CALIBRATION_DIR"
	)
	policy: Literal["mmd", "allow", "quota", "rate"] = Field(default="mmd", alias="HARNESS_POLICY")
	accumulator: str = Field(default="paper", alias="HARNESS_ACCUMULATOR")
	thresholds: Literal["calibrated", "ville"] = Field(
		default="calibrated", alias="HARNESS_THRESHOLDS"
	)
	# "direct" replaces the graduated actions with blocking at the L4 threshold (an ablation).
	actions: Literal["graduated", "direct"] = Field(default="graduated", alias="HARNESS_ACTIONS")
	l4_action: Literal["refuse", "deceive"] = Field(default="deceive", alias="HARNESS_L4_ACTION")
	deception_rate: float = Field(default=1.0, ge=0, le=1, alias="HARNESS_DECEPTION_RATE")
	# Hex secret, at least 16 bytes. Without it, the key file is read, or created on first start.
	deception_key: str = Field(default="", alias="HARNESS_DECEPTION_KEY")
	deception_key_file: Path = Field(
		default=Path("artifacts/harness/deception.key"), alias="HARNESS_DECEPTION_KEY_FILE"
	)
	# The endpoint's output format for noise: ag_news, squad, gsm8k, or text for general chat.
	task: Literal["ag_news", "squad", "gsm8k", "coding", "text", "gateway"] = Field(
		default="text", alias="HARNESS_TASK"
	)
	gateway_caps: dict[str, int] = Field(default_factory=dict, alias="HARNESS_GATEWAY_CAPS")
	pow_difficulty: int = Field(default=20, ge=1, le=64, alias="HARNESS_POW_DIFFICULTY")
	challenge_ttl_seconds: float = Field(default=300, gt=0, alias="HARNESS_CHALLENGE_TTL_SECONDS")
	min_interval_seconds: float = Field(default=1.0, ge=0, alias="HARNESS_MIN_INTERVAL_SECONDS")

	# The teacher inference API this service protects.
	teacher_url: str = Field(default="http://127.0.0.1:8000", alias="HARNESS_TEACHER_URL")
	teacher_token: str = Field(default="", alias="HARNESS_TEACHER_TOKEN")
	timeout_seconds: float = Field(default=120.0, gt=0, alias="HARNESS_TIMEOUT_SECONDS")

	# JSON object mapping the SHA-256 hex of each API key to its account ID.
	accounts_file: Path | None = Field(default=None, alias="HARNESS_ACCOUNTS_FILE")
	# Test only: take the account from X-Benchmark-Actor, which any client can set.
	trust_actor_header: bool = Field(default=False, alias="HARNESS_TRUST_ACTOR_HEADER")
	# Test only: take the time from X-Harness-Time and never wait in real time.
	test_clock: bool = Field(default=False, alias="HARNESS_TEST_CLOCK")

	decision_log: Path = Field(
		default=Path("outputs/harness/decisions.jsonl"), alias="HARNESS_DECISION_LOG"
	)
	encoder_device: str | None = Field(default=None, alias="HARNESS_ENCODER_DEVICE")
	# Victim tokenizer for recounting output tokens. Without it, output token counts are
	# omitted from every response, so a deceived response cannot be told apart by its count.
	tokenizer_model: str = Field(default="", alias="HARNESS_TOKENIZER_MODEL")
	tokenizer_revision: str = Field(default="", alias="HARNESS_TOKENIZER_REVISION")

	host: str = Field(default="127.0.0.1", alias="HARNESS_HOST")
	port: int = Field(default=8003, ge=1, le=65535, alias="HARNESS_PORT")
	log_level: str = Field(default="info", alias="HARNESS_LOG_LEVEL")

	@model_validator(mode="after")
	def coding_l4(self) -> Settings:
		if self.task in {"coding", "gateway"} and "l4_action" not in self.model_fields_set:
			self.l4_action = "refuse"
		if self.task in {"coding", "gateway"} and self.l4_action != "refuse":
			raise ValueError("coding and gateway modes support L4 refusal only")
		if self.task == "gateway" and (
			len(self.gateway_caps) < 2
			or set(self.gateway_caps) - {"ag_news", "squad", "gsm8k", "coding"}
			or any(cap < 1 for cap in self.gateway_caps.values())
		):
			raise ValueError(
				"gateway mode requires positive caps for at least two supported domains"
			)
		return self
