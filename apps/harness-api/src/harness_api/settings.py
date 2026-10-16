"""Environment-backed settings for the protected teacher API."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	model_config = SettingsConfigDict(env_file=".env", extra="ignore")

	# The frozen artifact from `harness calibrate`.
	calibration_dir: Path = Field(default=Path("artifacts/harness"), alias="HARNESS_CALIBRATION_DIR")
	policy: Literal["mmd", "allow", "quota", "rate"] = Field(default="mmd", alias="HARNESS_POLICY")
	accumulator: str = Field(default="paper", alias="HARNESS_ACCUMULATOR")
	thresholds: Literal["calibrated", "ville"] = Field(
		default="calibrated", alias="HARNESS_THRESHOLDS"
	)
	l4_action: Literal["refuse", "deceive"] = Field(default="refuse", alias="HARNESS_L4_ACTION")
	deception_rate: float = Field(default=1.0, ge=0, le=1, alias="HARNESS_DECEPTION_RATE")
	# Hex secret for deception and watermark tests; at least 16 bytes.
	deception_key: str = Field(default="", alias="HARNESS_DECEPTION_KEY")
	# The task this endpoint serves: ag_news, squad, or gsm8k. Required for deception.
	task: str = Field(default="", alias="HARNESS_TASK")
	# GSM8K deceptions are rewritten by a second teacher call with its own output limit.
	gsm8k_rewrite: bool = Field(default=True, alias="HARNESS_GSM8K_REWRITE")
	rewrite_max_output_tokens: int = Field(default=512, ge=1, alias="HARNESS_REWRITE_MAX_TOKENS")
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
