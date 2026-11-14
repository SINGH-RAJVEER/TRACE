"""Environment-backed settings for the protected teacher API."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from harness_api.output_policy import OutputPolicy

Tier = Literal["new", "established", "verified"]
TIERS: tuple[Tier, ...] = ("new", "established", "verified")


class Settings(BaseSettings):
	model_config = SettingsConfigDict(env_file=".env", extra="ignore")

	# The frozen artifact from `harness calibrate`.
	calibration_dir: Path = Field(
		default=Path("artifacts/harness"), alias="HARNESS_CALIBRATION_DIR"
	)
	policy: Literal["mmd", "allow", "quota", "quota_tokens", "rate", "path"] = Field(
		default="mmd", alias="HARNESS_POLICY"
	)
	path_calibration: Path | None = Field(default=None, alias="HARNESS_PATH_CALIBRATION")
	accumulator: str = Field(default="paper", alias="HARNESS_ACCUMULATOR")
	thresholds: Literal["calibrated", "ville"] = Field(
		default="calibrated", alias="HARNESS_THRESHOLDS"
	)
	# "direct" replaces the graduated actions with blocking at the L4 threshold (an ablation).
	actions: Literal["graduated", "direct"] = Field(default="graduated", alias="HARNESS_ACTIONS")
	policy_version: Literal["legacy-v1", "verify-risk-v2"] = Field(
		default="legacy-v1", alias="HARNESS_POLICY_VERSION"
	)
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
	# Echoed by /healthz so a launcher can tell its own process from another on the same port.
	instance_id: str | None = Field(default=None, alias="HARNESS_INSTANCE_ID")

	# Durable enforcement state: account horizons, open challenges, and the release ledger.
	# Without it, state lives in this process and a restart renews every account. Use a new
	# file for each arm; a file written under another policy or calibration is rejected.
	state_db: Path | None = Field(default=None, alias="HARNESS_STATE_DB")
	request_quota: int | None = Field(default=None, gt=0, alias="HARNESS_REQUEST_QUOTA")
	output_token_budget: int | None = Field(default=None, gt=0, alias="HARNESS_OUTPUT_TOKEN_BUDGET")
	organization_token_budget: int | None = Field(
		default=None, gt=0, alias="HARNESS_ORGANIZATION_TOKEN_BUDGET"
	)
	in_flight_limit: int | None = Field(default=None, gt=0, alias="HARNESS_IN_FLIGHT_LIMIT")
	reservation_lease_seconds: float = Field(
		default=600, gt=0, alias="HARNESS_RESERVATION_LEASE_SECONDS"
	)
	# Limits over a rolling window, which a policy horizon does not renew: per principal, and
	# per declared organization over the same window.
	rolling_window_seconds: float = Field(
		default=604_800, gt=0, alias="HARNESS_ROLLING_WINDOW_SECONDS"
	)
	rolling_request_budget: int | None = Field(
		default=None, gt=0, alias="HARNESS_ROLLING_REQUEST_BUDGET"
	)
	rolling_token_budget: int | None = Field(
		default=None, gt=0, alias="HARNESS_ROLLING_TOKEN_BUDGET"
	)
	organization_rolling_request_budget: int | None = Field(
		default=None, gt=0, alias="HARNESS_ORGANIZATION_ROLLING_REQUEST_BUDGET"
	)
	organization_rolling_token_budget: int | None = Field(
		default=None, gt=0, alias="HARNESS_ORGANIZATION_ROLLING_TOKEN_BUDGET"
	)

	rolling_supervision_budget: float | None = Field(
		default=None, gt=0, allow_inf_nan=False, alias="HARNESS_ROLLING_SUPERVISION_BUDGET"
	)
	supervision_weights_file: Path | None = Field(
		default=None, alias="HARNESS_SUPERVISION_WEIGHTS_FILE"
	)
	supervision_repeat_factor: float = Field(
		default=1, ge=0, le=1, alias="HARNESS_SUPERVISION_REPEAT_FACTOR"
	)
	trust_tiers: bool = Field(default=False, alias="HARNESS_TRUST_TIERS")
	new_tier_fraction: float = Field(default=0.5, gt=0, le=1, alias="HARNESS_NEW_TIER_FRACTION")
	new_tier_min_age_seconds: float = Field(
		default=604800, ge=0, allow_inf_nan=False, alias="HARNESS_NEW_TIER_MIN_AGE_SECONDS"
	)
	new_tier_min_releases: int = Field(default=10, ge=1, alias="HARNESS_NEW_TIER_MIN_RELEASES")
	# Release detail by trust tier. JSON caps on max_output_tokens per tier, such as
	# {"new": 256, "established": 1024}; an omitted tier is uncapped.
	tier_output_caps: dict[Tier, int] = Field(
		default_factory=dict, alias="HARNESS_TIER_OUTPUT_CAPS"
	)
	# Default-on keyed watermark: each principal's requests carry its key ID to the teacher,
	# which creates its secret file if missing. Unmarked output is withheld when enabled.
	watermark: bool = Field(default=True, alias="HARNESS_WATERMARK")
	watermark_namespace: str = Field(
		default="trace", min_length=1, alias="HARNESS_WATERMARK_NAMESPACE"
	)
	# Opt-in request-path account linkage. The JSON file holds a `linkage-v1` config, and the
	# accounts file may give principals declared "attributes". Inferred clusters then share
	# these rolling limits over HARNESS_ROLLING_WINDOW_SECONDS. Unqualified heuristics.
	linkage_config: Path | None = Field(default=None, alias="HARNESS_LINKAGE_CONFIG")
	cluster_rolling_request_budget: int | None = Field(
		default=None, ge=1, alias="HARNESS_CLUSTER_ROLLING_REQUEST_BUDGET"
	)
	cluster_rolling_token_budget: int | None = Field(
		default=None, ge=1, alias="HARNESS_CLUSTER_ROLLING_TOKEN_BUDGET"
	)
	# Lowest tier that may set temperature, seed, top_p or top_k. Lower tiers get the
	# deterministic default, so resending a prompt cannot collect alternative answers.
	sampling_min_tier: Tier | None = Field(default=None, alias="HARNESS_SAMPLING_MIN_TIER")

	# Review is opt-in and requires a durable state database.
	review_enabled: bool = Field(default=False, alias="HARNESS_REVIEW_ENABLED")
	review_admin_key: str = Field(default="", alias="HARNESS_REVIEW_ADMIN_KEY")
	review_verify_codes: set[str] = Field(default_factory=set, alias="HARNESS_REVIEW_VERIFY_CODES")

	# Opt-in experimental campaign detector. The JSON file holds a `campaign-v2` config. In
	# observe mode it only records evidence; in throttle mode it can refuse requests, which
	# needs the explicit diagnostic override because the score is not calibrated.
	campaign_config: Path | None = Field(default=None, alias="HARNESS_CAMPAIGN_CONFIG")
	campaign_allow_unqualified_enforcement: bool = Field(
		default=False, alias="HARNESS_CAMPAIGN_ALLOW_UNQUALIFIED_ENFORCEMENT"
	)

	# Opt-in JSON: replace the teacher's draft before release. See docs/harness.md.
	output_policy: OutputPolicy | None = Field(default=None, alias="HARNESS_OUTPUT_POLICY")

	# Opt-in whole-campaign evidence monitor from `harness evidence-monitor-calibrate`. It logs
	# fused feature ranks and a path-calibrated alert at each look and never changes a
	# response. It needs the encoder the calibration was fitted with and HARNESS_STATE_DB.
	campaign_evidence: Path | None = Field(default=None, alias="HARNESS_CAMPAIGN_EVIDENCE")
	# Log the shadow scorers from a `calibrate --shadow` artifact on every decision. Telemetry
	# only: no level, action, or response depends on them. Not part of the policy revision.
	shadow_scorers: bool = Field(default=False, alias="HARNESS_SHADOW_SCORERS")

	decision_log: Path = Field(
		default=Path("artifacts/harness/decisions.jsonl"), alias="HARNESS_DECISION_LOG"
	)
	encoder_device: str | None = Field(default=None, alias="HARNESS_ENCODER_DEVICE")
	encoder_batch_size: int = Field(default=64, ge=1, alias="HARNESS_ENCODER_BATCH_SIZE")
	encoder_batch_wait_ms: float = Field(
		default=0, ge=0, le=1000, alias="HARNESS_ENCODER_BATCH_WAIT_MS"
	)
	encoder_queue_capacity: int = Field(default=1024, ge=1, alias="HARNESS_ENCODER_QUEUE_CAPACITY")
	scoring_mode: Literal["synchronous", "off_path"] = Field(
		default="synchronous", alias="HARNESS_SCORING_MODE"
	)
	scoring_queue_capacity: int = Field(default=128, ge=1, alias="HARNESS_SCORING_QUEUE_CAPACITY")
	# Victim tokenizer for recounting output tokens. Without it, output token counts are
	# omitted from every response, so a deceived response cannot be told apart by its count.
	tokenizer_model: str = Field(default="", alias="HARNESS_TOKENIZER_MODEL")
	tokenizer_revision: str = Field(default="", alias="HARNESS_TOKENIZER_REVISION")

	host: str = Field(default="127.0.0.1", alias="HARNESS_HOST")
	port: int = Field(default=8003, ge=1, le=65535, alias="HARNESS_PORT")
	log_level: str = Field(default="info", alias="HARNESS_LOG_LEVEL")

	@model_validator(mode="after")
	def validate_combinations(self) -> Settings:
		if (self.policy == "path") != (self.path_calibration is not None):
			raise ValueError("path policy requires HARNESS_PATH_CALIBRATION, exclusively")
		if self.policy == "path" and self.scoring_mode == "off_path":
			raise ValueError("path stops are incompatible with off_path scoring")
		if self.policy == "path" and self.state_db is None:
			raise ValueError("path stops require HARNESS_STATE_DB for durable campaign state")
		if self.policy == "path":
			if "l4_action" not in self.model_fields_set:
				self.l4_action = "refuse"
			if self.l4_action != "refuse":
				raise ValueError("path policy supports terminal refusal only")
		if self.policy == "quota_tokens" and (
			self.state_db is None or self.output_token_budget is None
		):
			raise ValueError(
				"quota_tokens requires HARNESS_STATE_DB and HARNESS_OUTPUT_TOKEN_BUDGET"
			)
		if self.request_quota is not None and self.policy not in {"quota", "quota_tokens"}:
			raise ValueError("HARNESS_REQUEST_QUOTA requires quota or quota_tokens")
		if self.campaign_evidence is not None and self.state_db is None:
			raise ValueError(
				"HARNESS_CAMPAIGN_EVIDENCE requires HARNESS_STATE_DB for durable state"
			)
		if self.campaign_config is not None and self.state_db is None:
			raise ValueError("HARNESS_CAMPAIGN_CONFIG requires HARNESS_STATE_DB for durable state")
		if self.campaign_allow_unqualified_enforcement and self.campaign_config is None:
			raise ValueError(
				"HARNESS_CAMPAIGN_ALLOW_UNQUALIFIED_ENFORCEMENT requires HARNESS_CAMPAIGN_CONFIG"
			)
		if (self.tier_output_caps or self.sampling_min_tier) and not self.trust_tiers:
			raise ValueError(
				"HARNESS_TIER_OUTPUT_CAPS and HARNESS_SAMPLING_MIN_TIER require HARNESS_TRUST_TIERS"
			)
		if self.linkage_config is not None and self.state_db is None:
			raise ValueError("HARNESS_LINKAGE_CONFIG requires HARNESS_STATE_DB for durable state")
		if (
			self.cluster_rolling_request_budget or self.cluster_rolling_token_budget
		) and self.linkage_config is None:
			raise ValueError("cluster rolling budgets require HARNESS_LINKAGE_CONFIG")
		if any(cap < 1 for cap in self.tier_output_caps.values()):
			raise ValueError("tier output caps must be positive")
		if self.review_enabled and self.state_db is None:
			raise ValueError("review requires HARNESS_STATE_DB")
		if self.review_verify_codes and not self.review_enabled:
			raise ValueError("review verification requires HARNESS_REVIEW_ENABLED")
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
		if (self.rolling_supervision_budget is None) != (self.supervision_weights_file is None):
			raise ValueError("supervision budget and weights file must be configured together")
		if self.output_policy is not None:
			self._validate_output_policy(self.output_policy)
		return self

	def _validate_output_policy(self, output: OutputPolicy) -> None:
		domains = set(self.gateway_caps) if self.task == "gateway" else {self.task}
		if not domains & set(output.tasks):
			raise ValueError(
				f"HARNESS_OUTPUT_POLICY declares {list(output.tasks)}, but this service serves "
				f"{sorted(domains)}, so no request would be protected"
			)
		if not self.tokenizer_model:
			raise ValueError(
				"HARNESS_OUTPUT_POLICY requires HARNESS_TOKENIZER_MODEL: released text is built "
				"by the server, so no teacher usage count describes it"
			)
		if (
			self.policy == "mmd"
			and self.l4_action == "deceive"
			and (self.policy_version == "legacy-v1")
		):
			raise ValueError(
				"HARNESS_OUTPUT_POLICY cannot be combined with L4 deception: noise is shaped "
				"like unmodified teacher output, so it would be told apart; set "
				"HARNESS_L4_ACTION=refuse"
			)
		if (
			self.in_flight_limit is not None
			and self.reservation_lease_seconds
			< self.timeout_seconds + (output.rewrite_timeout_seconds or 0)
		):
			raise ValueError(
				"HARNESS_RESERVATION_LEASE_SECONDS must cover the draft and rewrite timeouts"
			)
