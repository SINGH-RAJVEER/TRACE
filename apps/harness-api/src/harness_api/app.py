"""FastAPI service that applies the extraction harness in front of the teacher API.

It serves the teacher API's contract, POST /v1/teacher/generate, so collectors, evaluations,
and the traffic benchmark can point at it unchanged. Every response carries
X-Harness-Decision: allow (L0 and L1), throttle (L2 and L3, including answered challenges),
or block (L4 refusals). An L3 response is HTTP 428 with a proof-of-work challenge; the client
resends the same request with X-Harness-Proof: NONCE:COUNTER. An account at L4 is refused by
default. With HARNESS_L4_ACTION=deceive it instead keeps receiving L3-shaped responses whose
content is keyed noise, built without calling the teacher.

Admission and release are separate events. A `decision` event records each admission. Before
any output is produced, the admission is reserved in the release ledger, and a `release` event
records how the reservation settled: released, failed, or cancelled. With HARNESS_STATE_DB,
account horizons, open challenges, the ledger, and aggregate monitoring survive restarts.
The noise response template is local to each worker.
"""

import asyncio
import hashlib
import hmac
import json
import math
import os
import random
import secrets
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from distillation.io import canonical_json
from distillation.schemas import (
	Message,
	TeacherGeneration,
	TeacherGenerationRequest,
	TokenUsage,
)
from distillation.teacher import Teacher, TeacherAPIGateway, TeacherUpstreamError
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from harness import review
from harness.calibration import Calibration, load_aggregate, load_calibration, load_shadow
from harness.campaign import (
	CampaignConfig,
	CampaignError,
	campaign_cooldown,
	campaign_key,
	load_campaign_config,
	observe_campaign,
	prepare_session_observation,
	text_identity,
)
from harness.deception import noise
from harness.encoder import Encoder
from harness.mmd import WindowScorer
from harness.monitor import AggregateMonitor, AggregateWindow
from harness.path_runtime import RuntimePathCalibration, encoder_contract
from harness.policy import (
	Decision,
	MMDPolicy,
	PathStopPolicy,
	Policy,
	ScoreWindow,
	StalePathAdmission,
	request_digest,
)
from harness.replay import make_policy
from harness.serialization import detector_text, gateway_turn, valid_gateway_turn
from harness.shadow import ShadowMonitor
from harness.store import (
	IdempotencyRecord,
	MemoryStore,
	Outcome,
	Reservation,
	RollingBudget,
	Settlement,
	SQLiteStore,
	StateStore,
	SupervisionBudget,
	TierBudget,
	Transaction,
	admission_key,
	reserve_entitlement,
	rolling_refusal,
)
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.types import Message as ASGIMessage

from harness_api.batching import EncoderBatcher
from harness_api.output_policy import (
	RECEIPT_HEADER,
	OutputAttempt,
	ProviderCall,
	apply_policy,
)
from harness_api.settings import Settings


class ScoringUnavailable(Exception):
	"""The off-path scorer cannot accept a window, so the admission must not commit."""


class CampaignObservationFailed(Exception):
	"""The campaign observation was refused, so its attempt rolls back and runs without it."""


def current_turn(request: TeacherGenerationRequest) -> str:
	"""The latest user message: the paper embeds the current turn only."""
	return next(m.content for m in reversed(request.messages) if m.role == "user")


def decision_header(decision: Decision) -> str:
	if decision.action == "refuse":
		return "block"
	return "allow" if decision.level < 2 else "throttle"


ROLLING_REFUSALS = {
	"supervision_budget_exceeded": (429, "supervision budget is used up", "block"),
	"tier_budget_exceeded": (429, "tier budget is used up", "block"),
	"rolling_budget_exceeded": (429, "the account's rolling budget is used up", "block"),
	"organization_rolling_budget_exceeded": (
		429,
		"the organization's rolling budget is used up",
		"block",
	),
}


def _error(status: int, code: str, message: str, decision: str | None) -> JSONResponse:
	"""An error body; decision is None for failures the harness did not decide."""
	return JSONResponse(
		{"detail": {"code": code, "message": message}},
		status_code=status,
		headers={"X-Harness-Decision": decision} if decision else None,
	)


class DecisionLog:
	"""Experimenter-only JSON lines: decisions, deceptions, and aggregate alerts.

	Requests appear as SHA-256 digests, never as text.
	"""

	def __init__(self, path: Path) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		self.path = path

	def write(self, event: str, **fields: Any) -> None:
		with self.path.open("a", encoding="utf-8") as sink:
			sink.write(json.dumps({"event": event, **fields}) + "\n")


def _key(settings: Settings) -> bytes | None:
	"""The deception secret: HARNESS_DECEPTION_KEY, else the key file, else a new key there.

	A persisted key keeps noise consistent across restarts and keeps watermark tests possible.
	"""
	if settings.l4_action != "deceive" or settings.policy_version == "verify-risk-v2":
		return None
	if settings.deception_key:
		key = bytes.fromhex(settings.deception_key)
	elif settings.deception_key_file.exists():
		key = bytes.fromhex(settings.deception_key_file.read_text().strip())
	else:
		key = secrets.token_bytes(32)
		settings.deception_key_file.parent.mkdir(parents=True, exist_ok=True)
		descriptor = os.open(
			settings.deception_key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
		)
		with os.fdopen(descriptor, "w") as sink:
			sink.write(key.hex() + "\n")
	if len(key) < 16:
		raise ValueError("the deception key must hold at least 16 bytes")
	return key


def _mimic(identifier: str | None) -> str | None:
	"""A fresh identifier shaped like a real one: same prefix, random hex of the same length."""
	if identifier is None:
		return None
	prefix, _, suffix = identifier.rpartition("-")
	head = f"{prefix}-" if prefix else ""
	tail = suffix or identifier
	return head + secrets.token_hex((len(tail) + 1) // 2)[: len(tail)]


def _policy_fields(
	settings: Settings,
	calibration: Calibration,
	path: RuntimePathCalibration | None = None,
	campaign: CampaignConfig | None = None,
) -> dict[str, Any]:
	"""The policy, its actions and limits, and the calibration behind a decision."""
	fields = settings.model_dump(
		include={
			"policy",
			"accumulator",
			"thresholds",
			"actions",
			"l4_action",
			"deception_rate",
			"task",
			"gateway_caps",
			"pow_difficulty",
			"challenge_ttl_seconds",
			"min_interval_seconds",
			"output_token_budget",
			"organization_token_budget",
			"in_flight_limit",
			"reservation_lease_seconds",
			"tokenizer_model",
			"tokenizer_revision",
		}
	)
	if settings.request_quota is not None:
		fields["request_quota"] = settings.request_quota
	# Omitted for the default version so existing legacy state files keep their scope.
	if settings.policy_version != "legacy-v1":
		fields["policy_version"] = settings.policy_version
	if path is not None:
		fields["path_calibration"] = path.input_hash
		fields["path_policy_version"] = path.recipe.version
		fields["path_qualified"] = False
	# Likewise omitted when unset, so a file written without rolling limits keeps its scope.
	if settings.rolling_request_budget or settings.rolling_token_budget:
		fields["rolling"] = [
			settings.rolling_window_seconds,
			settings.rolling_request_budget,
			settings.rolling_token_budget,
		]
	if settings.organization_rolling_request_budget or settings.organization_rolling_token_budget:
		fields["organization_rolling"] = [
			settings.rolling_window_seconds,
			settings.organization_rolling_request_budget,
			settings.organization_rolling_token_budget,
		]
	if settings.rolling_supervision_budget is not None:
		assert settings.supervision_weights_file is not None
		fields["supervision"] = [
			settings.rolling_window_seconds,
			settings.rolling_supervision_budget,
			settings.supervision_repeat_factor,
			hashlib.sha256(settings.supervision_weights_file.read_bytes()).hexdigest(),
		]
	if settings.trust_tiers:
		fields["trust_tiers"] = [
			settings.new_tier_fraction,
			settings.new_tier_min_age_seconds,
			settings.new_tier_min_releases,
			hashlib.sha256(settings.accounts_file.read_bytes()).hexdigest()
			if settings.accounts_file
			else None,
		]
	# Present only when the campaign detector is configured, so other state keeps its scope.
	if campaign is not None:
		fields["campaign"] = {
			"config_sha256": campaign.digest,
			"unqualified_enforcement": settings.campaign_allow_unqualified_enforcement,
		}
	# Likewise omitted when unset, so the default revision and state scope do not change.
	if settings.output_policy is not None:
		fields["output_policy"] = settings.output_policy.binding()
	fields["calibration"] = hashlib.sha256(calibration.model_dump_json().encode()).hexdigest()
	return fields


def _scope(
	settings: Settings,
	calibration: Calibration,
	scorer: WindowScorer,
	aggregate_scorer: WindowScorer | None,
	deception_key: bytes | None,
	path: RuntimePathCalibration | None = None,
	campaign: CampaignConfig | None = None,
) -> str:
	"""Everything that gives stored state its meaning: the policy, its actions, and the
	calibration. Secrets and service addresses are left out."""
	fields = _policy_fields(settings, calibration, path, campaign)
	fields["references"] = hashlib.sha256(scorer.references.tobytes()).hexdigest()
	fields["scorer"] = [scorer.sigma, scorer.scales]
	fields["deception_key"] = hashlib.sha256(deception_key).hexdigest() if deception_key else None
	fields["aggregate_references"] = (
		hashlib.sha256(aggregate_scorer.references.tobytes()).hexdigest()
		if aggregate_scorer is not None
		else None
	)
	fields["accounts"] = (
		hashlib.sha256(settings.accounts_file.read_bytes()).hexdigest()
		if settings.accounts_file
		else None
	)
	return canonical_json(fields)


def _accounts(settings: Settings) -> dict[str, tuple[str, str | None]]:
	if settings.accounts_file is None:
		return {}
	mapping = json.loads(settings.accounts_file.read_text())
	result: dict[str, tuple[str, str | None]] = {}
	for digest, entry in mapping.items():
		if isinstance(entry, str):
			result[digest.lower()] = (entry, None)
		else:
			result[digest.lower()] = (entry["principal"], entry.get("organization"))
	organizations: dict[str, str | None] = {}
	for principal, organization in result.values():
		if principal in organizations and organizations[principal] != organization:
			raise ValueError("a principal must map to one organization")
		organizations[principal] = organization
	return result


def create_app(
	*,
	settings: Settings | None = None,
	teacher: Teacher | None = None,
	encoder: Encoder | None = None,
	calibration: Calibration | None = None,
	scorer: WindowScorer | None = None,
	aggregate_scorer: WindowScorer | None = None,
	token_counter: Callable[[str], int] | None = None,
	clock: Callable[[], float] = time.time,
) -> FastAPI:
	"""Build the service. Tests inject the teacher, encoder, calibration, and clock."""
	config = settings or Settings()
	accounts = _accounts(config)
	if not accounts and not config.trust_actor_header:
		raise ValueError("configure HARNESS_ACCOUNTS_FILE, or HARNESS_TRUST_ACTOR_HEADER for tests")
	key = _key(config)
	if calibration is None or scorer is None:
		calibration, scorer = load_calibration(config.calibration_dir)
		if aggregate_scorer is None:
			aggregate_scorer = load_aggregate(config.calibration_dir, calibration)
	# The detector serialization follows the calibration, so both must name the served task.
	served_task = calibration.config.task
	if served_task != config.task:
		raise ValueError(f"HARNESS_TASK is {config.task}, but the calibration is for {served_task}")
	path = (
		RuntimePathCalibration.model_validate_json(config.path_calibration.read_text())
		if config.path_calibration is not None
		else None
	)
	if path is not None:
		if encoder is None:
			from harness.encoder import BGEEncoder

			encoder = BGEEncoder(device=config.encoder_device, batch_size=config.encoder_batch_size)
		path.require_compatible(calibration, scorer, encoder.identity())
	campaign = (
		load_campaign_config(config.campaign_config) if config.campaign_config is not None else None
	)
	campaign_needs_embeddings = campaign is not None and (
		campaign.coverage is not None
		or (campaign.session is not None and campaign.session.encoder_identity is not None)
	)
	if campaign_needs_embeddings:
		if encoder is None:
			from harness.encoder import BGEEncoder

			encoder = BGEEncoder(device=config.encoder_device, batch_size=config.encoder_batch_size)
		assert campaign is not None
		identities = []
		if campaign.session is not None and campaign.session.encoder_identity is not None:
			identities.append(campaign.session.encoder_identity)
		if campaign.coverage is not None:
			identities.append(campaign.coverage.monitor.encoder_identity)
		if any(encoder_contract(encoder.identity()) != identity for identity in identities):
			raise ValueError("campaign encoder representation differs from configured identity")
	if (
		campaign is not None
		and campaign.coverage is not None
		and campaign.coverage.monitor.serialization
		!= ("gateway_turn" if config.task == "gateway" else "raw_turn")
	):
		raise ValueError("coverage serialization differs from the served task's encoder input")
	if (
		campaign is not None
		and campaign.mode == "throttle"
		and not config.campaign_allow_unqualified_enforcement
	):
		raise ValueError(
			"campaign throttle mode is experimental and unqualified; set "
			"HARNESS_CAMPAIGN_ALLOW_UNQUALIFIED_ENFORCEMENT to enforce it for diagnosis"
		)
	if config.scoring_mode == "off_path" and config.state_db is not None:
		# Queued windows live in this process, so a restart would silently drop their evidence.
		raise ValueError("off-path scoring keeps queued windows in memory; unset HARNESS_STATE_DB")
	store: StateStore = (
		SQLiteStore(
			config.state_db,
			_scope(config, calibration, scorer, aggregate_scorer, key, path, campaign),
			review_enabled=config.review_enabled,
		)
		if config.state_db is not None
		else MemoryStore()
	)
	if campaign is not None:
		if "campaign_throttled" in config.review_verify_codes:
			raise ValueError("campaign throttling cannot be converted to verification")
		store.bind_campaign(campaign.digest)
	output_policy = config.output_policy
	if output_policy is not None:
		store.enable_output_receipts()
	# Binds every logged event to the exact policy and calibration behind it.
	policy_revision = hashlib.sha256(
		canonical_json(_policy_fields(config, calibration, path, campaign)).encode()
	).hexdigest()
	path_fields = (
		{
			"path_policy": {
				"calibration_hash": path.input_hash,
				"policy_version": path.recipe.version,
				"qualified": False,
			}
		}
		if path is not None
		else {}
	)
	# Present only with an output policy, so other decision logs keep their schema.
	event_fields = {
		**path_fields,
		**(output_policy.event_fields() if output_policy is not None else {}),
	}
	rolling = (
		RollingBudget(
			config.rolling_window_seconds,
			config.rolling_request_budget,
			config.rolling_token_budget,
		)
		if config.rolling_request_budget or config.rolling_token_budget
		else None
	)
	organization_rolling = (
		RollingBudget(
			config.rolling_window_seconds,
			config.organization_rolling_request_budget,
			config.organization_rolling_token_budget,
			scope="organization",
		)
		if config.organization_rolling_request_budget or config.organization_rolling_token_budget
		else None
	)
	supervision = None
	if config.rolling_supervision_budget is not None:
		assert config.supervision_weights_file is not None
		data = json.loads(config.supervision_weights_file.read_text())
		weights = data.get("weights")
		domains = set(config.gateway_caps) if config.task == "gateway" else {config.task}
		if (
			data.get("version") != 1
			or not isinstance(weights, dict)
			or not domains <= weights.keys()
			or any(
				isinstance(value, bool)
				or not isinstance(value, (int, float))
				or not math.isfinite(value)
				or value <= 0
				for value in weights.values()
			)
		):
			raise ValueError(
				"weights must be version 1 with positive finite weights for every domain"
			)
		supervision = SupervisionBudget(
			config.rolling_window_seconds,
			config.rolling_supervision_budget,
			weights,
			config.supervision_repeat_factor,
		)
	tier = None
	if config.trust_tiers:
		tiers: dict[str, str] = {}
		contracts: dict[str, RollingBudget] = {}
		entries = (
			json.loads(config.accounts_file.read_text()).values() if config.accounts_file else []
		)
		for entry in entries:
			if isinstance(entry, str):
				entry = {"principal": entry}
			principal = entry["principal"]
			level = entry.get("tier", "new")
			if level not in {"new", "established", "verified"} or (
				principal in tiers and tiers[principal] != level
			):
				raise ValueError("invalid or inconsistent principal tier")
			tiers[principal] = level
			contract = entry.get("contractual_rolling_budget")
			if contract is not None:
				org = entry.get("organization")
				if (
					level != "verified"
					or not org
					or not isinstance(contract, dict)
					or set(contract) - {"requests", "tokens"}
					or not contract
					or any(type(v) is not int or v <= 0 for v in contract.values())
				):
					raise ValueError(
						"contract requires a verified organization and positive limits"
					)
				budget = RollingBudget(
					config.rolling_window_seconds,
					contract.get("requests"),
					contract.get("tokens"),
					scope="organization",
				)
				if org in contracts and contracts[org] != budget:
					raise ValueError("inconsistent organization contract")
				contracts[org] = budget
		tier = TierBudget(
			config.new_tier_fraction,
			config.new_tier_min_age_seconds,
			config.new_tier_min_releases,
			tiers,
			contracts,
		)
	queue: asyncio.Queue[ScoreWindow | None] = asyncio.Queue(maxsize=config.scoring_queue_capacity)
	score_worker: asyncio.Task[None] | None = None
	score_failure: list[str] = []
	score_metrics: dict[int, dict[str, Any]] = {}
	score_accounts: dict[int, str] = {}
	inflight: dict[int, str] = {}
	next_release_id = 0

	def submit_window(work: ScoreWindow) -> None:
		nonlocal score_worker
		if score_failure:
			raise ScoringUnavailable("off-path scorer unavailable")
		if not config.test_clock:
			work = replace(work, closed_at=clock())
		try:
			queue.put_nowait(work)
		except asyncio.QueueFull as error:
			score_failure.append("queue_full")
			raise ScoringUnavailable("off-path scoring queue is full") from error
		score_metrics[work.sequence] = {
			"admissions": 0,
			"released_responses": 0,
			"released_tokens": 0,
			"unknown_token_responses": 0,
			"inflight_at_closure": sum(a == work.account for a in inflight.values()),
		}
		score_accounts[work.sequence] = work.account
		if score_worker is None:
			score_worker = asyncio.create_task(run_scores())

	policy: Policy
	if config.policy == "mmd":
		policy = MMDPolicy(
			calibration,
			scorer,
			accumulator=config.accumulator,
			thresholds=config.thresholds,
			min_interval=config.min_interval_seconds,
			difficulty=config.pow_difficulty,
			challenge_ttl=config.challenge_ttl_seconds,
			l4_action=config.l4_action,
			deception_rate=config.deception_rate,
			deception_key=key,
			actions=config.actions,
			store=store,
			policy_version=config.policy_version,
			submit_window=submit_window if config.scoring_mode == "off_path" else None,
		)
	elif path is not None:
		policy = PathStopPolicy(calibration, scorer, path, store=store)
	else:
		if config.scoring_mode == "off_path":
			raise ValueError("off-path scoring requires the MMD policy")
		policy = make_policy(
			"quota" if config.policy == "quota_tokens" else config.policy,
			calibration,
			scorer,
			store=store,
			request_quota=config.request_quota,
		)
	monitor = (
		AggregateMonitor(aggregate_scorer, calibration.aggregate)
		if config.policy != "path"
		and aggregate_scorer is not None
		and calibration.aggregate is not None
		else None
	)
	base_needs_embeddings = config.policy in {"mmd", "path"} or monitor is not None
	needs_embeddings = base_needs_embeddings or campaign_needs_embeddings
	shadow: ShadowMonitor | None = None
	if config.shadow_scorers:
		if calibration.shadow is None:
			raise ValueError("HARNESS_SHADOW_SCORERS needs a calibration fitted with --shadow")
		if calibration.shadow._centroids is None:
			load_shadow(config.calibration_dir, calibration)
		shadow = ShadowMonitor(
			calibration.shadow,
			task=served_task,
			window_size=calibration.config.window_size,
			scales=calibration.config.scales,
			distinct_prompts=calibration.config.distinct_prompts,
		)
	if encoder is None and (needs_embeddings or shadow is not None):
		from harness.encoder import BGEEncoder

		encoder = BGEEncoder(device=config.encoder_device, batch_size=config.encoder_batch_size)
	if token_counter is None and config.tokenizer_model:
		from transformers import AutoTokenizer

		tokenizer = AutoTokenizer.from_pretrained(
			config.tokenizer_model, revision=config.tokenizer_revision or None
		)

		def token_counter(text: str) -> int:
			return len(tokenizer(text, add_special_tokens=False)["input_ids"])

	upstream = teacher or TeacherAPIGateway(
		base_url=config.teacher_url,
		token=config.teacher_token,
		timeout_seconds=config.timeout_seconds,
	)
	batcher = (
		EncoderBatcher(
			encoder,
			batch_size=config.encoder_batch_size,
			wait_seconds=config.encoder_batch_wait_ms / 1000,
			capacity=config.encoder_queue_capacity,
		)
		if encoder is not None and needs_embeddings and config.encoder_batch_wait_ms > 0
		else None
	)
	log = DecisionLog(config.decision_log)
	if encoder is not None and needs_embeddings:
		log.write(
			"encoder_config",
			device=getattr(encoder, "device", "unknown"),
			batch_size=config.encoder_batch_size,
			batch_wait_ms=config.encoder_batch_wait_ms,
			queue_capacity=config.encoder_queue_capacity,
			identity=encoder.identity(),
		)
	# Orders off-path score application with release accounting; the store orders admissions.
	lock = asyncio.Lock()
	# Real responses teach the shape and timing that noise responses copy.
	template: list[TeacherGeneration] = []
	latencies: deque[float] = deque(maxlen=256)

	async def run_scores() -> None:
		assert isinstance(policy, MMDPolicy)
		while (work := await queue.get()) is not None:
			try:
				if score_failure:
					raise RuntimeError("an earlier score failed")
				score = await asyncio.to_thread(policy.score_window, work)
				async with lock:
					with store.transaction() as tx:
						current = tx.account(work.account)
						previous_level = (
							current.level
							if current is not None and current.horizon_start == work.horizon_start
							else None
						)
						state = policy.apply_window(tx, work, score)
					applied = state is not None
					level = state.level if state is not None else None
					metrics = score_metrics.pop(work.sequence)
					score_accounts.pop(work.sequence)
					applied_at = clock()
					escalated = (
						level is not None and previous_level is not None and level > previous_level
					)
					log.write(
						"off_path_score",
						account=work.account,
						sequence=work.sequence,
						horizon_sequence=work.horizon_sequence,
						horizon_start=work.horizon_start,
						closed_at=work.closed_at,
						applied_at=applied_at,
						enforced_at=applied_at if escalated else None,
						delay_seconds=max(0, applied_at - work.closed_at),
						applied=applied,
						escalated=escalated,
						previous_level=previous_level,
						level=level,
						reservations=None,
						total_released_tokens=(
							None
							if metrics["unknown_token_responses"]
							else metrics["released_tokens"]
						),
						inflight_at_enforcement=sum(a == work.account for a in inflight.values()),
						**metrics,
					)
			except Exception as error:
				score_failure.append(type(error).__name__)
				metrics = score_metrics.pop(work.sequence, {})
				score_accounts.pop(work.sequence, None)
				log.write(
					"off_path_score_error",
					account=work.account,
					sequence=work.sequence,
					horizon_sequence=work.horizon_sequence,
					closed_at=work.closed_at,
					**metrics,
					error=type(error).__name__,
				)
			finally:
				queue.task_done()

	def observe_shadow(
		decision: Decision, text: str, embedding: Any, digest: str
	) -> dict[str, Any] | None:
		"""Shadow scores after the admission committed, without yielding to the event loop, so
		a cancellation cannot fall between the commit and the decision log or release. A
		failure is logged, never raised, so the shadow scorers cannot change any response."""
		assert shadow is not None
		if embedding is None:
			return None
		try:
			return shadow.observe(
				decision.account, decision.horizon, text, embedding, request_digest(text)
			)
		except Exception as error:
			log.write(
				"shadow_error",
				account=decision.account,
				request_digest=digest,
				error=type(error).__name__,
			)
			return None

	@asynccontextmanager
	async def lifespan(_: FastAPI) -> AsyncIterator[None]:
		yield
		if batcher is not None:
			await batcher.close()
		if score_worker is not None:
			await queue.put(None)
			await score_worker
		await upstream.close()
		if isinstance(store, SQLiteStore):
			store.close()

	app = FastAPI(title="TRACE protected teacher API", version="0.1.0", lifespan=lifespan)

	class ReleaseAccounting:
		def __init__(self, app: ASGIApp) -> None:
			self.app = app

		async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
			if scope["type"] != "http":
				await self.app(scope, receive, send)
				return
			state = scope.setdefault("state", {})

			async def counted_send(message: ASGIMessage) -> None:
				if message["type"] == "http.response.body" and not message.get("more_body", False):
					async with lock:
						await send(message)
						if (account := state.get("release_account")) is not None:
							for sequence, metrics in score_metrics.items():
								if score_accounts[sequence] == account:
									metrics["released_responses"] += 1
									tokens = state.get("release_tokens")
									if tokens is None:
										metrics["unknown_token_responses"] += 1
									else:
										metrics["released_tokens"] += tokens
				else:
					await send(message)

			try:
				await self.app(scope, receive, counted_send)
			finally:
				async with lock:
					if (release_id := state.get("release_id")) is not None:
						inflight.pop(release_id, None)

	app.add_middleware(ReleaseAccounting)

	def now_from(header: str | None) -> float:
		if config.test_clock and header is not None:
			return float(header)
		return clock()

	def resolve(authorization: str | None, actor: str | None) -> tuple[str, str | None] | None:
		if authorization and authorization.startswith("Bearer ") and accounts:
			digest = hashlib.sha256(authorization.removeprefix("Bearer ").encode()).hexdigest()
			if digest in accounts:
				return accounts[digest]
		if config.trust_actor_header and actor:
			return actor, None
		return None

	def previous(
		tx: Transaction,
		account: str,
		key: str | None,
		digest: str,
		*,
		proof: bool = False,
		now: float | None = None,
	) -> JSONResponse | None:
		if key is None:
			return None
		record = tx.idempotency(account, key)
		if record is None:
			return None
		if record.request_digest != digest:
			return _error(409, "idempotency_conflict", "key belongs to another request", None)
		if record.outcome is None:
			return _error(
				409, "idempotency_pending", "request is pending or its outcome is unknown", None
			)
		if campaign is not None and now is not None:
			body = json.loads(record.outcome.body)
			if body.get("detail", {}).get("code") == "campaign_throttled":
				until = body.get("campaign", {}).get("throttle_until")
				if isinstance(until, int | float) and now >= until:
					return None
				if isinstance(until, int | float):
					remaining = max(1, math.ceil(until - now))
					body["campaign"]["retry_after_seconds"] = remaining
					headers = {
						k: v
						for k, v in record.outcome.headers.items()
						if k.lower() not in {"content-length", "retry-after"}
					}
					headers["Retry-After"] = str(remaining)
					return JSONResponse(
						body, status_code=record.outcome.status_code, headers=headers
					)
		if proof and record.outcome.status_code == 428:
			return None
		return JSONResponse(
			json.loads(record.outcome.body),
			status_code=record.outcome.status_code,
			headers=record.outcome.headers,
		)

	def review_response(
		tx: Transaction,
		response: JSONResponse,
		account: str,
		organization: str | None,
		digest: str,
		now: float,
		code: str,
		horizon: float | None = None,
		reset_at: float | None = None,
		scope: str = "account",
		window: float | None = None,
	) -> JSONResponse:
		if not config.review_enabled:
			return response
		assert hasattr(tx, "db")
		prior_appeal = None
		campaign_state = None
		if code == "campaign_throttled" and campaign is not None:
			campaign_state = tx.campaign(campaign_key(campaign, account))
			if campaign_state is not None:
				cached = campaign_state.get("api_appeal", {})
				if cached.get("until") == reset_at:
					prior_appeal = cached.get("ref")
		ref = prior_appeal or review.open_appeal(
			tx.db,
			account=account,
			organization=organization,
			code=code,
			scope=scope,
			reset_at=reset_at,
			horizon=horizon,
			window_seconds=window,
			now=now,
			digest=digest,
		)
		if campaign_state is not None and prior_appeal is None and campaign is not None:
			campaign_state["api_appeal"] = {"until": reset_at, "ref": ref}
			tx.put_campaign(campaign_key(campaign, account), campaign_state)
		body = json.loads(bytes(response.body))
		body["detail"].update(
			{
				"limit_code": code,
				"scope": "principal" if scope == "account" else scope,
				"reset_at": reset_at,
				"appeal_ref": ref,
			}
		)
		status = response.status_code
		headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
		if code in config.review_verify_codes and status != 428:
			status = 428
			body["detail"]["message"] = "verification review required"
			body["action"] = "verify"
			headers["x-harness-decision"] = "throttle"
		return JSONResponse(body, status_code=status, headers=headers)

	def cooldown_until(
		tx: Transaction, account: str, organization: str | None, now: float
	) -> float | None:
		assert campaign is not None
		return campaign_cooldown(tx, campaign, account, now)

	def throttle_response(
		tx: Transaction,
		account: str,
		organization: str | None,
		digest: str,
		now: float,
		until: float,
	) -> JSONResponse:
		retry_after = max(1, math.ceil(until - now))
		response = JSONResponse(
			{
				"detail": {
					"code": "campaign_throttled",
					"message": "experimental campaign throttle is active for this account",
				},
				"campaign": {
					"qualified": False,
					"mode": "throttle",
					"throttle_until": until,
					"retry_after_seconds": retry_after,
				},
			},
			status_code=429,
			headers={"X-Harness-Decision": "throttle", "Retry-After": str(retry_after)},
		)
		return review_response(
			tx,
			response,
			account,
			organization,
			digest,
			now,
			"campaign_throttled",
			reset_at=until,
		)

	def reserve(
		tx: Transaction,
		decision: Decision,
		request: TeacherGenerationRequest,
		digest: str,
		now: float,
		organization: str | None,
		idem: str | None,
		proof: bool = False,
		domain: str = "text",
	) -> tuple[Reservation | None, JSONResponse | None]:
		old = previous(tx, decision.account, idem, digest, proof=proof, now=now)
		if old is not None:
			return None, old
		row = Reservation(
			account=decision.account,
			horizon_start=decision.horizon,
			attempt=decision.attempt,
			request_digest=digest,
			reserved_at=now,
			deceive=decision.deceive,
			reserved_tokens=request.max_output_tokens,
			organization=organization,
			domain=domain,
			clean=decision.level == 0,
			expires_at=now + config.reservation_lease_seconds
			if config.in_flight_limit is not None
			else None,
		)
		reserved_row, reason = reserve_entitlement(
			tx,
			row,
			now=now,
			horizon_seconds=calibration.config.horizon_seconds,
			principal_budget=config.output_token_budget,
			organization_budget=config.organization_token_budget,
			in_flight_limit=config.in_flight_limit,
			rolling=rolling,
			organization_rolling=organization_rolling,
			supervision=supervision,
			tier=tier,
		)
		if reason is not None:
			messages = {
				"organization_required": (403, "account has no organization", "block"),
				"token_budget_exceeded": (429, "output token entitlement exhausted", "block"),
				**ROLLING_REFUSALS,
				"in_flight_limit": (429, "too many requests in flight", "throttle"),
				"already_released": (409, "this admission was already released", None),
			}
			status_code, message, decision_header_value = messages[reason]
			response = _error(status_code, reason, message, decision_header_value)
			scope = (
				"organization" if reason == "organization_rolling_budget_exceeded" else "account"
			)
			org_horizon = tx.organization(organization) if organization else None
			if (
				reason == "token_budget_exceeded"
				and organization
				and config.organization_token_budget is not None
			):
				org_start = (
					org_horizon
					if org_horizon is not None
					and now - org_horizon < calibration.config.horizon_seconds
					else now
				)
				used = tx.exposure("organization", organization, org_start, now)
				credit = tx.credits("organization", organization, "horizon", now, org_start, None)
				principal_used = tx.exposure("account", decision.account, decision.horizon, now)
				principal_credit = tx.credits(
					"account", decision.account, "horizon", now, decision.horizon, None
				)
				principal_exceeded = config.output_token_budget is not None and (
					principal_used.tokens + request.max_output_tokens
					> config.output_token_budget + principal_credit.tokens
				)
				if not principal_exceeded and (
					used.tokens + request.max_output_tokens
					> config.organization_token_budget + credit.tokens
				):
					scope = "organization"
				org_horizon = org_start
			window = config.rolling_window_seconds if "rolling_budget" in reason else None
			horizon = (
				(org_horizon if scope == "organization" else decision.horizon)
				if reason == "token_budget_exceeded"
				else None
			)
			reset = horizon + calibration.config.horizon_seconds if horizon is not None else None
			return None, review_response(
				tx,
				response,
				decision.account,
				organization,
				digest,
				now,
				reason,
				horizon,
				reset,
				scope,
				window,
			)
		assert reserved_row is not None
		if idem is not None:
			tx.put_idempotency(IdempotencyRecord(decision.account, idem, digest, now))
		return reserved_row, None

	def remember(
		tx: Transaction,
		account: str,
		idem: str | None,
		digest: str,
		now: float,
		response: JSONResponse,
	) -> None:
		if idem is not None:
			tx.put_idempotency(
				IdempotencyRecord(
					account,
					idem,
					digest,
					now,
					Outcome(
						response.status_code, bytes(response.body).decode(), dict(response.headers)
					),
				)
			)

	def policy_response(decision: Decision) -> JSONResponse:
		if decision.action == "refuse":
			if decision.reason == "path_stop":
				return _error(403, "path_stop", "campaign path stopped", "block")
			return _error(403, "blocked", "request refused", "block")
		return JSONResponse(
			{
				"detail": {
					"code": "challenge_required",
					"message": "answer the challenge and resend this request with "
					"X-Harness-Proof: NONCE:COUNTER",
				},
				**decision.client_view(),
			},
			status_code=428,
			headers={"X-Harness-Decision": "throttle"},
		)

	async def call_teacher(request: TeacherGenerationRequest) -> TeacherGeneration:
		started = time.perf_counter()
		generation = await upstream.generate(request)
		latencies.append(time.perf_counter() - started)
		template[:] = [generation]
		return generation

	def fabricate(content: str) -> TeacherGeneration:
		"""A response with a real one's provider, model, endpoint, and identifier shapes."""
		model = template[0]
		return TeacherGeneration(
			id=_mimic(model.id) or secrets.token_hex(12),
			provider=model.provider,
			model=model.model,
			generated_at=datetime.now(UTC),
			message=Message(role="assistant", content=content),
			finish_reason="stop",
			usage=TokenUsage(),
			request_id=_mimic(model.request_id),
			endpoint=model.endpoint,
		)

	async def release(
		http_request: Request,
		decision: Decision,
		request: TeacherGenerationRequest,
		text: str,
		binding: str,
		benchmark_request: str | None,
		now: float,
		reservation: Reservation,
		idem: str | None,
	) -> JSONResponse:
		nonlocal next_release_id
		async with lock:
			next_release_id += 1
			release_id = next_release_id
			inflight[release_id] = decision.account
		# The release accounting middleware clears the in-flight entry after the response.
		http_request.state.release_id = release_id
		attempt = (
			OutputAttempt(output_policy, reservation.domain) if output_policy is not None else None
		)
		status: Settlement = "failed"
		tokens: int | None = None
		response: JSONResponse | None = None
		try:
			response, tokens = await produce(
				decision, request, text, binding, benchmark_request, attempt
			)
			if response.status_code == 200:
				if (
					config.output_token_budget is not None
					or config.organization_token_budget is not None
					or supervision is not None
					or tier is not None
				) and (tokens is None or tokens < 0 or tokens > reservation.reserved_tokens):
					response = _error(
						502,
						"unaccountable_output",
						"teacher output cannot be safely accounted",
						None,
					)
				else:
					status = "released"
		except asyncio.CancelledError:
			status = "cancelled"
			raise
		finally:
			settled_at = now if config.test_clock else clock()
			with store.transaction() as tx:
				settled = tx.settle(
					reservation.key, status, settled_at, tokens if status == "released" else None
				)
				if status == "released" and not settled:
					response = _error(
						503, "reservation_expired", "reservation expired before release", None
					)
					status = "failed"
					tx.settle(reservation.key, status, settled_at)
				if idem is not None and response is not None:
					tx.put_idempotency(
						IdempotencyRecord(
							decision.account,
							idem,
							reservation.request_digest,
							now,
							Outcome(
								response.status_code,
								bytes(response.body).decode(),
								dict(response.headers),
							),
						)
					)
				if attempt is not None:
					# Settled with the reservation, so the two ledgers cannot disagree.
					attempt.settle(status)
					tx.put_output_receipt(
						reservation.key, decision.account, settled_at, attempt.receipt()
					)
			log.write(
				"release",
				time=settled_at,
				account=decision.account,
				horizon=decision.horizon,
				attempt=decision.attempt,
				status=status,
				**(
					{"path": decision.path, "reason": decision.reason}
					if decision.path is not None
					else {}
				),
				output_tokens=tokens if status == "released" else None,
				deceive=decision.deceive,
				policy_revision=policy_revision,
				**path_fields,
				**({"output_policy": attempt.receipt()} if attempt is not None else {}),
				request_digest=reservation.request_digest,
				benchmark_request=benchmark_request,
			)
		if status == "released":
			http_request.state.release_account = decision.account
			http_request.state.release_tokens = tokens
		assert response is not None
		return response

	async def produce(
		decision: Decision,
		request: TeacherGenerationRequest,
		text: str,
		binding: str,
		benchmark_request: str | None,
		attempt: OutputAttempt | None = None,
	) -> tuple[JSONResponse, int | None]:
		"""The response, and the output tokens it releases by the server's own count: the
		victim tokenizer's when configured, else the teacher's usage for a truthful answer."""
		headers = {"X-Harness-Decision": decision_header(decision)}
		if config.test_clock:
			headers["X-Harness-Release-At"] = repr(decision.release_at)
		else:
			await asyncio.sleep(max(0.0, decision.release_at - clock()))
		content: str | None = None
		if decision.deceive:
			assert key is not None
			content = noise(config.task, key, text, request.max_output_tokens)
			log.write(
				"noise",
				account=decision.account,
				request_digest=hashlib.sha256(binding.encode()).hexdigest(),
				benchmark_request=benchmark_request,
			)
		if attempt is not None:
			# Settings refuse deception with an output policy, so this is always a real draft.
			assert content is None
			attempt.calls["draft"] = ProviderCall(
				"interrupted", max_output_tokens=request.max_output_tokens
			)
		try:
			if content is not None and template:
				# Noise needs no teacher call; wait like a real call so timing gives nothing away.
				if latencies and not config.test_clock:
					await asyncio.sleep(random.choice(latencies))
				generation = fabricate(content)
			else:
				# Until one real response has been seen there is no shape to copy.
				generation = await call_teacher(request)
		except TeacherUpstreamError as error:
			failed = _error(
				502, "teacher_upstream_error", str(error), headers["X-Harness-Decision"]
			)
			if attempt is not None:
				attempt.calls["draft"] = ProviderCall(
					"error",
					max_output_tokens=request.max_output_tokens,
					error=type(error).__name__,
				)
			return failed, None
		if attempt is not None:
			return await release_under_policy(attempt, request, generation, headers)
		tokens = generation.usage.output_tokens if content is None else None
		if content is not None:
			generation.message = Message(role="assistant", content=content)
			generation.finish_reason = "stop"
		if token_counter:
			tokens = token_counter(generation.message.content)
		# Every response carries the same usage fields: output tokens recounted or omitted, and
		# no input count, which a fabricated response could not know.
		if config.l4_action == "deceive":
			generation.usage = TokenUsage(output_tokens=tokens if token_counter else None)
		return JSONResponse(generation.model_dump(mode="json"), headers=headers), tokens

	async def release_under_policy(
		attempt: OutputAttempt,
		request: TeacherGenerationRequest,
		draft: TeacherGeneration,
		headers: dict[str, str],
	) -> tuple[JSONResponse, int | None]:
		"""Replace the draft before any response exists. No store transaction is open here.

		Everything client-visible is rebuilt from the released text: the draft's finish reason
		and output count would reveal its truncation and length. Any failure after the draft
		exists withholds; an unexpected error is logged by type only.
		"""
		assert token_counter is not None
		try:
			released = await apply_policy(attempt, request, draft, upstream.generate, token_counter)
		except Exception as error:
			attempt.failure = f"internal_error:{type(error).__name__}"
			released = None
		if released is None:
			attempt.withhold("internal_error")
			withheld = _error(
				502,
				"output_withheld",
				"the response could not be released under the configured output policy",
				headers["X-Harness-Decision"],
			)
			withheld.headers[RECEIPT_HEADER] = attempt.receipt_header()
			return withheld, None
		if attempt.protected:
			draft.finish_reason = "stop"
		draft.message = Message(role="assistant", content=released)
		draft.usage = TokenUsage(
			input_tokens=None if config.l4_action == "deceive" else draft.usage.input_tokens,
			output_tokens=attempt.released_tokens,
		)
		headers[RECEIPT_HEADER] = attempt.receipt_header()
		return JSONResponse(draft.model_dump(mode="json"), headers=headers), attempt.released_tokens

	class ReviewDecision(BaseModel):
		action: str
		operator: str
		reason: str
		requests: int = Field(default=0, ge=0)
		tokens: int = Field(default=0, ge=0)
		expires_at: float | None = None

	def admin_authorized(key: str | None) -> bool:
		return bool(
			config.review_enabled
			and config.review_admin_key
			and key
			and hmac.compare_digest(key, config.review_admin_key)
		)

	@app.get("/admin/reviews")
	async def review_list(x_harness_admin_key: str | None = Header(default=None)) -> Any:
		if not admin_authorized(x_harness_admin_key):
			return _error(401, "unauthorized", "admin key required", None)
		with store.transaction() as tx:
			assert hasattr(tx, "db")
			return review.list_appeals(tx.db)

	@app.get("/admin/reviews/{ref}")
	async def review_show(ref: str, x_harness_admin_key: str | None = Header(default=None)) -> Any:
		if not admin_authorized(x_harness_admin_key):
			return _error(401, "unauthorized", "admin key required", None)
		with store.transaction() as tx:
			assert hasattr(tx, "db")
			return review.show(tx.db, ref) or _error(404, "not_found", "unknown appeal", None)

	@app.post("/admin/reviews/{ref}/decisions")
	async def review_decide(
		ref: str, body: ReviewDecision, x_harness_admin_key: str | None = Header(default=None)
	) -> Any:
		if not admin_authorized(x_harness_admin_key):
			return _error(401, "unauthorized", "admin key required", None)
		try:
			with store.transaction() as tx:
				assert hasattr(tx, "db")
				return {"id": review.decide(tx.db, ref, **body.model_dump())}
		except ValueError as error:
			return _error(400, "invalid_decision", str(error), None)

	@app.get("/healthz")
	async def health() -> dict[str, str]:
		return {"status": "ready", "policy": config.policy}

	@app.post("/v1/teacher/generate", response_model=None)
	async def generate(
		request: TeacherGenerationRequest,
		http_request: Request,
		authorization: str | None = Header(default=None),
		x_benchmark_actor: str | None = Header(default=None),
		x_harness_proof: str | None = Header(default=None),
		x_harness_time: str | None = Header(default=None),
		x_benchmark_request: str | None = Header(default=None),
		x_gateway_domain: str | None = Header(default=None),
		idempotency_key: str | None = Header(default=None),
	) -> JSONResponse:
		# X-Benchmark-Request is a test correlation ID: it is logged, never used to decide.
		identity = resolve(authorization, x_benchmark_actor)
		if identity is None:
			return _error(401, "unauthorized", "unknown API key", None)
		account, organization = identity
		if score_failure:
			return _error(503, "scoring_unavailable", "off-path scoring failed", None)
		now = now_from(x_harness_time)
		text = current_turn(request)
		if served_task == "gateway":
			if (
				x_gateway_domain not in config.gateway_caps
				or request.max_output_tokens != config.gateway_caps[x_gateway_domain]
			):
				return _error(
					422, "invalid_domain", "declare a configured domain and its frozen cap", None
				)
			assert x_gateway_domain is not None
			if (
				len(request.messages) != 1
				or request.messages[0].role != "user"
				or not valid_gateway_turn(x_gateway_domain, text)
			):
				return _error(
					422,
					"invalid_template",
					"use the declared domain task template in one user message",
					None,
				)
			text = gateway_turn(x_gateway_domain, text)
		binding = canonical_json(
			{"request": request.model_dump(mode="json"), "detector_turn": text}
		)
		digest = hashlib.sha256(binding.encode()).hexdigest()
		if (
			output_policy is not None
			and output_policy.unsupported_task == "refuse"
			and (x_gateway_domain if served_task == "gateway" else served_task)
			not in output_policy.tasks
		):
			# Nothing was admitted or generated, so nothing is charged or observed.
			log.write(
				"output_policy_refusal",
				time=now,
				account=account,
				policy_revision=policy_revision,
				**output_policy.event_fields(),
				request_digest=digest,
				benchmark_request=x_benchmark_request,
			)
			return _error(
				422,
				"output_policy_unsupported_task",
				"the configured output policy does not cover this task",
				None,
			)
		with store.transaction() as tx:
			prior = previous(
				tx, account, idempotency_key, digest, proof=x_harness_proof is not None, now=now
			)
		if prior is not None:
			return prior
		if x_harness_proof is not None:
			nonce, _, counter = x_harness_proof.partition(":")
			# The challenge is bound to the account and the whole canonical request, so the
			# resent request is the one that was challenged and can be released as sent.
			held: Decision | None = None
			reservation = None
			rejection = None
			proof_throttle: JSONResponse | None = None
			proof_until: float | None = None
			with store.transaction() as tx:
				prior = previous(tx, account, idempotency_key, digest, proof=True, now=now)
				if prior is not None:
					return prior
				# An active experimental cooldown also stops redemption. The challenge stays
				# unspent, its reservation and the stored 428 stay as they are, and the
				# refusal is not remembered, so the same proof can be sent after expiry.
				if campaign is not None and campaign.mode == "throttle":
					proof_until = cooldown_until(tx, account, organization, now)
					# Held proofs reuse pre-grant reservations, so reservation-counted
					# review allowances cannot safely bound their redemption. Grants apply
					# to fresh admissions only; held proofs wait for cooldown expiry.
					if proof_until is not None:
						proof_throttle = throttle_response(
							tx, account, organization, digest, now, proof_until
						)
				if proof_throttle is None and isinstance(policy, MMDPolicy) and counter.isdigit():
					challenge_record = tx.challenge(nonce)
					if (
						challenge_record is not None
						and challenge_record[1].get("idempotency_key") != idempotency_key
					):
						return _error(
							409,
							"idempotency_conflict",
							"proof must use the challenge's idempotency key",
							None,
						)
					held = policy.redeem_in_transaction(
						tx, nonce, account, binding, int(counter), now
					)
				if held is not None:
					reservation = tx.reservation(admission_key(account, held.horizon, held.attempt))
					if reservation is None or not reservation.live(now):
						rejection = _error(
							409, "reservation_expired", "reservation is no longer available", None
						)
						remember(tx, account, idempotency_key, digest, now, rejection)
					elif idempotency_key is not None:
						tx.put_idempotency(IdempotencyRecord(account, idempotency_key, digest, now))
			if proof_throttle is not None:
				log.write(
					"campaign_throttle",
					time=now,
					account=account,
					stage="proof",
					throttle_until=proof_until,
					policy_revision=policy_revision,
					qualified=False,
					request_digest=digest,
					benchmark_request=x_benchmark_request,
				)
				return proof_throttle
			log.write(
				"redeem",
				account=account,
				request_digest=digest,
				solved=held is not None,
				benchmark_request=x_benchmark_request,
			)
			if held is None:
				return _error(
					403, "challenge_failed", "the challenge answer was not accepted", "throttle"
				)
			if rejection is not None:
				return rejection
			assert reservation is not None
			return await release(
				http_request,
				held,
				request,
				text,
				binding,
				x_benchmark_request,
				now,
				reservation,
				idempotency_key,
			)
		# A spent rolling budget is cheap to read, so it costs no embedding or policy state.
		# A principal without an organization is left to admission, which refuses it.
		spent = None
		code = None
		domain = x_gateway_domain if served_task == "gateway" else served_task
		assert domain is not None
		candidate = Reservation(
			account,
			now,
			0,
			digest,
			now,
			reserved_tokens=request.max_output_tokens,
			organization=organization,
			domain=domain,
		)
		with store.transaction() as tx:
			# Admission drops an unlinked principal's organization; the early check matches it.
			if tx.unlinked(account, organization):
				candidate = replace(candidate, organization=None)
			code = rolling_refusal(
				tx, candidate, now, rolling, organization_rolling, supervision, tier
			)
			if code is not None:
				status_code, message, header = ROLLING_REFUSALS[code]
				spent = _error(status_code, code, message, header)
				spent = review_response(
					tx,
					spent,
					account,
					candidate.organization,
					digest,
					now,
					code,
					scope="organization" if code.startswith("organization_") else "account",
					window=config.rolling_window_seconds,
				)
				remember(tx, account, idempotency_key, digest, now, spent)
		if spent is not None:
			log.write(
				"decision",
				time=now,
				account=account,
				attempt=None,
				horizon=None,
				level=4,
				action="refuse",
				reason=None,
				policy_revision=policy_revision,
				**event_fields,
				deceive=False,
				window_completed=False,
				p_value=None,
				statistic=None,
				witness=[],
				request_digest=digest,
				benchmark_request=x_benchmark_request,
				entitlement=code,
			)
			return spent
		embedding = None
		campaign_embedding = None
		campaign_semantic_complete: bool | None = None
		if needs_embeddings and encoder is not None:
			try:
				encoded_text = detector_text(served_task, text) if base_needs_embeddings else None
				if encoded_text is not None:
					embedding = (
						await batcher.encode(encoded_text)
						if batcher is not None
						else (await asyncio.to_thread(encoder.encode, [encoded_text]))[0]
					)
				if campaign_needs_embeddings:
					assert campaign is not None
					# Tokenizer work happens before the write transaction. Unknown or
					# truncated representations cannot supply semantic evidence.
					campaign_text = text[: campaign.max_text_chars]
					complete = getattr(encoder, "input_complete", None)
					if campaign_text != text:
						campaign_semantic_complete = False
					elif callable(complete):
						campaign_semantic_complete = await asyncio.to_thread(
							complete, campaign_text
						)
						if not isinstance(campaign_semantic_complete, bool):
							raise ValueError("encoder completeness must be a boolean")
					if campaign_semantic_complete is True:
						campaign_embedding = (
							embedding
							if campaign_text == encoded_text
							else (
								await batcher.encode(campaign_text)
								if batcher is not None
								else (await asyncio.to_thread(encoder.encode, [campaign_text]))[0]
							)
						)
			except asyncio.QueueFull:
				return _error(503, "encoder_overloaded", "encoder queue is full", None)
			except Exception:
				log.write("encoder_error", account=account, request_digest=digest)
				return _error(503, "encoder_error", "encoder failed", None)
		if campaign is not None and campaign.session is None and campaign.coverage is None:
			# Preserve v2 telemetry when the experimental extension is disabled.
			campaign_embedding = embedding
		if shadow is not None and embedding is None and encoder is not None:
			# Shadow-only encoding happens before anything commits; a failure only skips it.
			try:
				shadow_text = detector_text(served_task, text)
				embedding_for_shadow = (await asyncio.to_thread(encoder.encode, [shadow_text]))[0]
			except Exception as error:
				log.write(
					"shadow_error",
					account=account,
					request_digest=digest,
					error=type(error).__name__,
				)
				embedding_for_shadow = None
		else:
			embedding_for_shadow = embedding
		if score_failure:
			return _error(503, "scoring_unavailable", "off-path scoring failed", None)
		reservation = None
		rejection = None
		policy_result = None
		alert = None
		# The campaign evidence of the committed attempt only: a rolled-back attempt leaves none.
		campaign_result: dict[str, Any] | None = None
		campaign_override = False
		campaign_throttle: JSONResponse | None = None
		campaign_until: float | None = None
		# One identity per real request: a retry carries its idempotency key, and replays return
		# before this point, so a retry is never observed twice.
		campaign_request = (
			hashlib.sha256(canonical_json([account, idempotency_key]).encode()).hexdigest()
			if idempotency_key is not None
			else secrets.token_hex(16)
		)
		observe_failed = False
		campaign_identity = text_identity(text) if campaign is not None else None
		campaign_session_observation = (
			prepare_session_observation(text[: campaign.max_text_chars])
			if campaign is not None and campaign.session is not None
			else None
		)
		while True:
			campaign_result = None
			campaign_override = False
			prepared = (
				await asyncio.to_thread(policy.prepare, account, text, now, embedding, binding)
				if isinstance(policy, PathStopPolicy)
				else None
			)
			# A failed off-path submission rolls the whole admission back.
			try:
				with store.transaction() as tx:
					prior = previous(tx, account, idempotency_key, digest, now=now)
					if prior is not None:
						return prior
					code = (
						rolling_refusal(
							tx, candidate, now, rolling, organization_rolling, supervision, tier
						)
						if supervision is not None or tier is not None
						else None
					)
					if code is not None:
						status_code, message, header = ROLLING_REFUSALS[code]
						refusal = _error(status_code, code, message, header)
						remember(tx, account, idempotency_key, digest, now, refusal)
						log.write(
							"decision",
							time=now,
							account=account,
							attempt=None,
							horizon=None,
							level=4,
							action="refuse",
							reason=None,
							policy_revision=policy_revision,
							**event_fields,
							deceive=False,
							window_completed=False,
							p_value=None,
							statistic=None,
							witness=[],
							request_digest=digest,
							benchmark_request=x_benchmark_request,
							entitlement=code,
						)
						return refusal
					if campaign is not None:
						# The observation and any cooldown it starts commit with this admission,
						# so a concurrent worker sees both or neither.
						if not observe_failed:
							try:
								campaign_result = observe_campaign(
									tx,
									campaign,
									principal=account,
									organization=organization,
									request_id=campaign_request,
									text=text[: campaign.max_text_chars],
									identity=campaign_identity,
									resampling=request.temperature > 0,
									semantic_complete=campaign_semantic_complete,
									session_observation=campaign_session_observation,
									now=now,
									domain=domain,
									embedding=(
										np.asarray(campaign_embedding, dtype=np.float64).tolist()
										if campaign_embedding is not None
										else None
									),
								)
							except CampaignError as error:
								raise CampaignObservationFailed(type(error).__name__) from error
							if campaign_result.get("qualified") is not False:
								raise RuntimeError("campaign evidence must stay unqualified")
						if campaign.mode == "throttle":
							campaign_until = cooldown_until(tx, account, organization, now)
							if (
								campaign_result is not None
								and campaign_result["should_throttle"]
								and campaign_result["throttle_until"] is not None
							):
								campaign_until = float(campaign_result["throttle_until"])
							if campaign_until is not None:
								if tx.allowance(
									account, "campaign_throttled", now, request.max_output_tokens
								):
									campaign_override = True
								else:
									campaign_throttle = throttle_response(
										tx, account, organization, digest, now, campaign_until
									)
									remember(
										tx, account, idempotency_key, digest, now, campaign_throttle
									)
									break
					decision = policy.admit_in_transaction(
						tx,
						account,
						text,
						now,
						embedding=embedding,
						binding=binding,
						prepared=prepared,
					)
					if (
						decision.action == "refuse"
						and decision.reason != "path_stop"
						and tx.allowance(
							account, decision.reason or "blocked", now, request.max_output_tokens
						)
					):
						decision = replace(decision, action="serve", release_at=now)
					if (
						monitor
						and embedding is not None
						and tx.append_aggregate(embedding, account, now)
						>= monitor.calibration.window_size
					):
						index, rows = tx.take_aggregate()
						window = AggregateWindow(
							index,
							np.stack([r[0] for r in rows]),
							[r[1] for r in rows],
							rows[0][2],
							rows[-1][2],
						)
						alert = monitor.score(window)
						tx.put_aggregate_alert(index, asdict(alert))
					if decision.action in {"serve", "challenge"}:
						reservation, rejection = reserve(
							tx,
							decision,
							request,
							digest,
							now,
							organization,
							idempotency_key,
							domain=domain,
						)
						if rejection is not None:
							remember(tx, account, idempotency_key, digest, now, rejection)
						elif decision.action == "challenge":
							assert decision.challenge is not None
							challenge_record = tx.challenge(decision.challenge.nonce)
							assert challenge_record is not None
							challenge, held_data = challenge_record
							held_data["idempotency_key"] = idempotency_key
							# Replace the held metadata in the same transaction as the reservation.
							tx.delete_challenge(challenge.nonce)
							tx.put_challenge(challenge, held_data)
							policy_result = review_response(
								tx,
								policy_response(decision),
								account,
								organization,
								digest,
								now,
								decision.reason or "challenge_required",
								decision.horizon,
								decision.horizon + calibration.config.horizon_seconds,
							)
							remember(tx, account, idempotency_key, digest, now, policy_result)
					else:
						policy_result = review_response(
							tx,
							policy_response(decision),
							account,
							organization,
							digest,
							now,
							decision.reason or "blocked",
							decision.horizon,
							decision.horizon + calibration.config.horizon_seconds,
						)
						remember(tx, account, idempotency_key, digest, now, policy_result)
			except StalePathAdmission:
				continue
			except CampaignObservationFailed as error:
				# The attempt rolled back, so nothing was observed. Rerun it without an
				# observation; an enforced cooldown on record still applies.
				if campaign is not None and campaign.mode == "throttle":
					with store.transaction() as tx:
						active = cooldown_until(tx, account, organization, now)
					if active is None:
						log.write(
							"campaign_error",
							time=now,
							account=account,
							error=str(error),
							policy_revision=policy_revision,
							request_digest=digest,
						)
						return _error(
							503, "campaign_unavailable", "campaign scoring unavailable", None
						)
				observe_failed = True
				log.write(
					"campaign_error",
					time=now,
					account=account,
					error=str(error),
					policy_revision=policy_revision,
					request_digest=digest,
					benchmark_request=x_benchmark_request,
				)
				continue
			except ScoringUnavailable:
				log.write("off_path_queue_error", account=account, request_digest=digest)
				return _error(503, "scoring_unavailable", "off-path scoring queue failed", None)
			break
		if campaign_throttle is not None:
			log.write(
				"campaign_throttle",
				time=now,
				account=account,
				stage="admission",
				throttle_until=campaign_until,
				policy_revision=policy_revision,
				qualified=False,
				campaign=campaign_result,
				request_digest=digest,
				benchmark_request=x_benchmark_request,
			)
			return campaign_throttle
		for sequence, metrics in score_metrics.items():
			if score_accounts[sequence] == account:
				metrics["admissions"] += 1
		if alert is not None:
			log.write("aggregate", **asdict(alert))
		# Present only with shadow scorers enabled, so other decision logs keep their schema.
		shadow_fields = (
			{"shadow": observe_shadow(decision, text, embedding_for_shadow, digest)}
			if shadow is not None
			else {}
		)
		log.write(
			"decision",
			time=now,
			account=account,
			attempt=decision.attempt,
			horizon=decision.horizon,
			level=decision.level,
			action=decision.action,
			reason=decision.reason,
			policy_revision=policy_revision,
			**event_fields,
			deceive=decision.deceive,
			window_completed=decision.window_completed,
			p_value=decision.p_value,
			statistic=None
			if decision.statistic is None or math.isinf(decision.statistic)
			else decision.statistic,
			witness=list(decision.witness),
			request_digest=digest,
			benchmark_request=x_benchmark_request,
			# The admission counted against the horizon even when its entitlement was refused.
			entitlement=None
			if rejection is None
			else json.loads(bytes(rejection.body))["detail"]["code"],
			**shadow_fields,
			**({"path": decision.path} if decision.path is not None else {}),
			# Present only with the campaign detector configured. None means no observation
			# was committed for this admission; an override means a review grant served it.
			**(
				{"campaign": campaign_result, "campaign_override": campaign_override}
				if campaign is not None
				else {}
			),
		)
		if rejection is not None:
			return rejection
		if decision.action != "serve":
			assert policy_result is not None
			return policy_result
		assert reservation is not None
		return await release(
			http_request,
			decision,
			request,
			text,
			binding,
			x_benchmark_request,
			now,
			reservation,
			idempotency_key,
		)

	return app
