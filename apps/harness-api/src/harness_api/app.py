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
account horizons, open challenges, and the ledger survive restarts. The aggregate monitor and
the noise template still live in this process, so the service runs as a single worker.
"""

import asyncio
import hashlib
import json
import math
import os
import random
import secrets
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from distillation.io import canonical_json
from distillation.schemas import (
	Message,
	TeacherGeneration,
	TeacherGenerationRequest,
	TokenUsage,
)
from distillation.teacher import Teacher, TeacherAPIGateway, TeacherUpstreamError
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from harness.calibration import Calibration, load_aggregate, load_calibration
from harness.deception import noise
from harness.encoder import Encoder
from harness.mmd import WindowScorer
from harness.monitor import AggregateMonitor, AggregateWindow
from harness.policy import Decision, MMDPolicy, Policy
from harness.replay import make_policy
from harness.serialization import detector_text, gateway_turn, valid_gateway_turn
from harness.store import MemoryStore, Reservation, Settlement, SQLiteStore, StateStore

from harness_api.settings import Settings


def current_turn(request: TeacherGenerationRequest) -> str:
	"""The latest user message: the paper embeds the current turn only."""
	return next(m.content for m in reversed(request.messages) if m.role == "user")


def decision_header(decision: Decision) -> str:
	if decision.action == "refuse":
		return "block"
	return "allow" if decision.level < 2 else "throttle"


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


def _scope(settings: Settings, calibration: Calibration) -> str:
	"""Everything that gives stored state its meaning: the policy, its actions, and the
	calibration. Secrets and service addresses are left out."""
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
		}
	)
	# Omitted for the default version so existing legacy state files keep their scope.
	if settings.policy_version != "legacy-v1":
		fields["policy_version"] = settings.policy_version
	fields["calibration"] = hashlib.sha256(calibration.model_dump_json().encode()).hexdigest()
	return canonical_json(fields)


def _accounts(settings: Settings) -> dict[str, str]:
	if settings.accounts_file is None:
		return {}
	mapping: dict[str, str] = json.loads(settings.accounts_file.read_text())
	return {digest.lower(): account for digest, account in mapping.items()}


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
	store: StateStore = (
		SQLiteStore(config.state_db, _scope(config, calibration))
		if config.state_db is not None
		else MemoryStore()
	)
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
		)
	else:
		policy = make_policy(config.policy, calibration, scorer, store=store)
	monitor = (
		AggregateMonitor(aggregate_scorer, calibration.aggregate)
		if aggregate_scorer is not None and calibration.aggregate is not None
		else None
	)
	needs_embeddings = config.policy == "mmd" or monitor is not None
	if encoder is None and needs_embeddings:
		from harness.encoder import BGEEncoder

		encoder = BGEEncoder(device=config.encoder_device)
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
	log = DecisionLog(config.decision_log)
	lock = asyncio.Lock()
	background: set[asyncio.Task[None]] = set()
	# Real responses teach the shape and timing that noise responses copy.
	template: list[TeacherGeneration] = []
	latencies: deque[float] = deque(maxlen=256)

	@asynccontextmanager
	async def lifespan(_: FastAPI) -> AsyncIterator[None]:
		yield
		if background:
			await asyncio.gather(*background, return_exceptions=True)
		await upstream.close()
		if isinstance(store, SQLiteStore):
			store.close()

	app = FastAPI(title="TRACE protected teacher API", version="0.1.0", lifespan=lifespan)

	def now_from(header: str | None) -> float:
		if config.test_clock and header is not None:
			return float(header)
		return clock()

	def resolve(authorization: str | None, actor: str | None) -> str | None:
		if authorization and authorization.startswith("Bearer ") and accounts:
			digest = hashlib.sha256(authorization.removeprefix("Bearer ").encode()).hexdigest()
			if digest in accounts:
				return accounts[digest]
		if config.trust_actor_header and actor:
			return actor
		return None

	async def score_aggregate(window: AggregateWindow) -> None:
		assert monitor is not None
		alert = await asyncio.to_thread(monitor.score, window)
		log.write("aggregate", **asdict(alert))

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
		decision: Decision,
		request: TeacherGenerationRequest,
		text: str,
		binding: str,
		benchmark_request: str | None,
		now: float,
	) -> JSONResponse:
		"""Reserve the admission, produce its output, and settle the reservation once.

		Output is produced only under a new reservation, so one admission releases at most
		once. A response counts as released when it is built, before it is sent: a sent
		response cannot be shown to have arrived, so the ledger errs towards acquisition.
		"""
		digest = hashlib.sha256(binding.encode()).hexdigest()
		reservation = Reservation(
			account=decision.account,
			horizon_start=decision.horizon,
			attempt=decision.attempt,
			request_digest=digest,
			reserved_at=now,
			deceive=decision.deceive,
		)
		with store.transaction() as tx:
			reserved = tx.reserve(reservation)
		if not reserved:
			return _error(409, "already_released", "this admission was already released", None)
		status: Settlement = "failed"
		tokens: int | None = None
		try:
			response, produced = await produce(decision, request, text, binding, benchmark_request)
			if response.status_code == 200:
				status, tokens = "released", produced
			return response
		except asyncio.CancelledError:
			status = "cancelled"
			raise
		finally:
			settled_at = now if config.test_clock else clock()
			with store.transaction() as tx:
				tx.settle(reservation.key, status, settled_at, tokens)
			log.write(
				"release",
				time=settled_at,
				account=decision.account,
				horizon=decision.horizon,
				attempt=decision.attempt,
				status=status,
				output_tokens=tokens,
				deceive=decision.deceive,
				request_digest=digest,
				benchmark_request=benchmark_request,
			)

	async def produce(
		decision: Decision,
		request: TeacherGenerationRequest,
		text: str,
		binding: str,
		benchmark_request: str | None,
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
			return failed, None
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

	@app.get("/healthz")
	async def health() -> dict[str, str]:
		return {"status": "ready", "policy": config.policy}

	@app.post("/v1/teacher/generate", response_model=None)
	async def generate(
		request: TeacherGenerationRequest,
		authorization: str | None = Header(default=None),
		x_benchmark_actor: str | None = Header(default=None),
		x_harness_proof: str | None = Header(default=None),
		x_harness_time: str | None = Header(default=None),
		x_benchmark_request: str | None = Header(default=None),
		x_gateway_domain: str | None = Header(default=None),
	) -> JSONResponse:
		# X-Benchmark-Request is a test correlation ID: it is logged, never used to decide.
		account = resolve(authorization, x_benchmark_actor)
		if account is None:
			return _error(401, "unauthorized", "unknown API key", None)
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
		if x_harness_proof is not None:
			nonce, _, counter = x_harness_proof.partition(":")
			# The challenge is bound to the account and the whole canonical request, so the
			# resent request is the one that was challenged and can be released as sent.
			held: Decision | None = None
			if isinstance(policy, MMDPolicy) and counter.isdigit():
				async with lock:
					held = policy.redeem(nonce, account, binding, int(counter), now)
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
			return await release(held, request, text, binding, x_benchmark_request, now)
		embedding = (
			(await asyncio.to_thread(encoder.encode, [detector_text(served_task, text)]))[0]
			if needs_embeddings and encoder is not None
			else None
		)
		async with lock:
			decision = policy.admit(account, text, now, embedding=embedding, binding=binding)
			window = (
				monitor.add(embedding, account, now) if monitor and embedding is not None else None
			)
		if window is not None:
			task = asyncio.create_task(score_aggregate(window))
			background.add(task)
			task.add_done_callback(background.discard)
		log.write(
			"decision",
			time=now,
			account=account,
			attempt=decision.attempt,
			horizon=decision.horizon,
			level=decision.level,
			action=decision.action,
			deceive=decision.deceive,
			window_completed=decision.window_completed,
			p_value=decision.p_value,
			statistic=None
			if decision.statistic is None or math.isinf(decision.statistic)
			else decision.statistic,
			witness=list(decision.witness),
			request_digest=digest,
			benchmark_request=x_benchmark_request,
		)
		if decision.action == "refuse":
			return _error(403, "blocked", "request refused", "block")
		if decision.action == "challenge":
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
		return await release(decision, request, text, binding, x_benchmark_request, now)

	return app
