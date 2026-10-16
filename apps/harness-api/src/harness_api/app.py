"""FastAPI service that applies the extraction harness in front of the teacher API.

It serves the teacher API's contract, POST /v1/teacher/generate, so collectors, evaluations,
and the traffic benchmark can point at it unchanged. Every response carries
X-Harness-Decision: allow (L0 and L1), throttle (L2 and L3, including answered challenges),
or block (L4 refusals). An L3 response is HTTP 428 with a proof-of-work challenge; the client
resends the same request with X-Harness-Proof: NONCE:COUNTER. By default an account at L4 is
not refused: it keeps receiving L3-shaped responses whose content is keyed noise, built
without calling the teacher. Account state lives in this process, so the service must run as
a single worker.
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
from dataclasses import asdict, dataclass
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

from harness_api.settings import Settings


@dataclass
class Pending:
	"""A request waiting for its challenge to be answered."""

	decision: Decision
	request: TeacherGenerationRequest
	binding: str
	text: str


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
	if settings.l4_action != "deceive":
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
		)
	else:
		policy = make_policy(config.policy, calibration, scorer)
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
	pending: dict[str, Pending] = {}
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

	app = FastAPI(title="Protected teacher API", version="0.1.0", lifespan=lifespan)

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
		decision: Decision, request: TeacherGenerationRequest, text: str, binding: str
	) -> JSONResponse:
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
			return _error(502, "teacher_upstream_error", str(error), headers["X-Harness-Decision"])
		if content is not None:
			generation.message = Message(role="assistant", content=content)
			generation.finish_reason = "stop"
		# Every response carries the same usage fields: output tokens recounted or omitted, and
		# no input count, which a fabricated response could not know.
		generation.usage = TokenUsage(
			output_tokens=token_counter(generation.message.content) if token_counter else None
		)
		return JSONResponse(generation.model_dump(mode="json"), headers=headers)

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
	) -> JSONResponse:
		account = resolve(authorization, x_benchmark_actor)
		if account is None:
			return _error(401, "unauthorized", "unknown API key", None)
		now = now_from(x_harness_time)
		text = current_turn(request)
		binding = canonical_json(request.model_dump(mode="json"))
		digest = hashlib.sha256(binding.encode()).hexdigest()
		if x_harness_proof is not None:
			nonce, _, counter = x_harness_proof.partition(":")
			async with lock:
				for stale in [
					n
					for n, p in pending.items()
					if p.decision.challenge and p.decision.challenge.expires_at < now
				]:
					del pending[stale]
				waiting = pending.pop(nonce, None)
				solved = (
					waiting is not None
					and waiting.decision.account == account
					and waiting.binding == binding
					and counter.isdigit()
					and isinstance(policy, MMDPolicy)
					and policy.redeem(waiting.decision, binding, int(counter), now)
				)
			log.write("redeem", account=account, request_digest=digest, solved=solved)
			if not solved or waiting is None:
				return _error(
					403, "challenge_failed", "the challenge answer was not accepted", "throttle"
				)
			return await release(waiting.decision, waiting.request, waiting.text, binding)
		embedding = (
			(await asyncio.to_thread(encoder.encode, [text]))[0]
			if needs_embeddings and encoder is not None
			else None
		)
		async with lock:
			for stale in [
				n
				for n, p in pending.items()
				if p.decision.challenge and p.decision.challenge.expires_at < now
			]:
				del pending[stale]
			decision = policy.admit(account, text, now, embedding=embedding, binding=binding)
			window = (
				monitor.add(embedding, account, now) if monitor and embedding is not None else None
			)
			if decision.challenge is not None:
				pending[decision.challenge.nonce] = Pending(decision, request, binding, text)
		if window is not None:
			task = asyncio.create_task(score_aggregate(window))
			background.add(task)
			task.add_done_callback(background.discard)
		log.write(
			"decision",
			time=now,
			account=account,
			attempt=decision.attempt,
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
		return await release(decision, request, text, binding)

	return app
