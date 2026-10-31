"""Replay a fixed schedule against the real HTTP path without hidden retries."""

import asyncio
import hashlib
import json
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
from pydantic import BaseModel, Field, ValidationError

from distillation.benchmarks.models import save_json
from distillation.benchmarks.traffic import (
	Decision,
	Phase,
	TrafficConfig,
	TrafficRequest,
	TrafficRun,
	TrafficSample,
	load_traffic_suite,
)
from distillation.io import append_jsonl
from distillation.schemas import TeacherGeneration


class _ChatMessage(BaseModel):
	content: str


class _ChatChoice(BaseModel):
	message: _ChatMessage
	finish_reason: str | None = None


class _ChatUsage(BaseModel):
	completion_tokens: int | None = Field(default=None, ge=0)


class _ChatResponse(BaseModel):
	model: str = Field(min_length=1)
	choices: list[_ChatChoice] = Field(min_length=1)
	usage: _ChatUsage | None = None


def solve_challenge(prefix: str, difficulty: int) -> int:
	"""The smallest counter whose SHA-256 of prefix + counter has `difficulty` leading zero bits."""
	counter = 0
	while True:
		digest = hashlib.sha256(f"{prefix}{counter}".encode()).digest()
		if int.from_bytes(digest) >> (256 - difficulty) == 0:
			return counter
		counter += 1


def _harness_events(log: Path, run_id: str) -> tuple[set[str], dict[str, int]]:
	"""Correlate noise and detector levels with this run's requests."""
	if not log.is_file():
		raise ValueError(f"harness log not found: {log}")
	prefix = f"{run_id}:"
	noised: set[str] = set()
	levels: dict[str, int] = {}
	for line in log.read_text(encoding="utf-8").split("\n"):
		if not line.strip():
			continue
		event = json.loads(line)
		correlation = event.get("benchmark_request") or ""
		if correlation.startswith(prefix):
			request_id = correlation.removeprefix(prefix)
			if event.get("event") == "noise":
				noised.add(request_id)
			elif event.get("event") == "decision" and isinstance(event.get("level"), int):
				levels[request_id] = event["level"]
	return noised, levels


def _completion(response: httpx.Response, backend: str) -> tuple[bool, int | None, str]:
	if backend == "teacher_api":
		generation = TeacherGeneration.model_validate(response.json())
		text, finish = generation.message.content, generation.finish_reason
		tokens, model = generation.usage.output_tokens, generation.model
	else:
		body = _ChatResponse.model_validate(response.json())
		choice = body.choices[0]
		text, finish = choice.message.content, choice.finish_reason
		model = body.model
		tokens = body.usage.completion_tokens if body.usage else None
	valid = (
		isinstance(text, str)
		and bool(text.strip())
		and finish in {"stop", "eos", "end_turn"}
		and "<think>" not in text
		and "</think>" not in text
	)
	if not isinstance(model, str) or not model:
		raise ValueError("response must identify the served model")
	return valid, tokens, model


async def run_traffic(
	config: TrafficConfig,
	phase: Phase,
	*,
	transport: httpx.AsyncBaseTransport | None = None,
) -> TrafficRun:
	suite = load_traffic_suite(config.suite_path)
	target = config.targets[phase]
	if target.gateway_domain:
		from harness.serialization import valid_gateway_turn

		if target.backend != "teacher_api":
			raise ValueError("gateway_domain requires the teacher_api backend")
		if any(
			len(row.messages) != 1
			or row.messages[0].role != "user"
			or not valid_gateway_turn(target.gateway_domain, row.messages[0].content)
			for row in suite.requests
		):
			raise ValueError("traffic suite does not match the configured gateway domain")
	actors = {actor.id for actor in suite.actors}
	if set(target.actor_key_envs) - actors:
		raise ValueError("credential configuration references unknown traffic actors")
	keys = {}
	for actor in actors:
		env = target.actor_key_envs.get(actor, target.api_key_env)
		if env and not os.environ.get(env):
			raise ValueError(f"missing environment variable {env}")
		keys[actor] = os.environ[env] if env else ""
	run_id = uuid.uuid4().hex
	semaphore = asyncio.Semaphore(config.max_in_flight)
	loop = asyncio.get_running_loop()
	# Challenges are solved in processes: hashlib holds the GIL for short inputs.
	pool = ProcessPoolExecutor()
	async with httpx.AsyncClient(
		base_url=f"{target.base_url.rstrip('/')}/",
		timeout=config.timeout_seconds,
		limits=httpx.Limits(max_connections=config.max_in_flight),
		transport=transport,
	) as client:
		started_at = datetime.now(UTC)
		start = time.perf_counter()

		async def send(item: TrafficRequest) -> TrafficSample:
			await asyncio.sleep(max(0, start + item.at_seconds - time.perf_counter()))
			async with semaphore:
				began = time.perf_counter() - start
				result = TrafficSample(
					request_id=item.id,
					scheduled_seconds=item.at_seconds,
					started_seconds=began,
					completed_seconds=began,
					status="error",
				)
				headers = {
					"Accept": "application/json",
					"X-Benchmark-Actor": item.actor_id,
					# A run-unique correlation ID that the harness logs but never decides on.
					"X-Benchmark-Request": f"{run_id}:{item.id}",
				}
				if target.gateway_domain:
					headers["X-Gateway-Domain"] = target.gateway_domain
				if key := keys[item.actor_id]:
					headers["Authorization"] = f"Bearer {key}"
				payload: dict[str, Any] = {
					"messages": [message.model_dump() for message in item.messages],
					"temperature": 0,
					"seed": config.generation.seed,
				}
				if target.backend == "teacher_api":
					path = "v1/teacher/generate"
					payload.update(
						max_output_tokens=config.generation.max_output_tokens, enable_thinking=False
					)
				else:
					path = "chat/completions"
					payload.update(
						model=config.model,
						max_tokens=config.generation.max_output_tokens,
						chat_template_kwargs={"enable_thinking": False},
					)
				try:
					response = await client.post(path, json=payload, headers=headers)
					if response.status_code == 428:
						# Answering the harness's challenge is part of the protocol, not a retry:
						# the same frozen request stays one sample.
						challenge = response.json()["challenge"]
						solving = time.perf_counter()
						counter = await loop.run_in_executor(
							pool,
							solve_challenge,
							str(challenge["prefix"]),
							int(challenge["difficulty"]),
						)
						result.challenge_seconds = time.perf_counter() - solving
						proof = {"X-Harness-Proof": f"{challenge['nonce']}:{counter}"}
						response = await client.post(path, json=payload, headers=headers | proof)
					result.http_status = response.status_code
					result.retry_after = response.headers.get("retry-after")
					result.provider_request_id = response.headers.get("x-request-id")
					decision = response.headers.get("x-harness-decision", "").lower()
					if decision in {"allow", "throttle", "block"}:
						result.decision = cast(Decision, decision)
					if result.decision == "block":
						result.status = "blocked"
					elif response.status_code == 429:
						result.status = "rate_limited"
					elif not response.is_success:
						result.status = "http_error"
					else:
						valid, tokens, model = _completion(response, target.backend)
						result.succeeded = valid
						result.status = "ok" if valid else "invalid"
						result.output_tokens = tokens
						result.served_model = model
				except httpx.TimeoutException:
					result.status = "timeout"
				except httpx.HTTPError:
					result.status = "error"
				except ValidationError, ValueError, KeyError, IndexError, TypeError:
					result.status = "invalid"
				result.completed_seconds = time.perf_counter() - start
				# Bodies and credentials are deliberately excluded from result artifacts.
				append_jsonl(config.output_dir / f"{run_id}.samples.jsonl", [result])
				return result

		try:
			samples = await asyncio.gather(*(send(item) for item in suite.requests))
		finally:
			pool.shutdown(cancel_futures=True)
	if target.harness_log is not None:
		noised, levels = _harness_events(target.harness_log, run_id)
		for sample in samples:
			sample.level = levels.get(sample.request_id)
			if sample.level is not None:
				sample.noised = sample.request_id in noised
	run = TrafficRun(
		id=run_id,
		phase=phase,
		started_at=started_at,
		completed_at=datetime.now(UTC),
		suite_sha256=suite.sha256,
		config=config,
		samples=samples,
	)
	save_json(config.output_dir / f"traffic-suite-{suite.sha256}.json", suite)
	save_json(config.output_dir / f"{run_id}.json", run)
	return run
