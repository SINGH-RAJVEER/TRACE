"""Replay a fixed schedule against the real HTTP path without hidden retries."""

import asyncio
import os
import time
import uuid
from datetime import UTC, datetime
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
				headers = {"Accept": "application/json", "X-Benchmark-Actor": item.actor_id}
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

		samples = await asyncio.gather(*(send(item) for item in suite.requests))
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
