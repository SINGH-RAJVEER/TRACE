"""Bounded single-account campaigns; every HTTP attempt has a durable event record."""

import asyncio
import hashlib
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from anyio import Path as AsyncPath

from distillation.collection import CollectionOptions, _collect_one
from distillation.io import append_jsonl, read_jsonl
from distillation.schemas import (
	LedgerRecord,
	PromptRecord,
	TeacherGeneration,
	TeacherGenerationRequest,
)


class PolicyRefusal(Exception):
	"""A refusal explicitly attributed to the policy."""


class AttemptFailed(Exception):
	"""An unsuccessful attempt whose cost remains in the ledger."""


class BudgetExhausted(Exception):
	"""The attempt or elapsed-time allowance is spent."""


MAX_CONSECUTIVE_FAILURES = 10


class ProtectedTeacher:
	def __init__(
		self, url: str, token: str, *, transport: httpx.AsyncBaseTransport | None = None
	) -> None:
		self.client = httpx.AsyncClient(
			base_url=f"{url.rstrip('/')}/",
			headers={"Authorization": f"Bearer {token}"},
			timeout=300,
			transport=transport,
		)
		self.challenges = 0
		self.challenge_seconds = 0.0
		self.hashes = 0
		self.events: list[dict[str, Any]] = []
		self.event_path: Path | None = None
		self.example_id = ""
		self.contract = ""
		self.max_requests = 2**63
		self.started = time.monotonic()
		self.deadline = float("inf")
		self.last_event_id = ""
		self.prompt_request_id = ""

	def remaining(self) -> float:
		return self.deadline - time.monotonic()

	def _write(self, event: dict[str, Any]) -> None:
		if self.event_path:
			with self.event_path.open("a", encoding="utf-8") as output:
				output.write(json.dumps(event) + "\n")
		self.events.append(event)

	async def _post(self, payload: dict[str, Any], proof: str | None = None) -> httpx.Response:
		count = sum(event["event"] == "attempt" for event in self.events)
		if count >= self.max_requests or self.remaining() <= 0:
			raise BudgetExhausted
		request_id = uuid4().hex
		self.last_event_id = request_id
		if proof is None:
			self.prompt_request_id = request_id
		started = time.monotonic()
		self._write(
			{
				"event": "attempt",
				"id": request_id,
				"policy_request_id": self.prompt_request_id,
				"example_id": self.example_id,
				"index": count + 1,
				"kind": "proof" if proof else "query",
				"at_seconds": started - self.started,
				"contract": self.contract,
			}
		)
		headers = {"X-Benchmark-Request": self.prompt_request_id}
		if proof:
			headers["X-Harness-Proof"] = proof
		outcome: dict[str, Any] = {"event": "outcome", "id": request_id, "status": "failed"}
		try:
			async with asyncio.timeout(min(300.0, self.remaining())):
				response = await self.client.post(
					"v1/teacher/generate", json=payload, headers=headers
				)
			outcome.update(
				http_status=response.status_code,
				decision=response.headers.get("X-Harness-Decision"),
			)
			outcome["status"] = (
				"challenge"
				if response.status_code == 428
				else (
					"blocked"
					if response.status_code == 403
					and response.headers.get("X-Harness-Decision") == "block"
					else "received"
					if response.is_success
					else "failed"
				)
			)
			if response.is_success:
				generation = TeacherGeneration.model_validate(response.json())
				outcome["output_tokens"] = generation.usage.output_tokens
			return response
		except (httpx.TransportError, TimeoutError, ValueError) as error:
			outcome["error"] = type(error).__name__
			outcome["status"] = "deadline" if self.remaining() <= 0 else "failed"
			raise AttemptFailed(type(error).__name__) from error
		finally:
			outcome["at_seconds"] = time.monotonic() - self.started
			outcome["latency_seconds"] = time.monotonic() - started
			self._write(outcome)

	async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
		payload = request.model_dump(mode="json")
		response = await self._post(payload)
		if response.status_code == 428:
			challenge = response.json()["challenge"]
			started = time.monotonic()
			self.challenges += 1
			counter = 0
			try:
				while True:
					if self.remaining() <= 0:
						raise BudgetExhausted
					digest = hashlib.sha256(f"{challenge['prefix']}{counter}".encode()).digest()
					self.hashes += 1
					if 256 - int.from_bytes(digest).bit_length() >= challenge["difficulty"]:
						break
					counter += 1
					if counter % 4096 == 0:
						await asyncio.sleep(0)
			finally:
				elapsed = time.monotonic() - started
				self.challenge_seconds += elapsed
				self._write(
					{
						"event": "solve",
						"policy_request_id": self.prompt_request_id,
						"hashes": counter + 1,
						"seconds": elapsed,
						"hardware": platform.machine(),
						"processor": platform.processor(),
					}
				)
			response = await self._post(payload, f"{challenge['nonce']}:{counter}")
		if response.status_code == 403 and response.headers.get("X-Harness-Decision") == "block":
			raise PolicyRefusal
		if response.status_code in {401, 403}:
			response.raise_for_status()
		if response.is_error:
			raise AttemptFailed(f"HTTP {response.status_code}")
		return TeacherGeneration.model_validate(response.json())

	async def close(self) -> None:
		await self.client.aclose()


def event_summary(
	events: list[dict[str, Any]],
	levels: tuple[int, ...] = (1, 2, 3, 4),
	*,
	budget: int | None = None,
) -> dict[str, Any]:
	"""J counts attempts through first reach; Q counts released outputs before that attempt.

	No reach is right-censored. Missing policy attribution is reported separately.
	"""
	attempts = [
		e for e in events if e["event"] == "attempt" and (budget is None or e["index"] <= budget)
	]
	ids = {e["id"] for e in attempts}
	outcomes = {e["id"]: e for e in events if e["event"] == "outcome" and e["id"] in ids}
	decisions = {e["policy_request_id"]: e for e in events if e["event"] == "policy"}
	counts = {
		status: sum(e["status"] == status for e in outcomes.values())
		for status in ("received", "blocked", "failed", "deadline", "challenge")
	}
	level_results: dict[str, Any] = {}
	for level in range(1, 5):
		if level not in levels:
			level_results[f"L{level}"] = None
			continue
		reach = next(
			(
				a
				for a in attempts
				if decisions.get(a["policy_request_id"], {}).get("level", -1) >= level
			),
			None,
		)
		known = all(
			a["policy_request_id"] in decisions
			for a in attempts
			if a["kind"] == "query" and outcomes.get(a["id"], {}).get("decision") is not None
		)
		level_results[f"L{level}"] = {
			"J": reach["index"] if reach else len(attempts),
			"Q": sum(
				o["status"] == "received" and o["at_seconds"] < reach["at_seconds"]
				for o in outcomes.values()
			)
			if reach
			else counts["received"],
			"reached": reach is not None,
			"censored": reach is None,
			"attribution_complete": known and bool(decisions),
		}
	tokens = [e.get("output_tokens") for e in outcomes.values() if e["status"] == "received"]
	return {
		"queries": len(attempts),
		**counts,
		"pending": len(ids - outcomes.keys()),
		"released_output_tokens": sum(tokens) if all(t is not None for t in tokens) else None,
		"levels": level_results,
		"proof_requests": sum(a["kind"] == "proof" for a in attempts),
	}


def campaign_contract(prompts: Path, options: CollectionOptions) -> str:
	return hashlib.sha256(
		prompts.read_bytes() + json.dumps(asdict(options), sort_keys=True).encode()
	).hexdigest()


async def collect_campaign(
	*,
	prompts_path: Path,
	ledger_path: Path,
	summary_path: Path,
	teacher: ProtectedTeacher,
	options: CollectionOptions,
	max_requests: int | None = None,
	deadline_seconds: float = 86400,
	decision_log: Path | None = None,
	levels: tuple[int, ...] = (1, 2, 3, 4),
) -> dict[str, Any]:
	"""Run a fresh campaign or reuse a completed one; never replenish an interrupted budget."""
	from distillation.benchmarks.models import save_json

	try:
		prompts = read_jsonl(prompts_path, PromptRecord)
		if len({p.example_id for p in prompts}) != len(prompts) or any(
			p.split == "test" for p in prompts
		):
			raise ValueError("campaign requires unique non-test prompts")
		limit = max_requests if max_requests is not None else len(prompts)
		if limit < 1 or deadline_seconds <= 0:
			raise ValueError("campaign budgets must be positive")
		contract = hashlib.sha256(
			(
				f"campaign-v2:{campaign_contract(prompts_path, options)}"
				f":{limit}:{deadline_seconds}:{levels}"
			).encode()
		).hexdigest()
		events = summary_path.with_suffix(".events.jsonl")
		if await AsyncPath(summary_path).exists():
			summary = json.loads(await AsyncPath(summary_path).read_text())
			if summary["contract"] != contract:
				raise ValueError("campaign options changed; use a new run directory")
			for key, path in (("ledger_sha256", ledger_path), ("events_sha256", events)):
				if hashlib.sha256(await AsyncPath(path).read_bytes()).hexdigest() != summary[key]:
					raise ValueError("completed campaign artifacts changed")
			return dict(summary)
		if await AsyncPath(ledger_path).exists() or await AsyncPath(events).exists():
			raise ValueError(
				"interrupted campaign retained its spent budget; keep its artifacts and "
				"declare a new campaign directory with fresh policy state"
			)
		ledger_path.parent.mkdir(parents=True, exist_ok=True)
		await AsyncPath(ledger_path).touch()
		events.parent.mkdir(parents=True, exist_ok=True)
		await AsyncPath(events).touch()
		teacher.event_path, teacher.contract = events, contract
		teacher.max_requests = limit
		teacher.started = time.monotonic()
		teacher.deadline = teacher.started + deadline_seconds
		attempt_log = ledger_path.with_name(f"{ledger_path.stem}.attempts.jsonl")
		reason = "prompts_exhausted"
		consecutive = 0
		for prompt in prompts:
			teacher.example_id = prompt.example_id
			try:
				record = await _collect_one(
					prompt, teacher, replace(options, max_attempts=1), attempt_log, lambda: None
				)
			except BudgetExhausted:
				reason = "deadline" if teacher.remaining() <= 0 else "attempt_budget"
				break
			except PolicyRefusal:
				consecutive = 0
			except AttemptFailed:
				consecutive += 1
				if consecutive >= MAX_CONSECUTIVE_FAILURES:
					reason = "service_failures"
					break
			else:
				record.attempt_id = teacher.last_event_id
				append_jsonl(ledger_path, [record])
				consecutive = 0
		if decision_log and await AsyncPath(decision_log).exists():
			wanted = {e["policy_request_id"] for e in teacher.events if e["event"] == "attempt"}
			for line in (await AsyncPath(decision_log).read_text()).splitlines():
				decision = json.loads(line)
				if (
					decision.get("event") == "decision"
					and decision.get("benchmark_request") in wanted
				):
					teacher._write(
						{
							"event": "policy",
							"policy_request_id": decision["benchmark_request"],
							"level": decision["level"],
							"action": decision["action"],
						}
					)
		summary = {
			"contract": contract,
			"attempt_budget": limit,
			"deadline_seconds": deadline_seconds,
			"termination": reason,
			"elapsed_seconds": time.monotonic() - teacher.started,
			**event_summary(teacher.events, levels),
			"ledger_records": len(read_jsonl(ledger_path, LedgerRecord)),
			"challenges": teacher.challenges,
			"challenge_seconds": teacher.challenge_seconds,
			"hashes": teacher.hashes,
			"ledger_sha256": hashlib.sha256(await AsyncPath(ledger_path).read_bytes()).hexdigest(),
			"events_sha256": hashlib.sha256(await AsyncPath(events).read_bytes()).hexdigest(),
		}
		save_json(summary_path, summary)
		return summary
	finally:
		await teacher.close()
