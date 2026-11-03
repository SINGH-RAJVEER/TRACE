"""Campaign allowance, protected teacher client and event summaries shared by gateway campaigns."""

import asyncio
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from distillation.proof import SOLVER_VERSION, ProofSolver, SolveMetrics
from distillation.schemas import (
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


def service_failures(summary: dict[str, Any]) -> list[str]:
	"""Accounts a campaign stopped after consecutive transport failures.

	Policy refusals reset the failure count, so these stops measure the service, not a defense.
	"""
	reasons = summary.get("termination_by_account") or {"campaign": summary.get("termination")}
	return sorted(account for account, reason in reasons.items() if reason == "service_failures")


@dataclass
class CampaignAllowance:
	"""One HTTP allowance, shared even when policy accounts or horizons change."""

	limit: int
	used: int = 0
	exhausted: asyncio.Event | None = None


class ProtectedTeacher:
	def __init__(
		self,
		url: str,
		token: str,
		*,
		transport: httpx.AsyncBaseTransport | None = None,
		solver: ProofSolver | None = None,
	) -> None:
		self.client = httpx.AsyncClient(
			base_url=f"{url.rstrip('/')}/",
			headers={"Authorization": f"Bearer {token}"} if token else {},
			timeout=300,
			transport=transport,
		)
		self.solver = solver or ProofSolver()
		self._owns_solver = solver is None
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
		self.allowance: CampaignAllowance | None = None
		self.account_limit: int | None = None
		self.account = ""
		self.domain = ""
		self.scheduled_id = ""

	def remaining(self) -> float:
		return self.deadline - time.monotonic()

	def _write(self, event: dict[str, Any]) -> None:
		if self.account:
			event.update(account=self.account, domain=self.domain, scheduled_id=self.scheduled_id)
		if self.event_path:
			with self.event_path.open("a", encoding="utf-8") as output:
				output.write(json.dumps(event) + "\n")
		self.events.append(event)

	async def _post(self, payload: dict[str, Any], proof: str | None = None) -> httpx.Response:
		count = (
			self.allowance.used
			if self.allowance
			else sum(event["event"] == "attempt" for event in self.events)
		)
		account_spent = (
			sum(e["event"] == "attempt" and e.get("account") == self.account for e in self.events)
			if self.account_limit is not None
			else 0
		)
		if self.account_limit is not None and account_spent >= self.account_limit:
			raise BudgetExhausted
		if count >= self.max_requests or self.remaining() <= 0:
			raise BudgetExhausted
		if self.allowance:
			self.allowance.used += 1
			if self.allowance.used >= self.allowance.limit and self.allowance.exhausted:
				self.allowance.exhausted.set()
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
		if self.domain:
			headers["X-Gateway-Domain"] = self.domain
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
			metrics = SolveMetrics(workers=self.solver.workers)
			# Server expiries use Unix time; process workers share the host monotonic clock.
			expires = started + max(0.0, float(challenge["expires_at"]) - time.time())
			deadline = min(self.deadline, expires)
			try:
				counter = await self.solver.solve(
					challenge["prefix"], challenge["difficulty"], deadline, metrics
				)
				if counter is None:
					if self.remaining() <= 0:
						raise BudgetExhausted
					raise AttemptFailed("proof challenge expired")
			except Exception:
				if metrics.status == "cancelled":
					metrics.status = "error"
				raise
			finally:
				elapsed = time.monotonic() - started
				self.hashes += metrics.hashes
				self.challenge_seconds += elapsed
				self._write(
					{
						"event": "solve",
						"policy_request_id": self.prompt_request_id,
						**asdict(metrics),
						"seconds": elapsed,
						"solver": SOLVER_VERSION,
						"hardware": platform.machine(),
						"processor": platform.processor(),
					}
				)
			if time.monotonic() >= deadline:
				if self.remaining() <= 0:
					raise BudgetExhausted
				raise AttemptFailed("proof challenge expired before redemption")
			response = await self._post(payload, f"{challenge['nonce']}:{counter}")
		if response.status_code == 403 and response.headers.get("X-Harness-Decision") == "block":
			raise PolicyRefusal
		if response.status_code in {401, 403}:
			response.raise_for_status()
		if response.is_error:
			raise AttemptFailed(f"HTTP {response.status_code}")
		return TeacherGeneration.model_validate(response.json())

	async def close(self) -> None:
		try:
			await self.client.aclose()
		finally:
			if self._owns_solver:
				await self.solver.close()


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
