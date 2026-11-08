"""Campaign allowance, protected teacher client and event summaries shared by gateway campaigns."""

import asyncio
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from harness.campaign import load_campaign_config

from distillation.proof import SOLVER_VERSION, ProofSolver, SolveMetrics
from distillation.schemas import (
	TeacherGeneration,
	TeacherGenerationRequest,
)


class PolicyRefusal(Exception):
	"""A refusal explicitly attributed to the policy."""


class AttemptFailed(Exception):
	"""An unsuccessful attempt whose cost remains in the ledger."""


class PolicyRestricted(AttemptFailed):
	"""A 429 attributed to the harness, raised only when a client opts into attribution.

	It subclasses AttemptFailed so that callers unaware of it still see an unsuccessful attempt.
	"""

	def __init__(self, restriction: dict[str, Any]) -> None:
		super().__init__(f"policy restriction: {restriction['cause']}")
		self.restriction = restriction


class BudgetExhausted(Exception):
	"""The attempt or elapsed-time allowance is spent."""


MAX_CONSECUTIVE_FAILURES = 10
ATTRIBUTION_VERSION = "campaign-attribution-v1"
RESTRICTION_STRATEGIES = ("give_up", "wait_and_retry")
CAMPAIGN_CONFIG_FILE = "campaign-config.json"
DECISIONS_SNAPSHOT_FILE = "decisions.snapshot.jsonl"
# Harness 429 codes. A code outside this table stays cause-attributed but of unknown kind.
OUTPUT_RECEIPT_HEADER = "X-Harness-Output-Receipt"
RESTRICTION_KINDS = {
	"campaign_throttled": "campaign",
	"token_budget_exceeded": "entitlement",
	"rolling_budget_exceeded": "entitlement",
	"organization_rolling_budget_exceeded": "entitlement",
	"supervision_budget_exceeded": "entitlement",
	"tier_budget_exceeded": "entitlement",
	"in_flight_limit": "rate",
	"output_withheld": "output",
}


def _error_body(response: httpx.Response) -> dict[str, Any]:
	try:
		body = response.json()
	except ValueError:
		return {}
	return body if isinstance(body, dict) else {}


def _error_code(body: dict[str, Any]) -> str | None:
	detail = body.get("detail")
	code = detail.get("code") if isinstance(detail, dict) else None
	return code if isinstance(code, str) else None


def _seconds(value: Any) -> float | None:
	"""A finite nonnegative delay. An HTTP-date Retry-After stays unknown: clocks differ."""
	try:
		seconds = float(value)
	except TypeError, ValueError:
		return None
	return seconds if math.isfinite(seconds) and seconds >= 0 else None


def classify_restriction(response: httpx.Response) -> dict[str, Any] | None:
	"""The cause and retry timing of a harness 429 or a withheld output; else a service failure.

	An upstream or proxy 429 has neither a harness decision header nor a known code, so it is
	not relabeled as a defense. A harness decision with an unrecognized code keeps kind unknown.
	"""
	body = _error_body(response)
	code = _error_code(body)
	if response.status_code != 429 and not (
		response.status_code == 502 and code == "output_withheld"
	):
		return None
	decision = response.headers.get("X-Harness-Decision")
	if decision is None and code not in RESTRICTION_KINDS:
		return None
	campaign = body.get("campaign")
	campaign = campaign if isinstance(campaign, dict) else {}
	detail = body.get("detail")
	detail = detail if isinstance(detail, dict) else {}
	retry_after = _seconds(response.headers.get("Retry-After"))
	if retry_after is None:
		retry_after = _seconds(campaign.get("retry_after_seconds"))
	return {
		"kind": RESTRICTION_KINDS.get(code or "", "unknown"),
		"cause": code,
		"decision": decision,
		"retry_after_seconds": retry_after,
		"retry_at_server_seconds": campaign.get("throttle_until", detail.get("reset_at")),
	}


def output_receipt(response: httpx.Response) -> dict[str, Any] | None:
	"""The gateway's output receipt, and whether the released hash matches what arrived."""
	header = response.headers.get(OUTPUT_RECEIPT_HEADER)
	if header is None:
		return None
	try:
		receipt = json.loads(header)
	except ValueError:
		return {"valid": False}
	if not isinstance(receipt, dict):
		return {"valid": False}
	matches = None
	if response.is_success:
		try:
			text = TeacherGeneration.model_validate(response.json()).message.content
		except ValueError:
			text = None
		matches = text is not None and hashlib.sha256(text.encode()).hexdigest() == receipt.get(
			"released_sha256"
		)
	return {**receipt, "valid": True, "released_hash_matches": matches}


@dataclass(frozen=True)
class ExperimentBinding:
	"""An opt-in declaration of what a gateway campaign measures and how it handles 429s."""

	base_policy: str
	restriction_strategy: str
	max_retries_per_prompt: int = 0
	campaign_config: Path | None = None
	campaign_mode: str | None = None

	def record(self) -> dict[str, Any]:
		"""The contract member; loading the configuration refuses an invalid or wrong-mode file."""
		if self.base_policy == "none":
			raise ValueError("an experiment binding needs a harness policy, not a direct teacher")
		if self.restriction_strategy not in RESTRICTION_STRATEGIES:
			raise ValueError(f"restriction strategy must be one of {RESTRICTION_STRATEGIES}")
		retrying = self.restriction_strategy == "wait_and_retry"
		if self.max_retries_per_prompt < (1 if retrying else 0) or (
			not retrying and self.max_retries_per_prompt
		):
			raise ValueError("wait_and_retry needs positive retries; give_up allows none")
		if (self.campaign_config is None) != (self.campaign_mode is None):
			raise ValueError("a campaign configuration and its declared mode go together")
		campaign = None
		if self.campaign_config is not None:
			config = load_campaign_config(self.campaign_config)
			if config.mode != self.campaign_mode:
				raise ValueError(
					f"campaign configuration has mode {config.mode}, not {self.campaign_mode}"
				)
			campaign = {
				"config_sha256": hashlib.sha256(self.campaign_config.read_bytes()).hexdigest(),
				"digest": config.digest,
				"version": config.version,
				"mode": config.mode,
				"session": config.session.version if config.session is not None else None,
			}
		return {
			"attribution": ATTRIBUTION_VERSION,
			"base_policy": self.base_policy,
			"campaign": campaign,
			"restriction_strategy": {
				"name": self.restriction_strategy,
				"max_retries_per_prompt": self.max_retries_per_prompt,
				"missing_retry_after": "give up on the prompt",
				"retry_submission": "fresh query; a held proof is not re-sent",
				"charged": "each rejected submission spends the HTTP allowance; waits spend time",
			},
		}


def policy_binding(log: bytes, campaign: dict[str, Any] | None) -> dict[str, Any]:
	"""What a decision log proves was served, compared with the declared configuration.

	The revision is observed here because the API does not report it before a run. `bound`
	needs one revision and matching detector evidence, `unknown` means the log has no revision,
	and `conflict` lists what disagrees. Export and resume refuse anything but `bound`.
	"""
	revisions: set[str] = set()
	digests: set[str] = set()
	unparsable = decisions = throttles = 0
	for line in log.splitlines():
		if not line.strip():
			continue
		try:
			row = json.loads(line)
		except ValueError:
			unparsable += 1
			continue
		if not isinstance(row, dict):
			unparsable += 1
			continue
		if isinstance(row.get("policy_revision"), str):
			revisions.add(row["policy_revision"])
		evidence = row.get("campaign")
		if isinstance(evidence, dict) and isinstance(evidence.get("config_sha256"), str):
			digests.add(evidence["config_sha256"])
		decisions += row.get("event") == "decision"
		throttles += row.get("event") == "campaign_throttle"
	problems = []
	if unparsable:
		problems.append(f"{unparsable} unparsable decision log rows")
	if len(revisions) > 1:
		problems.append("more than one policy revision")
	declared = campaign["digest"] if campaign else None
	if digests - ({declared} if declared else set()):
		problems.append("detector evidence from an undeclared campaign configuration")
	if declared and (decisions or revisions) and declared not in digests:
		problems.append("no row carries evidence from the declared campaign configuration")
	if throttles and (campaign is None or campaign["mode"] != "throttle"):
		problems.append("campaign throttle logged by an arm that did not declare throttling")
	status = "conflict" if problems else "bound" if revisions else "unknown"
	return {
		"status": status,
		"policy_revision": next(iter(revisions)) if len(revisions) == 1 else None,
		"policy_revisions": sorted(revisions),
		"campaign_digests": sorted(digests),
		"problems": problems,
		"decision_log_sha256": hashlib.sha256(log).hexdigest(),
	}


def verify_experiment_binding(
	directory: Path, contract: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
	"""Refuse a bound campaign whose frozen configuration or served policy does not check out."""
	experiment = contract.get("experiment")
	if not isinstance(experiment, dict) or "campaign" not in experiment:
		raise ValueError("gateway contract lacks its experiment binding")
	campaign = experiment["campaign"]
	if campaign is not None:
		path = directory / CAMPAIGN_CONFIG_FILE
		if (
			hashlib.sha256(path.read_bytes()).hexdigest() != campaign["config_sha256"]
			or load_campaign_config(path).digest != campaign["digest"]
		):
			raise ValueError("frozen campaign configuration does not match the contract")
	binding = policy_binding((directory / DECISIONS_SNAPSHOT_FILE).read_bytes(), campaign)
	if summary.get("policy_binding") != binding:
		raise ValueError("decision log snapshot does not match the recorded policy binding")
	if binding["status"] != "bound":
		problems = "; ".join(binding["problems"]) or "no policy revision was logged"
		raise ValueError(f"policy binding is {binding['status']}: {problems}")
	return binding


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
		# Off by default: a 429 stays `failed`, with the legacy events and exception.
		self.attribute_restrictions = False

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
		attempt: dict[str, Any] = {
			"event": "attempt",
			"id": request_id,
			"policy_request_id": self.prompt_request_id,
			"example_id": self.example_id,
			"index": count + 1,
			"kind": "proof" if proof else "query",
			"at_seconds": started - self.started,
			"contract": self.contract,
		}
		if self.attribute_restrictions:
			attempt["attribution"] = ATTRIBUTION_VERSION
		self._write(attempt)
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
			receipt = output_receipt(response)
			if receipt is not None:
				outcome["output_receipt"] = receipt
			if self.attribute_restrictions and response.is_error:
				outcome["code"] = _error_code(_error_body(response))
				restriction = classify_restriction(response)
				if restriction is not None:
					outcome.update(status="restricted", restriction=restriction)
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
		if self.attribute_restrictions:
			restriction = classify_restriction(response)
			if restriction is not None:
				raise PolicyRestricted(restriction)
		if response.status_code in {401, 403}:
			response.raise_for_status()
		if response.is_error:
			raise AttemptFailed(f"HTTP {response.status_code}")
		return TeacherGeneration.model_validate(response.json())

	def record_restriction_event(self, event: str, **fields: Any) -> None:
		"""Charge a retry-after decision to the attacker, in the campaign's event log."""
		self._write(
			{
				"event": event,
				"id": self.last_event_id,
				"policy_request_id": self.prompt_request_id,
				"at_seconds": time.monotonic() - self.started,
				"contract": self.contract,
				**fields,
			}
		)

	async def close(self) -> None:
		try:
			await self.client.aclose()
		finally:
			if self._owns_solver:
				await self.solver.close()
