"""Collect one ordered extraction campaign through the protected HTTP endpoint."""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from anyio import Path as AsyncPath

from distillation.collection import CollectionOptions, _collect_one, _existing_example_ids
from distillation.io import append_jsonl, read_jsonl
from distillation.schemas import PromptRecord, TeacherGeneration, TeacherGenerationRequest


class PolicyRefusal(Exception):
	"""An explicitly attributed refusal, which consumes a query but returns no training text."""


class AttemptFailed(Exception):
	"""One request that failed without a policy refusal, such as a server error, timeout, or
	rejected proof. It consumes a query and returns no training text (paper Section IV-C)."""


# A run of failures this long means the service, not one request, is broken; stop collection.
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

	async def _generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
		payload = request.model_dump(mode="json")
		response = await self.client.post("v1/teacher/generate", json=payload)
		if response.status_code == 428:
			# The same payload redeems a challenge; it must not consume another policy attempt.
			from distillation.benchmarks.traffic_runner import solve_challenge

			challenge = response.json()["challenge"]
			started = time.perf_counter()
			counter = await asyncio.to_thread(
				solve_challenge, challenge["prefix"], challenge["difficulty"]
			)
			self.challenge_seconds += time.perf_counter() - started
			self.challenges += 1
			response = await self.client.post(
				"v1/teacher/generate",
				json=payload,
				headers={"X-Harness-Proof": f"{challenge['nonce']}:{counter}"},
			)
		if response.status_code == 403 and response.headers.get("X-Harness-Decision") == "block":
			raise PolicyRefusal
		# Missing or rejected credentials, and generic 403s without a harness decision, mean the
		# run is misconfigured, so they stop it. Other failures are charged to the budget as
		# failed attempts and never counted as evidence that the policy blocked a request.
		if response.status_code == 401 or (
			response.status_code == 403 and "X-Harness-Decision" not in response.headers
		):
			response.raise_for_status()
		if response.is_error:
			raise AttemptFailed(f"HTTP {response.status_code}")
		return TeacherGeneration.model_validate(response.json())

	async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
		"""One attempted request. Transport errors and timeouts are failed attempts."""
		try:
			return await self._generate(request)
		except httpx.TransportError as error:
			raise AttemptFailed(type(error).__name__) from error

	async def close(self) -> None:
		await self.client.aclose()


def campaign_contract(prompts: Path, options: CollectionOptions) -> str:
	from dataclasses import asdict

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
) -> dict[str, Any]:
	"""One account, one ordered query stream, with successful outputs in an immutable ledger.

	A completed campaign is reused. An interrupted campaign is archived and restarted in
	full against a fresh service. Its partial responses never mix with the new campaign.
	"""
	from distillation.benchmarks.models import save_json

	try:
		contract = campaign_contract(prompts_path, options)
		prompts = read_jsonl(prompts_path, PromptRecord)
		if len({p.example_id for p in prompts}) != len(prompts):
			raise ValueError("duplicate campaign example_id")
		_existing_example_ids(ledger_path, prompts, options)
		if any(prompt.split == "test" for prompt in prompts):
			raise ValueError("campaign cannot collect held-out test prompts")
		events = summary_path.with_suffix(".events.jsonl")
		if await AsyncPath(summary_path).exists():
			summary: dict[str, Any] = json.loads(await AsyncPath(summary_path).read_text())
			if summary["contract"] != contract:
				raise ValueError("campaign options changed; choose a new run directory")
			if (
				summary["ledger_sha256"]
				!= hashlib.sha256(await AsyncPath(ledger_path).read_bytes()).hexdigest()
			):
				raise ValueError("completed campaign ledger changed")
			return summary
		if await AsyncPath(events).exists():
			for line in (await AsyncPath(events).read_text()).splitlines():
				if json.loads(line)["contract"] != contract:
					raise ValueError("campaign options changed; choose a new run directory")
		attempts = ledger_path.with_name(f"{ledger_path.stem}.attempts.jsonl")
		if await AsyncPath(ledger_path).exists() or await AsyncPath(events).exists():
			archive = ledger_path.parent / "interrupted" / uuid4().hex
			await AsyncPath(archive).mkdir(parents=True)
			for path in (ledger_path, events, attempts):
				if await AsyncPath(path).exists():
					await AsyncPath(path).rename(archive / path.name)
		await AsyncPath(ledger_path.parent).mkdir(parents=True, exist_ok=True)
		await AsyncPath(ledger_path).touch(exist_ok=True)
		blocked = failed = consecutive = 0
		for index, prompt in enumerate(prompts, 1):
			status, error_detail = "error", None
			challenges = teacher.challenges
			challenge_seconds = teacher.challenge_seconds
			try:
				record = await _collect_one(prompt, teacher, options, attempts, lambda: None)
			except PolicyRefusal:
				blocked += 1
				status = "blocked"
				consecutive = 0
			except AttemptFailed as error:
				failed += 1
				consecutive += 1
				status, error_detail = "failed", str(error)
			else:
				status = "received"
				consecutive = 0
				append_jsonl(ledger_path, [record])
			finally:
				with events.open("a", encoding="utf-8") as output:
					output.write(
						json.dumps(
							{
								"example_id": prompt.example_id,
								"status": status,
								"error": error_detail,
								"contract": contract,
								"challenges": teacher.challenges - challenges,
								"challenge_seconds": teacher.challenge_seconds - challenge_seconds,
							}
						)
						+ "\n"
					)
			from distillation.progress import console

			console.print(f"Protected collection {index}/{len(prompts)}: {error_detail or status}")
			if consecutive >= MAX_CONSECUTIVE_FAILURES:
				raise RuntimeError(
					f"{consecutive} consecutive failed requests; the protected service is down"
				)
		summary = {
			"contract": contract,
			"queries": len(prompts),
			"blocked": blocked,
			# Failed attempts are charged to the budget and never replaced.
			"failed": failed,
			"received": len(prompts) - blocked - failed,
			"ledger_records": sum(
				bool(line.strip())
				for line in (await AsyncPath(ledger_path).read_text()).splitlines()
			),
			"interrupted_attempts": len(list(ledger_path.parent.glob("interrupted/*"))),
			"challenges": teacher.challenges,
			"challenge_seconds": teacher.challenge_seconds,
			"ledger_sha256": hashlib.sha256(await AsyncPath(ledger_path).read_bytes()).hexdigest(),
		}
		save_json(summary_path, summary)
		return summary
	finally:
		await teacher.close()
