"""Execute immutable mixed-domain schedules with one durable campaign allowance."""

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from contextlib import suppress
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import httpx
from anyio import Path as AsyncPath
from harness.calibration import HORIZON_POLICY

from distillation.benchmarks.models import save_json
from distillation.campaign import (
	MAX_CONSECUTIVE_FAILURES,
	AttemptFailed,
	BudgetExhausted,
	CampaignAllowance,
	PolicyRefusal,
	ProtectedTeacher,
	event_summary,
)
from distillation.collection import CollectionOptions, _collect_one
from distillation.io import append_jsonl, read_jsonl
from distillation.proof import CHUNK_HASHES, SOLVER_VERSION, ProofSolver
from distillation.schemas import GatewayRequest, LedgerRecord


def digest(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


async def collect_gateway(
	*,
	schedule: Path,
	output: Path,
	url: str,
	keys: dict[str, str],
	options: CollectionOptions,
	attempt_budget: int,
	deadline_seconds: float = 86400,
	solver_workers: int = 4,
	decision_log: Path | None = None,
	transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
	"""One in-flight request per account; HTTP failures and proofs spend the global budget.

	Completed artifacts are verified before reuse. An interrupted campaign cannot restart
	with fresh policy state or replenish its spent allowance in the same output directory.
	"""
	if solver_workers < 1:
		raise ValueError("solver workers must be positive")
	rows = read_jsonl(schedule, GatewayRequest)
	if not rows or attempt_budget < 1 or not 0 < deadline_seconds < float("inf"):
		raise ValueError("a gateway campaign needs requests and finite positive budgets")
	if len({r.request_id for r in rows}) != len(rows) or len(
		{r.prompt.example_id for r in rows}
	) != len(rows):
		raise ValueError("campaign request and example IDs must be unique")
	if any(r.prompt.split != "train" for r in rows):
		raise ValueError("gateway extraction requires training prompts")
	if any(r.account not in keys and "default" not in keys for r in rows):
		raise ValueError("missing account credentials")
	contract = {
		"version": "gateway-execution-v2",
		"horizon_policy": HORIZON_POLICY,
		"solver": {
			"version": SOLVER_VERSION,
			"workers": solver_workers,
			"chunk_hashes": CHUNK_HASHES,
		},
		"schedule_sha256": digest(schedule),
		"options": asdict(options),
		"attempt_budget": attempt_budget,
		"deadline_seconds": deadline_seconds,
		"url": url,
		"credential_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in keys.items()},
		"decision_log": str(decision_log) if decision_log else None,
	}
	contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
	summary_path = output / "campaign.json"
	ledger, events_path = output / "ledger.jsonl", output / "campaign.events.jsonl"
	if summary_path.exists():
		summary = json.loads(summary_path.read_text())
		if summary["contract"] != contract_hash:
			raise ValueError("gateway campaign changed; use a new run directory")
		if summary["sha256"] != {"ledger": digest(ledger), "events": digest(events_path)}:
			raise ValueError("completed gateway campaign artifacts changed")
		return dict(summary)
	if await AsyncPath(output).exists() and any(
		[item async for item in AsyncPath(output).iterdir()]
	):
		raise ValueError("interrupted gateway campaign: preserve artifacts and use a new directory")
	await AsyncPath(output).mkdir(parents=True, exist_ok=False)
	save_json(output / "contract.json", contract)
	ledger.touch()
	events_path.touch()
	allowance = CampaignAllowance(attempt_budget, exhausted=asyncio.Event())
	solver = ProofSolver(solver_workers)
	try:
		await solver.warmup()
	except BaseException:
		await solver.close()
		raise
	started = time.monotonic()
	events: list[dict[str, Any]] = []
	clients: list[ProtectedTeacher] = []
	groups: dict[str, list[GatewayRequest]] = defaultdict(list)
	for row in rows:
		groups[row.account].append(row)
	semaphore = asyncio.Semaphore(options.concurrency)
	terminations: dict[str, str] = {}

	async def worker(account: str, requests: list[GatewayRequest]) -> None:
		teacher = ProtectedTeacher(
			url, keys.get(account, keys.get("default", "")), transport=transport, solver=solver
		)
		clients.append(teacher)
		teacher.account = account
		teacher.events, teacher.event_path = events, events_path
		teacher.contract = contract_hash
		teacher.allowance, teacher.max_requests = allowance, attempt_budget
		teacher.started, teacher.deadline = started, started + deadline_seconds
		consecutive = 0
		terminations[account] = "prompts_exhausted"
		for row in sorted(requests, key=lambda r: r.time):
			if allowance.used >= attempt_budget:
				terminations[account] = "attempt_budget"
				break
			wait = row.time - (time.monotonic() - started)
			if wait > 0:
				assert allowance.exhausted is not None
				with suppress(TimeoutError):
					await asyncio.wait_for(
						allowance.exhausted.wait(), timeout=min(wait, max(0, teacher.remaining()))
					)
			teacher.domain, teacher.example_id = row.domain, row.prompt.example_id
			teacher.scheduled_id = row.request_id
			try:
				async with semaphore:
					record = await _collect_one(
						row.prompt,
						teacher,
						replace(options, max_attempts=1, max_output_tokens=row.max_output_tokens),
						output / "attempts.jsonl",
						lambda: None,
					)
			except BudgetExhausted:
				terminations[account] = "deadline" if teacher.remaining() <= 0 else "attempt_budget"
				break
			except PolicyRefusal:
				consecutive = 0
			except AttemptFailed:
				if teacher.remaining() <= 0:
					terminations[account] = "deadline"
					break
				consecutive += 1
				if consecutive >= MAX_CONSECUTIVE_FAILURES:
					terminations[account] = "service_failures"
					break
			else:
				record.attempt_id = teacher.last_event_id
				append_jsonl(ledger, [record])
				consecutive = 0

	tasks = [asyncio.create_task(worker(account, requests)) for account, requests in groups.items()]
	try:
		await asyncio.gather(*tasks)
		elapsed = time.monotonic() - started
	finally:
		for task in tasks:
			if not task.done():
				task.cancel()
		await asyncio.gather(*tasks, return_exceptions=True)
		try:
			await asyncio.gather(*(client.close() for client in clients))
		finally:
			await solver.close()
	if decision_log:
		attempts = {e["policy_request_id"]: e for e in events if e["event"] == "attempt"}
		for line in (await AsyncPath(decision_log).read_text()).splitlines():
			decision = json.loads(line)
			request_id = decision.get("benchmark_request")
			if decision.get("event") == "decision" and request_id in attempts:
				event = {
					**attempts[request_id],
					"event": "policy",
					"level": decision["level"],
					"action": decision["action"],
				}
				with events_path.open("a") as stream:
					stream.write(json.dumps(event) + "\n")
				events.append(event)
	records = read_jsonl(ledger, LedgerRecord)
	summary = {
		"contract": contract_hash,
		"horizon_policy": HORIZON_POLICY,
		"solver": contract["solver"],
		"attempt_budget": attempt_budget,
		"deadline_seconds": deadline_seconds,
		"elapsed_seconds": elapsed,
		"termination_by_account": terminations,
		**event_summary(events),
		"index_unit": "global campaign HTTP submissions",
		"domains": {
			domain: event_summary([e for e in events if e.get("domain") == domain])
			for domain in sorted({r.domain for r in rows})
		},
		"ledger_records": len(records),
		"ledger_records_by_domain": {
			domain: sum(r.task == domain for r in records)
			for domain in sorted({r.domain for r in rows})
		},
		"hashes": sum(c.hashes for c in clients),
		"challenge_seconds": sum(c.challenge_seconds for c in clients),
		"sha256": {"ledger": digest(ledger), "events": digest(events_path)},
	}
	save_json(summary_path, summary)
	return summary
