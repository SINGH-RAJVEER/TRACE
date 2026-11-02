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

from distillation.adaptive import CoverageSelector, UncertaintyHook
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
from distillation.detectability_gate import contract_record
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
	variants: dict[str, Any] | None = None,
	selector_seed: int = 42,
	uncertainty_hook: UncertaintyHook | None = None,
	uncertainty_hook_id: str | None = None,
	detectability: dict[str, Any] | None = None,
	alternative_url: str | None = None,
	alternative_share: float = 0.0,
) -> dict[str, Any]:
	"""One in-flight request per account; HTTP failures and proofs spend the global budget.

	Completed artifacts are verified before reuse. An interrupted campaign cannot restart
	with fresh policy state or replenish its spent allowance in the same output directory.
	A protected arm passes its detectability gate record, which binds the contract and is
	saved beside it; unprotected arms keep their earlier contracts.
	"""
	if not 0 <= alternative_share <= 1 or bool(alternative_url) != bool(alternative_share):
		raise ValueError("alternative teacher URL and positive share must be supplied together")
	if solver_workers < 1:
		raise ValueError("solver workers must be positive")
	variants = variants or {}
	online = bool(variants.get("online"))
	account_limit = variants.get("per_account_limit")
	if account_limit is not None and (type(account_limit) is not int or account_limit < 1):
		raise ValueError("per-account HTTP limit must be positive")
	round_size = variants.get("round_size", 16)
	if type(round_size) is not int or round_size < 1:
		raise ValueError("round size must be positive")
	if uncertainty_hook is not None and (not online or not uncertainty_hook_id):
		raise ValueError("uncertainty hook requires A8 and a reproducible hook identity")
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
	if variants:
		contract["variants"] = variants
		contract["selector_seed"] = selector_seed
		contract["uncertainty_hook_id"] = uncertainty_hook_id
	if alternative_url:
		contract["alternative_teacher"] = {
			"url": alternative_url,
			"share": alternative_share,
			"assignment": "sha256(seed:request_id) threshold",
			"seed": selector_seed,
			"offline_compute": "none",
		}
	if detectability is not None:
		contract["detectability"] = contract_record(detectability)
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
	if detectability is not None:
		save_json(output / "detectability.json", detectability)
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
	failures: dict[str, int] = defaultdict(int)
	teachers: dict[tuple[str, str], ProtectedTeacher] = {}

	async def worker(account: str, requests: list[GatewayRequest]) -> None:
		consecutive = failures[account]
		terminations[account] = "prompts_exhausted"
		for row in sorted(requests, key=lambda r: r.time):
			fraction = (
				int.from_bytes(
					hashlib.sha256(f"{selector_seed}:{row.request_id}".encode()).digest(), "big"
				)
				/ 2**256
			)
			selected_url = (
				alternative_url if alternative_url and fraction < alternative_share else url
			)
			teacher_key = (account, selected_url)
			if teacher_key not in teachers:
				teacher = ProtectedTeacher(
					selected_url,
					keys.get(account, keys.get("default", "")),
					transport=transport,
					solver=solver,
				)
				teachers[teacher_key] = teacher
				clients.append(teacher)
				teacher.account_limit = account_limit
				teacher.account = account
				teacher.events, teacher.event_path = events, events_path
				teacher.contract = contract_hash
				teacher.allowance, teacher.max_requests = allowance, attempt_budget
				teacher.started, teacher.deadline = started, started + deadline_seconds
			teacher = teachers[teacher_key]
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
				terminations[account] = (
					"deadline"
					if teacher.remaining() <= 0
					else "account_limit"
					if account_limit and allowance.used < attempt_budget
					else "attempt_budget"
				)
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
				if alternative_url:
					record.teacher.endpoint = selected_url
				append_jsonl(ledger, [record])
				consecutive = 0

		failures[account] = consecutive

	async def adaptive_worker() -> None:
		selector = CoverageSelector(rows, selector_seed)
		round_index, cursor = 0, 0
		next_at = 0.0
		interval = rows[1].time - rows[0].time if len(rows) > 1 else 0.0
		previous_feedback: list[dict[str, Any]] = []
		while (
			selector.pending
			and allowance.used < attempt_budget
			and time.monotonic() < started + deadline_seconds
		):
			batch, scores = (
				selector.select(
					min(round_size, attempt_budget - allowance.used),
					read_jsonl(ledger, LedgerRecord),
					previous_feedback,
					uncertainty_hook,
				)
				if online
				else ([selector.pending.pop(next(iter(selector.pending)))], {})
			)
			event = {
				"event": "selection_round",
				"round": round_index,
				"at_seconds": time.monotonic() - started,
				"selected_ids": [r.request_id for r in batch],
				"scores": scores,
				"feedback": list(previous_feedback),
				"uncertainty_scores": dict(selector.uncertainty),
				"attempts_spent": allowance.used,
				"released_example_ids": [r.example_id for r in read_jsonl(ledger, LedgerRecord)],
				"contract": contract_hash,
			}
			with events_path.open("a") as stream:
				stream.write(json.dumps(event) + "\n")
			events.append(event)
			feedback_start = len(events)
			for row in batch:
				account = row.account
				if account_limit:
					while (
						sum(
							e["event"] == "attempt" and e.get("account") == rows[cursor].account
							for e in events
						)
						>= account_limit
					):
						cursor += 1
						if cursor >= len(rows):
							return
					account = rows[cursor].account
				await worker(
					account, [row.model_copy(update={"account": account, "time": next_at})]
				)
				next_at = time.monotonic() - started + interval
				if terminations[account] in {"deadline", "attempt_budget", "service_failures"}:
					return
			# Only HTTP-visible outcomes enter the selector; never policy sidecar decisions.
			previous_feedback.extend(
				{
					key: e[key]
					for key in ("event", "domain", "status", "latency_seconds", "http_status")
					if key in e
				}
				for e in events[feedback_start:]
				if e["event"] == "outcome"
			)
			round_index += 1
		if not terminations:
			terminations[rows[0].account] = "prompts_exhausted"
		if terminations:
			last_account = next(reversed(terminations))
			if allowance.used >= attempt_budget:
				terminations[last_account] = "attempt_budget"
			elif time.monotonic() >= started + deadline_seconds:
				terminations[last_account] = "deadline"

	tasks = (
		[asyncio.create_task(adaptive_worker())]
		if online or account_limit
		else [
			asyncio.create_task(worker(account, requests)) for account, requests in groups.items()
		]
	)
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
	if detectability is not None:
		summary["reading"] = detectability["reading"]
		summary["allow_undetected"] = detectability["allow_undetected"]
	save_json(summary_path, summary)
	return summary
