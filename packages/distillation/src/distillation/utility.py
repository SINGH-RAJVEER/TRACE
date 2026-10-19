"""Task-scored utility under identical synthetic arrival schedules and bounded concurrency."""

import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
from anyio import Path as AsyncPath
from harness.calibration import BenignRequest

from distillation.benchmarks.models import BenchmarkItem, save_json
from distillation.benchmarks.scoring import evaluate_answer
from distillation.campaign import AttemptFailed, BudgetExhausted, PolicyRefusal, ProtectedTeacher
from distillation.proof import CHUNK_HASHES, SOLVER_VERSION, ProofSolver
from distillation.schemas import TeacherGeneration, TeacherGenerationRequest


async def measure_utility(
	*,
	requests: list[BenignRequest],
	references: dict[str, BenchmarkItem],
	url: str,
	keys: dict[str, str],
	max_output_tokens: int | dict[str, int],
	output: Path,
	account_profiles: dict[str, str] | None = None,
	concurrency: int = 32,
	solver_workers: int = 4,
	request_deadline: float = 120,
	transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
	"""Measure response utility with a declared process allowance for proof solving."""
	solver = ProofSolver(solver_workers)
	try:
		return await _measure_utility(
			requests=requests,
			references=references,
			url=url,
			keys=keys,
			max_output_tokens=max_output_tokens,
			output=output,
			account_profiles=account_profiles,
			concurrency=concurrency,
			request_deadline=request_deadline,
			transport=transport,
			solver=solver,
		)
	finally:
		await solver.close()


async def _measure_utility(
	*,
	solver: ProofSolver,
	requests: list[BenignRequest],
	references: dict[str, BenchmarkItem],
	url: str,
	keys: dict[str, str],
	max_output_tokens: int | dict[str, int],
	output: Path,
	account_profiles: dict[str, str] | None = None,
	concurrency: int = 32,
	request_deadline: float = 120,
	transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
	"""Quality among released responses and zero-for-nonresponse utility over every submission.

	The same absolute arrivals are used for both services. An account has one request in flight;
	client and semaphore waiting count against its deadline. Warmup uses a separate account.
	"""
	contract = hashlib.sha256(
		json.dumps(
			{
				"measurement_version": "utility-response-latency-v2",
				"solver": {
					"version": SOLVER_VERSION,
					"workers": solver.workers,
					"chunk_hashes": CHUNK_HASHES,
				},
				"requests": [r.model_dump() for r in requests],
				"references": {k: v.model_dump(mode="json") for k, v in references.items()},
				"cap": max_output_tokens,
				"account_profiles": account_profiles,
				"concurrency": concurrency,
				"deadline": request_deadline,
				"url": url,
			},
			sort_keys=True,
		).encode()
	).hexdigest()
	if await AsyncPath(output).exists():
		previous = json.loads(await AsyncPath(output).read_text())
		if previous["contract"] != contract:
			raise ValueError("utility workload or serving configuration changed")
		return dict(previous)
	if not requests:
		raise ValueError("utility requires at least one request")
	if any(item.metric == "pass_at_1" for item in references.values()):
		from coding.scoring import ensure_runtime

		await asyncio.to_thread(ensure_runtime)
	await solver.warmup()
	warmup = ProtectedTeacher(
		url, keys.get("warmup", keys.get("default", "")), transport=transport, solver=solver
	)
	try:
		item = references[str(requests[0].example_id)]
		if isinstance(max_output_tokens, dict):
			warmup.domain = item.benchmark
		for _ in range(5):
			warmup.deadline = time.monotonic() + request_deadline
			try:
				await warmup.generate(
					TeacherGenerationRequest(
						messages=item.messages,
						max_output_tokens=(
							max_output_tokens[item.benchmark]
							if isinstance(max_output_tokens, dict)
							else max_output_tokens
						),
						enable_thinking=False,
					)
				)
			except PolicyRefusal, AttemptFailed, BudgetExhausted, httpx.HTTPError:
				continue
	finally:
		await warmup.close()
	streams: dict[str, list[BenignRequest]] = {}
	for request in requests:
		streams.setdefault(request.account, []).append(request)
	semaphore = asyncio.Semaphore(concurrency)
	started = time.monotonic()
	results: list[dict[str, Any]] = []
	answers: list[tuple[dict[str, Any], BenchmarkItem, TeacherGeneration]] = []

	async def client(account: str, rows: list[BenignRequest]) -> None:
		teacher = ProtectedTeacher(
			url, keys.get(account, keys.get("default", "")), transport=transport, solver=solver
		)
		try:
			for index, row in enumerate(rows):
				arrival = started + (row.time or 0)
				await asyncio.sleep(max(0, arrival - time.monotonic()))
				teacher.deadline = arrival + request_deadline
				item = references[str(row.example_id)]
				if isinstance(max_output_tokens, dict):
					teacher.domain = item.benchmark
				status, quality, received = "abandoned", 0.0, False
				generation = None
				event_offset = len(teacher.events)
				admitted = None
				before = teacher.hashes
				attempt_started = time.monotonic()
				try:
					async with asyncio.timeout(max(0, teacher.remaining())):
						async with semaphore:
							admitted = time.monotonic()
							generation = await teacher.generate(
								TeacherGenerationRequest(
									messages=item.messages,
									max_output_tokens=(
										max_output_tokens[item.benchmark]
										if isinstance(max_output_tokens, dict)
										else max_output_tokens
									),
									enable_thinking=False,
								)
							)
				except PolicyRefusal:
					completed = time.monotonic()
					status = "refused"
				except TimeoutError, BudgetExhausted:
					completed = time.monotonic()
					status = "timeout"
				except AttemptFailed, httpx.HTTPError, ValueError:
					completed = time.monotonic()
					status = "error"
				else:
					completed = time.monotonic()
					received = True

				solve_events = [e for e in teacher.events[event_offset:] if e["event"] == "solve"]
				http_events = [e for e in teacher.events[event_offset:] if e["event"] == "outcome"]
				results.append(
					{
						"account": account,
						"domain": item.benchmark,
						"domain_profile": (account_profiles or {}).get(account),
						"sequence": index,
						"example_id": row.example_id,
						"source_group": row.source_group,
						"client_type": row.client_type,
						"status": status,
						"received": received,
						"quality": quality,
						"latency_seconds": completed - arrival,
						"arrival_delay_seconds": attempt_started - arrival,
						"client_queue_seconds": (admitted or completed) - attempt_started,
						"http_seconds": sum(e["latency_seconds"] for e in http_events),
						"proof_seconds": sum(e["seconds"] for e in solve_events),
						"proof_queue_seconds": sum(e["queue_seconds"] for e in solve_events),
						"proof_worker_seconds": sum(e["worker_seconds"] for e in solve_events),
						"proof_cpu_seconds": sum(e["cpu_seconds"] for e in solve_events),
						"service_and_queue_seconds": completed - attempt_started,
						"hashes": teacher.hashes - before,
					}
				)
				if generation is not None:
					answers.append((results[-1], item, generation))
		finally:
			await teacher.close()

	tasks = [asyncio.create_task(client(account, rows)) for account, rows in streams.items()]
	try:
		await asyncio.gather(*tasks)
	finally:
		for task in tasks:
			if not task.done():
				task.cancel()
		await asyncio.gather(*tasks, return_exceptions=True)
	elapsed = time.monotonic() - started
	# Scoring can use a Python loop or a coding sandbox. Keep it out of request scheduling.
	scoring_started = time.monotonic()
	for result, item, generation in answers:
		score = await evaluate_answer(item, generation.message.content, generation.finish_reason)
		valid = score.status == "ok"
		result["status"] = "valid" if valid else score.status
		result["quality"] = (
			(score.f1 if score.f1 is not None else float(score.correct)) if valid else 0.0
		)
	scoring_seconds = time.monotonic() - scoring_started
	served = [r for r in results if r["received"]]
	latencies = sorted(r["latency_seconds"] for r in served)
	summary = {
		"contract": contract,
		"submitted": len(results),
		"served": len(served),
		"served_only_quality": statistics.fmean(r["quality"] for r in served) if served else None,
		"unconditional_utility": statistics.fmean(r["quality"] for r in results),
		"completion_rate": len(served) / len(results),
		"elapsed_seconds": elapsed,
		"scoring_seconds": scoring_seconds,
		"throughput": len(served) / elapsed,
		"latency_p50": statistics.median(latencies) if latencies else None,
		"latency_p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
		if latencies
		else None,
		"strata": {
			field: {
				str(value): {
					"submitted": len(rows),
					"served": sum(r["received"] for r in rows),
					"unconditional_utility": statistics.fmean(r["quality"] for r in rows),
					"served_only_quality": statistics.fmean(
						r["quality"] for r in rows if r["received"]
					)
					if any(r["received"] for r in rows)
					else None,
				}
				for value in {r.get(field) for r in results}
				if (rows := [r for r in results if r.get(field) == value])
			}
			for field in ("domain", "domain_profile", "client_type")
		},
		"measurement_version": "utility-response-latency-v2",
		"solver": {
			"version": SOLVER_VERSION,
			"workers": solver.workers,
			"chunk_hashes": CHUNK_HASHES,
		},
		"load": {
			"concurrency": concurrency,
			"deadline_seconds": request_deadline,
			"warmup": 5,
			"arrival_times": "synthetic, unchanged across arms",
		},
		"ttft": None,
		"ttft_reason": "endpoint is non-streaming",
		"server_peak_memory": None,
		"results": results,
	}
	save_json(output, summary)
	return summary
