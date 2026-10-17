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
from distillation.benchmarks.scoring import score_answer, squad_f1
from distillation.campaign import AttemptFailed, BudgetExhausted, PolicyRefusal, ProtectedTeacher
from distillation.schemas import TeacherGenerationRequest


async def measure_utility(
	*,
	requests: list[BenignRequest],
	references: dict[str, BenchmarkItem],
	url: str,
	keys: dict[str, str],
	max_output_tokens: int,
	output: Path,
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
				"requests": [r.model_dump() for r in requests],
				"references": {k: v.model_dump(mode="json") for k, v in references.items()},
				"cap": max_output_tokens,
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
	warmup = ProtectedTeacher(url, keys.get("warmup", keys.get("default", "")), transport=transport)
	try:
		item = references[str(requests[0].example_id)]
		for _ in range(5):
			await warmup.generate(
				TeacherGenerationRequest(
					messages=item.messages,
					max_output_tokens=max_output_tokens,
					enable_thinking=False,
				)
			)
	finally:
		await warmup.close()
	streams: dict[str, list[BenignRequest]] = {}
	for request in requests:
		streams.setdefault(request.account, []).append(request)
	semaphore = asyncio.Semaphore(concurrency)
	started = time.monotonic()
	results: list[dict[str, Any]] = []

	async def client(account: str, rows: list[BenignRequest]) -> None:
		teacher = ProtectedTeacher(
			url, keys.get(account, keys.get("default", "")), transport=transport
		)
		try:
			for index, row in enumerate(rows):
				arrival = started + (row.time or 0)
				await asyncio.sleep(max(0, arrival - time.monotonic()))
				teacher.deadline = arrival + request_deadline
				item = references[str(row.example_id)]
				status, quality, received = "abandoned", 0.0, False
				before = teacher.hashes
				attempt_started = time.monotonic()
				try:
					async with asyncio.timeout(max(0, teacher.remaining())):
						async with semaphore:
							generation = await teacher.generate(
								TeacherGenerationRequest(
									messages=item.messages,
									max_output_tokens=max_output_tokens,
									enable_thinking=False,
								)
							)
				except PolicyRefusal:
					status = "refused"
				except TimeoutError, BudgetExhausted:
					status = "timeout"
				except AttemptFailed, httpx.HTTPError, ValueError:
					status = "error"
				else:
					received = True
					text = generation.message.content
					prediction, correct = score_answer(item, text)
					valid = (
						prediction is not None
						and generation.finish_reason in {"stop", "eos", "end_turn", "completed"}
						and "<think>" not in text
						and "</think>" not in text
					)
					status = "valid" if valid else "invalid"
					quality = (
						(squad_f1(item, text) if item.metric == "squad" else float(correct))
						if valid
						else 0.0
					)
				results.append(
					{
						"account": account,
						"sequence": index,
						"example_id": row.example_id,
						"source_group": row.source_group,
						"client_type": row.client_type,
						"status": status,
						"received": received,
						"quality": quality,
						"latency_seconds": time.monotonic() - arrival,
						"service_and_queue_seconds": time.monotonic() - attempt_started,
						"hashes": teacher.hashes - before,
					}
				)
		finally:
			await teacher.close()

	await asyncio.gather(*(client(account, rows) for account, rows in streams.items()))
	elapsed = time.monotonic() - started
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
		"throughput": len(served) / elapsed,
		"latency_p50": statistics.median(latencies) if latencies else None,
		"latency_p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
		if latencies
		else None,
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
