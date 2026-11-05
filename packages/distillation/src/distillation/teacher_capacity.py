"""Open-loop, undefended teacher capacity preflight against a frozen schedule."""

import asyncio
import hashlib
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from distillation.schemas import GatewayRequest, TeacherGeneration, TeacherGenerationRequest


@dataclass(frozen=True)
class CapacitySLO:
	min_offered_rps: float
	min_completed_rps: float
	max_p95_seconds: float
	max_error_fraction: float
	max_timeout_fraction: float
	max_queued_at_end: int
	max_queue_growth: int

	def __post_init__(self) -> None:
		values = (
			self.min_offered_rps,
			self.min_completed_rps,
			self.max_p95_seconds,
			self.max_error_fraction,
			self.max_timeout_fraction,
		)
		if any(not math.isfinite(value) or value < 0 for value in values):
			raise ValueError("SLO values must be finite and nonnegative")
		if self.min_offered_rps <= 0 or self.min_completed_rps <= 0 or self.max_p95_seconds <= 0:
			raise ValueError("throughput and latency SLOs must be positive")
		if self.max_error_fraction > 1 or self.max_timeout_fraction > 1:
			raise ValueError("fractions must not exceed one")
		if self.max_queued_at_end < 0 or self.max_queue_growth < 0:
			raise ValueError("queue limits must be nonnegative")


def require_capacity_pass(result: dict[str, Any], *, schedule_sha256: str) -> None:
	"""Reject absent, failed, or mismatched capacity evidence for qualification."""
	if (
		result.get("version") != "teacher-capacity-v1"
		or result.get("passed") is not True
		or result.get("schedule_sha256") != schedule_sha256
		or not isinstance(result.get("slo"), dict)
		or not isinstance(result.get("checks"), dict)
		or set(result["checks"])
		!= {
			"offered_throughput",
			"completed_throughput",
			"latency_p95",
			"errors",
			"timeouts",
			"queue_end",
			"queue_growth",
			"undefended",
		}
		or not all(value is True for value in result["checks"].values())
	):
		raise ValueError("a passing capacity preflight for this schedule is required")


async def preflight(
	rows: list[GatewayRequest],
	*,
	url: str,
	token: str,
	schedule_sha256: str,
	slo: CapacitySLO,
	concurrency: int,
	timeout_seconds: float,
	transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
	if (
		len(rows) < 2
		or concurrency < 1
		or not math.isfinite(timeout_seconds)
		or timeout_seconds <= 0
	):
		raise ValueError("preflight needs at least two requests, concurrency and a finite timeout")
	if len({row.request_id for row in rows}) != len(rows):
		raise ValueError("request IDs must be unique")
	rows = sorted(rows, key=lambda row: row.time)
	duration = rows[-1].time - rows[0].time
	if duration <= 0:
		raise ValueError("schedule must span a positive time interval")
	queue: asyncio.Queue[GatewayRequest | None] = asyncio.Queue()
	started = time.monotonic()
	latencies: list[float] = []
	counts = {"completed": 0, "errors": 0, "timeouts": 0, "defended_responses": 0}
	queue_mid = 0
	queue_end = 0
	max_queue = 0
	headers = {"Authorization": f"Bearer {token}"} if token else {}
	async with httpx.AsyncClient(
		base_url=url.rstrip("/") + "/",
		headers=headers,
		timeout=timeout_seconds,
		transport=transport,
	) as client:

		async def worker() -> None:
			while (row := await queue.get()) is not None:
				request = TeacherGenerationRequest(
					messages=row.prompt.prompt,
					max_output_tokens=row.max_output_tokens,
					enable_thinking=False,
				)
				try:
					response = await asyncio.wait_for(
						client.post("v1/teacher/generate", json=request.model_dump(mode="json")),
						timeout_seconds,
					)
					if "x-harness-decision" in response.headers:
						counts["defended_responses"] += 1
					response.raise_for_status()
					TeacherGeneration.model_validate(response.json())
				except TimeoutError, httpx.TimeoutException:
					counts["timeouts"] += 1
				except httpx.HTTPError, ValidationError, ValueError:
					counts["errors"] += 1
				else:
					counts["completed"] += 1
					latencies.append(time.monotonic() - started - (row.time - rows[0].time))
				queue.task_done()

		workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
		try:
			for row in rows:
				await asyncio.sleep(max(0, started + row.time - rows[0].time - time.monotonic()))
				queue.put_nowait(row)
				max_queue = max(max_queue, queue.qsize())
				if row.time - rows[0].time <= duration / 2:
					queue_mid = queue.qsize()
			queue_end = queue.qsize()
			await queue.join()
		finally:
			for task in workers:
				task.cancel()
			await asyncio.gather(*workers, return_exceptions=True)
	elapsed = time.monotonic() - started
	latencies.sort()
	p95 = latencies[math.ceil(len(latencies) * 0.95) - 1] if latencies else None
	p50 = latencies[math.ceil(len(latencies) * 0.50) - 1] if latencies else None
	p99 = latencies[math.ceil(len(latencies) * 0.99) - 1] if latencies else None
	offered_rps = len(rows) / duration
	completed_rps = counts["completed"] / elapsed
	checks = {
		"offered_throughput": offered_rps >= slo.min_offered_rps,
		"completed_throughput": completed_rps >= slo.min_completed_rps,
		"latency_p95": p95 is not None and p95 <= slo.max_p95_seconds,
		"errors": counts["errors"] / len(rows) <= slo.max_error_fraction,
		"timeouts": counts["timeouts"] / len(rows) <= slo.max_timeout_fraction,
		"queue_end": queue_end <= slo.max_queued_at_end,
		"queue_growth": max(0, queue_end - queue_mid) <= slo.max_queue_growth,
		"undefended": counts["defended_responses"] == 0,
	}
	return {
		"version": "teacher-capacity-v1",
		"schedule_sha256": schedule_sha256,
		"endpoint_sha256": hashlib.sha256(url.encode()).hexdigest(),
		"slo": asdict(slo),
		"offered": len(rows),
		**counts,
		"offered_rps": offered_rps,
		"completed_rps": completed_rps,
		"elapsed_seconds": elapsed,
		"latency_p95_seconds": p95,
		"latency_p50_seconds": p50,
		"latency_p99_seconds": p99,
		"queued_mid": queue_mid,
		"queued_at_end": queue_end,
		"max_queue": max_queue,
		"concurrency": concurrency,
		"timeout_seconds": timeout_seconds,
		"checks": checks,
		"passed": all(checks.values()),
	}
