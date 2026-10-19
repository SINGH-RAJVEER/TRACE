"""Bounded process workers keep proof hashing off the HTTP event loop."""

import asyncio
import hashlib
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing.synchronize import Barrier

SOLVER_VERSION = "process-sha256-v1"
CHUNK_HASHES = 4096


def _ready(barrier: Barrier) -> None:
	# Do not begin timed work until every configured worker has started its interpreter.
	barrier.wait(timeout=60)


@dataclass
class SolveMetrics:
	hashes: int = 0
	queue_seconds: float = 0.0
	worker_seconds: float = 0.0
	cpu_seconds: float = 0.0
	workers: int = 1
	worker_pids: list[int] = field(default_factory=list)
	status: str = "cancelled"


@dataclass(frozen=True)
class _Chunk:
	counter: int | None
	hashes: int
	started: float
	seconds: float
	cpu_seconds: float
	pid: int


def _hash_chunk(prefix: str, difficulty: int, offset: int, deadline: float) -> _Chunk:
	started, cpu = time.monotonic(), time.process_time()
	hashes, solution = 0, None
	for counter in range(offset, offset + CHUNK_HASHES):
		if counter % 256 == 0 and time.monotonic() >= deadline:
			break
		digest = hashlib.sha256(f"{prefix}{counter}".encode()).digest()
		hashes += 1
		if 256 - int.from_bytes(digest).bit_length() >= difficulty:
			solution = counter
			break
	return _Chunk(
		solution,
		hashes,
		started,
		time.monotonic() - started,
		time.process_time() - cpu,
		os.getpid(),
	)


class ProofSolver:
	"""One shared compute allowance, with at most one bounded chunk per active solve.

	Cancellation drains the current chunk, then submits no further work. Deadlines are
	checked inside workers as well as before submission, including time spent queueing.
	"""

	def __init__(self, workers: int = 1) -> None:
		if workers < 1:
			raise ValueError("solver workers must be positive")
		self.workers = workers
		self._pool: ProcessPoolExecutor | None = None
		self._warming: asyncio.Task[None] | None = None

	def _executor(self) -> ProcessPoolExecutor:
		if self._pool is None:
			context = multiprocessing.get_context("spawn")
			self._pool = ProcessPoolExecutor(
				max_workers=self.workers,
				mp_context=context,
				initializer=_ready,
				initargs=(context.Barrier(self.workers),),
			)
		return self._pool

	async def warmup(self) -> None:
		if self._warming is None:
			self._warming = asyncio.create_task(self._warmup())
		await asyncio.shield(self._warming)

	async def _warmup(self) -> None:
		loop = asyncio.get_running_loop()
		await asyncio.gather(
			*(
				loop.run_in_executor(self._executor(), _hash_chunk, "warmup", 0, 0, float("inf"))
				for _ in range(self.workers)
			)
		)

	async def solve(
		self, prefix: str, difficulty: int, deadline: float, metrics: SolveMetrics
	) -> int | None:
		if not 0 <= difficulty <= 256:
			raise ValueError("invalid proof difficulty")
		metrics.workers = self.workers
		await self.warmup()
		loop = asyncio.get_running_loop()
		while time.monotonic() < deadline:
			submitted = time.monotonic()
			future = loop.run_in_executor(
				self._executor(), _hash_chunk, prefix, difficulty, metrics.hashes, deadline
			)
			cancelled = False
			try:
				chunk = await asyncio.shield(future)
			except asyncio.CancelledError:
				# A running executor future cannot be cancelled. Drain just this bounded chunk.
				chunk = await future
				cancelled = True
			metrics.hashes += chunk.hashes
			metrics.queue_seconds += max(0.0, chunk.started - submitted)
			metrics.worker_seconds += chunk.seconds
			metrics.cpu_seconds += chunk.cpu_seconds
			if chunk.pid not in metrics.worker_pids:
				metrics.worker_pids.append(chunk.pid)
			if cancelled:
				raise asyncio.CancelledError
			if time.monotonic() >= deadline:
				break
			if chunk.counter is not None:
				metrics.status = "solved"
				return chunk.counter
		metrics.status = "expired"
		return None

	async def close(self) -> None:
		if self._pool is not None:
			pool, self._pool = self._pool, None
			await asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=True)
		self._warming = None
