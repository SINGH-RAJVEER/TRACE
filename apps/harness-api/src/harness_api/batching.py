"""Bounded FIFO microbatching for the serving encoder."""

import asyncio
from dataclasses import dataclass

from harness.encoder import Encoder
from harness.mmd import Array


@dataclass
class _Item:
	text: str
	future: asyncio.Future[Array]


class EncoderBatcher:
	def __init__(self, encoder: Encoder, *, batch_size: int, wait_seconds: float, capacity: int):
		self.encoder = encoder
		self.batch_size = batch_size
		self.wait_seconds = wait_seconds
		self.queue: asyncio.Queue[_Item | None] = asyncio.Queue(maxsize=capacity)
		self.worker: asyncio.Task[None] | None = None
		self.closed = False

	async def encode(self, text: str) -> Array:
		if self.closed:
			raise RuntimeError("encoder batcher is closed")
		future: asyncio.Future[Array] = asyncio.get_running_loop().create_future()
		self.queue.put_nowait(_Item(text, future))
		if self.worker is None:
			self.worker = asyncio.create_task(self._run())
		return await future

	async def _run(self) -> None:
		while True:
			first = await self.queue.get()
			if first is None:
				return
			items = [first]
			deadline = asyncio.get_running_loop().time() + self.wait_seconds
			while len(items) < self.batch_size:
				remaining = deadline - asyncio.get_running_loop().time()
				if remaining <= 0:
					break
				try:
					item = await asyncio.wait_for(self.queue.get(), remaining)
				except TimeoutError:
					break
				if item is None:
					# Process the partial batch before stopping.
					self.queue.put_nowait(None)
					break
				items.append(item)
			active = [item for item in items if not item.future.cancelled()]
			if not active:
				continue
			try:
				rows = await asyncio.to_thread(self.encoder.encode, [item.text for item in active])
				if len(rows) != len(active):
					raise RuntimeError("encoder returned the wrong number of rows")
			except Exception as error:
				for item in active:
					if not item.future.done():
						item.future.set_exception(error)
			else:
				for item, row in zip(active, rows, strict=True):
					if not item.future.done():
						item.future.set_result(row)

	async def close(self) -> None:
		self.closed = True
		if self.worker is not None:
			# FIFO sentinel follows all accepted items, including a partially filled batch.
			await self.queue.put(None)
			await self.worker
