import asyncio
import threading
from typing import Any

import numpy as np
import pytest
from harness_api.batching import EncoderBatcher


class FakeEncoder:
	def __init__(self) -> None:
		self.batches: list[list[str]] = []
		self.fail = False
		self.started = threading.Event()
		self.resume = threading.Event()
		self.block = False

	def encode(self, texts: list[str]) -> Any:
		self.batches.append(texts)
		self.started.set()
		if self.block:
			self.resume.wait(5)
		if self.fail:
			raise ValueError("failure")
		return np.array([[int(text)] for text in texts])

	def identity(self) -> dict[str, Any]:
		return {"model": "fake"}


async def test_batch_order_and_shutdown_drain() -> None:
	encoder = FakeEncoder()
	batcher = EncoderBatcher(encoder, batch_size=3, wait_seconds=0.02, capacity=8)
	tasks = [asyncio.create_task(batcher.encode(str(i))) for i in range(5)]
	await asyncio.sleep(0)
	await batcher.close()
	assert [int((await task)[0]) for task in tasks] == list(range(5))
	assert encoder.batches == [["0", "1", "2"], ["3", "4"]]
	with pytest.raises(RuntimeError, match="closed"):
		await batcher.encode("6")


async def test_queue_full_cancel_and_failure() -> None:
	encoder = FakeEncoder()
	encoder.block = True
	batcher = EncoderBatcher(encoder, batch_size=1, wait_seconds=0.01, capacity=1)
	first = asyncio.create_task(batcher.encode("1"))
	await asyncio.to_thread(encoder.started.wait, 2)
	second = asyncio.create_task(batcher.encode("2"))
	await asyncio.sleep(0)
	with pytest.raises(asyncio.QueueFull):
		await batcher.encode("3")
	second.cancel()
	with pytest.raises(asyncio.CancelledError):
		await second
	encoder.resume.set()
	assert int((await first)[0]) == 1
	await batcher.close()
	assert encoder.batches == [["1"]]
	encoder = FakeEncoder()
	encoder.fail = True
	batcher = EncoderBatcher(encoder, batch_size=2, wait_seconds=0.01, capacity=2)
	results = await asyncio.gather(batcher.encode("1"), batcher.encode("2"), return_exceptions=True)
	assert all(isinstance(result, ValueError) for result in results)
	await batcher.close()
