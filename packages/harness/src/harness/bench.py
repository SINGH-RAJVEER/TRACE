"""Serving-cost measurements for the harness's own work (paper Table X).

Each unit is timed separately so a window-scoring cost is never reported as a per-request
latency. Scopes are named: per-request encoder time, per-completed-window MMD and risk
update, per-request noise, and policy admission split into requests that complete a window
and requests that do not. Victim inference is excluded; measure it with the traffic
benchmark at the same load.
"""

import time
import tracemalloc
from typing import Any

import numpy as np

from harness.calibration import Calibration
from harness.deception import noise
from harness.encoder import Encoder
from harness.mmd import Array, WindowScorer
from harness.policy import MMDPolicy


def _quantiles(seconds: list[float]) -> dict[str, float | int | None]:
	if not seconds:
		return {"n": 0, "p50_ms": None, "p95_ms": None}
	values = np.asarray(seconds) * 1000
	return {
		"n": len(seconds),
		"p50_ms": float(np.percentile(values, 50)),
		"p95_ms": float(np.percentile(values, 95)),
	}


def serving_costs(
	*,
	calibration: Calibration,
	scorer: WindowScorer,
	texts: list[str],
	encoder: Encoder | None = None,
	task: str = "text",
	accumulator: str = "paper",
	key: bytes = bytes(32),
	max_tokens: int = 256,
	seed: int = 0,
) -> dict[str, Any]:
	"""Time each harness unit over the given requests, replayed as one account.

	Without an encoder, embeddings are random unit vectors and the embedding row is absent;
	the scoring, admission, and noise costs do not depend on embedding content.
	"""
	if len(texts) < scorer.window_size:
		raise ValueError(f"benchmarking needs at least one window of {scorer.window_size} requests")
	embed_seconds: list[float] = []
	if encoder is not None:
		rows = []
		for text in texts:
			started = time.perf_counter()
			rows.append(encoder.encode([text])[0])
			embed_seconds.append(time.perf_counter() - started)
		embeddings: Array = np.stack(rows)
	else:
		rng = np.random.default_rng(seed)
		dimension = scorer.references.shape[2]
		embeddings = rng.normal(size=(len(texts), dimension))
		embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

	size = scorer.window_size
	windows = [embeddings[i : i + size] for i in range(0, len(texts) - size + 1, size)]
	spec = calibration.config.accumulators[accumulator]
	mmd_seconds: list[float] = []
	update_seconds: list[float] = []
	for window in windows:
		started = time.perf_counter()
		score = scorer.score(window)
		mmd_seconds.append(time.perf_counter() - started)
		p_value = calibration.p_value(score)
		started = time.perf_counter()
		spec.update(1.0, p_value)
		update_seconds.append(time.perf_counter() - started)
	noise_seconds: list[float] = []
	for text in texts:
		started = time.perf_counter()
		noise(task, key, text, max_tokens)
		noise_seconds.append(time.perf_counter() - started)

	def admit_all() -> list[tuple[bool, float]]:
		policy = MMDPolicy(calibration, scorer, accumulator=accumulator, deception_key=key)
		timings = []
		for index, text in enumerate(texts):
			started = time.perf_counter()
			decision = policy.admit("bench", text, float(index), embedding=embeddings[index])
			timings.append((decision.window_completed, time.perf_counter() - started))
		return timings

	timings = admit_all()
	trigger_seconds = [seconds for completed, seconds in timings if completed]
	other_seconds = [seconds for completed, seconds in timings if not completed]
	# Memory is traced in a separate pass, because tracing slows the timed one.
	tracemalloc.start()
	admit_all()
	_, peak = tracemalloc.get_traced_memory()
	tracemalloc.stop()
	return {
		"requests": len(texts),
		"window_size": size,
		"reference_windows": int(scorer.references.shape[0]),
		"task": task,
		"units": {
			"embedding_per_request": _quantiles(embed_seconds) if encoder else None,
			"mmd_per_completed_window": _quantiles(mmd_seconds),
			"risk_update_per_completed_window": _quantiles(update_seconds),
			"noise_per_request": _quantiles(noise_seconds),
			"admit_per_scoring_trigger_request": _quantiles(trigger_seconds),
			"admit_per_other_request": _quantiles(other_seconds),
			"admit_per_request": _quantiles(trigger_seconds + other_seconds),
		},
		# Python heap allocated while admitting one account's requests: buffers and state.
		"peak_policy_memory_bytes": peak,
		"encoder": encoder.identity() if encoder else None,
	}
