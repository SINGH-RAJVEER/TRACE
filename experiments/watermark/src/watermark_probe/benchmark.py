"""CPU latency of the seed, mask, bias, batch and detection paths, on synthetic inputs.

These numbers describe the arithmetic on one machine at one moment. They include array allocation
and exclude any model, tokenizer or serving engine, so they bound nothing about real generation.
"""

import os
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from watermark_probe.bias import KgwBatchProcessor, apply_bias
from watermark_probe.core import WatermarkKey, WatermarkProfile, context_seed, green_mask
from watermark_probe.detect import detect_candidates

_KEYS = [WatermarkKey(f"bench-{i}", 0, f"public-benchmark-key-{i:04d}".encode()) for i in range(8)]
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _profile(vocab_size: int, context_width: int = 2) -> WatermarkProfile:
	return WatermarkProfile(
		gamma=0.25, delta=1.0, context_width=context_width, vocab_size=vocab_size
	)


def _timed(run: Callable[[], object], repeats: int) -> dict[str, float | int]:
	samples = []
	for _ in range(repeats):
		start = time.perf_counter_ns()
		run()
		samples.append((time.perf_counter_ns() - start) / 1000)
	samples.sort()
	return {
		"repeats": repeats,
		"p50_us": statistics.median(samples),
		"p95_us": samples[min(repeats - 1, int(repeats * 0.95))],
	}


def _cpu_model() -> str | None:
	try:
		for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
			if line.startswith("model name"):
				return line.split(":", 1)[1].strip()
	except OSError:
		pass
	return platform.processor() or None


def _environment() -> dict[str, Any]:
	return {
		"cpu_model": _cpu_model(),
		"cpu_count": os.cpu_count(),
		"python": platform.python_version(),
		"numpy": np.__version__,
		"threads": {name: os.environ.get(name) for name in _THREAD_VARIABLES},
		"nice": os.nice(0),
		"load_average_start": list(os.getloadavg()),
	}


def _seed_cases(repeats: int) -> list[dict[str, Any]]:
	cases = []
	for width in (1, 4):
		profile = _profile(50_000, width)
		context = tuple(range(width))
		timing = _timed(lambda: context_seed(_KEYS[0], profile, context), repeats)  # noqa: B023
		cases.append({"context_width": width, **timing})
	return cases


def _mask_cases(vocabularies: tuple[int, ...], repeats: int) -> list[dict[str, Any]]:
	cases = []
	for vocab_size in vocabularies:
		profile = _profile(vocab_size)
		seed = context_seed(_KEYS[0], profile, (1, 2))
		timing = _timed(lambda: green_mask(seed, profile), repeats)  # noqa: B023
		cases.append({"vocab_size": vocab_size, **timing})
	return cases


def _bias_cases(vocabularies: tuple[int, ...], repeats: int) -> list[dict[str, Any]]:
	cases = []
	for vocab_size in vocabularies:
		profile = _profile(vocab_size)
		for dtype in (np.float32, np.float64):
			logits: Any = np.zeros(vocab_size, dtype=dtype)
			timing = _timed(
				lambda: apply_bias(logits, (1, 2), _KEYS[0], profile),  # noqa: B023
				repeats,
			)
			cases.append({"vocab_size": vocab_size, "dtype": dtype.__name__, **timing})
	return cases


def _batch_cases(
	vocabularies: tuple[int, ...], batch_sizes: tuple[int, ...], repeats: int
) -> list[dict[str, Any]]:
	cases = []
	for vocab_size in vocabularies:
		profile = _profile(vocab_size)
		for batch_size in batch_sizes:
			processor = KgwBatchProcessor(profile)
			slots = list(range(batch_size))
			for slot in slots:
				processor.add_request(slot, f"r{slot}", [slot, slot + 1], _KEYS[slot % len(_KEYS)])
			logits = np.zeros((batch_size, vocab_size), dtype=np.float32)
			step = [0]

			def run() -> None:
				step[0] += 1  # noqa: B023
				processor.apply(logits, slots)  # noqa: B023
				for slot in slots:  # noqa: B023
					processor.record_token(slot, (step[0] + slot) % vocab_size)  # noqa: B023

			timing = _timed(run, repeats)
			cases.append(
				{
					"vocab_size": vocab_size,
					"batch_size": batch_size,
					"tokens_per_second": batch_size / (float(timing["p50_us"]) / 1e6),
					**timing,
				}
			)
	return cases


def _detect_cases(lengths: tuple[int, ...], repeats: int) -> list[dict[str, Any]]:
	profile = _profile(50_000)
	cases = []
	for length in lengths:
		unique = list(range(length))
		repeated = [index % 16 for index in range(length)]
		for pattern, sequence in (("unique", unique), ("repeated", repeated)):
			for candidates in (1, 8):
				keys = _KEYS[:candidates]
				timing = _timed(
					lambda: detect_candidates(
						sequence,  # noqa: B023
						keys,  # noqa: B023
						profile,
						alpha=1e-3,
						min_effective=1,
					),
					repeats,
				)
				cases.append(
					{"length": length, "pattern": pattern, "candidates": candidates, **timing}
				)
	return cases


def run_benchmarks(quick: bool = False) -> dict[str, Any]:
	"""Measure every path. `quick` only proves the benchmark runs; its timings mean little."""
	vocabularies = (5_000,) if quick else (5_000, 50_000, 200_000)
	batch_sizes = (1, 4) if quick else (1, 4, 16)
	lengths = (64,) if quick else (256, 1024, 4096)
	seed_repeats, array_repeats, detect_repeats = (5, 5, 2) if quick else (20_000, 300, 30)
	report: dict[str, Any] = {"environment": _environment()}
	report["seed"] = _seed_cases(seed_repeats)
	report["mask"] = _mask_cases(vocabularies, array_repeats)
	report["bias"] = _bias_cases(vocabularies, array_repeats)
	report["batch"] = _batch_cases(vocabularies, batch_sizes, array_repeats)
	report["detect"] = _detect_cases(lengths, detect_repeats)
	report["environment"]["load_average_end"] = list(os.getloadavg())
	return report
