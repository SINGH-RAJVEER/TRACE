"""Wilson score intervals, percentiles and latency distributions."""

import math
from typing import Any


def wilson(correct: int, total: int) -> list[float]:
	if not total:
		return [0, 0]
	z = 1.959963984540054
	p = correct / total
	denominator = 1 + z * z / total
	center = (p + z * z / (2 * total)) / denominator
	radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
	return [100 * max(0, center - radius), 100 * min(1, center + radius)]


def percentile(values: list[float], fraction: float) -> float | None:
	if not values:
		return None
	ordered = sorted(values)
	position = (len(ordered) - 1) * fraction
	lower = math.floor(position)
	upper = math.ceil(position)
	return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


QUANTILES = (0.5, 0.9, 0.95, 0.99)


def distribution(values: list[float]) -> dict[str, Any]:
	"""Count, mean, max, and p50/p90/p95/p99 by linear interpolation.

	A quantile q is `supported` only with at least 1/(1 - q) observations; below that it is
	an interpolation toward the sample maximum, not an estimate of the tail.
	"""
	result: dict[str, Any] = {
		"n": len(values),
		"mean": sum(values) / len(values) if values else None,
		"max": max(values) if values else None,
	}
	for q in QUANTILES:
		name = f"p{round(q * 100)}"
		result[name] = percentile(values, q)
		result[f"{name}_supported"] = len(values) >= math.ceil(1 / (1 - q) - 1e-9)
	return result
