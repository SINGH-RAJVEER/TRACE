"""Minimum detectable paired effects of the frozen domain suites, before any model runs.

The study interprets a gain only when it is at least the declared minimum with a paired
95% interval excluding zero. A suite too small to resolve that minimum cannot support the
rule, whatever the models do. These planning numbers use a normal approximation for a
paired binary difference; SQuAD token F1 is treated as binary, which is conservative.
"""

import math
from collections import Counter
from pathlib import Path
from typing import Any

from distillation.benchmarks.models import load_suite

POWER_VERSION = "suite-power-v1"
Z_TWO_SIDED_95 = 1.959964
Z_POWER_80 = 0.841621
# Planning assumption: the share of items whose correctness differs between two paired
# students. It is replaced by the observed discordance once paired runs exist.
DEFAULT_DISCORDANCE = 0.2


def minimum_detectable_pp(
	items: int,
	*,
	discordance: float = DEFAULT_DISCORDANCE,
	design_effect: float = 1.0,
) -> float:
	"""Smallest paired difference, in points, detected with 80% power at two-sided 5%."""
	if items < 1 or not 0 < discordance <= 1 or design_effect < 1:
		raise ValueError("power needs items >= 1, 0 < discordance <= 1, design_effect >= 1")
	return 100 * (Z_TWO_SIDED_95 + Z_POWER_80) * math.sqrt(discordance * design_effect / items)


def half_width_pp(items: int, rate: float = 0.5) -> float:
	"""Normal-approximation 95% half-width of one binary score at `rate`, in points."""
	return 100 * Z_TWO_SIDED_95 * math.sqrt(rate * (1 - rate) / items)


def design_effect(clusters: Counter[str]) -> float:
	"""Kish design effect for equal within-cluster correlation of one, an upper bound."""
	sizes = list(clusters.values())
	total = sum(sizes)
	return sum(size * size for size in sizes) / total if total else 1.0


def suite_power(
	path: Path,
	*,
	minimum_gain_pp: float = 2.0,
	discordance: float = DEFAULT_DISCORDANCE,
	clustered: bool = False,
) -> dict[str, Any]:
	"""Resolution of one frozen suite against the declared minimum gain.

	`clustered` uses item categories as source clusters (SQuAD article titles) and reports
	the worst case where items within a cluster are perfectly correlated; the realised
	design effect lies between one and that bound.
	"""
	suite = load_suite(path)
	clusters: Counter[str] = Counter(item.category for item in suite.items)
	bound = design_effect(clusters) if clustered else 1.0
	items = len(suite.items)
	mde = minimum_detectable_pp(items, discordance=discordance)
	clustered_mde = minimum_detectable_pp(items, discordance=discordance, design_effect=bound)
	return {
		"suite": str(path),
		"suite_sha256": suite.sha256,
		"items": items,
		"clusters": len(clusters),
		"score_half_width_pp_at_50": half_width_pp(items),
		"minimum_detectable_pp": mde,
		"clustered_design_effect_bound": bound,
		"clustered_minimum_detectable_pp_bound": clustered_mde,
		"minimum_gain_pp": minimum_gain_pp,
		"resolves_minimum_gain": mde <= minimum_gain_pp,
		"resolves_minimum_gain_clustered_bound": clustered_mde <= minimum_gain_pp,
	}


def study_power(
	tasks_dir: Path,
	tasks: list[str],
	*,
	minimum_gain_pp: dict[str, float] | None = None,
	discordance: float = DEFAULT_DISCORDANCE,
) -> dict[str, Any]:
	"""Per-domain resolution of the development and test suites of existing task builds."""
	from distillation.tasks import task_directory

	margins = minimum_gain_pp or {}
	domains: dict[str, Any] = {}
	for task in tasks:
		domains[task] = {
			name: suite_power(
				task_directory(task, root=tasks_dir) / f"{name}-suite.json",
				minimum_gain_pp=margins.get(task, 2.0),
				discordance=discordance,
				clustered=task == "squad",
			)
			for name in ("development", "test")
			if (task_directory(task, root=tasks_dir) / f"{name}-suite.json").exists()
		}
	return {
		"version": POWER_VERSION,
		"alpha": 0.05,
		"power": 0.8,
		"discordance_assumption": discordance,
		"method": "normal approximation to a paired binary difference (McNemar)",
		"domains": domains,
	}
