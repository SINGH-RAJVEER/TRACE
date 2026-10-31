"""Frozen development qualification and paired capability effect inference."""

import hashlib
import math
from pathlib import Path
from typing import Any

from distillation.benchmarks.models import SampleResult, load_suite
from distillation.launcher import LaunchError

ANALYSIS_VERSION = "transfer-v2"
BOOTSTRAPS = 5000


def analysis_protocol(margins: dict[str, float], tasks: list[str]) -> dict[str, Any]:
	if set(margins) - set(tasks) or any(
		not math.isfinite(value) or not 0 < value < 100 for value in margins.values()
	):
		raise LaunchError(
			"--minimum-gain requires known tasks and finite margins with 0 < PP < 100"
		)
	margins = {task: margins.get(task, 2.0) for task in tasks}
	return {
		"version": ANALYSIS_VERSION,
		"minimum_gain_pp": margins,
		"confidence": 0.95,
		"score_unit": "percentage_points",
		"bootstrap_replicates": BOOTSTRAPS,
		"bootstrap_seed": 42,
		"qualification_rule": "lower two-sided 95% CI > minimum_gain_pp",
		"split": "source-disjoint development halves; SHA256 order; selection then qualification",
		"uncertainty": "paired source bootstrap conditional on trained models; "
		"not training variance",
	}


def development_suites(source: Path, directory: Path) -> dict[str, Path]:
	"""Partition before selection; SQuAD titles stay together as in paired reporting."""
	from distillation.study import freeze

	suite = load_suite(source)
	groups = {
		(item.benchmark, item.category if item.benchmark == "squad" else item.id)
		for item in suite.items
	}
	if len(groups) < 4:
		raise LaunchError("development qualification requires at least four source groups")
	ordered = sorted(groups, key=lambda key: hashlib.sha256(repr(key).encode()).hexdigest())
	selection = set(ordered[: len(ordered) // 2])
	paths = {}
	for split in ("selection", "qualification"):
		items = [
			item
			for item in suite.items
			if (
				(item.benchmark, item.category if item.benchmark == "squad" else item.id)
				in selection
			)
			== (split == "selection")
		]
		part = suite.model_copy(
			update={
				"items": items,
				"sources": [
					s.model_copy(
						update={"selected": sum(i.benchmark == s.benchmark for i in items)}
					)
					for s in suite.sources
					if any(i.benchmark == s.benchmark for i in items)
				],
			}
		)
		part.sha256 = part.digest()
		part.verify()
		path = directory / f"{split}-suite.json"
		freeze(path, part.model_dump(mode="json"))
		paths[split] = path
	freeze(
		directory / "split.json",
		{
			"version": ANALYSIS_VERSION,
			"parent_sha256": suite.sha256,
			**{split: load_suite(path).sha256 for split, path in paths.items()},
		},
	)
	return paths


def _scores(samples: list[SampleResult]) -> dict[str, SampleResult]:
	rows = {s.item_id: s for s in samples}
	if not rows or len(rows) != len(samples):
		raise ValueError("inference requires nonempty samples with unique IDs")
	return rows


def _resample(*arms: list[SampleResult]) -> tuple[Any, Any, int]:
	import numpy as np

	rows = [_scores(arm) for arm in arms]
	if any(row.keys() != rows[0].keys() for row in rows):
		raise ValueError("inference requires identical paired item IDs")
	groups: dict[tuple[str, str], list[list[float]]] = {}
	for key in sorted(rows[0]):
		sample = rows[0][key]
		paired = [row[key] for row in rows]
		if any((s.benchmark, s.category) != (sample.benchmark, sample.category) for s in paired):
			raise ValueError("paired source metadata changed")
		group = sample.benchmark, sample.category if sample.benchmark == "squad" else key
		values = [float(s.f1 or 0) if s.benchmark == "squad" else float(s.correct) for s in paired]
		if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
			raise ValueError("invalid primary score")
		groups.setdefault(group, []).append(values)
	if len(groups) < 2:
		raise ValueError("inference requires at least two source groups")
	sums = np.array([np.sum(values, axis=0) for values in groups.values()])
	sizes = np.array([len(values) for values in groups.values()])
	rng = np.random.default_rng(42)
	# Chunk draws to bound memory for full evaluation suites.
	draws = []
	for start in range(0, BOOTSTRAPS, 100):
		indices = rng.integers(0, len(groups), size=(min(100, BOOTSTRAPS - start), len(groups)))
		draws.append(100 * sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)[:, None])
	return 100 * sums.sum(axis=0) / sizes.sum(), np.concatenate(draws), len(groups)


def _ci(values: Any) -> list[float]:
	import numpy as np

	return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def _exceeds(value: float, margin: float) -> bool:
	# A nominal equality must not pass because summation rounded upward.
	return value > margin and not math.isclose(value, margin, rel_tol=1e-10, abs_tol=1e-10)


def _check_margin(margin: float) -> None:
	if not math.isfinite(margin) or not 0 < margin < 100:
		raise ValueError(
			"minimum useful gain must be finite and between 0 and 100 percentage points"
		)


def qualification(
	before: list[SampleResult], after: list[SampleResult], margin: float
) -> dict[str, Any]:
	_check_margin(margin)
	point, draws, clusters = _resample(before, after)
	interval = _ci(draws[:, 1] - draws[:, 0])
	return {
		"base_score": float(point[0]),
		"score": float(point[1]),
		"gain_pp": float(point[1] - point[0]),
		"gain_ci95_pp": interval,
		"minimum_gain_pp": margin,
		"passed": _exceeds(interval[0], margin),
		"clusters": clusters,
		"analysis_version": ANALYSIS_VERSION,
	}


def effect_inference(
	base: list[SampleResult],
	undefended: list[SampleResult],
	defended: list[SampleResult],
	*,
	margin: float,
	qualified: bool,
) -> dict[str, Any]:
	import numpy as np

	_check_margin(margin)
	point, draws, clusters = _resample(base, undefended, defended)
	gain = float(point[1] - point[0])
	gain_draws = draws[:, 1] - draws[:, 0]
	removed = float(point[1] - point[2])
	removed_draws = draws[:, 1] - draws[:, 2]
	interval = _ci(gain_draws)
	reason = (
		"qualification_failed"
		if not qualified
		else "negligible_gain"
		if not _exceeds(gain, margin)
		else "uncertain_gain"
		if not _exceeds(interval[0], margin)
		else None
	)
	# Never drop denominator-crossing draws to manufacture a bounded ratio interval.
	if reason is None and np.any(gain_draws <= 0):
		reason = "unstable_ratio_interval"
	valid = reason is None
	return {
		"analysis_version": ANALYSIS_VERSION,
		"minimum_gain_pp": margin,
		"sample_scores_pp": dict(
			zip(("base", "undefended", "defended"), map(float, point), strict=True)
		),
		"clusters": clusters,
		"undefended_gain_ci95_pp": interval,
		"defended_gain_ci95_pp": _ci(draws[:, 2] - draws[:, 0]),
		"gain_removed": removed,
		"gain_removed_ci95_pp": _ci(removed_draws),
		"retained_gain_fraction": float((point[2] - point[0]) / gain) if valid else None,
		"gain_removed_fraction": removed / gain if valid else None,
		"retained_gain_fraction_ci95": _ci((draws[:, 2] - draws[:, 0]) / gain_draws)
		if valid
		else None,
		"gain_removed_fraction_ci95": _ci(removed_draws / gain_draws) if valid else None,
		"ratio_valid": valid,
		"ratio_invalid_reason": reason,
		"positive_removal_supported": _exceeds(_ci(removed_draws)[0], 0),
		"uncertainty": "paired source bootstrap conditional on trained models",
	}
