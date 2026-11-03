"""Student-capability metrics from paired per-item correctness; no models or teacher calls.

Input: ``correct[role][family]`` is a list of per-item correctness flags. The roles are
``teacher``, ``base``, ``clean`` (student trained on clean teacher outputs) and ``defended``
(student trained on protected outputs). Every role must list the same families with the same
number of items; position within a family identifies the item, so the four roles are paired.
Accuracies are fractions in [0, 1] and all thresholds are fractions (0.10 is ten points).
"""

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Literal

ROLES = ("teacher", "base", "clean", "defended")
Correct = Mapping[str, Mapping[str, Sequence[bool]]]
Weighting = Literal["item", "family"]
SINGLE_SEED = "Development evidence: a single training seed."


def _decimal(value: float) -> Fraction:
	"""A threshold as written, so 0.10 is exactly one tenth rather than its binary neighbor."""
	return Fraction(str(value))


def _floats(values: list[Fraction]) -> list[float]:
	return [float(v) for v in values]


def _bounds(samples: list[Fraction]) -> list[Fraction]:
	"""Percentile 95% bounds, indexed as in evaluation_protocol.cluster_interval."""
	samples = sorted(samples)
	n = len(samples)
	return [samples[int(0.025 * n)], samples[min(n - 1, int(0.975 * n))]]


def _validate(correct: Correct) -> list[str]:
	if not isinstance(correct, Mapping):
		raise ValueError("correct must map each role to a mapping of task families")
	if set(correct) != set(ROLES):
		raise ValueError(f"roles must be exactly {', '.join(ROLES)}")
	if not all(isinstance(correct[role], Mapping) for role in ROLES):
		raise ValueError("correct must map each role to a mapping of task families")
	families = sorted(correct["teacher"])
	if len(families) < 2:
		raise ValueError("need at least two task families")
	for role in ROLES:
		if sorted(correct[role]) != families:
			raise ValueError("every role must cover the same task families")
		for family in families:
			items = correct[role][family]
			if isinstance(items, str | bytes) or not isinstance(items, Sequence):
				raise ValueError("each task family needs a list of correctness flags")
			if not items or len(items) != len(correct["teacher"][family]):
				raise ValueError("every role needs the same nonzero item count in each family")
			if any(type(v) not in (bool, int) or v not in (0, 1) for v in items):
				raise ValueError("correctness must be boolean")
	return families


def _ratio(numerator: Fraction, denominator: Fraction) -> Fraction | None:
	return numerator / denominator if denominator > 0 else None


def _derive(a: dict[str, Fraction]) -> dict[str, Any]:
	"""Differences and ratios from four accuracies; ratios are None when the denominator <= 0."""
	headroom = a["teacher"] - a["base"]
	clean_gain = a["clean"] - a["base"]
	defended_gain = a["defended"] - a["base"]
	suppression = _ratio(defended_gain, clean_gain)
	return {
		"headroom": headroom,
		"clean_gain": clean_gain,
		"defended_gain": defended_gain,
		"effect": a["clean"] - a["defended"],
		"r_clean": _ratio(clean_gain, headroom),
		"r_defended": _ratio(defended_gain, headroom),
		"suppression": None if suppression is None else 1 - suppression,
	}


def _gate(
	estimate: Fraction, interval: list[Fraction], minimum: float, what: str
) -> dict[str, Any]:
	"""A denominator is usable only when its interval excludes zero and it meets the minimum."""
	reason = None
	if interval[0] <= 0 <= interval[1]:
		reason = f"{what} interval includes zero"
	elif estimate <= 0:
		reason = f"{what} is not positive"
	elif estimate < _decimal(minimum):
		reason = f"{what} {float(estimate):.4f} is below the minimum {minimum}"
	return {
		"estimate": float(estimate),
		"interval": _floats(interval),
		"minimum": minimum,
		"usable": not reason,
		"reason": reason,
	}


def _quantity(
	point: Fraction | None, draws: list[Fraction], gate: dict[str, Any], replicates: int
) -> dict[str, Any]:
	if not gate["usable"] or point is None:
		return {"value": None, "interval": None, "reason": gate["reason"] or "undefined"}
	return {
		"value": float(point),
		"interval": _floats(_bounds(draws)),
		"replicates_used": len(draws),
		"replicates": replicates,
		"reason": None,
	}


def transfer_report(
	correct: Correct,
	*,
	min_headroom: float = 0.10,
	min_clean_gain: float = 0.05,
	min_effect: float = 0.05,
	min_suppression_fraction: float = 0.25,
	replicates: int = 2000,
	seed: int = 42,
	weighting: Weighting = "item",
) -> dict[str, Any]:
	"""Raw accuracies, normalized transfer, suppression and the decision rule for one seed.

	Intervals are percentile bootstraps over task families: each replicate resamples families
	with replacement, keeping a family's items together and the four roles paired, so
	denominator uncertainty propagates into R and S. ``weighting`` is "item" (pooled items,
	default) or "family" (each family counts equally). Nothing is clipped. A ratio is None with
	a reason unless its denominator interval excludes zero and meets the declared minimum;
	bootstrap replicates with a non-positive denominator are left out of that ratio's interval.
	Accuracies, differences and ratios are exact fractions and thresholds are read as written,
	so boundaries and zero are compared exactly; values are converted to floats for output.
	"""
	if replicates < 100:
		raise ValueError("need at least 100 bootstrap replicates")
	if weighting not in ("item", "family"):
		raise ValueError("weighting must be item or family")
	thresholds = (min_headroom, min_clean_gain, min_effect, min_suppression_fraction)
	if not all(isinstance(t, int | float) and math.isfinite(t) for t in thresholds):
		raise ValueError("thresholds must be finite numbers")
	families = _validate(correct)
	sizes = [len(correct["teacher"][f]) for f in families]
	if weighting == "item":
		numerators = {r: [Fraction(sum(correct[r][f])) for f in families] for r in ROLES}
		weights = [Fraction(n) for n in sizes]
	else:
		numerators = {
			r: [Fraction(sum(correct[r][f]), n) for f, n in zip(families, sizes, strict=True)]
			for r in ROLES
		}
		weights = [Fraction(1)] * len(families)

	def accuracies(indices: Sequence[int]) -> dict[str, Fraction]:
		total = sum((weights[i] for i in indices), Fraction(0))
		return {r: sum((numerators[r][i] for i in indices), Fraction(0)) / total for r in ROLES}

	everything = list(range(len(families)))
	point_acc = accuracies(everything)
	point = _derive(point_acc)
	rng = random.Random(seed)
	draws_acc: dict[str, list[Fraction]] = {r: [] for r in ROLES}
	draws: dict[str, list[Fraction]] = {}
	for _ in range(replicates):
		acc = accuracies(rng.choices(everything, k=len(everything)))
		for role in ROLES:
			draws_acc[role].append(acc[role])
		for key, value in _derive(acc).items():
			if value is not None:
				draws.setdefault(key, []).append(value)

	headroom = _gate(
		point["headroom"], _bounds(draws["headroom"]), min_headroom, "teacher-base headroom"
	)
	clean_gain = _gate(
		point["clean_gain"], _bounds(draws["clean_gain"]), min_clean_gain, "clean gain"
	)

	def get(key: str) -> list[Fraction]:
		return draws.get(key, [])

	transfer = {
		"clean": _quantity(point["r_clean"], get("r_clean"), headroom, replicates),
		"defended": _quantity(point["r_defended"], get("r_defended"), headroom, replicates),
	}
	suppression = _quantity(point["suppression"], get("suppression"), clean_gain, replicates)
	effect = point["effect"]
	effect_interval = _bounds(get("effect"))
	fraction_met = fraction_lower = None
	if suppression["value"] is not None:
		fraction = _decimal(min_suppression_fraction)
		fraction_met = point["suppression"] >= fraction
		fraction_lower = _bounds(get("suppression"))[0] >= fraction
	effect_met = effect >= _decimal(min_effect)
	effect_lower = effect_interval[0] >= _decimal(min_effect)
	by_family = {
		f: {"items": n, **{r: sum(correct[r][f]) / n for r in ROLES}}
		for f, n in zip(families, sizes, strict=True)
	}
	return {
		"evidence": SINGLE_SEED,
		"weighting": weighting,
		"families": len(families),
		"items": sum(sizes),
		"raw": {
			"overall": {r: float(point_acc[r]) for r in ROLES},
			"overall_interval": {r: _floats(_bounds(draws_acc[r])) for r in ROLES},
			"by_family": by_family,
		},
		"headroom": headroom,
		"clean_gain": clean_gain,
		"normalized_transfer": transfer,
		"suppression": suppression,
		"decision": {
			"min_effect": min_effect,
			"min_suppression_fraction": min_suppression_fraction,
			"clean_minus_defended": {
				"value": float(effect),
				"interval": _floats(effect_interval),
			},
			"effect_met": effect_met,
			"suppression_met": fraction_met,
			"meets_target": effect_met or bool(fraction_met),
			"meets_target_at_interval_lower": effect_lower or bool(fraction_lower),
		},
		"bootstrap": {
			"method": "percentile, paired resampling of task families",
			"replicates": replicates,
			"seed": seed,
		},
	}


def _spread(values: dict[str, float | None]) -> dict[str, Any]:
	defined = [v for v in values.values() if v is not None]
	return {
		"values": values,
		"defined": len(defined),
		"mean": statistics.fmean(defined) if defined else None,
		"min": min(defined) if defined else None,
		"max": max(defined) if defined else None,
		"stdev": statistics.stdev(defined) if len(defined) > 1 else None,
	}


def transfer_report_seeds(runs: Mapping[str, Correct], **options: Any) -> dict[str, Any]:
	"""One report per training seed plus the spread of point estimates across seeds.

	``runs`` maps a training-seed label to a ``correct`` mapping. Seed variability is the spread
	of per-seed point estimates; it is not mixed into the within-seed bootstrap intervals.
	"""
	if not isinstance(runs, Mapping) or not runs:
		raise ValueError("runs must map at least one training-seed label to its results")
	reports = {label: transfer_report(correct, **options) for label, correct in runs.items()}
	rows = list(reports.values())
	quantities: dict[str, dict[str, float | None]] = {
		**{f"accuracy_{r}": {k: v["raw"]["overall"][r] for k, v in reports.items()} for r in ROLES},
		"r_clean": {k: v["normalized_transfer"]["clean"]["value"] for k, v in reports.items()},
		"r_defended": {
			k: v["normalized_transfer"]["defended"]["value"] for k, v in reports.items()
		},
		"suppression": {k: v["suppression"]["value"] for k, v in reports.items()},
		"clean_minus_defended": {
			k: v["decision"]["clean_minus_defended"]["value"] for k, v in reports.items()
		},
	}
	return {
		"evidence": SINGLE_SEED
		if len(runs) == 1
		else f"{len(runs)} training seeds; seed variability is reported separately from "
		"the bootstrap intervals.",
		"seeds": reports,
		"seed_variability": {name: _spread(values) for name, values in quantities.items()},
		"seeds_meeting_target": sum(r["decision"]["meets_target"] for r in rows),
	}
