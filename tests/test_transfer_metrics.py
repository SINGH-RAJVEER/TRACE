"""Tests use synthetic per-item correctness, without a teacher or student training."""

from typing import Any

import pytest
from distillation import transfer_metrics
from distillation.transfer_metrics import SuppressionGate

COVERED = SuppressionGate(covered=True, reason=None, digest="fixture", identity={})


# The metrics themselves are tested under a covering gate; the refusal is tested without one.
def transfer_report(correct: Any, **options: Any) -> dict[str, Any]:
	return transfer_metrics.transfer_report(correct, gate=COVERED, **options)


def transfer_report_seeds(runs: Any, **options: Any) -> dict[str, Any]:
	return transfer_metrics.transfer_report_seeds(runs, gate=COVERED, **options)


def flags(correct: int, total: int) -> list[bool]:
	return [True] * correct + [False] * (total - correct)


def uniform(
	teacher: int, base: int, clean: int, defended: int, *, families: int = 6, items: int = 10
) -> dict[str, dict[str, list[bool]]]:
	"""Identical families, so every bootstrap resample gives the same accuracies."""
	counts = {"teacher": teacher, "base": base, "clean": clean, "defended": defended}
	return {role: {f"f{i}": flags(k, items) for i in range(families)} for role, k in counts.items()}


def test_hand_computed_values() -> None:
	report = transfer_report(uniform(9, 3, 7, 5))
	overall = report["raw"]["overall"]
	assert overall == pytest.approx({"teacher": 0.9, "base": 0.3, "clean": 0.7, "defended": 0.5})
	assert report["raw"]["by_family"]["f0"]["items"] == 10
	assert report["raw"]["by_family"]["f3"]["defended"] == pytest.approx(0.5)
	transfer = report["normalized_transfer"]
	assert transfer["clean"]["value"] == pytest.approx(0.4 / 0.6)
	assert transfer["defended"]["value"] == pytest.approx(0.2 / 0.6)
	assert transfer["defended"]["interval"] == pytest.approx([0.2 / 0.6] * 2)
	assert report["suppression"]["value"] == pytest.approx(0.5)
	assert report["evidence"].startswith("Development evidence")


def test_undefined_without_headroom_or_clean_gain() -> None:
	flat = transfer_report(uniform(5, 5, 7, 6))
	assert flat["normalized_transfer"]["clean"]["value"] is None
	assert "interval includes zero" in flat["normalized_transfer"]["clean"]["reason"]
	assert flat["suppression"]["value"] == pytest.approx(0.5)
	small = transfer_report(uniform(4, 3, 3, 3), min_headroom=0.15)
	assert "below the minimum" in small["normalized_transfer"]["defended"]["reason"]
	stalled = transfer_report(uniform(9, 3, 3, 2))
	assert stalled["suppression"]["value"] is None
	assert "clean gain" in stalled["suppression"]["reason"]
	assert stalled["normalized_transfer"]["clean"]["value"] == pytest.approx(0.0)
	worse = transfer_report(uniform(3, 9, 9, 9))
	assert "not positive" in worse["headroom"]["reason"]


def test_values_are_not_clipped() -> None:
	beats_teacher = transfer_report(uniform(9, 3, 10, 2))
	assert beats_teacher["normalized_transfer"]["clean"]["value"] > 1
	assert beats_teacher["normalized_transfer"]["defended"]["value"] < 0
	assert beats_teacher["suppression"]["value"] > 1
	reversed_effect = transfer_report(uniform(9, 3, 5, 7))
	assert reversed_effect["suppression"]["value"] == pytest.approx(-1.0)


def test_bootstrap_resamples_whole_families() -> None:
	correct = {
		"teacher": {"a": flags(4, 4), "b": flags(0, 2)},
		"base": {"a": flags(0, 4), "b": flags(0, 2)},
		"clean": {"a": flags(2, 4), "b": flags(1, 2)},
		"defended": {"a": flags(1, 4), "b": flags(1, 2)},
	}
	report = transfer_report(correct, seed=3)
	# Resampling families gives 25%, 50% and 25% mass on {a,a}, {a,b} and {b,b}; resampling
	# the six items independently would almost never reach an accuracy of zero.
	assert report["raw"]["overall_interval"]["teacher"] == [0.0, 1.0]
	family = transfer_report(correct, weighting="family")
	assert family["raw"]["overall"]["teacher"] == pytest.approx(0.5)
	assert report["raw"]["overall"]["teacher"] == pytest.approx(4 / 6)


def test_seeded_and_paired() -> None:
	# Families differ widely in difficulty, but every family gains exactly two items from
	# base to clean, so only paired resampling gives a degenerate clean-gain interval.
	base = [0, 2, 4, 6, 8]
	correct = {
		role: {f"f{i}": flags(min(10, k + shift), 10) for i, k in enumerate(base)}
		for role, shift in {"teacher": 4, "base": 0, "clean": 2, "defended": 1}.items()
	}
	first = transfer_report(correct, seed=11, replicates=300)
	assert first["clean_gain"]["interval"] == pytest.approx([0.2, 0.2])
	low, high = first["raw"]["overall_interval"]["base"]
	assert high - low > 0.2
	assert first == transfer_report(correct, seed=11, replicates=300)
	varied = {
		role: {f"f{i}": flags((i * 7 + shift) % 11, 10) for i in range(40)}
		for role, shift in {"teacher": 6, "base": 0, "clean": 3, "defended": 2}.items()
	}
	intervals = [
		transfer_report(varied, seed=seed, replicates=300)["raw"]["overall_interval"]
		for seed in (11, 12)
	]
	assert intervals[0] != intervals[1]
	with pytest.raises(ValueError):
		transfer_report(correct, replicates=10)


def test_inclusive_thresholds_at_exact_boundaries() -> None:
	at_headroom = transfer_report(uniform(90, 80, 85, 82, items=100))
	assert at_headroom["headroom"]["usable"]
	at_gain = transfer_report(uniform(90, 30, 35, 33, items=100))
	assert at_gain["clean_gain"]["usable"]
	at_effect = transfer_report(uniform(90, 30, 70, 65, items=100))["decision"]
	assert at_effect["effect_met"] and at_effect["meets_target"]
	at_fraction = transfer_report(uniform(90, 4, 12, 10, items=100), min_clean_gain=0.05)
	assert at_fraction["suppression"]["value"] == pytest.approx(0.25)
	assert at_fraction["decision"]["suppression_met"]


def test_exact_zero_and_sub_item_gaps() -> None:
	# Family weighting once left a floating 5.6e-17 lower bound that passed the zero gate.
	def counts(teacher, base, clean, defended, sizes):
		roles = {"teacher": teacher, "base": base, "clean": clean, "defended": defended}
		return {
			role: {f"f{i}": flags(k, n) for i, (k, n) in enumerate(zip(ks, sizes, strict=True))}
			for role, ks in roles.items()
		}

	zero = counts([3, 3, 3, 2], [0, 0, 0, 3], [3] * 4, [2] * 4, [5] * 4)
	for weighting in ("item", "family"):
		report = transfer_report(zero, weighting=weighting)
		assert report["headroom"]["interval"][0] == 0
		assert not report["headroom"]["usable"]
	# The exact family-weighted effect is 0.04999999920009599, just below 0.05.
	sizes = [25_000, 25_003]
	near = counts(sizes, [0, 0], [15_000, 10_000], [4_167, 18_334], sizes)
	decision = transfer_report(near, weighting="family", replicates=100)["decision"]
	assert not decision["effect_met"] and not decision["meets_target"]


def test_decision_rule() -> None:
	large = transfer_report(uniform(90, 30, 70, 50, items=100))
	assert large["decision"]["effect_met"] and large["decision"]["meets_target"]
	by_fraction = transfer_report(uniform(90, 30, 40, 36, items=100))
	decision = by_fraction["decision"]
	assert not decision["effect_met"] and decision["suppression_met"]
	assert decision["meets_target"] and decision["meets_target_at_interval_lower"]
	strict = transfer_report(uniform(90, 30, 40, 36, items=100), min_suppression_fraction=0.5)
	assert not strict["decision"]["meets_target"]
	assert strict["decision"]["clean_minus_defended"]["value"] == pytest.approx(0.04)


def test_seed_variability_is_separate() -> None:
	runs = {"1": uniform(9, 3, 7, 5), "2": uniform(9, 3, 6, 5)}
	report = transfer_report_seeds(runs)
	spread = report["seed_variability"]["suppression"]
	assert spread["values"]["1"] == pytest.approx(0.5)
	assert spread["mean"] == pytest.approx((0.5 + 1 / 3) / 2)
	assert spread["stdev"] is not None
	assert report["seeds_meeting_target"] == 2
	assert report["seeds"]["1"]["suppression"]["interval"] == pytest.approx([0.5, 0.5])
	assert transfer_report_seeds({"1": runs["1"]})["evidence"].startswith("Development evidence")


def test_input_validation() -> None:
	good = uniform(9, 3, 7, 5)
	with pytest.raises(ValueError, match="roles"):
		transfer_report({k: v for k, v in good.items() if k != "base"})
	with pytest.raises(ValueError, match="same task families"):
		transfer_report({**good, "base": {"f0": good["base"]["f0"], "x": good["base"]["f1"]}})
	with pytest.raises(ValueError, match="item count"):
		transfer_report({**good, "clean": {**good["clean"], "f0": flags(3, 5)}})
	with pytest.raises(ValueError, match="two task families"):
		transfer_report(uniform(9, 3, 7, 5, families=1))
	for name in ("min_headroom", "min_clean_gain", "min_effect", "min_suppression_fraction"):
		with pytest.raises(ValueError, match="finite"):
			transfer_report(good, **{name: float("nan")})
	for runs in ([good], "seed", {}):
		with pytest.raises(ValueError, match="runs"):
			transfer_report_seeds(runs)
	with pytest.raises(ValueError, match="mapping"):
		transfer_report({**good, "clean": [True]})  # type: ignore[dict-item]


def test_suppression_and_decision_are_refused_without_a_covering_gate() -> None:
	correct = uniform(9, 3, 7, 5)
	report = transfer_metrics.transfer_report(correct)
	assert report["suppression"]["value"] is None
	assert "No passing clean-baseline gate" in report["suppression"]["reason"]
	assert report["suppression_gate"]["covered"] is False
	assert report["decision"]["meets_target"] is None
	assert report["decision"]["effect_met"] is None
	assert report["decision"]["clean_minus_defended"]["value"] == pytest.approx(0.2)
	assert report["raw"]["overall"]["clean"] == pytest.approx(0.7)
	failing = SuppressionGate(covered=False, reason="other student", digest="d", identity={})
	assert transfer_metrics.transfer_report(correct, gate=failing)["suppression"]["reason"] == (
		"other student"
	)
	runs = transfer_metrics.transfer_report_seeds({"1": correct, "2": correct})
	assert runs["seeds_meeting_target"] is None
	assert runs["seed_variability"]["suppression"]["defined"] == 0
