"""Tests use synthetic per-item correctness, without a teacher or student training."""

import pytest
from distillation.transfer_metrics import transfer_report, transfer_report_seeds


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
	# Resampling families gives 25% mass each on {a,a}, {a,b} and {b,b}; resampling
	# the six items independently would almost never reach an accuracy of zero.
	assert report["raw"]["overall_interval"]["teacher"] == [0.0, 1.0]
	family = transfer_report(correct, weighting="family")
	assert family["raw"]["overall"]["teacher"] == pytest.approx(0.5)
	assert report["raw"]["overall"]["teacher"] == pytest.approx(4 / 6)


def test_seeded_and_paired() -> None:
	correct = {
		role: {f"f{i}": flags((i + shift) % 7, 6) for i in range(8)}
		for role, shift in {"teacher": 5, "base": 0, "clean": 3, "defended": 2}.items()
	}
	first = transfer_report(correct, seed=11, replicates=300)
	assert first == transfer_report(correct, seed=11, replicates=300)
	assert first != transfer_report(correct, seed=12, replicates=300)
	with pytest.raises(ValueError):
		transfer_report(correct, replicates=10)


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
