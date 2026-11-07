"""Focused tests for the log-det coverage proxy."""

import numpy as np
import pytest
from harness.logdet import COVERAGE_DIM, CoverageSpec, LogDetCoverage
from harness.signals import SignalCalibration, SignalRow, fit_signals, raw_signals, score_signals


def stream(count: int, seed: int, width: int = 16) -> list[SignalRow]:
	"""Rows drawn i.i.d. from one fixed distribution."""
	draws = np.random.default_rng(seed).standard_normal((count, width))
	return [SignalRow(f"request {i}", float(i), tuple(row)) for i, row in enumerate(draws)]


def test_sherman_morrison_matches_direct_log_det() -> None:
	rng = np.random.default_rng(0)
	tracker = LogDetCoverage(COVERAGE_DIM, ridge=0.5)
	a = 0.5 * np.eye(COVERAGE_DIM)
	for _ in range(300):
		v = rng.standard_normal(COVERAGE_DIM)
		v *= rng.uniform(0.1, 1.0) / np.linalg.norm(v)
		before = np.linalg.slogdet(a)[1]
		a += np.outer(v, v)
		assert tracker.update(v) == pytest.approx(np.linalg.slogdet(a)[1] - before, abs=1e-9)
	tracker.reset()
	assert tracker.update(np.zeros(COVERAGE_DIM)) == 0.0


def test_repeats_shrink_and_new_directions_gain_more() -> None:
	tracker = LogDetCoverage(COVERAGE_DIM)
	e = np.eye(COVERAGE_DIM)
	repeats = [tracker.update(e[0]) for _ in range(5)]
	assert all(b < a for a, b in zip(repeats, repeats[1:], strict=False))
	assert tracker.update(e[1]) > repeats[-1]
	with pytest.raises(ValueError, match="norm"):
		tracker.update(2 * e[0])


def test_windowed_feature_does_not_grow_with_window_length() -> None:
	spec = CoverageSpec(seed=7, input_dim=16)
	short = raw_signals(stream(50, 1), coverage=spec)["coverage_mean_logdet"]
	long = raw_signals(stream(400, 1), coverage=spec)["coverage_mean_logdet"]
	assert 0 < long < short
	assert "coverage_mean_logdet" not in raw_signals(stream(50, 1))


def test_seeded_projection_is_deterministic_and_recorded() -> None:
	spec = CoverageSpec(seed=7, input_dim=16)
	same = CoverageSpec(seed=7, input_dim=16)
	other = CoverageSpec(seed=8, input_dim=16)
	assert np.array_equal(spec.matrix(), same.matrix())
	assert spec.projection_digest == same.projection_digest != other.projection_digest
	rows = stream(30, 2)
	assert raw_signals(rows, coverage=spec) == raw_signals(rows, coverage=same)
	projected = spec.project(np.asarray([r.embedding for r in rows]))
	assert projected.shape == (30, COVERAGE_DIM)
	assert np.linalg.norm(projected, axis=1).max() <= 1 + 1e-12


def test_calibration_records_spec_and_rejects_a_changed_projection() -> None:
	spec = CoverageSpec(seed=7, input_dim=16)
	horizons = [stream(20, i) for i in range(19)]
	fitted = fit_signals(horizons, profiles=["g"] * 19, coverage=spec)
	restored = SignalCalibration.model_validate_json(fitted.model_dump_json())
	assert restored.coverage == spec
	entry = score_signals(stream(20, 99), profile="g", calibration=restored)["metrics"]
	assert entry["coverage_mean_logdet"]["p_value"] is not None
	legacy = fit_signals(horizons, profiles=["g"] * 19)
	assert (
		"coverage_mean_logdet"
		not in score_signals(horizons[0], profile="g", calibration=legacy)["metrics"]
	)
	payload = fitted.model_dump()
	payload["coverage"]["projection_digest"] = "0" * 64
	with pytest.raises(ValueError, match="digest"):
		SignalCalibration.model_validate(payload)


def test_ranks_use_matched_request_count_bands() -> None:
	spec = CoverageSpec(seed=7, input_dim=2)
	same = [SignalRow(f"r{i}", float(i), (1.0, 0.0)) for i in range(100)]
	fitted = fit_signals([same] * 19, profiles=["g"] * 19, coverage=spec)
	assert set(fitted.nulls["g"]) >= {"coverage_mean_logdet[64-127]"}
	short = score_signals(same[:2], profile="g", calibration=fitted)["metrics"]
	assert short["coverage_mean_logdet"]["p_value"] is None
	assert "of 2-3 requests" in short["coverage_mean_logdet"]["abstain"]
	matched = score_signals(same, profile="g", calibration=fitted)["metrics"]
	assert matched["coverage_mean_logdet"]["length_band"] == "64-127"
	assert matched["coverage_mean_logdet"]["p_value"] == 1.0


def test_ridge_bounds_and_frozen_spec() -> None:
	for ridge in (1e-20, float("inf"), 0.0):
		with pytest.raises(ValueError):
			CoverageSpec(seed=7, input_dim=2, ridge=ridge)
		with pytest.raises(ValueError):
			LogDetCoverage(COVERAGE_DIM, ridge)
	tracker = LogDetCoverage(COVERAGE_DIM, 1e-3)
	e = np.eye(COVERAGE_DIM)[0]
	for k in range(200):
		assert tracker.update(e) == pytest.approx(np.log1p(1 / (1e-3 + k)), rel=1e-12)
	spec = CoverageSpec(seed=7, input_dim=2)
	with pytest.raises(ValueError):
		spec.seed = 8
	same = [SignalRow(f"r{i}", float(i), (1.0, float(i % 3))) for i in range(10)]
	fitted = fit_signals([same] * 19, profiles=["g"] * 19, coverage=spec)
	with pytest.raises(ValueError):
		fitted.coverage = CoverageSpec(seed=8, input_dim=2, ridge=0.001)


def test_evidence_round_trip_and_paired_band_comparison() -> None:
	from harness.calibration import BenignRequest, Calibration, HarnessConfig
	from harness.evidence import check_evidence, fit_evidence

	calibration = Calibration.model_construct(config=HarnessConfig(horizon_attempts=100))
	training = [
		BenignRequest(account=f"fit-{a}", text=f"fit {a} {i}") for a in range(19) for i in range(4)
	]
	vectors = np.random.default_rng(7).normal(size=(len(training), 4))
	spec = CoverageSpec(seed=9, input_dim=4, ridge=0.5)
	fitted = fit_evidence(training, vectors, calibration, coverage=spec)
	restored = SignalCalibration.model_validate_json(fitted.model_dump_json())
	assert restored.coverage == spec
	assert "novelty_mean_nn[4-7]" in restored.nulls["all"]
	heldout = [
		BenignRequest(account=f"test-{a}", text=f"heldout {a} {i}")
		for a, n in enumerate((4, 2, 1))
		for i in range(n)
	]
	report = check_evidence(heldout, vectors[:7], calibration, restored)
	comparison = report["coverage_comparison"]
	assert comparison["qualified"] is False
	assert len(comparison["bands"]) == 3
	for band in comparison["bands"]:
		assert set(band["metrics"]) == {"coverage_mean_logdet", "novelty_mean_nn"}
		for metric in band["metrics"].values():
			assert metric["ranked"] + metric["abstained"] == band["horizons"]
		if band["length_band"] == "4-7":
			assert band["paired_ranked"] == 1
		else:
			assert band["paired_ranked"] == 0
			assert all(m["abstained"] == 1 for m in band["metrics"].values())
	legacy = fit_evidence(training, vectors, calibration)
	assert "coverage_comparison" not in check_evidence(heldout, vectors[:7], calibration, legacy)
	assert all(
		"coverage_mean_logdet" not in u["signals"]["metrics"]
		for u in check_evidence(heldout, vectors[:7], calibration, legacy)["units"]
	)


def test_incremental_snapshot_replay() -> None:
	"""A serialized inverse reproduces uninterrupted gains without prefix storage."""
	import json

	from harness.logdet import LogDetCoverage

	tracker = LogDetCoverage(dim=3)
	tracker.update(np.array([1.0, 0.0, 0.0]))
	state = json.loads(json.dumps(tracker.snapshot(), allow_nan=False))
	restored = LogDetCoverage.restore(state)
	for vector in np.eye(3):
		assert restored.update(vector) == tracker.update(vector)
	inverse = restored.snapshot()["inverse"]
	assert isinstance(inverse, list)
	assert len(inverse) == 3


def test_incremental_snapshot_rejects_corruption() -> None:
	from harness.logdet import LogDetCoverage

	state = LogDetCoverage(dim=2).snapshot()
	state["inverse"] = [[1.0, 2.0], [2.0, 1.0]]
	with pytest.raises(ValueError, match="positive ridge"):
		LogDetCoverage.restore(state)
