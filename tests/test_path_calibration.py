import math

import numpy as np
import pytest
from harness.calibration import binomial_upper
from harness.path_calibration import (
	PathCalibration,
	PathScoring,
	PathState,
	audit_upper_bound,
	fit_path_calibration,
	period_alpha,
)

SCORING = PathScoring(name="test-v1", streams=("s",))


def _fit(maxima, **options):
	return fit_path_calibration([[[float(m)]] for m in maxima], scoring=SCORING, **options)


def test_rank_ties_and_running_max() -> None:
	state = PathState(_fit([1, 2, 2, 3], alpha=0.4, horizon=4, stratum="a"))
	results = [state.observe([score]).p_value for score in [1, 2, 1, 3]]
	assert results == [1, 0.8, 0.8, 0.4]
	assert state.stopped and state.stop_look == 4


def test_resolution_and_stratum_abstention() -> None:
	insufficient = _fit(range(98), alpha=0.01, horizon=2, stratum="small")
	adequate = _fit(range(99), alpha=0.01, horizon=2, stratum="large")
	assert insufficient.abstained and insufficient.abstention_reason
	assert not adequate.abstained
	result = PathState(insufficient).observe([1000])
	assert result.decision == "abstain" and result.p_value is None and not result.stopped


def test_stop_is_durable_and_horizon_expires() -> None:
	calibration = _fit(range(9), alpha=0.1, horizon=2, stratum="a")
	state = PathState(calibration)
	assert state.observe([10]).decision == "stop"
	assert state.observe([-100]).decision == "stop"
	assert state.stop_look == 1 and state.look_count == 1
	other = PathState(calibration)
	other.observe([0])
	other.observe([1])
	expired = other.observe([100])
	assert expired.decision == "horizon expired" and not expired.stopped
	assert expired.look_count == 2 and other.running_max == 1
	last = PathState(calibration)
	last.observe([0])
	last.observe([100])
	assert last.observe([0]).decision == "horizon expired" and last.stopped


def test_scoring_is_shared_by_fitting_and_runtime() -> None:
	weighted = PathScoring(name="two-v1", streams=("a", "b"), weights=(1.0, 0.5))
	assert weighted.path_maximum([[1, 3], [4, 2]], horizon=2) == 4
	assert weighted.path_maximum([[1, 10]], horizon=2) == 5
	# 99 identical benign campaigns cannot stop one another once both sides share the scoring.
	calibration = fit_path_calibration(
		[[[0, 100]]] * 99, scoring=weighted, alpha=0.01, horizon=1, stratum="a"
	)
	assert not PathState(calibration).observe([0, 100]).stopped
	with pytest.raises(ValueError, match="one score per stream"):
		PathState(calibration).observe([0])
	with pytest.raises(ValueError, match="longer than the horizon"):
		fit_path_calibration([[[1, 1]] * 3], scoring=weighted, alpha=0.5, horizon=2, stratum="a")
	for path in ([], [[]], [[float("nan"), 0]]):
		with pytest.raises(ValueError):
			weighted.path_maximum(path, horizon=2)
	for bad in ({"streams": ()}, {"streams": ("a", "a")}, {"streams": ("a",), "weights": (0,)}):
		with pytest.raises(ValueError):
			PathScoring(name="x", **bad)
	with pytest.raises(ValueError):
		PathState(_fit([], alpha=0.1, horizon=1, stratum="a")).observe([float("nan")])


def test_artifact_round_trip_and_signed_zero_hash() -> None:
	calibration = _fit([0.0, -0.0, 1], alpha=0.5, horizon=1, stratum="a")
	assert PathCalibration.model_validate_json(calibration.model_dump_json()) == calibration
	assert _fit([-0.0, 0.0, 1], alpha=0.5, horizon=1, stratum="a").input_hash == (
		calibration.input_hash
	)
	other = PathScoring(name="test-v2", streams=("s",))
	assert (
		fit_path_calibration(
			[[[0.0]], [[1.0]]], scoring=other, alpha=0.5, horizon=1, stratum="a"
		).input_hash
		!= _fit([0, 1], alpha=0.5, horizon=1, stratum="a").input_hash
	)


def test_period_schedule_and_audit() -> None:
	assert sum(period_alpha(0.05, j) for j in range(1, 100_001)) <= 0.05
	assert audit_upper_bound(0, 299) < 0.01
	assert audit_upper_bound(0, 2995) < 0.001
	assert audit_upper_bound(0, 299) == pytest.approx(1 - 0.05 ** (1 / 299))
	assert audit_upper_bound(3, 10) == pytest.approx(0.6066242161)
	assert 200 / 2995 < audit_upper_bound(200, 2995) < 0.08
	assert binomial_upper(10, 1000) == pytest.approx(audit_upper_bound(10, 1000))
	for counts in ((1.5, 2), (0, 1.5), (True, 2), (0, 0), (3, 2), (-1, 2)):
		with pytest.raises(ValueError):
			audit_upper_bound(*counts)


def test_exchangeable_dependent_paths() -> None:
	rng = np.random.default_rng(826)
	trials, n, horizon, alpha = 4000, 99, 8, 0.05
	false_stops = 0
	for _ in range(trials):
		paths = np.cumsum(rng.normal(size=(n + 1, horizon)), axis=1)
		calibration = fit_path_calibration(
			[[[float(x)] for x in path] for path in paths[:n]],
			scoring=SCORING,
			alpha=alpha,
			horizon=horizon,
			stratum="a",
		)
		state = PathState(calibration)
		for score in paths[n]:
			state.observe([float(score)])
		false_stops += state.stopped
	assert false_stops / trials <= alpha + 4 * math.sqrt(alpha * (1 - alpha) / trials)


def test_matched_observable_paths() -> None:
	calibration = _fit(range(19), alpha=0.05, horizon=4, stratum="a")
	first, second = PathState(calibration), PathState(calibration)
	for score in [1, 3, 25, 0]:
		assert first.observe([score]) == second.observe([score])
