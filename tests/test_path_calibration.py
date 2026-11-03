import math

import numpy as np
import pytest
from harness.path_calibration import (
	PathState,
	audit_upper_bound,
	fit_path_calibration,
	path_maximum,
	period_alpha,
)


def test_rank_ties_and_running_max() -> None:
	state = PathState(fit_path_calibration([1, 2, 2, 3], alpha=0.4, horizon=4, stratum="a"))
	results = [state.observe(score).p_value for score in [1, 2, 1, 3]]
	assert results == [1, 0.8, 0.8, 0.4]
	assert state.stopped and state.stop_look == 4


def test_resolution_and_stratum_abstention() -> None:
	insufficient = fit_path_calibration(range(98), alpha=0.01, horizon=2, stratum="small")
	adequate = fit_path_calibration(range(99), alpha=0.01, horizon=2, stratum="large")
	assert insufficient.abstained and insufficient.abstention_reason
	assert not adequate.abstained
	result = PathState(insufficient).observe(1000)
	assert result.decision == "abstain" and result.p_value is None and not result.stopped


def test_stop_is_durable_and_horizon_expires() -> None:
	calibration = fit_path_calibration(range(9), alpha=0.1, horizon=2, stratum="a")
	state = PathState(calibration)
	assert state.observe(10).decision == "stop"
	assert state.observe(-100).decision == "stop"
	assert state.stop_look == 1 and state.look_count == 1
	other = PathState(calibration)
	other.observe(0)
	other.observe(1)
	expired = other.observe(100)
	assert expired.decision == "horizon expired" and not expired.stopped
	assert expired.look_count == 2 and other.running_max == 1
	last = PathState(calibration)
	last.observe(0)
	last.observe(100)
	assert last.observe(0).decision == "horizon expired" and last.stopped


def test_path_maximum_and_input_validation() -> None:
	assert path_maximum([[1, 3], [4, 2]]) == 4
	assert path_maximum([[1, 3], [4, 2]], transform=lambda scores: scores[0] - scores[1]) == 2
	for scores in ([], [[]], [[float("nan")]]):
		with pytest.raises(ValueError):
			path_maximum(scores)
	with pytest.raises(ValueError):
		fit_path_calibration([float("inf")], alpha=0.1, horizon=1, stratum="a")
	with pytest.raises(ValueError):
		PathState(fit_path_calibration([], alpha=0.1, horizon=1, stratum="a")).observe(float("nan"))


def test_period_schedule_and_audit() -> None:
	assert sum(period_alpha(0.05, j) for j in range(1, 100_001)) <= 0.05
	assert audit_upper_bound(0, 299) < 0.01
	assert audit_upper_bound(0, 2995) < 0.001
	assert audit_upper_bound(0, 299) == pytest.approx(1 - 0.05 ** (1 / 299))
	assert audit_upper_bound(3, 10) == pytest.approx(0.6066242161)


def test_exchangeable_dependent_paths() -> None:
	rng = np.random.default_rng(826)
	trials, n, horizon, alpha = 4000, 99, 8, 0.05
	false_stops = 0
	for _ in range(trials):
		paths = np.cumsum(rng.normal(size=(n + 1, horizon)), axis=1)
		calibration = fit_path_calibration(
			paths[:n].max(axis=1), alpha=alpha, horizon=horizon, stratum="a"
		)
		state = PathState(calibration)
		for score in paths[n]:
			state.observe(float(score))
		false_stops += state.stopped
	assert false_stops / trials <= alpha + 4 * math.sqrt(alpha * (1 - alpha) / trials)


def test_matched_observable_paths() -> None:
	calibration = fit_path_calibration(range(19), alpha=0.05, horizon=4, stratum="a")
	first, second = PathState(calibration), PathState(calibration)
	for score in [1, 3, 25, 0]:
		assert first.observe(score) == second.observe(score)
