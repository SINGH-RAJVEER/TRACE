import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from harness.accumulators import AccumulatorSpec
from harness.bench import serving_costs
from harness.calibration import (
	BenignRequest,
	Calibration,
	HarnessConfig,
	LevelThresholds,
	calibrate,
	fit_threshold,
	load_calibration,
	peak_per_minute,
	save_calibration,
)
from harness.mmd import Array, WindowScorer, kernel, mmd2_unbiased
from harness.policy import MMDPolicy, QuotaPolicy, RatePolicy
from harness.pow import solve
from harness.replay import replay

DIMENSION = 8


class FakeEncoder:
	"""Benign texts come from one Gaussian; texts starting with "attack" from a shifted one."""

	def encode(self, texts: list[str]) -> Array:
		rows = []
		for text in texts:
			seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
			vector = np.random.default_rng(seed).normal(size=DIMENSION)
			if text.startswith("attack"):
				vector[0] += 3.0
			rows.append(vector / np.linalg.norm(vector))
		return np.array(rows)

	def identity(self) -> dict[str, Any]:
		return {"model": "fake"}


def _benign(prefix: str, accounts: int, per_account: int) -> list[BenignRequest]:
	return [
		BenignRequest(account=f"{prefix}-{a}", text=f"{prefix} {a} {i}", time=float(i))
		for a in range(accounts)
		for i in range(per_account)
	]


@pytest.fixture(scope="module")
def calibrated() -> tuple[Calibration, WindowScorer]:
	calibration, references = calibrate(
		config=HarnessConfig(window_size=10, reference_windows=3, bandwidth_sample=60),
		encoder=FakeEncoder(),
		fitting=_benign("fit", 4, 30),
		score_calibration=_benign("score", 20, 50),
		policy_calibration=_benign("policy", 200, 40),
	)
	return calibration, WindowScorer(references, calibration.sigma)


def test_unbiased_mmd_matches_its_definition() -> None:
	rng = np.random.default_rng(0)
	x, z = rng.normal(size=(5, 3)), rng.normal(size=(6, 3))

	def k(a: Array, b: Array) -> float:
		return float(kernel(np.array([[np.sum((a - b) ** 2)]]), 1.3)[0, 0])

	xx = sum(k(x[i], x[j]) for i in range(5) for j in range(5) if i != j) / 20
	zz = sum(k(z[i], z[j]) for i in range(6) for j in range(6) if i != j) / 30
	xz = sum(k(a, b) for a in x for b in z) / 30
	assert mmd2_unbiased(x, z, 1.3) == pytest.approx(xx + zz - 2 * xz)
	references = np.stack([rng.normal(size=(5, 3)) for _ in range(3)])
	expected = np.mean([mmd2_unbiased(x, r, 1.3) for r in references])
	assert WindowScorer(references, 1.3).score(x) == pytest.approx(expected)


def test_thresholds_use_strict_exceedance_and_report_achieved_rates() -> None:
	assert fit_threshold([5, 4, 3, 2, 1, 0, 0, 0, 0, 0], 0.2) == (3, 0.2)
	assert fit_threshold([7, 7, 7, 1], 0.25) == (7, 0.0)
	assert fit_threshold([3, 2, 1], 0.01) == (3, 0.0)
	assert peak_per_minute([0, 10, 59.9, 60, 61, 200]) == 4


def test_accumulators_follow_their_update_rules() -> None:
	p = math.exp(-3)
	assert AccumulatorSpec(kind="instantaneous").update(9.0, p) == pytest.approx(3)
	paper = AccumulatorSpec(kind="decayed_cusum", decay=0.9, drift=1.0)
	assert paper.update(2.0, p) == pytest.approx(0.9 * 2 + 3 - 1)
	assert paper.update(0.0, 1.0) == 0.0
	log_e = math.log(0.5) - 0.5 * math.log(p)
	assert AccumulatorSpec(kind="e_process").update(-5.0, p) == pytest.approx(-5 + log_e)
	assert AccumulatorSpec(kind="e_cusum").update(-5.0, p) == pytest.approx(log_e)


def test_calibration_meets_benign_targets_and_round_trips(
	calibrated: tuple[Calibration, WindowScorer], tmp_path: Path
) -> None:
	calibration, scorer = calibrated
	assert calibration.counts["score_calibration_windows"] == 100
	assert calibration.counts["policy_calibration_horizons"] == 200
	for thresholds in calibration.thresholds.values():
		assert thresholds.thresholds == sorted(thresholds.thresholds)
		for achieved, target in zip(thresholds.achieved, (0.10, 0.05, 0.02, 0.01), strict=True):
			assert achieved <= target
	assert calibration.baselines["quota"].thresholds == [40]
	assert "rate_per_minute" in calibration.baselines
	save_calibration(tmp_path, calibration, scorer.references)
	loaded, loaded_scorer = load_calibration(tmp_path)
	assert loaded.thresholds == calibration.thresholds
	window = FakeEncoder().encode([f"probe {i}" for i in range(10)])
	assert loaded_scorer.score(window) == pytest.approx(scorer.score(window))
	np.save(tmp_path / "references.npy", scorer.references + 1)
	with pytest.raises(ValueError, match="reference windows differ"):
		load_calibration(tmp_path)


def test_shifted_traffic_escalates_and_benign_traffic_mostly_does_not(
	calibrated: tuple[Calibration, WindowScorer], tmp_path: Path
) -> None:
	calibration, scorer = calibrated
	requests = _benign("held-out", 50, 40) + [
		BenignRequest(account="attacker", text=f"attack {i}") for i in range(60)
	]
	summary = replay(
		calibration=calibration,
		scorer=scorer,
		requests=requests,
		output=tmp_path / "decisions.jsonl",
		policy="mmd",
		encoder=FakeEncoder(),
	)
	assert summary["horizons"] == 51
	assert summary["reached"]["L4"] >= 1
	assert summary["reached_rate"]["L1"] < 0.5
	lines = (tmp_path / "decisions.jsonl").read_text().splitlines()
	assert '"action": "refuse"' in lines[-1]


def test_graduated_actions_space_challenge_and_refuse(
	calibrated: tuple[Calibration, WindowScorer],
) -> None:
	calibration, scorer = calibrated
	forced = calibration.model_copy(deep=True)
	forced.thresholds["paper"] = LevelThresholds(
		thresholds=[-1, -1, -1, 1e9], achieved=[0, 0, 0, 0], units=1
	)
	policy = MMDPolicy(forced, scorer, encoder=FakeEncoder(), difficulty=4)
	decisions = [policy.admit("a", f"q {i}", now=100.0) for i in range(12)]
	assert decisions[8].action == "serve" and decisions[9].window_completed
	assert decisions[9].action == "challenge" and decisions[9].level == 3
	assert [d.release_at for d in decisions[9:12]] == [100.0, 101.0, 102.0]
	challenge = decisions[10]
	counter, _ = solve(challenge.challenge)  # type: ignore[arg-type]
	assert challenge.client_view()["challenge"]["prefix"].endswith(":")
	assert not policy.redeem(challenge, "different text", counter, now=100.0)
	assert policy.redeem(challenge, "q 10", counter, now=100.0)
	assert not policy.redeem(challenge, "q 10", counter, now=100.0)
	late = decisions[11]
	assert not policy.redeem(late, "q 11", solve(late.challenge)[0], now=1000.0)  # type: ignore[arg-type]
	forced.thresholds["paper"].thresholds = [-1, -1, -1, -1]
	blocked = MMDPolicy(forced, scorer, encoder=FakeEncoder())
	outcome = [blocked.admit("b", f"q {i}", now=0.0) for i in range(12)]
	assert outcome[9].action == "refuse" and outcome[11].action == "refuse"
	fresh = blocked.admit("b", "q 12", now=calibration.config.horizon_seconds)
	assert (fresh.attempt, fresh.level, fresh.action) == (1, 0, "serve")


def test_quota_and_rate_baselines_block_for_the_horizon() -> None:
	quota = QuotaPolicy(3, horizon_seconds=100, horizon_attempts=1000)
	assert [quota.admit("a", "q", float(t)).action for t in range(5)][-2:] == ["refuse"] * 2
	assert quota.admit("a", "q", 100.0).action == "serve"
	rate = RatePolicy(2, horizon_seconds=1000, horizon_attempts=1000)
	actions = [rate.admit("a", "q", t).action for t in (0, 1, 2, 100)]
	assert actions == ["serve", "serve", "refuse", "refuse"]


def test_serving_cost_benchmark_times_each_unit_separately(
	calibrated: tuple[Calibration, WindowScorer],
) -> None:
	calibration, scorer = calibrated
	texts = [f"request {i}" for i in range(60)]
	report = serving_costs(
		calibration=calibration, scorer=scorer, texts=texts, encoder=FakeEncoder(), task="gsm8k"
	)
	units = report["units"]
	assert units["embedding_per_request"]["n"] == 60
	assert units["mmd_per_completed_window"]["n"] == 6
	assert units["admit_per_scoring_trigger_request"]["n"] == 6
	assert units["admit_per_other_request"]["n"] == 54
	assert units["noise_per_request"]["p95_ms"] >= units["noise_per_request"]["p50_ms"] >= 0
	assert report["peak_policy_memory_bytes"] > 0
	plain = serving_costs(calibration=calibration, scorer=scorer, texts=texts)
	assert plain["units"]["embedding_per_request"] is None
	with pytest.raises(ValueError, match="at least one window"):
		serving_costs(calibration=calibration, scorer=scorer, texts=texts[:5])
