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
	nested_thresholds,
	peak_per_minute,
	pooled_accounts,
	save_calibration,
)
from harness.mmd import Array, WindowScorer, kernel, mmd2_unbiased
from harness.policy import AllowPolicy, MMDPolicy, Policy, QuotaPolicy, RatePolicy
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
	forced = nested_thresholds(
		[100, 4, 3, 2, 1], [0.4, 0.2, 0.01], [True, False, False, False, False]
	)
	assert forced.thresholds == [3, 4, 4]
	assert forced.achieved == [0.4, 0.2, 0.2]
	assert nested_thresholds([1, 2], [0.01], [True, True]).achieved == [1.0]


def test_pooled_accounts_break_up_source_ordered_groups() -> None:
	# Five articles of 100 consecutive questions each, as SQuAD pools are written.
	rows = [(f"q{index}", f"article-{index // 100}") for index in range(500)]
	accounts = pooled_accounts(rows, "pool", 100, seed=42)
	assert accounts == pooled_accounts(rows, "pool", 100, seed=42)
	assert sorted(request.text for request in accounts) == sorted(text for _, text in rows)
	first = [request.text for request in accounts if request.account == "pool-00000"]
	assert len(first) == 100 and len(set(first)) == 5


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
	# Spacing counts from the last L0 admission, so the first throttled request also waits.
	assert [d.release_at for d in decisions[9:12]] == [101.0, 102.0, 103.0]
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
	forced.thresholds["paper"].thresholds = [-1, -1, -1, 1e9]
	direct = MMDPolicy(forced, scorer, encoder=FakeEncoder(), actions="direct")
	below = [direct.admit("c", f"q {i}", now=0.0) for i in range(12)]
	assert {(d.level, d.action, d.release_at) for d in below} == {(0, "serve", 0.0)}
	forced.thresholds["paper"].thresholds = [-1, -1, -1, -1]
	direct = MMDPolicy(forced, scorer, encoder=FakeEncoder(), actions="direct")
	above = [direct.admit("d", f"q {i}", now=0.0) for i in range(10)]
	assert above[9].level == 4 and above[9].action == "refuse"


def test_coding_policy_refuses_deception(calibrated: tuple[Calibration, WindowScorer]) -> None:
	calibration, scorer = calibrated
	calibration = calibration.model_copy(deep=True)
	calibration.config.task = "coding"
	with pytest.raises(ValueError, match="coding deception"):
		MMDPolicy(calibration, scorer, l4_action="deceive", deception_key=b"fixture")


def test_quota_and_rate_baselines_block_for_the_horizon() -> None:
	quota = QuotaPolicy(3, horizon_seconds=100, horizon_attempts=1000)
	assert [quota.admit("a", "q", float(t)).action for t in range(5)][-2:] == ["refuse"] * 2
	assert quota.admit("a", "q", 100.0).action == "serve"
	rate = RatePolicy(2, horizon_seconds=1000, horizon_attempts=1000)
	actions = [rate.admit("a", "q", t).action for t in (0, 1, 2, 100)]
	assert actions == ["serve", "serve", "refuse", "refuse"]


@pytest.mark.parametrize("name", ["quota", "rate", "mmd"])
def test_refused_attempts_cannot_clear_a_block(
	name: str, calibrated: tuple[Calibration, WindowScorer]
) -> None:
	calibration, scorer = calibrated
	forced = calibration.model_copy(deep=True)
	forced.config.horizon_seconds = 100
	forced.config.horizon_attempts = 12
	forced.thresholds["paper"].thresholds = [-1] * 4
	policy: Policy
	if name == "quota":
		policy = QuotaPolicy(1, 100, 12)
	elif name == "rate":
		policy = RatePolicy(1, 100, 12)
	else:
		policy = MMDPolicy(forced, scorer, encoder=FakeEncoder())
	decisions = [policy.admit("a", f"q{i}", 7.0) for i in range(40)]
	assert all(d.action == "refuse" and d.level == 4 for d in decisions[9:])
	assert [d.attempt for d in decisions] == list(range(1, 41))
	assert policy.admit("a", "still blocked", 106.999).action == "refuse"
	fresh = policy.admit("a", "fresh", 107.0)
	assert (fresh.attempt, fresh.level, fresh.action) == (1, 0, "serve")


@pytest.mark.parametrize("deceive", [False, True])
def test_mmd_attempt_cap_refuses_without_reset_even_for_deception(
	deceive: bool, calibrated: tuple[Calibration, WindowScorer]
) -> None:
	calibration, scorer = calibrated
	forced = calibration.model_copy(deep=True)
	forced.config.horizon_attempts = 10
	forced.thresholds["paper"].thresholds = [1e9] * 4
	policy = MMDPolicy(
		forced,
		scorer,
		encoder=FakeEncoder(),
		l4_action="deceive" if deceive else "refuse",
		deception_key=b"fixture",
	)
	decisions = [policy.admit("a", f"q{i}", 0) for i in range(31)]
	assert all(d.action == "serve" for d in decisions[:10])
	assert all(d.action == "refuse" and d.level == 4 for d in decisions[10:])
	assert policy.accounts["a"].windows == 1
	assert policy.accounts["a"].attempts == 31
	# The undefended arm must remain undefended at the same count.
	allow = AllowPolicy(86400, 10)
	assert all(allow.admit("a", "q", 0).action == "serve" for _ in range(31))


def test_full_pool_quota_does_not_renew_after_refusal_spam() -> None:
	policy = QuotaPolicy(500, 86400, 10000)
	served = sum(policy.admit("a", "q", 0).action == "serve" for _ in range(53893))
	assert served == 500


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


def test_calibration_and_replay_share_time_horizons_and_attempt_caps(tmp_path: Path) -> None:
	from harness.calibration import HORIZON_POLICY

	calibration, references = calibrate(
		config=HarnessConfig(
			window_size=2,
			reference_windows=2,
			bandwidth_sample=8,
			horizon_seconds=10,
			horizon_attempts=4,
			aggregate_enabled=False,
		),
		encoder=FakeEncoder(),
		fitting=_benign("fit", 4, 6),
		score_calibration=_benign("score", 2, 12),
		policy_calibration=_benign("policy", 2, 12),
	)
	assert calibration.horizon_policy == HORIZON_POLICY
	assert calibration.counts["score_calibration_windows"] == 6
	assert calibration.counts["policy_calibration_horizons"] == 4
	assert calibration.counts["policy_calibration_capped_horizons"] == 2
	# Demand above the cap stays in the original horizon rather than receiving a new quota.
	assert calibration.baselines["quota"].thresholds == [2]
	assert calibration.baselines["quota"].achieved == [0.5]
	summary = replay(
		calibration=calibration,
		scorer=WindowScorer(references, calibration.sigma),
		requests=_benign("policy", 2, 12),
		output=tmp_path / "replay.jsonl",
		policy="quota",
	)
	assert summary["horizons"] == 4
	assert summary["reached"]["L4"] == 2
	assert summary["horizon_policy"] == HORIZON_POLICY
	# The cap imposes a refusal floor independent of the fitted statistical threshold.
	assert summary["reached_rate"]["L4"] == 0.5


@pytest.mark.parametrize("version", [None, "count-reset-v0"])
def test_legacy_horizon_calibration_requires_regeneration(
	version: str | None, calibrated: tuple[Calibration, WindowScorer], tmp_path: Path
) -> None:
	import json

	from harness.replay import make_policy

	calibration, scorer = calibrated
	legacy = calibration.model_copy(update={"horizon_policy": version})
	with pytest.raises(ValueError, match="recalibrate in a new directory"):
		MMDPolicy(legacy, scorer)
	with pytest.raises(ValueError, match="recalibrate in a new directory"):
		make_policy("quota", legacy, scorer)
	save_calibration(tmp_path, legacy, scorer.references)
	path = tmp_path / "calibration.json"
	if version is None:
		payload = json.loads(path.read_text())
		del payload["horizon_policy"]
		path.write_text(json.dumps(payload))
	with pytest.raises(ValueError, match="recalibrate in a new directory"):
		load_calibration(tmp_path)
