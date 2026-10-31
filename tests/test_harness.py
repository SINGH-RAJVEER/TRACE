import hashlib
import math
import threading
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
	binomial_upper,
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
from harness.store import MemoryStore, Reservation, SQLiteStore

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
	# Zero of 20 benign horizons bounds the rate only near 14%, far above a 1% target.
	assert binomial_upper(0, 20) == pytest.approx(1 - 0.05 ** (1 / 20), abs=1e-6)
	assert binomial_upper(5, 5) == 1.0
	assert 0.01 < binomial_upper(10, 1000) < 0.02
	forced = nested_thresholds(
		[100, 4, 3, 2, 1], [0.4, 0.2, 0.01], [True, False, False, False, False]
	)
	assert forced.thresholds == [3, 4, 4]
	assert forced.achieved == [0.4, 0.2, 0.2]
	assert nested_thresholds([1, 2], [0.01], [True, True]).achieved == [1.0]
	assert forced.level_results is not None
	assert [row["target"] for row in forced.level_results] == [0.4, 0.2, 0.01]
	assert [row["reached"] for row in forced.level_results] == [2, 1, 1]
	assert [row["capped"] for row in forced.level_results] == [1, 1, 1]
	assert forced.level_results[0]["scored"] == 5


def test_level_resolution_and_legacy_evidence(calibrated: tuple[Calibration, WindowScorer]) -> None:
	calibration = calibrated[0].model_copy(deep=True)
	report = calibration.adequacy(accumulators=("paper",), levels=(1, 4))
	assert report["levels"]["L1"]["minimum_policy_horizons"] == 10
	assert report["levels"]["L4"]["minimum_policy_horizons"] == 100
	result = report["levels"]["L1"]["results"]["paper"]
	assert result["scored"] == 200 and result["censored"] == 0
	assert result["upper95"] >= result["achieved"]
	calibration.thresholds["paper"].level_results = None
	legacy = Calibration.model_validate_json(calibration.model_dump_json())
	assert (
		legacy.adequacy(accumulators=("paper",))["levels"]["L1"]["evidence_version"]
		== "legacy-inferred"
	)
	calibration.thresholds["paper"].level_results = []
	assert not calibration.adequacy(accumulators=("paper",), levels=(4,))["passed"]
	with pytest.raises(ValueError):
		HarnessConfig(level_targets=(0.1, 0.2, 0.02, 0.01))


def test_risk_verification_rejects_stale_and_capped_proofs(
	calibrated: tuple[Calibration, WindowScorer],
) -> None:
	calibration, scorer = calibrated
	calibration = calibration.model_copy(deep=True)
	calibration.config.horizon_attempts = 4
	horizon = calibration.config.horizon_seconds
	policy = MMDPolicy(calibration, scorer, difficulty=1, policy_version="verify-risk-v2")

	def set_level(level: int) -> None:
		with policy.store.transaction() as tx:
			state = tx.account("a")
			assert state is not None
			state.level = level
			tx.put_account("a", state)

	def challenge(text: str, now: float) -> tuple[str, int]:
		decision = policy.admit("a", text, now, embedding=np.zeros(DIMENSION))
		assert decision.action == "challenge" and decision.challenge is not None
		return decision.challenge.nonce, solve(decision.challenge)[0]

	policy.admit("a", "first", 0, embedding=np.zeros(DIMENSION))
	set_level(3)
	# A valid proof releases only its request and leaves the level and attempts unchanged.
	nonce, counter = challenge("second", 1)
	held = policy.redeem(nonce, "a", "second", counter, 1)
	assert held is not None and held.level == 3 and not held.deceive
	state = policy.state("a")
	assert state is not None and (state.level, state.attempts) == (3, 2)
	# A stronger current level rejects a proof issued at a weaker one.
	nonce, counter = challenge("third", 2)
	set_level(4)
	assert policy.redeem(nonce, "a", "third", counter, 2) is None
	# L4 still offers a proof rather than refusing, but an exhausted cap rejects it.
	nonce, counter = challenge("fourth", 3)
	assert policy.admit("a", "fifth", 4, embedding=np.zeros(DIMENSION)).action == "refuse"
	assert policy.redeem(nonce, "a", "fourth", counter, 4) is None
	# An expired horizon rejects an unexpired proof before a new admission resets the state.
	policy = MMDPolicy(
		calibration,
		scorer,
		difficulty=1,
		challenge_ttl=2 * horizon,
		policy_version="verify-risk-v2",
	)
	policy.admit("a", "first", 0, embedding=np.zeros(DIMENSION))
	set_level(3)
	nonce, counter = challenge("second", 1)
	assert policy.redeem(nonce, "a", "second", counter, horizon) is None


def test_calibration_adequacy_rejects_small_pools_and_overlapping_window_inflation(
	calibrated: tuple[Calibration, WindowScorer],
) -> None:
	calibration = calibrated[0].model_copy(deep=True)
	calibration.counts.update(
		score_calibration_independent_windows=5,
		score_calibration_windows=1000,
		policy_calibration_scored_horizons=2,
	)
	calibration.calibration_scores = [0.0] * 1000
	calibration.thresholds["paper"].thresholds = [1.0] * 4
	calibration.thresholds["paper"].units = 2
	calibration.thresholds["paper"].level_results = None
	report = calibration.adequacy(accumulators=("paper",))
	assert report["passed"] is False
	assert report["minimum_score_windows"] == 99
	assert report["minimum_policy_horizons"] == 100
	assert report["nonoverlapping_score_windows"] == 5
	assert report["empirical_p_value_floor"] == pytest.approx(1 / 1001)
	assert len(report["reasons"]) == 3


class ClusteredEncoder:
	"""Each benign account asks about one of eight topics, as a specialist does."""

	def encode(self, texts: list[str]) -> Array:
		centers = np.random.default_rng(0).normal(size=(8, DIMENSION)) * 3
		rows = []
		for text in texts:
			seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
			vector = centers[int(text.split()[1]) % 8] + np.random.default_rng(seed).normal(
				size=DIMENSION
			)
			rows.append(vector / np.linalg.norm(vector))
		return np.array(rows)

	def identity(self) -> dict[str, Any]:
		return {"model": "clustered"}


def test_mixture_control_warns_when_broad_traffic_scores_below_benign_accounts(
	calibrated: tuple[Calibration, WindowScorer],
) -> None:
	# Exchangeable benign accounts: a pooled mixture ranks like any account, so no warning.
	control = calibrated[0].mixture_control
	assert control is not None and control.share_below_null_median < 0.9
	assert calibrated[0].adequacy(accumulators=("paper",))["detectability_warnings"] == []
	# Topic-concentrated accounts: every pooled window sits below the account null.
	specialists, _ = calibrate(
		config=HarnessConfig(window_size=10, reference_windows=3, bandwidth_sample=60),
		encoder=ClusteredEncoder(),
		fitting=_benign("fit", 16, 30),
		score_calibration=_benign("score", 40, 50),
		policy_calibration=_benign("policy", 200, 20),
	)
	assert specialists.mixture_control is not None
	assert specialists.mixture_control.share_below_null_median >= 0.9
	report = specialists.adequacy(accumulators=("paper",))
	assert len(report["detectability_warnings"]) == 1
	# A warning describes what the study can show; it never changes the gate's verdict.
	unchecked = specialists.model_copy(update={"mixture_control": None})
	plain = unchecked.adequacy(accumulators=("paper",))
	assert (report["passed"], report["status"]) == (plain["passed"], plain["status"])
	assert specialists.adequacy()["detectability_warnings"] == []


def test_calibrations_without_a_mixture_control_load_and_report_no_warning(
	calibrated: tuple[Calibration, WindowScorer],
) -> None:
	raw = calibrated[0].model_dump(mode="json")
	raw.pop("mixture_control")
	legacy = Calibration.model_validate(raw)
	assert legacy.mixture_control is None
	assert legacy.adequacy(accumulators=("paper",))["detectability_warnings"] == []


def test_required_mixture_control_is_recorded_and_fails_closed(
	calibrated: tuple[Calibration, WindowScorer],
) -> None:
	artifact = calibrated[0].model_copy(deep=True)
	artifact.config.mixture_control_required = True
	assert artifact.mixture_control is not None
	artifact.mixture_control.share_below_null_median = 0.9
	roundtrip = Calibration.model_validate(artifact.model_dump(mode="json"))
	report = roundtrip.adequacy(accumulators=("paper",))
	assert report["mixture_control_required"] is True
	assert not report["passed"]
	assert any("required mixture control failed" in reason for reason in report["reasons"])
	roundtrip.mixture_control = None
	assert (
		"required mixture control is missing or has no windows"
		in roundtrip.adequacy(accumulators=("paper",), allow_resolution_limited=True)["reasons"]
	)


def test_calibration_adequacy_checks_selected_defenses_and_legacy_counts(
	calibrated: tuple[Calibration, WindowScorer],
) -> None:
	calibration = calibrated[0].model_copy(deep=True)
	calibration.counts["policy_calibration_scored_horizons"] = 100
	calibration.counts["score_calibration_independent_windows"] = 99
	calibration.thresholds["paper"].thresholds = [1.0, 2.0, 3.0, 4.0]
	calibration.thresholds["paper"].units = 100
	calibration.thresholds["paper"].level_results = None
	assert calibration.adequacy(accumulators=("paper",))["passed"] is True
	calibration.thresholds["instantaneous"].thresholds = [1.0] * 4
	assert calibration.adequacy(accumulators=("instantaneous",))["passed"] is False
	calibration.counts.pop("policy_calibration_scored_horizons")
	assert calibration.adequacy(accumulators=("paper",))["passed"] is False
	# Quota shares the scored denominator but does not need an MMD null.
	calibration.counts["policy_calibration_scored_horizons"] = 200
	calibration.counts["score_calibration_independent_windows"] = 0
	assert calibration.adequacy(accumulators=(), baselines=("quota",))["passed"] is True
	calibration.baselines["quota"].units = 12
	assert calibration.adequacy(accumulators=(), baselines=("quota",))["passed"] is False


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
	assert calibration.counts["score_calibration_independent_windows"] == 100
	assert calibration.counts["score_calibration_windows"] > 100
	assert calibration.counts["policy_calibration_horizons"] == 200
	for thresholds in calibration.thresholds.values():
		assert thresholds.thresholds == sorted(thresholds.thresholds)
		for achieved, target in zip(thresholds.achieved, (0.10, 0.05, 0.02, 0.01), strict=True):
			assert achieved <= target
		assert all(
			upper >= achieved
			for upper, achieved in zip(thresholds.upper95, thresholds.achieved, strict=True)
		)
	assert calibration.baselines["quota"].thresholds == [40]
	assert "rate_per_minute" in calibration.baselines
	save_calibration(tmp_path, calibration, scorer.references)
	loaded, loaded_scorer = load_calibration(tmp_path)
	assert loaded.thresholds == calibration.thresholds
	window = FakeEncoder().encode([f"probe {i}" for i in range(10)])
	assert loaded_scorer.score(window) == pytest.approx(scorer.score(window))
	assert loaded.config.distinct_prompts is True
	# Artifacts from before distinct-prompt scoring keep scoring every request.
	import json

	raw = json.loads((tmp_path / "calibration.json").read_text())
	del raw["config"]["distinct_prompts"]
	(tmp_path / "calibration.json").write_text(json.dumps(raw))
	assert load_calibration(tmp_path)[0].config.distinct_prompts is False
	np.save(tmp_path / "references.npy", scorer.references + 1)
	with pytest.raises(ValueError, match="reference windows differ"):
		load_calibration(tmp_path)


def test_thresholds_and_baselines_fit_the_same_scored_horizons() -> None:
	calibration, _ = calibrate(
		config=HarnessConfig(window_size=10, reference_windows=3, bandwidth_sample=60),
		encoder=FakeEncoder(),
		fitting=_benign("fit", 4, 30),
		score_calibration=_benign("score", 20, 50),
		policy_calibration=_benign("policy", 60, 40) + _benign("short", 140, 5),
	)
	assert calibration.counts["policy_calibration_scored_horizons"] == 60
	assert calibration.counts["policy_calibration_unscored_horizons"] == 140
	assert {thresholds.units for thresholds in calibration.thresholds.values()} == {60}
	assert calibration.baselines["quota"].units == 60
	assert calibration.adequacy(baselines=("quota",))["reasons"] == [
		"quota: need 100 baseline policy horizons"
	]


def test_repeated_prompts_are_scored_once_per_window() -> None:
	rng = np.random.default_rng(1)
	references = np.stack([rng.normal(size=(6, 3)) for _ in range(2)])
	scorer = WindowScorer(references, 1.0)
	distinct_rows = rng.normal(size=(2, 3))
	window = distinct_rows[[0, 0, 0, 1, 1, 1]]
	keys = ["a", "a", "a", "b", "b", "b"]
	expected = np.mean([mmd2_unbiased(distinct_rows, r, 1.0) for r in references])
	assert scorer.score(window, keys) == pytest.approx(expected)
	# Without keys the identical points inflate the within-window term.
	assert scorer.score(window) > scorer.score(window, keys)
	assert np.isfinite(scorer.score(window[[0] * 6], ["a"] * 6))


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
	assert summary["reached_rate_scored"]["L1"] < 0.5
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
	nonce = challenge.challenge.nonce  # type: ignore[union-attr]
	assert policy.redeem(nonce, "a", "different text", counter, now=100.0) is None
	assert policy.redeem(nonce, "other", "q 10", counter, now=100.0) is None
	held = policy.redeem(nonce, "a", "q 10", counter, now=100.0)
	assert held is not None and held.release_at == challenge.release_at == 102.0
	assert (held.attempt, held.horizon) == (challenge.attempt, challenge.horizon) == (11, 100.0)
	assert policy.redeem(nonce, "a", "q 10", counter, now=100.0) is None
	late = decisions[11]
	late_nonce = late.challenge.nonce  # type: ignore[union-attr]
	assert policy.redeem(late_nonce, "a", "q 11", solve(late.challenge)[0], now=1000.0) is None  # type: ignore[arg-type]
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
	state = policy.state("a")
	assert state is not None and (state.windows, state.attempts) == (1, 31)
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
	noise = units["noise_per_request"]
	assert noise["p99_ms"] >= noise["p95_ms"] >= noise["p90_ms"] >= noise["p50_ms"] >= 0
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
		policy_calibration=_benign("policy", 2, 12) + _benign("short", 2, 1),
	)
	assert calibration.horizon_policy == HORIZON_POLICY
	assert calibration.counts["score_calibration_windows"] == 8
	assert calibration.counts["score_calibration_independent_windows"] == 6
	assert calibration.counts["policy_calibration_horizons"] == 6
	assert calibration.counts["policy_calibration_scored_horizons"] == 4
	assert calibration.counts["policy_calibration_unscored_horizons"] == 2
	assert calibration.counts["policy_calibration_capped_horizons"] == 2
	# Demand above the cap stays in the original horizon rather than receiving a new quota.
	assert calibration.baselines["quota"].thresholds == [2]
	assert calibration.baselines["quota"].achieved == [0.5]
	assert calibration.baselines["quota"].units == 4
	assert calibration.baselines["quota"].upper95 == [binomial_upper(2, 4)]
	summary = replay(
		calibration=calibration,
		scorer=WindowScorer(references, calibration.sigma),
		requests=_benign("policy", 2, 12) + _benign("short", 2, 1),
		output=tmp_path / "replay.jsonl",
		policy="quota",
	)
	assert summary["horizons"] == 6
	assert summary["reached"]["L4"] == 2
	assert summary["horizon_policy"] == HORIZON_POLICY
	# The cap imposes a refusal floor independent of the fitted statistical threshold.
	assert summary["reached_rate"]["L4"] == pytest.approx(2 / 6)
	assert summary["reached_rate_scored"]["L4"] == 0.5


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


def _forced(calibration: Calibration, levels: list[float]) -> Calibration:
	forced = calibration.model_copy(deep=True)
	forced.config.horizon_seconds = 100
	forced.config.horizon_attempts = 12
	forced.thresholds["paper"].thresholds = levels
	return forced


def test_durable_state_resumes_the_horizon_window_and_cap_after_restart(
	calibrated: tuple[Calibration, WindowScorer], tmp_path: Path
) -> None:
	calibration, scorer = calibrated
	forced = _forced(calibration, [1e9] * 4)
	texts = [f"q {i}" for i in range(13)]
	uninterrupted = MMDPolicy(forced, scorer, encoder=FakeEncoder())
	expected = [uninterrupted.admit("a", text, 0.0) for text in texts]
	path = tmp_path / "state.sqlite3"

	def restarted() -> MMDPolicy:
		return MMDPolicy(forced, scorer, encoder=FakeEncoder(), store=SQLiteStore(path, "arm"))

	# Restart halfway through the first window and again after the cap is reached.
	before = [restarted().admit("a", text, 0.0) for text in texts[:5]]
	after = restarted()
	resumed = before + [after.admit("a", text, 0.0) for text in texts[5:]]
	assert [d.attempt for d in resumed] == list(range(1, 14))
	assert resumed[9].window_completed and resumed[9].p_value == expected[9].p_value
	assert (
		[d.action for d in resumed] == [d.action for d in expected] == ["serve"] * 12 + ["refuse"]
	)
	assert restarted().admit("a", "spam", 99.999).action == "refuse"
	fresh = restarted().admit("a", "fresh", 100.0)
	assert (fresh.attempt, fresh.action, fresh.horizon) == (1, "serve", 100.0)
	with pytest.raises(ValueError, match="another policy or calibration"):
		SQLiteStore(path, "another arm")


def test_challenges_survive_restart_and_redeem_once(
	calibrated: tuple[Calibration, WindowScorer], tmp_path: Path
) -> None:
	calibration, scorer = calibrated
	forced = _forced(calibration, [-1, -1, -1, 1e9])
	path = tmp_path / "state.sqlite3"
	first = MMDPolicy(forced, scorer, encoder=FakeEncoder(), store=SQLiteStore(path, "arm"))
	decisions = [first.admit("a", f"q {i}", 0.0) for i in range(10)]
	assert decisions[9].action == "challenge" and decisions[9].challenge is not None
	counter, _ = solve(decisions[9].challenge)
	nonce = decisions[9].challenge.nonce
	# Two processes sharing the file after a restart: the proof releases once in total.
	second = MMDPolicy(forced, scorer, encoder=FakeEncoder(), store=SQLiteStore(path, "arm"))
	third = MMDPolicy(forced, scorer, encoder=FakeEncoder(), store=SQLiteStore(path, "arm"))
	held = second.redeem(nonce, "a", "q 9", counter, 1.0)
	assert held is not None and (held.attempt, held.level, held.horizon) == (10, 3, 0.0)
	assert third.redeem(nonce, "a", "q 9", counter, 1.0) is None
	assert second.redeem(nonce, "a", "q 9", counter, 1.0) is None
	# The level persisted, so the next request after the restart is challenged too.
	assert third.admit("a", "q 10", 2.0).action == "challenge"


def test_concurrent_connections_admit_exactly_the_cap(tmp_path: Path) -> None:
	path = tmp_path / "state.sqlite3"
	workers = 4
	policies = [QuotaPolicy(5, 100, 1000, store=SQLiteStore(path, "arm")) for _ in range(workers)]
	start = threading.Barrier(workers)
	results: list[list[tuple[int, str]]] = [[] for _ in range(workers)]

	def run(worker: int) -> None:
		start.wait()
		for _ in range(10):
			decision = policies[worker].admit("a", "q", 1.0)
			results[worker].append((decision.attempt, decision.action))

	threads = [threading.Thread(target=run, args=(w,)) for w in range(workers)]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()
	decisions = sorted(d for worker in results for d in worker)
	assert [attempt for attempt, _ in decisions] == list(range(1, 41))
	assert [action for _, action in decisions] == ["serve"] * 5 + ["refuse"] * 35


@pytest.mark.parametrize("durable", [False, True])
def test_release_ledger_settles_once_and_keeps_unsettled_reservations(
	durable: bool, tmp_path: Path
) -> None:
	path = tmp_path / "state.sqlite3"

	def open_store() -> MemoryStore | SQLiteStore:
		return SQLiteStore(path, "arm") if durable else memory

	memory = MemoryStore()
	served, crashed = (
		Reservation(account="a", horizon_start=0.0, attempt=n, request_digest="d", reserved_at=0)
		for n in (1, 2)
	)
	store = open_store()
	with store.transaction() as tx:
		assert tx.reserve(served) and not tx.reserve(served)
		assert tx.settle(served.key, "released", 1.0, 7)
		assert not tx.settle(served.key, "cancelled", 2.0)
		# The process dies after reserving this one, possibly after emitting its output.
		assert tx.reserve(crashed)
	with open_store().transaction() as tx:
		usage = tx.usage("a", 0.0)
		assert (usage.released, usage.reserved, usage.acquired, usage.output_tokens) == (1, 1, 2, 7)
		assert not tx.reserve(crashed)
		assert tx.usage("a", 100.0).acquired == 0
