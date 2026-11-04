import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from harness.calibration import BenignRequest, Calibration, HarnessConfig, binomial_upper, calibrate
from harness.mmd import Array, WindowScorer
from harness.path_calibration import (
	PathCalibration,
	PathScoring,
	PathState,
	audit_upper_bound,
	fit_path_calibration,
	period_alpha,
)
from harness.path_runtime import PathRecipe, RuntimePathCalibration, detector_hash
from harness.policy import PathStopPolicy, StalePathAdmission, request_digest
from harness.store import SQLiteStore

SCORING = PathScoring(name="test-v1", streams=("s",))


def _fit(maxima: Sequence[float], **options: Any) -> PathCalibration:
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
	bad_paths: list[list[list[float]]] = [[], [[]], [[float("nan"), 0]]]
	for path in bad_paths:
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


def test_artifact_edits_and_overflowing_scores_are_rejected() -> None:
	weighted = PathScoring(name="two-v1", streams=("a", "b"), weights=(1.0, 0.5))
	fitted = fit_path_calibration(
		[[[0, 0]]] * 99, scoring=weighted, alpha=0.01, horizon=1, stratum="a"
	)
	tampered = fitted.model_dump_json().replace("0.5", "2.0")
	with pytest.raises(ValueError, match="input hash"):
		PathCalibration.model_validate_json(tampered)
	with pytest.raises(ValueError, match="input hash"):
		PathCalibration(**{**fitted.model_dump(), "input_hash": "anything"})
	huge = PathScoring(name="huge-v1", streams=("s",), weights=(1e308,))
	with pytest.raises(ValueError, match="finite"):
		huge.look([2.0])


@pytest.fixture
def runtime_components() -> tuple[Calibration, WindowScorer, RuntimePathCalibration]:
	from test_harness_api import Encoder

	rows = [BenignRequest(account=f"a{i // 4}", text=f"q{i}") for i in range(48)]
	calibration, refs = calibrate(
		config=HarnessConfig(
			window_size=2, reference_windows=2, bandwidth_sample=8, horizon_attempts=20
		),
		encoder=Encoder(),
		fitting=rows[:16],
		score_calibration=rows[16:32],
		policy_calibration=rows[32:],
	)
	scorer = WindowScorer(refs, calibration.sigma)
	fitted = fit_path_calibration(
		[[[10.0]]] * 99,
		scoring=PathScoring(name="prompt-path-v1", streams=("mmd",)),
		alpha=0.01,
		horizon=2,
		stratum="all",
	)
	artifact = RuntimePathCalibration(
		recipe=PathRecipe(
			detector_hash=detector_hash(calibration, scorer),
			encoder={"model": "fake"},
			request_horizon=4,
			window_size=2,
		),
		calibration=fitted,
	)
	return calibration, scorer, artifact


def test_runtime_recipe_binding_and_forbidden_sum(
	runtime_components: tuple[Calibration, WindowScorer, RuntimePathCalibration],
) -> None:
	calibration, scorer, artifact = runtime_components
	assert RuntimePathCalibration.model_validate_json(artifact.model_dump_json()) == artifact
	with pytest.raises(ValueError, match="feature recipe"):
		RuntimePathCalibration.model_validate_json(
			artifact.model_dump_json().replace('"model":"fake"', '"model":"other"')
		)
	with pytest.raises(ValueError, match="different encoder"):
		artifact.require_compatible(calibration, scorer, {"model": "other"})
	for stream in ("coverage_sum_logdet", "request_count", "off_path"):
		fitted = fit_path_calibration(
			[[[1]]] * 99,
			scoring=PathScoring(name="prompt-path-v1", streams=(stream,)),
			alpha=0.01,
			horizon=2,
			stratum="all",
		)
		with pytest.raises(ValueError, match="unsupported"):
			RuntimePathCalibration(recipe=artifact.recipe, calibration=fitted)


def test_ordered_prefix_and_no_write_lock_during_features(
	runtime_components: tuple[Calibration, WindowScorer, RuntimePathCalibration], tmp_path: Path
) -> None:
	from concurrent.futures import ThreadPoolExecutor
	from threading import Barrier, Lock

	from test_harness_api import Encoder

	calibration, scorer, artifact = runtime_components
	barrier, guard = Barrier(2), Lock()
	calls = 0
	stores = [SQLiteStore(tmp_path / "order.db", "same") for _ in range(2)]

	class CheckedScorer(WindowScorer):
		def score(self, window: Array, keys: Sequence[str] | None = None) -> float:
			probe = SQLiteStore(tmp_path / "order.db", "same", timeout_seconds=0.1)
			probe.close()
			return super().score(window, keys)

	scorer = CheckedScorer(scorer.references, scorer.sigma, scorer.scales)

	class OrderedEncoder(Encoder):
		def encode(self, texts: list[str]) -> Array:
			nonlocal calls
			# A separate connection must acquire a write transaction during encoding.
			probe = SQLiteStore(tmp_path / "order.db", "same", timeout_seconds=0.1)
			probe.close()
			with guard:
				calls += 1
				wait = calls <= 2
			if wait:
				barrier.wait(timeout=5)
			return super().encode(texts)

	policies = [
		PathStopPolicy(calibration, scorer, artifact, encoder=OrderedEncoder(), store=s)
		for s in stores
	]
	with ThreadPoolExecutor(max_workers=2) as pool:
		futures = [pool.submit(p.admit, "a", f"prompt {i}", 0) for i, p in enumerate(policies)]
		decisions = [f.result(timeout=10) for f in futures]
	assert sorted(d.path["revision"] for d in decisions if d.path is not None) == [1, 2]
	state = policies[0].state("a")
	assert state is not None and state.path is not None
	ordered = sorted(
		zip(decisions, ["prompt 0", "prompt 1"], strict=True), key=lambda x: x[0].attempt
	)
	expected = Encoder().encode([text for _, text in ordered])
	assert np.array_equal(np.asarray(state.path.vectors), expected)
	assert state.path.look_count == 1 and calls == 3
	assert state.path.running_max == pytest.approx(
		scorer.score(expected, [request_digest(text) for _, text in ordered])
	)
	for store in stores:
		store.close()


def test_prepared_prefix_requires_compare_and_swap(
	runtime_components: tuple[Calibration, WindowScorer, RuntimePathCalibration],
) -> None:
	from test_harness_api import Encoder

	calibration, scorer, artifact = runtime_components
	policy = PathStopPolicy(calibration, scorer, artifact, encoder=Encoder())
	first, stale = policy.prepare("a", "first", 0), policy.prepare("a", "second", 0)
	with policy.store.transaction() as tx:
		policy.admit_in_transaction(tx, "a", "first", 0, prepared=first)
	with pytest.raises(StalePathAdmission), policy.store.transaction() as tx:
		policy.admit_in_transaction(tx, "a", "second", 0, prepared=stale)
	with (
		pytest.raises(ValueError, match="outside the transaction"),
		policy.store.transaction() as tx,
	):
		policy.admit_in_transaction(tx, "a", "second", 0)
	assert policy.admit("a", "second", 0).attempt == 2


def test_durable_stop_and_store_binding(
	runtime_components: tuple[Calibration, WindowScorer, RuntimePathCalibration], tmp_path: Path
) -> None:
	from test_harness_api import Encoder

	calibration, scorer, artifact = runtime_components
	fitted = fit_path_calibration(
		[[[-10]]] * 99, scoring=artifact.calibration.scoring, alpha=0.01, horizon=2, stratum="all"
	)
	artifact = RuntimePathCalibration(recipe=artifact.recipe, calibration=fitted)
	store = SQLiteStore(tmp_path / "state.db", "same")
	policy = PathStopPolicy(calibration, scorer, artifact, encoder=Encoder(), store=store)
	policy.admit("a", "one", 0)
	stopped = policy.admit("a", "two", 1)
	assert stopped.reason == "path_stop" and stopped.action == "refuse"
	store.close()
	store = SQLiteStore(tmp_path / "state.db", "same")
	policy = PathStopPolicy(calibration, scorer, artifact, encoder=Encoder(), store=store)
	later = policy.admit("a", "different context", calibration.config.horizon_seconds + 1)
	assert later.reason == "path_stop" and later.path is not None
	assert later.path["stop_look"] == 1 and later.path["qualified"] is False
	assert later.path["requests"] == 2 and later.attempt == 1
	with pytest.raises(ValueError, match="another path calibration"):
		PathStopPolicy(calibration, scorer, runtime_components[2], store=store)
	store.close()


@pytest.mark.parametrize(("n", "alpha"), [(98, 0.01), (998, 0.001)])
def test_runtime_resolution_abstention(
	runtime_components: tuple[Calibration, WindowScorer, RuntimePathCalibration],
	n: int,
	alpha: float,
) -> None:
	from test_harness_api import Encoder

	calibration, scorer, artifact = runtime_components
	fitted = fit_path_calibration(
		[[[-10]]] * n, scoring=artifact.calibration.scoring, alpha=alpha, horizon=2, stratum="all"
	)
	policy = PathStopPolicy(
		calibration,
		scorer,
		RuntimePathCalibration(recipe=artifact.recipe, calibration=fitted),
		encoder=Encoder(),
	)
	for i in range(6):
		decision = policy.admit("a", str(i), i * calibration.config.horizon_seconds)
		assert decision.action == "serve" and decision.path is not None
		assert decision.path["status"] == "abstain" and decision.path["abstention_reason"]
		assert decision.path["p_value"] is None
	assert decision.path is not None and decision.path["requests"] == 4


def test_path_cli_and_explicit_coverage_options(
	runtime_components: tuple[Calibration, WindowScorer, RuntimePathCalibration],
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	import json

	from harness import cli
	from harness.calibration import save_calibration
	from test_harness_api import Encoder
	from typer.testing import CliRunner

	calibration, scorer, _ = runtime_components
	directory = tmp_path / "calibration"
	save_calibration(directory, calibration, scorer.references)
	monkeypatch.setattr(cli, "make_encoder", lambda embeddings, device: Encoder())
	runner = CliRunner()
	fit = tmp_path / "feature-fit.jsonl"
	fit.write_text(
		"".join(
			BenignRequest(account=f"a{i // 4}", text=f"q{i}").model_dump_json() + "\n"
			for i in range(16, 32)
		)
	)
	evidence = tmp_path / "evidence.json"
	args = [
		"evidence-calibrate",
		"--calibration",
		str(directory),
		"--score-calibration",
		str(fit),
		"--output",
		str(evidence),
		"--coverage-seed",
		"7",
	]
	partial = runner.invoke(cli.app, args)
	assert partial.exit_code != 0 and "explicit seed, width and ridge" in partial.output
	result = runner.invoke(cli.app, args + ["--coverage-width", "4", "--coverage-ridge", "0.5"])
	assert result.exit_code == 0, result.output
	assert json.loads(evidence.read_text())["null"]["coverage"]["ridge"] == 0.5
	heldout = tmp_path / "benign-paths.jsonl"
	heldout.write_text(
		"".join(
			BenignRequest(account=f"path-{a}", text=f"path {a} {i}").model_dump_json() + "\n"
			for a in range(99)
			for i in range(4)
		)
	)
	report = tmp_path / "report.json"
	check_args = [
		"evidence-check",
		"--calibration",
		str(directory),
		"--evidence",
		str(evidence),
		"--requests",
		str(heldout),
		"--output",
		str(report),
	]
	result = runner.invoke(cli.app, check_args)
	assert result.exit_code == 0, result.output
	comparison = json.loads(report.read_text())["coverage_comparison"]
	assert comparison["bands"][0]["metrics"]["coverage_mean_logdet"]["abstained"] == 99
	result = runner.invoke(cli.app, check_args[:5] + ["--requests", str(fit)])
	assert result.exit_code != 0 and "held-out accounts and prompts" in result.output
	path = tmp_path / "path.json"
	args = [
		"path-calibrate",
		"--calibration",
		str(directory),
		"--campaigns",
		str(heldout),
		"--output",
		str(path),
		"--request-horizon",
		"4",
		"--stratum",
		"all",
	]
	result = runner.invoke(cli.app, args)
	assert result.exit_code == 0, result.output
	artifact = RuntimePathCalibration.model_validate_json(path.read_text())
	assert artifact.calibration.n == 99 and not artifact.calibration.abstained
	assert artifact.qualified is False and artifact.recipe.request_horizon == 4
	assert runner.invoke(cli.app, args).exit_code != 0


@pytest.mark.parametrize("stop", [False, True])
def test_stratum_switching_cannot_reset_evidence_or_stop(
	runtime_components: tuple[Calibration, WindowScorer, RuntimePathCalibration],
	stop: bool,
) -> None:
	from harness.serialization import gateway_turn
	from test_harness_api import Encoder

	calibration, scorer, artifact = runtime_components
	calibration = calibration.model_copy(deep=True)
	calibration.config.task = "gateway"
	recipe = artifact.recipe.model_copy(
		update={"detector_hash": detector_hash(calibration, scorer)}
	)
	fitted = fit_path_calibration(
		[[[(-10.0 if stop else 10.0)]]] * 99,
		scoring=artifact.calibration.scoring,
		alpha=0.01,
		horizon=2,
		stratum="single_domain",
	)
	policy = PathStopPolicy(
		calibration,
		scorer,
		RuntimePathCalibration(recipe=recipe, calibration=fitted),
		encoder=Encoder(),
	)
	for i in range(2):
		first = policy.admit("a", gateway_turn("ag_news", f"first {i}"), i)
	assert first.path is not None
	maximum = first.path["running_max"]
	for i in range(2):
		changed = policy.admit(
			"a", gateway_turn("gsm8k", f"switch {i}"), calibration.config.horizon_seconds * (i + 1)
		)
	assert changed.path is not None and changed.path["running_max"] == maximum
	assert changed.path["status"] == ("stop" if stop else "abstain")
	assert changed.reason == ("path_stop" if stop else None)
	assert changed.path["look_count"] == 1


def test_sparse_coverage_band_abstains_without_dropping_a_declared_stream(
	runtime_components: tuple[Calibration, WindowScorer, RuntimePathCalibration],
) -> None:
	from harness.logdet import CoverageSpec
	from harness.signals import fit_signals
	from test_harness_api import Encoder

	calibration, scorer, artifact = runtime_components
	null = fit_signals([], profiles=[], coverage=CoverageSpec(seed=4, input_dim=4))
	recipe = artifact.recipe.model_copy(update={"evidence": null})
	fitted = fit_path_calibration(
		[[[-10, -10]]] * 99,
		scoring=PathScoring(name="prompt-path-v1", streams=("mmd", "coverage_mean_logdet")),
		alpha=0.01,
		horizon=2,
		stratum="all",
	)
	policy = PathStopPolicy(
		calibration,
		scorer,
		RuntimePathCalibration(recipe=recipe, calibration=fitted),
		encoder=Encoder(),
	)
	policy.admit("a", "one", 0)
	decision = policy.admit("a", "two", 1)
	assert decision.action == "serve" and decision.path is not None
	assert decision.path["status"] == "abstain"
	assert "insufficient coverage null" in decision.path["abstention_reason"]
	assert decision.path["running_max"] is None and decision.path["look_count"] == 0
