"""Whole-campaign evidence monitor on synthetic embeddings, then observe-only API logging."""

import json
from typing import Any

import numpy as np
import pytest
from harness.campaign_evidence import (
	EvidenceCalibration,
	EvidenceSettings,
	calibrate,
	features,
	new_state,
	observe,
	update,
)

DIM = 16
SETTINGS = EvidenceSettings(look_every=20, rarefied_size=10, alpha=0.05, cells=8)


def _unit(rows: np.ndarray) -> np.ndarray:
	return np.asarray(rows / np.linalg.norm(rows, axis=1, keepdims=True))


def _specialist(rng: np.random.Generator, length: int) -> list[tuple[str, str, Any]]:
	"""A benign account: distinct questions around one topic."""
	center = rng.standard_normal(DIM)
	rows = _unit(center + 0.6 * rng.standard_normal((length, DIM)))
	return [(f"question {rng.integers(1 << 40)}", "gsm8k", row) for row in rows]


def _sweep(rng: np.random.Generator, length: int) -> list[tuple[str, str, Any]]:
	"""Broad extraction: distinct prompts spread over the whole space."""
	rows = _unit(rng.standard_normal((length, DIM)))
	return [(f"prompt {rng.integers(1 << 40)}", "gsm8k", row) for row in rows]


@pytest.fixture(scope="module")
def monitor() -> EvidenceCalibration:
	rng = np.random.default_rng(3)
	fitting = _unit(rng.standard_normal((400, DIM)))
	return calibrate(
		fitting=fitting,
		score_campaigns=[_specialist(rng, 60) for _ in range(60)],
		policy_campaigns=[_specialist(rng, 60) for _ in range(60)],
		settings=SETTINGS,
		encoder={"model": "fixture"},
	)


def _replay(monitor: EvidenceCalibration, campaign: list[tuple[str, str, Any]]) -> list[Any]:
	state = new_state()
	centroids = monitor.cell_centroids()
	looks = []
	for text, domain, vector in campaign:
		result = observe(
			state,
			text=text,
			domain=domain,
			embedding=vector,
			calibration=monitor,
			centroids=centroids,
		)
		if result is not None:
			looks.append(result)
	return looks


def test_broad_sweeps_alert_and_fresh_specialists_rarely_do(monitor: EvidenceCalibration) -> None:
	rng = np.random.default_rng(11)
	assert monitor.horizon_looks == {"single_domain": 3}
	sweeps = [_replay(monitor, _sweep(rng, 60)) for _ in range(20)]
	benign = [_replay(monitor, _specialist(rng, 60)) for _ in range(100)]
	assert sum(any(look["decision"] == "alert" for look in s) for s in sweeps) >= 18
	# Fresh benign accounts from the calibration generator: at most a few alerts at alpha .05.
	assert sum(any(look["decision"] == "alert" for look in b) for b in benign) <= 12
	assert all(look["qualified"] is False for look in sweeps[0])


def test_monitor_expires_past_the_benign_horizon_and_state_is_bounded(
	monitor: EvidenceCalibration,
) -> None:
	rng = np.random.default_rng(5)
	looks = _replay(monitor, _sweep(rng, 100))
	assert [look["decision"] for look in looks][3] == "expired" and len(looks) == 4
	state = new_state()
	centroids = monitor.cell_centroids()
	for text, domain, vector in _sweep(rng, 300):
		update(
			state,
			text=text,
			domain=domain,
			embedding=vector,
			settings=SETTINGS,
			coverage=monitor.coverage,
			centroids=centroids,
		)
	assert len(state["reservoir"]) == 128 and len(json.dumps(state)) < 200_000
	assert features(state, SETTINGS)["template_diversity"] == pytest.approx(1 / 300)


def test_repeats_add_no_coverage(monitor: EvidenceCalibration) -> None:
	rng = np.random.default_rng(9)
	campaign = _sweep(rng, 10)
	state = new_state()
	for text, domain, vector in campaign + campaign:
		update(
			state,
			text=text,
			domain=domain,
			embedding=vector,
			settings=SETTINGS,
			coverage=monitor.coverage,
			centroids=monitor.cell_centroids(),
		)
	assert state["count"] == 20 and state["distinct"] == 10 and state["gain_count"] == 9


def test_offline_commands_calibrate_and_report_first_alerts(tmp_path: Any) -> None:
	import hashlib

	from harness.calibration import BenignRequest
	from harness.cli import app
	from typer.testing import CliRunner

	rng = np.random.default_rng(21)
	texts: dict[str, np.ndarray] = {}

	def write(name: str, campaigns: list[list[tuple[str, str, Any]]]) -> str:
		path = tmp_path / f"{name}.jsonl"
		rows = []
		for index, campaign in enumerate(campaigns):
			for text, _, vector in campaign:
				texts[text] = vector
				rows.append(BenignRequest(account=f"{name}-{index}", text=text).model_dump_json())
		path.write_text("\n".join(rows) + "\n")
		return str(path)

	fitting = write("fit", [_sweep(rng, 200)])
	score = write("score", [_specialist(rng, 60) for _ in range(40)])
	policy = write("policy", [_specialist(rng, 60) for _ in range(40)])
	suspect = write("suspect", [_sweep(rng, 60), _specialist(rng, 60)])
	cache = tmp_path / "cache.npz"
	keys = list(texts)
	np.savez(
		cache,
		keys=np.array([hashlib.sha256(k.encode()).hexdigest() for k in keys]),
		vectors=np.stack([texts[k] for k in keys]),
	)
	runner = CliRunner()
	monitor = tmp_path / "monitor.json"
	common = ["--task", "text", "--embeddings", str(cache)]
	declared = ["--encoder-model", "fixture", "--encoder-revision", "r1"]
	fitted = runner.invoke(
		app,
		[
			"evidence-monitor-calibrate",
			"--fitting",
			fitting,
			"--score-calibration",
			score,
			"--policy-calibration",
			policy,
			"--output",
			str(monitor),
			"--look-every",
			"20",
			"--rarefied-size",
			"10",
			"--alpha",
			"0.05",
			"--cells",
			"8",
			*common,
			*declared,
		],
	)
	assert fitted.exit_code == 0, fitted.output
	assert json.loads(monitor.read_text())["encoder"]["model"] == "fixture"
	report = tmp_path / "report.json"
	checked = runner.invoke(
		app,
		[
			"evidence-monitor-check",
			"--monitor",
			str(monitor),
			"--requests",
			suspect,
			"--output",
			str(report),
			*common,
		],
	)
	assert checked.exit_code == 0, checked.output
	results = {r["account"]: r for r in json.loads(report.read_text())["results"]}
	assert results["suspect-0"]["alert_requests"] is not None
	overlap = runner.invoke(
		app,
		[
			"evidence-monitor-calibrate",
			"--fitting",
			fitting,
			"--score-calibration",
			score,
			"--policy-calibration",
			score,
			"--output",
			str(tmp_path / "bad.json"),
			*common,
			*declared,
		],
	)
	assert overlap.exit_code != 0
	undeclared = runner.invoke(
		app,
		[
			"evidence-monitor-calibrate",
			"--fitting",
			fitting,
			"--score-calibration",
			score,
			"--policy-calibration",
			policy,
			"--output",
			str(tmp_path / "undeclared.json"),
			*common,
		],
	)
	assert undeclared.exit_code != 0 and "--encoder-model" in undeclared.output


def test_leaving_the_first_profile_expires_without_alerting(monitor: EvidenceCalibration) -> None:
	rng = np.random.default_rng(13)
	campaign = _specialist(rng, 20) + [
		(text, "squad", vector) for text, _, vector in _sweep(rng, 40)
	]
	looks = _replay(monitor, campaign)
	assert looks[0]["profile"] == "single_domain"
	assert looks[1]["decision"] == "expired" and len(looks) == 2
