"""Whole-campaign rank validity and CPU-only calibration CLI fixtures."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from harness.campaign_coverage import (
	CoverageCalibration,
	CoverageConfig,
	CoverageMonitor,
	advance_coverage,
	coverage_evidence,
	fit_coverage_calibration,
)
from harness.cli import app
from harness.logdet import CoverageSpec
from harness.serialization import gateway_turn
from typer.testing import CliRunner


def monitor() -> CoverageMonitor:
	return CoverageMonitor(
		spec=CoverageSpec(seed=1, input_dim=2),
		encoder_identity={"model": "fixture"},
		serialization="raw_turn",
		domain="math",
		looks=(1, 2, 3),
		scales=(1, 2),
	)


def test_exchangeable_rank_with_ties_and_all_looks() -> None:
	rng = np.random.default_rng(20)
	paths = [rng.integers(-1, 2, (3, 2)).astype(float) for _ in range(30)]
	for path in paths:
		path[~path.any(axis=1)] = [1.0, 0.0]
	# Leave-one-out ranks on an exchangeable simulated null, including tied paths.
	paths += [paths[0].copy()] * 5
	stops = 0
	for i, path in enumerate(paths):
		cal = fit_coverage_calibration(monitor(), paths[:i] + paths[i + 1 :], 0.2)
		cfg = CoverageConfig(monitor=monitor(), calibration=cal, calibration_sha256=cal.digest)
		state: dict[str, Any] = {}
		alert = False
		for vector in path:
			state = advance_coverage(monitor(), state, vector, True)
			result = coverage_evidence(cfg, state, "math")
			alert |= result["alert"]
		stops += alert
		means = [0.0] * len(monitor().scales)
		expected = []
		for count, gain in enumerate(monitor().spec.gains(path), 1):
			means = [
				gain if count == 1 else old + (gain - old) / scale
				for old, scale in zip(means, monitor().scales, strict=True)
			]
			if count in monitor().looks:
				expected.extend(means)
		assert state["maximum"] == max(expected)
		assert result["p_value"] == (1 + sum(x >= state["maximum"] for x in cal.maxima)) / (
			len(cal.maxima) + 1
		)
	assert stops / len(paths) <= 0.2


def test_artifact_binding_resolution_and_complete_horizon() -> None:
	cal = fit_coverage_calibration(monitor(), [[[1.0, 0.0]] * 3], 0.01)
	cfg = CoverageConfig(monitor=monitor(), calibration=cal, calibration_sha256=cal.digest)
	state = advance_coverage(monitor(), {}, [1.0, 0.0], True)
	assert coverage_evidence(cfg, state, "math")["abstention_reason"] == "insufficient_campaigns"
	with pytest.raises(ValueError, match="mismatch"):
		CoverageConfig(monitor=monitor(), calibration=cal, calibration_sha256="bad")
	with pytest.raises(ValueError, match="horizon"):
		fit_coverage_calibration(monitor(), [[[1.0, 0.0]] * 2], 0.1)


def test_offline_cli_fixture_encoder(tmp_path: Path) -> None:
	cache = tmp_path / "embeddings.npz"
	texts = ["one", "two", "three"]
	np.savez(
		cache,
		keys=[hashlib.sha256(t.encode()).hexdigest() for t in texts],
		vectors=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
	)
	recipe = monitor().model_copy(
		update={
			"encoder_identity": {
				"model": "cached",
				"sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
			}
		}
	)
	monitor_file = tmp_path / "monitor.json"
	monitor_file.write_text(recipe.model_dump_json())
	requests = tmp_path / "requests.jsonl"
	requests.write_text("\n".join(json.dumps({"account": "benign", "text": t}) for t in texts))
	output = tmp_path / "cal.json"
	result = CliRunner().invoke(
		app,
		[
			"campaign-coverage-calibrate",
			"--monitor",
			str(monitor_file),
			"--campaigns",
			str(requests),
			"--output",
			str(output),
			"--embeddings",
			str(cache),
		],
	)
	assert result.exit_code == 0, result.output
	assert CoverageCalibration.model_validate_json(output.read_text()).monitor == recipe


def test_gateway_monitor_calibrates_on_the_input_serving_encodes(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	seen: list[str] = []

	class Fixture:
		def identity(self) -> dict[str, str]:
			return {"model": "fixture"}

		def input_complete(self, text: str) -> bool:
			return True

		def encode(self, texts: list[str]) -> np.ndarray:
			seen.extend(texts)
			return np.tile([1.0, 0.0], (len(texts), 1))

	monkeypatch.setattr("harness.cli.make_encoder", lambda *args: Fixture())
	recipe = monitor().model_copy(update={"serialization": "gateway_turn"})
	monitor_file = tmp_path / "monitor.json"
	monitor_file.write_text(recipe.model_dump_json())
	requests = tmp_path / "requests.jsonl"
	requests.write_text(
		"".join(
			json.dumps({"account": str(a), "text": "turn"}) + "\n" for a in range(9) for _ in "abc"
		)
	)
	result = CliRunner().invoke(
		app,
		[
			"campaign-coverage-calibrate",
			"--monitor",
			str(monitor_file),
			"--campaigns",
			str(requests),
			"--output",
			str(tmp_path / "coverage.json"),
			"--alpha",
			"0.1",
		],
	)
	assert result.exit_code == 0, result.output
	assert set(seen) == {gateway_turn("math", "turn")}
