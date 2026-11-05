"""Measured-grid quota selection, independent of experimental output rewriting."""

import json
from pathlib import Path
from typing import Any

import pytest
from distillation.cli import app
from distillation.quota_matching import match_quotas
from typer.testing import CliRunner


def development() -> dict[str, Any]:
	return {
		"split": "development",
		"population_id": "fixture-clients",
		"schedule_sha256": "fixture-schedule",
		"tolerance": 0.02,
		"candidate": {"arm": "path", "loss_by_stratum": {"casual": 0.1, "specialist": 0.2}},
		"settings": [
			{
				"policy": "quota",
				"request_quota": 4,
				"loss_by_stratum": {"casual": 0.1, "specialist": 0.2},
			},
			{
				"policy": "quota_tokens",
				"request_quota": 4,
				"output_token_budget": 200,
				"loss_by_stratum": {"casual": 0.1, "specialist": 0.2},
			},
		],
	}


def test_matching_requires_one_observed_setting_for_every_stratum(tmp_path: Path) -> None:
	dev = development()
	dev["settings"][0]["loss_by_stratum"]["specialist"] = 0.3
	dev["settings"].append(
		{
			"policy": "quota",
			"request_quota": 7,
			"loss_by_stratum": {"casual": 0.2, "specialist": 0.2},
		}
	)
	result = match_quotas(dev)
	assert result["baselines"]["quota"]["status"] == "not matched"
	assert result["baselines"]["quota"]["selected"] is None
	assert result["baselines"]["quota_tokens"]["status"] == "matched"
	assert not result["interpolated"]
	assert (
		result["baselines"]["quota"]["trials"][0]["strata"]["specialist"]["status"] == "not matched"
	)
	source, output = tmp_path / "development.json", tmp_path / "matches.json"
	source.write_text(json.dumps(dev))
	args = ["baseline", "match-quotas", "--development", str(source), "--output", str(output)]
	runner = CliRunner()
	assert runner.invoke(app, args).exit_code == 0
	assert json.loads(output.read_text()) == result
	assert runner.invoke(app, args).exit_code == 2
	dev["split"] = "test"
	with pytest.raises(ValueError, match="development"):
		match_quotas(dev)


def test_matching_decimal_tolerance_boundary() -> None:
	dev = development()
	dev["settings"][0]["loss_by_stratum"]["casual"] = 0.08
	assert match_quotas(dev)["baselines"]["quota"]["status"] == "matched"
