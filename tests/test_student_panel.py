"""Guard the published score snapshot against partial and tampered benchmark results."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def paper_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))


def test_paired_comparisons_keep_each_model_on_the_same_items() -> None:
	module = runpy.run_module("paper.figures.student_panel")
	cells = {"base": {"correctness": "0101"}, "student": {"correctness": "1101"}}
	metrics, pairs = module["summarize"](cells, ["a", "b", "c", "d"], resamples=100)
	assert metrics["base"]["correct"] == 2
	assert metrics["student"]["correct"] == 3
	pair = pairs["base minus student"]
	assert pair["a_only"] == 0 and pair["b_only"] == 1
	assert pair["difference"] == -0.25
	assert pair["mcnemar_exact_p"] <= pair["mcnemar_holm_p"] <= 1


@pytest.mark.parametrize("bits", ["01", "0102", "01x1", "01010", [0, 1, 0, 1]])
def test_partial_and_nonbinary_results_fail_closed(bits: object) -> None:
	module = runpy.run_module("paper.figures.student_panel")
	with pytest.raises(ValueError, match="invalid or incomplete"):
		module["summarize"]({"base": {"correctness": bits}}, ["a", "b", "c", "d"])


def test_published_statistics_cannot_be_changed_without_detection(tmp_path: Path) -> None:
	module = runpy.run_module("paper.figures.student_panel")
	cells: dict[str, dict[str, object]] = {
		"base": {"correctness": "0101"},
		"student": {"correctness": "1101"},
	}
	metrics, pairs = module["summarize"](cells, ["a", "b", "c", "d"])
	for name, values in metrics.items():
		cells[name].update(values)
	cells["student"]["accuracy"] = 1.0
	path = tmp_path / "tampered.json"
	path.write_text(json.dumps({"cells": cells, "pairs": pairs, "item_ids": ["a", "b", "c", "d"]}))
	result = subprocess.run(
		[sys.executable, "-m", "paper.figures.student_panel", "--verify", str(path)],
		capture_output=True,
		text=True,
		check=False,
	)
	assert result.returncode != 0
	assert "statistics do not match" in result.stderr
