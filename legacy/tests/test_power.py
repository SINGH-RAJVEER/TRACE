"""Suite resolution is reported before any model runs."""

from collections import Counter
from pathlib import Path

import pytest
from distillation.benchmarks.models import save_json
from distillation.power import design_effect, half_width_pp, minimum_detectable_pp, study_power
from distillation.tasks import _benchmark_item, _suite, gsm8k_example


def test_minimum_detectable_effect_shrinks_with_suite_size() -> None:
	assert minimum_detectable_pp(163) > 9 > minimum_detectable_pp(539) > 5
	assert minimum_detectable_pp(7600) < 2 < minimum_detectable_pp(1319)
	assert half_width_pp(163) == pytest.approx(7.68, abs=0.01)
	assert design_effect(Counter({"a": 3, "b": 1})) == pytest.approx(2.5)
	with pytest.raises(ValueError):
		minimum_detectable_pp(0)


def test_study_power_reads_frozen_suites(tmp_path: Path) -> None:
	items = [
		_benchmark_item(
			"gsm8k", "test", gsm8k_example(i, {"question": f"Q{i}", "answer": "1 #### 1"})
		)
		for i in range(50)
	]
	save_json(tmp_path / "gsm8k" / "test-suite.json", _suite("gsm8k", "test", items, 50, 42))
	result = study_power(tmp_path, ["gsm8k"], minimum_gain_pp={"gsm8k": 30.0})
	test = result["domains"]["gsm8k"]["test"]
	assert test["items"] == 50 and "development" not in result["domains"]["gsm8k"]
	assert test["resolves_minimum_gain"] == (test["minimum_detectable_pp"] <= 30)
