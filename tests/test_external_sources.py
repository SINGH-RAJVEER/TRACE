"""Independent A7 sources follow each domain's frozen gateway template."""

import json
from pathlib import Path

import pytest
from distillation.benchmarks.models import save_json
from distillation.external_sources import EXTERNAL_SOURCES, build_external
from distillation.io import read_jsonl
from distillation.schemas import PromptRecord
from distillation.tasks import _benchmark_item, _suite, gsm8k_example
from harness.serialization import valid_gateway_turn

ROWS = {
	"ag_news": [{"text": "wales want rugby league training"}, {"text": "  "}],
	"squad": [
		{"context": "CNN -- A court ruled.", "question": "Who ruled?"},
		{"context": "CNN -- A court ruled.", "question": "Who ruled?"},
	],
	"gsm8k": [{"question_without_options": "A train covers 60 km in 2 hours. What is its speed?"}],
	"coding": [
		{
			"instruction": "Write a function `add` that returns the sum of two integers.",
			"response": (
				"Here it is.\n```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```"
			),
		},
		{"instruction": "Explain recursion.", "response": "No code here."},
		{
			"instruction": "Two functions.",
			"response": "```python\ndef f():\n    pass\n\ndef g():\n    pass\n```",
		},
	],
}


@pytest.mark.parametrize("task", sorted(ROWS))
def test_external_pools_use_the_gateway_template_and_record_their_pin(
	tmp_path: Path, task: str
) -> None:
	manifest = build_external(task=task, task_dir=tmp_path, rows={"train": ROWS[task]})
	records = read_jsonl(tmp_path / "external" / "attacker.jsonl", PromptRecord)
	assert manifest["prompts"] == len(records) == 1
	assert manifest["dataset"]["revision"] == EXTERNAL_SOURCES[task].dataset.revision
	assert sum(manifest["filtered"].values()) == len(ROWS[task]) - 1
	for record in records:
		assert record.split == "train" and record.task == task
		assert record.source.corpus == EXTERNAL_SOURCES[task].dataset.repository
		assert valid_gateway_turn(task, record.prompt[0].content)
	with pytest.raises(ValueError, match="already holds"):
		build_external(task=task, task_dir=tmp_path, rows={"train": ROWS[task]})


def test_external_pools_drop_exact_evaluation_overlap(tmp_path: Path) -> None:
	problem = ROWS["gsm8k"][0]["question_without_options"]
	example = gsm8k_example(0, {"question": problem, "answer": "30 #### 30"})
	suite = _suite("gsm8k", "test", [_benchmark_item("gsm8k", "test", example)], 1, 42)
	save_json(tmp_path / "test-suite.json", suite)
	with pytest.raises(ValueError, match="no eligible"):
		build_external(task="gsm8k", task_dir=tmp_path, rows={"train": ROWS["gsm8k"]})
	assert not (tmp_path / "external" / "manifest.json").exists()
	assert json.loads(json.dumps(EXTERNAL_SOURCES["gsm8k"].splits)) == ["train"]
