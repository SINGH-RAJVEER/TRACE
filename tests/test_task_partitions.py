"""Attacker pools are split into sealed, source-disjoint exploratory and confirmatory partitions."""

import json
from pathlib import Path

import pytest
from distillation.gateway_cli import _pools
from distillation.io import read_jsonl
from distillation.schemas import PromptRecord
from distillation.tasks import build_task


def _rows(count: int, prefix: str) -> list[dict[str, str]]:
	return [
		{
			"question": f"{prefix} {index}: what is {index} plus 1?",
			"answer": f"Add.\n#### {index + 1}",
		}
		for index in range(count)
	]


@pytest.fixture
def math_dir(tmp_path: Path) -> Path:
	directory = tmp_path / "math"
	build_task(
		task="gsm8k",
		output_dir=directory,
		rows={"train": _rows(600, "Train"), "test": _rows(3, "Test")},
	)
	return directory


def test_partitions_are_disjoint_and_cover_the_attacker_role(math_dir: Path) -> None:
	manifest = json.loads((math_dir / "manifest.json").read_text())
	exploratory = read_jsonl(math_dir / "attacker.jsonl", PromptRecord)
	confirmatory = read_jsonl(math_dir / "attacker-confirmatory.jsonl", PromptRecord)
	groups = [{prompt.source.group_id for prompt in pool} for pool in (exploratory, confirmatory)]
	assert exploratory and confirmatory
	assert not groups[0] & groups[1]
	assert len(exploratory) + len(confirmatory) == manifest["roles"]["attacker"]
	assert manifest["attacker_partitions"]["prompts"] == {
		"exploratory": len(exploratory),
		"confirmatory": len(confirmatory),
	}


def test_builds_without_sealed_partitions_are_refused(math_dir: Path) -> None:
	# Builds without sealed partitions would mix confirmatory prompts into campaigns.
	manifest = json.loads((math_dir / "manifest.json").read_text())
	del manifest["attacker_partitions"]
	(math_dir / "manifest.json").write_text(json.dumps(manifest))
	with pytest.raises(ValueError, match="predates sealed attacker partitions"):
		_pools(math_dir.parent, "gsm8k,squad", "attacker")
