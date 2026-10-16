import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from distillation.benchmarks.models import BenchmarkItem, load_suite
from distillation.benchmarks.scoring import score_answer, squad_f1
from distillation.io import canonical_json, read_jsonl, write_jsonl
from distillation.schemas import (
	CollectionAttempt,
	LedgerRecord,
	Message,
	PromptRecord,
	QualityRecord,
	SFTRecord,
	TeacherMetadata,
	TokenUsage,
)
from distillation.tasks import GoldRecord, budget_run, build_task


def _ag_news_rows() -> dict[str, list[dict[str, Any]]]:
	train = [{"text": f"Story {i} about markets and teams", "label": i % 4} for i in range(400)]
	train.append({"text": "Shared headline in both splits", "label": 1})
	test = [
		{"text": "Shared headline in both splits", "label": 1},
		{"text": "A held-out story", "label": 3},
	]
	return {"train": train, "test": test}


def test_task_build_separates_pools_and_keeps_gold_out_of_attacker_files(tmp_path: Path) -> None:
	manifest = build_task(task="ag_news", output_dir=tmp_path, rows=_ag_news_rows())
	assert manifest["filtered"] == {"test_overlap": 1}
	assert sum(manifest["roles"].values()) == 400
	attacker = read_jsonl(tmp_path / "attacker.jsonl", PromptRecord)
	legitimate = [
		record
		for path in sorted((tmp_path / "legitimate").glob("*.jsonl"))
		for record in read_jsonl(path, PromptRecord)
	]
	development = load_suite(tmp_path / "development-suite.json")
	attacker_groups = {record.source.group_id for record in attacker}
	legitimate_groups = {record.source.group_id for record in legitimate}
	assert attacker_groups and legitimate_groups and development.items
	assert not attacker_groups & legitimate_groups
	assert {record.split for record in attacker} == {"train"}
	assert {record.split for record in legitimate} == {"test"}
	gold = read_jsonl(tmp_path / "gold-answers.jsonl", GoldRecord)
	assert [record.example_id for record in gold] == [record.example_id for record in attacker]
	assert len(load_suite(tmp_path / "test-suite.json").items) == 2
	with pytest.raises(ValueError, match="already holds"):
		build_task(task="ag_news", output_dir=tmp_path, rows=_ag_news_rows())


def _ledger(prompt: PromptRecord) -> LedgerRecord:
	now = datetime.now(UTC)
	return LedgerRecord(
		example_id=prompt.example_id,
		attempt_id="att",
		split=prompt.split,
		task=prompt.task,
		source=prompt.source,
		prompt=prompt.prompt,
		teacher=TeacherMetadata(
			provider="fake",
			model="fake",
			endpoint="fake",
			generated_at=now,
			parameters={"max_output_tokens": 16},
			terms_review_id="review",
		),
		completion=[Message(role="assistant", content="World")],
		usage=TokenUsage(),
		quality=QualityRecord(status="pending", checks=["non_empty"], reviewer="collector-v1"),
		attempts=[CollectionAttempt(attempt_id="att", attempted_at=now, status="succeeded")],
	)


def test_budgets_nest_per_seed_and_subset_the_ledger(tmp_path: Path) -> None:
	task_dir = tmp_path / "task"
	build_task(task="ag_news", output_dir=task_dir, rows=_ag_news_rows())
	small = budget_run(task_dir=task_dir, budget=20, seed=1, output_dir=tmp_path / "small")
	large = budget_run(task_dir=task_dir, budget=50, seed=1, output_dir=tmp_path / "large")
	assert small["pool"] == large["pool"]
	first = read_jsonl(tmp_path / "small" / "prompts.jsonl", PromptRecord)
	second = read_jsonl(tmp_path / "large" / "prompts.jsonl", PromptRecord)
	assert second[:20] == first
	other = budget_run(task_dir=task_dir, budget=20, seed=20, output_dir=tmp_path / "other")
	assert other["sha256"]["prompts.jsonl"] != small["sha256"]["prompts.jsonl"]
	gold = read_jsonl(tmp_path / "small" / "gold-train.jsonl", SFTRecord)
	assert [record.prompt for record in gold] == [prompt.prompt for prompt in first]
	ledger = tmp_path / "ledger.jsonl"
	write_jsonl(ledger, [_ledger(prompt) for prompt in reversed(second[:30])])
	summary = budget_run(
		task_dir=task_dir, budget=50, seed=1, output_dir=tmp_path / "large", ledger_path=ledger
	)
	assert (summary["collected"], summary["missing"]) == (30, 20)
	assert summary["rejected_by_collector"] == 0
	subset = read_jsonl(tmp_path / "large" / "ledger.jsonl", LedgerRecord)
	assert [record.example_id for record in subset] == [p.example_id for p in second[:30]]
	mismatched = _ledger(second[0])
	mismatched.teacher.parameters["max_output_tokens"] = 1024
	write_jsonl(ledger, [mismatched])
	with pytest.raises(ValueError, match="max-output-tokens 16"):
		budget_run(
			task_dir=task_dir, budget=50, seed=1, output_dir=tmp_path / "bad", ledger_path=ledger
		)
	with pytest.raises(ValueError, match="attacker pool size"):
		budget_run(task_dir=task_dir, budget=10_000, seed=1, output_dir=tmp_path / "huge")


def test_squad_and_label_scoring_follow_the_task_contracts() -> None:
	squad = BenchmarkItem(
		id="squad:validation:1",
		benchmark="squad",
		category="Normans",
		question="q",
		messages=[Message(role="user", content="q")],
		metric="squad",
		answer="in France",
		answers=["in France", "Normandy, France"],
	)
	assert score_answer(squad, "In France.") == ("in france", True)
	assert score_answer(squad, "The north of France")[1] is False
	assert squad_f1(squad, "north of France") == pytest.approx(0.4)
	assert score_answer(squad, "...")[0] is None
	label = BenchmarkItem(
		id="ag_news:test:1",
		benchmark="ag_news",
		category="Sci/Tech",
		question="q",
		messages=[Message(role="user", content="q")],
		metric="label_accuracy",
		answer="Sci/Tech",
		choices=["World", "Sports", "Business", "Sci/Tech"],
	)
	assert score_answer(label, " sci/tech.\n") == ("Sci/Tech", True)
	assert score_answer(label, "Topic: Sci/Tech") == (None, False)


def test_suites_without_gold_answer_lists_keep_their_hash(tmp_path: Path) -> None:
	build_task(task="ag_news", output_dir=tmp_path, rows=_ag_news_rows())
	suite = load_suite(tmp_path / "test-suite.json")
	payload = suite.model_dump(mode="json", exclude={"sha256", "created_at", "items"})
	payload["items"] = [item.model_dump(mode="json", exclude={"answers"}) for item in suite.items]
	legacy = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
	assert suite.sha256 == legacy
	assert json.loads((tmp_path / "manifest.json").read_text())["max_output_tokens"] == 16
