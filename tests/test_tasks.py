import hashlib
import json
import tomllib
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


def test_paper_tasks_scenario_keeps_its_own_ledger_and_resumes(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	import distillation.launcher as launcher
	from distillation.cli import app
	from typer.testing import CliRunner

	monkeypatch.chdir(tmp_path)
	monkeypatch.setenv("TEACHER_TERMS_REVIEW_ID", "test-approval")
	monkeypatch.setattr(launcher.Launcher, "install", lambda self, component: None)
	build_task(task="ag_news", output_dir=tmp_path / "data/tasks/ag_news", rows=_ag_news_rows())
	(tmp_path / "configs").mkdir()
	(tmp_path / "configs/train.task.toml").write_text(
		'base_model = "student"\ndataset_path = "unused.jsonl"\noutput_dir = "unused"\n'
		"seed = 0\nlearning_rate = 2e-4\nchat_template_kwargs = { enable_thinking = false }\n"
	)
	steps: list[str] = []

	def distill(self: launcher.Launcher, *args: str, env: dict[str, str] | None = None) -> None:
		if args[0] == "collect":
			assert args[args.index("--max-output-tokens") + 1] == "16"
			steps.append("collect")
			prompts = read_jsonl(Path(args[args.index("--input") + 1]), PromptRecord)
			output = Path(args[args.index("--output") + 1])
			have = (
				{r.example_id for r in read_jsonl(output, LedgerRecord)}
				if output.is_file()
				else set()
			)
			with output.open("a") as ledger:
				for prompt in prompts:
					if prompt.example_id not in have:
						record = _ledger(prompt)
						record.teacher.parameters = {
							"temperature": 0.0,
							"max_output_tokens": 16,
							"seed": None,
							"enable_thinking": False,
						}
						record.teacher.terms_review_id = "test-approval"
						ledger.write(record.model_dump_json() + "\n")
		elif args[0] == "train":
			config = Path(args[args.index("--config") + 1])
			values = tomllib.loads(config.read_text())
			steps.append(f"train {config.parent.name} {config.name}")
			output = Path(values["output_dir"])
			output.mkdir(parents=True)
			digest = hashlib.sha256(Path(values["dataset_path"]).read_bytes()).hexdigest()
			(output / "training-manifest.json").write_text(
				json.dumps({"dataset_sha256": digest, "config": values})
			)
		else:
			if args[0] == "review":
				# The length check loads the tokenizer, which needs the training stack.
				assert "--references" not in args
				args = args[: args.index("--config")]
			result = CliRunner().invoke(app, list(args))
			assert result.exit_code == 0, result.output

	monkeypatch.setattr(launcher.Launcher, "distill", distill)
	monkeypatch.setattr(
		launcher.Launcher,
		"start_teacher_stack",
		lambda self, host, port, upstream_url, api_log: steps.append("serve"),
	)
	monkeypatch.setattr(launcher.Launcher, "stop_background", lambda self: steps.append("stop"))
	command = ["paper-tasks", "--tasks", "ag_news", "--seeds", "42,1", "--budgets", "40,80,99999"]

	result = CliRunner().invoke(launcher.run_app, command)
	assert result.exit_code == 0, result.output
	trained = [
		f"train s{s}-b{b} train.{k}.toml"
		for s in (42, 1)
		for b in (40, 80)
		for k in ("attacker", "gold")
	]
	assert steps == ["serve", "collect", "collect", "stop", *trained, "stop"]
	task_dir = tmp_path / "data/tasks/ag_news"
	assert len(read_jsonl(task_dir / "ledger.jsonl", LedgerRecord)) <= 160
	assert not (tmp_path / "data/teacher").exists()
	run = task_dir / "runs/s42-b40"
	attacker = tomllib.loads((run / "train.attacker.toml").read_text())
	assert attacker["dataset_path"] == "data/tasks/ag_news/runs/s42-b40/train.jsonl"
	assert attacker["output_dir"] == "artifacts/tasks/ag_news/s42-b40/attacker"
	assert attacker["seed"] == 42
	assert attacker["chat_template_kwargs"] == {"enable_thinking": False}
	assert len(read_jsonl(run / "train.jsonl", SFTRecord)) == 40
	gold = tomllib.loads((run / "train.gold.toml").read_text())
	assert gold["dataset_path"] == "data/tasks/ag_news/runs/s42-b40/gold-train.jsonl"

	# Everything is collected and trained, so a rerun neither serves the teacher nor trains.
	steps.clear()
	result = CliRunner().invoke(launcher.run_app, command)
	assert result.exit_code == 0, result.output
	assert steps == ["stop"]
