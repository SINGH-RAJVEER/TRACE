"""Optional coding checks include real Linux sandbox containment, without model calls."""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("coding", reason="install the optional coding extra")

from coding.dataset import Decontaminator, Ineligible, convert, prepare_corpus  # noqa: E402
from coding.extract import extract_code  # noqa: E402
from coding.sandbox import Limits, SandboxUnavailable, run_python  # noqa: E402
from coding.scoring import CodingItem, Score, pass_at_1  # noqa: E402


@pytest.fixture
def sandbox() -> None:
	try:
		result = run_python("print('ready')")
	except SandboxUnavailable as error:
		pytest.skip(f"sandbox unavailable: {error}")
	assert result.returncode == 0 and result.stdout == "ready\n"


def test_extract_and_generation_statuses() -> None:
	assert extract_code("Explanation\n```python\ndef f(): return 1\n```") == "def f(): return 1"
	assert extract_code("def f(): return 1") is not None
	assert extract_code("```python\nnot valid python!\n```\n```python\nx=1\n```") is None
	assert extract_code("# only a comment") is None
	item = CodingItem("assert f() == 1")
	assert pass_at_1(item, "invalid!").status == "invalid"
	assert pass_at_1(item, "def f(): return 1", finish_reason="length").status == "truncated"
	assert pass_at_1(item, "def f(): return 1", finish_reason=None).status == "error"
	with pytest.raises(ValueError, match="top-level"):
		CodingItem("def test_unused():\n\tassert False")


def test_sandbox_required(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("coding.sandbox.shutil.which", lambda name: None)
	with pytest.raises(SandboxUnavailable, match="bubblewrap"):
		run_python("raise AssertionError('must not run')")


def test_correct_incorrect_and_early_exit(sandbox: None) -> None:
	item = CodingItem("assert add(2, 3) == 5\nassert add(-1, 1) == 0")
	assert pass_at_1(item, "def add(a, b): return a + b").correct
	wrong = pass_at_1(item, "def add(a, b): return a - b")
	assert wrong.status == "ok" and not wrong.correct
	assert not pass_at_1(item, "import os; os._exit(0)").correct


def test_loop_timeout(sandbox: None) -> None:
	score = pass_at_1(
		CodingItem("assert f() == 1"),
		"def f():\n\twhile True: pass",
		limits=Limits(wall_seconds=0.25, cpu_seconds=2),
	)
	assert score.status == "timeout" and not score.correct


@pytest.mark.parametrize(
	"operation",
	[
		"__import__('socket').socket()",
		"__import__('os').fork()",
		"__import__('os').posix_spawn('/bin/true', ['/bin/true'], {})",
	],
)
def test_network_and_process_creation_denied(sandbox: None, operation: str) -> None:
	result = run_python(
		f"try:\n\t{operation}\nexcept PermissionError:\n\tprint('denied')\n"
		"else:\n\traise AssertionError('operation escaped')"
	)
	assert result.returncode == 0 and result.stdout == "denied\n"


def test_filesystem_and_secrets(sandbox: None, tmp_path: Path) -> None:
	secret = tmp_path / "host-secret"
	secret.write_text("must not be visible")
	result = run_python(
		f"from pathlib import Path\nassert not Path({str(secret)!r}).exists()\n"
		"for path in ['/outside', '/dev/escape', '/dev/shm/escape', '/work/run.py']:\n"
		"\ttry: Path(path).write_text('escape')\n"
		"\texcept OSError: pass\n"
		"\telse: raise AssertionError(path)\n"
		"Path('/tmp/allowed').write_text('yes')\nprint('contained')"
	)
	assert result.returncode == 0 and result.stdout == "contained\n"
	assert secret.read_text() == "must not be visible"


def test_memory_and_output_bounded(sandbox: None) -> None:
	result = run_python(
		"try: value = bytearray(1024 ** 3)\n"
		"except MemoryError: print('bounded')\nelse: raise AssertionError('unbounded')"
	)
	assert result.returncode == 0 and result.stdout == "bounded\n"
	output = run_python("while True: print('x' * 10000)", limits=Limits(file_bytes=4096))
	assert len(output.stdout.encode()) <= 4096 and output.returncode != 0


def row(identifier: int = 1) -> dict[str, Any]:
	return {
		"id": identifier,
		"input": f"Return the number {identifier}.",
		"output": f"def value(): return {identifier}",
		"unit_tests": repr([f"assert value() == {identifier}"]),
		"domain": "numbers",
	}


def test_conversion_and_decontamination() -> None:
	example = convert(row())
	assert "assert" not in example.user and "return 1" not in example.user
	assert example.to_task_example().gold_completion.startswith("```python")
	evaluation = [
		{"id": "external", "specification": "Other wording", "solution": "def renamed(): return 1"}
	]
	assert Decontaminator(evaluation).overlap(example) == "external"
	text = " ".join(f"token{i}" for i in range(40))
	index = Decontaminator([{"id": "overlap", "specification": text, "solution": "x = 0"}])
	assert index.overlap(convert({**row(), "input": "An instruction " + text})) == "overlap"
	with pytest.raises(Ineligible):
		convert({**row(), "unit_tests": "__import__('os').system('false')"})


def test_prepare_all_rows_and_audit(sandbox: None, tmp_path: Path) -> None:
	output = tmp_path / "corpus"
	manifest = prepare_corpus(
		[row(), row(), row(2), {**row(3), "output": "def value(): return -1"}],
		[{"id": "test", "specification": "held out", "solution": "def value(): return 2"}],
		output,
	)
	assert manifest["complete"] and manifest["limit"] is None
	assert manifest["counts"] == {
		"raw": 4,
		"eligible": 1,
		"duplicate": 1,
		"test_overlap": 1,
		"reference_failed": 1,
	}
	assert len((output / "exclusions.jsonl").read_text().splitlines()) == 3
	assert json.loads((output / "manifest.json").read_text())["eligible_sha256"]
	with pytest.raises(FileExistsError):
		prepare_corpus([], [{"id": "t", "specification": "t", "solution": "x=0"}], output)


def test_interrupted_scan_is_incomplete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
	def unavailable(*args: Any, **kwargs: Any) -> None:
		raise SandboxUnavailable("host failure")

	monkeypatch.setattr("coding.dataset.pass_at_1", unavailable)
	with pytest.raises(SandboxUnavailable):
		prepare_corpus(
			[row()], [{"id": "t", "specification": "t", "solution": "x=0"}], tmp_path / "corpus"
		)
	assert not json.loads((tmp_path / "corpus/manifest.json").read_text())["complete"]


def test_full_corpus_exceeds_old_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
	monkeypatch.setattr("coding.dataset.pass_at_1", lambda *args, **kwargs: Score("ok", True))
	manifest = prepare_corpus(
		(row(index) for index in range(8003)),
		[{"id": "t", "specification": "held out", "solution": "x = 0"}],
		tmp_path / "full",
	)
	assert manifest["counts"] == {"raw": 8003, "eligible": 8003}


@pytest.fixture(scope="module")
def task_build(tmp_path_factory: pytest.TempPathFactory) -> Path:
	from distillation.tasks import build_task

	directory = tmp_path_factory.mktemp("coding-task") / "coding"
	build_task(
		task="coding",
		output_dir=directory,
		development_limit=5,
		rows={
			"train": [row(i) for i in range(160)],
			"test": [
				{
					"task_id": "Fixture/double",
					"entry_point": "double",
					"prompt": 'def double(x):\n\t"""Return twice the input."""\n',
					"canonical_solution": "\treturn x * 2\n",
					"test": "import numpy as np\ndef check(candidate):\n"
					"\tassert np.allclose(candidate(3), 6)\n",
				}
			],
		},
	)
	return directory


def test_default_coding_build_has_disjoint_gold_blind_pools(task_build: Path) -> None:
	from distillation.benchmarks.models import load_suite
	from distillation.io import read_jsonl
	from distillation.schemas import PromptRecord
	from distillation.tasks import LEGITIMATE_ROLES, ROLES, budget_run
	from distillation.workloads import task_references

	manifest = json.loads((task_build / "manifest.json").read_text())
	assert manifest["complete"] and manifest["corpus_limit"] is None
	assert sum(manifest["roles"].values()) == 160
	assert all(manifest["roles"][role] > 0 for role in ROLES)
	assert manifest["max_output_tokens"] == 512
	assert (
		load_suite(task_build / "test-suite.json").sources[0].repository == "evalplus/humanevalplus"
	)
	assert len(load_suite(task_build / "development-suite.json").items) == 5
	groups: set[str] = set()
	for role in ("attacker", *LEGITIMATE_ROLES):
		path = task_build / ("attacker.jsonl" if role == "attacker" else f"legitimate/{role}.jsonl")
		prompts = read_jsonl(path, PromptRecord)
		assert len(prompts) == manifest["roles"][role]
		assert all(p.source.group_id is not None for p in prompts)
		current = {str(p.source.group_id) for p in prompts}
		assert not groups & current
		groups.update(current)
		assert all(
			"assert value" not in m.content and "return " not in m.content
			for p in prompts
			for m in p.prompt
		)
		assert all(p.task == "coding" for p in prompts)
	assert (
		sum(1 for _ in (task_build / "gold-answers.jsonl").open()) == manifest["roles"]["attacker"]
	)
	references = task_references(task_build)
	assert references and all(item.test_code for item in references.values())
	budget_run(
		task_dir=task_build,
		budget=manifest["roles"]["attacker"],
		seed=42,
		output_dir=task_build / "runs" / f"s42-b{manifest['roles']['attacker']}",
	)
	assert (
		len(
			read_jsonl(
				task_build / "runs" / f"s42-b{manifest['roles']['attacker']}" / "prompts.jsonl",
				PromptRecord,
			)
		)
		== manifest["roles"]["attacker"]
	)


@pytest.mark.asyncio
async def test_coding_runs_through_existing_benchmark_and_report(
	task_build: Path, tmp_path: Path
) -> None:
	from datetime import UTC, datetime

	from distillation.benchmarks.models import EvaluationConfig, ModelConfig, load_suite
	from distillation.benchmarks.reporting import build_report
	from distillation.benchmarks.runner import run_evaluation
	from distillation.schemas import Message, TeacherGeneration

	class Teacher:
		async def generate(self, request: Any) -> TeacherGeneration:
			assert "assert np.allclose" not in str(request.messages)
			return TeacherGeneration(
				id="fixture",
				provider="fixture",
				model="fixture",
				generated_at=datetime.now(UTC),
				message=Message(
					role="assistant", content="```python\ndef double(x): return x * 2\n```"
				),
				finish_reason="stop",
				endpoint="fixture",
			)

		async def close(self) -> None:
			pass

	config = EvaluationConfig(
		experiment="coding-fixture",
		suite_path=task_build / "test-suite.json",
		output_dir=tmp_path,
		models={
			"teacher": ModelConfig(
				backend="teacher_api",
				model="fixture",
				revision="fixture",
				base_url="http://fixture",
			)
		},
	)
	run = await run_evaluation(config, "teacher", factory=lambda config: Teacher())
	assert run.suite_sha256 == load_suite(config.suite_path).sha256
	assert run.samples[0].correct and run.samples[0].status == "ok"
	report = build_report(tmp_path, run.comparison_key())
	assert report["scores"]["teacher"]["coding"]["primary_metric"] == "pass_at_1"
	assert report["scores"]["teacher"]["coding"]["primary_score"] == 100


@pytest.mark.asyncio
async def test_coding_scoring_retains_failures_and_propagates_infrastructure_errors(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	from distillation.benchmarks.models import BenchmarkItem
	from distillation.benchmarks.scoring import evaluate_answer
	from distillation.schemas import Message

	item = BenchmarkItem(
		id="coding:test:1",
		benchmark="coding",
		category="fixture",
		question="Return one",
		messages=[Message(role="user", content="Return one")],
		metric="pass_at_1",
		answer="def f(): return 1",
		test_code="assert f() == 1",
		entry_point="f",
	)
	assert (await evaluate_answer(item, "def f(): return 2", "stop")).correct is False
	assert (await evaluate_answer(item, "invalid!", "stop")).status == "invalid"
	assert (await evaluate_answer(item, "def f(): return 1", "length")).status == "truncated"

	def unavailable(*args: Any, **kwargs: Any) -> None:
		raise SandboxUnavailable("fixture unavailable")

	monkeypatch.setattr("coding.scoring.run_python", unavailable)
	with pytest.raises(SandboxUnavailable):
		await evaluate_answer(item, "def f(): return 1", "stop")


@pytest.mark.asyncio
async def test_coding_collection_prepares_response_token_training_data(
	task_build: Path, tmp_path: Path
) -> None:
	import re
	from datetime import UTC, datetime

	from distillation.collection import CollectionOptions, collect
	from distillation.io import read_jsonl
	from distillation.preparation import prepare_sft_dataset
	from distillation.review import review_ledger
	from distillation.schemas import Message, SFTRecord, TeacherGeneration
	from distillation.tasks import budget_run

	budget_run(task_dir=task_build, budget=3, seed=42, output_dir=tmp_path)

	class Teacher:
		async def generate(self, request: Any) -> TeacherGeneration:
			assert request.max_output_tokens == 512 and not request.enable_thinking
			match = re.search(r"Return the number (\d+)", request.messages[-1].content)
			assert match is not None
			return TeacherGeneration(
				id="fixture",
				provider="fixture",
				model="fixture",
				generated_at=datetime.now(UTC),
				message=Message(role="assistant", content=f"def value(): return {match[1]}"),
				finish_reason="stop",
				endpoint="fixture",
			)

		async def close(self) -> None:
			pass

	ledger, reviews, dataset = (
		tmp_path / name for name in ("ledger.jsonl", "reviews.jsonl", "train.jsonl")
	)
	await collect(
		input_path=tmp_path / "prompts.jsonl",
		output_path=ledger,
		teacher=Teacher(),
		options=CollectionOptions(
			terms_review_id="fixture", max_output_tokens=512, enable_thinking=False
		),
	)
	review_ledger(ledger_path=ledger, output_path=reviews)
	written, skipped = prepare_sft_dataset(
		input_path=ledger,
		output_path=dataset,
		reviews_path=reviews,
		prompts_path=tmp_path / "prompts.jsonl",
	)
	assert written == 3 and skipped == 0
	assert all("def value" in row.completion[0].content for row in read_jsonl(dataset, SFTRecord))
