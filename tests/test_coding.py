"""Optional coding checks include real Linux sandbox containment, without model calls."""

import json
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from coding.dataset import (
	Decontaminator,
	Ineligible,
	ast_hash,
	convert,
	prepare_corpus,
)
from coding.extract import extract_code
from coding.sandbox import Limits, SandboxUnavailable, run_python
from coding.scoring import CodingItem, Score, pass_at_1


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
	cpu = pass_at_1(
		item=CodingItem("assert f() == 1"),
		response="def f():\n\twhile True: pass",
		limits=Limits(wall_seconds=4, cpu_seconds=1),
	)
	assert cpu.status == "timeout" and not cpu.correct
	# CPU exhaustion can terminate the payload before the wall clock expires.
	cpu = pass_at_1(
		CodingItem("assert f() == 1"),
		"def f():\n\twhile True: pass",
		limits=Limits(wall_seconds=4, cpu_seconds=1),
	)
	assert cpu.status == "timeout" and not cpu.correct


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


def reference_only(item: CodingItem, response: str, **kwargs: Any) -> Score:
	"""Passes every reference and fails every null stub, without a sandbox."""
	return Score("ok", "return None" not in response)


def test_conversion_and_decontamination() -> None:
	assert ast_hash("def f(a, b): return a + b") == ast_hash("def g(x, y): return x + y")
	assert ast_hash("def f(a, b): return a + b") != ast_hash("def f(a, b): return a + a")
	assert ast_hash("def f(a, b): return a + b") == ast_hash("def g(x, y): return x + y")
	assert ast_hash("def f(a, b): return a + b") != ast_hash("def f(a, b): return a + a")
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
		[
			row(),
			row(),
			row(2),
			{**row(3), "output": "def value(): return -1"},
			{
				**row(4),
				"output": "def value(): print(4)",
				"unit_tests": "['assert value() is None']",
			},
		],
		[{"id": "test", "specification": "held out", "solution": "def value(): return 2"}],
		output,
	)
	assert manifest["complete"] and manifest["limit"] is None
	assert manifest["counts"] == {
		"raw": 5,
		"eligible": 1,
		"duplicate": 1,
		"test_overlap": 1,
		"reference_failed": 1,
		"tests_accept_null_stub": 1,
	}
	assert len((output / "exclusions.jsonl").read_text().splitlines()) == 4
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


def test_slow_first_reference_does_not_starve_other_workers(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	later_started = Event()
	first_saw_later: list[bool] = []

	def score(item: CodingItem, response: str, **kwargs: Any) -> Score:
		if response == row(0)["output"]:
			first_saw_later.append(later_started.wait(timeout=2))
		if response == row(24)["output"]:
			later_started.set()
		return reference_only(item, response)

	monkeypatch.setattr("coding.dataset.pass_at_1", score)
	manifest = prepare_corpus(
		[row(index) for index in range(40)],
		[{"id": "test", "specification": "held out", "solution": "pass"}],
		tmp_path / "corpus",
		workers=2,
	)
	assert first_saw_later == [True]
	assert manifest["counts"] == {"raw": 40, "eligible": 40}


def test_parallel_qualification_matches_sequential_source_order(
	sandbox: None, tmp_path: Path
) -> None:
	rows = [
		{**row(1), "output": "def value(): return -1"},
		row(1),
		row(1),
		row(2),
		{**row(3), "unit_tests": "not a test list"},
		row(4),
		row(5),
	]
	evaluation = [{"id": "test", "specification": "held out", "solution": "def value(): return 4"}]
	sequential = prepare_corpus(rows, evaluation, tmp_path / "sequential", workers=1)
	parallel = prepare_corpus(iter(rows), evaluation, tmp_path / "parallel")
	assert parallel == sequential
	for name in ("eligible.jsonl", "exclusions.jsonl"):
		assert (tmp_path / "sequential" / name).read_bytes() == (
			tmp_path / "parallel" / name
		).read_bytes()
	assert parallel["counts"] == {
		"raw": 7,
		"reference_failed": 1,
		"eligible": 3,
		"duplicate": 1,
		"malformed_or_unsupported": 1,
		"test_overlap": 1,
	}


@pytest.mark.parametrize("workers", [None, 1])
def test_task_build_cli_forwards_parallel_default_and_sequential_override(
	workers: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
	from coding import DEFAULT_QUALIFICATION_WORKERS
	from distillation.cli import app
	from typer.testing import CliRunner

	seen = []

	def build(**kwargs: Any) -> dict[str, Any]:
		seen.append(kwargs["qualification_workers"])
		return {"roles": {}, "filtered": {}, "truncated": {}, "max_output_tokens": 512}

	monkeypatch.setattr("distillation.tasks.build_task", build)
	args = ["tasks", "build", "--task", "coding"]
	if workers is not None:
		args += ["--qualification-workers", str(workers)]
	result = CliRunner().invoke(app, args)
	assert result.exit_code == 0, result.output
	assert seen == [DEFAULT_QUALIFICATION_WORKERS if workers is None else workers]


def test_full_corpus_exceeds_old_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
	monkeypatch.setattr("coding.dataset.pass_at_1", reference_only)
	manifest = prepare_corpus(
		(row(index) for index in range(8003)),
		[{"id": "t", "specification": "held out", "solution": "x = 0"}],
		tmp_path / "full",
	)
	assert manifest["counts"] == {"raw": 8003, "eligible": 8003}


@pytest.mark.parametrize("workers", [1, 4])
def test_explicit_subset_stops_at_eligible_target_and_records_decided_rows(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, workers: int
) -> None:
	monkeypatch.setattr("coding.dataset.pass_at_1", reference_only)
	output = tmp_path / "subset"
	manifest = prepare_corpus(
		[row(0), row(0), *(row(i) for i in range(1, 30))],
		[{"id": "heldout", "specification": "held out", "solution": "x = 0"}],
		output,
		workers=workers,
		eligible_limit=5,
	)
	assert manifest["complete"] and manifest["limit"] == 5
	assert manifest["scope"] == "eligible-subset"
	assert manifest["counts"] == {"raw": 6, "eligible": 5, "duplicate": 1}
	assert [
		json.loads(line)["key"] for line in (output / "eligible.jsonl").read_text().splitlines()
	] == [str(i) for i in range(5)]
	assert [
		json.loads(line) for line in (output / "source-ids.jsonl").read_text().splitlines()
	] == ["0", "0", "1", "2", "3", "4"]


def test_subset_with_too_few_eligible_rows_is_not_complete(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	monkeypatch.setattr("coding.dataset.pass_at_1", reference_only)
	output = tmp_path / "subset"
	with pytest.raises(ValueError, match="eligible target"):
		prepare_corpus(
			[row(1)],
			[{"id": "heldout", "specification": "held out", "solution": "x=0"}],
			output,
			eligible_limit=2,
		)
	assert json.loads((output / "manifest.json").read_text())["complete"] is False


def test_global_source_permutation_is_seeded_and_not_a_prefix(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	datasets = pytest.importorskip("datasets")
	from distillation.coding_task import training_source
	from distillation.tasks import TASKS

	rows = [row(i) for i in range(100)]
	dataset = datasets.Dataset.from_list(rows)
	monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: dataset)
	first, metadata = training_source(TASKS["coding"].dataset, 42, 5)
	second, _ = training_source(TASKS["coding"].dataset, 42, 5)
	other, _ = training_source(TASKS["coding"].dataset, 43, 5)
	ids = [r["id"] for r in first]
	assert ids == [r["id"] for r in second]
	assert ids != [r["id"] for r in other]
	assert sorted(ids) == list(range(100)) and ids[:5] != list(range(5))
	assert metadata["source_rows"] == 100 and metadata["seed"] == 42
	assert metadata["algorithm"] == "datasets-global-shuffle-v1"


def test_subset_export_is_reproducible_disjoint_and_keeps_external_suite(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	from distillation.tasks import build_task

	monkeypatch.setattr("coding.dataset.pass_at_1", reference_only)
	rows = {
		"train": [row(i) for i in range(300)],
		"test": [
			{
				"task_id": "Fixture/heldout",
				"entry_point": "heldout",
				"prompt": "def heldout(x):\n",
				"canonical_solution": "\treturn x * 2\n",
				"test": "def check(candidate):\n\tassert candidate(2) == 4\n",
			}
		],
	}
	manifests = [
		build_task(
			task="coding",
			output_dir=tmp_path / str(workers),
			rows=rows,
			coding_pool_size=120,
			qualification_workers=workers,
		)
		for workers in (1, 4)
	]
	assert manifests[0]["roles"] == manifests[1]["roles"]
	assert sum(manifests[0]["roles"].values()) == 120
	assert manifests[0]["corpus_limit"] == 120
	assert manifests[0]["selection"]["source_rows"] == 300
	for filename in ("eligible.jsonl", "exclusions.jsonl", "source-ids.jsonl"):
		assert (tmp_path / "1/qualified" / filename).read_bytes() == (
			tmp_path / "4/qualified" / filename
		).read_bytes()
	from distillation.benchmarks.models import load_suite

	assert (
		load_suite(tmp_path / "1/test-suite.json").sha256
		== load_suite(tmp_path / "4/test-suite.json").sha256
	)
	groups: list[set[str]] = []
	for path in [tmp_path / "1/attacker.jsonl", *(tmp_path / "1/legitimate").glob("*.jsonl")]:
		current = {json.loads(line)["source"]["group_id"] for line in path.read_text().splitlines()}
		assert not any(current & previous for previous in groups)
		groups.append(current)


@pytest.fixture(scope="module")
def task_build(tmp_path_factory: pytest.TempPathFactory) -> Path:
	from distillation.tasks import build_task

	directory = tmp_path_factory.mktemp("coding-task") / "coding"
	build_task(
		task="coding",
		output_dir=directory,
		coding_pool_size=0,
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
			"test_mbpp": [
				{
					"task_id": 7,
					"code": "def triple(x):\n    return x * 3\n",
					"prompt": "Write a function to triple a number.",
					"test_imports": [],
					"test_list": ["assert triple(2) == 6"],
					"test": "def assertion(out, exp, atol):\n    assert out == exp\n"
					"inputs = [[2], [5]]\nresults = [6, 15]\n"
					"for inp, exp in zip(inputs, results):\n"
					"    assertion(triple(*inp), exp, 0)\n",
				},
				{
					"task_id": 255,
					"code": "def excluded(x):\n    return x\n",
					"prompt": "Declared exclusion.",
					"test_imports": [],
					"test_list": [],
					"test": "assert False\n",
				},
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
	from harness.serialization import detector_text

	manifest = json.loads((task_build / "manifest.json").read_text())
	assert manifest["complete"] and manifest["corpus_limit"] is None
	assert sum(manifest["roles"].values()) == 160
	assert all(manifest["roles"][role] > 0 for role in ROLES)
	assert manifest["max_output_tokens"] == 512
	suite = load_suite(task_build / "test-suite.json")
	assert [source.repository for source in suite.sources] == [
		"evalplus/humanevalplus",
		"evalplus/mbppplus",
	]
	assert [item.id for item in suite.items] == ["coding:test:Fixture/double", "coding:test:Mbpp/7"]
	mbpp = suite.items[1]
	assert mbpp.entry_point == "triple" and "Example: assert triple(2) == 6" in mbpp.question
	assert [e["id"] for e in manifest["evaluation_exclusions"]] == ["Mbpp/255"]
	assert len(load_suite(task_build / "development-suite.json").items) == 5
	for item in load_suite(task_build / "test-suite.json").items:
		assert detector_text("coding", item.messages[0].content) == item.question
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
					role="assistant",
					content=(
						"```python\ndef double(x): return x * 2\ndef triple(x): return x * 3\n```"
					),
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
	assert float(run.coding_runtime["cpu_seconds"]) > 0
	changed = run.model_copy(update={"coding_runtime": run.coding_runtime | {"wall_seconds": 999}})
	assert changed.comparison_key() != run.comparison_key()
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


@pytest.mark.asyncio
async def test_coding_agreement_compares_outputs_including_shared_wrong_answers() -> None:
	from distillation.benchmarks.models import BenchmarkItem, SampleResult
	from distillation.benchmarks.reporting import paired
	from distillation.benchmarks.scoring import evaluate_answer
	from distillation.schemas import Message

	item = BenchmarkItem(
		id="coding:agreement",
		benchmark="coding",
		category="fixture",
		question="Increment",
		messages=[Message(role="user", content="Increment")],
		metric="pass_at_1",
		answer="def f(x): return x + 1",
		entry_point="f",
		test_code="assert f(1) == 2\nassert f(3) == 4",
	)

	async def sample(code: str) -> SampleResult:
		score = await evaluate_answer(item, code, "stop")
		return SampleResult(
			item_id=item.id,
			benchmark="coding",
			category="fixture",
			response=code,
			prediction=score.prediction,
			correct=score.correct,
			status=score.status,
			latency_seconds=0,
			functional_trace=score.functional_trace,
		)

	left = await sample("def f(x): return x + 3")
	right = await sample("def f(x): return 3 + x")
	assert not left.correct and not right.correct
	comparison = paired([left], [right])
	assert comparison["functional_agreement"] == 100
	assert comparison["functional_agreement_n"] == 1
	assert comparison["functional_agreement_coverage"] == 1
	different = await sample("def f(x): return x + 5")
	assert paired([left], [different])["functional_agreement"] == 0
	invalid = await sample("def f(x): raise ValueError('no output')")
	assert paired([left], [invalid])["functional_agreement_n"] == 0
	# A wrong return type can break an oracle expression while still producing valid,
	# comparable function outputs on every reference-covered input.
	item = item.model_copy(
		update={"answer": "def f(x): return []", "test_code": "assert len(f(1)) == 0"}
	)
	wrong_type = await sample("def f(x): return 7")
	also_wrong_type = await sample("def f(x): return 3 + 4")
	assert not wrong_type.correct and not also_wrong_type.correct
	assert paired([wrong_type], [also_wrong_type])["functional_agreement"] == 100


@pytest.mark.parametrize(
	"finish,decision", [("stop", "cap_passed"), ("length", "raise_cap"), (None, "inconclusive")]
)
def test_development_cap_pilot_reports_and_exits_on_bad_caps(
	task_build: Path,
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	finish: str | None,
	decision: str,
) -> None:
	from datetime import UTC, datetime

	from distillation.cli import app
	from distillation.schemas import Message, TeacherGeneration
	from typer.testing import CliRunner

	class Teacher:
		closed = False

		async def generate(self, request: Any) -> TeacherGeneration:
			assert request.max_output_tokens == 512 and not request.enable_thinking
			return TeacherGeneration(
				id="pilot-fixture",
				provider="fixture",
				model="fixture",
				generated_at=datetime.now(UTC),
				message=Message(role="assistant", content="def value(): return 1"),
				finish_reason=finish,
				endpoint="fixture",
			)

		async def close(self) -> None:
			self.closed = True

	teacher = Teacher()
	monkeypatch.setattr("distillation.cli.TeacherAPIGateway", lambda **kwargs: teacher)
	result = CliRunner().invoke(
		app,
		[
			"tasks",
			"pilot",
			"--task-dir",
			str(task_build),
			"--output-dir",
			str(tmp_path / "pilot"),
			"--teacher-url",
			"http://fixture",
			"--limit",
			"3",
		],
	)
	assert result.exit_code == (0 if decision == "cap_passed" else 1), result.output
	summary = json.loads((tmp_path / "pilot" / "pilot.json").read_text())
	assert summary["complete"] and summary["n"] == 3
	assert summary["decision"] == decision
	assert teacher.closed
