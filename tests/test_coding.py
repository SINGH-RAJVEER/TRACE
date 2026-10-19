"""Optional coding checks include real Linux sandbox containment, without model calls."""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("coding", reason="install the optional coding extra")

from coding.dataset import (  # noqa: E402
	Decontaminator,
	Ineligible,
	ast_hash,
	convert,
	prepare_corpus,
)
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


def test_conversion_and_decontamination() -> None:
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
