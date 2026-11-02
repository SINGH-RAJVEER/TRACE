"""CPU-only learning curve contracts, with synthetic ledgers and scores."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from distillation.benchmarks.models import BenchmarkItem, DatasetSource, FrozenSuite
from distillation.cli import app
from distillation.learning_curve import (
	digest,
	execute_curve,
	plan_curve,
	summarize_curve,
	write,
)
from distillation.schemas import Message
from typer.testing import CliRunner


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
	def ledger(path: Path, prefix: str, count: int) -> None:
		rows = []
		for i in range(count):
			rows.append(
				json.dumps(
					{
						"example_id": f"{prefix}{i}",
						"attempt_id": "a",
						"split": "train",
						"task": "ag_news",
						"source": {"corpus": "fake", "source_id": str(i), "license": "test"},
						"prompt": [{"role": "user", "content": f"article {prefix}{i}"}],
						"completion": [{"role": "assistant", "content": "World"}],
						"teacher": {
							"provider": "fake",
							"model": "fake",
							"endpoint": "fake",
							"generated_at": "2026-01-01T00:00:00Z",
							"parameters": {},
							"terms_review_id": "test",
						},
						"finish_reason": "stop",
						"usage": {},
						"quality": {"status": "accepted", "checks": [], "reviewer": "test"},
						"attempts": [
							{
								"attempt_id": "a",
								"attempted_at": "2026-01-01T00:00:00Z",
								"status": "succeeded",
							}
						],
					}
				)
			)
		path.write_text("\n".join(rows) + "\n")

	dev, a1, suite_path = (tmp_path / name for name in ("dev.jsonl", "a1.jsonl", "suite.json"))
	ledger(dev, "dev", 500)
	ledger(a1, "a1", 1000)
	suite = FrozenSuite(
		seed=42,
		created_at=datetime.now(UTC),
		sources=[
			DatasetSource(
				benchmark="ag_news",
				repository="fake",
				revision="fake",
				configuration="fake",
				split="development",
				license="test",
				url="fake",
				available=1000,
				selected=1000,
			)
		],
		items=[
			BenchmarkItem(
				id=f"test{i}",
				benchmark="ag_news",
				category="World",
				question="test",
				messages=[Message(role="user", content=f"test article {i}")],
				metric="label_accuracy",
				answer="World",
				choices=["World", "Sports"],
			)
			for i in range(1000)
		],
	)
	suite.sha256 = suite.digest()
	suite_path.write_text(suite.model_dump_json())
	return dev, a1, suite_path


def test_nested_deterministic_sources(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
	first, second = tmp_path / "one", tmp_path / "two"
	plan = plan_curve(first, *inputs)
	assert plan == plan_curve(second, *inputs)
	previous: list[dict[str, Any]] = []
	for n in plan["sizes"]:
		rows = json.loads((first / f"subset-{n}.json").read_text())
		assert rows[: len(previous)] == previous
		assert len(rows) == n
		assert all(r["source"] == "development" for r in rows[: min(n, 500)])
		assert all(r["source"] == "A1/s42/undefended" for r in rows[500:])
		assert digest(first / f"subset-{n}.json") == digest(second / f"subset-{n}.json")
		assert all(r["record_sha256"] and r["line"] > 0 for r in rows)
		previous = rows
	third = tmp_path / "three"
	plan_curve(third, *inputs, subset_seed=99)
	assert digest(first / "subset-50.json") != digest(third / "subset-50.json")
	assert plan["estimated_gpu_hours"] == [4, 7]
	assert not (first / "scores").exists()


def test_resume_summary_and_integrity(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
	output = tmp_path / "run"
	plan_curve(output, *inputs, sizes=(50, 1500), fraction=0.5)
	calls: list[tuple[int | None, int]] = []

	def runner(path: Path, recipe: dict[str, Any], n: int | None, seed: int) -> dict[str, Any]:
		assert recipe["learning_rate"] == 1e-4 and recipe["epochs"] == 3
		calls.append((n, seed))
		# Paired base accuracy .2, n50 accuracy .4, n1500 accuracy .6.
		correct = 200 if n is None else 400 if n == 50 else 600
		if n is not None:
			correct += (seed - 43) * 100
		return {f"test{i}": int(i < correct) for i in range(1000)}

	assert execute_curve(output, runner) == {"completed": 7, "skipped": 0}
	assert execute_curve(output, runner) == {"completed": 0, "skipped": 7}
	assert len(calls) == 7
	(output / "scores/n50-s43.json").unlink()
	assert execute_curve(output, runner) == {"completed": 1, "skipped": 6}
	result = summarize_curve(output, replicates=100)
	assert [r["mean_gain_pp"] for r in result["results"]] == [20, 40]
	assert result["results"][0]["seed_gains_pp"] == [10, 20, 30]
	assert result["results"][1]["seed_gains_pp"] == [30, 40, 50]
	assert result["smallest_n"] == 50
	assert result["target_gain_pp"] == 20
	assert result["results"][0]["ci95_pp"][0] < 20 < result["results"][0]["ci95_pp"][1]
	assert result == summarize_curve(output, replicates=100)
	# A nonpositive reference cannot define a useful gain threshold.
	for path in (output / "scores").glob("n*.json"):
		data = json.loads(path.read_text())
		data["scores"] = {f"test{i}": int(i < 200) for i in range(1000)}
		write(path, data)
	assert summarize_curve(output, replicates=100)["smallest_n"] is None
	(output / "subset-50.json").write_text("[]")
	with pytest.raises(ValueError, match="hash mismatch"):
		execute_curve(output, runner)


def test_partial_failure_and_score_validation(
	inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
	output = tmp_path / "run"
	plan_curve(output, *inputs, sizes=(1500,))

	def fail(path: Path, recipe: dict[str, Any], n: int | None, seed: int) -> dict[str, Any]:
		if n is not None:
			raise RuntimeError("interrupted")
		return {f"test{i}": 0 for i in range(1000)}

	with pytest.raises(RuntimeError, match="interrupted"):
		execute_curve(output, fail)
	assert list((output / "scores").glob("*.json")) == [output / "scores/base.json"]
	with pytest.raises(ValueError, match="every suite item"):
		execute_curve(output, lambda *args: {"test0": 1})
	with pytest.raises(FileNotFoundError):
		summarize_curve(output)
	data = json.loads((output / "scores/base.json").read_text())
	data["contract"] = "wrong"
	write(output / "scores/base.json", data)
	with pytest.raises(ValueError, match="different plan"):
		execute_curve(output, fail)


def test_cli_plan(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
	dev, a1, suite = inputs
	result = CliRunner().invoke(
		app,
		[
			"learning-curve",
			"plan",
			"--output",
			str(tmp_path / "cli"),
			"--development",
			str(dev),
			"--a1",
			str(a1),
			"--suite",
			str(suite),
		],
	)
	assert result.exit_code == 0, result.output
	assert "One domain" in result.output
	assert len(json.loads(result.output)["cells"]) == 21
