"""CPU-only learning curve contracts, with synthetic ledgers and scores."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from distillation.benchmarks.models import BenchmarkItem, DatasetSource, FrozenSuite
from distillation.cli import app
from distillation.learning_curve import (
	MATH_RECIPE,
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
	assert plan["version"] == "ag-news-learning-curve-v1"
	assert "evaluation_max_new_tokens" not in plan["recipe"]
	assert plan["cells"] == [{"n": n, "seed": seed} for n in plan["sizes"] for seed in (42, 43, 44)]
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


def test_math_curve(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
	from distillation.learning_curve import math_family, plan_math_curve
	from distillation.results_index import student_rows
	from distillation.tasks import GSM8K_EVAL_SUFFIX

	dev, _, suite_path = inputs
	rows = [json.loads(line) for line in dev.read_text().splitlines()[:4]]
	for i, row in enumerate(rows):
		row["task"] = "gsm8k"
		row["source"]["group_id"] = math_family(f"training {i}")
		row["prompt"][0]["content"] = f"training {i}{GSM8K_EVAL_SUFFIX}"
		row["completion"][0]["content"] = "work\n#### 2"
	ledger = tmp_path / "math.jsonl"
	ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
	gold = tmp_path / "gold.jsonl"
	gold.write_text(
		"".join(
			json.dumps(
				{"example_id": row["example_id"], "completion": f"gold {row['example_id']}\n#### 2"}
			)
			+ "\n"
			for row in rows
		)
	)
	suite = FrozenSuite.model_validate_json(suite_path.read_text())
	suite.sources[0].benchmark = "gsm8k"
	suite.sources[0].selected = 2
	suite.items = suite.items[:2]
	for i, item in enumerate(suite.items):
		item.benchmark = "gsm8k"
		item.metric = "numeric_exact_match"
		item.question = f"evaluation {i}"
		item.messages = [Message(role="user", content=f"evaluation {i}")]
		item.answer = "2"
	suite.sha256 = suite.digest()
	math_suite = tmp_path / "math-suite.json"
	math_suite.write_text(suite.model_dump_json())
	cli = CliRunner().invoke(
		app,
		[
			"learning-curve",
			"plan-math",
			"--output",
			str(tmp_path / "math-cli"),
			"--ledger",
			str(ledger),
			"--suite",
			str(math_suite),
			"--gold",
			str(gold),
			"--n",
			"2",
			"--seed",
			"1",
		],
	)
	assert cli.exit_code == 0, cli.output
	output = tmp_path / "math-run"
	plan = plan_math_curve(output, ledger, math_suite, gold_path=gold, sizes=(2, 4), seeds=(1, 2))
	assert plan["version"] == "math-learning-curve-v1"
	assert plan["recipe"]["evaluation_max_new_tokens"] == 512
	assert plan["recipe"]["max_length"] == 2048
	assert plan["recipe"]["overlength_policy"] == "error_before_training"
	assert json.loads((output / "plan.json").read_text()) == plan
	assert len(plan["cells"]) == 8
	teacher = json.loads((output / "subset-2.json").read_text())
	gold_rows = json.loads((output / "gold-subset-2.json").read_text())
	assert [r["example_id"] for r in teacher] == [r["example_id"] for r in gold_rows]
	assert all(
		r["sft"]["completion"][0]["content"] == f"gold {r['example_id']}\n#### 2" for r in gold_rows
	)
	calls: list[str] = []

	def runner(path: Path, recipe: dict[str, Any], n: int | None, seed: int) -> dict[str, Any]:
		calls.append(recipe.get("_arm", "teacher"))
		return {"test0": int(n is not None), "test1": 0}

	def too_long(path: Path, recipe: dict[str, Any], files: list[str]) -> None:
		calls.append("preflight:" + ",".join(files))
		raise ValueError("training response would be truncated")

	with pytest.raises(ValueError, match="truncated"):
		execute_curve(output, runner, too_long)
	assert calls == ["preflight:subset-4.json,gold-subset-4.json"]
	calls.clear()
	assert execute_curve(output, runner, lambda *args: calls.append("preflight")) == {
		"completed": 9,
		"skipped": 0,
	}
	assert calls[0] == "preflight" and calls.count("gold") == 4
	assert execute_curve(output, runner) == {"completed": 0, "skipped": 9}
	summary = summarize_curve(output, replicates=100)
	assert len(summary["gold_results"]) == 2
	index = student_rows(output, [])
	assert {row["arm"] for row in index} == {"gold", "undefended"}
	assert {row["student_task"] for row in index} == {"gsm8k"}
	rows[0]["source"]["group_id"] = math_family("evaluation 0")
	ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
	with pytest.raises(ValueError, match="source family"):
		plan_math_curve(tmp_path / "overlap", ledger, math_suite, sizes=(2,))
	# An evaluation question in another wrapper, without group_id, is refused by template.
	del rows[0]["source"]["group_id"]
	rows[0]["prompt"][0]["content"] = "Question: evaluation 0\nAnswer with a number."
	ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
	with pytest.raises(ValueError, match="task builder template"):
		plan_math_curve(tmp_path / "wrapped", ledger, math_suite, sizes=(2,))


def _gsm8k_suite(path: Path, inputs: tuple[Path, Path, Path], items: int = 40) -> Path:
	suite = FrozenSuite.model_validate_json(inputs[2].read_text())
	suite.sources[0].benchmark = "gsm8k"
	suite.sources[0].selected = items
	suite.items = suite.items[:items]
	for i, item in enumerate(suite.items):
		item.benchmark = "gsm8k"
		item.metric = "numeric_exact_match"
		item.question = f"evaluation {i}"
		item.messages = [Message(role="user", content=f"evaluation {i}")]
		item.answer = "2"
	suite.sha256 = suite.digest()
	path.write_text(suite.model_dump_json())
	return path


def _problem_scores(path: Path, role: str, suite: Path, correct: int) -> Path:
	from distillation.baseline_gates import SCORES_VERSION, read_items, suite_digest

	items = read_items(suite)
	scores = {item.id: int(i < correct) for i, item in enumerate(items)}
	path.write_text(
		json.dumps(
			{
				"version": SCORES_VERSION,
				"role": role,
				"model": MATH_RECIPE["base_model"] if role == "base" else "teacher",
				"revision": MATH_RECIPE["revision"] if role == "base" else "t1",
				"suite_sha256": suite_digest(items),
				"scores": scores,
			}
		)
	)
	return path


def _headroom(tmp_path: Path, suite: Path, teacher: int, margin: float = 10) -> dict[str, Any]:
	from distillation.baseline_gates import headroom_gate

	return headroom_gate(
		_problem_scores(tmp_path / f"base-{teacher}.json", "base", suite, 10),
		_problem_scores(tmp_path / f"teacher-{teacher}.json", "teacher", suite, teacher),
		suite,
		margin_points=margin,
		replicates=1000,
	)


def _good_cell(path: Path, recipe: dict[str, Any], n: int | None, seed: int) -> dict[str, Any]:
	"""A stand-in runner: the base gets 10 of 40 right and each student 25."""
	correct = 10 if n is None else 25
	return {f"test{i}": int(i < correct) for i in range(40)}


def _curve(
	tmp_path: Path,
	inputs: tuple[Path, Path, Path],
	runner: Any = _good_cell,
	seeds: tuple[int, ...] = (42, 43, 44),
	gold: bool = True,
	policy: str = "clean",
) -> tuple[Path, Path, Path]:
	"""A planned and scored stand-in curve, with its headroom gate and acquisition manifest."""
	from distillation.acquisition_manifest import build_manifest, write_manifest
	from distillation.baseline_gates import write_gate
	from distillation.learning_curve import math_family, plan_math_curve
	from distillation.tasks import GSM8K_EVAL_SUFFIX

	tmp_path.mkdir(exist_ok=True)
	suite = _gsm8k_suite(tmp_path / "gsm8k-suite.json", inputs)
	template = json.loads(inputs[1].read_text().splitlines()[0])
	rows = []
	for i in range(1000):
		row = json.loads(json.dumps(template))
		row["example_id"] = f"train{i}"
		row["task"] = "gsm8k"
		row["source"]["group_id"] = math_family(f"training {i}")
		row["prompt"][0]["content"] = f"training {i}{GSM8K_EVAL_SUFFIX}"
		row["completion"][0]["content"] = "work\n#### 2"
		rows.append(row)
	ledger = tmp_path / "gsm8k-ledger.jsonl"
	ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
	gold_path = tmp_path / "gsm8k-gold.jsonl"
	gold_path.write_text(
		"".join(
			json.dumps({"example_id": row["example_id"], "completion": "#### 2"}) + "\n"
			for row in rows
		)
	)
	curve = tmp_path / "curve"
	plan_math_curve(curve, ledger, suite, gold_path=gold_path if gold else None, seeds=seeds)
	execute_curve(curve, runner)
	headroom = tmp_path / "headroom.json"
	write_gate(_headroom(tmp_path, suite, 30), headroom)
	manifest = tmp_path / "manifest.json"
	write_manifest(build_manifest([ledger], response_policy=policy), manifest)
	return curve, headroom, manifest


def test_headroom_gate_stores_the_margin_and_fails_with_a_reason(
	inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
	suite = _gsm8k_suite(tmp_path / "suite-gsm8k.json", inputs)
	passed = _headroom(tmp_path, suite, 30)
	assert passed["passed"] is True and passed["status"] == "passed"
	assert passed["results"]["headroom"]["gain_pp"] == pytest.approx(50)
	assert passed["parameters"]["margin_points"] == 10
	assert "not established" in passed["parameters"]["margin_status"]
	assert set(passed["inputs"]) == {"base_scores", "teacher_scores", "suite"}
	short = _headroom(tmp_path, suite, 30, margin=60)
	assert short["passed"] is False and "below the declared 60" in short["reasons"][0]
	assert short["action"].startswith("Change the domain")
	# Exactly at the margin passes; identical scores have no interval excluding zero.
	assert _headroom(tmp_path, suite, 14)["passed"] is True
	flat = _headroom(tmp_path, suite, 10, margin=0.01)
	assert flat["passed"] is False and any("exclude zero" in r for r in flat["reasons"])
	with pytest.raises(ValueError, match="margin"):
		_headroom(tmp_path, suite, 30, margin=0)
	other = _problem_scores(tmp_path / "other.json", "teacher", suite, 30)
	other.write_text(json.dumps({**json.loads(other.read_text()), "suite_sha256": "x"}))
	from distillation.baseline_gates import headroom_gate

	with pytest.raises(ValueError, match="different suite"):
		headroom_gate(tmp_path / "base-30.json", other, suite, margin_points=10)


def test_clean_gate_passes_only_when_the_gain_repeats(
	inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
	from distillation.baseline_gates import FIX_LEARNING, clean_gate

	curve, headroom, manifest = _curve(tmp_path / "good", inputs)
	report = clean_gate(curve, headroom, manifest, replicates=1000)
	assert report["passed"] is True, report["reasons"]
	assert [row["n"] for row in report["results"]["clean"]] == [300, 1000]
	assert report["results"]["clean"][0]["seeds"]["43"]["gain_pp"] == pytest.approx(37.5)
	assert report["gold_diagnostic"]["learned"] is True
	assert "scores/gold-n1000-s44.json" in report["inputs"]

	def one_seed_flat(
		path: Path, recipe: dict[str, Any], n: int | None, seed: int
	) -> dict[str, Any]:
		correct = 10 if n is None or (n == 300 and seed == 43) else 25
		return {f"test{i}": int(i < correct) for i in range(40)}

	curve, headroom, manifest = _curve(tmp_path / "flat", inputs, runner=one_seed_flat)
	failed = clean_gate(curve, headroom, manifest, replicates=1000)
	assert failed["status"] == "failed" and failed["action"] == FIX_LEARNING
	assert failed["reasons"][0].startswith("teacher n=300, seed 43")
	# A missing cell cannot pass, and a gold-less plan lacks the diagnostic.
	curve, headroom, manifest = _curve(tmp_path / "partial", inputs)
	(curve / "scores" / "n1000-s44.json").unlink()
	assert any("missing" in r for r in clean_gate(curve, headroom, manifest)["reasons"])
	curve, headroom, manifest = _curve(tmp_path / "nogold", inputs, gold=False)
	nogold = clean_gate(curve, headroom, manifest)
	assert not nogold["passed"] and any("gold" in r for r in nogold["reasons"])


def test_clean_gate_refuses_one_seed_as_development_evidence(
	inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
	from distillation.baseline_gates import ONE_SEED, clean_gate

	curve, headroom, manifest = _curve(tmp_path, inputs, seeds=(42,))
	report = clean_gate(curve, headroom, manifest)
	assert report["passed"] is False and report["status"] == "development-only"
	assert report["reasons"] == [ONE_SEED]
	assert report["results"]["clean"][0]["pooled"]["positive_excluding_zero"] is True


def test_clean_gate_needs_passing_headroom_and_a_clean_manifest(
	inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
	from distillation.baseline_gates import clean_gate, write_gate

	curve, headroom, manifest = _curve(tmp_path / "policy", inputs, policy="watermarked")
	reasons = clean_gate(curve, headroom, manifest)["reasons"]
	assert any("response policy" in r for r in reasons)
	curve, headroom, manifest = _curve(tmp_path / "head", inputs)
	failed = tmp_path / "failed-headroom.json"
	suite = tmp_path / "head" / "gsm8k-suite.json"
	write_gate(_headroom(tmp_path, suite, 30, margin=90), failed)
	reasons = clean_gate(curve, failed, manifest)["reasons"]
	assert reasons == ["the headroom gate did not pass"]


def test_gate_report_is_hashed_indexed_and_never_overwritten(
	inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
	from distillation.acquisition_manifest import ManifestError
	from distillation.baseline_gates import clean_gate, load_gate, write_gate
	from distillation.results_index import read_index

	curve, headroom, manifest = _curve(tmp_path, inputs)
	output, index = tmp_path / "gate.json", tmp_path / "index.tsv"
	result = CliRunner().invoke(
		app,
		[
			"baseline",
			"clean",
			"--curve",
			str(curve),
			"--headroom",
			str(headroom),
			"--manifest",
			str(manifest),
			"--output",
			str(output),
			"--index",
			str(index),
		],
	)
	assert result.exit_code == 0, result.output
	report = load_gate(output)
	(row,) = read_index(index)
	assert (row["kind"], row["attack"], row["status"]) == ("gate", "phase1-clean", "passed")
	assert row["gate_report_sha256"] == digest(output)
	assert row["gate_inputs_sha256"] == report["inputs_sha256"]
	assert row["run_id"] == f"gate:{report['inputs_sha256'][:24]}"
	assert row["student_revision"] == report["student_revision"]
	with pytest.raises(ManifestError, match="overwritten"):
		write_gate(clean_gate(curve, headroom, manifest), output)
	output.write_text(json.dumps({**report, "passed": False}))
	with pytest.raises(ValueError, match="intact"):
		load_gate(output)


def test_transfer_refuses_suppression_unless_the_gate_covers_this_student(
	inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
	from distillation.baseline_gates import clean_gate, covering_gate, plan_identity, write_gate

	curve, headroom, manifest = _curve(tmp_path, inputs)
	passing = clean_gate(curve, headroom, manifest)
	identity = plan_identity(curve)
	assert covering_gate(passing, identity).covered is True
	for field in ("task", "base_model", "base_revision", "student_revision"):
		other = covering_gate(passing, {**identity, field: "other"})
		assert other.covered is False and "different" in str(other.reason)
	assert "headroom" in str(covering_gate(_read(headroom), identity).reason)
	assert "not a passing" in str(covering_gate({**passing, "passed": False}, identity).reason)

	gate = tmp_path / "gate.json"
	write_gate(passing, gate)
	counts = {"teacher": 9, "base": 3, "clean": 7, "defended": 5}
	results = tmp_path / "results.json"
	results.write_text(
		json.dumps(
			{
				role: {f"f{i}": [True] * k + [False] * (10 - k) for i in range(6)}
				for role, k in counts.items()
			}
		)
	)
	args = ["learning-curve", "transfer", "--results", str(results)]
	refused = CliRunner().invoke(app, args)
	assert refused.exit_code == 0, refused.output
	assert json.loads(refused.output)["suppression"]["value"] is None
	covered = CliRunner().invoke(app, [*args, "--gate", str(gate), "--student-plan", str(curve)])
	assert covered.exit_code == 0, covered.output
	assert json.loads(covered.output)["suppression"]["value"] == pytest.approx(0.5)
	assert CliRunner().invoke(app, [*args, "--gate", str(gate)]).exit_code != 0


def _read(path: Path) -> dict[str, Any]:
	value: dict[str, Any] = json.loads(path.read_text())
	return value


def test_evaluations_write_score_files_the_gate_reads(
	inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
	from distillation.baseline_gates import evaluate_base, evaluate_teacher, headroom_gate

	suite = _gsm8k_suite(tmp_path / "suite-gsm8k.json", inputs)
	seen: list[dict[str, Any]] = []

	def teacher(request: httpx.Request) -> httpx.Response:
		seen.append(json.loads(request.content))
		return httpx.Response(
			200,
			json={
				"id": "1",
				"provider": "fake",
				"model": "teacher",
				"generated_at": "2026-01-01T00:00:00Z",
				"message": {"role": "assistant", "content": "work\n#### 2"},
				"finish_reason": "stop",
				"endpoint": "fake",
			},
		)

	base_path, teacher_path = tmp_path / "base.json", tmp_path / "teacher.json"
	evaluate_base(suite, base_path, runner=_good_cell)
	evaluate_teacher(
		suite,
		teacher_path,
		url="http://teacher",
		model="teacher",
		revision="t1",
		transport=httpx.MockTransport(teacher),
	)
	assert seen[0]["enable_thinking"] is False and seen[0]["max_output_tokens"] == 512
	report = headroom_gate(base_path, teacher_path, suite, margin_points=10, replicates=1000)
	assert report["results"]["teacher_accuracy"] == 1 and report["passed"] is True

	def harness(request: httpx.Request) -> httpx.Response:
		return httpx.Response(200, headers={"x-harness-decision": "allow"}, json={})

	failed = tmp_path / "failed.json"
	with pytest.raises(ValueError, match="no scores written"):
		evaluate_teacher(
			suite,
			failed,
			url="http://teacher",
			model="teacher",
			revision="t1",
			transport=httpx.MockTransport(harness),
		)
	assert not failed.exists()


def test_clean_collection_selects_deterministically_and_feeds_the_manifest(
	tmp_path: Path,
) -> None:
	from distillation.acquisition_manifest import build_manifest
	from distillation.baseline_gates import collect_clean
	from distillation.tasks import GSM8K_EVAL_SUFFIX

	pool = tmp_path / "attacker.jsonl"
	pool.write_text(
		"".join(
			json.dumps(
				{
					"example_id": f"p{i}",
					"split": "train",
					"task": "gsm8k",
					"source": {"corpus": "fake", "source_id": str(i), "license": "test"},
					"prompt": [{"role": "user", "content": f"q{i}{GSM8K_EVAL_SUFFIX}"}],
				}
			)
			+ "\n"
			for i in range(5)
		)
	)

	def teacher(request: httpx.Request) -> httpx.Response:
		assert json.loads(request.content)["enable_thinking"] is False
		return httpx.Response(
			200,
			json={
				"id": "1",
				"provider": "fake",
				"model": "teacher",
				"generated_at": "2026-01-01T00:00:00Z",
				"message": {"role": "assistant", "content": "work\n#### 2"},
				"finish_reason": "stop",
				"endpoint": "fake",
			},
		)

	options: dict[str, Any] = {
		"count": 3,
		"seed": 7,
		"url": "http://teacher",
		"terms_review_id": "review",
		"transport": httpx.MockTransport(teacher),
	}
	ledger = tmp_path / "collection" / "ledger.jsonl"
	assert collect_clean(pool, ledger, **options) == (3, 0)
	assert collect_clean(pool, ledger, **options) == (0, 3)
	manifest = build_manifest([ledger], response_policy="clean")
	assert manifest["totals"]["examples"] == 3
	with pytest.raises(ValueError, match="fewer than 9"):
		collect_clean(pool, ledger, **{**options, "count": 9})
