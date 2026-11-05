"""CPU-only checkpoint learning contracts with a contract-shaped export and injected runner."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from distillation import checkpoint_cli, checkpoint_learning
from distillation.benchmarks.models import BenchmarkItem, DatasetSource, FrozenSuite
from distillation.checkpoint_learning import (
	execute_checkpoint_curve,
	plan_checkpoint_curve,
	summarize_checkpoint_curve,
)
from distillation.io import canonical_json
from distillation.learning_curve import digest, math_family
from distillation.schemas import Message
from distillation.tasks import GSM8K_EVAL_SUFFIX
from typer.testing import CliRunner

SUITE_SIZE = 4


def _record(
	name: str, status: str = "accepted", finish: str = "stop", **fields: Any
) -> dict[str, Any]:
	question = f"training {name}"
	row: dict[str, Any] = {
		"example_id": name,
		"attempt_id": f"attempt-{name}",
		"split": "train",
		"task": "gsm8k",
		"source": {
			"corpus": "fake",
			"source_id": name,
			"license": "test",
			"group_id": math_family(question),
		},
		"prompt": [{"role": "user", "content": question + GSM8K_EVAL_SUFFIX}],
		"completion": [{"role": "assistant", "content": f"work {name}\n#### 2"}],
		"teacher": {
			"provider": "fake",
			"model": "fake",
			"endpoint": "fake",
			"generated_at": "2026-01-01T00:00:00Z",
			"parameters": {},
			"terms_review_id": "test",
		},
		"finish_reason": finish,
		"usage": {},
		"quality": {"status": status, "checks": [], "reviewer": "test"},
		"attempts": [
			{
				"attempt_id": f"attempt-{name}",
				"attempted_at": "2026-01-01T00:00:00Z",
				"status": "succeeded",
			}
		],
	}
	row.update(fields)
	return row


RECORDS = {
	"e0": _record("e0", "pending"),
	"e1": _record("e1", "rejected"),
	"e2": _record("e2"),
	"e3": _record("e3"),
	"e4": _record("e4", finish="length"),
	"e5": _record("e5"),
	"h6": _record("h6", split="validation"),
}
# Arrival order differs from id order and from the later snapshots' ledger order.
SNAPSHOTS = [
	("zero", "response-0", 1.0, []),
	("early", "response-2", 5.0, ["e2", "e0"]),
	("late", "first_restriction_observed", 9.0, ["e2", "e0", "e1", "e3", "e4", "h6"]),
	("end", "end", 12.0, ["e2", "e0", "e1", "e3", "e4", "h6", "e5"]),
]


def build_export(root: Path, snapshots: Any = SNAPSHOTS, records: Any = RECORDS) -> Path:
	"""Write a manifest and ledgers shaped like the exporter contract."""
	entries: list[dict[str, Any]] = []
	for name, kind, cutoff, ids in snapshots:
		ledger = root / "ledgers" / f"{name}.jsonl"
		ledger.parent.mkdir(parents=True, exist_ok=True)
		ledger.write_text("".join(json.dumps(records[i]) + "\n" for i in ids))
		entries.append(
			{
				"id": name,
				"kind": kind,
				"reached": True,
				"cutoff_seconds": cutoff,
				"ledger_file": f"ledgers/{name}.jsonl",
				"ledger_sha256": digest(ledger),
				"example_ids": ids,
				"record_count": len(ids),
				"raw_successful_responses": len(ids) + 1,
				"timing_source": "client_receipt",
			}
		)
	entries.append(
		{
			"id": "response-100",
			"kind": "response_count",
			"reached": False,
			"cutoff_seconds": None,
			"ledger_file": None,
			"ledger_sha256": None,
			"example_ids": [],
			"record_count": 0,
		}
	)
	inputs = {}
	for name in ("contract", "summary", "events", "ledger"):
		source = root / f"source-{name}.json"
		source.write_text("{}")
		inputs[name] = {"path": str(source.resolve()), "sha256": digest(source)}
	manifest: dict[str, Any] = {
		"version": "campaign-checkpoints-v1",
		"inputs": inputs,
		"checkpoints": entries,
		"interventions": {"first_restriction": {"status": "reached", "bound": "upper"}},
		"limitations": ["fixture"],
	}
	manifest["digest"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
	(root / "manifest.json").write_text(json.dumps(manifest))
	return root


def fixture_loader(root: Path) -> dict[str, Any]:
	manifest: dict[str, Any] = json.loads((root / "manifest.json").read_text())
	body = {k: v for k, v in manifest.items() if k != "digest"}
	if hashlib.sha256(canonical_json(body).encode()).hexdigest() != manifest["digest"]:
		raise ValueError("export manifest digest mismatch")
	return manifest


def build_suite(path: Path, benchmark: str = "gsm8k", metric: str = "numeric_exact_match") -> Path:
	suite = FrozenSuite(
		seed=42,
		created_at=datetime.now(UTC),
		sources=[
			DatasetSource(
				benchmark=benchmark,
				repository="fake",
				revision="fake",
				configuration="fake",
				split="development",
				license="test",
				url="fake",
				available=SUITE_SIZE,
				selected=SUITE_SIZE,
			)
		],
		items=[
			BenchmarkItem(
				id=f"test{i}",
				benchmark=benchmark,
				category="math",
				question=f"evaluation {i}",
				messages=[Message(role="user", content=f"evaluation {i}")],
				metric=metric,  # type: ignore[arg-type]
				answer="2",
			)
			for i in range(SUITE_SIZE)
		],
	)
	suite.sha256 = suite.digest()
	path.write_text(suite.model_dump_json())
	return path


@pytest.fixture
def export(tmp_path: Path) -> Path:
	return build_export(tmp_path / "export")


@pytest.fixture
def suite(tmp_path: Path) -> Path:
	return build_suite(tmp_path / "suite.json")


def plan(tmp_path: Path, export: Path, suite: Path, **options: Any) -> dict[str, Any]:
	options.setdefault("target_accuracy", 0.7)
	options.setdefault("seeds", (1, 2))
	return plan_checkpoint_curve(tmp_path / "plan", export, suite, loader=fixture_loader, **options)


def stage_ids(output: Path, name: str, n: int) -> list[str]:
	rows = json.loads((output / "stages" / name / f"subset-{n}.json").read_text())
	return [r["example_id"] for r in rows]


class Recorder:
	"""Learning improves with training size and one seed does slightly better."""

	def __init__(self) -> None:
		self.calls: list[tuple[str, int | None, int]] = []

	def __call__(
		self, stage: Path, recipe: dict[str, Any], n: int | None, seed: int
	) -> dict[str, Any]:
		assert (stage / "suite.json").is_file()
		if n is not None:
			assert (stage / f"subset-{n}.json").is_file()
		self.calls.append((stage.name, n, seed))
		ids = [i["id"] for i in json.loads((stage / "suite.json").read_text())]
		solved = 0 if n is None else n + (seed == 2)
		return {item: int(index < solved) for index, item in enumerate(ids)}


def test_plan_freezes_chronological_eligible_sets(
	tmp_path: Path, export: Path, suite: Path
) -> None:
	frozen = plan(tmp_path, export, suite)
	output = tmp_path / "plan"
	by_id = {c["id"]: c for c in frozen["checkpoints"]}
	assert by_id["zero"]["status"] == "base" and by_id["zero"]["trainable_count"] == 0
	assert by_id["response-100"]["status"] == "censored"
	assert "stage" not in by_id["response-100"] and "stage" not in by_id["zero"]
	assert (by_id["early"]["acquired_count"], by_id["early"]["trainable_count"]) == (2, 1)
	assert by_id["early"]["filtered"] == {"pending_excluded": 1}
	assert by_id["late"]["filtered"] == {
		"held_out_split": 1,
		"nonterminal_finish": 1,
		"pending_excluded": 1,
		"rejected": 1,
	}
	assert by_id["end"]["evidence"] == {
		"raw_successful_responses": 8,
		"timing_source": "client_receipt",
	}
	# Exactly the eligible rows in arrival order, nested by time; pending and rejected excluded.
	assert stage_ids(output, "early", 1) == ["e2"]
	assert stage_ids(output, "late", 2) == ["e2", "e3"]
	assert stage_ids(output, "end", 3) == ["e2", "e3", "e5"]
	assert frozen["timeline"] == ["zero", "early", "late", "end"]
	assert len(frozen["cells"]) == 6 and frozen["target_accuracy"] == 0.7
	assert frozen["export"]["manifest_sha256"] == digest(export / "manifest.json")
	assert frozen["interventions"] == {"first_restriction": {"status": "reached", "bound": "upper"}}
	assert json.loads((output / "plan.json").read_text()) == frozen


def test_default_loader_is_the_exporter_verification(
	tmp_path: Path, export: Path, suite: Path
) -> None:
	frozen = plan_checkpoint_curve(
		tmp_path / "real", export, suite, target_accuracy=0.7, seeds=(1,)
	)
	assert frozen["export"]["manifest_version"] == "campaign-checkpoints-v1"
	ledger = export / "ledgers" / "end.jsonl"
	ledger.write_text(ledger.read_text() + "\n")
	with pytest.raises(ValueError, match="digest mismatch"):
		plan_checkpoint_curve(tmp_path / "real2", export, suite, target_accuracy=0.7)


def test_pending_requires_explicit_inclusion_and_rejected_never(
	tmp_path: Path, export: Path, suite: Path
) -> None:
	plan(tmp_path, export, suite, include_pending=True)
	output = tmp_path / "plan"
	assert stage_ids(output, "early", 2) == ["e2", "e0"]
	assert stage_ids(output, "end", 4) == ["e2", "e0", "e3", "e5"]


def test_end_to_end_resume_and_summary(tmp_path: Path, export: Path, suite: Path) -> None:
	plan(tmp_path, export, suite)
	output = tmp_path / "plan"
	runner = Recorder()
	checked: list[list[str]] = []
	result = execute_checkpoint_curve(
		output, runner, lambda path, recipe, files: checked.append(files)
	)
	assert result == {"completed": 7, "skipped": 0}
	assert checked == [
		["stages/early/subset-1.json", "stages/late/subset-2.json", "stages/end/subset-3.json"]
	]
	assert runner.calls[0] == ("base", None, 1)
	assert {stage for stage, _, _ in runner.calls} == {"base", "early", "late", "end"}
	assert execute_checkpoint_curve(output, runner) == {"completed": 0, "skipped": 7}
	assert len(runner.calls) == 7

	summary = summarize_checkpoint_curve(output)
	rows = {r["id"]: r for r in summary["checkpoints"]}
	assert summary["base_accuracy"] == 0 and not summary["base_meets_target"]
	assert rows["zero"]["source"] == "base_model" and rows["zero"]["mean_gain_pp"] == 0
	assert rows["response-100"]["status"] == "censored" and rows["response-100"]["seeds"] is None
	assert [s["accuracy"] for s in rows["end"]["seeds"]] == [0.75, 1.0]
	assert rows["end"]["seeds"][0]["improved"] == 3 and rows["end"]["seeds"][0]["regressed"] == 0
	assert rows["end"]["seeds_reaching_target"] == 2
	assert rows["late"]["mean_accuracy"] == 0.625 and not rows["late"]["mean_reaches_target"]
	assert rows["late"]["seed_sd"] is not None
	assert summary["first_reaching_target"]["id"] == "end"
	assert summary["first_reaching_target"]["acquired_count"] == 7
	assert summary["first_reaching_target"]["trainable_count"] == 3
	assert summary["target_status"] == "observed"
	assert summary["censored_checkpoints"] == ["response-100"]
	assert "prevent" not in " ".join(k for k in summary if "verdict" in k)
	assert "not evidence that any defense prevents" in summary["claim_boundary"]
	bindings = summary["bindings"]
	assert bindings["plan_sha256"] == digest(output / "plan.json")
	assert bindings["export_manifest_sha256"] == digest(export / "manifest.json")
	assert set(bindings["scores"]) == {
		"base",
		*(f"{c}-s{s}" for c in ("early", "late", "end") for s in (1, 2)),
	}
	assert json.loads((output / "summary.json").read_text()) == summary


def test_target_not_observed_is_not_a_prevention_claim(
	tmp_path: Path, export: Path, suite: Path
) -> None:
	plan(tmp_path, export, suite, target_accuracy=1.0, seeds=(1,))
	output = tmp_path / "plan"
	execute_checkpoint_curve(output, Recorder())
	summary = summarize_checkpoint_curve(output)
	assert summary["first_reaching_target"] is None
	assert summary["target_status"] == "not_observed_among_reached_checkpoints"
	assert summary["censored_checkpoints"] == ["response-100"]


def test_interrupted_cells_resume_from_verified_scores(
	tmp_path: Path, export: Path, suite: Path
) -> None:
	plan(tmp_path, export, suite)
	output = tmp_path / "plan"
	runner = Recorder()

	def flaky(stage: Path, recipe: dict[str, Any], n: int | None, seed: int) -> dict[str, Any]:
		if stage.name == "late" and seed == 1:
			raise RuntimeError("gpu lost")
		return runner(stage, recipe, n, seed)

	with pytest.raises(RuntimeError):
		execute_checkpoint_curve(output, flaky)
	assert (output / "scores" / "early-s2.json").exists()
	assert not (output / "scores" / "late-s1.json").exists()
	with pytest.raises(ValueError, match="missing score"):
		summarize_checkpoint_curve(output)
	runner.calls.clear()
	assert execute_checkpoint_curve(output, runner) == {"completed": 4, "skipped": 3}
	assert [stage for stage, _, _ in runner.calls] == ["late", "late", "end", "end"]


def test_changed_artifacts_scores_and_plans_are_refused(
	tmp_path: Path, export: Path, suite: Path
) -> None:
	plan(tmp_path, export, suite)
	output = tmp_path / "plan"
	execute_checkpoint_curve(output, Recorder())
	score = output / "scores" / "end-s1.json"
	original = score.read_text()
	tampered = json.loads(original)
	tampered["scores"]["test0"] = 0
	score.write_text(json.dumps(tampered))
	with pytest.raises(ValueError, match="content hash"):
		execute_checkpoint_curve(output, Recorder())
	with pytest.raises(ValueError, match="content hash"):
		summarize_checkpoint_curve(output)
	moved = json.loads(original)
	moved["seed"] = 2
	score.write_text(json.dumps(moved))
	with pytest.raises(ValueError, match="different plan or cell"):
		execute_checkpoint_curve(output, Recorder())
	score.write_text("{not json")
	with pytest.raises(ValueError, match="unusable score artifact"):
		summarize_checkpoint_curve(output)
	score.write_text(original)
	frozen = json.loads((output / "plan.json").read_text())
	frozen["target_accuracy"] = 0.1
	(output / "plan.json").write_text(json.dumps(frozen, indent="\t"))
	with pytest.raises(ValueError, match="different plan or cell"):
		summarize_checkpoint_curve(output)
	(output / "plan.json").write_text(json.dumps({**frozen, "version": "other"}))
	with pytest.raises(ValueError, match="not a checkpoint learning plan"):
		execute_checkpoint_curve(output, Recorder())


def test_stage_and_input_tampering_is_refused_before_training(
	tmp_path: Path, export: Path, suite: Path
) -> None:
	plan(tmp_path, export, suite)
	output = tmp_path / "plan"
	subset = output / "stages" / "late" / "subset-2.json"
	original = subset.read_text()
	subset.write_text(original.replace("work e3", "work other"))
	runner = Recorder()
	with pytest.raises(ValueError, match="artifact hash mismatch"):
		execute_checkpoint_curve(output, runner)
	assert runner.calls == []
	subset.write_text(original)
	ledger = export / "ledgers" / "end.jsonl"
	ledger.write_text(ledger.read_text() + "\n")
	with pytest.raises(ValueError, match="ledger changed"):
		execute_checkpoint_curve(output, runner)
	assert runner.calls == []


def test_ledger_and_manifest_corruption_is_refused(
	tmp_path: Path, export: Path, suite: Path
) -> None:
	ledger = export / "ledgers" / "early.jsonl"
	ledger.write_text(ledger.read_text() + "\n")
	with pytest.raises(ValueError, match="ledger hash mismatch"):
		plan(tmp_path, export, suite)
	assert not (tmp_path / "plan").exists()
	manifest = json.loads((export / "manifest.json").read_text())
	manifest["limitations"] = ["edited"]
	(export / "manifest.json").write_text(json.dumps(manifest))
	with pytest.raises(ValueError, match="digest mismatch"):
		plan(tmp_path, export, suite)


def test_missing_or_changed_sources_refused_before_resume(
	tmp_path: Path, export: Path, suite: Path
) -> None:
	plan(tmp_path, export, suite)
	output = tmp_path / "plan"
	runner = Recorder()
	source = export / "source-events.json"
	original = source.read_text()
	source.write_text("changed")
	with pytest.raises(ValueError, match="input changed"):
		execute_checkpoint_curve(output, runner)
	source.unlink()
	with pytest.raises(ValueError, match="input is missing"):
		execute_checkpoint_curve(output, runner)
	source.write_text(original)
	ledger = export / "ledgers" / "early.jsonl"
	ledger.unlink()
	with pytest.raises(ValueError, match="ledger is missing"):
		execute_checkpoint_curve(output, runner)
	assert runner.calls == []


def test_manifest_entry_validation(tmp_path: Path, suite: Path) -> None:
	def rewrite(root: Path, edit: Any) -> None:
		manifest = json.loads((root / "manifest.json").read_text())
		edit(manifest)
		del manifest["digest"]
		manifest["digest"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
		(root / "manifest.json").write_text(json.dumps(manifest))

	def escape(manifest: dict[str, Any]) -> None:
		manifest["checkpoints"][1]["ledger_file"] = "../outside.jsonl"

	def count(manifest: dict[str, Any]) -> None:
		manifest["checkpoints"][1]["record_count"] = 5

	def unreached(manifest: dict[str, Any]) -> None:
		manifest["checkpoints"][-1]["ledger_file"] = "ledgers/early.jsonl"

	def unsafe(manifest: dict[str, Any]) -> None:
		manifest["checkpoints"][1]["id"] = "a/b"

	def reordered(manifest: dict[str, Any]) -> None:
		manifest["checkpoints"][1]["example_ids"].reverse()

	for number, (edit, message) in enumerate(
		[
			(escape, "relative and contained"),
			(count, "malformed"),
			(unreached, "no ledger"),
			(unsafe, "filesystem-safe"),
			(reordered, "do not match the manifest"),
		]
	):
		root = build_export(tmp_path / f"export{number}")
		rewrite(root, edit)
		with pytest.raises(ValueError, match=message):
			plan_checkpoint_curve(
				tmp_path / f"plan{number}",
				root,
				suite,
				target_accuracy=0.5,
				loader=fixture_loader,
			)


def test_membership_must_be_chronological(tmp_path: Path, suite: Path) -> None:
	snapshots = [
		("early", "response-2", 5.0, ["e2", "e3"]),
		("late", "end", 9.0, ["e2", "e5"]),
	]
	root = build_export(tmp_path / "export", snapshots)
	with pytest.raises(ValueError, match="not chronological"):
		plan_checkpoint_curve(
			tmp_path / "plan", root, suite, target_accuracy=0.5, loader=fixture_loader
		)


def test_record_changing_between_snapshots_is_refused(tmp_path: Path, suite: Path) -> None:
	changed = {**RECORDS, "e2b": {**RECORDS["e2"], "quality": RECORDS["e0"]["quality"]}}
	root = build_export(tmp_path / "export", [("early", "response-1", 1.0, ["e2"])], changed)
	late = root / "ledgers" / "late.jsonl"
	late.write_text(json.dumps(changed["e2b"] | {"example_id": "e2"}) + "\n")
	manifest = json.loads((root / "manifest.json").read_text())
	manifest["checkpoints"].insert(
		1,
		{
			**manifest["checkpoints"][0],
			"id": "late",
			"cutoff_seconds": 2.0,
			"ledger_file": "ledgers/late.jsonl",
			"ledger_sha256": digest(late),
		},
	)
	del manifest["digest"]
	manifest["digest"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
	(root / "manifest.json").write_text(json.dumps(manifest))
	with pytest.raises(ValueError, match="differs between checkpoint ledgers"):
		plan_checkpoint_curve(
			tmp_path / "plan", root, suite, target_accuracy=0.5, loader=fixture_loader
		)


def test_overlap_and_template_checks_cover_every_reached_checkpoint(
	tmp_path: Path, suite: Path
) -> None:
	leak = _record("leak", group_id=None)
	leak["source"].pop("group_id")
	leak["prompt"][0]["content"] = "evaluation 0" + GSM8K_EVAL_SUFFIX
	root = build_export(
		tmp_path / "leak",
		[("early", "response-1", 1.0, ["e2"]), ("end", "end", 2.0, ["e2", "leak"])],
		{**RECORDS, "leak": leak},
	)
	with pytest.raises(ValueError, match="evaluation overlap"):
		plan_checkpoint_curve(
			tmp_path / "p1", root, suite, target_accuracy=0.5, loader=fixture_loader
		)
	grouped = _record("grouped")
	grouped["source"]["group_id"] = math_family("evaluation 1")
	root = build_export(
		tmp_path / "group", [("end", "end", 2.0, ["grouped"])], {"grouped": grouped}
	)
	with pytest.raises(ValueError, match="evaluation overlap"):
		plan_checkpoint_curve(
			tmp_path / "p2", root, suite, target_accuracy=0.5, loader=fixture_loader
		)
	wrapped = _record("wrapped")
	wrapped["source"].pop("group_id")
	wrapped["prompt"][0]["content"] = "Question: evaluation 0"
	root = build_export(tmp_path / "wrap", [("end", "end", 2.0, ["wrapped"])], {"wrapped": wrapped})
	with pytest.raises(ValueError, match="task builder template"):
		plan_checkpoint_curve(
			tmp_path / "p3", root, suite, target_accuracy=0.5, loader=fixture_loader
		)
	assert not any((tmp_path / name).exists() for name in ("p1", "p2", "p3"))


def test_plan_input_contracts(tmp_path: Path, export: Path, suite: Path) -> None:
	for target in (0.0, 1.5, float("nan"), -1.0):
		with pytest.raises(ValueError, match="target_accuracy"):
			plan(tmp_path, export, suite, target_accuracy=target)
	with pytest.raises(ValueError, match="unsupported task"):
		plan(tmp_path, export, suite, task="ag_news")
	with pytest.raises(ValueError, match="unique integer seeds"):
		plan(tmp_path, export, suite, seeds=(1, 1))
	mixed = build_suite(tmp_path / "ag.json", benchmark="ag_news", metric="label_accuracy")
	with pytest.raises(ValueError, match="GSM8K numeric"):
		plan(tmp_path, export, mixed)
	with pytest.raises(ValueError, match="inside the export"):
		plan_checkpoint_curve(
			export / "plan", export, suite, target_accuracy=0.5, loader=fixture_loader
		)
	plan(tmp_path, export, suite)
	with pytest.raises(ValueError, match="new directory"):
		plan(tmp_path, export, suite)


def test_only_unreached_or_empty_checkpoints_need_no_model(tmp_path: Path, suite: Path) -> None:
	root = build_export(tmp_path / "export", [("zero", "response-0", 1.0, [])])
	frozen = plan_checkpoint_curve(
		tmp_path / "plan", root, suite, target_accuracy=0.5, seeds=(1,), loader=fixture_loader
	)
	assert frozen["cells"] == []
	runner = Recorder()
	assert execute_checkpoint_curve(tmp_path / "plan", runner) == {"completed": 1, "skipped": 0}
	assert runner.calls == [("base", None, 1)]
	summary = summarize_checkpoint_curve(tmp_path / "plan")
	assert summary["first_reaching_target"] is None
	assert summary["censored_checkpoints"] == ["response-100"]


def test_cli_group_runs_the_whole_flow(
	tmp_path: Path, export: Path, suite: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	import distillation.learning_curve_runner as gpu

	runner = Recorder()
	monkeypatch.setattr(checkpoint_learning, "load_export", fixture_loader)
	monkeypatch.setattr(gpu, "run_cell", runner)
	monkeypatch.setattr(gpu, "preflight", lambda *args: None)
	exported: list[Any] = []

	def fake_export(campaign: Path, output: Path, counts: Any) -> dict[str, Any]:
		exported.append((campaign, output, tuple(counts)))
		return {"version": "campaign-checkpoints-v1"}

	monkeypatch.setattr(checkpoint_cli, "run_export", fake_export)
	cli = CliRunner()
	output = tmp_path / "cli-plan"
	result = cli.invoke(
		checkpoint_cli.app,
		["export", "--campaign", str(tmp_path), "--output", "o", "--response-count", "5"]
		+ ["--response-count", "10"],
	)
	assert result.exit_code == 0, result.output
	assert exported == [(tmp_path, Path("o"), (5, 10))]
	missing = cli.invoke(
		checkpoint_cli.app,
		["plan", "--output", str(output), "--export", str(export)] + ["--suite", str(suite)],
	)
	assert missing.exit_code != 0 and not output.exists()
	args = ["--output", str(output)]
	planned = cli.invoke(
		checkpoint_cli.app,
		["plan", *args, "--export", str(export), "--suite", str(suite)]
		+ ["--target-accuracy", "0.7", "--seed", "1", "--include-pending"],
	)
	assert planned.exit_code == 0, planned.output
	assert json.loads(planned.output)["include_pending"] is True
	refused = cli.invoke(
		checkpoint_cli.app,
		["plan", "--output", str(tmp_path / "x"), "--export", str(export), "--suite", str(suite)]
		+ ["--target-accuracy", "0.7", "--task", "ag_news"],
	)
	assert refused.exit_code != 0
	executed = cli.invoke(checkpoint_cli.app, ["execute", *args])
	assert executed.exit_code == 0, executed.output
	assert json.loads(executed.output) == {"completed": 4, "skipped": 0}
	summarized = cli.invoke(checkpoint_cli.app, ["summary", *args])
	assert summarized.exit_code == 0, summarized.output
	assert json.loads(summarized.output)["target_accuracy"] == 0.7
