"""Qualification separates selection evidence and rejects unstable effect denominators."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from distillation.benchmarks.models import (
	BenchmarkItem,
	DatasetSource,
	FrozenSuite,
	SampleResult,
	load_suite,
	save_json,
)
from distillation.launcher import LaunchError, _gain_margins
from distillation.schemas import Message
from distillation.study_reporting import defense_effect
from distillation.study_training import primary_score
from distillation.transfer import (
	analysis_protocol,
	development_suites,
	effect_inference,
	qualification,
)


def samples(values: Sequence[float], *, clustered: bool = False) -> list[SampleResult]:
	return [
		SampleResult(
			item_id=str(index),
			benchmark="squad",
			category=str(index // 2 if clustered else index),
			response="fixture",
			prediction="fixture",
			correct=value == 1,
			f1=value,
			status="ok",
			latency_seconds=0,
		)
		for index, value in enumerate(values)
	]


def test_protocol_freezes_positive_finite_task_margins() -> None:
	assert analysis_protocol({}, ["squad", "gsm8k"])["minimum_gain_pp"] == {"squad": 2, "gsm8k": 2}
	assert analysis_protocol(_gain_margins(["squad=3"]), ["squad"])["minimum_gain_pp"] == {
		"squad": 3
	}
	for value in (0, -1, 100, float("nan"), float("inf")):
		with pytest.raises(LaunchError):
			analysis_protocol({"squad": value}, ["squad"])
	with pytest.raises(LaunchError):
		analysis_protocol({"typo": 2}, ["squad"])
	with pytest.raises(LaunchError):
		_gain_margins(["squad=2", "squad=3"])


def test_qualification_requires_lower_bound_above_practical_margin() -> None:
	base = samples([0] * 20)
	assert qualification(base, samples([0.01] * 20), 2)["passed"] is False
	assert qualification(base, samples([0.02] * 20), 2)["passed"] is False
	assert qualification(base, samples([0.1] * 20), 2)["passed"] is True
	uncertain = qualification(base, samples([1] + [0] * 19), 2)
	assert uncertain["gain_pp"] > 2
	assert uncertain["gain_ci95_pp"][0] == 0
	assert uncertain["passed"] is False
	assert qualification(samples([1] * 20), base, 2)["passed"] is False
	assert primary_score(samples([0.1] * 20), "squad") == pytest.approx(10)


@pytest.mark.parametrize(
	"values,qualified,reason",
	[
		([0.01] * 20, True, "negligible_gain"),
		([0] * 20, True, "negligible_gain"),
		([1] + [0] * 19, True, "uncertain_gain"),
		([1] * 20, False, "qualification_failed"),
	],
)
def test_invalid_ratios_keep_absolute_effects(
	values: list[float], qualified: bool, reason: str
) -> None:
	result = effect_inference(
		samples([0] * 20), samples(values), samples([0] * 20), margin=2, qualified=qualified
	)
	assert result["ratio_valid"] is False
	assert result["ratio_invalid_reason"] == reason
	assert result["retained_gain_fraction"] is None
	assert result["gain_removed_fraction_ci95"] is None
	assert result["gain_removed"] >= 0
	assert len(result["gain_removed_ci95_pp"]) == 2


def test_joint_effect_ratios_preserve_pairing_and_historical_alias() -> None:
	base, undefended, defended = samples([0.2] * 20), samples([0.6] * 20), samples([0.4] * 20)
	result = effect_inference(base, undefended, defended, margin=2, qualified=True)
	assert result["ratio_valid"] is True
	assert result["retained_gain_fraction"] == pytest.approx(0.5)
	assert result["gain_removed_fraction"] == pytest.approx(0.5)
	assert result["retained_gain_fraction_ci95"] == pytest.approx([0.5, 0.5])
	legacy = defense_effect(20, 60, 40, development_gate=True, inference=result)
	assert legacy["R_D"] == legacy["retained_gain_fraction"]
	failed_gate = defense_effect(20, 60, 40, development_gate=False, inference=result)
	assert failed_gate["retained_gain_fraction_ci95"] is None
	assert failed_gate["R_D_invalid_reason"] == "qualification_failed"
	with pytest.raises(ValueError, match="disagree"):
		defense_effect(20, 90, 40, development_gate=True, inference=result)
	assert defense_effect(20, 60, 40, development_gate=True)["R_D"] is None
	# Defended scores below base and above undefended must remain visible.
	for score, retained in ((0.1, -0.25), (0.8, 1.5)):
		result = effect_inference(base, undefended, samples([score] * 20), margin=2, qualified=True)
		assert result["retained_gain_fraction"] == pytest.approx(retained)
		assert result["gain_removed_fraction"] == pytest.approx(1 - retained)


def test_source_clusters_and_pair_validation() -> None:
	base = samples([0] * 8, clustered=True)
	result = qualification(base, samples([1] * 8, clustered=True), 2)
	assert result["clusters"] == 4
	with pytest.raises(ValueError, match="item IDs"):
		qualification(base, base[:-1], 2)
	with pytest.raises(ValueError, match="unique IDs"):
		qualification(base, base + base, 2)
	with pytest.raises(ValueError, match="metadata"):
		qualification(base, samples([1] * 8), 2)


def test_development_partition_freezes_disjoint_source_groups(tmp_path: Path) -> None:
	items = [
		BenchmarkItem(
			id=str(i),
			benchmark="squad",
			category=f"document-{i // 2}",
			question="question",
			messages=[Message(role="user", content="question")],
			metric="squad",
			answer="answer",
		)
		for i in range(12)
	]
	suite = FrozenSuite(
		seed=42,
		created_at=datetime.now(UTC),
		items=items,
		sources=[
			DatasetSource(
				benchmark="squad",
				repository="fixture",
				revision="fixture",
				configuration="fixture",
				split="development",
				license="fixture",
				url="https://example.com",
				available=12,
				selected=12,
			)
		],
	)
	suite.sha256 = suite.digest()
	source = tmp_path / "development.json"
	save_json(source, suite)
	paths = development_suites(source, tmp_path / "splits")
	selection, qualification_suite = (
		load_suite(paths[key]) for key in ("selection", "qualification")
	)
	assert {i.id for i in selection.items}.isdisjoint(i.id for i in qualification_suite.items)
	assert {i.category for i in selection.items}.isdisjoint(
		i.category for i in qualification_suite.items
	)
	assert len(selection.items) + len(qualification_suite.items) == len(items)
	assert development_suites(source, tmp_path / "splits") == paths
	items[0].answer = "changed"
	suite.sha256 = suite.digest()
	save_json(source, suite)
	with pytest.raises(LaunchError, match="new --run-dir"):
		development_suites(source, tmp_path / "splits")


def test_task_selection_cannot_promote_selection_gain_into_qualification(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	import json

	from distillation.benchmarks.models import EvaluationRun, Slot, load_config
	from distillation.launcher import Launcher, TaskPipeline
	from distillation.study import Study
	from distillation.study_training import select_recipe
	from distillation.tasks import build_task, task_directory

	monkeypatch.chdir(tmp_path)
	build_task(
		task="ag_news",
		output_dir=task_directory("ag_news"),
		rows={
			"train": [{"text": f"Article {i}", "label": i % 4} for i in range(500)],
			"test": [{"text": "Final article", "label": 0}],
		},
	)
	manifest_path = task_directory("ag_news") / "manifest.json"
	manifest = json.loads(manifest_path.read_text())
	manifest["roles"]["attacker"] = 2000
	save_json(manifest_path, manifest)
	config = tmp_path / "train.toml"
	config.write_text('base_model="fixture"\nbase_revision="' + "a" * 40 + '"\n')
	pipeline = TaskPipeline(
		Launcher("fixture"), "fixture", ["ag_news"], [42], [2000], config, 1, False
	)
	pipeline.plan = {"ag_news": [2000]}
	ledger = tmp_path / "ledger.jsonl"
	ledger.write_text("")
	pipeline.run_ledgers[("ag_news", 42)] = ledger
	study = Study(pipeline, tmp_path / "run", "", "", 8000, "", 8003, None, "refuse")
	monkeypatch.setattr(TaskPipeline, "train_run", lambda *args: None)
	calls: list[tuple[str, Slot]] = []

	def benchmark(self: Study, path: Path, slot: Slot) -> None:
		settings = load_config(path)
		suite = load_suite(settings.suite_path)
		split = str(settings.dimensions["split"])
		if split == "qualification":
			assert (self.root / "ag_news/selected-recipe.json").exists()
		calls.append((split, slot))
		now = datetime.now(UTC)
		run = EvaluationRun(
			id=slot,
			experiment=settings.experiment,
			slot=slot,
			started_at=now,
			completed_at=now,
			suite_sha256=suite.sha256,
			generation=settings.generation,
			model=settings.models[slot],
			dimensions=settings.dimensions,
			samples=[
				SampleResult(
					item_id=item.id,
					benchmark=item.benchmark,
					category=item.category,
					response="fixture",
					prediction="fixture",
					status="ok",
					latency_seconds=0,
					correct=slot == "student_after" and split == "selection",
				)
				for item in suite.items
			],
		)
		save_json(settings.output_dir / f"{slot}.json", run)

	monkeypatch.setattr(Study, "benchmark", benchmark)
	selection = select_recipe(study, "ag_news")
	assert selection["development_gain"] == 100
	assert selection["qualification"]["gain_pp"] == 0
	assert selection["transfer_gate_passed"] is False
	assert len([c for c in calls if c[0] == "selection"]) == 12
	assert len([c for c in calls if c[0] == "qualification"]) == 2
	assert selection["selected"] == {"learning_rate": 1e-4, "epochs": 1}
	assert select_recipe(study, "ag_news") == selection
	assert len(calls) == 14
	study.minimum_gain = {"ag_news": 3}
	with pytest.raises(LaunchError, match="new --run-dir"):
		select_recipe(study, "ag_news")
