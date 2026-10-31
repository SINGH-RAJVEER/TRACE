"""Profile generation, role separation and legacy workload regression."""

import hashlib
import json
from pathlib import Path

import pytest
from distillation.benchmarks.models import BenchmarkItem
from distillation.io import read_jsonl, write_jsonl
from distillation.profiles import TopicSampler, profile_clients
from distillation.role_workflow import SourceItem, build_roles, screen
from distillation.schemas import PromptRecord
from distillation.tasks import task_directory
from distillation.workloads import mixed_clients, task_references
from workload_fixtures import CAPS4, prompt_record, task_tree


def test_legacy_workload_golden(tmp_path: Path) -> None:
	task_tree(tmp_path, 500, roles=("fitting",))
	pools = {
		domain: read_jsonl(
			task_directory(domain, root=tmp_path) / "legitimate" / "fitting.jsonl", PromptRecord
		)
		for domain in CAPS4
	}
	refs = {
		key: item
		for domain in CAPS4
		for key, item in task_references(task_directory(domain, root=tmp_path)).items()
	}
	rows, manifest = mixed_clients(
		pools=pools,
		caps=CAPS4,
		role="fitting",
		accounts=25,
		generalist_share=0.3,
		requests_per_account=20,
		seed=7,
		references=refs,
		concentrated=0.4,
		length_tail=None,
	)
	payload = "".join(row.model_dump_json(by_alias=True) + "\n" for row in rows)
	assert (
		hashlib.sha256(payload.encode()).hexdigest()
		== "685fe1d4360c050885581b7c8e04ae631e6b539ecae4b0fbf1caaad285c5d57f"
	)
	assert (
		hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
		== "d6fcaa7c9e44a2ded4cac407c2391c7f002867af8c23aa575066e85ad5cff676"
	)


def source_item(domain: str, index: int) -> SourceItem:
	prompt, category = prompt_record(domain, "source", index)
	return SourceItem(
		prompt=prompt,
		reference=BenchmarkItem(
			id=prompt.example_id,
			benchmark=domain,
			category=category,
			question="fixture",
			messages=prompt.prompt,
			metric="label_accuracy",
			answer="fixture",
		),
		family=f"family-{index}",
		family_kind={
			"ag_news": "publisher_topic_period",
			"squad": "document_title",
			"gsm8k": "problem_family",
			"coding": "problem_lineage",
		}[domain],
		provenance="fixture audited",
	)


def test_balanced_topic_sampler_covers_partial_pool() -> None:
	items = [source_item("ag_news", index) for index in range(100)]
	by_id = {item.prompt.example_id: item.reference.category for item in items}
	sampler = TopicSampler([item.prompt for item in items], lambda p: by_id[p.example_id], 42)
	for _ in range(20):
		assert sampler.draw() is not None
	assert sorted(sampler.used().values()) == [5, 5, 5, 5]


def test_roles_reject_unknown_and_cross_role_near_duplicate(tmp_path: Path) -> None:
	items = [source_item("ag_news", i) for i in range(40)]
	duplicate = items[1].model_copy(
		update={"prompt": items[1].prompt.model_copy(update={"prompt": items[0].prompt.prompt})}
	)
	assert screen(
		[items[0], duplicate], {items[0].prompt.example_id: "a", duplicate.prompt.example_id: "b"}
	)
	items[0].provenance = "unknown"
	write_jsonl(tmp_path / "source.jsonl", items)
	with pytest.raises(ValueError, match="unknown provenance"):
		build_roles(tmp_path / "source.jsonl", tmp_path / "roles")


def test_profile_generator_exercises_all_profiles() -> None:
	items = [source_item(domain, index) for domain in CAPS4 for index in range(300)]
	pools = {
		domain: [item.prompt for item in items if item.prompt.task == domain] for domain in CAPS4
	}
	rows, manifest = profile_clients(
		pools=pools,
		caps=CAPS4,
		role="locked_test",
		accounts=8,
		references={item.prompt.example_id: item.reference for item in items},
		requests_per_account=30,
		max_requests_per_account=30,
		length_tail=None,
		short_share=0,
	)
	assert len({row.client_type for row in rows}) == 8
	assert all(length == 30 for length in manifest["account_lengths"].values())
	assert manifest["topic_coverage"]["ag_news"]["passed"]
	assert not manifest["topic_coverage"]["squad"]["passed"]


def test_build_and_validate_pipeline_fixture(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	from distillation.gateway_cli import app
	from distillation.schemas import Message
	from typer.testing import CliRunner

	items = []
	for domain in CAPS4:
		for index in range(3000):
			item = source_item(domain, index)
			# Each family is independently identifiable, with balanced topics.
			item.reference.category = "math" if domain == "gsm8k" else f"topic-{index % 2}"
			item.prompt.source.group_id = f"family-{index}"
			item.prompt.prompt = [
				Message(
					role="user",
					content=f"{domain} {index} "
					+ " ".join(f"{domain}term{index}x{j}" for j in range(12)),
				)
			]
			items.append(item)
	source = tmp_path / "source.jsonl"
	write_jsonl(source, items)
	runner = CliRunner()
	root = tmp_path / "build"
	result = runner.invoke(
		app,
		[
			"profile-build",
			"--source",
			str(source),
			"--output",
			str(root),
			"--horizons-per-profile",
			"1",
		],
	)
	assert result.exit_code == 0, result.output
	result = runner.invoke(
		app,
		[
			"profile-validate",
			"--input-dir",
			str(root),
			"--output",
			str(root / "validation.json"),
			"--min-horizons",
			"1",
		],
	)
	assert result.exit_code == 0, result.output
	from harness import encoder
	from harness.cli import app as harness_app
	from test_harness import FakeEncoder

	monkeypatch.setattr(encoder, "BGEEncoder", lambda device=None: FakeEncoder())
	args = [
		"calibrate",
		"--fitting",
		str(root / "reference_fit.detector.jsonl"),
		"--score-calibration",
		str(root / "score_calibration.detector.jsonl"),
		"--policy-calibration",
		str(root / "policy_development.detector.jsonl"),
		"--output-dir",
		str(root / "calibration"),
		"--task",
		"gateway",
		"--window-size",
		"10",
	]
	assert runner.invoke(harness_app, args).exit_code != 0
	calibration = runner.invoke(
		harness_app, [*args, "--validation-report", str(root / "validation.json")]
	)
	assert calibration.exit_code == 0, str(calibration.exception) + calibration.output
	adequacy = runner.invoke(
		harness_app,
		[
			"adequacy",
			"--calibration",
			str(root / "calibration"),
			"--output",
			str(root / "adequacy.json"),
		],
	)
	assert adequacy.exit_code != 0
	assert not json.loads((root / "adequacy.json").read_text())["passed"]
	path = root / "reference_fit.detector.jsonl"
	path.write_text(
		path.read_text().replace(
			'"workload_version":"gateway-clients-v5"', '"workload_version":"changed"', 1
		)
	)
	result = runner.invoke(
		app,
		[
			"profile-validate",
			"--input-dir",
			str(root),
			"--output",
			str(root / "bad.json"),
			"--min-horizons",
			"1",
		],
	)
	assert result.exit_code != 0
	assert "detector inputs differ" in result.output
