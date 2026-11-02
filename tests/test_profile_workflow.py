"""Profile generation, role separation and legacy workload regression."""

import json
from pathlib import Path

import pytest
from distillation.benchmarks.models import BenchmarkItem
from distillation.io import write_jsonl
from distillation.profiles import TopicSampler, profile_clients
from distillation.role_workflow import SourceItem, build_roles, screen
from distillation.workloads import mixed_clients
from workload_fixtures import CAPS4, prompt_record


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
	assert manifest["topic_coverage"]["squad"]["passed"]


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


@pytest.mark.parametrize("version", ["v3", "v5"])
@pytest.mark.parametrize(
	"role", ["fitting", "score_calibration", "policy_calibration", "benign_test"]
)
def test_partial_workloads_balance_categories_and_keep_topic_blocks(
	version: str, role: str
) -> None:
	from collections import Counter, defaultdict

	from distillation.tasks import LEGITIMATE_ROLES

	items = [source_item(domain, index) for domain in CAPS4 for index in range(1000)]
	# Unequal category shares exercise proportional selection rather than equal cycling.
	for index, item in enumerate(items):
		item.reference.category = f"topic-{min(index % 10, 3)}"
	pools = {d: [i.prompt for i in items if i.prompt.task == d] for d in CAPS4}
	refs = {i.prompt.example_id: i.reference for i in items}
	if version == "v3":
		rows, manifest = mixed_clients(
			role=role,
			pools=pools,
			caps=CAPS4,
			accounts=24,
			references=refs,
			requests_per_account=40,
			length_tail=None,
		)
	else:
		roles = ("reference_fit", "score_calibration", "policy_development", "locked_test")
		rows, manifest = profile_clients(
			role=roles[LEGITIMATE_ROLES.index(role)],
			pools=pools,
			caps=CAPS4,
			accounts=24,
			references=refs,
			requests_per_account=40,
			length_tail=None,
		)
	for domain, pool in pools.items():
		used = {r.prompt.example_id for r in rows if r.domain == domain}
		counts = Counter(refs[key].category for key in used)
		expected = Counter(refs[p.example_id].category for p in pool)
		assert 0 < len(used) < len(pool)
		for category, count in expected.items():
			assert abs(counts[category] / len(used) - count / len(pool)) <= 0.03
		coverage = manifest["topic_coverage"][domain]
		assert coverage["used_category_counts"] == dict(counts)
		assert coverage["pool_category_counts"] == dict(expected)
	by_account = defaultdict(list)
	for row in rows:
		by_account[row.account].append(row)
	for account_rows in by_account.values():
		if account_rows[0].client_type == "specialist":
			topics = [refs[r.prompt.example_id].category for r in account_rows]
			assert topics == sorted(topics)


def test_changed_workload_cannot_overwrite_frozen_inventory(tmp_path: Path) -> None:
	from distillation.gateway_cli import _write

	output = tmp_path / "frozen"
	_write(output, [], {"version": "old"})
	before = {p.name: p.read_bytes() for p in output.iterdir()}
	with pytest.raises(FileExistsError):
		_write(output, [], {"version": "changed"})
	assert {p.name: p.read_bytes() for p in output.iterdir()} == before
