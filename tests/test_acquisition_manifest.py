"""Acquisition manifest contracts, with synthetic ledgers and no teacher."""

import json
from pathlib import Path
from typing import Any

import pytest
from distillation.acquisition_manifest import (
	MANIFEST_VERSION,
	ManifestError,
	build_manifest,
	load_manifest,
	select_rows,
	write_manifest,
)
from distillation.cli import app
from typer.testing import CliRunner

PARAMETERS: dict[str, Any] = {"temperature": 0.0, "max_output_tokens": 256, "seed": 7}


def record(
	example_id: str,
	*,
	prompt: str | None = None,
	tokens: int | None = 10,
	status: str = "accepted",
	task: str = "ag_news",
	group_id: str | None = None,
	model: str = "teacher-a",
	parameters: dict[str, Any] | None = None,
	split: str = "train",
) -> dict[str, Any]:
	source: dict[str, Any] = {"corpus": "fake", "source_id": example_id, "license": "test"}
	if group_id is not None:
		source["group_id"] = group_id
	return {
		"example_id": example_id,
		"attempt_id": "a",
		"split": split,
		"task": task,
		"source": source,
		"prompt": [{"role": "user", "content": prompt or f"prompt {example_id}"}],
		"teacher": {
			"provider": "fake",
			"model": model,
			"endpoint": "http://teacher",
			"generated_at": "2026-01-01T00:00:00Z",
			"parameters": PARAMETERS if parameters is None else parameters,
			"terms_review_id": "test",
		},
		"completion": [{"role": "assistant", "content": f"answer {example_id}"}],
		"finish_reason": "stop",
		"usage": {"input_tokens": 5, "output_tokens": tokens},
		"quality": {"status": status, "checks": [], "reviewer": "test"},
		"attempts": [
			{"attempt_id": "a", "attempted_at": "2026-01-01T00:00:00Z", "status": "succeeded"}
		],
	}


def ledger(path: Path, rows: list[dict[str, Any]]) -> Path:
	path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
	return path


@pytest.fixture
def ledgers(tmp_path: Path) -> list[Path]:
	first = ledger(
		tmp_path / "one.jsonl",
		[
			record("a0", tokens=10, group_id="g1"),
			record("a1", tokens=20),
			record("a2", tokens=None, status="rejected"),
		],
	)
	second = ledger(
		tmp_path / "two.jsonl", [record("b0", tokens=30, task="squad"), record("b1", tokens=40)]
	)
	return [first, second]


def many(tmp_path: Path, tokens: list[int | None]) -> Path:
	return ledger(
		tmp_path / "many.jsonl", [record(f"e{i}", tokens=t) for i, t in enumerate(tokens)]
	)


def test_manifest_fields_and_totals(ledgers: list[Path]) -> None:
	manifest = build_manifest(ledgers, response_policy="clean")
	assert manifest["version"] == MANIFEST_VERSION == "acquisition-manifest-v1"
	assert manifest["teacher"] == {
		"provider": "fake",
		"model": "teacher-a",
		"endpoint": "http://teacher",
		"parameters": PARAMETERS,
	}
	assert [entry["rows"] for entry in manifest["inputs"]] == [3, 2]
	assert all(len(entry["sha256"]) == 64 for entry in manifest["inputs"])
	first = manifest["examples"][0]
	assert first["source_family"] == "g1"
	assert first["source_family_basis"] == "group_id"
	assert (first["corpus"], first["source_id"]) == ("fake", "a0")
	assert first["response_policy"] == "clean"
	assert (first["output_tokens"], first["input_tokens"], first["finish_reason"]) == (
		10,
		5,
		"stop",
	)
	assert first["ledger"] == manifest["inputs"][0]["path"]
	assert first["line"] == 1
	assert all(
		len(first[key]) == 64 for key in ("prompt_sha256", "completion_sha256", "record_sha256")
	)
	assert manifest["examples"][1]["source_family"] == "fake"
	assert manifest["examples"][1]["source_family_basis"] == "corpus"
	assert manifest["totals"] == {
		"examples": 5,
		"accepted_examples": 4,
		"known_output_tokens": 100,
		"unknown_token_rows": 1,
		"examples_per_task": {"ag_news": 4, "squad": 1},
		"examples_per_source_family": {"fake": 4, "g1": 1},
	}
	assert manifest["filter"] == {"accepted_only": False, "rows_read": 5, "rows_dropped": 0}


def test_accepted_only_filter_is_recorded(ledgers: list[Path]) -> None:
	manifest = build_manifest(ledgers, response_policy="clean", accepted_only=True)
	assert manifest["filter"] == {"accepted_only": True, "rows_read": 5, "rows_dropped": 1}
	assert manifest["totals"]["examples"] == manifest["totals"]["accepted_examples"] == 4
	assert manifest["totals"]["unknown_token_rows"] == 0


def test_duplicate_ids_and_prompt_collisions_are_refused(tmp_path: Path) -> None:
	one = ledger(tmp_path / "one.jsonl", [record("x"), record("y")])
	two = ledger(tmp_path / "two.jsonl", [record("x")])
	with pytest.raises(ManifestError, match=r"duplicate example IDs: x at .*one.jsonl:1 and .*two"):
		build_manifest([one, two], response_policy="clean")
	same = ledger(
		tmp_path / "same.jsonl", [record("p", prompt="shared"), record("q", prompt="shared")]
	)
	with pytest.raises(ManifestError, match=r"different example IDs: .*p at .*:1 and q at .*:2"):
		build_manifest([same], response_policy="clean")


def test_mixed_settings_are_refused(tmp_path: Path) -> None:
	mixed_model = ledger(tmp_path / "m.jsonl", [record("a"), record("b", model="teacher-b")])
	with pytest.raises(ManifestError, match="not frozen"):
		build_manifest([mixed_model], response_policy="clean")
	other = {**PARAMETERS, "temperature": 0.7}
	mixed_parameters = ledger(tmp_path / "p.jsonl", [record("a"), record("b", parameters=other)])
	with pytest.raises(ManifestError, match="not frozen"):
		build_manifest([mixed_parameters], response_policy="clean")


@pytest.mark.parametrize("policy", ["", " ", "has space", "-lead", "a/b"])
def test_response_policy_must_be_an_identifier(ledgers: list[Path], policy: str) -> None:
	with pytest.raises(ManifestError, match="response policy"):
		build_manifest(ledgers, response_policy=policy)


def test_unknown_tokens_stay_none(ledgers: list[Path]) -> None:
	manifest = build_manifest(ledgers, response_policy="clean")
	row = next(r for r in manifest["examples"] if r["example_id"] == "a2")
	assert row["output_tokens"] is None


def test_digest_round_trip_and_tamper_detection(ledgers: list[Path], tmp_path: Path) -> None:
	manifest = build_manifest(ledgers, response_policy="clean")
	path = tmp_path / "manifest.json"
	write_manifest(manifest, path)
	assert load_manifest(path, require_ledgers=True) == manifest
	with pytest.raises(ManifestError, match="already exists"):
		write_manifest(manifest, path)

	tampered = json.loads(path.read_text())
	tampered["examples"][0]["output_tokens"] = 11
	bad = tmp_path / "tampered.json"
	bad.write_text(json.dumps(tampered))
	with pytest.raises(ManifestError, match="digest"):
		load_manifest(bad)

	with ledgers[0].open("a") as handle:
		handle.write("\n")
	with pytest.raises(ManifestError, match="changed"):
		load_manifest(path)
	ledgers[0].unlink()
	assert load_manifest(path)["digest"] == manifest["digest"]
	with pytest.raises(ManifestError, match="not available"):
		load_manifest(path, require_ledgers=True)


def test_query_selection_is_seeded_and_nested(tmp_path: Path) -> None:
	manifest = build_manifest([many(tmp_path, [10] * 12)], response_policy="clean")
	small = select_rows(manifest, by="queries", budget=4, seed=1)
	large = select_rows(manifest, by="queries", budget=8, seed=1)
	assert small == select_rows(manifest, by="queries", budget=4, seed=1)
	assert (small["count"], small["output_tokens"]) == (4, 40)
	assert large["example_ids"][:4] == small["example_ids"]
	assert (
		select_rows(manifest, by="queries", budget=4, seed=2)["example_ids"] != small["example_ids"]
	)
	with pytest.raises(ManifestError, match="exceeds"):
		select_rows(manifest, by="queries", budget=13, seed=1)


def test_query_selection_reports_unknown_tokens_without_guessing(tmp_path: Path) -> None:
	manifest = build_manifest([many(tmp_path, [10, None, 10])], response_policy="clean")
	selection = select_rows(manifest, by="queries", budget=3, seed=0)
	assert (selection["output_tokens"], selection["unknown_token_rows"]) == (None, 1)


def test_token_selection_never_exceeds_the_budget(tmp_path: Path) -> None:
	manifest = build_manifest(
		[many(tmp_path, [30, 50, 20, 70, 10, 40, 60, 25])], response_policy="clean"
	)
	by_id = {r["example_id"]: r["output_tokens"] for r in manifest["examples"]}
	previous: list[str] = []
	for budget in (1, 40, 100, 150, 305, 1000):
		selection = select_rows(manifest, by="tokens", budget=budget, seed=3)
		spent = sum(by_id[i] for i in selection["example_ids"])
		assert selection["output_tokens"] == spent <= budget
		assert selection["count"] == len(selection["example_ids"])
		assert selection["example_ids"][: len(previous)] == previous
		previous = selection["example_ids"]
	assert selection["exhausted"] and selection["count"] == 8
	assert not select_rows(manifest, by="tokens", budget=100, seed=3)["exhausted"]


def test_token_selection_is_refused_with_unknown_tokens(tmp_path: Path) -> None:
	manifest = build_manifest([many(tmp_path, [10, None, 10])], response_policy="clean")
	with pytest.raises(ManifestError, match="unknown output tokens"):
		select_rows(manifest, by="tokens", budget=5, seed=0)


def test_selection_excludes_rejected_and_requires_pending_opt_in(
	ledgers: list[Path], tmp_path: Path
) -> None:
	manifest = build_manifest(ledgers, response_policy="clean")
	default = select_rows(manifest, by="queries", budget=4, seed=0)
	assert (default["candidates"], default["excluded_unaccepted"]) == (4, 1)
	assert "a2" not in default["example_ids"]
	with pytest.raises(ManifestError, match="exceeds"):
		select_rows(manifest, by="queries", budget=5, seed=0, include_pending=True)
	pending = ledger(tmp_path / "pending.jsonl", [record("pending", status="pending")])
	manifest = build_manifest([*ledgers, pending], response_policy="clean")
	selected = select_rows(manifest, by="queries", budget=5, seed=0, include_pending=True)
	assert "pending" in selected["example_ids"] and "a2" not in selected["example_ids"]


def test_cli_manifest_then_select(ledgers: list[Path], tmp_path: Path) -> None:
	runner = CliRunner()
	output = tmp_path / "out" / "manifest.json"
	arguments = ["acquisition", "manifest", "--response-policy", "clean", "--output", str(output)]
	for path in ledgers:
		arguments += ["--ledger", str(path)]
	result = runner.invoke(app, arguments)
	assert result.exit_code == 0, result.output
	assert "examples=5 unknown_token_rows=1" in result.output
	assert runner.invoke(app, arguments).exit_code != 0

	selected = runner.invoke(
		app,
		["acquisition", "select", "--manifest", str(output), "--by", "queries"]
		+ ["--budget", "2", "--seed", "9"],
	)
	assert selected.exit_code == 0, selected.output
	assert json.loads(selected.output)["count"] == 2
	refused = runner.invoke(
		app,
		["acquisition", "select", "--manifest", str(output), "--by", "tokens"]
		+ ["--budget", "50", "--seed", "9", "--include-unaccepted"],
	)
	assert refused.exit_code != 0


def test_token_budget_needs_the_clean_policy(tmp_path: Path) -> None:
	rows = [record(f"e{i}", tokens=10) for i in range(4)]
	protected = build_manifest([ledger(tmp_path / "p.jsonl", rows)], response_policy="protected-v1")
	with pytest.raises(ManifestError, match="token budget refused"):
		select_rows(protected, by="tokens", budget=20, seed=1)
	assert select_rows(protected, by="queries", budget=2, seed=1)["count"] == 2


def test_selection_is_limited_to_one_split(tmp_path: Path) -> None:
	rows = [record("tr", split="train"), record("te", split="test")]
	manifest = build_manifest([ledger(tmp_path / "s.jsonl", rows)], response_policy="clean")
	train = select_rows(manifest, by="queries", budget=1, seed=1)
	assert train["example_ids"] == ["tr"] and train["excluded_other_splits"] == 1
	with pytest.raises(ManifestError):
		select_rows(manifest, by="queries", budget=2, seed=1)
	assert select_rows(manifest, by="queries", budget=1, seed=1, split="test")["example_ids"] == [
		"te"
	]


def test_numeric_settings_compare_by_value(tmp_path: Path) -> None:
	rows = [
		record("z0", parameters={**PARAMETERS, "temperature": 0}),
		record("z1", parameters={**PARAMETERS, "temperature": -0.0}),
	]
	manifest = build_manifest([ledger(tmp_path / "z.jsonl", rows)], response_policy="clean")
	assert manifest["totals"]["examples"] == 2


def test_outputs_are_never_overwritten(ledgers: list[Path], tmp_path: Path) -> None:
	path = tmp_path / "manifest.json"
	runner = CliRunner()
	common = ["acquisition", "manifest", "--response-policy", "clean", "--output", str(path)]
	sources = [arg for ledger_path in ledgers for arg in ("--ledger", str(ledger_path))]
	assert runner.invoke(app, [*common, *sources]).exit_code == 0
	before = path.read_bytes()
	select = ["acquisition", "select", "--manifest", str(path), "--by", "queries"]
	result = runner.invoke(app, [*select, "--budget", "1", "--seed", "1", "--output", str(path)])
	assert result.exit_code == 2 and path.read_bytes() == before
	assert not list(tmp_path.glob(".*.tmp"))
	path.write_text("{truncated")
	result = runner.invoke(app, [*select, "--budget", "1", "--seed", "1"])
	assert result.exit_code == 2 and "not valid manifest JSON" in result.output


def test_temporary_files_cannot_overwrite_or_linger(tmp_path: Path, monkeypatch: Any) -> None:
	import os

	from distillation.acquisition_manifest import write_new

	frozen = tmp_path / "frozen.json"
	frozen.write_text("frozen")
	# A planted symlink at a guessable temporary name no longer redirects the write.
	for pid in range(os.getpid() - 2, os.getpid() + 3):
		(tmp_path / f".out.json.{pid}.tmp").symlink_to(frozen)
	write_new(tmp_path / "out.json", "new")
	assert frozen.read_text() == "frozen" and (tmp_path / "out.json").read_text() == "new"

	def broken_link(*args: Any) -> None:
		raise OSError("no hard links")

	monkeypatch.setattr(os, "link", broken_link)
	with pytest.raises(ManifestError, match="exclusively"):
		write_new(tmp_path / "other.json", "text")
	assert not list(tmp_path.glob(".other.json.*"))


def test_large_integer_settings_stay_distinct(tmp_path: Path) -> None:
	rows = [
		record("s0", parameters={**PARAMETERS, "seed": 9007199254740992}),
		record("s1", parameters={**PARAMETERS, "seed": 9007199254740993}),
	]
	with pytest.raises(ManifestError, match="not frozen"):
		build_manifest([ledger(tmp_path / "s.jsonl", rows)], response_policy="clean")


def test_unknown_split_is_refused(ledgers: list[Path]) -> None:
	manifest = build_manifest(ledgers, response_policy="clean")
	with pytest.raises(ManifestError, match="split"):
		select_rows(manifest, by="tokens", budget=10, seed=1, split="trian")


def test_manifest_structure_and_publication_errors_are_reported(
	ledgers: list[Path], tmp_path: Path, monkeypatch: Any
) -> None:
	import tempfile

	from distillation.acquisition_manifest import _sha256_json, write_new

	manifest = build_manifest(ledgers, response_policy="clean")
	del manifest["inputs"]
	manifest["digest"] = _sha256_json({k: v for k, v in manifest.items() if k != "digest"})
	path = tmp_path / "malformed.json"
	path.write_text(json.dumps(manifest))
	with pytest.raises(ManifestError, match="malformed manifest inputs"):
		load_manifest(path)

	def denied(*args: Any, **kwargs: Any) -> None:
		raise PermissionError("read only")

	monkeypatch.setattr(tempfile, "mkstemp", denied)
	with pytest.raises(ManifestError, match="cannot publish"):
		write_new(tmp_path / "new.json", "content")
