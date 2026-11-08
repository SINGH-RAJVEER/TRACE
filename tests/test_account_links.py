"""Offline linking proposals are corroboration and retain uncertain proxy behavior."""

import json
from pathlib import Path

import pytest
from harness.account_links import propose_links
from harness.calibration import BenignRequest
from harness.cli import app
from harness.clusters import LinkEvidence
from typer.testing import CliRunner


def rows(account: str, prompts: list[str]) -> list[BenignRequest]:
	return [BenignRequest(account=account, text=p, time=i * 10) for i, p in enumerate(prompts)]


def population() -> list[BenignRequest]:
	return (
		rows("left", ["private sweep alpha 1", "private sweep beta 2"])
		+ rows("right", ["private sweep alpha 3", "private sweep beta 4"])
		+ rows("benign-a", ["public benchmark alpha", "public benchmark beta"])
		+ rows("benign-b", ["public benchmark alpha", "public benchmark beta"])
	)


def test_split_sweep_cli_and_determinism(tmp_path: Path) -> None:
	requests = population()
	result = propose_links(requests)
	assert [(e["left"], e["right"]) for e in result["edges"]] == [
		("benign-a", "benign-b"),
		("left", "right"),
	]
	assert all(LinkEvidence(**edge).inferred for edge in result["edges"])
	assert result == propose_links(list(reversed(requests)))
	assert result["organizations"] == {} and result["enforcement"] is False
	source, output = tmp_path / "requests.jsonl", tmp_path / "links.json"
	source.write_text("\n".join(r.model_dump_json() for r in requests))
	assert (
		CliRunner()
		.invoke(app, ["link-accounts", "--requests", str(source), "--output", str(output)])
		.exit_code
		== 0
	)
	assert json.loads(output.read_text()) == result


def test_public_benchmark_and_shared_ip_only() -> None:
	requests = sum((rows(a, ["public alpha", "public beta"]) for a in ("a", "b", "c", "d")), [])
	assert propose_links(requests)["edges"] == []
	# Infrastructure is not an input feature; same timing without prompt overlap cannot link.
	assert propose_links(rows("a", ["one", "two"]) + rows("b", ["three", "four"]))["edges"] == []
	assert propose_links(rows("a", ["public alpha"]) + rows("b", ["public alpha"]))["edges"] == []


def test_mixed_proxy_is_uncertain_and_bounded() -> None:
	requests = population() + rows(
		"proxy",
		[
			"private sweep alpha 5",
			"private sweep beta 6",
			"unrelated person one",
			"unrelated person two",
			"unrelated person three",
		],
	)
	result = propose_links(requests, max_df=0.7)
	assert not any("proxy" in (e["left"], e["right"]) for e in result["edges"])
	assert any(e["left"] == "left" for e in result["edges"])
	with pytest.raises(ValueError, match="bounds"):
		propose_links(requests, max_accounts=2)


def test_df_precedes_bounded_sketch_and_timing_is_required() -> None:
	requests = population()
	assert propose_links([r.model_copy(update={"time": None}) for r in requests])["edges"] == []
	# A common benchmark remains excluded even when accounts have different private distractors.
	public = ["public benchmark alpha", "public benchmark beta", "public benchmark gamma"]
	requests = sum(
		(
			rows(a, public + [f"private {a} {word}" for word in ("one", "two", "three")])
			for a in ("a", "b", "c", "d")
		),
		[],
	)
	result = propose_links(requests, max_features=2)
	assert result["edges"] == []
	assert result["provenance"]["excluded_common_templates"] == 3
