"""The pre-campaign detectability gate refuses undetected protected arms unless overridden."""

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from distillation import campaign_timing, detectability_gate, gateway_campaign, gateway_cli
from distillation.benchmarks.models import save_json
from distillation.collection import CollectionOptions
from distillation.detectability_gate import CachedEncoder, contract_record, detectability, gate
from distillation.gateway_campaign import collect_gateway
from distillation.gateway_cli import app
from distillation.io import write_jsonl
from distillation.schemas import (
	GatewayRequest,
	Message,
	PromptRecord,
	PromptSource,
	TeacherGenerationRequest,
)
from distillation.tasks import AG_NEWS_PROMPT
from harness.calibration import BenignRequest, HarnessConfig, calibrate, save_calibration
from harness.mmd import Array
from harness.serialization import gateway_turn
from test_harness_api import Teacher
from typer.testing import CliRunner

DIMENSION = 8


class MarkerEncoder:
	"""Gateway detector text from one Gaussian; articles mentioning EXTRACT are shifted."""

	def __init__(self) -> None:
		self.calls = 0

	def encode(self, texts: list[str]) -> Array:
		self.calls += 1
		rows = []
		for text in texts:
			seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
			vector = np.random.default_rng(seed).normal(size=DIMENSION)
			if "EXTRACT" in text:
				vector[0] += 3.0
			rows.append(vector / np.linalg.norm(vector))
		return np.array(rows)

	def identity(self) -> dict[str, Any]:
		return {"model": "marker", "encoded": self.calls}


class OfflineEncoder(MarkerEncoder):
	"""The same identity, but any embedding it would compute is a failure: cache hits only."""

	def encode(self, texts: list[str]) -> Array:
		raise AssertionError("the gate recomputed a cached embedding")


def _benign(prefix: str, accounts: int, per_account: int) -> list[BenignRequest]:
	return [
		BenignRequest(
			account=f"{prefix}-{a}",
			text=gateway_turn("ag_news", AG_NEWS_PROMPT.format(text=f"{prefix} {a} {i}")),
			time=float(i),
			source_group=f"ag_news:{prefix}-{a}-{i}",
		)
		for a in range(accounts)
		for i in range(per_account)
	]


def _schedule(directory: Path, marker: str, count: int = 40) -> Path:
	rows = [
		GatewayRequest(
			request_id=f"r{index}",
			account="extractor",
			time=0.0,
			prompt=PromptRecord(
				example_id=f"ag_news-{index}",
				split="train",
				task="ag_news",
				source=PromptSource(
					corpus="ag_news", source_id=str(index), group_id=f"g{index}", license="test"
				),
				prompt=[
					Message(role="user", content=AG_NEWS_PROMPT.format(text=f"{marker} {index}"))
				],
			),
			max_output_tokens=16,
		)
		for index in range(count)
	]
	directory.mkdir()
	write_jsonl(directory / "requests.jsonl", rows)
	save_json(
		directory / "manifest.json",
		{
			"sha256": {
				"requests.jsonl": hashlib.sha256(
					(directory / "requests.jsonl").read_bytes()
				).hexdigest()
			},
			"attempt_budget": count,
			"deadline_seconds": 60,
		},
	)
	return directory


@pytest.fixture(scope="module")
def frozen(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
	root = tmp_path_factory.mktemp("gate")
	calibration, references = calibrate(
		config=HarnessConfig(
			task="gateway", window_size=10, reference_windows=3, bandwidth_sample=60
		),
		encoder=MarkerEncoder(),
		fitting=_benign("fit", 4, 30),
		score_calibration=_benign("score", 20, 50),
		policy_calibration=_benign("policy", 200, 40),
	)
	save_calibration(root / "calibration", calibration, references)
	held_out = root / "benign_test.detector.jsonl"
	held_out.write_text("".join(row.model_dump_json() + "\n" for row in _benign("held", 60, 40)))
	return {
		"calibration": root / "calibration",
		"benign_test": held_out,
		"undetected": _schedule(root / "undetected", "ordinary story"),
		"detected": _schedule(root / "detected", "EXTRACT story"),
	}


def _hashes(*directories: Path) -> dict[str, str]:
	return {
		str(path): hashlib.sha256(path.read_bytes()).hexdigest()
		for directory in directories
		for path in sorted(directory.rglob("*"))
		if path.is_file()
	}


def _collect(
	frozen: dict[str, Path], schedule: str, output: Path, cache: Path, *extra: str
) -> list[str]:
	return [
		"collect",
		"--input-dir",
		str(frozen[schedule]),
		"--output",
		str(output),
		"--policy",
		"mmd",
		"--calibration",
		str(frozen["calibration"]),
		"--benign-test",
		str(frozen["benign_test"]),
		"--embedding-cache",
		str(cache),
		"--terms-review-id",
		"fixture",
		*extra,
	]


@pytest.fixture
def offline_cli(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
	"""The collect command with its gateway, BGE model and campaign replaced by fixtures."""
	captured: dict[str, Any] = {}

	async def fake_collect(**kwargs: Any) -> dict[str, Any]:
		captured.update(kwargs)
		kwargs["output"].mkdir()
		return {}

	monkeypatch.setattr(gateway_cli, "_served_policy", lambda url: "mmd")
	monkeypatch.setattr(gateway_campaign, "collect_gateway", fake_collect)
	monkeypatch.setattr(campaign_timing, "campaign_timing", lambda *args, **kwargs: {})
	monkeypatch.setattr(
		detectability_gate,
		"bge_cache",
		lambda directory, calibration: CachedEncoder(directory, {"model": "marker"}, MarkerEncoder),
	)
	return captured


def test_gate_refuses_an_undetectable_extractor_before_the_campaign_starts(
	frozen: dict[str, Path], offline_cli: dict[str, Any], tmp_path: Path
) -> None:
	before = _hashes(frozen["calibration"], frozen["undetected"])
	output = tmp_path / "run"
	result = CliRunner().invoke(app, _collect(frozen, "undetected", output, tmp_path / "cache"))
	assert result.exit_code == 1, result.output
	assert "no MMD arm escalates the attacker" in result.output
	assert "--allow-undetected" in result.output
	assert not offline_cli and not output.exists()
	assert _hashes(frozen["calibration"], frozen["undetected"]) == before
	# Without the gate's inputs a protected arm cannot start at all.
	bare = CliRunner().invoke(
		app,
		[
			"collect",
			"--input-dir",
			str(frozen["undetected"]),
			"--output",
			str(output),
			"--policy",
			"mmd",
		],
	)
	assert bare.exit_code != 0 and "--benign-test" in bare.output and not output.exists()


@pytest.mark.asyncio
async def test_override_records_a_quota_comparison_in_the_campaign_contract(
	frozen: dict[str, Path], tmp_path: Path
) -> None:
	cache = tmp_path / "cache"
	inputs: dict[str, Any] = {
		"calibration_dir": frozen["calibration"],
		"schedule_dir": frozen["undetected"],
		"benign_test": frozen["benign_test"],
		"embedding_cache": cache,
	}
	record = gate(detectability(**inputs, encoder=MarkerEncoder()), allow_undetected=True)
	assert (record["detected"], record["reading"]) == (False, "quota comparison")
	# A rerun reads only cached embeddings and binds the same contract, so resume still works.
	again = gate(detectability(**inputs, encoder=OfflineEncoder()), allow_undetected=True)
	assert contract_record(again) == contract_record(record)
	teacher = Teacher()

	async def respond(request: httpx.Request) -> httpx.Response:
		result = await teacher.generate(
			TeacherGenerationRequest.model_validate_json(request.content)
		)
		return httpx.Response(200, json=result.model_dump(mode="json"))

	kwargs: dict[str, Any] = {
		"schedule": frozen["undetected"] / "requests.jsonl",
		"output": tmp_path / "run",
		"url": "http://fixture",
		"keys": {"default": "fixture"},
		"options": CollectionOptions(terms_review_id="fixture"),
		"attempt_budget": 2,
		"transport": httpx.MockTransport(respond),
	}
	summary = await collect_gateway(**kwargs, detectability=record)
	contract = json.loads((tmp_path / "run" / "contract.json").read_text())
	assert contract["detectability"]["allow_undetected"] is True
	assert contract["detectability"]["reading"] == "quota comparison"
	assert (summary["reading"], summary["allow_undetected"]) == ("quota comparison", True)
	assert json.loads((tmp_path / "run" / "detectability.json").read_text())["arms"]
	assert await collect_gateway(**kwargs, detectability=again) == summary


def test_detectable_extractor_passes_and_binds_its_gate_record(
	frozen: dict[str, Path], offline_cli: dict[str, Any], tmp_path: Path
) -> None:
	result = CliRunner().invoke(
		app, _collect(frozen, "detected", tmp_path / "run", tmp_path / "cache")
	)
	assert result.exit_code == 0, result.output
	record = offline_cli["detectability"]
	assert (record["detected"], record["reading"], record["allow_undetected"]) == (
		True,
		"detection",
		False,
	)
	assert any(arm["escalates_beyond_benign"] for arm in record["arms"].values())
	assert "not evidence of effectiveness" in record["purpose"]
