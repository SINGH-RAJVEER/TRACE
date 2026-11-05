"""CPU fixtures for chronological campaign exports."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from distillation.checkpoint_learning import (
	execute_checkpoint_curve,
	plan_checkpoint_curve,
	summarize_checkpoint_curve,
)
from distillation.checkpoints import (
	CheckpointError,
	export_campaign_checkpoints,
	load_checkpoint_export,
)
from distillation.collection import CollectionOptions
from distillation.gateway_campaign import collect_gateway
from distillation.io import write_jsonl
from distillation.learning_curve import math_family
from distillation.schedules import campaign_schedule
from distillation.schemas import TeacherGenerationRequest
from test_checkpoint_learning import Recorder, build_suite
from test_gateway import CAPS, _pools
from test_harness_api import Teacher


def _fixture(path: Path) -> None:
	path.mkdir()
	contract = {"version": "gateway-execution-v2"}
	contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
	events: list[dict[str, Any]] = []
	for index, (example, kind, start, end, status) in enumerate(
		[
			("late", "query", 0, 5, "received"),
			("proof", "query", 1, 2, "challenge"),
			("proof", "proof", 3, 4, "received"),
			("tie", "query", 2, 4, "received"),
		],
		1,
	):
		events.extend(
			[
				{
					"event": "attempt",
					"id": str(index),
					"index": index,
					"kind": kind,
					"example_id": example,
					"policy_request_id": "2" if index == 3 else str(index),
					"at_seconds": start,
					"contract": contract_hash,
				},
				{
					"event": "outcome",
					"id": str(index),
					"status": status,
					"at_seconds": end,
					"output_tokens": None,
				},
			]
		)

	def record(ident: str, example: str) -> dict[str, Any]:
		return {
			"example_id": example,
			"attempt_id": ident,
			"split": "train",
			"task": "gsm8k",
			"source": {"corpus": "test", "source_id": example, "license": "test"},
			"prompt": [{"role": "user", "content": "q"}],
			"teacher": {
				"provider": "test",
				"model": "test",
				"endpoint": "test",
				"generated_at": "2026-01-01T00:00:00Z",
				"parameters": {},
				"terms_review_id": "test",
			},
			"completion": [{"role": "assistant", "content": "a"}],
			"usage": {},
			"quality": {"status": "pending", "checks": [], "reviewer": "test"},
			"attempts": [
				{
					"attempt_id": "inner",
					"attempted_at": "2026-01-01T00:00:00Z",
					"status": "succeeded",
				}
			],
		}

	ledger = b"".join(
		(json.dumps(record(i, x)) + "\n").encode()
		for i, x in [("1", "late"), ("3", "proof"), ("4", "tie")]
	)
	event_bytes = b"".join((json.dumps(e) + "\n").encode() for e in events)
	(path / "contract.json").write_text(json.dumps(contract))
	(path / "ledger.jsonl").write_bytes(ledger)
	(path / "campaign.events.jsonl").write_bytes(event_bytes)
	(path / "campaign.json").write_text(
		json.dumps(
			{
				"contract": contract_hash,
				"ledger_records": 3,
				"sha256": {
					"ledger": hashlib.sha256(ledger).hexdigest(),
					"events": hashlib.sha256(event_bytes).hexdigest(),
				},
			}
		)
	)


def test_receipts_proofs_and_bounds(tmp_path: Path) -> None:
	campaign = tmp_path / "campaign"
	_fixture(campaign)
	out = tmp_path / "out"
	manifest = export_campaign_checkpoints(campaign, out, response_counts=[0, 1, 4])
	by_id = {c["id"]: c for c in manifest["checkpoints"]}
	assert by_id["response-0"]["example_ids"] == []
	assert by_id["response-1"]["example_ids"] == ["proof", "tie"]
	assert not by_id["response-4"]["reached"]
	assert by_id["end"]["example_ids"] == ["proof", "tie", "late"]
	assert manifest["interventions"]["first_alert"]["status"] == "unknown"
	assert manifest["interventions"]["first_restriction"]["observed_seconds"] == 2
	assert manifest["late_inflight_responses"]["first_restriction"] == 1
	assert load_checkpoint_export(out) == manifest
	with pytest.raises(CheckpointError):
		export_campaign_checkpoints(campaign, out)
	(campaign / "ledger.jsonl").write_text("corrupt")
	with pytest.raises(CheckpointError, match="digest"):
		load_checkpoint_export(out)


def test_corrupt_source_and_output(tmp_path: Path) -> None:
	campaign = tmp_path / "campaign"
	_fixture(campaign)
	out = tmp_path / "out"
	export_campaign_checkpoints(campaign, out)
	(out / "end.jsonl").write_text("changed")
	with pytest.raises(CheckpointError, match="digest"):
		load_checkpoint_export(out)
	(campaign / "campaign.events.jsonl").write_text("bad")
	with pytest.raises(CheckpointError, match="digest"):
		export_campaign_checkpoints(campaign, tmp_path / "other")


def test_concurrent_intervention_bounds_and_late_failure(tmp_path: Path) -> None:
	campaign = tmp_path / "campaign"
	_fixture(campaign)
	events = [
		json.loads(line) for line in (campaign / "campaign.events.jsonl").read_text().splitlines()
	]
	contract_hash = events[0]["contract"]
	events.append({**events[0], "event": "policy", "level": 2, "action": "serve"})
	for ident, start, end, status in (
		("5", 0, 10, "blocked"),
		("6", 5, 6, "blocked"),
		("7", 11, 12, "failed"),
	):
		events.extend(
			[
				{
					"event": "attempt",
					"id": ident,
					"index": int(ident),
					"kind": "query",
					"example_id": ident,
					"policy_request_id": ident,
					"at_seconds": start,
					"contract": contract_hash,
				},
				{"event": "outcome", "id": ident, "status": status, "at_seconds": end},
			]
		)
	content = b"".join((json.dumps(e) + "\n").encode() for e in events)
	(campaign / "campaign.events.jsonl").write_bytes(content)
	summary = json.loads((campaign / "campaign.json").read_text())
	summary["sha256"]["events"] = hashlib.sha256(content).hexdigest()
	(campaign / "campaign.json").write_text(json.dumps(summary))
	manifest = export_campaign_checkpoints(campaign, tmp_path / "out")
	assert manifest["interventions"]["first_alert"]["observed_seconds"] == 5
	assert manifest["interventions"]["first_restriction"]["lower_seconds"] == 0
	assert manifest["interventions"]["first_refusal"]["lower_seconds"] == 0
	assert manifest["interventions"]["first_refusal"]["observed_seconds"] == 6
	end = manifest["checkpoints"][-1]
	assert end["cutoff_seconds"] == 12
	assert end["cost"]["attempts_started"] == 7
	assert end["cost"]["outcomes"]["failed"] == 1
	assert end["cost"]["unknown_output_token_receipts"] == 3


@pytest.mark.asyncio
async def test_real_collector_export_with_proof(tmp_path: Path) -> None:
	pools = _pools(2)
	for prompt in pools["gsm8k"]:
		prompt.source.group_id = math_family(prompt.prompt[0].content)
	rows, _ = campaign_schedule(pools=pools, caps=CAPS, attack="A1", budget=3, interval_seconds=0)
	schedule = tmp_path / "requests.jsonl"
	write_jsonl(schedule, rows)
	teacher = Teacher()

	async def respond(request: httpx.Request) -> httpx.Response:
		if "X-Harness-Proof" not in request.headers:
			return httpx.Response(
				428,
				json={
					"challenge": {
						"prefix": "fixture",
						"difficulty": 0,
						"expires_at": time.time() + 60,
						"nonce": "fixture",
					}
				},
			)
		result = await teacher.generate(
			TeacherGenerationRequest.model_validate_json(request.content)
		)
		return httpx.Response(200, json=result.model_dump(mode="json"))

	campaign = tmp_path / "campaign"
	summary = await collect_gateway(
		schedule=schedule,
		output=campaign,
		url="http://fixture",
		keys={"default": "fixture"},
		options=CollectionOptions(terms_review_id="fixture"),
		attempt_budget=6,
		transport=httpx.MockTransport(respond),
	)
	manifest = export_campaign_checkpoints(campaign, tmp_path / "export", response_counts=[0, 1, 4])
	assert summary["attempts"] == 6
	assert manifest["successful_responses"] == 3
	end = manifest["checkpoints"][-1]
	assert end["cost"]["proof_attempts_started"] == 3
	assert end["cost"]["outcomes"]["challenge"] == 3
	assert load_checkpoint_export(tmp_path / "export") == manifest
	students = tmp_path / "students"
	plan = plan_checkpoint_curve(
		students,
		tmp_path / "export",
		build_suite(tmp_path / "suite.json"),
		target_accuracy=0.25,
		seeds=(1,),
		include_pending=True,
	)
	assert plan["checkpoints"][-1]["trainable_count"] == 1
	assert execute_checkpoint_curve(students, Recorder())["completed"] >= 2
	report = summarize_checkpoint_curve(students)
	assert report["checkpoints"][-1]["evidence"]["cost"]["attempts_started"] == 6
	assert report["checkpoints"][-1]["mean_accuracy"] == 0.25
	assert report["first_reaching_target"]["evidence"]["cost"]["attempts_started"] >= 2
