"""CPU fixtures for chronological campaign exports."""

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from distillation.campaign import ExperimentBinding
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
from distillation.gateway_cli import check_declared_campaign
from distillation.io import write_jsonl
from distillation.learning_curve import math_family
from distillation.schedules import campaign_schedule
from distillation.schemas import TeacherGenerationRequest
from harness.campaign import CampaignConfig
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


def _config(path: Path, mode: str) -> CampaignConfig:
	config = CampaignConfig.model_validate({"namespace": "fixture", "mode": mode})
	path.write_text(json.dumps(config.model_dump(mode="json", exclude_none=True)))
	return config


def _served(request: httpx.Request, count: int) -> httpx.Response:
	return httpx.Response(
		200,
		json={
			"id": f"r{count}",
			"provider": "fixture",
			"model": "fixture",
			"generated_at": "2026-01-01T00:00:00Z",
			"message": {"role": "assistant", "content": "World"},
			"finish_reason": "stop",
			"usage": {"input_tokens": 1, "output_tokens": 1},
			"endpoint": "fixture",
		},
		headers={"X-Harness-Decision": "allow"},
	)


def _decision(request: httpx.Request, **fields: Any) -> dict[str, Any]:
	row = {
		"event": "decision",
		"time": 1000.0,
		"attempt": 1,
		"level": 0,
		"action": "serve",
		"reason": None,
		"policy_revision": "revision",
		"entitlement": None,
		"benchmark_request": request.headers["X-Benchmark-Request"],
	}
	return row | fields


def _throttled(retry_after: str = "0.01") -> httpx.Response:
	return httpx.Response(
		429,
		json={
			"detail": {"code": "campaign_throttled", "message": "cooldown"},
			"campaign": {"qualified": False, "mode": "throttle", "throttle_until": 2000.0},
		},
		headers={"X-Harness-Decision": "throttle", "Retry-After": retry_after},
	)


async def _bound_campaign(
	tmp_path: Path,
	script: Callable[[httpx.Request, int, list[dict[str, Any]]], httpx.Response],
	*,
	mode: str = "throttle",
	strategy: str = "wait_and_retry",
	retries: int = 2,
	budget: int = 6,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
	"""Run the collector against a scripted gateway that writes the decision log it serves."""
	pools = _pools()
	for prompt in pools["gsm8k"]:
		prompt.source.group_id = math_family(prompt.prompt[0].content)
	rows, _ = campaign_schedule(
		pools=pools, caps=CAPS, attack="A5", budget=budget, accounts=3, interval_seconds=0
	)
	# One account, so submissions are sequential and consecutive-failure counting is exercised.
	for row in rows:
		row.account = "one"
	schedule = tmp_path / "requests.jsonl"
	write_jsonl(schedule, rows)
	log = tmp_path / "decisions.jsonl"
	log.touch()
	calls = 0

	async def respond(request: httpx.Request) -> httpx.Response:
		nonlocal calls
		calls += 1
		written: list[dict[str, Any]] = []
		response = script(request, calls, written)
		with log.open("a") as stream:
			stream.writelines(json.dumps(row) + "\n" for row in written)
		return response

	config = tmp_path / "campaign-source.json"
	_config(config, mode)
	kwargs: dict[str, Any] = dict(
		schedule=schedule,
		output=tmp_path / "campaign",
		url="http://fixture",
		keys={"default": "fixture"},
		options=CollectionOptions(terms_review_id="fixture", concurrency=1),
		attempt_budget=budget * 4,
		transport=httpx.MockTransport(respond),
		decision_log=log,
		experiment=ExperimentBinding(
			base_policy="allow",
			restriction_strategy=strategy,
			max_retries_per_prompt=retries,
			campaign_config=config,
			campaign_mode=mode,
		),
	)
	return await collect_gateway(**kwargs), kwargs, tmp_path / "campaign"


def _evidence(digest: str, status: str = "observed") -> dict[str, Any]:
	return {"config_sha256": digest, "qualified": False, "status": status}


def _events(campaign: Path) -> list[dict[str, Any]]:
	return [
		json.loads(line) for line in (campaign / "campaign.events.jsonl").read_text().splitlines()
	]


@pytest.mark.asyncio
async def test_throttle_arm_attributes_restrictions_and_keeps_interventions_distinct(
	tmp_path: Path,
) -> None:
	digest = CampaignConfig(namespace="fixture", mode="throttle").digest

	def script(request: httpx.Request, count: int, rows: list[dict[str, Any]]) -> httpx.Response:
		root = request.headers["X-Benchmark-Request"]
		if count == 2:
			rows.append(_decision(request, campaign=_evidence(digest, "alert")))
		elif count == 3:
			rows.append(
				{
					"event": "campaign_throttle",
					"time": 1001.0,
					"stage": "admission",
					"throttle_until": 2000.0,
					"policy_revision": "revision",
					"qualified": False,
					"campaign": _evidence(digest, "alert"),
					"benchmark_request": root,
				}
			)
			return _throttled()
		elif count == 5:
			# The early entitlement path logs a level 4 refusal without any detector decision.
			rows.append(
				_decision(
					request,
					attempt=None,
					level=4,
					action="refuse",
					entitlement="rolling_budget_exceeded",
					campaign=None,
				)
			)
			return httpx.Response(
				429,
				json={"detail": {"code": "rolling_budget_exceeded", "message": "used up"}},
				headers={"X-Harness-Decision": "block"},
			)
		else:
			rows.append(_decision(request, campaign=_evidence(digest)))
		return _served(request, count)

	summary, kwargs, campaign = await _bound_campaign(tmp_path, script)
	assert summary["summary_version"] == "campaign-summary-v2"
	assert summary["restricted"] == 2 and summary["failed"] == 0
	assert summary["restrictions_by_kind"] == {"campaign": 1, "entitlement": 1}
	assert summary["restriction_wait_seconds"] > 0
	assert summary["restriction_give_ups"] == 1
	# Neither the campaign alert nor the entitlement refusal is a detector level.
	assert not summary["levels"]["L1"]["reached"] and not summary["levels"]["L4"]["reached"]
	assert summary["campaign"]["first_alert_seconds"] == summary["first_alert_seconds"]
	assert summary["campaign"]["throttles"] == 1
	assert summary["policy_binding"]["status"] == "bound"
	assert summary["policy_binding"]["policy_revision"] == "revision"
	assert not any(
		reason == "service_failures" for reason in summary["termination_by_account"].values()
	)

	events = _events(campaign)
	throttle = next(e for e in events if e["event"] == "campaign_throttle")
	wait = next(e for e in events if e["event"] == "restriction_wait")
	assert wait["requested_seconds"] == 0.01 and wait["cause"] == "campaign_throttled"
	retried = [
		e for e in events if e["event"] == "attempt" and e["example_id"] == throttle["example_id"]
	]
	# The retry is a fresh submission of the same prompt, under a new root.
	assert len(retried) == 2 and retried[0]["policy_request_id"] != retried[1]["policy_request_id"]
	entitlement = next(e for e in events if e["event"] == "policy" and e["entitlement"])
	assert entitlement["level"] is None and entitlement["logged_level"] == 4

	manifest = export_campaign_checkpoints(campaign, tmp_path / "export")
	assert manifest["version"] == "campaign-checkpoints-v2"
	found = manifest["interventions"]
	assert found["first_alert"]["observed_seconds"] < found["first_restriction"]["observed_seconds"]
	assert (
		found["first_restriction"]["observed_seconds"] < found["first_refusal"]["observed_seconds"]
	)
	assert found["first_campaign_restriction"]["causes"] == ["campaign"]
	assert found["first_campaign_restriction"]["checkpoints_from"] == "first_restriction"
	assert found["first_refusal"]["causes"] == ["entitlement"]
	assert found["first_entitlement_restriction"]["checkpoints_from"] == "first_refusal"
	end = next(c for c in manifest["checkpoints"] if c["id"] == "end")
	assert end["cost"]["outcomes"]["restricted"] == 2
	assert end["cost"]["restriction_wait_seconds"] > 0
	assert manifest["policy_binding"]["policy_revision"] == "revision"
	assert load_checkpoint_export(tmp_path / "export") == manifest

	assert await collect_gateway(**kwargs) == summary
	(campaign / "decisions.snapshot.jsonl").write_text("")
	with pytest.raises(ValueError, match="does not match"):
		await collect_gateway(**kwargs)
	with pytest.raises(CheckpointError, match="binding"):
		export_campaign_checkpoints(campaign, tmp_path / "refused")
	_config(kwargs["experiment"].campaign_config, "observe")
	with pytest.raises(ValueError, match="mode"):
		await collect_gateway(**kwargs)


@pytest.mark.asyncio
async def test_separate_campaign_bounds_get_their_own_checkpoints_and_plan(tmp_path: Path) -> None:
	digest = CampaignConfig(namespace="fixture", mode="throttle").digest

	def script(request: httpx.Request, count: int, rows: list[dict[str, Any]]) -> httpx.Response:
		if count == 2:
			# A reserve-time refusal keeps the real detector level beside its entitlement code.
			rows.append(_decision(request, entitlement="token_budget_exceeded"))
			return httpx.Response(
				429,
				json={"detail": {"code": "token_budget_exceeded", "message": "used up"}},
				headers={"X-Harness-Decision": "block"},
			)
		if count == 4:
			rows.append(
				{
					"event": "campaign_throttle",
					"stage": "admission",
					"policy_revision": "revision",
					"benchmark_request": request.headers["X-Benchmark-Request"],
				}
			)
			return _throttled()
		rows.append(_decision(request, campaign=_evidence(digest)))
		return _served(request, count)

	_, _, campaign = await _bound_campaign(tmp_path, script, strategy="give_up", retries=0)
	export = tmp_path / "export"
	manifest = export_campaign_checkpoints(campaign, export)
	found = manifest["interventions"]
	assert found["first_restriction"]["causes"] == ["entitlement"]
	assert found["first_entitlement_restriction"]["checkpoints_from"] == "first_restriction"
	assert found["first_campaign_restriction"]["causes"] == ["campaign"]
	assert "checkpoints_from" not in found["first_campaign_restriction"]
	by_id = {c["id"]: c for c in manifest["checkpoints"]}
	assert by_id["first_campaign_restriction-lower"]["kind"] == "intervention_lower"
	assert (
		by_id["first_campaign_restriction-observed"]["record_count"]
		> by_id["first_restriction-observed"]["record_count"]
	)
	assert "first_entitlement_restriction-lower" not in by_id
	assert load_checkpoint_export(export) == manifest
	plan = plan_checkpoint_curve(
		tmp_path / "students",
		export,
		build_suite(tmp_path / "suite.json"),
		target_accuracy=0.25,
		seeds=(1,),
		include_pending=True,
	)
	assert "first_campaign_restriction-observed" in {c["id"] for c in plan["checkpoints"]}
	assert plan["interventions"] == found


@pytest.mark.asyncio
async def test_observe_alert_is_an_alert_and_never_a_level_or_restriction(tmp_path: Path) -> None:
	digest = CampaignConfig(namespace="fixture", mode="observe").digest

	def script(request: httpx.Request, count: int, rows: list[dict[str, Any]]) -> httpx.Response:
		rows.append(
			_decision(request, campaign=_evidence(digest, "alert"), campaign_override=False)
		)
		return _served(request, count)

	summary, _, campaign = await _bound_campaign(tmp_path, script, mode="observe")
	assert summary["first_alert_seconds"] is not None
	assert summary["first_restriction_seconds"] is None and summary["restricted"] == 0
	assert not summary["levels"]["L1"]["reached"]
	assert summary["attribution_complete"]
	found = export_campaign_checkpoints(campaign, tmp_path / "export")["interventions"]
	assert found["first_campaign_alert"]["status"] == "reached"
	assert found["first_campaign_alert"]["checkpoints_from"] == "first_alert"
	assert found["first_alert"]["causes"] == ["campaign"]
	for name in ("first_restriction", "first_refusal", "first_campaign_restriction"):
		assert found[name]["status"] == "unknown"


@pytest.mark.asyncio
async def test_restrictions_never_end_a_campaign_as_service_failures(tmp_path: Path) -> None:
	def script(request: httpx.Request, count: int, rows: list[dict[str, Any]]) -> httpx.Response:
		return _throttled("600")

	summary, _, campaign = await _bound_campaign(
		tmp_path, script, strategy="give_up", retries=0, budget=12
	)
	assert summary["attempts"] == summary["restricted"] == 12
	assert summary["received"] == summary["failed"] == 0
	assert summary["restriction_give_ups"] == 12 and summary["restriction_wait_seconds"] == 0
	assert set(summary["termination_by_account"].values()) == {"prompts_exhausted"}
	# Nothing in the log says which revision served this, so export refuses rather than guesses.
	assert summary["policy_binding"]["status"] == "unknown"
	with pytest.raises(CheckpointError, match="unknown"):
		export_campaign_checkpoints(campaign, tmp_path / "export")


@pytest.mark.asyncio
async def test_unattributed_429_stays_a_service_failure(tmp_path: Path) -> None:
	def script(request: httpx.Request, count: int, rows: list[dict[str, Any]]) -> httpx.Response:
		return httpx.Response(429, text="upstream overloaded")

	summary, _, _ = await _bound_campaign(tmp_path, script, budget=12)
	assert summary["restricted"] == 0 and summary["failed"] == summary["attempts"] > 0
	assert "service_failures" in summary["termination_by_account"].values()


@pytest.mark.asyncio
async def test_proof_join_keeps_root_and_submission_and_refuses_ambiguity(tmp_path: Path) -> None:
	digest = CampaignConfig(namespace="fixture", mode="throttle").digest

	def script(request: httpx.Request, count: int, rows: list[dict[str, Any]]) -> httpx.Response:
		root = request.headers["X-Benchmark-Request"]
		if "X-Harness-Proof" in request.headers:
			rows.append(
				{
					"event": "campaign_throttle",
					"time": 1002.0,
					"stage": "proof",
					"policy_revision": "revision",
					"benchmark_request": root,
				}
			)
			return _throttled()
		rows.append(_decision(request, level=3, action="challenge", campaign=_evidence(digest)))
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
			headers={"X-Harness-Decision": "throttle"},
		)

	first = tmp_path / "joined"
	first.mkdir()
	_, _, campaign = await _bound_campaign(first, script, strategy="give_up", retries=0, budget=3)
	events = _events(campaign)
	query = next(e for e in events if e["event"] == "attempt" and e["kind"] == "query")
	proof = next(e for e in events if e["event"] == "attempt" and e["kind"] == "proof")
	policy = next(e for e in events if e["event"] == "policy")
	throttle = next(e for e in events if e["event"] == "campaign_throttle")
	assert query["policy_request_id"] == proof["policy_request_id"] == policy["root_id"]
	assert policy["submission_id"] == query["id"] and policy["action"] == "challenge"
	assert throttle["submission_id"] == proof["id"] and throttle["root_id"] == query["id"]
	manifest = export_campaign_checkpoints(campaign, first / "export")
	assert manifest["interventions"]["first_campaign_restriction"]["status"] == "reached"

	def duplicated(
		request: httpx.Request, count: int, rows: list[dict[str, Any]]
	) -> httpx.Response:
		rows.extend([_decision(request, level=1), _decision(request, level=4, action="refuse")])
		return _served(request, count)

	second = tmp_path / "ambiguous"
	second.mkdir()
	summary, _, campaign = await _bound_campaign(
		second, duplicated, mode="observe", strategy="give_up", retries=0, budget=3
	)
	assert summary["ambiguous_policy_rows"] == 3 and not summary["attribution_complete"]
	assert not summary["levels"]["L1"]["reached"]
	assert not any(e["event"] == "policy" for e in _events(campaign))
	with pytest.raises(CheckpointError):
		export_campaign_checkpoints(campaign, second / "export")


@pytest.mark.asyncio
async def test_undeclared_detector_evidence_is_recorded_and_refused(tmp_path: Path) -> None:
	def script(request: httpx.Request, count: int, rows: list[dict[str, Any]]) -> httpx.Response:
		rows.append(_decision(request, campaign=_evidence("another-configuration")))
		return _served(request, count)

	summary, kwargs, campaign = await _bound_campaign(tmp_path, script, mode="observe", budget=3)
	# The spent run stays a completed campaign; resume and export are what refuse it.
	assert summary["policy_binding"]["status"] == "conflict"
	assert summary["received"] == 3
	with pytest.raises(ValueError, match="conflict"):
		await collect_gateway(**kwargs)
	with pytest.raises(CheckpointError, match="conflict"):
		export_campaign_checkpoints(campaign, tmp_path / "export")


@pytest.mark.asyncio
async def test_unbound_collection_keeps_the_legacy_contract_and_failures(tmp_path: Path) -> None:
	rows, _ = campaign_schedule(
		pools=_pools(), caps=CAPS, attack="A5", budget=3, accounts=3, interval_seconds=0
	)
	schedule = tmp_path / "requests.jsonl"
	write_jsonl(schedule, rows)

	async def respond(request: httpx.Request) -> httpx.Response:
		return _throttled()

	summary = await collect_gateway(
		schedule=schedule,
		output=tmp_path / "campaign",
		url="http://fixture",
		keys={"default": "fixture"},
		options=CollectionOptions(terms_review_id="fixture"),
		attempt_budget=3,
		transport=httpx.MockTransport(respond),
	)
	contract = json.loads((tmp_path / "campaign" / "contract.json").read_text())
	assert contract["version"] == "gateway-execution-v2" and "experiment" not in contract
	assert summary["summary_version"] == "campaign-summary-v1"
	assert summary["failed"] == 3 and "restricted" not in summary
	assert all("attribution" not in e for e in _events(tmp_path / "campaign"))
	manifest = export_campaign_checkpoints(tmp_path / "campaign", tmp_path / "export")
	assert manifest["version"] == "campaign-checkpoints-v1"
	assert set(manifest["interventions"]) == {"first_alert", "first_restriction", "first_refusal"}
	assert manifest["interventions"]["first_restriction"]["status"] == "unknown"
	assert "restricted" not in manifest["checkpoints"][-1]["cost"]["outcomes"]


@pytest.mark.asyncio
async def test_a_declared_campaign_needs_its_own_evidence_to_bind(tmp_path: Path) -> None:
	def script(request: httpx.Request, count: int, rows: list[dict[str, Any]]) -> httpx.Response:
		rows.append({"event": "release", "policy_revision": "revision"})
		return _served(request, count)

	summary, _, campaign = await _bound_campaign(tmp_path, script, mode="observe", budget=3)
	assert summary["policy_binding"]["status"] == "conflict"
	with pytest.raises(CheckpointError, match="conflict"):
		export_campaign_checkpoints(campaign, tmp_path / "export")


@pytest.mark.asyncio
async def test_output_receipts_are_kept_and_withheld_outputs_are_not_service_failures(
	tmp_path: Path,
) -> None:
	digest = CampaignConfig(namespace="fixture", mode="observe").digest

	def script(request: httpx.Request, count: int, rows: list[dict[str, Any]]) -> httpx.Response:
		rows.append(_decision(request, campaign=_evidence(digest)))
		if count > 2:
			return httpx.Response(
				502,
				json={"detail": {"code": "output_withheld"}},
				headers={
					"X-Harness-Decision": "allow",
					"X-Harness-Output-Receipt": json.dumps({"outcome": "withheld"}),
				},
			)
		served = _served(request, count)
		released = "0" * 64 if count == 2 else hashlib.sha256(b"World").hexdigest()
		served.headers["X-Harness-Output-Receipt"] = json.dumps(
			{"outcome": "rewritten", "released_sha256": released}
		)
		return served

	summary, _, _ = await _bound_campaign(
		tmp_path, script, mode="observe", strategy="give_up", retries=0, budget=14
	)
	assert summary["restrictions_by_kind"] == {"output": 12}
	assert summary["output_receipts"] == {
		"received": 14,
		"by_outcome": {"rewritten": 2, "withheld": 12},
		"invalid": 0,
		"released_hash_mismatches": 1,
	}
	assert not any(
		reason == "service_failures" for reason in summary["termination_by_account"].values()
	)


def test_a_reporting_gateway_must_serve_the_declared_campaign() -> None:
	check_declared_campaign({}, "declared")
	check_declared_campaign({"policy_revision": "r", "campaign_digest": "declared"}, "declared")
	check_declared_campaign({"policy_revision": "r"}, None)
	with pytest.raises(ValueError, match="not the declared"):
		check_declared_campaign({"policy_revision": "r", "campaign_digest": "other"}, "declared")
	with pytest.raises(ValueError, match="not the declared"):
		check_declared_campaign({"policy_revision": "r"}, "declared")
