"""Frozen schedules and online campaign accounting use synthetic local HTTP fixtures."""

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import cast

import httpx
import pytest
from distillation.collection import CollectionOptions
from distillation.external_sources import EXTERNAL_SOURCES
from distillation.gateway_campaign import collect_gateway
from distillation.io import canonical_json, read_jsonl, write_jsonl
from distillation.schedules import Attack, campaign_schedule
from distillation.schemas import LedgerRecord, TeacherGenerationRequest
from test_gateway import CAPS, _pools
from test_harness_api import Teacher

FROZEN = {
	"A1": "7f377de5b8f9c35db1bf0d3a7bbf5e6622a616659af99f896fd6059a1e6802a0",
	"A2": "473ae479e8df5414f227f7abafc56d0807951bd3bb3be0389eddea50b67ac7ab",
	"A3": "36abddcaf561b9f3546188118d9d34df0c88597d11132c7b508f8e847e923dc1",
	"A4": "c734d27124baf8ea695fbb35c2e6819b2b0e664b93a161be3e3b93098c810808",
	"A5": "6cc6c423e84d4dfa977b41d05fc5f03ea37e41cccf04cc2c6fa706fb87a8548f",
	"A6": "dfce49f81f31dd76cc8366e11e08c05f702ab1967954e942119aaf85f38bf1f5",
	"A7": "8dbe69ebc59101f5c497f8a36df63a943a56c62bf46bb299b726041a5bb8f35d",
}


@pytest.mark.parametrize("attack", FROZEN)
def test_frozen_schedule_and_manifest_bytes(attack: str) -> None:
	# Digests captured from schedules.py before adding any adaptive options.
	pools = _pools()
	external = {
		d: [
			p.model_copy(
				update={
					"example_id": "external-" + p.example_id,
					"source": p.source.model_copy(
						update={"corpus": EXTERNAL_SOURCES[d].dataset.repository}
					),
				}
			)
			for p in pool
		]
		for d, pool in pools.items()
	}
	rows, manifest = campaign_schedule(
		pools=external if attack in {"A3", "A7"} else pools,
		caps=CAPS,
		attack=cast(Attack, attack),
		budget=60,
		cover_pools=pools if attack == "A3" else None,
	)
	assert (
		hashlib.sha256(
			canonical_json(
				[
					[r.model_dump(mode="json") for r in rows],
					manifest,
				]
			).encode()
		).hexdigest()
		== FROZEN[attack]
	)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, 1, 2])
async def test_online_rounds_count_failures_proofs_and_account_limits(
	tmp_path: Path,
	limit: int | None,
) -> None:
	from distillation.adaptive import CoverageSelector

	rows, manifest = campaign_schedule(
		pools=_pools(5),
		caps=CAPS,
		attack="A8",
		budget=9,
		round_size=2,
		per_account_limit=limit,
		interval_seconds=0,
	)
	assert len(rows) == 15  # Full candidate pool, rather than a selected nine-query schedule.
	schedule = tmp_path / "requests.jsonl"
	write_jsonl(schedule, rows)
	teacher = Teacher()
	calls = 0

	async def respond(request: httpx.Request) -> httpx.Response:
		nonlocal calls
		calls += 1
		if calls == 1:
			return httpx.Response(500)
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

	output = tmp_path / "run"
	result = await collect_gateway(
		schedule=schedule,
		output=output,
		url="http://fixture",
		keys={"default": "fixture"},
		options=CollectionOptions(terms_review_id="fixture"),
		attempt_budget=9,
		variants=manifest["variants"],
		transport=httpx.MockTransport(respond),
	)
	assert calls == result["queries"] == 9
	assert result["failed"] == 1
	events = [
		json.loads(line) for line in (output / "campaign.events.jsonl").read_text().splitlines()
	]
	attempts = [e for e in events if e["event"] == "attempt"]
	if limit:
		assert max(Counter(e["account"] for e in attempts).values()) <= limit
	if limit == 1:
		assert result["proof_requests"] == 0
	else:
		assert result["proof_requests"] > 0
	rounds = [e for e in events if e["event"] == "selection_round"]
	assert len(rounds) > 1
	assert rounds[0]["attempts_spent"] == 0
	assert rounds[1]["feedback"]
	assert len({key for e in rounds for key in e["selected_ids"]}) == sum(
		len(e["selected_ids"]) for e in rounds
	)
	selector = CoverageSelector(rows)
	ledger = read_jsonl(output / "ledger.jsonl", LedgerRecord)
	for event in rounds:
		batch, scores = selector.select(
			2 if 9 - event["attempts_spent"] >= 2 else 1,
			[r for r in ledger if r.example_id in event["released_example_ids"]],
			event["feedback"],
		)
		assert [r.request_id for r in batch] == event["selected_ids"]
		assert scores == event["scores"]


def test_explicit_variants_preserve_sources_and_do_not_mutate_pools() -> None:
	pools = _pools(10)
	before = canonical_json({d: [p.model_dump() for p in v] for d, v in pools.items()})
	rows, manifest = campaign_schedule(
		pools=pools,
		caps=CAPS,
		attack="A1",
		budget=30,
		topic_sorted=True,
		answer_only=True,
		deduplicate=True,
		per_account_limit=3,
	)
	assert [r.domain for r in rows] == sorted(r.domain for r in rows)
	assert max(Counter(r.account for r in rows).values()) == 3
	assert all("Solve step by step" not in r.prompt.prompt[0].content for r in rows)
	assert manifest["variants"]["topic_sorted"]
	assert before == canonical_json({d: [p.model_dump() for p in v] for d, v in pools.items()})
	duplicates = _pools(5)
	for pool in duplicates.values():
		for prompt in pool:
			prompt.prompt = pool[0].prompt
	unique, _ = campaign_schedule(
		pools=duplicates, caps=CAPS, attack="A8", budget=9, deduplicate=True
	)
	assert len(unique) == 3


@pytest.mark.asyncio
async def test_a8_deadline_and_failure_stop(tmp_path: Path) -> None:
	import asyncio

	from distillation.campaign import MAX_CONSECUTIVE_FAILURES

	rows, manifest = campaign_schedule(
		pools=_pools(MAX_CONSECUTIVE_FAILURES),
		caps=CAPS,
		attack="A8",
		budget=60,
		interval_seconds=0,
		round_size=1,
	)
	schedule = tmp_path / "requests.jsonl"
	write_jsonl(schedule, rows)

	async def slow(request: httpx.Request) -> httpx.Response:
		await asyncio.sleep(10)
		return httpx.Response(500)

	for name, transport, deadline in [
		("deadline", httpx.MockTransport(slow), 0.1),
		("service_failures", httpx.MockTransport(lambda request: httpx.Response(500)), 60),
	]:
		result = await collect_gateway(
			schedule=schedule,
			output=tmp_path / name,
			url="http://fixture",
			keys={"default": "fixture"},
			options=CollectionOptions(terms_review_id="fixture"),
			attempt_budget=60,
			deadline_seconds=deadline,
			variants=manifest["variants"],
			transport=transport,
		)
		assert set(result["termination_by_account"].values()) == {name}
		assert result["queries"] == (1 if name == "deadline" else MAX_CONSECUTIVE_FAILURES)


@pytest.mark.asyncio
async def test_uncertainty_hook_receives_only_candidates_and_released_ledger(
	tmp_path: Path,
) -> None:
	from collections.abc import Sequence

	from distillation.adaptive import CoverageSelector
	from distillation.schemas import GatewayRequest

	rows, manifest = campaign_schedule(
		pools=_pools(3),
		caps=CAPS,
		attack="A8",
		budget=4,
		round_size=1,
		interval_seconds=0,
	)
	schedule = tmp_path / "requests.jsonl"
	write_jsonl(schedule, rows)
	observed: list[int] = []

	def hook(
		candidates: Sequence[GatewayRequest], released: Sequence[LedgerRecord]
	) -> dict[str, float]:
		observed.append(len(released))
		return {r.request_id: 100 if r.domain == "gsm8k" else 0 for r in candidates}

	teacher = Teacher()

	async def respond(request: httpx.Request) -> httpx.Response:
		result = await teacher.generate(
			TeacherGenerationRequest.model_validate_json(request.content)
		)
		return httpx.Response(200, json=result.model_dump(mode="json"))

	output = tmp_path / "run"
	await collect_gateway(
		schedule=schedule,
		output=output,
		url="http://fixture",
		keys={"default": "fixture"},
		options=CollectionOptions(terms_review_id="fixture"),
		attempt_budget=4,
		variants=manifest["variants"],
		transport=httpx.MockTransport(respond),
		uncertainty_hook=hook,
		uncertainty_hook_id="fixture-v1",
	)
	assert observed == [0, 1, 2, 3]
	assert json.loads((output / "contract.json").read_text())["uncertainty_hook_id"] == "fixture-v1"
	ledger = read_jsonl(output / "ledger.jsonl", LedgerRecord)
	assert [r.task for r in ledger[:3]] == ["gsm8k"] * 3
	events = [
		json.loads(line) for line in (output / "campaign.events.jsonl").read_text().splitlines()
	]
	selector = CoverageSelector(rows)
	for event in events:
		if event["event"] != "selection_round":
			continue

		def replay_hook(
			candidates: Sequence[GatewayRequest],
			released: Sequence[LedgerRecord],
			scores: dict[str, float] = event["uncertainty_scores"],
		) -> dict[str, float]:
			return scores

		batch, _ = selector.select(
			1,
			[r for r in ledger if r.example_id in event["released_example_ids"]],
			event["feedback"],
			replay_hook,
		)
		assert [r.request_id for r in batch] == event["selected_ids"]


def test_cli_passes_online_contract_to_collection(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	from typing import Any

	from distillation import campaign_timing, gateway_campaign, gateway_cli
	from typer.testing import CliRunner

	monkeypatch.setattr(gateway_cli, "_pools", lambda *args, **kwargs: (_pools(3), CAPS, {}))
	runner = CliRunner()
	inventory = tmp_path / "inventory"
	result = runner.invoke(
		gateway_cli.app,
		[
			"schedule",
			"--attack",
			"A8",
			"--budget",
			"4",
			"--round-size",
			"1",
			"--per-account-limit",
			"2",
			"--answer-only",
			"--deduplicate",
			"--output",
			str(inventory),
		],
	)
	assert result.exit_code == 0, result.output
	captured: dict[str, Any] = {}

	async def collect(**kwargs: Any) -> dict[str, Any]:
		captured.update(kwargs)
		kwargs["output"].mkdir()
		return {"queries": 0}

	def timing(directory: Path, *, schedule: Path | None) -> dict[str, Any]:
		assert schedule is None  # Candidate timestamps are not actual scheduled arrivals.
		return {}

	monkeypatch.setattr(gateway_campaign, "collect_gateway", collect)
	monkeypatch.setattr(campaign_timing, "campaign_timing", timing)
	result = runner.invoke(
		gateway_cli.app,
		[
			"collect",
			"--terms-review-id",
			"fixture",
			"--input-dir",
			str(inventory),
			"--output",
			str(tmp_path / "run"),
			"--policy",
			"none",
		],
	)
	assert result.exit_code == 0, result.output
	assert captured["attempt_budget"] == 4
	assert captured["variants"]["online"]
	assert captured["variants"]["per_account_limit"] == 2
	assert captured["variants"]["round_size"] == 1
	help_result = runner.invoke(gateway_cli.app, ["schedule", "--help"])
	assert help_result.exit_code == 0
	assert "A8" in help_result.output
	assert "--topic-sorted" in help_result.output


@pytest.mark.asyncio
async def test_alternative_teacher_shares_http_budget_and_records_contract(tmp_path: Path) -> None:
	rows, manifest = campaign_schedule(
		pools=_pools(5), caps=CAPS, attack="A1", budget=9, interval_seconds=0
	)
	schedule = tmp_path / "requests.jsonl"
	write_jsonl(schedule, rows)
	teacher = Teacher()
	hosts: list[str] = []

	async def respond(request: httpx.Request) -> httpx.Response:
		hosts.append(request.url.host)
		result = await teacher.generate(
			TeacherGenerationRequest.model_validate_json(request.content)
		)
		return httpx.Response(200, json=result.model_dump(mode="json"))

	output = tmp_path / "run"
	result = await collect_gateway(
		schedule=schedule,
		output=output,
		url="http://primary",
		keys={"default": "fixture"},
		options=CollectionOptions(terms_review_id="fixture"),
		attempt_budget=4,
		alternative_url="http://alternative",
		alternative_share=0.5,
		transport=httpx.MockTransport(respond),
		selector_seed=42,
	)
	contract = json.loads((output / "contract.json").read_text())
	assert contract["alternative_teacher"]["share"] == 0.5
	assert contract["alternative_teacher"]["seed"] == 42
	assert result["queries"] == len(hosts) == 4
	ledger = read_jsonl(output / "ledger.jsonl", LedgerRecord)
	assert [record.teacher.endpoint.removeprefix("http://") for record in ledger] == hosts
	assert set(hosts) == {"primary", "alternative"}
	assert manifest["attempt_budget"] == 9


def test_a8_candidate_inventory_bytes_match_main() -> None:
	rows, manifest = campaign_schedule(pools=_pools(5), caps=CAPS, attack="A8", budget=9)
	payload = canonical_json([[r.model_dump(mode="json") for r in rows], manifest]).encode()
	assert (
		hashlib.sha256(payload).hexdigest()
		== "a06dc8effe0596bae6f1d38ec93074a121a338fe6412bdbb6579e30e52d89a1c"
	)


@pytest.mark.parametrize(
	"option", ["paraphrase", "user_prefix", "assistant_prefill", "human_bursts"]
)
def test_schedule_transforms_are_opt_in_and_declared(option: str) -> None:
	from distillation.attack_transforms import offline_rewrite, transform_inventory

	rows, _ = campaign_schedule(pools=_pools(5), caps=CAPS, attack="A1", budget=9)
	kwargs = {
		"paraphrase": {"rewriter": offline_rewrite, "rewriter_id": "offline-phrases-v1"},
		"user_prefix": {"user_prefix": "Answer briefly."},
		"assistant_prefill": {"assistant_prefill": "The answer is"},
		"human_bursts": {"bursts": True},
	}[option]
	transformed, settings = transform_inventory(rows, **kwargs)
	assert len(transformed) == len(rows)
	assert settings
	assert [r.request_id for r in transformed] == [r.request_id for r in rows]
	assert [r.model_dump() for r in transform_inventory(rows, **kwargs)[0]] == [
		r.model_dump() for r in transformed
	]
	if option == "assistant_prefill":
		assert transformed[0].prompt.prompt[-1].role == "assistant"
	if option == "user_prefix":
		assert transformed[0].prompt.prompt[0].content.startswith("Answer briefly.")
	if option == "human_bursts":
		assert transformed[6].time - transformed[5].time >= 600


@pytest.mark.asyncio
async def test_verifier_and_selective_retry_share_budget(tmp_path: Path) -> None:
	from distillation.benchmarks.models import BenchmarkItem

	rows, _ = campaign_schedule(
		pools=_pools(2), caps=CAPS, attack="A1", budget=3, interval_seconds=0
	)
	schedule = tmp_path / "requests.jsonl"
	write_jsonl(schedule, rows)
	items = [
		BenchmarkItem(
			id=r.prompt.example_id,
			benchmark="fixture",
			category="fixture",
			question="fixture",
			messages=r.prompt.prompt,
			metric="label_accuracy",
			answer="correct",
			choices=["correct", "wrong"],
		)
		for r in rows
	]
	verifier = tmp_path / "verifier.jsonl"
	write_jsonl(verifier, items)
	teacher = Teacher()
	calls = 0

	async def respond(request: httpx.Request) -> httpx.Response:
		nonlocal calls
		calls += 1
		generation = await teacher.generate(
			TeacherGenerationRequest.model_validate_json(request.content)
		)
		generation.message.content = "wrong" if calls == 1 else "correct"
		return httpx.Response(200, json=generation.model_dump(mode="json"))

	output = tmp_path / "run"
	result = await collect_gateway(
		schedule=schedule,
		output=output,
		url="http://fixture",
		keys={"default": "fixture"},
		options=CollectionOptions(terms_review_id="fixture"),
		attempt_budget=2,
		verifier_file=verifier,
		selective_retries=1,
		transport=httpx.MockTransport(respond),
	)
	assert calls == result["queries"] == 2
	assert result["ledger_records"] == 1
	contract = json.loads((output / "contract.json").read_text())
	assert contract["response_selection"]["selective_retries"] == 1
	assert (
		contract["response_selection"]["verifier_sha256"]
		== hashlib.sha256(verifier.read_bytes()).hexdigest()
	)


@pytest.mark.asyncio
async def test_session_replay_consumes_shared_budget_without_entering_ledger(
	tmp_path: Path,
) -> None:
	from distillation.attack_transforms import replay_sessions

	rows, _ = campaign_schedule(
		pools=_pools(4), caps=CAPS, attack="A1", budget=6, interval_seconds=0
	)
	benign = []
	for index, row in enumerate(rows[:2]):
		prompt = row.prompt.model_copy(update={"example_id": f"cover-{index}"})
		benign.append(
			row.model_copy(
				update={"request_id": f"cover-{index}", "prompt": prompt, "account": "cover"}
			)
		)
	combined, replay_ids = replay_sessions(rows, benign, 2)
	schedule = tmp_path / "requests.jsonl"
	write_jsonl(schedule, combined)
	teacher = Teacher()

	async def respond(request: httpx.Request) -> httpx.Response:
		result = await teacher.generate(
			TeacherGenerationRequest.model_validate_json(request.content)
		)
		return httpx.Response(200, json=result.model_dump(mode="json"))

	output = tmp_path / "run"
	result = await collect_gateway(
		schedule=schedule,
		output=output,
		url="http://fixture",
		keys={"default": "fixture"},
		options=CollectionOptions(terms_review_id="fixture"),
		attempt_budget=6,
		variants={"session_replay": {"request_ids": replay_ids, "sha256": "fixture"}},
		transport=httpx.MockTransport(respond),
	)
	assert result["queries"] == 6
	assert result["ledger_records"] == 4
	assert (
		json.loads((output / "contract.json").read_text())["variants"]["session_replay"][
			"request_ids"
		]
		== replay_ids
	)


@pytest.mark.asyncio
async def test_batched_requests_charge_once_and_retain_members(tmp_path: Path) -> None:
	from distillation.attack_transforms import batch_inventory

	rows, _ = campaign_schedule(
		pools=_pools(4), caps=CAPS, attack="A2", budget=6, interval_seconds=0
	)
	batched, members = batch_inventory(rows, 2)
	assert members
	assert all(row.max_output_tokens <= CAPS[row.domain] for row in batched)
	schedule = tmp_path / "requests.jsonl"
	write_jsonl(schedule, batched)
	teacher = Teacher()

	async def respond(request: httpx.Request) -> httpx.Response:
		result = await teacher.generate(
			TeacherGenerationRequest.model_validate_json(request.content)
		)
		result.message.content = json.dumps(["answer", "answer"])
		return httpx.Response(200, json=result.model_dump(mode="json"))

	output = tmp_path / "run"
	result = await collect_gateway(
		schedule=schedule,
		output=output,
		url="http://fixture",
		keys={"default": "fixture"},
		options=CollectionOptions(terms_review_id="fixture"),
		attempt_budget=1,
		variants={"batching": {"size": 2, "members": members}},
		transport=httpx.MockTransport(respond),
	)
	assert result["queries"] == 1
	assert result["ledger_records"] == 2
	assert json.loads((output / "contract.json").read_text())["variants"]["batching"]["size"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
	"variant", ["rewriter", "user_prefix", "assistant_prefill", "arrival_pattern"]
)
async def test_schedule_variant_contract_and_http_budget(tmp_path: Path, variant: str) -> None:
	from distillation.attack_transforms import offline_rewrite, transform_inventory

	rows, _ = campaign_schedule(
		pools=_pools(3), caps=CAPS, attack="A1", budget=3, interval_seconds=0
	)
	settings = {
		"rewriter": {"rewriter": offline_rewrite, "rewriter_id": "offline-phrases-v1"},
		"user_prefix": {"user_prefix": "Answer briefly."},
		"assistant_prefill": {"assistant_prefill": "Answer:"},
		"arrival_pattern": {"bursts": True},
	}[variant]
	rows, declared = transform_inventory(rows, **settings)
	schedule = tmp_path / "requests.jsonl"
	write_jsonl(schedule, rows)
	teacher = Teacher()

	async def respond(request: httpx.Request) -> httpx.Response:
		result = await teacher.generate(
			TeacherGenerationRequest.model_validate_json(request.content)
		)
		return httpx.Response(200, json=result.model_dump(mode="json"))

	output = tmp_path / "run"
	result = await collect_gateway(
		schedule=schedule,
		output=output,
		url="http://fixture",
		keys={"default": "fixture"},
		options=CollectionOptions(terms_review_id="fixture"),
		attempt_budget=1,
		variants=declared,
		transport=httpx.MockTransport(respond),
	)
	assert result["queries"] == 1
	assert (
		json.loads((output / "contract.json").read_text())["variants"][variant] == declared[variant]
	)


@pytest.mark.asyncio
async def test_low_confidence_retry_records_scorer_and_spends_budget(tmp_path: Path) -> None:
	rows, _ = campaign_schedule(
		pools=_pools(2), caps=CAPS, attack="A1", budget=3, interval_seconds=0
	)
	schedule = tmp_path / "requests.jsonl"
	write_jsonl(schedule, rows)
	teacher = Teacher()
	calls = 0

	async def respond(request: httpx.Request) -> httpx.Response:
		nonlocal calls
		calls += 1
		result = await teacher.generate(
			TeacherGenerationRequest.model_validate_json(request.content)
		)
		result.message.content = "uncertain" if calls == 1 else "confident"
		return httpx.Response(200, json=result.model_dump(mode="json"))

	output = tmp_path / "run"
	result = await collect_gateway(
		schedule=schedule,
		output=output,
		url="http://fixture",
		keys={"default": "fixture"},
		options=CollectionOptions(terms_review_id="fixture"),
		attempt_budget=2,
		selective_retries=1,
		confidence_scorer=lambda record: (
			0.1 if record.completion[0].content == "uncertain" else 0.9
		),
		confidence_scorer_id="fixture:confidence",
		confidence_threshold=0.5,
		transport=httpx.MockTransport(respond),
	)
	assert calls == result["queries"] == 2
	assert result["ledger_records"] == 1
	selection = json.loads((output / "contract.json").read_text())["response_selection"]
	assert selection["confidence_scorer"] == "fixture:confidence"
	assert selection["confidence_threshold"] == 0.5
