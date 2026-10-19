"""Gateway inputs preserve paired prompts, account profiles, and one campaign budget."""

import hashlib
import json
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import pytest
from distillation.benchmarks.models import BenchmarkItem
from distillation.external_sources import EXTERNAL_SOURCES
from distillation.gateway_cli import app
from distillation.io import read_jsonl, write_jsonl
from distillation.schedules import Attack, campaign_schedule
from distillation.schemas import GatewayRequest, Message, PromptRecord, PromptSource
from distillation.tasks import AG_NEWS_PROMPT, GSM8K_EVAL_SUFFIX, SQUAD_PROMPT
from distillation.workloads import (
	CLIENT_TYPES,
	LEGACY_CONCENTRATED_SHARE,
	assign_types,
	client_shares,
	clients,
	mixed_clients,
)
from typer.testing import CliRunner

CAPS = {"ag_news": 16, "squad": 64, "gsm8k": 256}


def _pools(count: int = 100, *, split: str = "train") -> dict[str, list[PromptRecord]]:
	return {
		domain: [
			PromptRecord(
				example_id=f"{domain}-{index}",
				split="train" if split == "train" else "test",
				task=domain,
				source=PromptSource(
					corpus=domain,
					source_id=str(index),
					group_id=f"source-{index // 4}",
					license="test",
				),
				prompt=[
					Message(
						role="user",
						content=(
							AG_NEWS_PROMPT.format(text=f"Story {index}")
							if domain == "ag_news"
							else SQUAD_PROMPT.format(context=f"Passage {index}", question="What?")
							if domain == "squad"
							else f"Problem {index}{GSM8K_EVAL_SUFFIX}"
						),
					)
				],
			)
			for index in range(count)
		]
		for domain in CAPS
	}


def test_attack_schedules_pair_prompts_and_share_the_budget_across_specialist_accounts() -> None:
	pools = _pools()
	attacks: tuple[Attack, ...] = ("A1", "A2", "A5")
	results = {
		attack: campaign_schedule(
			pools=pools,
			caps=CAPS,
			attack=attack,
			budget=61,
			accounts=5 if attack == "A5" else 1,
		)
		for attack in attacks
	}
	ids = [{r.prompt.example_id for r in rows} for rows, _ in results.values()]
	assert ids[0] == ids[1] == ids[2]
	for rows, manifest in results.values():
		assert len(rows) == len({r.request_id for r in rows}) == 61
		assert manifest["attempt_budget"] == 61
		assert manifest["domain_allocation"] == {"ag_news": 21, "gsm8k": 20, "squad": 20}
		assert Counter(r.domain for r in rows) == manifest["domain_allocation"]
		assert [r.time for r in rows] == list(range(61))
		assert all(r.max_output_tokens == CAPS[r.domain] for r in rows)
		assert all(r.prompt in pools[r.domain] for r in rows)
		assert manifest["execution_status"] == "not_run"
	sequential = results["A2"][0]
	assert sum(a.domain != b.domain for a, b in pairwise(sequential)) == 2
	interleaved = results["A1"][0]
	assert sum(a.domain != b.domain for a, b in pairwise(interleaved)) > 2
	assert len({r.account for r in interleaved}) == 1
	sybil, manifest = results["A5"]
	assert len(manifest["accounts"]) == 5
	assert all(len(domains) == 1 for domains in manifest["account_domains"].values())
	assert sum(Counter(r.account for r in sybil).values()) == 61
	assert results["A1"] == campaign_schedule(pools=pools, caps=CAPS, attack="A1", budget=61)
	assert (
		results["A1"][0]
		!= campaign_schedule(pools=pools, caps=CAPS, attack="A1", budget=61, seed=1)[0]
	)
	assert results["A1"] == campaign_schedule(
		pools={k: list(reversed(v)) for k, v in reversed(list(pools.items()))},
		caps=CAPS,
		attack="A1",
		budget=61,
	)


@pytest.mark.parametrize("share", [0.0, 0.1, 0.3, 1.0])
def test_mixed_benign_profiles_preserve_domain_identity_and_disclose_realized_mix(
	share: float,
) -> None:
	pools = _pools(400, split="test")
	rows, manifest = mixed_clients(
		pools=pools,
		caps=CAPS,
		role="benign_test",
		accounts=10,
		generalist_share=share,
		requests_per_account=30,
		concentrated=LEGACY_CONCENTRATED_SHARE,
	)
	by_account: dict[str, list[GatewayRequest]] = defaultdict(list)
	for row in rows:
		by_account[row.account].append(row)
		assert row.max_output_tokens == CAPS[row.domain]
		assert row.prompt in pools[row.domain]
	assert len(rows) == len({r.request_id for r in rows})
	assert [r.time for r in rows] == sorted(r.time for r in rows)
	assert manifest["generalist_accounts"] == share * 10
	assert manifest["generalist_share_achieved"] == share
	assert manifest["source_documents"] == len(
		{(r.prompt.source.corpus, r.prompt.source.group_id) for r in rows}
	)
	assert set(manifest["client_types"]) == {"specialist", "repetitive", "batch", "bursty", "short"}
	for account_rows in by_account.values():
		generalist = account_rows[0].domain_profile == "generalist"
		assert (len({r.domain for r in account_rows}) >= 2) == generalist
		if account_rows[0].client_type == "repetitive":
			assert set(Counter(r.prompt.example_id for r in account_rows).values()) == {3}
		if account_rows[0].client_type == "short":
			assert len(account_rows) == 20
		if account_rows[0].client_type == "batch":
			assert account_rows[1].time - account_rows[0].time == pytest.approx(0.1)
		if account_rows[0].client_type == "bursty":
			assert account_rows[25].time == 60
	# Repetition is confined to one account, and input ordering never changes sampling.
	owners: dict[str, set[str]] = defaultdict(set)
	for row in rows:
		owners[row.prompt.example_id].add(row.account)
	assert all(len(accounts) == 1 for accounts in owners.values())
	assert (rows, manifest) == mixed_clients(
		pools={domain: list(reversed(pool)) for domain, pool in pools.items()},
		caps=CAPS,
		role="benign_test",
		accounts=10,
		generalist_share=share,
		requests_per_account=30,
		concentrated=LEGACY_CONCENTRATED_SHARE,
	)


def test_benign_account_lengths_are_heavy_tailed_and_capacity_limited() -> None:
	pools = _pools(2000, split="test")
	rows, manifest = mixed_clients(
		pools=pools, caps=CAPS, role="benign_test", accounts=30, requests_per_account=30
	)
	lengths = Counter(r.account for r in rows)
	kinds = {r.account: r.client_type for r in rows}
	long = [n for account, n in lengths.items() if kinds[account] != "short"]
	assert min(long) >= 30 and max(long) > 60
	assert manifest["multi_window_accounts"] == sum(n >= 200 for n in lengths.values())
	assert manifest["length_distribution"]["pareto_shape"] == 1.5
	assert all(
		lengths[a] == manifest["account_lengths_requested"][a]
		for a in lengths
		if a not in manifest["capacity_limited_accounts"]
	)
	fixed, _ = mixed_clients(
		pools=pools,
		caps=CAPS,
		role="benign_test",
		accounts=30,
		requests_per_account=30,
		length_tail=None,
	)
	assert {n for a, n in Counter(r.account for r in fixed).items() if kinds[a] != "short"} == {30}
	# A tail larger than the pools is cut at the remaining capacity and disclosed.
	small, small_manifest = mixed_clients(
		pools=_pools(120, split="test"),
		caps=CAPS,
		role="benign_test",
		accounts=6,
		requests_per_account=20,
		max_requests_per_account=2000,
		length_tail=0.2,
	)
	assert small_manifest["capacity_limited_accounts"]
	assert len({r.prompt.example_id for r in small}) <= 360
	with pytest.raises(ValueError, match="max_requests_per_account"):
		mixed_clients(
			pools=pools,
			caps=CAPS,
			role="benign_test",
			accounts=2,
			requests_per_account=30,
			max_requests_per_account=10,
		)


def test_concentrated_share_sets_the_benign_client_mix() -> None:
	# The v1 equal split is plain round-robin; other shares are met to within one slot.
	assert assign_types(10, client_shares(LEGACY_CONCENTRATED_SHARE)) == [
		CLIENT_TYPES[i % 5] for i in range(10)
	]
	for share in (0.0, 0.1, 0.4):
		counts = Counter(assign_types(127, client_shares(share)))
		for kind, expected in client_shares(share).items():
			assert abs(counts[kind] - expected * 127) < 1
	with pytest.raises(ValueError, match="concentrated share"):
		client_shares(0.9)
	prompts = [prompt for pool in _pools(100).values() for prompt in pool]
	broad = clients(prompts, "fitting", concentrated=0.0)
	assert {r.client_type for r in broad} == {"batch", "bursty", "short"}
	assert len(broad) == len(prompts)


def test_gateway_rejects_unusable_pools_and_impossible_allocations() -> None:
	pools = _pools(10)
	with pytest.raises(ValueError, match="pool size"):
		campaign_schedule(pools=pools, caps=CAPS, attack="A1", budget=31)
	with pytest.raises(ValueError, match="at least one account per domain"):
		campaign_schedule(pools=pools, caps=CAPS, attack="A5", budget=12, accounts=2)
	with pytest.raises(ValueError, match="exactly one account"):
		campaign_schedule(pools=pools, caps=CAPS, attack="A2", budget=12, accounts=2)
	with pytest.raises(ValueError, match="sum to"):
		campaign_schedule(
			pools=pools, caps=CAPS, attack="A1", budget=12, allocation=dict.fromkeys(CAPS, 3)
		)
	with pytest.raises(ValueError, match="training-source"):
		campaign_schedule(pools=_pools(split="test"), caps=CAPS, attack="A1", budget=12)
	with pytest.raises(ValueError, match="exhaust"):
		mixed_clients(pools=pools, caps=CAPS, role="fitting", accounts=10)
	with pytest.raises(ValueError, match="generalist_share"):
		mixed_clients(
			pools=pools, caps=CAPS, role="fitting", accounts=2, generalist_share=float("nan")
		)
	with pytest.raises(ValueError, match="two distinct"):
		mixed_clients(
			pools=pools,
			caps=CAPS,
			role="fitting",
			accounts=1,
			generalist_share=1,
			requests_per_account=1,
		)
	pools["squad"][0] = pools["ag_news"][0].model_copy(update={"task": "squad"})
	with pytest.raises(ValueError, match="duplicate example_id"):
		campaign_schedule(pools=pools, caps=CAPS, attack="A1", budget=12)


@pytest.mark.parametrize("include_coding", [False, True])
def test_gateway_cli_writes_hashed_inputs_without_overwriting_them(
	tmp_path: Path, include_coding: bool
) -> None:
	pools = _pools(40)
	caps = dict(CAPS)
	if include_coding:
		pytest.importorskip("coding", reason="coding fixtures need the coding extra")
		from coding.dataset import PROMPT

		caps["coding"] = 512
		pools["coding"] = [
			PromptRecord(
				example_id=f"coding-{i}",
				split="train",
				task="coding",
				source=PromptSource(
					corpus="fixture-coding",
					source_id=str(i),
					group_id=f"coding-{i}",
					license="test",
				),
				prompt=[
					Message(
						role="user",
						content=PROMPT.format(
							specification=f"Return {i}", signature="def value():"
						),
					)
				],
			)
			for i in range(40)
		]
	for domain, prompts in pools.items():
		directory = tmp_path / "tasks" / domain
		write_jsonl(directory / "attacker.jsonl", prompts)
		write_jsonl(directory / "legitimate" / "fitting.jsonl", prompts)
		(directory / "legitimate-references.json").write_text(
			json.dumps(
				{
					prompt.example_id: BenchmarkItem(
						id=prompt.example_id,
						benchmark=domain,
						category="math"
						if domain == "gsm8k"
						else str(int(prompt.source.source_id) % 2),
						question="fixture",
						messages=prompt.prompt,
						metric="label_accuracy",
						answer="fixture",
					).model_dump(mode="json")
					for prompt in prompts
				}
			)
		)
		(directory / "manifest.json").write_text(
			json.dumps({"task": domain, "max_output_tokens": caps[domain]})
		)
	runner = CliRunner()
	common = ["--tasks-dir", str(tmp_path / "tasks")]
	if not include_coding:
		common += ["--domains", ",".join(CAPS)]
	args = [
		"schedule",
		*common,
		"--attack",
		"A5",
		"--budget",
		"30",
		"--output",
		str(tmp_path / "campaign"),
	]
	result = runner.invoke(app, args)
	assert result.exit_code == 0, result.output
	manifest = json.loads(result.output)
	path = tmp_path / "campaign" / "requests.jsonl"
	assert manifest["sha256"][path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
	assert len(read_jsonl(path, GatewayRequest)) == 30
	assert len(manifest["accounts"]) == 6
	assert set(manifest["domain_allocation"]) == set(caps)
	assert json.loads((path.parent / "manifest.json").read_text()) == manifest
	assert runner.invoke(app, args).exit_code != 0
	result = runner.invoke(
		app,
		[
			"benign",
			*common,
			"--role",
			"fitting",
			"--accounts",
			"5",
			"--generalist-share",
			"0.3",
			"--concentrated-share",
			"0.4",
			"--requests-per-account",
			"12",
			"--output",
			str(tmp_path / "benign"),
		],
	)
	assert result.exit_code == 0, result.output
	assert json.loads(result.output)["generalist_share_achieved"] == 0.4
	assert json.loads(result.output)["topic_order"] == "category"
	assert json.loads(result.output)["specialist_scope"]["gsm8k"] == "domain only"
	assert json.loads(result.output)["specialist_scope"]["ag_news"] == "topic ordered"
	rows = read_jsonl(tmp_path / "benign" / "requests.jsonl", GatewayRequest)
	for account in {row.account for row in rows}:
		account_rows = [row for row in rows if row.account == account]
		if account_rows[0].domain_profile == "specialist" and account_rows[0].domain != "gsm8k":
			assert {int(row.prompt.source.source_id) % 2 for row in account_rows} == {0}
	result = runner.invoke(
		app,
		[
			"benign",
			*common,
			"--role",
			"attacker",
			"--accounts",
			"1",
			"--output",
			str(tmp_path / "bad"),
		],
	)
	assert result.exit_code != 0
	assert not (tmp_path / "bad").exists()


def test_a5_defaults_to_six_total_accounts() -> None:
	rows, manifest = campaign_schedule(pools=_pools(), caps=CAPS, attack="A5", budget=30)
	assert len(manifest["accounts"]) == 6
	assert len(rows) == 30
	assert Counter(domains[0] for domains in manifest["account_domains"].values()) == dict.fromkeys(
		CAPS, 2
	)


@pytest.mark.asyncio
async def test_live_campaign_charges_failures_and_wakes_future_accounts_when_budget_is_spent(
	tmp_path: Path,
) -> None:
	import asyncio

	import httpx
	from distillation.collection import CollectionOptions
	from distillation.gateway_campaign import collect_gateway
	from test_harness_api import Teacher

	rows, _ = campaign_schedule(
		pools=_pools(), caps=CAPS, attack="A5", budget=6, interval_seconds=0
	)
	rows[-1].time = 1e6
	schedule = tmp_path / "requests.jsonl"
	write_jsonl(schedule, rows)
	calls = 0
	teacher = Teacher()

	async def respond(request: httpx.Request) -> httpx.Response:
		nonlocal calls
		calls += 1
		if calls == 1:
			return httpx.Response(500)
		from distillation.schemas import TeacherGenerationRequest

		result = await teacher.generate(
			TeacherGenerationRequest.model_validate_json(request.content)
		)
		return httpx.Response(200, json=result.model_dump(mode="json"))

	async with asyncio.timeout(1):
		result = await collect_gateway(
			schedule=schedule,
			output=tmp_path / "campaign",
			url="http://fixture",
			keys={"default": "fixture"},
			options=CollectionOptions(terms_review_id="fixture"),
			attempt_budget=2,
			transport=httpx.MockTransport(respond),
		)
	assert result["queries"] == calls == 2
	assert result["failed"] == result["received"] == 1
	assert result["pending"] == 0


@pytest.mark.asyncio
async def test_live_gateway_deadline_includes_waiting_and_pending_http(tmp_path: Path) -> None:
	import asyncio

	import httpx
	from distillation.collection import CollectionOptions
	from distillation.gateway_campaign import collect_gateway

	rows, _ = campaign_schedule(
		pools=_pools(), caps=CAPS, attack="A5", budget=6, interval_seconds=0
	)
	schedule = tmp_path / "requests.jsonl"
	write_jsonl(schedule, rows)

	async def respond(request: httpx.Request) -> httpx.Response:
		await asyncio.sleep(10)
		return httpx.Response(500)

	async with asyncio.timeout(1):
		result = await collect_gateway(
			schedule=schedule,
			output=tmp_path / "campaign",
			url="http://fixture",
			keys={"default": "fixture"},
			options=CollectionOptions(terms_review_id="fixture", concurrency=2),
			attempt_budget=6,
			deadline_seconds=0.02,
			transport=httpx.MockTransport(respond),
		)
	assert result["queries"] == result["deadline"] == 2
	assert result["pending"] == result["received"] == 0
	assert set(result["termination_by_account"].values()) == {"deadline"}


def _external(pools: dict[str, list[PromptRecord]]) -> dict[str, list[PromptRecord]]:
	return {
		domain: [
			p.model_copy(
				update={
					"example_id": p.example_id.replace(domain, f"{domain}-external", 1),
					"source": p.source.model_copy(
						update={"corpus": EXTERNAL_SOURCES[domain].dataset.repository}
					),
				}
			)
			for p in pool
		]
		for domain, pool in pools.items()
	}


def test_extended_strategies_share_one_budget_and_disclose_their_sources() -> None:
	pools = _pools()
	external = _external(pools)
	sybil, manifest = campaign_schedule(pools=pools, caps=CAPS, attack="A6", budget=60)
	baseline, _ = campaign_schedule(pools=pools, caps=CAPS, attack="A1", budget=60)
	assert {r.prompt.example_id for r in sybil} == {r.prompt.example_id for r in baseline}
	assert len(manifest["accounts"]) == 6
	assert all(len(domains) >= 2 for domains in manifest["account_domains"].values())
	independent, manifest = campaign_schedule(pools=external, caps=CAPS, attack="A7", budget=60)
	assert manifest["prompt_sources"] == "independent external"
	assert len({r.account for r in independent}) == 1
	assert all(
		r.prompt.source.corpus == EXTERNAL_SOURCES[r.domain].dataset.repository for r in independent
	)
	covered, manifest = campaign_schedule(
		pools=external, caps=CAPS, attack="A3", budget=60, cover_pools=pools, cover_share=0.5
	)
	kinds = Counter(r.client_type for r in covered)
	assert kinds == {"cover": 30, "extraction": 30}
	assert manifest["attempt_budget"] == len(covered) == 60
	assert sum(manifest["cover_allocation"].values()) == 30
	assert all((r.client_type == "cover") == (r.prompt.source.corpus == r.domain) for r in covered)
	with pytest.raises(ValueError, match="external"):
		campaign_schedule(pools=pools, caps=CAPS, attack="A7", budget=60)
	with pytest.raises(ValueError, match="task attacker pools"):
		campaign_schedule(pools=external, caps=CAPS, attack="A1", budget=60)
	with pytest.raises(ValueError, match="cover pools"):
		campaign_schedule(pools=external, caps=CAPS, attack="A3", budget=60)
	with pytest.raises(ValueError, match="disjoint"):
		campaign_schedule(pools=external, caps=CAPS, attack="A3", budget=60, cover_pools=external)
	with pytest.raises(ValueError, match="at least two accounts"):
		campaign_schedule(pools=pools, caps=CAPS, attack="A6", budget=60, accounts=1)
