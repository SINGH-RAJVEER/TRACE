"""Gateway inputs preserve paired prompts, account profiles, and one campaign budget."""

import hashlib
import json
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import pytest
from distillation.benchmarks.models import BenchmarkItem
from distillation.gateway_cli import app
from distillation.io import read_jsonl, write_jsonl
from distillation.schedules import Attack, campaign_schedule
from distillation.schemas import GatewayRequest, Message, PromptRecord, PromptSource
from distillation.tasks import AG_NEWS_PROMPT, GSM8K_EVAL_SUFFIX, SQUAD_PROMPT
from distillation.workloads import mixed_clients
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
	)


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


def test_gateway_cli_writes_hashed_inputs_without_overwriting_them(tmp_path: Path) -> None:
	for domain, prompts in _pools(40).items():
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
			json.dumps({"task": domain, "max_output_tokens": CAPS[domain]})
		)
	runner = CliRunner()
	common = ["--tasks-dir", str(tmp_path / "tasks"), "--domains", ",".join(CAPS)]
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
