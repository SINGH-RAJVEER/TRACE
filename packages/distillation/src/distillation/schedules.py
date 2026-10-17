"""Fixed multi-domain campaign inventories, before admission and response collection."""

import hashlib
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from distillation.io import canonical_json
from distillation.schemas import GatewayRequest, PromptRecord
from distillation.tasks import TASKS

SCHEDULE_VERSION = "gateway-schedules-v1"
Attack = Literal["A1", "A2", "A5"]


def validate_pools(pools: Mapping[str, Sequence[PromptRecord]], caps: Mapping[str, int]) -> None:
	"""Validate domain membership and retain globally unambiguous prompt identities."""
	if len(pools) < 2 or set(pools) - TASKS.keys():
		raise ValueError("gateway pools require at least two supported domains")
	if set(caps) != set(pools) or any(
		type(cap) is not int or not 1 <= cap <= 32768 for cap in caps.values()
	):
		raise ValueError("each domain requires its frozen output cap between 1 and 32768")
	seen = set()
	for domain, prompts in pools.items():
		if not prompts:
			raise ValueError(f"empty domain pool: {domain}")
		for prompt in prompts:
			if prompt.task != domain or not prompt.source.group_id:
				raise ValueError(f"domain mismatch or missing source group: {prompt.example_id}")
			if prompt.example_id in seen:
				raise ValueError(f"duplicate example_id across gateway pools: {prompt.example_id}")
			if not any(message.role == "user" for message in prompt.prompt):
				raise ValueError(f"missing user message: {prompt.example_id}")
			seen.add(prompt.example_id)


def pool_hashes(pools: Mapping[str, Sequence[PromptRecord]]) -> dict[str, str]:
	return {
		domain: hashlib.sha256(
			canonical_json(
				[p.model_dump(mode="json") for p in sorted(prompts, key=lambda p: p.example_id)]
			).encode()
		).hexdigest()
		for domain, prompts in sorted(pools.items())
	}


def campaign_schedule(
	*,
	pools: Mapping[str, Sequence[PromptRecord]],
	caps: Mapping[str, int],
	attack: Attack,
	budget: int,
	seed: int = 42,
	campaign_id: str = "campaign",
	accounts: int | None = None,
	allocation: Mapping[str, int] | None = None,
	interval_seconds: float = 1.0,
	deadline_seconds: float = 86400,
) -> tuple[list[GatewayRequest], dict[str, Any]]:
	"""Pair prompt selection across A1/A2/A5, changing order and account assignment only.

	The budget caps all future HTTP submissions, including proofs. The inventory contains
	at most that many ordinary queries; solving a challenge may leave later queries unsent.
	A5 partitions k specialist accounts across domains without multiplying this budget.
	"""
	validate_pools(pools, caps)
	domains = sorted(pools)
	if attack not in {"A1", "A2", "A5"}:
		raise ValueError("attack must be A1, A2, or A5")
	if not campaign_id.strip() or type(budget) is not int or budget < len(domains):
		raise ValueError("campaign needs an identity and at least one query per domain")
	if not math.isfinite(interval_seconds) or interval_seconds < 0:
		raise ValueError("interval_seconds must be finite and nonnegative")
	if not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
		raise ValueError("deadline_seconds must be finite and positive")
	if any(prompt.split != "train" for pool in pools.values() for prompt in pool):
		raise ValueError("attacker pools must contain training-source prompts only")
	if allocation is None:
		base, extra = divmod(budget, len(domains))
		allocation = {domain: base + (index < extra) for index, domain in enumerate(domains)}
	if (
		set(allocation) != set(domains)
		or any(type(count) is not int or count < 1 for count in allocation.values())
		or sum(allocation.values()) != budget
	):
		raise ValueError("domain allocation must be positive and sum to the campaign budget")
	selected = {}
	for domain in domains:
		pool = sorted(pools[domain], key=lambda prompt: prompt.example_id)
		if allocation[domain] > len(pool):
			raise ValueError(f"domain allocation exceeds {domain} pool size {len(pool)}")
		random.Random(f"{seed}:prompts:{domain}").shuffle(pool)
		selected[domain] = pool[: allocation[domain]]
	k = (6 if attack == "A5" else 1) if accounts is None else accounts
	if type(k) is not int or (attack != "A5" and k != 1):
		raise ValueError("A1 and A2 require exactly one account")
	if attack == "A5" and not len(domains) <= k <= budget:
		raise ValueError("A5 needs at least one account per domain and at most one per query")
	account_map = {}
	for index, domain in enumerate(domains):
		count = k // len(domains) + (index < k % len(domains)) if attack == "A5" else 1
		if count > allocation[domain]:
			raise ValueError(f"A5 allocation cannot populate all {domain} accounts")
		account_map[domain] = [
			f"{campaign_id}-{domain}-{i:03d}" if attack == "A5" else f"{campaign_id}-account"
			for i in range(count)
		]
	order = list(domains)
	random.Random(f"{seed}:domain-order").shuffle(order)
	prompts = [prompt for domain in order for prompt in selected[domain]]
	if attack != "A2":
		random.Random(f"{seed}:interleave").shuffle(prompts)
	used: Counter[str] = Counter()
	requests = []
	for index, prompt in enumerate(prompts):
		domain = prompt.task
		choices = account_map[domain]
		requests.append(
			GatewayRequest(
				request_id=f"{campaign_id}-{index:06d}",
				account=choices[used[domain] % len(choices)],
				time=index * interval_seconds,
				prompt=prompt,
				max_output_tokens=caps[domain],
				domain_profile="specialist" if attack == "A5" else "generalist",
			)
		)
		used[domain] += 1
	account_domains: dict[str, set[str]] = {}
	for request in requests:
		account_domains.setdefault(request.account, set()).add(request.domain)
	manifest = {
		"version": SCHEDULE_VERSION,
		"campaign_id": campaign_id,
		"attack": attack,
		"seed": seed,
		"attempt_budget": budget,
		"budget_unit": "campaign-wide HTTP submissions, including proof redemption and failures",
		"deadline_seconds": deadline_seconds,
		"planned_queries": len(requests),
		"domain_allocation": dict(allocation),
		"accounts": sorted(account_domains),
		"account_domains": {
			account: sorted(domains) for account, domains in sorted(account_domains.items())
		},
		"interval_seconds": interval_seconds,
		"arrival_times": "synthetic, global campaign schedule",
		"max_output_tokens": dict(caps),
		"pool_sha256": pool_hashes(pools),
		"execution_status": "not_run",
	}
	return requests, manifest
