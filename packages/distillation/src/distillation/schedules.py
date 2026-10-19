"""Fixed multi-domain campaign inventories, before admission and response collection."""

import hashlib
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from distillation.external_sources import EXTERNAL_SOURCES
from distillation.io import canonical_json
from distillation.schemas import GatewayRequest, PromptRecord
from distillation.tasks import TASKS

SCHEDULE_VERSION = "gateway-schedules-v2"
Attack = Literal["A1", "A2", "A3", "A5", "A6", "A7"]
ATTACKS: tuple[Attack, ...] = ("A1", "A2", "A3", "A5", "A6", "A7")
# Strategies whose extraction prompts must come from independently pinned sources.
EXTERNAL_ATTACKS = frozenset({"A3", "A7"})


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
	cover_pools: Mapping[str, Sequence[PromptRecord]] | None = None,
	cover_share: float = 0.5,
) -> tuple[list[GatewayRequest], dict[str, Any]]:
	"""Pair prompt selection across strategies, changing order and account assignment only.

	The budget caps all future HTTP submissions, including proofs. The inventory contains
	at most that many ordinary queries; solving a challenge may leave later queries unsent.
	A5 partitions k specialist accounts across domains and A6 splits the multiplexed stream
	across k accounts, neither multiplying this budget. A7 multiplexes prompts from
	independently pinned sources in one account. A3 interleaves A7 extraction with
	`cover_share` of the budget drawn from in-distribution `cover_pools`; every received
	response, cover included, may train the student. A1, A2, A5, and A6 select identical
	prompts for a matched seed, pools, and allocation.
	"""
	validate_pools(pools, caps)
	domains = sorted(pools)
	if attack not in ATTACKS:
		raise ValueError(f"attack must be one of {', '.join(ATTACKS)}")
	if not campaign_id.strip() or type(budget) is not int or budget < len(domains):
		raise ValueError("campaign needs an identity and at least one query per domain")
	external = {
		domain: all(p.source.corpus == EXTERNAL_SOURCES[domain].dataset.repository for p in pool)
		for domain, pool in pools.items()
	}
	if attack in EXTERNAL_ATTACKS and not all(external.values()):
		raise ValueError(f"{attack} requires independently pinned external prompt pools")
	if attack not in EXTERNAL_ATTACKS and any(external.values()):
		raise ValueError(f"{attack} uses the task attacker pools, not external sources")
	if (attack == "A3") != (cover_pools is not None):
		raise ValueError("cover pools are required for A3 and only for A3")
	if attack == "A3":
		if not math.isfinite(cover_share) or not 0 < cover_share < 1:
			raise ValueError("cover_share must be between zero and one")
		assert cover_pools is not None
		validate_pools(cover_pools, caps)
		if set(cover_pools) != set(pools):
			raise ValueError("cover pools must cover the same domains")
		if {p.example_id for v in cover_pools.values() for p in v} & {
			p.example_id for v in pools.values() for p in v
		}:
			raise ValueError("cover and extraction pools must be disjoint")
	cover_budget = round(budget * cover_share) if attack == "A3" else 0
	extraction_budget = budget - cover_budget
	if extraction_budget < len(domains) or (attack == "A3" and cover_budget < len(domains)):
		raise ValueError("cover and extraction each need at least one query per domain")
	if not math.isfinite(interval_seconds) or interval_seconds < 0:
		raise ValueError("interval_seconds must be finite and nonnegative")
	if not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
		raise ValueError("deadline_seconds must be finite and positive")
	sources = [pools, *([cover_pools] if cover_pools is not None else [])]
	if any(
		prompt.split != "train" for pool_set in sources for v in pool_set.values() for prompt in v
	):
		raise ValueError("attacker pools must contain training-source prompts only")
	if allocation is None:
		allocation = _balanced(extraction_budget, domains)
	if (
		set(allocation) != set(domains)
		or any(type(count) is not int or count < 1 for count in allocation.values())
		or sum(allocation.values()) != extraction_budget
	):
		raise ValueError("domain allocation must be positive and sum to the campaign budget")
	selected = _select(pools, allocation, seed, "prompts")
	cover_allocation = _balanced(cover_budget, domains) if cover_pools is not None else {}
	cover = _select(cover_pools, cover_allocation, seed, "cover") if cover_pools is not None else {}
	k = (6 if attack in {"A5", "A6"} else 1) if accounts is None else accounts
	if type(k) is not int or (attack not in {"A5", "A6"} and k != 1):
		raise ValueError(f"{attack} requires exactly one account")
	if attack == "A5" and not len(domains) <= k <= budget:
		raise ValueError("A5 needs at least one account per domain and at most one per query")
	if attack == "A6" and not 2 <= k <= budget:
		raise ValueError("A6 needs at least two accounts and at most one per query")
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
	prompts += [prompt for domain in order for prompt in cover.get(domain, [])]
	if attack != "A2":
		random.Random(f"{seed}:interleave").shuffle(prompts)
	cover_ids = {p.example_id for v in cover.values() for p in v}
	sybil = [f"{campaign_id}-sybil-{i:03d}" for i in range(k)] if attack == "A6" else []
	# A6 assigns accounts by a seeded permutation of positions, so every account multiplexes.
	positions = list(range(len(prompts)))
	random.Random(f"{seed}:sybil").shuffle(positions)
	owner = {position: sybil[rank % k] for rank, position in enumerate(positions)} if sybil else {}
	used: Counter[str] = Counter()
	requests = []
	for index, prompt in enumerate(prompts):
		domain = prompt.task
		choices = account_map[domain]
		requests.append(
			GatewayRequest(
				request_id=f"{campaign_id}-{index:06d}",
				account=owner[index] if sybil else choices[used[domain] % len(choices)],
				time=index * interval_seconds,
				prompt=prompt,
				max_output_tokens=caps[domain],
				client_type=(
					("cover" if prompt.example_id in cover_ids else "extraction")
					if attack == "A3"
					else None
				),
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
		"prompt_sources": "independent external" if attack in EXTERNAL_ATTACKS else "task pools",
		"source_corpora": {
			domain: sorted({p.source.corpus for p in selected[domain]}) for domain in domains
		},
		"cover_share": cover_share if attack == "A3" else None,
		"cover_allocation": dict(cover_allocation),
		"accounts": sorted(account_domains),
		"account_domains": {
			account: sorted(domains) for account, domains in sorted(account_domains.items())
		},
		"interval_seconds": interval_seconds,
		"arrival_times": "synthetic, global campaign schedule",
		"max_output_tokens": dict(caps),
		"pool_sha256": pool_hashes(pools),
		"cover_pool_sha256": pool_hashes(cover_pools) if cover_pools is not None else None,
		"execution_status": "not_run",
	}
	return requests, manifest


def _balanced(budget: int, domains: list[str]) -> dict[str, int]:
	base, extra = divmod(budget, len(domains))
	return {domain: base + (index < extra) for index, domain in enumerate(domains)}


def _select(
	pools: Mapping[str, Sequence[PromptRecord]],
	allocation: Mapping[str, int],
	seed: int,
	label: str,
) -> dict[str, list[PromptRecord]]:
	selected = {}
	for domain, count in allocation.items():
		pool = sorted(pools[domain], key=lambda prompt: prompt.example_id)
		if count > len(pool):
			raise ValueError(f"domain allocation exceeds {domain} pool size {len(pool)}")
		random.Random(f"{seed}:{label}:{domain}").shuffle(pool)
		selected[domain] = pool[:count]
	return selected
