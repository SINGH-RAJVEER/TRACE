"""Deterministic task clients with disclosed synthetic arrival times and source identities."""

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from harness.calibration import BenignRequest

from distillation.benchmarks.models import BenchmarkItem, save_json
from distillation.io import read_jsonl
from distillation.schedules import pool_hashes, validate_pools
from distillation.schemas import GatewayRequest, PromptRecord

WORKLOAD_VERSION = "task-clients-v2"
GATEWAY_WORKLOAD_VERSION = "gateway-clients-v2"
CLIENT_TYPES = ("specialist", "repetitive", "batch", "bursty", "short")
# Specialist and repetitive clients send concentrated windows that score above any broad window,
# so their share sets the benign null's upper tail. The v1 workloads used an equal five-way
# split (0.4); detectability is reported across a sweep of this share.
DEFAULT_CONCENTRATED_SHARE = 0.1
LEGACY_CONCENTRATED_SHARE = 0.4


def client_shares(concentrated: float) -> dict[str, float]:
	"""Shares of the five client types; short clients keep a fifth, the rest splits evenly."""
	if not math.isfinite(concentrated) or not 0 <= concentrated <= 0.8:
		raise ValueError("concentrated share must be between zero and 0.8")
	broad = (0.8 - concentrated) / 2
	return {
		"specialist": concentrated / 2,
		"repetitive": concentrated / 2,
		"batch": broad,
		"bursty": broad,
		"short": 0.2,
	}


def assign_types(count: int, shares: Mapping[str, float]) -> list[str]:
	"""Smooth weighted round-robin: each slot goes to the type furthest below its share.

	Ties follow CLIENT_TYPES order, so equal shares reproduce plain round-robin.
	"""
	assigned: Counter[str] = Counter()
	kinds = []
	for index in range(count):
		kind = max(CLIENT_TYPES, key=lambda k: shares[k] * (index + 1) - assigned[k])
		assigned[kind] += 1
		kinds.append(kind)
	return kinds


def task_references(directory: Path) -> dict[str, BenchmarkItem]:
	"""Recover scoring references for existing builds without rewriting pools or ledgers."""
	from distillation.tasks import CONVERTERS, LEGITIMATE_ROLES, TASKS, _benchmark_item, _load_rows

	path = directory / "legitimate-references.json"
	if path.exists():
		return {
			key: BenchmarkItem.model_validate(value)
			for key, value in json.loads(path.read_text()).items()
		}
	manifest = json.loads((directory / "manifest.json").read_text())
	task = manifest["task"]
	spec = TASKS[task]
	if manifest["dataset"]["revision"] != spec.dataset.revision:
		raise ValueError("task references require the build's pinned dataset revision")
	prompts = {
		p.example_id: p
		for role in LEGITIMATE_ROLES
		for p in read_jsonl(directory / "legitimate" / f"{role}.jsonl", PromptRecord)
	}
	items = {}
	for index, row in enumerate(_load_rows(spec.dataset, spec.pool_split)):
		example = CONVERTERS[task](index, row)
		key = f"{task}-{spec.pool_split}-{example.key}"
		if key in prompts:
			item = _benchmark_item(task, spec.pool_split, example)
			if (
				item.messages != prompts[key].prompt
				or example.group != prompts[key].source.group_id
			):
				raise ValueError(f"reference does not match frozen prompt {key}")
			items[key] = item
	if items.keys() != prompts.keys():
		raise ValueError("missing legitimate scoring references")
	save_json(path, {key: item.model_dump(mode="json") for key, item in items.items()})
	return items


def clients(
	prompts: list[PromptRecord],
	role: str,
	seed: int = 42,
	references: dict[str, BenchmarkItem] | None = None,
	concentrated: float = DEFAULT_CONCENTRATED_SHARE,
) -> list[BenignRequest]:
	"""Partition source groups between five client types, then form accounts within each type.

	`concentrated` is the share of source groups given to specialist and repetitive clients.
	Repetition reuses a prompt only within its original account. Document sharing between
	accounts is recorded and must be clustered in uncertainty estimates. No natural client
	identity or real-world arrival distribution is inferred from these synthetic accounts.
	"""
	groups: dict[str, list[PromptRecord]] = defaultdict(list)
	for prompt in prompts:
		if not prompt.source.group_id:
			raise ValueError(f"missing source group for {prompt.example_id}")
		groups[prompt.source.group_id].append(prompt)
	ordered = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest())
	buckets: dict[str, list[PromptRecord]] = {kind: [] for kind in CLIENT_TYPES}
	kinds = assign_types(len(ordered), client_shares(concentrated))
	for group, kind in zip(ordered, kinds, strict=True):
		buckets[kind].extend(groups[group])
	requests = []
	for kind, pool in buckets.items():
		pool.sort(key=lambda p: hashlib.sha256(f"{seed}:{p.example_id}".encode()).digest())
		if kind == "specialist":
			pool.sort(
				key=lambda p: (
					references[p.example_id].category if references else p.source.group_id or ""
				)
			)
		size = {"specialist": 200, "repetitive": 100, "batch": 500, "bursty": 300, "short": 20}[
			kind
		]
		for offset in range(0, len(pool), size):
			chunk = pool[offset : offset + size]
			if kind == "repetitive":
				chunk = [p for p in chunk for _ in range(3)]
			account = f"{role}-{kind}-{offset // size:05d}"
			for index, prompt in enumerate(chunk):
				at = index * 0.1 if kind == "batch" else index * 2.0
				if kind == "bursty":
					at = (index // 25) * 60 + (index % 25) * 0.1
				requests.append(
					BenignRequest(
						account=account,
						text=next(m.content for m in reversed(prompt.prompt) if m.role == "user"),
						time=at,
						example_id=prompt.example_id,
						source_group=prompt.source.group_id,
						client_type=kind,
					)
				)
	return requests


def workload_summary(
	rows: list[BenignRequest], concentrated: float = DEFAULT_CONCENTRATED_SHARE
) -> dict[str, Any]:
	lengths = Counter(row.account for row in rows)
	return {
		"version": WORKLOAD_VERSION,
		"concentrated_share": concentrated,
		"client_shares": client_shares(concentrated),
		"requests": len(rows),
		"accounts": len(lengths),
		"scored_accounts": sum(n >= 100 for n in lengths.values()),
		"short_accounts": sum(n < 100 for n in lengths.values()),
		"source_documents": len({r.source_group for r in rows}),
		"unique_prompts": len({r.example_id for r in rows}),
		"account_lengths": dict(lengths),
		"arrival_times": "synthetic",
		"client_types": dict(Counter(r.client_type for r in rows)),
	}


def mixed_clients(
	*,
	pools: Mapping[str, Sequence[PromptRecord]],
	caps: Mapping[str, int],
	role: str,
	accounts: int,
	generalist_share: float = 0.1,
	requests_per_account: int = 100,
	seed: int = 42,
	references: Mapping[str, BenchmarkItem] | None = None,
	concentrated: float = DEFAULT_CONCENTRATED_SHARE,
) -> tuple[list[GatewayRequest], dict[str, Any]]:
	"""Build specialist and generalist accounts independently of timing/repetition style.

	`concentrated` sets the share of accounts with specialist and repetitive timing styles;
	`generalist_share` sets domain specialization separately.

	Generalist share is an account fraction, rounded to the nearest account with ties up.
	Generalists use two or more domains. Prompts are consumed without replacement, except
	for disclosed repetition within repetitive accounts. Source sharing within a role is
	retained for clustered uncertainty; callers must use disjoint pools for different roles.
	"""
	from distillation.tasks import LEGITIMATE_ROLES

	validate_pools(pools, caps)
	if role not in LEGITIMATE_ROLES:
		raise ValueError("mixed benign clients require a legitimate pool role")
	if type(accounts) is not int or accounts < 1:
		raise ValueError("accounts must be positive")
	if type(requests_per_account) is not int or requests_per_account < 1:
		raise ValueError("requests_per_account must be positive")
	if not math.isfinite(generalist_share) or not 0 <= generalist_share <= 1:
		raise ValueError("generalist_share must be between zero and one")
	if references is not None and any(
		prompt.example_id not in references for pool in pools.values() for prompt in pool
	):
		raise ValueError("missing topic reference for a benign prompt")
	domains = sorted(pools)
	ordered = {
		domain: sorted(
			pool,
			key=lambda prompt: (
				references[prompt.example_id].category
				if references is not None
				else (prompt.source.corpus, prompt.source.group_id or ""),
				hashlib.sha256(f"{seed}:{prompt.example_id}".encode()).digest(),
			),
		)
		for domain, pool in pools.items()
	}
	rng = random.Random(f"{seed}:gateway-clients")
	generalist_count = math.floor(accounts * generalist_share + 0.5)
	generalists = set(rng.sample(range(accounts), generalist_count))
	rng.shuffle(domains)
	cursors: Counter[str] = Counter()
	requests = []
	for index, kind in enumerate(assign_types(accounts, client_shares(concentrated))):
		length = min(requests_per_account, 20) if kind == "short" else requests_per_account
		unique = math.ceil(length / 3) if kind == "repetitive" else length
		is_generalist = index in generalists
		if is_generalist and unique < 2:
			raise ValueError("generalist accounts need at least two distinct prompts")
		chosen = (
			rng.sample(domains, rng.randint(2, min(len(domains), unique)))
			if is_generalist
			else [domains[index % len(domains)]]
		)
		chunk = []
		for offset in range(unique):
			domain = chosen[offset % len(chosen)]
			if cursors[domain] >= len(ordered[domain]):
				raise ValueError(f"mixed benign clients exhaust {domain} pool; reduce account load")
			chunk.append(ordered[domain][cursors[domain]])
			cursors[domain] += 1
		if kind == "repetitive":
			chunk = [prompt for prompt in chunk for _ in range(3)][:length]
		for offset, prompt in enumerate(chunk):
			at = offset * 0.1 if kind == "batch" else offset * 2.0
			if kind == "bursty":
				at = (offset // 25) * 60 + (offset % 25) * 0.1
			requests.append(
				GatewayRequest(
					request_id=f"{role}-gateway-{index:05d}-{offset:05d}",
					account=f"{role}-gateway-{index:05d}",
					time=at,
					prompt=prompt,
					max_output_tokens=caps[prompt.task],
					client_type=kind,
					domain_profile="generalist" if is_generalist else "specialist",
				)
			)
	requests.sort(key=lambda request: (request.time, request.request_id))
	by_account: dict[str, list[GatewayRequest]] = defaultdict(list)
	for request in requests:
		by_account[request.account].append(request)
	manifest = {
		"version": GATEWAY_WORKLOAD_VERSION,
		"role": role,
		"seed": seed,
		"accounts": accounts,
		"concentrated_share": concentrated,
		"client_shares": client_shares(concentrated),
		"generalist_share_requested": generalist_share,
		"generalist_share_achieved": generalist_count / accounts,
		"generalist_accounts": generalist_count,
		"requests_per_account_requested": requests_per_account,
		"requests": len(requests),
		"domain_requests": dict(sorted(Counter(r.domain for r in requests).items())),
		"domain_unique_prompts": dict(cursors),
		"account_domains": {
			account: sorted({request.domain for request in rows})
			for account, rows in sorted(by_account.items())
		},
		"account_lengths": {account: len(rows) for account, rows in sorted(by_account.items())},
		"client_types": dict(Counter(rows[0].client_type for rows in by_account.values())),
		"scored_accounts": sum(len(rows) >= 100 for rows in by_account.values()),
		"source_documents": len(
			{(r.prompt.source.corpus, r.prompt.source.group_id) for r in requests}
		),
		"topic_order": "category" if references is not None else "source group",
		"domain_topic_categories": {
			domain: sorted({references[p.example_id].category for p in pool})
			for domain, pool in sorted(pools.items())
		}
		if references is not None
		else {},
		"specialist_scope": {
			domain: "topic ordered"
			if references is not None and len({references[p.example_id].category for p in pool}) > 1
			else "domain only"
			for domain, pool in sorted(pools.items())
		},
		"arrival_times": "synthetic, accounts start together; short accounts capped at 20",
		"max_output_tokens": dict(caps),
		"pool_sha256": pool_hashes(pools),
		"execution_status": "not_run",
	}
	return requests, manifest
