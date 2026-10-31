"""Client-profile benign workloads with topic-proportional sampling (gateway-clients-v5).

The v3 generator (`workloads.mixed_clients`) sorts each domain pool by topic and consumes it
through one cursor, so a role whose pool exceeds its demand uses only its first topics. Here
every draw comes from the topic furthest below its pool share, so the prompts a role uses keep
its pool's topic mix whatever the demand, and accounts follow declared client profiles. v3
stays the default for frozen reproductions.
"""

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from harness.calibration import binomial_upper

from distillation.benchmarks.models import BenchmarkItem
from distillation.schedules import pool_hashes, validate_pools
from distillation.schemas import GatewayRequest, PromptRecord
from distillation.workloads import (
	DEFAULT_LENGTH_TAIL,
	DEFAULT_MAX_REQUESTS_PER_ACCOUNT,
	account_length,
)

PROFILE_WORKLOAD_VERSION = "gateway-clients-v5"
HOUR = 3600.0
# Casual accounts stay below one detector window, as v3 short clients do.
SHORT_PROFILE = "short"
SHORT_LENGTH = 20
DEFAULT_SHORT_SHARE = 0.2
# Coverage acceptance from the v4 audit: each role's topic mix within 3 points of its pool.
TOPIC_TOLERANCE = 0.03
MIN_TOPIC_COVERAGE = 0.9


@dataclass(frozen=True)
class Profile:
	"""A declared legitimate use: domain scope, topic behavior, repetition, and timing.

	`broad` draws each prompt from the domain topic furthest below its share; `block` stays on
	one topic until it is exhausted; `document` asks 3 to 15 questions about one source
	document before moving on. Gaps are uniform in seconds; `batch` requests arrive at the
	gap, then pause. Accounts start at a seeded time of day in `start`.
	"""

	description: str
	scope: Literal["single", "multi"]
	topics: Literal["broad", "block", "document"]
	preferred: tuple[str, ...] = ()
	repeats: tuple[int, int] = (1, 1)
	gap: tuple[float, float] = (5.0, 60.0)
	batch: int = 0
	pause: float = 0.0
	start: tuple[float, float] = (8 * HOUR, 18 * HOUR)
	users: tuple[int, int] = (1, 1)


PROFILES: dict[str, Profile] = {
	"generalist": Profile("broad researcher across two or more domains", "multi", "broad"),
	"specialist": Profile("one topic of one domain at a time", "single", "block"),
	"batch_evaluation": Profile(
		"evaluation suite over a domain's topics in fast batches",
		"single",
		"broad",
		gap=(0.1, 0.1),
		batch=100,
		pause=300.0,
	),
	"repetitive_labeling": Profile(
		"labeling or translation-style stream sending each item three times",
		"single",
		"broad",
		preferred=("ag_news",),
		repeats=(3, 3),
		gap=(1.0, 3.0),
	),
	"iterative_coding": Profile(
		"coding session resubmitting each problem one to four times",
		"single",
		"broad",
		preferred=("coding",),
		repeats=(1, 4),
		gap=(10.0, 120.0),
	),
	"long_document_qa": Profile(
		"several questions about each long source document",
		"single",
		"document",
		preferred=("squad",),
		gap=(10.0, 90.0),
	),
	"shared_enterprise": Profile(
		"three to twelve users on one key, each on one domain topic",
		"multi",
		"block",
		gap=(10.0, 120.0),
		users=(3, 12),
	),
	"nighttime_automation": Profile(
		"scheduled overnight job at steady spacing",
		"single",
		"broad",
		gap=(2.0, 2.0),
		start=(1 * HOUR, 4 * HOUR),
	),
	SHORT_PROFILE: Profile("casual account below one detector window", "single", "broad"),
}
SCORED_PROFILES = tuple(name for name in PROFILES if name != SHORT_PROFILE)


def profile_shares(
	profiles: Sequence[str] = SCORED_PROFILES, short: float = DEFAULT_SHORT_SHARE
) -> dict[str, float]:
	"""Equal shares of the selected scored profiles after `short` casual accounts.

	Equal shares give every profile the same number of accounts, so each can support its own
	restriction bound instead of inheriting one from a dominant profile.
	"""
	if not profiles or len(set(profiles)) != len(profiles) or set(profiles) - set(SCORED_PROFILES):
		raise ValueError(f"profiles must be distinct names from {', '.join(SCORED_PROFILES)}")
	if not math.isfinite(short) or not 0 <= short < 1:
		raise ValueError("short share must be at least zero and below one")
	shares = dict.fromkeys(profiles, (1 - short) / len(profiles))
	return {**shares, SHORT_PROFILE: short} if short else shares


def assign_profiles(count: int, shares: Mapping[str, float]) -> list[str]:
	"""Smooth weighted round-robin, as `workloads.assign_types`, over the given profiles."""
	assigned: Counter[str] = Counter()
	kinds = []
	for index in range(count):
		kind = max(shares, key=lambda k: shares[k] * (index + 1) - assigned[k])
		assigned[kind] += 1
		kinds.append(kind)
	return kinds


def zero_event_bound(accounts: int) -> float | None:
	"""One-sided 95% upper bound on a restriction rate when no account is restricted.

	About 3/n: 30 independent accounts bound the rate near 10%, 150 near 2%, 300 near 1%.
	"""
	return binomial_upper(0, accounts) if accounts else None


class TopicSampler:
	"""One domain's prompts, drawn without replacement from the topic furthest below its share.

	Prompts are hash-ordered within a topic. Any number of draws keeps every topic within about
	one prompt of its pool share, except where callers pin a topic.
	"""

	def __init__(
		self,
		prompts: Sequence[PromptRecord],
		topic: Callable[[PromptRecord], str],
		seed: int,
		families: Mapping[str, str] | None = None,
	):
		self.queues: dict[str, list[PromptRecord]] = defaultdict(list)
		for prompt in sorted(
			prompts, key=lambda p: hashlib.sha256(f"{seed}:{p.example_id}".encode()).digest()
		):
			self.queues[topic(prompt)].append(prompt)
		self.order = sorted(self.queues)
		self.pool = {name: len(queue) for name, queue in self.queues.items()}
		self.total = sum(self.pool.values())
		self.taken: Counter[str] = Counter()
		self.drawn = 0
		self.families = families
		self.owners: dict[str, int] = {}

	def eligible(self, prompt: PromptRecord | None, owner: int | None) -> bool:
		if prompt is None:
			return False
		if self.families is None or owner is None:
			return True
		family = self.families[prompt.example_id]
		return self.owners.get(family, owner) == owner

	def open(self, name: str | None, owner: int | None = None) -> bool:
		return name is not None and any(self.eligible(p, owner) for p in self.queues[name])

	def topic(self, owner: int | None = None) -> str | None:
		"""Select the eligible topic furthest below its source share."""
		best, deficit = None, None
		for name in self.order:
			if self.open(name, owner):
				value = self.pool[name] * (self.drawn + 1) - self.taken[name] * self.total
				if deficit is None or value > deficit:
					best, deficit = name, value
		return best

	def draw(self, name: str | None = None, owner: int | None = None) -> PromptRecord | None:
		if not self.open(name, owner):
			name = self.topic(owner)
		if name is None:
			return None
		queue = self.queues[name]
		index = next(i for i, prompt in enumerate(queue) if self.eligible(prompt, owner))
		prompt = queue[index]
		assert prompt is not None
		queue.pop(index)
		if self.families is not None and owner is not None:
			self.owners[self.families[prompt.example_id]] = owner
		self.taken[name] += 1
		self.drawn += 1
		return prompt

	def used(self) -> Counter[str]:
		return Counter({name: count for name, count in self.taken.items() if count})


def topic_coverage(
	pool: Mapping[str, int],
	used: Mapping[str, int],
	*,
	tolerance: float = TOPIC_TOLERANCE,
	min_coverage: float = MIN_TOPIC_COVERAGE,
) -> dict[str, Any]:
	"""Compare the topic mix of the distinct prompts a role used with its pool's mix.

	Fails when any topic's used share differs from its pool share by more than `tolerance`, or
	when fewer than `min_coverage` of the topics expected to appear at least once (pool share
	times prompts used of at least one) do. The v4 policy-calibration AG News slice, 100%
	Business, fails both; its SQuAD slice, 56 of 215 titles, fails the second.
	"""
	pool_total, used_total = sum(pool.values()), sum(used.values())
	if set(used) - set(pool):
		raise ValueError("used topics must come from the pool")
	if not pool_total:
		raise ValueError("topic coverage needs a non-empty pool")
	deviation = (
		max(
			(
				abs(used.get(name, 0) / used_total - count / pool_total)
				for name, count in pool.items()
			),
			default=0.0,
		)
		if used_total
		else 0.0
	)
	expected = [name for name, count in pool.items() if count * used_total >= pool_total]
	covered = sum(used.get(name, 0) > 0 for name in expected)
	return {
		"pool_prompts": pool_total,
		"used_prompts": used_total,
		"pool_topics": len(pool),
		"used_topics": sum(value > 0 for value in used.values()),
		"expected_topics": len(expected),
		"covered_expected_topics": covered,
		"max_share_deviation": deviation,
		# Per-topic shares only where they are few enough to read.
		"topic_shares": {
			name: {"pool": count / pool_total, "used": used.get(name, 0) / used_total}
			for name, count in sorted(pool.items())
		}
		if len(pool) <= 20 and used_total
		else {},
		"tolerance": tolerance,
		"min_coverage": min_coverage,
		"passed": deviation <= tolerance + 1e-12 and covered >= min_coverage * len(expected),
	}


@dataclass
class _User:
	domains: list[str]
	topic: dict[str, str | None] = field(default_factory=dict)
	left: int = 0
	step: int = 0
	units: list[tuple[PromptRecord, int]] = field(default_factory=list)


@dataclass
class _Account:
	index: int
	profile: str
	minimum: int
	target: int
	rng: random.Random
	users: list[_User]
	requests: int = 0
	limited: bool = False


def profile_clients(
	*,
	pools: Mapping[str, Sequence[PromptRecord]],
	caps: Mapping[str, int],
	role: str,
	accounts: int,
	references: Mapping[str, BenchmarkItem],
	profiles: Sequence[str] = SCORED_PROFILES,
	families: Mapping[str, str] | None = None,
	short_share: float = DEFAULT_SHORT_SHARE,
	requests_per_account: int = 100,
	seed: int = 42,
	length_tail: float | None = DEFAULT_LENGTH_TAIL,
	max_requests_per_account: int = DEFAULT_MAX_REQUESTS_PER_ACCOUNT,
) -> tuple[list[GatewayRequest], dict[str, Any]]:
	"""Build benign accounts from declared client profiles over one role's pools.

	Topics are reference categories: AG News labels, SQuAD titles, coding domains, and one
	GSM8K topic. Prompts are consumed without replacement; repetition stays inside the account
	that repeats. Accounts draw in rounds, one prompt each, so long and concentrated accounts
	never take a pool's remaining prompts ahead of others. Every account first reaches its
	minimum length, then non-short accounts extend to a seeded Pareto length, as in v3; an
	account the pools cannot extend is disclosed as capacity limited. Times are synthetic
	seconds from midnight of the first day.
	"""

	validate_pools(pools, caps)
	if role not in {
		"representation_fit",
		"reference_fit",
		"score_calibration",
		"policy_development",
		"locked_test",
	}:
		raise ValueError("profile clients require a legitimate pool role")
	if type(accounts) is not int or accounts < 1:
		raise ValueError("accounts must be positive")
	if type(requests_per_account) is not int or requests_per_account < 1:
		raise ValueError("requests_per_account must be positive")
	if type(max_requests_per_account) is not int or max_requests_per_account < requests_per_account:
		raise ValueError("max_requests_per_account must be at least requests_per_account")
	if length_tail is not None and (not math.isfinite(length_tail) or length_tail <= 0):
		raise ValueError("length_tail must be positive or None")
	if any(prompt.example_id not in references for pool in pools.values() for prompt in pool):
		raise ValueError("missing topic reference for a benign prompt")
	shares = profile_shares(profiles, short_share)
	domains = sorted(pools)

	def topic(prompt: PromptRecord) -> str:
		return references[prompt.example_id].category

	samplers = {domain: TopicSampler(pools[domain], topic, seed, families) for domain in domains}
	capacity = {domain: len(pools[domain]) for domain in domains}
	# Single-domain profiles follow pool capacity among their preferred domains, by smooth
	# weighted round-robin as v3 specialists do, so a small domain is not drained first.
	placed: dict[str, Counter[str]] = defaultdict(Counter)

	def single_domain(name: str) -> str:
		allowed = [d for d in PROFILES[name].preferred if d in pools] or domains
		total = sum(capacity[d] for d in allowed)
		slot = sum(placed[name].values()) + 1
		choice = max(allowed, key=lambda d: capacity[d] * slot - placed[name][d] * total)
		placed[name][choice] += 1
		return choice

	plans: list[_Account] = []
	for index, name in enumerate(assign_profiles(accounts, shares)):
		profile = PROFILES[name]
		rng = random.Random(f"{seed}:gateway-profile:{index}")
		if name == SHORT_PROFILE:
			minimum = target = min(requests_per_account, SHORT_LENGTH)
		else:
			minimum = requests_per_account
			target = account_length(
				requests_per_account, length_tail, max_requests_per_account, seed, index
			)
		users: list[_User] = []
		if profile.users[1] > 1:
			order = rng.sample(domains, len(domains))
			count = rng.randint(*profile.users)
			users = [_User([order[user % len(order)]]) for user in range(count)]
		elif profile.scope == "multi":
			if len(domains) < 2:
				raise ValueError(f"{name} accounts need at least two domains")
			users = [_User(rng.sample(domains, rng.randint(2, len(domains))))]
		else:
			users = [_User([single_domain(name)])]
		plans.append(_Account(index, name, minimum, target, rng, users))

	def draw(account: _Account) -> bool:
		"""Add one prompt, with its repeats, to the account's next user; False if exhausted."""
		profile = PROFILES[account.profile]
		user = account.users[sum(len(u.units) for u in account.users) % len(account.users)]
		available = [d for d in user.domains if samplers[d].topic(account.index) is not None]
		if not available:
			return False
		domain = available[user.step % len(available)]
		sampler = samplers[domain]
		current = user.topic.get(domain)
		if profile.topics == "document" and user.left <= 0:
			current, user.left = None, account.rng.randint(3, 15)
		if profile.topics == "broad" or not sampler.open(current, account.index):
			current = sampler.topic(account.index)
		prompt = sampler.draw(current, account.index)
		assert prompt is not None
		user.topic[domain] = topic(prompt)
		user.left -= 1
		user.step += 1
		repeats = account.rng.randint(*profile.repeats)
		user.units.append((prompt, repeats))
		account.requests += repeats
		return True

	for phase in ("minimum", "target"):
		pending = list(plans)
		while pending:
			waiting = []
			for account in pending:
				goal = account.minimum if phase == "minimum" else account.target
				if account.requests >= goal:
					continue
				if not draw(account):
					if phase == "minimum":
						raise ValueError(
							f"profile clients exhaust the {account.profile} domains; "
							"reduce account load"
						)
					account.limited = True
					continue
				if account.requests < goal:
					waiting.append(account)
			pending = waiting

	requests = []
	account_users: dict[str, int] = {}
	for account in plans:
		profile = PROFILES[account.profile]
		name = f"{role}-v5-{account.index:05d}"
		start = account.rng.uniform(*profile.start)
		timed = []
		for number, user in enumerate(account.users):
			at = start + (account.rng.uniform(0, 4 * HOUR) if number else 0.0)
			sent = 0
			for prompt, repeats in user.units:
				for _ in range(repeats):
					if sent:
						at += (
							profile.pause
							if profile.batch and sent % profile.batch == 0
							else account.rng.uniform(*profile.gap)
						)
					timed.append((round(at, 3), number, sent, prompt))
					sent += 1
		timed.sort(key=lambda item: item[:3])
		timed = timed[: account.target]
		domains_used = {prompt.task for *_, prompt in timed}
		account_users[name] = len(account.users)
		for offset, (at, _, _, prompt) in enumerate(timed):
			requests.append(
				GatewayRequest(
					request_id=f"{name}-{offset:05d}",
					account=name,
					time=at,
					prompt=prompt,
					max_output_tokens=caps[prompt.task],
					client_type=account.profile,
					domain_profile="generalist" if len(domains_used) > 1 else "specialist",
				)
			)
	requests.sort(key=lambda request: (request.time, request.request_id))
	return requests, profile_manifest(
		requests,
		plans=plans,
		pools=pools,
		caps=caps,
		role=role,
		seed=seed,
		shares=shares,
		requests_per_account=requests_per_account,
		length_tail=length_tail,
		max_requests_per_account=max_requests_per_account,
		samplers=samplers,
		account_users=account_users,
	)


def profile_manifest(
	requests: list[GatewayRequest],
	*,
	plans: list[_Account],
	pools: Mapping[str, Sequence[PromptRecord]],
	caps: Mapping[str, int],
	role: str,
	seed: int,
	shares: Mapping[str, float],
	requests_per_account: int,
	length_tail: float | None,
	max_requests_per_account: int,
	samplers: Mapping[str, TopicSampler],
	account_users: Mapping[str, int],
) -> dict[str, Any]:
	by_account: dict[str, list[GatewayRequest]] = defaultdict(list)
	for request in requests:
		by_account[request.account].append(request)
	profile_of = {account: rows[0].client_type for account, rows in by_account.items()}
	scored = Counter(
		profile_of[account] for account, rows in by_account.items() if len(rows) >= 100
	)
	return {
		"version": PROFILE_WORKLOAD_VERSION,
		"role": role,
		"seed": seed,
		"accounts": len(plans),
		"profile_shares": dict(shares),
		"profiles": {
			name: {
				"description": PROFILES[name].description,
				"accounts": sum(plan.profile == name for plan in plans),
				"scored_accounts": scored[name],
				# Resolution only: what zero restricted accounts would bound, not a result.
				"zero_event_upper95": zero_event_bound(scored[name]),
			}
			for name in shares
		},
		"requests_per_account_requested": requests_per_account,
		"length_distribution": {
			"minimum": requests_per_account,
			"pareto_shape": length_tail,
			"maximum": max_requests_per_account,
			"short_accounts": f"capped at {SHORT_LENGTH}",
		},
		"account_lengths_requested": {f"{role}-v5-{plan.index:05d}": plan.target for plan in plans},
		"capacity_limited_accounts": sorted(
			f"{role}-v5-{plan.index:05d}" for plan in plans if plan.limited
		),
		"requests": len(requests),
		"domain_requests": dict(sorted(Counter(r.domain for r in requests).items())),
		"domain_unique_prompts": {domain: s.drawn for domain, s in sorted(samplers.items())},
		"account_profiles": dict(sorted(profile_of.items())),
		"account_users": dict(sorted(account_users.items())),
		"account_domains": {
			account: sorted({request.domain for request in rows})
			for account, rows in sorted(by_account.items())
		},
		"account_lengths": {account: len(rows) for account, rows in sorted(by_account.items())},
		"scored_accounts": sum(scored.values()),
		"complete_100_request_windows": sum(len(rows) // 100 for rows in by_account.values()),
		"multi_window_accounts": sum(len(rows) >= 200 for rows in by_account.values()),
		"accounts_spanning_a_day": sum(
			rows[-1].time - rows[0].time >= 86_400 for rows in by_account.values()
		),
		"source_documents": len(
			{(r.prompt.source.corpus, r.prompt.source.group_id) for r in requests}
		),
		"topic_order": "largest deficit against the pool's topic shares",
		"topic_coverage": {
			domain: topic_coverage(sampler.pool, sampler.used())
			for domain, sampler in sorted(samplers.items())
		},
		"arrival_times": "synthetic seconds from midnight; profiles declare start and spacing",
		"max_output_tokens": dict(caps),
		"pool_sha256": pool_hashes(pools),
		"execution_status": "not_run",
	}
