"""Opt-in, source-disjoint synthetic sessions for fresh shadow calibration only."""

import hashlib
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from distillation.benchmarks.models import BenchmarkItem
from distillation.schedules import pool_hashes, validate_pools
from distillation.schemas import GatewayRequest, Message, PromptRecord


@dataclass(frozen=True)
class SessionModel:
	"""Declared simulation parameters, not an estimate of natural user behavior."""

	follow_up_share: float = 0.12
	rephrase_share: float = 0.1
	retry_share: float = 0.08
	topic_drift_share: float = 0.2
	gap_log_mean: float = 3.4
	gap_log_sigma: float = 1.0
	session_break_share: float = 0.08
	session_break_seconds: float = 1800.0

	def validate(self) -> None:
		shares = (
			self.follow_up_share,
			self.rephrase_share,
			self.retry_share,
			self.topic_drift_share,
			self.session_break_share,
		)
		if any(not math.isfinite(v) or not 0 <= v <= 1 for v in shares):
			raise ValueError("session shares must be finite and between zero and one")
		if sum(shares[:3]) >= 1:
			raise ValueError("session model must retain fresh prompts")
		if (
			not all(
				math.isfinite(v)
				for v in (self.gap_log_mean, self.gap_log_sigma, self.session_break_seconds)
			)
			or min(self.gap_log_sigma, self.session_break_seconds) < 0
		):
			raise ValueError(
				"session gap parameters must be finite with nonnegative scale and pause"
			)


def session_clients(
	*,
	pools: Mapping[str, Sequence[PromptRecord]],
	caps: Mapping[str, int],
	role: str,
	references: Mapping[str, BenchmarkItem] | None = None,
	specialist_windows: int = 20,
	generalist_horizons: int = 150,
	window_size: int = 100,
	horizon_size: int = 100,
	seed: int = 42,
	model: SessionModel | None = None,
) -> tuple[list[GatewayRequest], dict[str, Any]]:
	"""Generate independent account windows with explicit synthetic session limitations.

	Each specialist account supplies exactly one nonoverlapping window. Generalists supply
	one fixed account horizon. Source groups belong to only one account, including across
	domains; callers must additionally provide role-disjoint pools. Insufficient capacity
	raises rather than silently shortening accounts or cloning source material.
	"""
	from distillation.tasks import LEGITIMATE_ROLES

	validate_pools(pools, caps)
	model = model or SessionModel()
	model.validate()
	if role not in LEGITIMATE_ROLES:
		raise ValueError("session clients require a legitimate pool role")
	if any(type(v) is not int or v < 1 for v in (specialist_windows, window_size, horizon_size)):
		raise ValueError(
			"specialist_windows, window_size and horizon_size must be positive integers"
		)
	if type(generalist_horizons) is not int or generalist_horizons < 0:
		raise ValueError("generalist_horizons must be a nonnegative integer")
	if generalist_horizons and (len(pools) < 2 or horizon_size < 2):
		raise ValueError("generalist horizons require at least two domains and requests")
	if not {"coding", "gsm8k"} <= pools.keys():
		raise ValueError("session calibration requires coding and gsm8k pools")
	if references is not None and any(
		p.example_id not in references for pool in pools.values() for p in pool
	):
		raise ValueError("missing session topic reference")
	queues = {
		d: sorted(pool, key=lambda p: hashlib.sha256(f"{seed}:{p.example_id}".encode()).digest())
		for d, pool in sorted(pools.items())
	}
	owners: dict[tuple[str, str], str] = {}

	def group(p: PromptRecord) -> tuple[str, str]:
		if not p.source.group_id:
			raise ValueError(f"missing source group for {p.example_id}")
		return p.source.corpus, p.source.group_id

	# Validate all provenance up front, even for unconsumed rows.
	for pool in queues.values():
		for prompt in pool:
			group(prompt)

	def topic(p: PromptRecord) -> str:
		return references[p.example_id].category if references else p.source.group_id or ""

	def draw(domain: str, account: str, previous: PromptRecord | None, drift: bool) -> PromptRecord:
		queue = queues[domain]
		index = next(
			(i for i, p in enumerate(queue) if owners.get(group(p), account) == account), None
		)
		if index is None:
			raise ValueError(
				f"session clients exhaust source-disjoint {domain} capacity at {account}; "
				"provide a larger fresh role pool or lower explicitly requested counts"
			)
		if references is not None and previous is not None and not drift:
			index = next(
				(
					i
					for i, p in enumerate(queue)
					if owners.get(group(p), account) == account and topic(p) == topic(previous)
				),
				index,
			)
		prompt = queue.pop(index)
		owners[group(prompt)] = account
		return prompt

	plans = [
		(f"{role}-sessions-{domain}-{i:04d}", [domain], window_size, "specialist")
		for domain in ("coding", "gsm8k")
		for i in range(specialist_windows)
	] + [
		(f"{role}-sessions-generalist-{i:04d}", sorted(pools), horizon_size, "generalist")
		for i in range(generalist_horizons)
	]
	rows = []
	events: Counter[str] = Counter()
	account_sources: dict[str, list[str]] = {}
	for account, domains, size, profile in plans:
		rng = random.Random(f"{seed}:{account}")
		at = rng.uniform(0, 86400)
		previous: PromptRecord | None = None
		original: PromptRecord | None = None
		for index in range(size):
			choice = rng.random()
			# Seed at least two distinct domains before synthesizing generalist revisits.
			fresh = (
				previous is None
				or index < len(domains)
				or choice >= (model.follow_up_share + model.rephrase_share + model.retry_share)
			)
			if fresh:
				domain = domains[index % len(domains)]
				original = draw(domain, account, original, rng.random() < model.topic_drift_share)
				prompt = original
				kind = "fresh"
			else:
				assert original is not None and previous is not None
				if choice < model.retry_share:
					prompt, kind = previous, "retry"
				else:
					kind = (
						"follow_up"
						if choice < model.retry_share + model.follow_up_share
						else "rephrase"
					)
					text = next(m.content for m in reversed(original.prompt) if m.role == "user")
					text = (
						text + "\nPlease explain the intermediate steps and check the result."
						if kind == "follow_up"
						else "Could you help me solve the following problem?\n" + text
					)
					messages = list(original.prompt)
					last_user = max(i for i, m in enumerate(messages) if m.role == "user")
					messages[last_user] = Message(role="user", content=text)
					prompt = original.model_copy(
						update={
							"example_id": f"{original.example_id}:session:{kind}:{index}",
							"prompt": messages,
						}
					)
			events[kind] += 1
			at += rng.lognormvariate(model.gap_log_mean, model.gap_log_sigma)
			if rng.random() < model.session_break_share:
				at += model.session_break_seconds
			rows.append(
				GatewayRequest(
					request_id=f"{account}-{index:05d}",
					account=account,
					time=at,
					prompt=prompt,
					max_output_tokens=caps[prompt.task],
					client_type=profile,
					domain_profile="generalist" if profile == "generalist" else "specialist",
				)
			)
			previous = prompt
		account_sources[account] = sorted(
			f"{c}:{g}" for (c, g), owner in owners.items() if owner == account
		)
	rows.sort(key=lambda r: (r.time, r.request_id))
	return rows, {
		"version": "gateway-session-clients-v1",
		"role": role,
		"seed": seed,
		"session_model": {
			"parameters": asdict(model),
			"events": dict(events),
			"follow_up": "append explicit explanation/check request to original user prompt",
			"rephrase": "prepend fixed polite problem-solving wrapper; not semantic paraphrasing",
			"retry": "repeat previous prompt verbatim",
			"topic_drift": "switch from preferred source category to seeded source order",
			"arrivals": "lognormal gaps with Bernoulli session pauses; random daily start",
			"limitation": "synthetic sessions bound behavior claims; not real-user validation",
		},
		"specialist_windows_per_domain": {d: specialist_windows for d in ("coding", "gsm8k")},
		"window_size": window_size,
		"generalist_horizons": generalist_horizons,
		"horizon_size": horizon_size,
		"accounts": len(plans),
		"requests": len(rows),
		"independence": (
			"source groups disjoint across accounts; "
			"role independence requires disjoint input pools"
		),
		"account_source_groups": account_sources,
		"account_profiles": {a: profile for a, _, _, profile in plans},
		"account_lengths": {a: size for a, _, size, _ in plans},
		"pool_sha256": pool_hashes(pools),
		"execution_status": "not_run",
	}
