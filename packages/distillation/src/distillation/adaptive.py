"""CPU coverage selection using public prompts and attacker-observable feedback only."""

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from distillation.schemas import GatewayRequest, LedgerRecord, Message, PromptRecord

UncertaintyHook = Callable[[Sequence[GatewayRequest], Sequence[LedgerRecord]], Mapping[str, float]]


def embedding(row: GatewayRequest) -> dict[str, float]:
	"""Sparse normalized token embedding; no model download or accelerator."""
	words = re.findall(r"\w+", " ".join(m.content for m in row.prompt.prompt).lower())
	counts = Counter(words)
	norm = math.sqrt(sum(v * v for v in counts.values())) or 1
	return {k: v / norm for k, v in counts.items()}


def similarity(left: dict[str, float], right: dict[str, float]) -> float:
	return sum(v * right.get(k, 0.0) for k, v in left.items())


def topic_order(rows: list[GatewayRequest]) -> list[GatewayRequest]:
	"""Domain blocks with eight deterministic farthest-first lexical topic clusters."""
	result = []
	for domain in sorted({r.domain for r in rows}):
		group = sorted((r for r in rows if r.domain == domain), key=lambda r: r.request_id)
		vectors = [embedding(r) for r in group]
		centers = [vectors[0]]
		for _ in range(min(8, len(group)) - 1):
			centers.append(min(vectors, key=lambda v: max(similarity(v, c) for c in centers)))
		result.extend(
			sorted(
				group,
				key=lambda r: (
					max(range(len(centers)), key=lambda i: similarity(embedding(r), centers[i])),
					r.request_id,
				),
			)
		)
	return result


def variant_schedule(
	*,
	pools: Mapping[str, Sequence[PromptRecord]],
	caps: Mapping[str, int],
	attack: str,
	budget: int,
	seed: int,
	campaign_id: str,
	accounts: int | None,
	allocation: Mapping[str, int] | None,
	interval_seconds: float,
	deadline_seconds: float,
	cover_pools: Mapping[str, Sequence[PromptRecord]] | None,
	cover_share: float,
	cover_allocation: Mapping[str, int] | None,
	topic_sorted: bool,
	deduplicate: bool,
	answer_only: bool,
	per_account_limit: int | None,
	round_size: int,
) -> tuple[list[GatewayRequest], dict[str, Any]]:
	from typing import cast

	from distillation.schedules import Attack, campaign_schedule

	if type(round_size) is not int or round_size < 1:
		raise ValueError("round_size must be positive")
	if per_account_limit is not None and (
		type(per_account_limit) is not int or per_account_limit < 1
	):
		raise ValueError("per_account_limit must be positive")
	if attack == "A8" and topic_sorted:
		raise ValueError("A8 selects order online; topic_sorted is a fixed-schedule variant")
	if attack == "A8" and (allocation is not None or cover_pools is not None):
		raise ValueError("A8 selects domain allocation online and does not use cover pools")
	if type(budget) is not int or budget < len(pools):
		raise ValueError("campaign needs at least one attempt per domain")
	rows, manifest = campaign_schedule(
		pools=pools,
		caps=caps,
		attack=cast(Attack, "A1" if attack == "A8" else attack),
		budget=sum(map(len, pools.values())) if attack == "A8" else budget,
		seed=seed,
		campaign_id=campaign_id,
		accounts=None if per_account_limit else accounts,
		allocation={d: len(p) for d, p in pools.items()} if attack == "A8" else allocation,
		interval_seconds=interval_seconds,
		deadline_seconds=deadline_seconds,
		cover_pools=cover_pools,
		cover_share=cover_share,
		cover_allocation=cover_allocation,
	)
	if accounts is not None and per_account_limit is not None:
		raise ValueError("rotation derives its account count from budget and per-account limit")
	if answer_only:
		from distillation.tasks import GSM8K_EVAL_SUFFIX

		for row in rows:
			row.prompt = row.prompt.model_copy(
				update={
					"prompt": [
						Message(
							role=m.role,
							content=m.content.removesuffix(GSM8K_EVAL_SUFFIX)
							+ "\nReturn only the final numeric answer, prefixed by ####.",
						)
						if row.domain == "gsm8k"
						and m.role == "user"
						and m.content.endswith(GSM8K_EVAL_SUFFIX)
						else m
						for m in row.prompt.prompt
					]
				}
			)
	if deduplicate:
		seen = set()
		unique = []
		for row in rows:
			key = tuple((m.role, " ".join(m.content.split())) for m in row.prompt.prompt)
			if key not in seen:
				unique.append(row)
				seen.add(key)
		rows = unique
	if topic_sorted:
		rows = topic_order(rows)
	for i, row in enumerate(rows):
		row.time = i * interval_seconds
		if per_account_limit:
			row.domain_profile = "generalist"
			row.account = f"{campaign_id}-rotation-{min(i, budget - 1) // per_account_limit:03d}"
	variants = {
		"online": attack == "A8",
		"selector": "token-cosine-coverage-v1",
		"round_size": round_size,
		"topic_sorted": topic_sorted,
		"topic_method": "domain-blocked lexical farthest-first k=8" if topic_sorted else None,
		"deduplicate": deduplicate,
		"answer_only": answer_only,
		"answer_only_templates": ["gsm8k frozen suffix; ag_news/squad already answer-only"],
		"per_account_limit": per_account_limit,
		"rotation": "before next query when HTTP attempts reach limit; proofs also count",
	}
	manifest.update(
		attack=attack,
		attempt_budget=budget,
		planned_queries=None if attack == "A8" else len(rows),
		candidate_count=len(rows),
		variants=variants,
		accounts=sorted({r.account for r in rows}),
	)
	manifest["account_domains"] = {
		a: sorted({r.domain for r in rows if per_account_limit or r.account == a})
		for a in manifest["accounts"]
	}
	if attack == "A8":
		manifest["domain_allocation"] = None
		manifest["arrival_times"] = "online rounds, minimum global inter-query interval"
	return rows, manifest


class CoverageSelector:
	def __init__(self, rows: list[GatewayRequest], seed: int = 42) -> None:
		self.pending = {r.request_id: r for r in rows}
		self.vectors = {r.request_id: embedding(r) for r in rows}
		self.seed = seed
		self.example_keys = {r.prompt.example_id: r.request_id for r in rows}
		self.uncertainty: dict[str, float] = {}
		self.acquired: set[str] = set()
		self.coverage = dict.fromkeys(self.pending, 0.0)

	def select(
		self,
		count: int,
		released: list[LedgerRecord],
		feedback: list[dict[str, Any]],
		hook: UncertaintyHook | None = None,
	) -> tuple[list[GatewayRequest], dict[str, float]]:
		uncertainty = dict(hook(list(self.pending.values()), released)) if hook else {}
		self.uncertainty = uncertainty
		if any(not math.isfinite(v) for v in uncertainty.values()):
			raise ValueError("uncertainty scores must be finite")
		# Outcomes are joined to their public request domains by the caller.
		cost: dict[str, float] = defaultdict(float)
		for event in feedback:
			if event.get("event") == "outcome":
				cost[event["domain"]] += (event.get("status") != "received") + float(
					event.get("latency_seconds", 0)
				)
		# Only released examples count as acquired coverage. Rejected prompts stay novel.
		covered = {self.example_keys[r.example_id] for r in released}
		for acquired in sorted(covered - self.acquired):
			for candidate in self.pending:
				self.coverage[candidate] = max(
					self.coverage[candidate],
					similarity(self.vectors[candidate], self.vectors[acquired]),
				)
		self.acquired.update(covered)
		nearest = {k: self.coverage[k] for k in self.pending}
		batch, scores = [], {}
		for _ in range(min(count, len(self.pending))):

			def score(key: str) -> tuple[float, str]:
				novelty = 1 - nearest[key]
				value = (novelty + uncertainty.get(key, 0)) / (1 + cost[self.pending[key].domain])
				return value, hashlib.sha256(f"{self.seed}:{key}".encode()).hexdigest()

			key = max(self.pending, key=score)
			scores[key] = score(key)[0]
			batch.append(self.pending.pop(key))
			for candidate in self.pending:
				nearest[candidate] = max(
					nearest[candidate], similarity(self.vectors[candidate], self.vectors[key])
				)
		return batch, scores
