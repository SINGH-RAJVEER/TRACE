"""Keyed detection for the KGW watermark: an exact binomial test with one event per context.

Tokens sharing a context share a seed, so their memberships are dependent, and a repeated n-gram is
no fresh draw at all. Each unique context therefore contributes only its first scorable token, a
choice that does not depend on the tested key. Distinct contexts get separate HMAC seeds. If HMAC
behaves as a PRF and the text is independent of the key, the green count follows
Binomial(unique_contexts, green_probability). That assumption is what makes the test exact.

Corrections cover candidate keys only. Tests across windows or profiles need their own correction.
"""

import math
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Literal

from watermark_probe.core import (
	ALGORITHM_ID,
	ALGORITHM_VERSION,
	WatermarkKey,
	WatermarkProfile,
	context_of,
	context_seed,
	exact_integer,
	is_green,
	token_id,
)

Decision = Literal["detected", "not_detected", "insufficient_evidence"]
_Event = tuple[tuple[int, ...], int]


@dataclass(frozen=True)
class Detection:
	"""One candidate key's result. Holds no secret material.

	On abstention both p-values are None. `z_score` is None only when unique_contexts is 0.
	`tested_hypotheses` is the corrected family size, which is 0 if every candidate abstained.
	A p-value below the float range reads as 0.0; use `binomial_log_upper_tail` for its magnitude.
	"""

	candidate_key_id: str
	key_epoch: int
	algorithm_id: str
	algorithm_version: int
	scored_tokens: int
	unique_contexts: int
	green_count: int
	z_score: float | None
	raw_p_value: float | None
	adjusted_p_value: float | None
	tested_hypotheses: int
	decision: Decision


def _log_pmf_sum(first: int, last: int, n: int, gamma: float) -> float:
	"""Log of the binomial probability mass summed over first..last inclusive."""
	log_gamma = math.log(gamma)
	log_rest = math.log1p(-gamma)
	log_n = math.lgamma(n + 1)
	terms = [
		log_n - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_gamma + (n - i) * log_rest
		for i in range(first, last + 1)
	]
	peak = max(terms)
	return peak + math.log(math.fsum(math.exp(term - peak) for term in terms))


def binomial_log_upper_tail(k: int, n: int, gamma: float) -> float:
	"""Exact log P(X >= k) for X ~ Binomial(n, gamma)."""
	if n < 0:
		raise ValueError("n cannot be negative")
	if not 0.0 < gamma < 1.0:
		raise ValueError("gamma must lie strictly between 0 and 1")
	if k <= 0:
		return 0.0
	if k > n:
		return -math.inf
	# Below the mean the upper tail is near 1, so sum the small side to keep precision.
	if k > n * gamma:
		return min(0.0, _log_pmf_sum(k, n, n, gamma))
	return math.log1p(-math.exp(_log_pmf_sum(0, k - 1, n, gamma)))


def binomial_upper_tail(k: int, n: int, gamma: float) -> float:
	"""Exact P(X >= k). Underflows to 0.0 below the float range."""
	return math.exp(binomial_log_upper_tail(k, n, gamma))


def binomial_z_score(k: int, n: int, gamma: float) -> float | None:
	"""Standard score of k against Binomial(n, gamma), or None when n is 0."""
	if n == 0:
		return None
	return (k - n * gamma) / math.sqrt(n * gamma * (1.0 - gamma))


def holm_adjust(raw_p_values: Sequence[float | None]) -> list[float | None]:
	"""Holm-Bonferroni adjusted p-values. None marks an abstention, left out of the family."""
	tested = sorted((p, i) for i, p in enumerate(raw_p_values) if p is not None)
	adjusted: list[float | None] = [None] * len(raw_p_values)
	running = 0.0
	for rank, (p, i) in enumerate(tested):
		running = max(running, min(1.0, (len(tested) - rank) * p))
		adjusted[i] = running
	return adjusted


def _collect_events(
	generated_ids: Sequence[int],
	profile: WatermarkProfile,
	prompt_ids: Sequence[int],
	excluded_ids: AbstractSet[int],
) -> tuple[int, list[_Event]]:
	"""The scored token count before selection, and each context's first scorable token."""
	width = profile.context_width
	prompt = [token_id(token, profile) for token in prompt_ids]
	full = prompt + [token_id(token, profile) for token in generated_ids]
	scored = 0
	events: dict[tuple[int, ...], int] = {}
	for position in range(len(prompt), len(full)):
		token = full[position]
		context = context_of(full[max(0, position - width) : position], width)
		if context is None or token in excluded_ids:
			continue
		scored += 1
		events.setdefault(context, token)
	return scored, list(events.items())


def _count_green(events: Sequence[_Event], key: WatermarkKey, profile: WatermarkProfile) -> int:
	return sum(
		is_green(context_seed(key, profile, context), token, profile) for context, token in events
	)


def _decide(adjusted_p_value: float | None, alpha: float) -> Decision:
	if adjusted_p_value is None:
		return "insufficient_evidence"
	return "detected" if adjusted_p_value <= alpha else "not_detected"


def detect_candidates(
	generated_ids: Sequence[int],
	candidates: Sequence[WatermarkKey],
	profile: WatermarkProfile,
	*,
	alpha: float,
	min_effective: int,
	prompt_ids: Sequence[int] = (),
	excluded_ids: AbstractSet[int] = frozenset(),
) -> list[Detection]:
	"""Test each candidate key under Holm-Bonferroni familywise correction, in input order."""
	if not 0.0 < alpha < 1.0:
		raise ValueError("alpha must lie strictly between 0 and 1")
	if exact_integer(min_effective, "min_effective") < 1:
		raise ValueError(
			"min_effective must be at least 1, or no evidence could read as a negative"
		)
	if not candidates:
		raise ValueError("at least one candidate key is required")

	scored, events = _collect_events(generated_ids, profile, prompt_ids, excluded_ids)
	n = len(events)
	p = profile.green_probability
	greens = [_count_green(events, key, profile) for key in candidates]
	raw = [binomial_upper_tail(k, n, p) if n >= min_effective else None for k in greens]
	adjusted = holm_adjust(raw)
	tested = sum(p is not None for p in raw)
	return [
		Detection(
			candidate_key_id=key.key_id,
			key_epoch=key.epoch,
			algorithm_id=ALGORITHM_ID,
			algorithm_version=ALGORITHM_VERSION,
			scored_tokens=scored,
			unique_contexts=n,
			green_count=green,
			z_score=binomial_z_score(green, n, p),
			raw_p_value=raw_p,
			adjusted_p_value=adjusted_p,
			tested_hypotheses=tested,
			decision=_decide(adjusted_p, alpha),
		)
		for key, green, raw_p, adjusted_p in zip(candidates, greens, raw, adjusted, strict=True)
	]


def detect(
	generated_ids: Sequence[int],
	key: WatermarkKey,
	profile: WatermarkProfile,
	prompt_ids: Sequence[int] = (),
	excluded_ids: AbstractSet[int] = frozenset(),
	*,
	alpha: float,
	min_effective: int,
) -> Detection:
	"""Test one key: the one-hypothesis case of `detect_candidates`, so adjusted equals raw."""
	return detect_candidates(
		generated_ids,
		[key],
		profile,
		alpha=alpha,
		min_effective=min_effective,
		prompt_ids=prompt_ids,
		excluded_ids=excluded_ids,
	)[0]
