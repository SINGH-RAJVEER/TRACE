"""Opt-in horizon telemetry; no policy, action, response or budget dependencies.

Ranks require exchangeable account horizons within each declared profile. Synthetic
session nulls support claims about that generator only. Novelty measures geometric
exploration, not intent or an unavoidable property of every active learner.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from harness.corpus import CorpusIndex, normalized, shingles
from harness.logdet import CoverageSpec, length_bin
from harness.shadow import minimum_null, upper_p


@dataclass(frozen=True)
class SignalRow:
	text: str
	time: float | None
	embedding: tuple[float, ...]
	domain: str = "all"
	topic: str | None = None


class SignalCalibration(BaseModel):
	version: str = "horizon-signals-v1"
	target: float = Field(default=0.05, gt=0, lt=1)
	nulls: dict[str, dict[str, list[float]]] = Field(default_factory=dict)
	corpus_digest: str | None = None
	coverage: CoverageSpec | None = None


def _entropy(values: Sequence[str]) -> float:
	counts = Counter(values)
	return -sum((n / len(values)) * math.log(n / len(values)) for n in counts.values())


def raw_signals(
	rows: Sequence[SignalRow],
	corpus: CorpusIndex | None = None,
	coverage: CoverageSpec | None = None,
) -> dict[str, float]:
	"""Sequence-dependent session/novelty summaries; corpus share is order invariant.

	Follow-ups use lexical overlap with any prior prompt or explicit follow-up language;
	rephrasing uses normalized trigram Jaccard >= .5 (including exact retries). Templates
	replace numbers and quoted strings. Topic entropy is over directed domain transitions.
	Novelty is mean Euclidean distance to the closest earlier embedding, excluding the first.
	With a `coverage` spec, `coverage_mean_logdet` adds the mean per-request log-det gain
	(see harness.logdet), excluding the first. It is a coverage proxy, not mutual information
	or a learning prediction. It depends on request count, so its nulls are matched by
	request-count band (see harness.logdet.length_bin). The cumulative sum is not exposed.
	"""
	if not rows:
		return {}
	followups = 0
	rephrases = 0
	prior_sizes: list[int] = []
	postings: dict[str, list[int]] = {}
	for row in rows:
		current = shingles(row.text)
		overlaps: Counter[int] = Counter()
		for part in current:
			overlaps.update(postings.get(part, []))
		near = max(
			(
				overlap / max(1, len(current) + prior_sizes[i] - overlap)
				for i, overlap in overlaps.items()
			),
			default=0.0,
		)
		followups += bool(prior_sizes) and (
			near >= 0.25
			or bool(
				re.search(r"\b(also|instead|again|previous|that|follow.up)\b", row.text.lower())
			)
		)
		rephrases += bool(prior_sizes) and near >= 0.5
		for part in current:
			postings.setdefault(part, []).append(len(prior_sizes))
		prior_sizes.append(len(current))
	n = len(rows)
	templates = {normalized(re.sub(r"\b\d+\b|\"[^\"]*\"|'[^']*'", "value", r.text)) for r in rows}
	times = [r.time for r in rows if r.time is not None]
	gaps = np.diff(times) if len(times) == len(rows) else np.asarray([])
	if not all(math.isfinite(t) for t in times):
		raise ValueError("arrival times must be finite")
	if np.any(gaps < 0):
		raise ValueError("signal rows must be in nondecreasing arrival order")
	result = {
		"request_count": float(n),
		"revisit_followup_rate": followups / max(1, n - 1),
		"rephrasing_rate": rephrases / max(1, n - 1),
		"template_diversity": len(templates) / n,
		"answer_only_share": sum(
			bool(
				re.search(
					r"\b(answer only|only (?:the )?answer|no explanation|just (?:the )?answer)\b",
					r.text.lower(),
				)
			)
			for r in rows
		)
		/ n,
	}
	if len(gaps) and float(np.mean(gaps)) > 0:
		result["interarrival_cv"] = float(np.std(gaps) / np.mean(gaps))
	if n > 1:
		transitions = [
			f"{a.topic or a.domain}->{b.topic or b.domain}"
			for a, b in zip(rows[:-1], rows[1:], strict=True)
		]
		result["topic_transition_entropy"] = _entropy(transitions)
		result["topic_switch_rate"] = sum(
			(a.topic or a.domain) != (b.topic or b.domain)
			for a, b in zip(rows[:-1], rows[1:], strict=True)
		) / (n - 1)
		if all(r.embedding for r in rows):
			embeddings = np.asarray([r.embedding for r in rows], dtype=np.float64)
			if not np.all(np.isfinite(embeddings)):
				raise ValueError("embeddings must be finite")
			result["novelty_mean_nn"] = float(
				np.mean(
					[
						float(np.linalg.norm(embeddings[:i] - embeddings[i], axis=1).min())
						for i in range(1, n)
					]
				)
			)
			if coverage is not None:
				result["coverage_mean_logdet"] = float(np.mean(coverage.gains(embeddings)[1:]))
	if corpus is not None:
		result["corpus_match_share"] = sum(bool(corpus.match(r.text)["matched"]) for r in rows) / n
	return result


def fit_signals(
	horizons: Sequence[Sequence[SignalRow]],
	*,
	profiles: Sequence[str],
	target: float = 0.05,
	corpus: CorpusIndex | None = None,
	coverage: CoverageSpec | None = None,
) -> SignalCalibration:
	"""Fit only from disjoint score_calibration account horizons, never locked/test data."""
	nulls: dict[str, dict[str, list[float]]] = {}
	for rows, profile in zip(horizons, profiles, strict=True):
		for metric, value in raw_signals(rows, corpus, coverage).items():
			if len(rows) < 2 and metric in {"revisit_followup_rate", "rephrasing_rate"}:
				continue
			if metric == "coverage_mean_logdet":
				# Coverage falls as a stream repeats its span, so its nulls are matched by
				# request-count band rather than pooled across horizon lengths.
				metric = f"{metric}[{length_bin(len(rows))}]"
			nulls.setdefault(profile, {}).setdefault(metric, []).append(value)
	return SignalCalibration(
		target=target,
		nulls={p: {k: sorted(v) for k, v in metrics.items()} for p, metrics in nulls.items()},
		corpus_digest=corpus.digest if corpus is not None else None,
		coverage=coverage,
	)


def score_signals(
	rows: Sequence[SignalRow],
	*,
	profile: str,
	calibration: SignalCalibration,
	corpus: CorpusIndex | None = None,
) -> dict[str, Any]:
	"""Report raw statistics and ranks. Session ranks are conservative two-sided ranks."""
	if (corpus.digest if corpus is not None else None) != calibration.corpus_digest:
		raise ValueError("corpus index digest differs from signal calibration")
	metrics: dict[str, Any] = {}
	values = raw_signals(rows, corpus, calibration.coverage)
	names: tuple[str, ...] = (
		"request_count",
		"revisit_followup_rate",
		"rephrasing_rate",
		"template_diversity",
		"answer_only_share",
		"interarrival_cv",
		"topic_transition_entropy",
		"topic_switch_rate",
		"novelty_mean_nn",
		"corpus_match_share",
	)
	if calibration.coverage is not None:
		names += ("coverage_mean_logdet",)
	for name in names:
		value = values.get(name)
		one_sided = name in {
			"novelty_mean_nn",
			"coverage_mean_logdet",
			"corpus_match_share",
			"request_count",
		}
		required = minimum_null(calibration.target if one_sided else calibration.target / 2)
		key = name
		if name == "coverage_mean_logdet" and len(rows) >= 2:
			key = f"{name}[{length_bin(len(rows))}]"
		null = calibration.nulls.get(profile, {}).get(key, [])
		entry: dict[str, Any] = {
			"value": value,
			"null_horizons": len(null),
			"minimum_null": required,
			"rank_floor": min(1.0, (1 if one_sided else 2) / (len(null) + 1)),
			"p_value": None,
		}
		if key != name:
			entry["length_band"] = key[len(name) + 1 : -1]
		if value is None or (
			len(rows) < 2 and name in {"revisit_followup_rate", "rephrasing_rate"}
		):
			entry["abstain"] = (
				"statistic unavailable: insufficient observations, gaps, embeddings or corpus"
			)
		elif len(null) < required:
			band = f" of {key[len(name) + 1 : -1]} requests" if key != name else ""
			entry["abstain"] = f"need {required} benign {profile} horizons{band}"
		else:
			upper = upper_p(null, value)
			lower = (1 + sum(v <= value for v in null)) / (len(null) + 1)
			entry.update(
				p_value=upper if one_sided else min(1.0, 2 * min(upper, lower)),
				upper_p=upper,
				lower_p=lower,
				abstain=None,
			)
		metrics[name] = entry
	return {
		"profile": profile,
		"requests": len(rows),
		"metrics": metrics,
		"corpus_digest": calibration.corpus_digest,
		"shadow_only": True,
	}
