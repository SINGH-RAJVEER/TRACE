"""Shadow-mode scorers: telemetry beside the policy that never changes a level or an action.

Two scores address the audit's blind spots of the pooled upper-tail MMD
(readings/v4/analysis/REPORT.md, F1 and the order-only evasion probes):

- Domain-conditional MMD. Each gateway domain has its own references, drawn from fitting
  requests of that domain, and its own null of per-account windows from single-domain
  score-calibration accounts. An account's requests in one domain form that domain's windows.
  A domain whose null has too few non-overlapping windows to resolve the target abstains.
- Cumulative coverage. Cells come from k-means on fitting embeddings. A horizon's coverage is
  the set of cells its distinct prompts occupy, which no reordering of the same prompts can
  change. Its rank uses rarefied coverage, the expected number of cells a random subset of a
  fixed size of those prompts occupies, against score-calibration horizons of each profile.

Neither score identifies extraction. Coverage measures breadth, which benign generalists share,
and neither score can separate extraction that follows the benign distribution.
"""

import json
import math
import random
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from harness.mmd import Array, WindowScorer, median_bandwidth, squared_distances

SHADOW_VERSION = "shadow-v1"
ALL = "all"
SINGLE = "single_domain"
MULTI = "multi_domain"


class ShadowConfig(BaseModel):
	"""Shadow scorer settings, recorded in the artifact. Chosen from benign counts only."""

	model_config = ConfigDict(extra="forbid")

	# The smallest p-value a reported rank must be able to reach. A null of m non-overlapping
	# windows or horizons resolves 1 / (m + 1), so fewer than ceil(1 / target) - 1 abstain.
	target: float = Field(default=0.05, gt=0, lt=1)
	cells: int = Field(default=128, ge=2)
	# Distinct-prompt counts at which rarefied coverage is compared. A horizon is ranked at the
	# largest checkpoint it reaches whose null still resolves the target.
	checkpoints: tuple[int, ...] = (20, 50, 100, 200, 500, 1000, 2000)
	# Null window start spacing within an account's domain stream; None uses a quarter window.
	null_stride: int | None = Field(default=None, ge=1)
	kmeans_iterations: int = Field(default=30, ge=1)

	@field_validator("checkpoints")
	@classmethod
	def increasing(cls, values: tuple[int, ...]) -> tuple[int, ...]:
		if not values or any(v < 2 for v in values) or list(values) != sorted(set(values)):
			raise ValueError("checkpoints must be increasing integers of at least 2")
		return values


class ConditionalDomain(BaseModel):
	"""One domain's references and null. `abstain` explains why the domain is not ranked."""

	sigma: float | None = None
	reference_windows: int = 0
	null_scores: list[float] = Field(default_factory=list)
	null_accounts: int = 0
	independent_windows: int = 0
	abstain: str | None = None


class CoverageNull(BaseModel):
	"""Benign coverage of one profile: rarefied cells per checkpoint and raw cells per horizon."""

	horizons: int
	rarefied: dict[str, list[float]]
	cells: list[float]


class ShadowCalibration(BaseModel):
	version: str = SHADOW_VERSION
	settings: ShadowConfig
	minimum_null: int
	domains: dict[str, ConditionalDomain]
	coverage: dict[str, CoverageNull]
	counts: dict[str, int] = Field(default_factory=dict)
	_references: dict[str, Array] = PrivateAttr(default_factory=dict)
	_centroids: Array | None = PrivateAttr(default=None)

	def arrays(self) -> dict[str, Array]:
		"""The arrays saved beside the artifact: centroids and each domain's references."""
		if self._centroids is None:
			raise ValueError("shadow arrays are not loaded")
		return {"centroids": self._centroids} | {
			f"references_{domain}": value for domain, value in self._references.items()
		}

	def attach(self, arrays: dict[str, Array]) -> None:
		self._centroids = arrays["centroids"]
		self._references = {
			domain: arrays[f"references_{domain}"]
			for domain, fitted in self.domains.items()
			if fitted.reference_windows
		}


def minimum_null(target: float) -> int:
	return math.ceil(1 / target - 1e-9) - 1


def upper_p(null: Sequence[float], value: float) -> float:
	"""The upper-tail rank of value among null values, as Calibration.p_value computes it."""
	below = int(np.searchsorted(null, value, side="left"))
	return (1 + len(null) - below) / (len(null) + 1)


def request_domain(task: str, text: str) -> str:
	"""The gateway's declared domain; other tasks have one domain, the task itself."""
	if task == "gateway":
		return str(json.loads(text)["domain"])
	return task


def profile(task: str, domains: set[str]) -> str:
	"""Single- or multi-domain from the domains seen so far; one profile outside the gateway."""
	if task != "gateway":
		return ALL
	return MULTI if len(domains) > 1 else SINGLE


def kmeans(rows: Array, cells: int, iterations: int, seed: int) -> Array:
	"""Seeded k-means++ and Lloyd iterations, so no clustering dependency is needed."""
	if len(rows) < cells:
		raise ValueError(f"coverage needs at least {cells} fitting requests for its cells")
	rng = np.random.default_rng(seed)
	centers = [rows[int(rng.integers(len(rows)))]]
	nearest = squared_distances(rows, centers[0][None, :])[:, 0]
	for _ in range(1, cells):
		total = float(nearest.sum())
		pick = (
			int(rng.choice(len(rows), p=nearest / total))
			if total > 0
			else int(rng.integers(len(rows)))
		)
		centers.append(rows[pick])
		nearest = np.minimum(nearest, squared_distances(rows, rows[pick][None, :])[:, 0])
	result = np.stack(centers)
	for _ in range(iterations):
		labels = assign(rows, result)
		updated = result.copy()
		for cell in range(cells):
			members = rows[labels == cell]
			if len(members):
				updated[cell] = members.mean(0)
		if np.allclose(updated, result):
			break
		result = updated
	return np.asarray(result, dtype=np.float64)


def assign(rows: Array, centroids: Array) -> Array:
	return np.asarray(squared_distances(rows, centroids).argmin(1))


def rarefied(counts: Sequence[int], size: int) -> float:
	"""Expected occupied cells in a uniform random subset of `size` of the distinct prompts.

	`counts` holds the distinct prompts per occupied cell. A cell with m of N prompts is missed
	with probability C(N - m, size) / C(N, size). The value depends only on the counts, never on
	the order the prompts arrived in.
	"""
	total = sum(counts)
	if size > total:
		raise ValueError("a rarefaction size cannot exceed the distinct prompts")
	expected = 0.0
	# Sorted, so the floating-point sum is identical for every arrival order.
	for m in sorted(counts):
		if total - m < size:
			expected += 1.0
			continue
		missed = (
			math.lgamma(total - m + 1)
			- math.lgamma(total - m - size + 1)
			- math.lgamma(total + 1)
			+ math.lgamma(total - size + 1)
		)
		expected += 1.0 - math.exp(missed)
	return expected


def coverage_report(
	shadow: ShadowCalibration, counts: Sequence[int], distinct: int, matched: str
) -> dict[str, Any]:
	"""Raw coverage and its rank against every calibrated profile, abstaining per profile."""
	ranks: dict[str, dict[str, Any]] = {}
	for name, null in shadow.coverage.items():
		reached = [c for c in shadow.settings.checkpoints if c <= distinct]
		usable = [c for c in reached if len(null.rarefied.get(str(c), [])) >= shadow.minimum_null]
		entry: dict[str, Any] = {"checkpoint": None, "rarefied": None, "p_value": None}
		if not reached:
			entry["abstain"] = f"{distinct} distinct prompts, under the first checkpoint"
		elif not usable:
			largest = len(null.rarefied.get(str(reached[0]), []))
			entry["abstain"] = (
				f"{largest} benign {name} horizons at {reached[0]} prompts; "
				f"need {shadow.minimum_null}"
			)
		else:
			checkpoint = usable[-1]
			values = null.rarefied[str(checkpoint)]
			value = rarefied(counts, checkpoint)
			entry.update(
				checkpoint=checkpoint,
				rarefied=value,
				p_value=upper_p(values, value),
				null_horizons=len(values),
				abstain=None,
			)
		# Raw cells grow with volume; this rank mixes breadth with the number of prompts.
		entry["cells_p_value"] = upper_p(null.cells, float(len(counts))) if null.cells else None
		ranks[name] = entry
	return {"distinct": distinct, "cells": len(counts), "profile": matched, "ranks": ranks}


@dataclass
class _Horizon:
	start: float
	digests: set[str] = field(default_factory=set)
	cells: Counter[int] = field(default_factory=Counter)
	domains: set[str] = field(default_factory=set)
	buffers: dict[str, list[tuple[Array, str]]] = field(default_factory=dict)
	windows: Counter[str] = field(default_factory=Counter)


class ShadowMonitor:
	"""Per-account shadow state for the current horizon, kept in this process only.

	`observe` returns the coverage after the request and, when the request completes a window
	in its domain, that window's conditional score. Nothing here is read by any policy.
	"""

	def __init__(
		self,
		shadow: ShadowCalibration,
		*,
		task: str,
		window_size: int,
		scales: tuple[float, ...],
		distinct_prompts: bool = True,
	) -> None:
		if shadow._centroids is None:
			raise ValueError("shadow arrays are not loaded")
		self.shadow = shadow
		self.task = task
		self.window_size = window_size
		self.distinct_prompts = distinct_prompts
		self.centroids = shadow._centroids
		self.scorers = {
			domain: WindowScorer(references, shadow.domains[domain].sigma or 1.0, scales)
			for domain, references in shadow._references.items()
		}
		self.accounts: dict[str, _Horizon] = {}
		self._lock = threading.Lock()

	def observe(
		self, account: str, horizon: float, text: str, embedding: Array, digest: str
	) -> dict[str, Any]:
		row = np.asarray(embedding, dtype=np.float64)
		domain = request_domain(self.task, text)
		with self._lock:
			state = self.accounts.get(account)
			if state is None or state.start != horizon:
				state = self.accounts[account] = _Horizon(horizon)
			state.domains.add(domain)
			if digest not in state.digests:
				state.digests.add(digest)
				state.cells[int(assign(row[None, :], self.centroids)[0])] += 1
			buffer = state.buffers.setdefault(domain, [])
			buffer.append((row, digest))
			window = None
			if len(buffer) == self.window_size:
				state.windows[domain] += 1
				window = (state.windows[domain], list(buffer))
				buffer.clear()
			counts = list(state.cells.values())
			distinct = len(state.digests)
			matched = profile(self.task, state.domains)
		conditional = None
		if window is not None:
			conditional = self.conditional(domain, window[0], window[1])
		return {
			"version": self.shadow.version,
			"coverage": coverage_report(self.shadow, counts, distinct, matched),
			"conditional": conditional,
		}

	def conditional(self, domain: str, index: int, rows: list[tuple[Array, str]]) -> dict[str, Any]:
		fitted = self.shadow.domains.get(domain)
		result: dict[str, Any] = {"domain": domain, "window": index, "score": None, "p_value": None}
		if fitted is None:
			return result | {"abstain": "domain absent from the shadow calibration"}
		if fitted.abstain is not None or domain not in self.scorers:
			return result | {"abstain": fitted.abstain or "no reference windows"}
		window = np.stack([row for row, _ in rows])
		keys = [key for _, key in rows] if self.distinct_prompts else None
		score = float(self.scorers[domain].score(window, keys))
		return result | {
			"score": score,
			"p_value": upper_p(fitted.null_scores, score),
			"null_windows": len(fitted.null_scores),
			"independent_windows": fitted.independent_windows,
			"abstain": None,
		}


def calibrate_shadow(
	*,
	settings: ShadowConfig,
	task: str,
	window_size: int,
	reference_windows: int,
	bandwidth_sample: int,
	scales: tuple[float, ...],
	horizon_streams: Callable[[Sequence[Any]], list[list[int]]],
	fitting: Sequence[Any],
	fitted: Array,
	score_calibration: Sequence[Any],
	null_embedded: Array,
	seed: int,
	distinct_prompts: bool = True,
) -> ShadowCalibration:
	"""Fit both shadow scorers from the fitting and score-calibration roles only.

	Fitting draws each domain's references and bandwidth and the coverage cells. The score
	calibration role gives each domain's null and the coverage null. `horizon_streams` splits a
	role into per-account horizons as lists of row indices, capped like the policy's horizons.
	Policy calibration is not used: shadow scores have no thresholds.
	"""
	rng = random.Random(seed + 3)
	minimum = minimum_null(settings.target)
	size = window_size
	stride = settings.null_stride or max(1, size // 4)
	fit_domains: dict[str, list[int]] = {}
	for index, row in enumerate(fitting):
		fit_domains.setdefault(request_domain(task, row.text), []).append(index)
	null_horizons = horizon_streams(score_calibration)
	null_domains = [
		{request_domain(task, score_calibration[i].text) for i in horizon}
		for horizon in null_horizons
	]
	domains: dict[str, ConditionalDomain] = {}
	references: dict[str, Array] = {}
	for domain in sorted(set(fit_domains) | {d for ds in null_domains for d in ds}):
		rows = fit_domains.get(domain, [])
		order = rng.sample(rows, len(rows))
		count = min(reference_windows, len(order) // size)
		if count < 1:
			domains[domain] = ConditionalDomain(
				abstain=f"{len(rows)} fitting requests; one reference window needs {size}"
			)
			continue
		windows = np.stack([fitted[order[i * size : (i + 1) * size]] for i in range(count)])
		sample = order[count * size :][:bandwidth_sample] or order[:bandwidth_sample]
		sigma = median_bandwidth(fitted[sorted(sample)])
		scorer = WindowScorer(windows, sigma, scales)
		scores: list[float] = []
		independent = 0
		accounts = 0
		for horizon, seen in zip(null_horizons, null_domains, strict=True):
			# Specialists only: a single-domain horizon in this domain.
			if seen != {domain}:
				continue
			accounts += 1
			independent += len(horizon) // size
			for start in range(0, len(horizon) - size + 1, stride):
				members = horizon[start : start + size]
				keys = [score_calibration[i].text for i in members] if distinct_prompts else None
				scores.append(scorer.score(null_embedded[members], keys))
		abstain = None
		if independent < minimum:
			abstain = (
				f"{independent} non-overlapping specialist windows; target {settings.target} "
				f"needs {minimum}"
			)
		domains[domain] = ConditionalDomain(
			sigma=sigma,
			reference_windows=count,
			null_scores=sorted(scores),
			null_accounts=accounts,
			independent_windows=independent,
			abstain=abstain,
		)
		references[domain] = windows
	centroids = kmeans(fitted, settings.cells, settings.kmeans_iterations, seed + 4)
	labels = assign(null_embedded, centroids)
	coverage: dict[str, dict[str, Any]] = {}
	for horizon, seen in zip(null_horizons, null_domains, strict=True):
		first: dict[str, int] = {}
		for i in horizon:
			first.setdefault(score_calibration[i].text, int(labels[i]))
		counts = list(Counter(first.values()).values())
		names = [ALL] if task != "gateway" else [ALL, profile(task, seen)]
		for name in names:
			null = coverage.setdefault(name, {"horizons": 0, "cells": [], "rarefied": {}})
			null["horizons"] += 1
			null["cells"].append(float(len(counts)))
			for checkpoint in settings.checkpoints:
				if checkpoint <= len(first):
					null["rarefied"].setdefault(str(checkpoint), []).append(
						rarefied(counts, checkpoint)
					)
	shadow = ShadowCalibration(
		settings=settings,
		minimum_null=minimum,
		domains=domains,
		coverage={
			name: CoverageNull(
				horizons=value["horizons"],
				cells=sorted(value["cells"]),
				rarefied={k: sorted(v) for k, v in value["rarefied"].items()},
			)
			for name, value in coverage.items()
		},
		counts={
			"fitting_requests": len(fitting),
			"score_calibration_requests": len(score_calibration),
			"score_calibration_horizons": len(null_horizons),
		},
	)
	shadow._centroids = centroids
	shadow._references = references
	return shadow
