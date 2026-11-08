"""Whole-campaign evidence monitor: bounded online features, fused ranks, path-calibrated alerts.

This replaces fixed-window MMD as the primary campaign signal. MMD asks whether a window
looks unlike benign traffic, which ranks broad extraction as ordinary. This monitor asks
whether a principal's campaign so far explores more of the request space, and more
systematically, than benign campaigns of the same profile and length:

- `coverage_gain`: mean incremental log-determinant over distinct prompts (harness.logdet).
- `novelty`: mean Euclidean distance from each distinct prompt to its nearest earlier one,
	over a bounded reservoir of recent prompts.
- `cell_coverage`: expected occupied k-means cells in a fixed-size rarefied subset, so it
	does not grow with volume and does not depend on order.
- `template_diversity` and `answer_only_share`: session shape from bounded counters.

Every `look_every` evaluated requests, each feature is ranked against benign campaigns of
the same profile at the same look, so a value is never ranked against another length. Ranks
become e-values with the fixed calibrator 0.5 / sqrt(p), averaged with equal weights; an
abstaining feature contributes e = 1. The
look score is the log of that average. A campaign alerts when the running maximum of its
look scores ranks at or below `alpha` among the running maxima of complete benign
calibration campaigns, the finite-horizon rule of harness.path_calibration. Under
exchangeability of benign campaigns with the same profile and horizon, that bounds the
chance a benign campaign ever alerts within the horizon by `alpha`. Past the horizon the
monitor expires and abstains; it does not rearm.

Calibration replays benign campaigns through the same `observe` code that serves requests,
so features cannot drift between fitting and serving. State per principal is bounded: a
64 x 64 inverse, fixed cell counts, a reservoir of recent projected prompts, and capped
digest sets. Nothing here is qualified: synthetic benign campaigns bound claims to their
generator, short benign horizons cap the useful horizon, and a campaign whose prompts
follow the benign distribution is indistinguishable by any content feature.
"""

import base64
import hashlib
import json
import math
import re
from bisect import bisect_left
from collections.abc import Callable, Sequence
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.logdet import COVERAGE_DIM, CoverageSpec
from harness.mmd import Array
from harness.shadow import assign, rarefied

FEATURES = (
	"coverage_gain",
	"novelty",
	"cell_coverage",
	"template_diversity",
	"answer_only_share",
)
# Bounds on per-principal state. Digest sets stop growing at the cap, so diversity is
# a lower bound for campaigns longer than it, and repeats beyond the cap count as distinct.
DIGEST_CAP = 4096
RESERVOIR = 128
ANSWER_ONLY = re.compile(
	r"\b(answer only|only (?:the )?answer|no explanation|just (?:the )?answer)\b", re.I
)


class EvidenceSettings(BaseModel):
	model_config = ConfigDict(extra="forbid", frozen=True)
	version: Literal["campaign-evidence-v1"] = "campaign-evidence-v1"
	look_every: int = Field(default=50, ge=2, le=10_000)
	rarefied_size: int = Field(default=40, ge=2)
	alpha: float = Field(default=0.01, gt=0, lt=1)
	cells: int = Field(default=64, ge=2, le=1024)
	seed: int = 0
	ridge: float = Field(default=1.0, ge=1e-3, le=1e3)

	@model_validator(mode="after")
	def validate_sizes(self) -> Self:
		if self.rarefied_size > self.look_every:
			raise ValueError("the rarefaction size cannot exceed the first look")
		return self


def _encode(values: Array) -> str:
	return base64.b64encode(np.ascontiguousarray(values, dtype=np.float64).tobytes()).decode()


def _decode(text: str, shape: tuple[int, ...]) -> Array:
	return np.frombuffer(base64.b64decode(text), dtype=np.float64).reshape(shape).copy()


class EvidenceCalibration(BaseModel):
	"""Frozen artifact: settings, projection, cells, feature nulls and benign path maxima."""

	model_config = ConfigDict(extra="forbid", frozen=True)
	settings: EvidenceSettings
	coverage: CoverageSpec
	centroids: str
	centroid_shape: tuple[int, int]
	# profile -> feature@requests -> sorted values, one per benign campaign per look.
	nulls: dict[str, dict[str, list[float]]]
	# profile -> sorted running maxima of complete benign campaigns, and their horizon.
	maxima: dict[str, list[float]]
	horizon_looks: dict[str, int]
	encoder: dict[str, Any] = Field(default_factory=dict)

	@property
	def digest(self) -> str:
		return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

	def cell_centroids(self) -> Array:
		return _decode(self.centroids, self.centroid_shape)


def new_state() -> dict[str, Any]:
	return {
		"version": "campaign-evidence-v1",
		"count": 0,
		"distinct": 0,
		"digests": [],
		"templates": [],
		"answer_only": 0,
		"cells": {},
		"inverse": None,
		"gain_sum": 0.0,
		"gain_count": 0,
		"reservoir": [],
		"nn_sum": 0.0,
		"nn_count": 0,
		"domains": [],
		"looks": 0,
		"profile": None,
		"running_max": None,
		"alert_look": None,
		"expired": False,
	}


def profile_of(domains: set[str]) -> str:
	if domains <= {"all"}:
		return "all"
	return "single_domain" if len(domains) == 1 else "multi_domain"


def _template(text: str) -> str:
	shape = re.sub(r"\b\d+\b|\"[^\"]*\"|'[^']*'", "value", text.lower())
	return hashlib.sha256(" ".join(re.findall(r"\w+", shape)).encode()).hexdigest()[:16]


def features(state: dict[str, Any], settings: EvidenceSettings) -> dict[str, float | None]:
	"""Current feature values; None where a feature has too few observations."""
	count = state["count"]
	counts = list(state["cells"].values())
	return {
		"coverage_gain": state["gain_sum"] / state["gain_count"] if state["gain_count"] else None,
		"novelty": state["nn_sum"] / state["nn_count"] if state["nn_count"] else None,
		"cell_coverage": (
			rarefied(counts, settings.rarefied_size)
			if state["distinct"] >= settings.rarefied_size
			else None
		),
		"template_diversity": len(state["templates"]) / count if count else None,
		"answer_only_share": state["answer_only"] / count if count else None,
	}


def update(
	state: dict[str, Any],
	*,
	text: str,
	domain: str,
	embedding: Sequence[float] | Array,
	settings: EvidenceSettings,
	coverage: CoverageSpec,
	centroids: Array,
) -> bool:
	"""Absorb one evaluated request. Returns True when it completes a look."""
	if state["expired"]:
		return False
	state["count"] += 1
	if domain not in state["domains"]:
		state["domains"] = sorted([*state["domains"], domain])
	if ANSWER_ONLY.search(text):
		state["answer_only"] += 1
	template = _template(text)
	if template not in state["templates"] and len(state["templates"]) < DIGEST_CAP:
		state["templates"].append(template)
	digest = hashlib.sha256(text.encode()).hexdigest()[:24]
	if digest not in state["digests"]:
		if len(state["digests"]) < DIGEST_CAP:
			state["digests"].append(digest)
		state["distinct"] += 1
		vector = np.asarray(embedding, dtype=np.float64)
		cell = str(int(assign(vector[None, :], centroids)[0]))
		state["cells"][cell] = state["cells"].get(cell, 0) + 1
		projected = coverage.project(vector[None, :])[0]
		inverse = (
			np.eye(COVERAGE_DIM) / settings.ridge
			if state["inverse"] is None
			else _decode(state["inverse"], (COVERAGE_DIM, COVERAGE_DIM))
		)
		u = inverse @ projected
		quad = max(0.0, float(projected @ u))
		if state["distinct"] > 1:
			state["gain_sum"] += math.log1p(quad)
			state["gain_count"] += 1
		inverse -= np.outer(u, u) / (1.0 + quad)
		state["inverse"] = _encode(inverse)
		if state["reservoir"]:
			earlier = np.asarray([_decode(v, (COVERAGE_DIM,)) for v in state["reservoir"]])
			state["nn_sum"] += float(np.sqrt(((earlier - projected) ** 2).sum(1)).min())
			state["nn_count"] += 1
		state["reservoir"] = [*state["reservoir"], _encode(projected)][-RESERVOIR:]
	return bool(state["count"] % settings.look_every == 0)


def _rank(null: list[float], value: float) -> float:
	return (1 + len(null) - bisect_left(null, value)) / (len(null) + 1)


def look_score(
	values: dict[str, float | None], profile: str, count: int, calibration: EvidenceCalibration
) -> tuple[float, dict[str, Any]]:
	"""Log of the equal-weight mean of per-feature e-values, and each feature's rank."""
	nulls = calibration.nulls.get(profile, {})
	minimum = math.ceil(1 / calibration.settings.alpha) - 1
	detail: dict[str, Any] = {}
	total = 0.0
	for name in FEATURES:
		value = values[name]
		null = nulls.get(f"{name}@{count}", [])
		if value is None or len(null) < minimum:
			detail[name] = {"value": value, "p_value": None, "null": len(null)}
			total += 1.0
			continue
		p = _rank(null, value)
		detail[name] = {"value": value, "p_value": p, "null": len(null)}
		total += 0.5 / math.sqrt(p)
	return math.log(total / len(FEATURES)), detail


def observe(
	state: dict[str, Any],
	*,
	text: str,
	domain: str,
	embedding: Sequence[float] | Array,
	calibration: EvidenceCalibration,
	centroids: Array,
) -> dict[str, Any] | None:
	"""Update a campaign and, at a look, return its evidence. Never changes a response."""
	settings = calibration.settings
	if not update(
		state,
		text=text,
		domain=domain,
		embedding=embedding,
		settings=settings,
		coverage=calibration.coverage,
		centroids=centroids,
	):
		return None
	state["looks"] += 1
	profile = profile_of(set(state["domains"]))
	if state["profile"] is None:
		state["profile"] = profile
	elif profile != state["profile"]:
		# The stratum is fixed at the first look, as in calibration; a campaign that leaves it
		# expires instead of comparing its running maximum against another profile's null.
		state["expired"] = True
		return {
			"look": state["looks"],
			"profile": profile,
			"decision": "expired",
			"reason": "left the profile fixed at the first look",
			"alert_look": state["alert_look"],
			"qualified": False,
		}
	horizon = calibration.horizon_looks.get(profile)
	maxima = calibration.maxima.get(profile, [])
	if horizon is None or state["looks"] > horizon:
		state["expired"] = True
		return {
			"look": state["looks"],
			"profile": profile,
			"decision": "expired" if horizon is not None else "abstain",
			"reason": "past the benign horizon" if horizon is not None else "no benign null",
			"qualified": False,
		}
	score, detail = look_score(features(state, settings), profile, state["count"], calibration)
	state["running_max"] = (
		score if state["running_max"] is None else max(state["running_max"], score)
	)
	if 1 / (len(maxima) + 1) > settings.alpha:
		decision, p = "abstain", None
	else:
		p = _rank(maxima, state["running_max"])
		if p <= settings.alpha and state["alert_look"] is None:
			state["alert_look"] = state["looks"]
		decision = "alert" if state["alert_look"] is not None else "continue"
	return {
		"look": state["looks"],
		"requests": state["count"],
		"profile": profile,
		"score": score,
		"running_max": state["running_max"],
		"p_value": p,
		"decision": decision,
		"alert_look": state["alert_look"],
		"features": detail,
		"qualified": False,
	}


Campaign = list[tuple[str, str, Array]]


def _replay(
	campaign: Campaign,
	settings: EvidenceSettings,
	coverage: CoverageSpec,
	centroids: Array,
	visit: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
	state = new_state()
	for text, domain, embedding in campaign:
		if update(
			state,
			text=text,
			domain=domain,
			embedding=embedding,
			settings=settings,
			coverage=coverage,
			centroids=centroids,
		):
			visit(state)
	return state


def calibrate(
	*,
	fitting: Array,
	score_campaigns: Sequence[Campaign],
	policy_campaigns: Sequence[Campaign],
	settings: EvidenceSettings,
	encoder: dict[str, Any] | None = None,
) -> EvidenceCalibration:
	"""Feature nulls from score campaigns, then path maxima from separate policy campaigns.

	Each benign campaign is one account's evaluated requests in order. Score and policy
	campaigns must be different accounts; neither may overlap the locked test.
	"""
	from harness.shadow import kmeans

	if fitting.ndim != 2 or len(fitting) < settings.cells:
		raise ValueError("fitting needs at least one embedding per cell")
	coverage = CoverageSpec(seed=settings.seed, input_dim=int(fitting.shape[1]))
	centroids = kmeans(fitting, settings.cells, 25, settings.seed)
	nulls: dict[str, dict[str, list[float]]] = {}
	for campaign in score_campaigns:

		def record(state: dict[str, Any]) -> None:
			profile = profile_of(set(state["domains"]))
			for name, value in features(state, settings).items():
				if value is not None:
					key = f"{name}@{state['count']}"
					nulls.setdefault(profile, {}).setdefault(key, []).append(value)

		_replay(campaign, settings, coverage, centroids, record)
	frozen_nulls = {p: {k: sorted(v) for k, v in m.items()} for p, m in nulls.items()}
	partial = EvidenceCalibration(
		settings=settings,
		coverage=coverage,
		centroids=_encode(centroids),
		centroid_shape=(int(centroids.shape[0]), int(centroids.shape[1])),
		nulls=frozen_nulls,
		maxima={},
		horizon_looks={},
		encoder=encoder or {},
	)
	paths: dict[str, list[tuple[list[float], bool]]] = {}
	for campaign in policy_campaigns:
		scores: list[tuple[str, float]] = []

		def score(state: dict[str, Any], scores: list[tuple[str, float]] = scores) -> None:
			profile = profile_of(set(state["domains"]))
			value, _ = look_score(features(state, settings), profile, state["count"], partial)
			scores.append((profile, value))

		_replay(campaign, settings, coverage, centroids, score)
		if not scores:
			continue
		# As at runtime, the stratum is fixed at the first look and the path ends on leaving it.
		profile = scores[0][0]
		kept = [
			v
			for _, v in scores[
				: next((i for i, (p, _) in enumerate(scores) if p != profile), len(scores))
			]
		]
		paths.setdefault(profile, []).append((kept, len(kept) == len(scores)))
	maxima: dict[str, list[float]] = {}
	horizons: dict[str, int] = {}
	for profile, entries in paths.items():
		# The horizon is the shortest benign path that stayed in its profile; every path is
		# then cut at it, so calibration and runtime see the same looks.
		complete = [len(path) for path, whole in entries if whole]
		horizon = min(complete) if complete else min(len(path) for path, _ in entries)
		horizons[profile] = horizon
		maxima[profile] = sorted(max(path[:horizon]) for path, _ in entries)
	return partial.model_copy(update={"maxima": maxima, "horizon_looks": horizons})


def json_state(state: dict[str, Any]) -> str:
	return json.dumps(state, separators=(",", ":"), allow_nan=False)
