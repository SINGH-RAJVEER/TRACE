"""Offline opt-in evidence analysis. No policy, service state or budgets are consulted."""

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harness.calibration import BenignRequest, Calibration, horizon_indices
from harness.corpus import CorpusIndex
from harness.fusion import fuse_evidence
from harness.logdet import CoverageSpec, length_bin
from harness.mmd import Array
from harness.shadow import assign, coverage_report, profile, request_domain
from harness.signals import SignalCalibration, SignalRow, fit_signals, score_signals


def signal_rows(
	requests: Sequence[BenignRequest], embeddings: Array, calibration: Calibration
) -> list[SignalRow]:
	if len(requests) != len(embeddings):
		raise ValueError("one embedding is required per request")
	task = calibration.config.task
	topics = (
		assign(embeddings, calibration.shadow._centroids)
		if calibration.shadow is not None and calibration.shadow._centroids is not None
		else None
	)
	return [
		SignalRow(
			text=json.loads(row.text)["turn"] if task == "gateway" else row.text,
			time=row.time,
			embedding=tuple(float(v) for v in embeddings[i]),
			domain=request_domain(task, row.text),
			topic=str(topics[i]) if topics is not None else None,
		)
		for i, row in enumerate(requests)
	]


def fit_evidence(
	requests: list[BenignRequest],
	embeddings: Array,
	calibration: Calibration,
	corpus: CorpusIndex | None = None,
	coverage: CoverageSpec | None = None,
) -> SignalCalibration:
	config = calibration.config
	rows = signal_rows(requests, embeddings, calibration)
	groups = horizon_indices(requests, config.horizon_seconds, config.horizon_attempts)
	horizons = [[rows[i] for i in group] for group in groups]
	profiles = [profile(config.task, {row.domain for row in horizon}) for horizon in horizons]
	return fit_signals(horizons, profiles=profiles, corpus=corpus, coverage=coverage)


def check_evidence(
	requests: list[BenignRequest],
	embeddings: Array,
	calibration: Calibration,
	null: SignalCalibration,
	corpus: CorpusIndex | None = None,
	reorder: str = "none",
) -> dict[str, Any]:
	config = calibration.config
	rows = signal_rows(requests, embeddings, calibration)
	groups = horizon_indices(requests, config.horizon_seconds, config.horizon_attempts)
	units: list[dict[str, Any]] = []
	for group in groups:
		members = [rows[i] for i in group]
		if reorder == "topic":
			# Preserve time slots, while changing the prompts assigned to those slots.
			ordered = sorted(members, key=lambda row: (row.domain, row.topic or "", row.embedding))
			members = [
				SignalRow(
					text=row.text,
					time=slot.time,
					embedding=row.embedding,
					domain=row.domain,
					topic=row.topic,
				)
				for row, slot in zip(ordered, members, strict=True)
			]
		matched = profile(config.task, {row.domain for row in members})
		report = score_signals(members, profile=matched, calibration=null, corpus=corpus)
		metrics = report["metrics"]
		# Frozen representatives; never choose the smallest correlated component rank.
		report["fusion"] = fuse_evidence(
			{
				"volume": metrics["request_count"]["p_value"],
				"session": metrics["revisit_followup_rate"]["p_value"],
				"novelty": metrics["novelty_mean_nn"]["p_value"],
				"linkage": None,
				"content": metrics["corpus_match_share"]["p_value"],
			}
		)
		units.append(
			{
				"account": requests[group[0]].account,
				"requests": len(group),
				"profile": matched,
				"signals": report,
			}
		)
	result: dict[str, Any] = {
		"development_only": True,
		"reorder": reorder,
		"horizons": len(units),
		"units": units,
	}
	if null.coverage is not None:
		bands: dict[str, Any] = {}
		for unit in units:
			band = length_bin(unit["requests"]) if unit["requests"] >= 2 else "insufficient"
			key = f"{unit['profile']}:{band}"
			entry = bands.setdefault(
				key,
				{
					"profile": unit["profile"],
					"length_band": band,
					"horizons": 0,
					"paired_ranked": 0,
					"metrics": {},
				},
			)
			entry["horizons"] += 1
			metrics = unit["signals"]["metrics"]
			names = ("coverage_mean_logdet", "novelty_mean_nn")
			entry["paired_ranked"] += all(metrics[n]["p_value"] is not None for n in names)
			for name in names:
				counts = entry["metrics"].setdefault(
					name,
					{
						"ranked": 0,
						"abstained": 0,
						"at_or_below_target": 0,
						"abstention_reasons": {},
					},
				)
				metric = metrics[name]
				p = metric["p_value"]
				counts["ranked"] += p is not None
				counts["abstained"] += p is None
				counts["at_or_below_target"] += p is not None and p <= null.target
				if p is None:
					reason = metric["abstain"]
					counts["abstention_reasons"][reason] = (
						counts["abstention_reasons"].get(reason, 0) + 1
					)
		result["coverage_comparison"] = {
			"qualified": False,
			"target": null.target,
			"bands": list(bands.values()),
			"source_disjointness": "operator must verify held-out source families",
			"interpretation": "paired descriptive ranks; no labeled separation or measured gain",
		}
	return result


def file_hash(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def cluster_evidence(
	requests: list[BenignRequest],
	embeddings: Array,
	calibration: Calibration,
	null: SignalCalibration,
	links: dict[str, Any],
	corpus: CorpusIndex | None = None,
) -> dict[str, Any]:
	"""Link document holds organizations, edges, metadata and optional benign truth."""
	from harness.clusters import LinkEvidence, PrincipalHorizon, analyze_clusters

	groups: dict[str, list[int]] = {}
	for i, request in enumerate(requests):
		groups.setdefault(request.account, []).append(i)
	principals = [
		PrincipalHorizon(
			principal=account,
			prompts=[requests[i].text for i in indices],
			embeddings=embeddings[indices],
			timestamps=[float(requests[i].time or 0) for i in indices]
			if all(requests[i].time is not None for i in indices)
			else [],
			created_at=links.get("created_at", {}).get(account),
			truth_group=links.get("benign_truth", {}).get(account),
		)
		for account, indices in groups.items()
	]
	shadow = calibration.shadow

	def score(prompts: Sequence[str], vectors: Array, times: Sequence[float]) -> dict[str, Any]:
		pooled = [
			BenignRequest(account="cluster", text=p, time=t)
			for p, t in zip(prompts, times, strict=True)
		]
		signal = signal_rows(pooled, vectors, calibration)
		matched = profile(calibration.config.task, {r.domain for r in signal})
		report = score_signals(signal, profile=matched, calibration=null, corpus=corpus)
		coverage = None
		if shadow is not None and shadow._centroids is not None:
			unique = {
				p: int(c) for p, c in zip(prompts, assign(vectors, shadow._centroids), strict=True)
			}
			coverage = coverage_report(
				shadow, list(Counter(unique.values()).values()), len(unique), matched
			)
		return {"signals": report, "coverage": coverage, "calibrated_cluster_claim": False}

	return analyze_clusters(
		principals,
		organization_mapping=links.get("organizations", {}),
		edges=[LinkEvidence(**edge) for edge in links.get("edges", [])],
		infer=links.get("infer", False),
		minimum_strength=links.get("minimum_strength", 0.8),
		centroids=shadow._centroids if shadow is not None else None,
		score_cluster=score,
	)
