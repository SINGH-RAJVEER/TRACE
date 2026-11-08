"""Offline, uncertain principal grouping and pooled shadow summaries.

Connected components are an analysis hypothesis, never an authorization relation.
Even declared organization membership does not establish shared intent. There is
no policy import, state mutation, decision, budget or transitive ban here.
"""

import hashlib
import itertools
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from harness.mmd import Array, squared_distances
from harness.shadow import assign


@dataclass(frozen=True)
class PrincipalHorizon:
	principal: str
	prompts: Sequence[str]
	embeddings: Array
	timestamps: Sequence[float] = ()
	created_at: float | None = None
	truth_group: str | None = None


@dataclass(frozen=True)
class LinkEvidence:
	left: str
	right: str
	source: str
	strength: float
	inferred: bool = False


def prompt_template(text: str) -> str:
	"""Case, whitespace and digits normalized, so numeric variants share one template."""
	return re.sub(r"\d+", "#", " ".join(text.lower().split()))


def infer_edges(accounts: Sequence[PrincipalHorizon]) -> list[LinkEvidence]:
	"""Optional heuristic proposals with explicit source labels, not probabilities."""
	result = []
	for left, right in itertools.combinations(accounts, 2):
		if (
			left.created_at is not None
			and right.created_at is not None
			and abs(left.created_at - right.created_at) <= 60
		):
			result.append(
				LinkEvidence(left.principal, right.principal, "creation_within_60s", 0.3, True)
			)
		lt, rt = (
			{prompt_template(p) for p in left.prompts},
			{prompt_template(p) for p in right.prompts},
		)
		if lt and rt:
			share = len(lt & rt) / len(lt | rt)
			if share >= 0.5:
				result.append(
					LinkEvidence(left.principal, right.principal, "shared_templates", share, True)
				)
		if len(left.timestamps) >= 3 and len(right.timestamps) >= 3:
			a, b = np.diff(sorted(left.timestamps)), np.diff(sorted(right.timestamps))
			n = min(len(a), len(b))
			if np.std(a[:n]) > 0 and np.std(b[:n]) > 0:
				correlation = float(np.corrcoef(a[:n], b[:n])[0, 1])
				if math.isfinite(correlation) and correlation >= 0.9:
					result.append(
						LinkEvidence(
							left.principal, right.principal, "gap_correlation", correlation, True
						)
					)
	return result


def analyze_clusters(
	principals: Sequence[PrincipalHorizon],
	*,
	organization_mapping: Mapping[str, str] | None = None,
	edges: Sequence[LinkEvidence] = (),
	infer: bool = False,
	minimum_strength: float = 0.8,
	centroids: Array | None = None,
	score_cluster: Callable[[Sequence[str], Array, Sequence[float]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
	"""Pool groups as one user; report pairwise errors where benign truth is known.

	Complementarity is one minus mean pairwise Jaccard of occupied semantic cells;
	without fitted centroids it uses exact canonical prompt sets and says so.
	Cluster selection itself is adaptive: account nulls do not certify cluster ranks.
	"""
	accounts = {a.principal: a for a in principals}
	if len(accounts) != len(principals):
		raise ValueError("principal identifiers must be unique")
	if not 0 <= minimum_strength <= 1:
		raise ValueError("minimum_strength must lie in [0, 1]")
	for account in principals:
		if account.embeddings.ndim != 2 or len(account.embeddings) != len(account.prompts):
			raise ValueError("each prompt needs one embedding")
		if not np.isfinite(account.embeddings).all():
			raise ValueError("embeddings must be finite")
		if any(not math.isfinite(t) for t in account.timestamps):
			raise ValueError("timestamps must be finite")
		if account.created_at is not None and not math.isfinite(account.created_at):
			raise ValueError("creation times must be finite")
		if account.timestamps and len(account.timestamps) != len(account.prompts):
			raise ValueError("timestamps must be absent or match prompts")
	evidence = list(edges)
	organizations: dict[str, list[str]] = {}
	for principal, organization in (organization_mapping or {}).items():
		if principal not in accounts:
			raise ValueError("organization mapping contains an unknown principal")
		organizations.setdefault(organization, []).append(principal)
	for organization, members in organizations.items():
		for left, right in itertools.combinations(sorted(members), 2):
			evidence.append(LinkEvidence(left, right, f"declared_organization:{organization}", 1.0))
	if infer:
		evidence.extend(infer_edges(principals))
	parents = {name: name for name in accounts}

	def root(name: str) -> str:
		while parents[name] != name:
			name = parents[name]
		return name

	reported_edges = []
	for edge in evidence:
		if edge.left not in accounts or edge.right not in accounts or edge.left == edge.right:
			raise ValueError("edge endpoints must be distinct known principals")
		if (
			not edge.source.strip()
			or not math.isfinite(edge.strength)
			or not 0 <= edge.strength <= 1
		):
			raise ValueError("edges require a source and strength in [0, 1]")
		included = edge.strength >= minimum_strength
		reported_edges.append(asdict(edge) | {"included": included})
		if included:
			parents[root(edge.right)] = root(edge.left)
	groups: dict[str, list[str]] = {}
	for name in sorted(accounts):
		groups.setdefault(root(name), []).append(name)
	reports = []
	for members in groups.values():
		# Stable prompt order for coverage; novelty follows timestamp order when present.
		items = [
			(accounts[m].timestamps[i] if accounts[m].timestamps else float(i), m, i, p)
			for m in members
			for i, p in enumerate(accounts[m].prompts)
		]
		items.sort()
		prompts = [p for _, _, _, p in items]
		rows = np.asarray([accounts[m].embeddings[i] for _, m, i, _ in items], dtype=np.float64)
		if not items:
			rows = np.empty((0, next(iter(accounts.values())).embeddings.shape[1]))
		occupied: list[set[str | int]] = []
		for member in members:
			account = accounts[member]
			if centroids is None:
				occupied.append(
					{
						hashlib.sha256(" ".join(p.lower().split()).encode()).hexdigest()
						for p in account.prompts
					}
				)
			else:
				occupied.append(set(map(int, assign(account.embeddings, centroids))))
		pairs = [
			1 - len(a & b) / len(a | b) for a, b in itertools.combinations(occupied, 2) if a | b
		]
		unique_rows: list[Array] = []
		seen = set()
		for p, row in zip(prompts, rows, strict=True):
			key = " ".join(p.lower().split())
			if key not in seen:
				seen.add(key)
				unique_rows.append(row)
		counts = (
			Counter(map(int, assign(np.asarray(unique_rows), centroids)))
			if centroids is not None and unique_rows
			else Counter()
		)
		novelty = [
			float(np.sqrt(np.min(squared_distances(rows[i : i + 1], rows[:i]))))
			for i in range(1, len(rows))
		]
		report: dict[str, Any] = {
			"members": members,
			"requests": len(prompts),
			"chronological_timestamps_complete": all(
				len(accounts[m].timestamps) == len(accounts[m].prompts) for m in members
			),
			"distinct_prompts": len(seen),
			"coverage_cells": len(counts) if centroids is not None else None,
			"coverage_counts": list(counts.values()),
			"novelty_mean": float(np.mean(novelty)) if novelty else None,
			"complementarity": float(np.mean(pairs)) if pairs else None,
			"complementarity_basis": "semantic_cells" if centroids is not None else "exact_prompts",
			"ranks": score_cluster(prompts, rows, [t for t, _, _, _ in items])
			if score_cluster
			else None,
			"rank_caveat": "account nulls do not calibrate adaptively selected clusters",
		}
		reports.append(report)
	merged = split = same = different = 0
	for a, b in itertools.combinations(principals, 2):
		if a.truth_group is None or b.truth_group is None:
			continue
		if a.truth_group == b.truth_group:
			same += 1
			split += root(a.principal) != root(b.principal)
		else:
			different += 1
			merged += root(a.principal) == root(b.principal)
	return {
		"shadow_only": True,
		"enforcement": False,
		"edges": reported_edges,
		"clusters": reports,
		"uncertainty": "strengths are heuristic or declared, not posterior identity probabilities",
		"benign_link_errors": {
			"false_merges": merged,
			"false_splits": split,
			"different_truth_pairs": different,
			"same_truth_pairs": same,
			"false_merge_rate": merged / different if different else None,
			"false_split_rate": split / same if same else None,
		},
	}
