"""Bounded offline overlap plus co-activity proposals, for corroboration only."""

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from harness.calibration import BenignRequest
from harness.clusters import prompt_template


def propose_links(
	requests: Sequence[BenignRequest],
	*,
	max_accounts: int = 256,
	max_requests: int = 512,
	max_features: int = 256,
	max_df: float = 0.5,
	minimum_overlap: float = 0.5,
	time_bin_seconds: float = 10,
	minimum_coactivity: float = 0.5,
) -> dict[str, Any]:
	"""Keep bottom-hash template sketches; require two rare shared templates and bins.

	Numbers normalize to #. Public templates present in more than max_df of accounts
	are excluded. No identity or infrastructure fields enter the features.
	"""
	if (
		min(max_accounts, max_requests, max_features) < 2
		or max_accounts > 1024
		or max_requests > 4096
		or max_features > 1024
	):
		raise ValueError("bounds must be at least two and within 1024/4096/1024")
	if not all(
		math.isfinite(v) and 0 < v <= 1 for v in (max_df, minimum_overlap, minimum_coactivity)
	):
		raise ValueError("thresholds must be in (0, 1]")
	if not math.isfinite(time_bin_seconds) or time_bin_seconds <= 0:
		raise ValueError("time bin must be positive and finite")
	groups: dict[str, list[BenignRequest]] = defaultdict(list)
	for row in requests:
		if row.time is not None and not math.isfinite(row.time):
			raise ValueError("timestamps must be finite")
		groups[row.account].append(row)
		if len(groups) > max_accounts or len(groups[row.account]) > max_requests:
			raise ValueError("input exceeds account/request bounds; select a fixed interval")
	sketches = {}
	bins = {}
	for account, rows in sorted(groups.items()):
		features = set()
		for row in rows:
			if len(row.text) > 32768:
				raise ValueError("prompt exceeds 32768 characters")
			template = prompt_template(row.text)
			if template:
				features.add(hashlib.sha256(template.encode()).hexdigest())
		sketches[account] = features
		bins[account] = {math.floor(r.time / time_bin_seconds) for r in rows if r.time is not None}
	df = Counter(feature for features in sketches.values() for feature in features)
	common = {feature for feature, count in df.items() if count > max_df * len(groups)}
	# An inverted index avoids an unbounded all-pairs search. Hard account bounds also cap pairs.
	index: dict[str, list[str]] = defaultdict(list)
	for account in sketches:
		sketches[account] = set(sorted(sketches[account] - common)[:max_features])
		for feature in sorted(sketches[account]):
			index[feature].append(account)
	pairs: Counter[tuple[str, str]] = Counter()
	for members in index.values():
		for i, left in enumerate(members):
			for right in members[i + 1 :]:
				pairs[left, right] += 1
	edges = []
	provenance = []
	for (left, right), shared in sorted(pairs.items()):
		overlap = shared / len(sketches[left] | sketches[right])
		shared_bins = len(bins[left] & bins[right])
		coactivity = shared_bins / len(bins[left] | bins[right]) if bins[left] | bins[right] else 0
		if (
			shared < 2
			or shared_bins < 2
			or overlap < minimum_overlap
			or coactivity < minimum_coactivity
		):
			continue
		edges.append(
			{
				"left": left,
				"right": right,
				"source": "rare_template_coactivity_v1",
				"strength": min(overlap, coactivity),
				"inferred": True,
			}
		)
		provenance.append(
			{
				"left": left,
				"right": right,
				"shared_templates": shared,
				"template_jaccard": overlap,
				"shared_time_bins": shared_bins,
				"time_jaccard": coactivity,
				"template_sha256": sorted(sketches[left] & sketches[right]),
			}
		)
	canonical = sorted((r.account, r.text, repr(r.time)) for r in requests)
	return {
		"version": "offline-account-links-v1",
		"organizations": {},
		"edges": edges,
		"infer": False,
		"minimum_strength": min(minimum_overlap, minimum_coactivity),
		"provenance": {
			"input_sha256": hashlib.sha256(repr(canonical).encode()).hexdigest(),
			"features": provenance,
			"excluded_common_templates": len(common),
			"bounds": {
				"max_accounts": max_accounts,
				"max_requests": max_requests,
				"max_features": max_features,
			},
			"thresholds": {
				"max_df": max_df,
				"minimum_overlap": minimum_overlap,
				"minimum_coactivity": minimum_coactivity,
				"time_bin_seconds": time_bin_seconds,
			},
		},
		"shadow_only": True,
		"enforcement": False,
		"qualified": False,
	}
