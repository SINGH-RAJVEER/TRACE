"""Fixed-horizon shadow evidence fusion; never an enforcement or sequential process.

Superuniform per-family p-values yield e-values with expectation at most one.
A fixed convex mixture retains that guarantee under arbitrary dependence. This
premise requires exchangeable calibration accounts and a prespecified statistic,
profile and horizon; empirical ranks alone do not establish it under drift.
"""

import math
from collections.abc import Mapping
from typing import Any

FAMILIES = ("volume", "session", "novelty", "linkage", "content")


def fuse_evidence(
	families: Mapping[str, float | None],
	*,
	weights: Mapping[str, float] | None = None,
	alpha: float = 0.5,
) -> dict[str, Any]:
	"""Convert prespecified family ranks to e-values and average with fixed weights.

	Missing/abstaining families contribute neutral e=1 without reallocating weight.
	The missingness rule must be fixed by design/profile independently of the
	realized family evidence; data-dependent substitution needs joint calibration.
	Never choose a family's minimum p-value from its correlated component scores.
	Weights, component selection and thresholds must be frozen on development data
	and evaluated once on locked accounts. No time-uniform claim is made.
	"""
	if set(families) - set(FAMILIES):
		raise ValueError("unknown evidence family")
	chosen = dict.fromkeys(FAMILIES, 1 / len(FAMILIES)) if weights is None else dict(weights)
	if set(chosen) != set(FAMILIES):
		raise ValueError("weights must specify every evidence family")
	if any(not math.isfinite(w) or w < 0 for w in chosen.values()) or not math.isclose(
		sum(chosen.values()), 1.0, abs_tol=1e-12
	):
		raise ValueError("weights must be nonnegative and sum to one")
	if not 0 < alpha < 1:
		raise ValueError("alpha must lie strictly between zero and one")
	components: dict[str, dict[str, Any]] = {}
	for family in FAMILIES:
		p = families.get(family)
		if p is not None and (not math.isfinite(p) or not 0 < p <= 1):
			raise ValueError("p-values must lie in (0, 1]")
		e = 1.0 if p is None else (1 - alpha) * p ** (-alpha)
		components[family] = {
			"p_value": p,
			"e_value": e,
			"weight": chosen[family],
			"abstain": p is None,
		}
	return {
		"shadow_only": True,
		"e_value": sum(chosen[name] * entry["e_value"] for name, entry in components.items()),
		"families": components,
		"alpha": alpha,
		"validity": "fixed horizon; superuniform ranks required; arbitrary dependence allowed",
		"sequential_validity": False,
	}
