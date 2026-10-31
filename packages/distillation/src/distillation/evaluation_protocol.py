"""Evidence-aware summaries and planning for acquisition and utility experiments."""

import math
import random
from collections import defaultdict
from typing import Any

CHECKPOINTS = (100, 300, 1000, 3000, 6000, 12000)
BASELINES = (
	"allow",
	"full_mmd",
	"instantaneous_mmd",
	"request_quota",
	"rate",
	"request_token_quota",
)


def campaign_report(events: list[dict[str, Any]]) -> dict[str, Any]:
	"""Summarize observed events without inferring admission from a successful response."""
	attempts = [e for e in events if e.get("event") == "attempt"]
	outcomes = {e["id"]: e for e in events if e.get("event") == "outcome"}
	decisions = {e["policy_request_id"]: e for e in events if e.get("event") == "policy"}
	released = [a for a in attempts if outcomes.get(a["id"], {}).get("status") == "received"]
	tokens = [outcomes[a["id"]].get("output_tokens") for a in released]

	def first(predicate: Any) -> float | None:
		return min((float(a["at_seconds"]) for a in attempts if predicate(a)), default=None)

	def decision(a: dict[str, Any]) -> dict[str, Any]:
		return decisions.get(a.get("policy_request_id"), {})

	return {
		"attempts": len(attempts),
		"admitted": None,
		"admission_reason": "current event log does not record admission for every attempt",
		"released": len(released),
		"failed": sum(
			outcomes.get(a["id"], {}).get("status") in ("failed", "deadline") for a in attempts
		),
		"cancelled": None,
		"pending": sum(a["id"] not in outcomes for a in attempts),
		"released_tokens": sum(tokens) if all(t is not None for t in tokens) else None,
		"known_released_tokens": sum(t for t in tokens if t is not None),
		"released_tokens_unknown_count": sum(t is None for t in tokens),
		"distinct_examples": len({a["example_id"] for a in released if a.get("example_id")})
		if all(a.get("example_id") for a in released)
		else None,
		"first_alert_seconds": first(lambda a: decision(a).get("level", 0) >= 1),
		# Policy events are keyed by prompt, so a redeemed challenge keeps only its later serve
		# decision; the per-attempt outcome status still records the challenge itself.
		"first_restriction_seconds": first(
			lambda a: (
				outcomes.get(a["id"], {}).get("status") in ("challenge", "blocked")
				or decision(a).get("action") in ("challenge", "refuse")
			)
		),
		"first_refusal_seconds": first(
			lambda a: (
				outcomes.get(a["id"], {}).get("status") == "blocked"
				or decision(a).get("action") == "refuse"
			)
		),
		"levels": {
			f"L{level}": first(lambda a, level=level: decision(a).get("level", 0) >= level)
			for level in range(1, 5)
		},
		"restriction_coverage": "challenge and refusal actions only; delayed serve is not recorded",
		"decision_attribution_complete": all(
			a.get("policy_request_id") in decisions for a in attempts if a.get("kind") == "query"
		),
	}


def benign_profiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
	"""Aggregate account records; missing outcomes remain unknown."""
	groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		groups[row["profile"]].append(row)
	result = {}
	for profile, group in sorted(groups.items()):
		entry: dict[str, Any] = {"accounts": len(group)}
		for field in ("escalated", "restricted", "deadline_completed"):
			values = [row.get(field) for row in group]
			entry[field] = {
				"yes": sum(v is True for v in values),
				"known": sum(v is not None for v in values),
				"total": len(values),
			}
		result[profile] = entry
	return result


def zero_event_upper(n: int, alpha: float = 0.05) -> float:
	"""Exact one-sided binomial zero-event bound, under independent trials."""
	if n < 1 or not 0 < alpha < 1:
		raise ValueError("need positive independent trials and alpha in (0, 1)")
	return 1.0 - math.exp(math.log(alpha) / n)


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> dict[str, Any]:
	"""Exact equal-tailed binomial interval, assuming independent trials."""
	if trials < 1 or not 0 <= successes <= trials or not 0 < alpha < 1:
		raise ValueError("invalid binomial counts or alpha")

	def cdf(k: int, probability: float) -> float:
		if probability == 0:
			return 1.0
		if probability == 1:
			return 0.0 if k < trials else 1.0
		terms = [
			math.lgamma(trials + 1)
			- math.lgamma(i + 1)
			- math.lgamma(trials - i + 1)
			+ i * math.log(probability)
			+ (trials - i) * math.log1p(-probability)
			for i in range(k + 1)
		]
		return min(1.0, sum(math.exp(term) for term in terms))

	def solve(k: int, target: float) -> float:
		lo, hi = 0.0, 1.0
		for _ in range(60):
			mid = (lo + hi) / 2
			if cdf(k, mid) > target:
				lo = mid
			else:
				hi = mid
		return (lo + hi) / 2

	return {
		"lower": solve(successes - 1, 1 - alpha / 2) if successes else 0.0,
		"upper": solve(successes, alpha / 2) if successes < trials else 1.0,
		"warning": "Assumes independent trials; shared source families make "
		"this interval optimistic.",
	}


def cluster_interval(
	rows: list[tuple[str, float]], *, seed: int = 42, replicates: int = 2000
) -> dict[str, Any]:
	"""Percentile bootstrap of cluster means, giving each independent cluster equal weight."""
	groups: dict[str, list[float]] = defaultdict(list)
	for cluster, value in rows:
		if not cluster or not math.isfinite(value):
			raise ValueError("cluster IDs and finite values are required")
		groups[cluster].append(value)
	if len(groups) < 2 or replicates < 100:
		raise ValueError("need at least two clusters and 100 replicates")
	means = [sum(v) / len(v) for _, v in sorted(groups.items())]
	rng = random.Random(seed)
	samples = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(replicates))
	return {
		"estimate": sum(means) / len(means),
		"lower": samples[int(0.025 * replicates)],
		"upper": samples[min(replicates - 1, int(0.975 * replicates))],
		"clusters": len(means),
		"warning": "Few independent clusters make bootstrap intervals unstable; "
		"inspect each cluster.",
	}


def compare_conditions(
	conditions: list[dict[str, Any]],
	view: str,
	*,
	capacity_evidence: dict[str, Any] | None = None,
	schedule_sha256: str | None = None,
) -> dict[str, Any]:
	"""Require explicitly declared matching dimensions before presenting a comparison."""
	if len(conditions) < 2:
		raise ValueError("need at least two conditions")
	fields = {
		"attacker_budget": (
			"query_budget",
			"output_token_budget",
			"money_budget",
			"deadline_seconds",
		),
		"legitimate_utility": (
			"utility_metric",
			"utility_target",
			"utility_population",
			"utility_tolerance",
		),
	}
	if view not in fields:
		raise ValueError("unknown comparison view")
	issues = []
	from distillation.teacher_capacity import require_capacity_pass

	try:
		if capacity_evidence is None or not schedule_sha256:
			raise ValueError("capacity evidence and schedule hash are required")
		require_capacity_pass(capacity_evidence, schedule_sha256=schedule_sha256)
	except ValueError:
		issues.append("teacher_capacity")
	for field in fields[view]:
		values = [c.get(field) for c in conditions]
		if any(v is None for v in values) or any(v != values[0] for v in values[1:]):
			issues.append(field)
	if view == "legitimate_utility" and not issues:
		for condition in conditions:
			observed = condition.get("utility_observed")
			target = condition["utility_target"]
			tolerance = condition["utility_tolerance"]
			if (
				not all(
					isinstance(v, (float, int)) and math.isfinite(v)
					for v in (observed, target, tolerance)
				)
				or tolerance < 0
				or abs(observed - target) > tolerance
			):
				issues.append("utility_observed")
				break
	return {
		"view": view,
		"comparable": not issues,
		"unmatched_or_missing": issues,
		"conditions": conditions,
		"note": "Declared matching requires independent verification against observed "
		"outcomes and schedules.",
	}


def student_plan(families: list[str], seeds: list[int], cost_per_run: float) -> dict[str, Any]:
	if (
		not families
		or not seeds
		or len(set(families)) != len(families)
		or len(set(seeds)) != len(seeds)
		or any(not f.strip() for f in families)
		or not math.isfinite(cost_per_run)
		or cost_per_run < 0
	):
		raise ValueError("provide distinct families and seeds and a nonnegative finite cost")
	runs = [
		{"family": f, "seed": s, "control": control, "checkpoint": checkpoint}
		for f in families
		for s in seeds
		for control, checkpoints in (
			("base", (None,)),
			("gold", CHECKPOINTS),
			("undefended_teacher", CHECKPOINTS),
			("defended_teacher", CHECKPOINTS),
		)
		for checkpoint in checkpoints
	]
	return {
		"version": "student-plan-v1",
		"checkpoints": list(CHECKPOINTS),
		"runs": runs,
		"cost_per_run": cost_per_run,
		"estimated_cost": len(runs) * cost_per_run,
		"cost_note": "User-supplied per-run estimate; excludes acquisition and evaluation "
		"unless included in that estimate.",
		"executes_training": False,
	}
