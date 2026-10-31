"""Step 4: compare defenses at fitted benign cost on held-out benign traffic and frozen attacks.

Window scores are computed once per account (consecutive non-overlapping 100-request windows,
as MMDPolicy cuts them). Every threshold is fitted on policy_calibration scored horizons at
the frozen targets and evaluated once on benign_test and the five seed-42 attack schedules.
L4 is the only level that withholds output (L2 spacing and the 20-bit proof cost little at
this budget), so "released" counts responses served before the first refusal.
"""
import collections
import json
import math

import numpy as np

from common import ATTACKS, READINGS, CachedEncoder, benign, schedule
from harness.accumulators import AccumulatorSpec
from harness.calibration import binomial_upper, load_calibration, nested_thresholds
from volume import cost, accounts

frozen, scorer = load_calibration(READINGS / "calibration")
encoder = CachedEncoder()
null = np.array(frozen.calibration_scores)
TARGETS = frozen.config.level_targets


def p_upper(s):
	return frozen.p_value(s)


def p_lower(s):
	return (1 + int(np.searchsorted(null, s, side="right"))) / (len(null) + 1)


def p_two(s):
	return min(1.0, 2 * min(p_upper(s), p_lower(s)))


def window_scores(rs):
	out = []
	for w in range(min(len(rs), frozen.config.horizon_attempts) // 100):
		chunk = rs[w * 100 : (w + 1) * 100]
		out.append(scorer.score(encoder.encode([r["detector"] for r in chunk]), [r["served"] for r in chunk]))
	return out


def path(scores, pfun, spec):
	state, states = 0.0, []
	for s in scores:
		state = spec.update(state, pfun(s))
		states.append(state)
	return states


PAPER = AccumulatorSpec(kind="decayed_cusum", decay=0.9, drift=1.0)
INST = AccumulatorSpec(kind="instantaneous")
VARIANTS = {
	"full (frozen)": (p_upper, PAPER, frozen.thresholds["paper"].thresholds),
	"instantaneous (frozen)": (p_upper, INST, frozen.thresholds["instantaneous"].thresholds),
	"full, lower tail (refit)": (p_lower, PAPER, None),
	"full, two-sided (refit)": (p_two, PAPER, None),
}
policy_scores = {a: window_scores(rs) for a, rs in accounts(benign("policy_calibration")).items()}
for name, (pfun, spec, thresholds) in list(VARIANTS.items()):
	if thresholds is None:
		maxima = [max(path(s, pfun, spec)) for s in policy_scores.values() if s]
		VARIANTS[name] = (pfun, spec, nested_thresholds(maxima, TARGETS).thresholds)
# Sanity: refitting the frozen variants on the same data must give the frozen thresholds.
for name in ("full (frozen)", "instantaneous (frozen)"):
	pfun, spec, thresholds = VARIANTS[name]
	refit = nested_thresholds([max(path(s, pfun, spec)) for s in policy_scores.values() if s], TARGETS).thresholds
	assert np.allclose(refit, thresholds), (name, refit, thresholds)
limits = json.load(open("results/volume.json"))["limits"]


def mmd_outcome(rs, scores, variant):
	pfun, spec, thresholds = VARIANTS[variant]
	states = path(scores, pfun, spec)
	levels = [sum(x > t for t in thresholds) for x in states]
	peak = max(levels, default=0)
	first = {lv: next((i for i, x in enumerate(levels) if x >= lv), None) for lv in (1, 4)}
	# The request completing the triggering window is itself refused.
	served = len(rs) if first[4] is None else 100 * (first[4] + 1) - 1
	return peak, served, None if first[1] is None else 100 * (first[1] + 1)


def quota_outcome(rs, limit, one):
	total = 0.0
	for i, r in enumerate(rs):
		total += one(r)
		if total > limit:
			return 4, i, i + 1
	return 0, len(rs), None


def outcome(rs, scores, defense):
	if defense == "request quota":
		return quota_outcome(rs, limits["request_quota"], lambda r: 1.0)
	if defense == "token quota (est.)":
		return quota_outcome(rs, limits["token_quota"], cost)
	if defense == "full + request quota":
		a, b = mmd_outcome(rs, scores, "full (frozen)"), quota_outcome(rs, limits["request_quota"], lambda r: 1.0)
		return max(a[0], b[0]), min(a[1], b[1]), min([x for x in (a[2], b[2]) if x is not None], default=None)
	return mmd_outcome(rs, scores, defense)


DEFENSES = [*VARIANTS, "request quota", "token quota (est.)", "full + request quota"]
if __name__ == "__main__":
	test = accounts(benign("benign_test"))
	test_scores = {a: window_scores(rs) for a, rs in test.items()}
	attack_rows = {k: accounts(schedule(k)) for k in ATTACKS}
	attack_scores = {k: {a: window_scores(rs) for a, rs in acc.items()} for k, acc in attack_rows.items()}
	result = {"thresholds": {k: v[2] for k, v in VARIANTS.items()}, "limits": limits, "defenses": {}}
	for defense in DEFENSES:
		ben = collections.defaultdict(lambda: collections.Counter())
		for a, rs in test.items():
			peak, served, _ = outcome(rs, test_scores[a], defense)
			if len(rs) < 100:
				continue
			for group in ("all scored", rs[0]["domain_profile"], f"{rs[0]['domain_profile']}/{rs[0]['client_type']}"):
				ben[group]["n"] += 1
				ben[group]["L1+"] += peak >= 1
				ben[group]["L4"] += peak >= 4
				ben[group]["refused_requests"] += len(rs) - served
		entry = {"benign_test": {}, "attacks": {}}
		for group, c in sorted(ben.items()):
			entry["benign_test"][group] = {
				"n": c["n"], "L1+": c["L1+"], "L4": c["L4"],
				"L4_rate": round(c["L4"] / c["n"], 3), "L4_upper95": round(binomial_upper(c["L4"], c["n"]), 3),
				"refused_requests": c["refused_requests"],
			}
		for k, acc in attack_rows.items():
			per = {a: outcome(rs, attack_scores[k][a], defense) for a, rs in acc.items()}
			entry["attacks"][k] = {
				"accounts": len(acc),
				"accounts_reaching_L4": sum(p[0] >= 4 for p in per.values()),
				"first_alert_request": min([p[2] for p in per.values() if p[2] is not None], default=None),
				"released_requests": sum(p[1] for p in per.values()),
				"released_tokens": round(sum(cost(r) for a, rs in acc.items() for r in rs[: per[a][1]])),
				"released_ag_news": sum(r["domain"] == "ag_news" for a, rs in acc.items() for r in rs[: per[a][1]]),
			}
		result["defenses"][defense] = entry
	result["attack_window_p_upper"] = {
		k: {a: [round(p_upper(s), 3) for s in v] for a, v in sc.items()} for k, sc in attack_scores.items()
	}
	json.dump(result, open("results/compare.json", "w"), indent="\t")
	for defense, e in result["defenses"].items():
		b = e["benign_test"]
		print(f"\n## {defense}  thresholds={result['thresholds'].get(defense)}")
		print("  benign scored L4:", b["all scored"], "| specialist:", b.get("specialist"), "| generalist:", b.get("generalist"))
		for k, v in e["attacks"].items():
			print(f"  {k}: {v}")
