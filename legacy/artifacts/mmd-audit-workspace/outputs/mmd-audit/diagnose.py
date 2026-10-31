"""Step 2: where attacker windows fall in the frozen benign null, against benign windows by profile.

Every account (benign or attacker) is cut into consecutive non-overlapping 100-request windows,
as the policy cuts them, and scored with the frozen references and sigma. Upper-tail p is the
deployed rank; lower-tail p asks whether a window is unusually close to the pooled reference.
"""
import collections
import json
import random
import sys

import numpy as np

from common import ATTACKS, READINGS, CachedEncoder, benign, schedule
from harness.calibration import load_calibration
from harness.mmd import WindowScorer, distinct

variant = sys.argv[1] if len(sys.argv) > 1 else "detector"
field = "detector" if variant == "detector" else "detector_noprefix"
frozen, scorer = load_calibration(READINGS / "calibration")
encoder = CachedEncoder(variant)
if variant != "detector":
	# Rebuild references and sigma from the same fitting rows in the ablated representation.
	from common import benign_requests
	from harness.calibration import HarnessConfig, calibrate

	config = HarnessConfig.model_validate(frozen.config.model_dump())

	class Ablated(CachedEncoder):
		def encode(self, texts):
			return self.lookup([json_to_noprefix[t] for t in texts])

	json_to_noprefix = {}
	for role in ("fitting", "score_calibration", "policy_calibration"):
		for r in benign(role):
			json_to_noprefix[r["detector"]] = r["detector_noprefix"]
	frozen, references = calibrate(
		config=config, encoder=Ablated(variant),
		**{r: benign_requests(benign(r)) for r in ("fitting", "score_calibration", "policy_calibration")},
	)
	scorer = WindowScorer(references, frozen.sigma, frozen.config.scales)
null = np.array(frozen.calibration_scores)


def upper(s: float) -> float:
	return frozen.p_value(s)


def lower(s: float) -> float:
	return (1 + int(np.searchsorted(null, s, side="right"))) / (len(null) + 1)


def windows(rows: list[dict]) -> list[dict]:
	by = collections.defaultdict(list)
	for r in rows:
		by[r["account"]].append(r)
	out = []
	for account, rs in by.items():
		for w in range(len(rs) // 100):
			chunk = rs[w * 100 : (w + 1) * 100]
			emb = encoder.encode([r[field] for r in chunk])
			score = scorer.score(emb, [r["served"] for r in chunk])
			domains = collections.Counter(r["domain"] for r in chunk)
			out.append({
				"account": account, "window": w, "score": score, "p_upper": upper(score), "p_lower": lower(score),
				"client_type": chunk[0].get("client_type"), "profile": chunk[0].get("domain_profile"),
				"domains": len(domains), "top_share": max(domains.values()) / 100,
				"distinct": len(distinct([r["served"] for r in chunk])),
			})
	return out


def summary(ws: list[dict]) -> dict:
	s = np.array([w["score"] for w in ws])
	pu = np.array([w["p_upper"] for w in ws])
	return {
		"windows": len(ws), "accounts": len({w["account"] for w in ws}),
		"score_q05_50_95": [round(float(x), 4) for x in np.percentile(s, [5, 50, 95])],
		"p_upper_min_med_max": [round(float(x), 3) for x in (pu.min(), np.median(pu), pu.max())],
		"share_p_upper_le_0.05": round(float((pu <= 0.05).mean()), 3),
		"share_p_lower_le_0.05": round(float((np.array([w["p_lower"] for w in ws]) <= 0.05).mean()), 3),
	}


result = {"variant": variant, "sigma": frozen.sigma, "null_q05_50_95": [round(float(x), 4) for x in np.percentile(null, [5, 50, 95])]}
test = windows(benign("benign_test"))
groups = collections.defaultdict(list)
for w in test:
	groups[f"benign_test/{w['profile']}/{w['client_type']}"].append(w)
	groups[f"benign_test/{w['profile']}"].append(w)
	groups[f"benign_test/{w['profile']}/domains={w['domains']}"].append(w)
for attack in ATTACKS:
	for w in windows(schedule(attack)):
		groups[attack].append(w)
		if attack == "A2":
			groups[f"A2/window{w['window']:02d}"].append(w)
# Synthetic probes from held-out benign prompts: a uniform shuffle of the whole pool (a
# "perfect mixture" account) and random, not topic-ordered, single-domain windows.
pool = benign("benign_test")
rng = random.Random(7)
shuffled = rng.sample(pool, len(pool))
probe = [dict(r, account=f"mixture-{i // 1000}") for i, r in enumerate(shuffled[:10000])]
groups["probe/uniform-mixture"] = windows(probe)
for domain in ("ag_news", "squad", "gsm8k", "coding"):
	rows = [r for r in shuffled if r["domain"] == domain][:1000]
	groups[f"probe/random-{domain}"] = windows([dict(r, account=f"rand-{domain}") for r in rows])
result["groups"] = {k: summary(v) for k, v in sorted(groups.items()) if v}
# Does domain concentration explain the score? Spearman-like rank correlation on benign test.
from scipy.stats import spearmanr
result["benign_test_spearman_score_vs_top_domain_share"] = float(spearmanr([w["score"] for w in test], [w["top_share"] for w in test]).statistic)
result["benign_test_spearman_score_vs_domain_count"] = float(spearmanr([w["score"] for w in test], [w["domains"] for w in test]).statistic)
json.dump(result, open(f"results/diagnose-{variant}.json", "w"), indent="\t")
for k, v in result["groups"].items():
	print(f"{k:45s} {v}")
print({k: v for k, v in result.items() if k != "groups"})
