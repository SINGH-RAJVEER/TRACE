"""
Compare domain-conditional A7 scores with benign specialist account windows.
"""

import collections
import json
import random

import numpy as np
from common import CachedEncoder, benign, schedule
from harness.mmd import WindowScorer, median_bandwidth

encoder = CachedEncoder()
rng = random.Random(11)


def emb(rows):
	return encoder.encode([r["detector"] for r in rows])


fitting, test, a7, a1 = benign("fitting"), benign("benign_test"), schedule("A7"), schedule("A1")
out = {}
for domain in ("ag_news", "squad", "gsm8k", "coding"):
	fit_d = rng.sample(
		[r for r in fitting if r["domain"] == domain], sum(r["domain"] == domain for r in fitting)
	)
	count = min(20, (len(fit_d) - 200) // 100)
	scorer = WindowScorer(
		np.stack([emb(fit_d[i * 100 : (i + 1) * 100]) for i in range(count)]),
		median_bandwidth(emb(fit_d[count * 100 :][:2000])),
	)

	def score(w, scorer=scorer):
		return scorer.score(emb(w), [r["served"] for r in w])

	accounts = collections.defaultdict(list)
	for r in test:
		if r["domain_profile"] == "specialist" and r["domain"] == domain:
			accounts[r["account"]].append(r)
	spec = [score(rs[i : i + 100]) for rs in accounts.values() for i in range(0, len(rs) - 99, 100)]
	a7s = [
		score(w)
		for w in (lambda rs: [rs[i : i + 100] for i in range(0, len(rs) - 99, 100)])(
			[r for r in a7 if r["domain"] == domain]
		)
	]
	a1s = [
		score(w)
		for w in (lambda rs: [rs[i : i + 100] for i in range(0, len(rs) - 99, 100)])(
			[r for r in a1 if r["domain"] == domain]
		)
	]

	def q(xs):
		return [round(float(x), 4) for x in np.percentile(xs, [5, 50, 95])] if xs else None

	out[domain] = {
		"specialist_windows": len(spec),
		"specialist_q05_50_95": q(spec),
		"A7_q05_50_95": q(a7s),
		"A1_q05_50_95": q(a1s),
		"share_specialist_windows_above_A7_median": round(
			float(np.mean(np.array(spec) > np.median(a7s))), 3
		)
		if spec
		else None,
	}
	print(domain, out[domain], flush=True)
with open("results/probe-specialists.json", "w") as sink:
	json.dump(out, sink, indent="\t")
