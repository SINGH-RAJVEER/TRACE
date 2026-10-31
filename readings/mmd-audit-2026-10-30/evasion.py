"""Step 5: can an attacker who controls only query order move its windows into the benign bulk?

Same 6,000 A1 prompts, same single account, no cost to the attacker: prompts are grouped by
domain and ordered by a k-means topic cluster of their own embeddings (the attacker may use any
sentence encoder; using BGE here is the strongest case for the attacker, a stated assumption).
Also: interleaving two domains per window, which mimics a two-domain generalist.
"""

import collections
import json

import numpy as np
from common import CachedEncoder, schedule
from compare import VARIANTS, p_lower, p_upper, path, window_scores
from sklearn.cluster import KMeans

encoder = CachedEncoder()
rows = schedule("A1")
by_domain = collections.defaultdict(list)
for r in rows:
	by_domain[r["domain"]].append(r)


def topic_sorted(k: int) -> list[dict]:
	out = []
	for domain in sorted(by_domain):
		rs = by_domain[domain]
		emb = encoder.encode([r["detector"] for r in rs])
		labels = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(emb)
		out += [r for _, r in sorted(zip(labels, rs, strict=True), key=lambda x: x[0])]
	return out


def paired_domains() -> list[dict]:
	"""Alternate two domains inside each window, cycling through domain pairs."""
	doms = sorted(by_domain)
	queues = {d: list(by_domain[d]) for d in doms}
	pairs = [(doms[i], doms[j]) for i in range(4) for j in range(i + 1, 4)]
	out, k = [], 0
	while any(queues.values()):
		a, b = pairs[k % len(pairs)]
		for i in range(100):
			d = (a, b)[i % 2]
			if not queues[d]:
				d = next((x for x in doms if queues[x]), None)
				if d is None:
					break
			out.append(queues[d].pop())
		k += 1
	return out


result = {}
for name, ordered in {
	"A1 as logged (interleaved)": rows,
	"A1 domain-blocked, topic-sorted k=8": topic_sorted(8),
	"A1 domain-blocked, topic-sorted k=30": topic_sorted(30),
	"A1 two domains per window": paired_domains(),
}.items():
	scores = window_scores(ordered)
	pu = [p_upper(s) for s in scores]
	pl = [p_lower(s) for s in scores]
	entry = {
		"windows": len(scores),
		"p_upper_min_med_max": [
			round(min(pu), 3),
			round(float(np.median(pu)), 3),
			round(max(pu), 3),
		],
		"p_lower_min_med_max": [
			round(min(pl), 3),
			round(float(np.median(pl)), 3),
			round(max(pl), 3),
		],
	}
	for variant in ("full (frozen)", "full, two-sided (refit)", "full, lower tail (refit)"):
		pfun, spec, thresholds = VARIANTS[variant]
		levels = [sum(x > t for t in thresholds) for x in path(scores, pfun, spec)]
		first4 = next((i for i, lv in enumerate(levels) if lv >= 4), None)
		entry[variant] = {
			"peak_level": max(levels),
			"released_before_L4": 6000 if first4 is None else 100 * (first4 + 1) - 1,
		}
	result[name] = entry
	print(name, json.dumps(entry))
with open("results/evasion.json", "w") as sink:
	json.dump(result, sink, indent="\t")
