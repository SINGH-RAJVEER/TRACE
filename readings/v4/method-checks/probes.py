"""Step 6: probe domain-conditional references and larger windows for independent-source A7.

(a) Domain-conditional: per domain, references and bandwidth from that domain's fitting rows;
    null = shuffled same-domain score_calibration windows; tests = A1 and A7 same-domain
    sub-streams in their logged order, and shuffled same-domain benign_test windows (control).
(b) Window size (Liu et al. style): one pooled benign set (fitting + score_calibration rows)
    is shuffled and split into reference windows and i.i.d. null windows, so reference and
    null share one mixture. Tests: A1 and A7 in logged order, and a domain-balanced shuffle of
    benign_test + policy_calibration prompts with A1's 25/25/25/25 mix (the benign control).
Null windows in (b) are resampled from the held-back pool and are dependent; p-values are
empirical upper-tail ranks and are diagnostics, not error-controlled tests.
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


def scorer_from(rows, size, count):
	order = rng.sample(rows, len(rows))
	refs = np.stack([emb(order[i * size : (i + 1) * size]) for i in range(count)])
	sigma = median_bandwidth(emb(order[count * size : count * size + 2000] or order[:2000]))
	return WindowScorer(refs, sigma), order[count * size :]


def chunks(rows, size):
	return [rows[i : i + size] for i in range(0, len(rows) - size + 1, size)]


def ranks(scorer, null_scores, windows):
	null = np.sort(null_scores)
	out = []
	for w in windows:
		s = scorer.score(emb(w), [r["served"] for r in w])
		out.append((1 + len(null) - int(np.searchsorted(null, s, side="left"))) / (len(null) + 1))
	return [round(p, 3) for p in out]


result = {"domain_conditional": {}, "window_size": {}}
fitting, score_cal = benign("fitting"), benign("score_calibration")
test_pool = benign("benign_test")  # policy_calibration AG News is Business-only (see role-shift)
a1, a7 = schedule("A1"), schedule("A7")
for domain in ("ag_news", "squad", "gsm8k", "coding"):
	fit_d = [r for r in fitting if r["domain"] == domain]
	count = min(20, (len(fit_d) - 200) // 100)
	scorer, _ = scorer_from(fit_d, 100, count)
	null_rows = [r for r in score_cal if r["domain"] == domain]
	null_scores = [
		scorer.score(emb(w), [r["served"] for r in w])
		for _ in range(5)
		for w in chunks(rng.sample(null_rows, len(null_rows)), 100)
	][:400]
	control = rng.sample(
		[r for r in test_pool if r["domain"] == domain],
		min(1000, sum(r["domain"] == domain for r in test_pool)),
	)
	result["domain_conditional"][domain] = {
		"reference_windows": count,
		"null_windows": len(null_scores),
		"A1_p": ranks(scorer, null_scores, chunks([r for r in a1 if r["domain"] == domain], 100)),
		"A7_p": ranks(scorer, null_scores, chunks([r for r in a7 if r["domain"] == domain], 100)),
		"benign_shuffled_control_p": ranks(scorer, null_scores, chunks(control, 100)),
	}
	print(domain, json.dumps(result["domain_conditional"][domain]), flush=True)
by_domain = collections.defaultdict(list)
for r in rng.sample(test_pool, len(test_pool)):
	by_domain[r["domain"]].append(r)
pooled = fitting + score_cal
for size in (100, 500, 1000):
	scorer, rest = scorer_from(pooled, size, 20)
	null_scores = [
		scorer.score(emb(w), [r["served"] for r in w])
		for w in (rng.sample(rest, size) for _ in range(200))
	]
	per = size // 4
	balanced = []
	for i in range(min(len(v) for v in by_domain.values()) // per):
		w = [r for d in sorted(by_domain) for r in by_domain[d][i * per : (i + 1) * per]]
		balanced.append(rng.sample(w, len(w)))
	result["window_size"][size] = {
		"null_windows": len(null_scores),
		"A1_p": ranks(scorer, null_scores, chunks(a1, size)),
		"A7_p": ranks(scorer, null_scores, chunks(a7, size)),
		"benign_balanced_control_p": ranks(scorer, null_scores, balanced[:20]),
	}
	print(size, json.dumps(result["window_size"][size]), flush=True)
with open("results/probes.json", "w") as sink:
	json.dump(result, sink, indent="\t")
