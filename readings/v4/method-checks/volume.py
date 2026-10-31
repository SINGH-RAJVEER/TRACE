"""Step 3: what count and token quotas release per account and campaign, at a fitted benign cost.

Attack token counts are exact teacher output tokens from the v2 undefended ledgers (A1, A2, A5
and A6 share prompts). Benign prompts never reached the bare teacher, so benign token use is
estimated by each domain's mean attacker output tokens; this is labelled an estimate.
"""

import collections
import json

from common import ATTACKS, ROOT, benign, schedule
from harness.calibration import binomial_upper, nested_thresholds

LEDGERS = ROOT / "readings/v4/evidence/v2"
tokens: dict[str, int] = {}
for attack in ("A1", "A7"):
	for line in (LEDGERS / attack / "s42-b6000/undefended/ledger.jsonl").open():
		r = json.loads(line)
		tokens[r["example_id"]] = r["usage"]["output_tokens"]
by_domain = collections.defaultdict(list)
for r in schedule("A1"):
	by_domain[r["domain"]].append(tokens[r["prompt"]["example_id"]])
mean_tokens = {d: sum(v) / len(v) for d, v in by_domain.items()}


def cost(r: dict) -> float:
	return tokens.get(r["prompt"]["example_id"], mean_tokens[r["domain"]])


def accounts(rows):
	by = collections.defaultdict(list)
	for r in rows:
		by[r["account"]].append(r)
	return by


def fit(rows, measure, target=0.01):
	"""Fit as calibration fits the request quota: scored horizons (>= 100 requests), 1% target."""
	values = [measure(rs) for rs in accounts(rows).values() if len(rs) >= 100]
	return nested_thresholds(values, [target]).thresholds[0]


def released(rs, limit, measure_one):
	"""Requests and tokens served before the cumulative measure first exceeds the limit."""
	total, n, tok = 0.0, 0, 0.0
	for r in rs:
		total += measure_one(r)
		if total > limit:
			break
		n, tok = n + 1, tok + cost(r)
	return n, tok


if __name__ == "__main__":
	policy = benign("policy_calibration")
	test = benign("benign_test")
	limits = {
		"request_quota": (fit(policy, len), lambda r: 1.0),
		"token_quota": (fit(policy, lambda rs: sum(cost(r) for r in rs)), cost),
	}
	out = {
		"mean_output_tokens_by_domain": mean_tokens,
		"limits": {k: v[0] for k, v in limits.items()},
	}
	for name, (limit, one) in limits.items():
		blocked = collections.Counter()
		totals = collections.Counter()
		lost = 0
		for rs in accounts(test).values():
			key = rs[0]["domain_profile"]
			totals[key] += 1
			n, _ = released(rs, limit, one)
			if n < len(rs):
				blocked[key] += 1
				lost += len(rs) - n
		scored = sum(len(rs) >= 100 for rs in accounts(test).values())
		b = sum(blocked.values())
		out[name] = {
			"benign_test_blocked_accounts": dict(blocked),
			"benign_test_accounts": dict(totals),
			"benign_test_blocked_of_scored": f"{b}/{scored}",
			"upper95": binomial_upper(b, scored),
			"benign_requests_refused": lost,
			"attacks": {},
		}
		for attack in ATTACKS:
			acc = accounts(schedule(attack))
			per = {a: released(rs, limit, one) for a, rs in acc.items()}
			out[name]["attacks"][attack] = {
				"accounts": len(acc),
				"requests_per_account": sorted({len(rs) for rs in acc.values()}),
				"released_requests": sum(n for n, _ in per.values()),
				"released_tokens": round(sum(t for _, t in per.values())),
				"campaign_tokens": round(sum(cost(r) for rs in acc.values() for r in rs)),
				"accounts_blocked": sum(n < len(acc[a]) for a, (n, _) in per.items()),
				"released_by_domain": dict(
					collections.Counter(
						r["domain"] for a, rs in acc.items() for r in rs[: per[a][0]]
					)
				),
			}
	print(json.dumps(out, indent="\t"))
	with open("results/volume.json", "w") as sink:
		json.dump(out, sink, indent="\t")
