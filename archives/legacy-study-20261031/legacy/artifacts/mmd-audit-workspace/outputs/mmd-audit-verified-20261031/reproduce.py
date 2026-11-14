"""Step 1: rebuild the frozen calibration from cached embeddings and replay logged campaigns.

A match shows that serving and calibration compute what the artifact says, so any miss is
method behaviour rather than an implementation fault.
"""
import json

import numpy as np

from common import READINGS, CachedEncoder, benign, benign_requests, schedule
from harness.calibration import HarnessConfig, calibrate, load_calibration
from harness.policy import MMDPolicy

frozen, scorer = load_calibration(READINGS / "calibration")
print("stored null sorted:", frozen.calibration_scores == sorted(frozen.calibration_scores))
encoder = CachedEncoder()
groups = {r: benign_requests(benign(r)) for r in ("fitting", "score_calibration", "policy_calibration")}
config = HarnessConfig.model_validate(frozen.config.model_dump())
rebuilt, references = calibrate(config=config, encoder=encoder, **groups)
stored_refs = np.load(READINGS / "calibration/references.npy")
null_a, null_b = np.array(frozen.calibration_scores), np.array(rebuilt.calibration_scores)
report = {
	"sha256_match": {k: frozen.sha256[k] == rebuilt.sha256[k] for k in groups},
	"sigma": [frozen.sigma, rebuilt.sigma],
	"references_max_abs_diff": float(np.abs(stored_refs - references).max()),
	"null_count": [len(null_a), len(null_b)],
	"null_max_abs_diff": float(np.abs(null_a - null_b).max()),
	"thresholds": {
		k: [frozen.thresholds[k].thresholds, rebuilt.thresholds[k].thresholds] for k in frozen.thresholds
	},
	"quota": [frozen.baselines["quota"].thresholds, rebuilt.baselines["quota"].thresholds],
	"counts_equal": frozen.counts == {**rebuilt.counts, **{k: v for k, v in frozen.counts.items() if k.endswith("truncated")}},
}


def replay(attack: str, accumulator: str, limit: int | None = None) -> list[dict]:
	policy = MMDPolicy(frozen, scorer, accumulator=accumulator)
	out = []
	for r in schedule(attack)[:limit]:
		emb = encoder.encode([r["detector"]])[0]
		d = policy.admit(r["account"], r["served"], r["time"], embedding=emb)
		out.append({"account": d.account, "attempt": d.attempt, "level": d.level, "p": d.p_value, "stat": d.statistic, "domain": r["domain"]})
	return out


def logged(path) -> list[dict]:
	return [json.loads(line) for line in open(path) if '"window_completed": true' in line]


for attack, arm, acc, limit in (
	("A1", "full", "paper", None),
	("A1", "instantaneous", "instantaneous", None),
	("A2", "full.interrupted-2026-10-30", "paper", 1536),
):
	ours = [d for d in replay(attack, acc, limit) if d["p"] is not None]
	theirs = [d for d in logged(READINGS / f"harness/{attack}/s42-b6000/{arm}/decisions.jsonl") if d["account"].startswith(attack)]
	pairs = list(zip([d["p"] for d in ours], [d["p_value"] for d in theirs]))
	report[f"{attack}/{arm}"] = {
		"windows": [len(ours), len(theirs)],
		"p_exact_matches": sum(a == b for a, b in pairs),
		"p_max_abs_diff": max(abs(a - b) for a, b in pairs),
		"max_level": max(d["level"] for d in ours),
	}
print(json.dumps(report, indent="\t"))
json.dump(report, open("results/reproduce.json", "w"), indent="\t")
