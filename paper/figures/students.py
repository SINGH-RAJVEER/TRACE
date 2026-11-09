"""Collect the GSM8K student scores the paper reports into figures/students.json.

Run from the repository root:
python paper/figures/students.py [ARTIFACTS]

ARTIFACTS defaults to ../TRACE-main/artifacts/students/v7, where the main-branch
workspace trained the students. The output keeps every per-item score, so the paper's
tests can be recomputed from this file alone.
"""

import json
import math
import os
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT = ROOT.parent / "TRACE-main" / "artifacts" / "students" / "v7"
OUT = HERE / "students.json"
CELLS = {
	"base": "base-dev.json",
	"teacher": "teacher-dev.json",
	"A5/full end": "plan-A5-full/scores/end-s42.json",
	"unprotected end": "plan-A6-unprotected/scores/end-s42.json",
}
PLANS = {"A5/full": "plan-A5-full/plan.json", "unprotected": "plan-A6-unprotected/plan.json"}
PAIRS = [
	("unprotected end", "A5/full end"),
	("A5/full end", "base"),
	("unprotected end", "base"),
	("teacher", "A5/full end"),
	("teacher", "unprotected end"),
]
Z = 1.959964


def wilson(k, n):
	p = k / n
	d = 1 + Z * Z / n
	centre = (p + Z * Z / (2 * n)) / d
	half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
	return [centre - half, centre + half]


def mcnemar(a_only, b_only):
	n = a_only + b_only
	k = min(a_only, b_only)
	return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2**n)


def bootstrap(diffs, draws=10000, seed=42):
	rng = random.Random(seed)
	n = len(diffs)
	means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(draws))
	return [means[int(0.025 * draws)], means[int(0.975 * draws) - 1]]


def main():
	source = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
	scores = {name: json.loads((source / path).read_text())["scores"] for name, path in CELLS.items()}
	ids = sorted(scores["base"])
	if any(sorted(s) != ids for s in scores.values()):
		raise ValueError("score files cover different items")
	n = len(ids)
	data = {"source": os.path.relpath(source.resolve(), ROOT.parent), "items": n, "target_accuracy": None, "cells": {}, "pairs": {}, "plans": {}}
	for name, plan_path in PLANS.items():
		plan = json.loads((source / plan_path).read_text())
		data["target_accuracy"] = plan["target_accuracy"]
		data["plans"][name] = {
			"recipe": plan["recipe"],
			"seeds": plan["seeds"],
			"checkpoints": [
				{key: c.get(key) for key in ("id", "acquired_count", "trainable_count", "filtered")}
				for c in plan["checkpoints"]
			],
		}
	for name, cell in scores.items():
		k = sum(cell[i] for i in ids)
		data["cells"][name] = {
			"correct": k,
			"accuracy": k / n,
			"wilson95": wilson(k, n),
			"scores": "".join(str(cell[i]) for i in ids),
		}
	for a, b in PAIRS:
		diffs = [scores[a][i] - scores[b][i] for i in ids]
		a_only = diffs.count(1)
		b_only = diffs.count(-1)
		data["pairs"][f"{a} - {b}"] = {
			"difference": sum(diffs) / n,
			"a_only": a_only,
			"b_only": b_only,
			"discordant_share": (a_only + b_only) / n,
			"mcnemar_exact_p": mcnemar(a_only, b_only),
			"bootstrap95": bootstrap(diffs),
		}
	gap = data["cells"]["teacher"]["accuracy"] - data["cells"]["base"]["accuracy"]
	data["gap_closed"] = {
		name: (data["cells"][name]["accuracy"] - data["cells"]["base"]["accuracy"]) / gap
		for name in ("A5/full end", "unprotected end")
	}
	OUT.write_text(json.dumps(data, indent="\t") + "\n")


if __name__ == "__main__":
	main()
