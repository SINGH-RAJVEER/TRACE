"""
Windows (of 100 requests) the frozen paper accumulator needs to reach each level at a constant p.
"""

import json
import math

from common import READINGS
from harness.calibration import load_calibration

frozen, _ = load_calibration(READINGS / "calibration")
thresholds = frozen.thresholds["paper"].thresholds
floor = 1 / (len(frozen.calibration_scores) + 1)
rows = {}
for p in (floor, 0.005, 0.01, 0.02, 0.05, 0.1, 0.142, 0.2, 0.3, 0.368, 0.5, 0.9):
	r, reached = 0.0, [None] * 4
	for n in range(1, 61):
		r = max(0.0, 0.9 * r - math.log(p) - 1)
		for level, t in enumerate(thresholds):
			if reached[level] is None and r > t:
				reached[level] = n
	rows[f"{p:.4f}"] = {
		"increment": round(-math.log(p) - 1, 3),
		"steady_state": round(max(0, 10 * (-math.log(p) - 1)), 2),
		"windows_to_L1_L4": reached,
	}
print(json.dumps({"thresholds": thresholds, "p_floor": floor, "constant_p": rows}, indent="\t"))
with open("results/arithmetic.json", "w") as sink:
	json.dump(
		{"thresholds": thresholds, "p_floor": floor, "constant_p": rows},
		sink,
		indent="\t",
	)
