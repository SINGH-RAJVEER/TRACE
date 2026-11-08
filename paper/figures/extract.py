"""Collect the v7 numbers the paper reports into figures/v7.json.

Run from the repository root: python paper/figures/extract.py
Reads only readings/v7; writes paper/figures/v7.json.
"""

import collections
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
V7 = ROOT / "readings" / "v7"
OUT = pathlib.Path(__file__).resolve().parent / "v7.json"
ATTACKS = ["A1", "A2", "A3", "A5", "A6", "A7"]
ARMS = ["full", "instantaneous", "quota"]


def jsonl(path):
	with open(path) as handle:
		for line in handle:
			line = line.strip()
			if line:
				yield json.loads(line)


def gate_report(attack, arm):
	"""Return the detectability JSON printed in a refused arm's log, or the saved file."""
	saved = V7 / "campaigns" / attack / arm / "campaign" / "detectability.json"
	if saved.exists():
		return json.loads(saved.read_text())["arms"]
	text = (V7 / "logs" / f"campaign-{attack}-{arm}.log").read_text()
	marker = text.rfind("detectability gate:")
	start = text.rfind('{\n\t"paper"', 0, marker)
	return json.loads(text[start:text.rfind("}", start, marker) + 1])


def campaign(attack, arm):
	base = V7 / "campaigns" / attack / arm
	outcome = (base / "outcome").read_text().strip()
	result = {"outcome": outcome}
	gate = gate_report(attack, arm) if arm != "quota" else None
	if gate:
		result["gate"] = {
			name: {
				"attacker_peak": value["attacker_peak"],
				"attacker_peak_by_account": value["attacker_peak_by_account"],
				"benign_peaks": value["benign"]["scored_peak_levels"],
			}
			for name, value in gate.items()
		}
	if outcome.startswith("gate refused"):
		return result
	report = json.loads((base / "evaluation-report.json").read_text())
	for key in ("attempts", "received", "blocked", "challenge", "failed", "released_tokens"):
		result[key] = report[key]
	result["first_L4_attempt"] = report["levels"].get("L4", {}).get("J")
	# Released responses and tokens by domain, in schedule order.
	by_domain = collections.defaultdict(lambda: [0, 0])
	curve = []
	total = 0
	for row in jsonl(base / "campaign" / "ledger.jsonl"):
		tokens = (row.get("usage") or {}).get("output_tokens") or 0
		by_domain[row["task"]][0] += 1
		by_domain[row["task"]][1] += tokens
		total += tokens
		curve.append(total)
	result["by_domain"] = {task: {"responses": n, "tokens": t} for task, (n, t) in by_domain.items()}
	result["ledger_tokens"] = total
	# Attempt index of every release, from the decision log.
	released = []
	attempt = 0
	result["first_L4_decision"] = None
	for row in jsonl(base / "decisions.jsonl"):
		if row.get("event") == "decision":
			attempt += 1
			if row.get("level") == 4 and result["first_L4_decision"] is None:
				result["first_L4_decision"] = attempt
		elif row.get("event") == "release" and row.get("status") == "released":
			released.append([attempt, row.get("output_tokens") or 0])
	result["release_curve"] = released
	return result


def windows(attack, arm):
	"""Window ranks the live MMD arm logged, per account."""
	path = V7 / "campaigns" / attack / arm / "decisions.jsonl"
	ranks = collections.defaultdict(list)
	for row in jsonl(path):
		if row.get("event") == "decision" and row.get("window_completed") and row.get("p_value") is not None:
			ranks[row["account"]].append(row["p_value"])
	return ranks


def utility(arm):
	data = json.loads((V7 / "utility" / arm / "utility.json").read_text())
	statuses = collections.Counter()
	for key, value in data.items():
		if isinstance(value, list):
			for row in value:
				if isinstance(row, dict) and "status" in row:
					statuses[row["status"]] += 1
	return {
		key: data[key]
		for key in ("submitted", "served", "served_only_quality", "completion_rate", "latency_p95")
	} | {"statuses": dict(statuses)}


def main():
	calibration = json.loads((V7 / "calibration" / "calibration.json").read_text())
	adequacy = json.loads((V7 / "adequacy.json").read_text())
	data = {
		"thresholds": {
			name: calibration["thresholds"][name]["thresholds"] for name in ("paper", "instantaneous")
		}
		| {"quota": calibration["baselines"]["quota"]["thresholds"][0]},
		"upper95": calibration["thresholds"]["paper"]["upper95"],
		"null_scores": len(calibration["calibration_scores"]),
		"adequacy": {
			key: adequacy.get(key)
			for key in ("status", "nonoverlapping_score_windows", "detectability_warnings", "empirical_p_value_floor")
		},
		"campaigns": {a: {arm: campaign(a, arm) for arm in ARMS} for a in ATTACKS},
		"windows": {
			f"{a}/{arm}": windows(a, arm) for a in ("A2", "A5") for arm in ("full", "instantaneous")
		},
		"utility": {arm: utility(arm) for arm in ARMS},
	}
	OUT.write_text(json.dumps(data, indent="\t") + "\n")


if __name__ == "__main__":
	main()
