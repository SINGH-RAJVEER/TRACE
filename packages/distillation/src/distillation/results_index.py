"""One row per campaign arm and benign utility arm of an exported run, kept in plain TSV.

Rows are keyed by run, kind, attack and arm, so a gate-refused or interrupted arm still has a
row. An empty cell means unknown, never zero. Re-exporting a run replaces its rows. Campaign
counts come from the canonical summary of the event log, so an interrupted arm reports what it
observed before stopping.
"""

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from distillation.campaign_summary import load_campaign_summary

INDEX_VERSION = "results-index-v2"
COLUMNS = (
	"reading",
	"run_id",
	"commit",
	"kind",
	"attack",
	"arm",
	"stratum",
	"policy",
	"accumulator",
	"seed",
	"attempt_budget",
	"deadline_seconds",
	"partition",
	"study",
	"status",
	"reading_label",
	"contract_sha256",
	"schedule_sha256",
	"calibration_sha256",
	"ledger_sha256",
	"policy_revision",
	"attempts",
	"received",
	"blocked",
	"challenge",
	"failed",
	"deadline",
	"pending",
	"released_tokens",
	"released_tokens_unknown",
	"distinct_examples",
	"first_alert_seconds",
	"first_restriction_seconds",
	"first_refusal_seconds",
	"attribution_complete",
	"unattributed_queries",
	*(f"L{level}_{field}" for level in (1, 2, 3, 4) for field in ("seconds", "J", "Q")),
	"submitted",
	"served",
	"completion_rate",
	"served_only_quality",
	"unconditional_utility",
	"latency_p95",
	"student_task",
	"student_n",
	"student_seeds",
	"student_gain_pp",
	"student_gain_lower",
	"student_gain_upper",
	"student_target_reached",
	"training_ledger_sha256",
	"training_runs",
)
KEY = ("run_id", "kind", "attack", "arm", "stratum")
STUDY = {"exploratory": "development", "confirmatory": "confirmatory"}


def _json(path: Path) -> dict[str, Any] | None:
	try:
		value = json.loads(path.read_text())
	except OSError, ValueError:
		return None
	return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str | None:
	return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _env(path: Path) -> dict[str, str]:
	if not path.is_file():
		return {}
	pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
	return {key: value for key, value in pairs if not key.startswith("inherited ")}


def _policy_revision(decisions: Path) -> str | None:
	if not decisions.is_file():
		return None
	with decisions.open() as stream:
		for line in stream:
			revision = json.loads(line).get("policy_revision") if line.strip() else None
			if revision:
				return str(revision)
	return None


# Arms of runs that predate harness.env, by the names the runner and the v3 study used.
ARM_POLICY = {
	"full": ("mmd", "paper"),
	"instantaneous": ("mmd", "instantaneous"),
	"quota": ("quota", None),
	"rate": ("rate", None),
	"allow": ("allow", None),
	"undefended": ("none", None),
}


def _arm_settings(directory: Path, arm: str, calibration: Path | None) -> dict[str, Any]:
	env = _env(directory / "harness.env")
	policy, accumulator = (
		(env.get("HARNESS_POLICY"), env.get("HARNESS_ACCUMULATOR"))
		if env
		else ARM_POLICY.get(arm, (None, None))
	)
	return {
		"policy": policy,
		"accumulator": accumulator if policy == "mmd" else None,
		"calibration_sha256": _sha256(calibration) if calibration and policy != "none" else None,
		"policy_revision": _policy_revision(directory / "decisions.jsonl"),
	}


def campaign_row(
	directory: Path,
	schedule_dir: Path | None,
	calibration: Path | None,
	attack: str,
	arm: str,
	*,
	campaign: Path | None = None,
) -> dict[str, Any]:
	"""One campaign arm: `directory` holds `campaign/` and, when protected, `decisions.jsonl`.

	`campaign` overrides the campaign directory for layouts that keep it elsewhere.
	"""
	# Schedules written before gateway-schedules-v2 was renamed keep the manifest as inventory.json.
	schedule = (
		(_json(schedule_dir / "manifest.json") or _json(schedule_dir / "inventory.json"))
		if schedule_dir
		else None
	) or {}
	partition = schedule.get("attacker_partition")
	campaign = campaign or directory / "campaign"
	summary = _json(campaign / "campaign.json")
	contract = _json(campaign / "contract.json")
	detectability = _json(campaign / "detectability.json") or {}
	outcome = (directory / "outcome").read_text() if (directory / "outcome").is_file() else ""
	if summary is not None:
		status = "complete"
	elif campaign.exists():
		status = "interrupted"
	elif outcome.startswith("gate refused"):
		status = "gate-refused"
	else:
		status = "not-run"
	if summary is not None:
		contract_hash = summary.get("contract")
	elif contract is not None:
		contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
	else:
		contract_hash = None
	report = load_campaign_summary(campaign) or {}
	levels = report.get("levels") or {}
	return {
		"kind": "campaign",
		"attack": attack,
		"arm": arm,
		"stratum": "all",
		"seed": schedule.get("seed"),
		"attempt_budget": schedule.get("attempt_budget"),
		"deadline_seconds": schedule.get("deadline_seconds"),
		"partition": partition,
		"study": STUDY.get(partition or "", "unknown"),
		"status": status,
		"reading_label": detectability.get("reading"),
		"contract_sha256": contract_hash,
		"schedule_sha256": _sha256(schedule_dir / "requests.jsonl") if schedule_dir else None,
		"ledger_sha256": _sha256(campaign / "ledger.jsonl"),
		**_arm_settings(directory, arm, calibration),
		**{
			field: report.get(field)
			for field in (
				"attempts",
				"received",
				"blocked",
				"challenge",
				"failed",
				"deadline",
				"pending",
				"released_tokens",
				"released_tokens_unknown",
			)
		},
		"distinct_examples": report.get("distinct_examples"),
		"first_alert_seconds": report.get("first_alert_seconds"),
		"first_restriction_seconds": report.get("first_restriction_seconds"),
		"first_refusal_seconds": report.get("first_refusal_seconds"),
		# Without policy events, empty level cells mean unattributed, not unescalated.
		"attribution_complete": report.get("attribution_complete"),
		"unattributed_queries": report.get("unattributed_queries"),
		# J and Q are left empty for an unreached level rather than given censored values.
		**{
			f"L{level}_{field}": (levels.get(f"L{level}") or {}).get(field)
			if (levels.get(f"L{level}") or {}).get("reached")
			else None
			for level in range(1, 5)
			for field in ("seconds", "J", "Q")
		},
	}


UTILITY_STRATA = ("domain", "domain_profile", "client_type")


def utility_rows(
	directory: Path, calibration: Path | None, arm: str, *, name: str = "utility.json"
) -> list[dict[str, Any]]:
	"""A pooled `all` row, then one row per domain, profile and client type in `utility.json`.

	A stratum's completion is its served share of submitted requests; its p95 latency is over
	served requests only.
	"""
	result = _json(directory / name)
	base = {
		"kind": "utility",
		"attack": "benign_test",
		"arm": arm,
		"status": "complete" if result is not None else "not-run",
		"study": None,
		**_arm_settings(directory, arm, calibration),
	}
	rows = [
		{
			**base,
			"stratum": "all",
			**{
				field: (result or {}).get(field)
				for field in (
					"submitted",
					"served",
					"completion_rate",
					"served_only_quality",
					"unconditional_utility",
					"latency_p95",
				)
			},
		}
	]
	for field in UTILITY_STRATA:
		for value, stratum in sorted(((result or {}).get("strata") or {}).get(field, {}).items()):
			submitted = stratum.get("submitted")
			rows.append(
				{
					**base,
					"stratum": f"{field}={value}",
					"submitted": submitted,
					"served": stratum.get("served"),
					"completion_rate": stratum["served"] / submitted if submitted else None,
					"served_only_quality": stratum.get("served_only_quality"),
					"unconditional_utility": stratum.get("unconditional_utility"),
					"latency_p95": (stratum.get("latency_seconds") or {}).get("p95"),
				}
			)
	return rows


def run_rows(run: Path, reading: str, commit: str) -> list[dict[str, Any]]:
	"""Rows for every campaign and utility arm directory in a `scripts/start.sh` run."""
	calibration = run / "calibration" / "calibration.json"
	rows = [
		campaign_row(arm, run / "schedules" / attack.name, calibration, attack.name, arm.name)
		for attack in sorted((run / "campaigns").glob("*"))
		for arm in sorted(attack.iterdir())
		if arm.is_dir()
	]
	rows += [
		row
		for arm in sorted((run / "utility").glob("*"))
		if arm.is_dir()
		for row in utility_rows(arm, calibration, arm.name)
	]
	for row in rows:
		row.update(reading=reading, run_id=run.name, commit=commit)
	return rows


def _seed_budget(name: str) -> tuple[int | None, int | None]:
	match = re.fullmatch(r"s(\d+)-b(\d+)", name)
	return (int(match[1]), int(match[2])) if match else (None, None)


def backfill_rows(readings: Path) -> list[dict[str, Any]]:
	"""Rows for the frozen v3 study and the v4 audit's undefended v2 evidence.

	Both predate the runner layout and the index, so their rows are labelled `backfill:` in
	`run_id`, carry no commit, and count as development evidence: the v4 audit inspected them.
	Fields the frozen files never recorded stay empty. The v2 ledgers have no event log, so
	their counts come from the translated legacy `campaign.json`.
	"""
	rows: list[dict[str, Any]] = []
	v3 = readings / "v3"
	calibration = v3 / "calibration" / "calibration.json"
	for directory in sorted((v3 / "harness").glob("*/*/*")):
		attack, schedule = directory.parent.parent.name, directory.parent.name
		arm = directory.name.split(".")[0]
		row = campaign_row(
			directory, v3 / "schedules" / attack / schedule, calibration, attack, arm
		)
		rows.append({**row, "run_id": "backfill:v3", "reading": "v3"})
		if (directory / "utility.json").is_file():
			rows += [
				{**u, "run_id": "backfill:v3", "reading": "v3"}
				for u in utility_rows(directory, calibration, arm)
			]
	if (v3 / "utility" / "undefended.json").is_file():
		rows += [
			{**u, "run_id": "backfill:v3", "reading": "v3"}
			for u in utility_rows(v3 / "utility", None, "undefended", name="undefended.json")
		]
	for directory in sorted((readings / "v4" / "evidence" / "v2").glob("*/*/*")):
		attack, arm = directory.parent.parent.name, directory.name
		seed, budget = _seed_budget(directory.parent.name)
		row = campaign_row(directory, None, None, attack, arm, campaign=directory)
		row.update(seed=seed, attempt_budget=budget, run_id="backfill:v4-evidence-v2")
		rows.append({**row, "reading": "v4"})
	for row in rows:
		row.update(commit=None, study="development")
	return rows


def student_rows(output: Path, index_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
	"""One row per training-set size of a summarized learning curve in `output`.

	The curve's training ledgers are listed by SHA-256 and resolved against campaign rows
	already in the index, so a student names the reading, run and arm its data came from. A
	ledger the index does not know stays listed by hash only.
	"""
	plan = _json(output / "plan.json")
	summary = _json(output / "summary.json")
	if plan is None or summary is None:
		raise ValueError(f"{output} needs plan.json and summary.json; run the summary first")
	ledgers = sorted(digest for path, digest in plan["inputs"].items() if path.endswith(".jsonl"))
	sources = sorted(
		f"{row['reading']}/{row['run_id']}/{row['attack']}/{row['arm']}"
		for row in index_rows
		if row["kind"] == "campaign" and row["ledger_sha256"] in ledgers
	)
	target = summary.get("target_gain_pp")
	return [
		{
			"run_id": f"learning-curve:{output.name}",
			"kind": "student",
			"attack": "learning-curve",
			"arm": "undefended",
			"stratum": f"n={result['n']}",
			"status": "complete",
			"study": "development",
			"student_task": "ag_news",
			"student_n": result["n"],
			"student_seeds": len(result["seed_gains_pp"]),
			"student_gain_pp": result["mean_gain_pp"],
			"student_gain_lower": result["ci95_pp"][0],
			"student_gain_upper": result["ci95_pp"][1],
			"student_target_reached": None if target is None else result["mean_gain_pp"] >= target,
			"training_ledger_sha256": ";".join(ledgers),
			"training_runs": ";".join(sources) or None,
		}
		for result in summary["results"]
	]


def _cell(value: Any) -> str:
	if value is None:
		return ""
	if isinstance(value, float):
		return repr(value)
	return str(value)


def read_index(path: Path) -> list[dict[str, str]]:
	if not path.is_file():
		return []
	with path.open(newline="") as stream:
		reader = csv.DictReader(stream, delimiter="\t")
		unknown = set(reader.fieldnames or ()) - set(COLUMNS)
		if unknown:
			raise ValueError(f"{path} has columns {INDEX_VERSION} does not know: {sorted(unknown)}")
		# Columns added since the file was written are empty, which reads as unknown.
		return [{column: row.get(column) or "" for column in COLUMNS} for row in reader]


def update_index(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
	"""Replace the rows of each run in `rows`, keep all others, and write the index sorted."""
	runs = {row["run_id"] for row in rows}
	kept = [row for row in read_index(path) if row["run_id"] not in runs]
	merged = kept + [{column: _cell(row.get(column)) for column in COLUMNS} for row in rows]
	keys = [tuple(row[column] for column in KEY) for row in merged]
	if len(set(keys)) != len(keys):
		raise ValueError("results index rows must be unique by run, kind, attack, arm and stratum")
	contracts = [row["contract_sha256"] for row in merged if row["contract_sha256"]]
	if len(set(contracts)) != len(contracts):
		raise ValueError("a campaign contract appears in more than one results index row")
	merged.sort(key=lambda row: tuple(row[column] for column in KEY))
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="") as stream:
		writer = csv.DictWriter(stream, COLUMNS, delimiter="\t", lineterminator="\n")
		writer.writeheader()
		writer.writerows(merged)
	return merged


# Campaign outcomes compared across arms. Each run is one independent unit.
OUTCOME_FIELDS = (
	"received",
	"released_tokens",
	"distinct_examples",
	"L4_J",
	"L4_Q",
	"first_refusal_seconds",
)
UTILITY_FIELDS = ("completion_rate", "served_only_quality", "unconditional_utility")


def _number(value: str) -> float | None:
	return float(value) if value else None


def _outcomes(rows: list[dict[str, str]], fields: tuple[str, ...]) -> dict[str, Any]:
	from distillation.evaluation_protocol import cluster_interval

	result: dict[str, Any] = {}
	for field in fields:
		units = [(row["run_id"], _number(row[field])) for row in rows]
		known = [(run, value) for run, value in units if value is not None]
		entry: dict[str, Any] = {
			"units": {run: value for run, value in units},
			"unknown_units": len(units) - len(known),
			"estimate": sum(v for _, v in known) / len(known) if known else None,
		}
		if len(known) >= 2:
			entry["interval"] = cluster_interval(known)
		else:
			entry["interval"] = None
			entry["interval_reason"] = "fewer than two independent runs with a known value"
		result[field] = entry
	return result


def comparison_from_index(
	rows: list[dict[str, str]],
	*,
	attack: str,
	arms: list[str],
	view: str,
	declared: dict[str, Any],
	include_backfill: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
	"""Conditions for `compare_conditions` and per-arm outcomes, built from index rows.

	Each arm pools its complete runs of `attack`; a run is the independent unit for the
	intervals. Query budget and deadline come from the rows and must agree within an arm.
	Fields the index does not record, such as the output-token and money budgets or the
	utility target, are taken only from `declared` and stay missing otherwise. An arm's runs
	must share policy, accumulator and attacker partition. Backfilled historical rows ran
	different code and workloads, so they are left out unless `include_backfill` is set.
	"""
	if len(set(arms)) != len(arms) or len(arms) < 2:
		raise ValueError("compare at least two distinct arms")
	conditions: list[dict[str, Any]] = []
	outcomes: dict[str, Any] = {}
	studies: set[str] = set()
	for arm in arms:
		selected = [
			r
			for r in rows
			if r["attack"] == attack
			and r["arm"] == arm
			and (include_backfill or not r["run_id"].startswith("backfill:"))
		]
		campaigns = [r for r in selected if r["kind"] == "campaign"]
		complete = [r for r in campaigns if r["status"] == "complete"]
		runs = {r["run_id"] for r in complete}
		measured = [
			r
			for r in rows
			if r["kind"] == "utility"
			and r["arm"] == arm
			and r["status"] == "complete"
			and r["run_id"] in runs
		]
		utility = [r for r in measured if r["stratum"] == "all"]
		strata: dict[str, list[dict[str, str]]] = defaultdict(list)
		for row in measured:
			if row["stratum"] != "all":
				strata[row["stratum"]].append(row)
		if not complete:
			raise ValueError(f"no complete {attack} campaign for arm {arm} in the index")
		studies.update(r["study"] for r in complete)
		for field in ("policy", "accumulator", "partition"):
			if len({r[field] for r in complete}) > 1:
				raise ValueError(f"arm {arm} pools runs with different {field} values")

		def shared(field: str, rows: list[dict[str, str]] = complete) -> float | None:
			values = {r[field] for r in rows}
			return _number(values.pop()) if len(values) == 1 else None

		condition: dict[str, Any] = {
			"arm": arm,
			"attack": attack,
			"policy": complete[0]["policy"] or None,
			"runs": sorted(runs),
			"commits": sorted({r["commit"] or "unknown" for r in complete}),
			"excluded_runs": {
				r["run_id"]: r["status"] for r in campaigns if r["status"] != "complete"
			},
			"query_budget": shared("attempt_budget"),
			"deadline_seconds": shared("deadline_seconds"),
			"output_token_budget": declared.get("output_token_budget"),
			"money_budget": declared.get("money_budget"),
			"declared": sorted(k for k, v in declared.items() if v is not None),
		}
		if view == "legitimate_utility":
			metric = declared.get("utility_metric")
			values = [_number(r.get(metric or "", "")) for r in utility]
			known = [v for v in values if v is not None]
			condition.update(
				utility_metric=metric,
				utility_target=declared.get("utility_target"),
				utility_tolerance=declared.get("utility_tolerance"),
				utility_population="benign_test" if utility else None,
				utility_observed=sum(known) / len(known) if known else None,
				utility_runs=sorted(r["run_id"] for r in utility),
			)
		conditions.append(condition)
		outcomes[arm] = {
			"campaign": _outcomes(complete, OUTCOME_FIELDS),
			"utility": _outcomes(utility, UTILITY_FIELDS) if utility else None,
			# A priority group can lose service while the pooled average looks acceptable.
			"utility_by_stratum": {
				name: _outcomes(group, UTILITY_FIELDS) for name, group in sorted(strata.items())
			},
		}
	if len(studies) > 1:
		raise ValueError("refusing to pool development and confirmatory runs in one comparison")
	return conditions, outcomes
