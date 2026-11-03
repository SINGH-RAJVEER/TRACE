"""One row per campaign arm and benign utility arm of an exported run, kept in plain TSV.

Rows are keyed by run, kind, attack and arm, so a gate-refused or interrupted arm still has a
row. An empty cell means unknown, never zero. Re-exporting a run replaces its rows. Campaign
counts come from the event log, so an interrupted arm reports what it observed before stopping.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from distillation.evaluation_protocol import campaign_report

INDEX_VERSION = "results-index-v1"
COLUMNS = (
	"reading",
	"run_id",
	"commit",
	"kind",
	"attack",
	"arm",
	"policy",
	"accumulator",
	"seed",
	"attempt_budget",
	"partition",
	"study",
	"status",
	"reading_label",
	"contract_sha256",
	"schedule_sha256",
	"calibration_sha256",
	"policy_revision",
	"attempts",
	"released",
	"failed",
	"pending",
	"released_tokens",
	"released_tokens_unknown",
	"distinct_examples",
	"first_alert_seconds",
	"first_restriction_seconds",
	"first_refusal_seconds",
	"L1_seconds",
	"L2_seconds",
	"L3_seconds",
	"L4_seconds",
	"submitted",
	"served",
	"completion_rate",
	"served_only_quality",
	"unconditional_utility",
	"latency_p95",
)
KEY = ("run_id", "kind", "attack", "arm")
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


def _arm_settings(directory: Path, run: Path) -> dict[str, Any]:
	env = _env(directory / "harness.env")
	return {
		"policy": env.get("HARNESS_POLICY"),
		"accumulator": env.get("HARNESS_ACCUMULATOR")
		if env.get("HARNESS_POLICY") == "mmd"
		else None,
		"calibration_sha256": _sha256(run / "calibration" / "calibration.json"),
		"policy_revision": _policy_revision(directory / "decisions.jsonl"),
	}


def campaign_row(run: Path, attack: str, arm: str) -> dict[str, Any]:
	directory = run / "campaigns" / attack / arm
	schedule = _json(run / "schedules" / attack / "manifest.json") or {}
	partition = schedule.get("attacker_partition")
	campaign = directory / "campaign"
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
	events = campaign / "campaign.events.jsonl"
	report = (
		campaign_report(
			[json.loads(line) for line in events.read_text().splitlines() if line.strip()]
		)
		if events.is_file()
		else {}
	)
	levels = report.get("levels") or {}
	return {
		"kind": "campaign",
		"attack": attack,
		"arm": arm,
		"seed": schedule.get("seed"),
		"attempt_budget": schedule.get("attempt_budget"),
		"partition": partition,
		"study": STUDY.get(partition or "", "unknown"),
		"status": status,
		"reading_label": detectability.get("reading"),
		"contract_sha256": contract_hash,
		"schedule_sha256": _sha256(run / "schedules" / attack / "requests.jsonl"),
		**_arm_settings(directory, run),
		"attempts": report.get("attempts"),
		"released": report.get("released"),
		"failed": report.get("failed"),
		"pending": report.get("pending"),
		"released_tokens": report.get("released_tokens"),
		"released_tokens_unknown": report.get("released_tokens_unknown_count"),
		"distinct_examples": report.get("distinct_examples"),
		"first_alert_seconds": report.get("first_alert_seconds"),
		"first_restriction_seconds": report.get("first_restriction_seconds"),
		"first_refusal_seconds": report.get("first_refusal_seconds"),
		**{f"L{level}_seconds": levels.get(f"L{level}") for level in range(1, 5)},
	}


def utility_row(run: Path, arm: str) -> dict[str, Any]:
	directory = run / "utility" / arm
	result = _json(directory / "utility.json")
	return {
		"kind": "utility",
		"attack": "benign_test",
		"arm": arm,
		"status": "complete" if result is not None else "not-run",
		"study": None,
		**_arm_settings(directory, run),
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


def run_rows(run: Path, reading: str, commit: str) -> list[dict[str, Any]]:
	"""Rows for every campaign and utility arm directory in a `scripts/start.sh` run."""
	rows = [
		campaign_row(run, attack.name, arm.name)
		for attack in sorted((run / "campaigns").glob("*"))
		for arm in sorted(attack.iterdir())
		if arm.is_dir()
	]
	rows += [
		utility_row(run, arm.name) for arm in sorted((run / "utility").glob("*")) if arm.is_dir()
	]
	for row in rows:
		row.update(reading=reading, run_id=run.name, commit=commit)
	return rows


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
		if tuple(reader.fieldnames or ()) != COLUMNS:
			raise ValueError(f"{path} columns differ from {INDEX_VERSION}; migrate it first")
		return list(reader)


def update_index(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
	"""Replace the rows of each run in `rows`, keep all others, and write the index sorted."""
	runs = {row["run_id"] for row in rows}
	kept = [row for row in read_index(path) if row["run_id"] not in runs]
	merged = kept + [{column: _cell(row.get(column)) for column in COLUMNS} for row in rows]
	keys = [tuple(row[column] for column in KEY) for row in merged]
	if len(set(keys)) != len(keys):
		raise ValueError("results index rows must be unique by run, kind, attack and arm")
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
