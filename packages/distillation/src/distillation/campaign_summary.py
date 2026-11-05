"""The one summary of a campaign event log, written to `campaign.json` and read everywhere else.

`attempts` counts every HTTP submission, proof redemptions included; `query_attempts` and
`proof_attempts` split it. Outcome counts keep `failed` (an HTTP or transport failure before
the deadline) apart from `deadline` (a failure after the campaign's time ran out).

Level reaches follow the manuscript. `J` is the first campaign attempt index whose account was
at or above the level, and `seconds` is that attempt's start, the decision time t. Attempt
indices and start times are assigned together, so index order and time order agree. `Q` counts
received responses that completed strictly before t, and `Q_by_domain` splits them with their
released tokens. An unreached level is censored: `J` is then the number of observed attempts
and `Q` every received response.

Unknown values stay None. Admission and cancellation are not in the event log. Released tokens
are None when any received response lacks usage; the known subtotal and the unknown count are
reported beside them.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SUMMARY_VERSION = "campaign-summary-v1"
LEVELS = (1, 2, 3, 4)
OUTCOMES = ("received", "blocked", "challenge", "failed", "deadline")


def summarize_campaign(events: list[dict[str, Any]]) -> dict[str, Any]:
	attempts = sorted((e for e in events if e["event"] == "attempt"), key=lambda a: a["index"])
	ids = {a["id"] for a in attempts}
	outcomes = {e["id"]: e for e in events if e["event"] == "outcome" and e["id"] in ids}
	# Policy events are keyed by prompt, so a redeemed challenge keeps only its later serve
	# decision; the per-attempt outcome status still records the challenge itself.
	decisions = {e["policy_request_id"]: e for e in events if e["event"] == "policy"}
	queries = [a for a in attempts if a["kind"] == "query"]
	received = [a for a in attempts if outcomes.get(a["id"], {}).get("status") == "received"]
	tokens = [outcomes[a["id"]].get("output_tokens") for a in received]

	def decision(attempt: dict[str, Any]) -> dict[str, Any]:
		return decisions.get(attempt["policy_request_id"], {})

	def first(predicate: Any) -> float | None:
		return next((float(a["at_seconds"]) for a in attempts if predicate(a)), None)

	# A query that never reached the policy, such as a transport failure, has no decision to
	# attribute; one that came back with a harness decision must have a policy event.
	attribution_complete = bool(decisions) and all(
		a["policy_request_id"] in decisions
		for a in queries
		if outcomes.get(a["id"], {}).get("decision") is not None
	)
	levels: dict[str, Any] = {}
	for level in LEVELS:
		reach = next((a for a in attempts if decision(a).get("level", -1) >= level), None)
		by_domain: dict[str, dict[str, int]] = defaultdict(
			lambda: {"responses": 0, "tokens": 0, "unknown_tokens": 0}
		)
		for attempt in received:
			outcome = outcomes[attempt["id"]]
			if reach is not None and outcome["at_seconds"] >= reach["at_seconds"]:
				continue
			row = by_domain[attempt.get("domain") or "unknown"]
			row["responses"] += 1
			if outcome.get("output_tokens") is None:
				row["unknown_tokens"] += 1
			else:
				row["tokens"] += int(outcome["output_tokens"])
		levels[f"L{level}"] = {
			"reached": reach is not None,
			"censored": reach is None,
			"J": reach["index"] if reach else len(attempts),
			"Q": sum(row["responses"] for row in by_domain.values()),
			"Q_by_domain": dict(sorted(by_domain.items())),
			"seconds": reach["at_seconds"] if reach else None,
			"account": reach.get("account") if reach else None,
			"domain": reach.get("domain") if reach else None,
			"attribution_complete": attribution_complete,
		}
	return {
		"summary_version": SUMMARY_VERSION,
		"attempts": len(attempts),
		"query_attempts": len(queries),
		"proof_attempts": len(attempts) - len(queries),
		**{status: sum(o["status"] == status for o in outcomes.values()) for status in OUTCOMES},
		"pending": len(ids - outcomes.keys()),
		"admitted": None,
		"cancelled": None,
		"unknown_reason": "the event log records neither admission nor cancellation",
		"released_tokens": sum(tokens) if all(t is not None for t in tokens) else None,
		"released_tokens_known": sum(t for t in tokens if t is not None),
		"released_tokens_unknown": sum(t is None for t in tokens),
		"distinct_examples": len({a["example_id"] for a in received})
		if all(a.get("example_id") for a in received)
		else None,
		"first_alert_seconds": levels["L1"]["seconds"],
		"first_restriction_seconds": first(
			lambda a: (
				outcomes.get(a["id"], {}).get("status") in ("challenge", "blocked")
				or decision(a).get("action") in ("challenge", "refuse")
			)
		),
		"first_refusal_seconds": first(
			lambda a: (
				outcomes.get(a["id"], {}).get("status") == "blocked"
				or decision(a).get("action") == "refuse"
			)
		),
		"restriction_coverage": "challenge and refusal actions only; delayed serve is not recorded",
		"levels": levels,
		"unattributed_queries": sum(a["policy_request_id"] not in decisions for a in queries),
		"attribution_complete": attribution_complete,
	}


def read_events(path: Path) -> list[dict[str, Any]]:
	return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_campaign_summary(directory: Path) -> dict[str, Any] | None:
	"""The canonical summary of a campaign directory, or None when it has neither file.

	`source` names what it was built from: the event log, a current `campaign.json`, or a
	legacy `campaign.json`.

	The event log is the source when present. A `campaign.json` written before this summary
	existed, without its event log, is translated: fields it never recorded stay None.
	"""
	events = directory / "campaign.events.jsonl"
	if events.is_file():
		return {**summarize_campaign(read_events(events)), "source": "event log"}
	path = directory / "campaign.json"
	if not path.is_file():
		return None
	stored = json.loads(path.read_text())
	if stored.get("summary_version") == SUMMARY_VERSION:
		return {
			**{key: stored.get(key) for key in summarize_campaign([])},
			"source": "campaign.json",
		}
	return _legacy(stored)


def _legacy(stored: dict[str, Any]) -> dict[str, Any]:
	summary: dict[str, Any] = dict.fromkeys(summarize_campaign([]))
	proofs = stored.get("proof_requests")
	summary.update(
		summary_version=None,
		attempts=stored.get("queries"),
		proof_attempts=proofs,
		query_attempts=stored["queries"] - proofs
		if stored.get("queries") is not None and proofs is not None
		else None,
		**{status: stored.get(status) for status in OUTCOMES},
		pending=stored.get("pending"),
		unknown_reason="translated from a legacy campaign.json without its event log",
		released_tokens=stored.get("released_output_tokens"),
		restriction_coverage=None,
		levels={
			name: None
			if level is None
			else {
				"reached": level.get("reached"),
				"censored": level.get("censored"),
				"J": level.get("J"),
				"Q": level.get("Q"),
				"Q_by_domain": None,
				"seconds": None,
				"account": None,
				"domain": None,
				"attribution_complete": level.get("attribution_complete"),
			}
			for name, level in (stored.get("levels") or {}).items()
		},
	)
	return {**summary, "source": "legacy campaign.json"}
