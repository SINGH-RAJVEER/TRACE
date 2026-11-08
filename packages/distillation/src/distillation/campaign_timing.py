"""Per-account stopping and latency of one executed campaign, from its event log.

Counts and level reaches, with J, Q and their time, account and domain, come from
`summarize_campaign`, the same summary `campaign.json` holds. This report adds when each
account was challenged, blocked and observed stopped, whether the whole campaign stopped,
deliveries per time window, and p50/p90/p95/p99 distributions of latency, solving and
end-to-end time per prompt. Times are seconds from the campaign start. Nothing here is
inferred from a schedule alone.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

from distillation.benchmarks.reporting import distribution
from distillation.campaign_summary import read_events, summarize_campaign
from distillation.io import read_jsonl
from distillation.schemas import GatewayRequest

TIMING_VERSION = "campaign-timing-v2"
DEFAULT_WINDOW_SECONDS = 60.0
# One refusal or failure is not a stop: at least this many blocked attempts must follow the
# account's last received response.
STOP_RUN = 2


def _stop(attempts: list[dict[str, Any]], outcomes: dict[str, dict[str, Any]]) -> dict[str, Any]:
	"""The final run of blocked attempts, started after the last received response completed."""
	ordered = sorted(attempts, key=lambda a: a["at_seconds"])
	received = [
		outcomes[a["id"]] for a in ordered if outcomes.get(a["id"], {}).get("status") == "received"
	]
	drained = max((o["at_seconds"] for o in received), default=float("-inf"))
	suffix: list[dict[str, Any]] = []
	for attempt in reversed(ordered):
		if outcomes.get(attempt["id"], {}).get("status") != "blocked":
			break
		suffix.append(attempt)
	suffix.reverse()
	after = [a for a in suffix if a["at_seconds"] >= drained]
	return {
		"stopped": len(after) >= STOP_RUN,
		"stop_seconds": after[0]["at_seconds"] if len(after) >= STOP_RUN else None,
		"final_blocked_attempts": len(suffix),
	}


def campaign_timing(
	directory: Path,
	*,
	schedule: Path | None = None,
	window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> dict[str, Any]:
	"""Summarise `directory/campaign.events.jsonl`; `schedule` adds scheduled-to-start delay."""
	events = read_events(directory / "campaign.events.jsonl")
	summary = summarize_campaign(events)
	attempts = [e for e in events if e["event"] == "attempt"]
	outcomes = {e["id"]: e for e in events if e["event"] == "outcome"}
	policy = {e["policy_request_id"]: e for e in events if e["event"] == "policy"}
	solves = [e for e in events if e["event"] == "solve"]
	by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for attempt in attempts:
		by_prompt[attempt["policy_request_id"]].append(attempt)

	def level(attempt: dict[str, Any]) -> int | None:
		decision = policy.get(attempt["policy_request_id"])
		# A row can lack a detector level: an entitlement refusal never ran the detector.
		value = None if decision is None else decision.get("level")
		return None if value is None else int(value)

	accounts: dict[str, Any] = {}
	grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for attempt in attempts:
		grouped[attempt.get("account") or "default"].append(attempt)
	for account, rows in sorted(grouped.items()):
		statuses = [(a, outcomes.get(a["id"], {}).get("status")) for a in rows]

		def first(status: str, rows: list[tuple[dict[str, Any], Any]] = statuses) -> float | None:
			times = [a["at_seconds"] for a, s in rows if s == status]
			return min(times) if times else None

		reached = {
			f"L{target}": min(
				(a["at_seconds"] for a in rows if (level(a) or -1) >= target), default=None
			)
			for target in range(1, 5)
		}
		accounts[account] = {
			"attempts": len(rows),
			"received": sum(s == "received" for _, s in statuses),
			"first_level_seconds": reached,
			"first_challenge_seconds": first("challenge"),
			"first_block_seconds": first("blocked"),
			**_stop(rows, outcomes),
		}
	stopped = bool(accounts) and all(a["stopped"] for a in accounts.values())
	first_block = min(
		(
			a["first_block_seconds"]
			for a in accounts.values()
			if a["first_block_seconds"] is not None
		),
		default=None,
	)
	received = [o for o in outcomes.values() if o.get("status") == "received"]
	end = max((o["at_seconds"] for o in outcomes.values()), default=0.0)
	windows = [0] * (int(end // window_seconds) + 1 if outcomes else 0)
	for outcome in received:
		windows[int(outcome["at_seconds"] // window_seconds)] += 1

	def latencies(rows: list[dict[str, Any]]) -> dict[str, Any]:
		return distribution([float(o["latency_seconds"]) for o in rows if "latency_seconds" in o])

	by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
	by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for attempt in attempts:
		served = outcomes.get(attempt["id"])
		if served is None or served.get("status") != "received":
			continue
		decision = policy.get(attempt["policy_request_id"], {})
		by_action[str(decision.get("action", "unattributed"))].append(served)
		by_domain[attempt.get("domain") or "unknown"].append(served)
	end_to_end = []
	for rows in by_prompt.values():
		finals = [outcomes[a["id"]] for a in rows if a["id"] in outcomes]
		if any(o.get("status") == "received" for o in finals):
			end_to_end.append(
				max(o["at_seconds"] for o in finals) - min(a["at_seconds"] for a in rows)
			)
	result: dict[str, Any] = {
		"version": TIMING_VERSION,
		"campaign": str(directory),
		"time_origin": "campaign start",
		"summary_version": summary["summary_version"],
		"attempts": summary["attempts"],
		"prompts": len(by_prompt),
		"received": summary["received"],
		"levels": summary["levels"],
		"accounts": accounts,
		"accounts_blocked": sum(a["first_block_seconds"] is not None for a in accounts.values()),
		"accounts_stopped": sum(a["stopped"] for a in accounts.values()),
		"campaign_stopped": stopped,
		"campaign_stop_seconds": max(a["stop_seconds"] for a in accounts.values())
		if stopped
		else None,
		"received_after_first_block": sum(o["at_seconds"] > first_block for o in received)
		if first_block is not None
		else 0,
		"stop_rule": f"every account ends with >= {STOP_RUN} blocked attempts after its last "
		"received response; an observed stop within the campaign, not permanent prevention",
		"delivery_windows": {"seconds": window_seconds, "received": windows},
		"latency_seconds": {
			"received": latencies(received),
			"by_action": {k: latencies(v) for k, v in sorted(by_action.items())},
			"by_domain": {k: latencies(v) for k, v in sorted(by_domain.items())},
		},
		"end_to_end_seconds": distribution(end_to_end),
		"challenge_solve_seconds": distribution([float(s["seconds"]) for s in solves]),
		"challenge_hashes": distribution([float(s["hashes"]) for s in solves]),
		"attribution_complete": summary["attribution_complete"],
	}
	if schedule is not None:
		planned = {r.request_id: r.time for r in read_jsonl(schedule, GatewayRequest)}
		delays = [
			min(a["at_seconds"] for a in rows) - planned[rows[0]["scheduled_id"]]
			for rows in by_prompt.values()
			if rows[0].get("scheduled_id") in planned
		]
		result["schedule_delay_seconds"] = distribution(delays)
	return result
