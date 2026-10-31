"""Campaign timing derives detection, throttling, and stopping from the event log only."""

import json
from pathlib import Path

from distillation.benchmarks.reporting import distribution
from distillation.campaign_timing import campaign_timing


def _write(directory: Path, rows: list[tuple[str, str, float, str, int, str]]) -> None:
	"""Rows are (account, domain, start, status, level, action) for one query each."""
	events = []
	for index, (account, domain, at, status, level, action) in enumerate(rows, 1):
		identifier = f"a{index}"
		base = {"account": account, "domain": domain, "policy_request_id": f"p{index}"}
		events.append(
			base
			| {
				"event": "attempt",
				"id": identifier,
				"index": index,
				"kind": "query",
				"at_seconds": at,
			}
		)
		events.append(
			{
				"event": "outcome",
				"id": identifier,
				"status": status,
				"at_seconds": at + 0.5,
				"latency_seconds": 0.5 if action == "allow" else 1.5,
				"output_tokens": 10 if status == "received" else None,
			}
		)
		events.append(base | {"event": "policy", "level": level, "action": action})
	events.append({"event": "solve", "policy_request_id": "p3", "hashes": 100, "seconds": 2.0})
	(directory / "campaign.events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def test_timing_reports_first_reach_per_account_stops_and_tail_latency(tmp_path: Path) -> None:
	_write(
		tmp_path,
		[
			("a", "ag_news", 0, "received", 0, "allow"),
			("a", "squad", 1, "received", 1, "allow"),
			("a", "squad", 2, "received", 2, "throttle"),
			("a", "ag_news", 3, "blocked", 4, "block"),
			("a", "ag_news", 4, "blocked", 4, "block"),
			("b", "gsm8k", 5, "received", 0, "allow"),
			("b", "gsm8k", 6, "blocked", 4, "block"),
		],
	)
	timing = campaign_timing(tmp_path)
	l1, l4 = timing["levels"]["L1"], timing["levels"]["L4"]
	assert (l1["attempt_index"], l1["seconds"], l1["account"], l1["domain"]) == (2, 1, "a", "squad")
	assert l1["received_before"] == {"ag_news": {"responses": 1, "tokens": 10, "unknown_tokens": 0}}
	assert l4["seconds"] == 3 and l4["received_before"]["squad"]["responses"] == 2
	assert timing["accounts"]["a"]["stopped"] and timing["accounts"]["a"]["stop_seconds"] == 3
	# One block after a success is not a stop, so the campaign has not stopped.
	assert not timing["accounts"]["b"]["stopped"] and not timing["campaign_stopped"]
	assert timing["accounts_blocked"] == 2 and timing["received_after_first_block"] == 1
	assert timing["latency_seconds"]["by_action"]["throttle"]["p50"] == 1.5
	assert timing["challenge_solve_seconds"]["n"] == 1
	assert timing["delivery_windows"]["received"] == [4]
	assert timing["attribution_complete"]


def test_distribution_marks_unsupported_tail_quantiles() -> None:
	values = [float(v) for v in range(1, 101)]
	result = distribution(values)
	assert result["p50"] == 50.5 and result["p99"] > result["p95"] > result["p90"]
	assert result["p99_supported"] and result["p90_supported"]
	small = distribution(values[:15])
	assert small["p90_supported"] and not small["p95_supported"] and not small["p99_supported"]
	assert distribution([])["p50"] is None
