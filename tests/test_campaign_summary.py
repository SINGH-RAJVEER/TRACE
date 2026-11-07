"""The canonical campaign summary, from synthetic event logs without a teacher or harness."""

import json
from pathlib import Path
from typing import Any

from distillation.campaign_summary import load_campaign_summary, summarize_campaign
from distillation.campaign_timing import campaign_timing


def _attempt(
	index: int, prompt: str, kind: str, at: float, account: str, domain: str
) -> dict[str, Any]:
	return {
		"event": "attempt",
		"id": f"a{index}",
		"policy_request_id": prompt,
		"example_id": f"x-{prompt}",
		"index": index,
		"kind": kind,
		"at_seconds": at,
		"account": account,
		"domain": domain,
	}


def _outcome(index: int, status: str, at: float, **extra: Any) -> dict[str, Any]:
	return {"event": "outcome", "id": f"a{index}", "status": status, "at_seconds": at, **extra}


def _policy(prompt: str, level: int, action: str) -> dict[str, Any]:
	return {"event": "policy", "policy_request_id": prompt, "level": level, "action": action}


# Two accounts and domains, a redeemed challenge, a refusal, a failure, a deadline, a pending
# attempt and one received response without usage.
EVENTS = [
	_attempt(1, "p1", "query", 0.0, "a", "ag_news"),
	_outcome(1, "received", 1.0, output_tokens=10, decision="serve"),
	_policy("p1", 0, "serve"),
	_attempt(2, "p2", "query", 0.5, "b", "squad"),
	_outcome(2, "received", 2.5, output_tokens=None, decision="serve"),
	_policy("p2", 1, "serve"),
	_attempt(3, "p3", "query", 2.0, "a", "squad"),
	_outcome(3, "challenge", 2.2, decision="challenge"),
	_attempt(4, "p3", "proof", 3.0, "a", "squad"),
	_outcome(4, "received", 4.0, output_tokens=7, decision="serve"),
	# The later redemption decision replaces the challenge decision for this prompt.
	_policy("p3", 3, "serve"),
	_attempt(5, "p4", "query", 5.0, "b", "ag_news"),
	_outcome(5, "blocked", 5.1, decision="block"),
	_policy("p4", 4, "refuse"),
	_attempt(6, "p5", "query", 6.0, "a", "ag_news"),
	_outcome(6, "failed", 6.5),
	_policy("p5", 2, "serve"),
	_attempt(7, "p6", "query", 7.0, "b", "squad"),
	_outcome(7, "deadline", 8.0),
	_attempt(8, "p7", "query", 9.0, "a", "squad"),
]


def test_summary_counts_unknowns_and_interventions() -> None:
	summary = summarize_campaign(EVENTS)
	assert (summary["attempts"], summary["query_attempts"], summary["proof_attempts"]) == (8, 7, 1)
	assert [summary[s] for s in ("received", "blocked", "challenge", "failed", "deadline")] == [
		3,
		1,
		1,
		1,
		1,
	]
	assert summary["pending"] == 1
	assert summary["admitted"] is None and summary["cancelled"] is None
	assert summary["released_tokens"] is None
	assert (summary["released_tokens_known"], summary["released_tokens_unknown"]) == (17, 1)
	assert summary["distinct_examples"] == 3
	# A redeemed challenge still counts as a restriction; the refusal comes later.
	assert summary["first_alert_seconds"] == 0.5
	assert summary["first_restriction_seconds"] == 2.0
	assert summary["first_refusal_seconds"] == 5.0
	# The deadline attempt never reached the policy, so attribution is complete but disclosed.
	assert summary["attribution_complete"] and summary["unattributed_queries"] == 2


def test_levels_follow_the_manuscript_definitions_of_j_and_q() -> None:
	levels = summarize_campaign(EVENTS)["levels"]
	assert {name: (level["J"], level["Q"]) for name, level in levels.items()} == {
		"L1": (2, 0),
		"L2": (3, 1),
		"L3": (3, 1),
		"L4": (5, 3),
	}
	l4 = levels["L4"]
	assert (l4["seconds"], l4["account"], l4["domain"]) == (5.0, "b", "ag_news")
	assert l4["Q_by_domain"] == {
		"ag_news": {"responses": 1, "tokens": 10, "unknown_tokens": 0},
		"squad": {"responses": 2, "tokens": 7, "unknown_tokens": 1},
	}
	censored = summarize_campaign(EVENTS[:3])["levels"]["L4"]
	assert censored["censored"] and (censored["J"], censored["Q"]) == (1, 1)


def test_timing_and_loader_share_the_summary(tmp_path: Path) -> None:
	(tmp_path / "campaign.events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in EVENTS))
	summary = summarize_campaign(EVENTS)
	timing = campaign_timing(tmp_path)
	assert timing["levels"] == summary["levels"]
	assert (timing["attempts"], timing["received"]) == (8, 3)
	assert timing["accounts"]["b"]["first_block_seconds"] == 5.0
	loaded = load_campaign_summary(tmp_path)
	assert loaded is not None and loaded["source"] == "event log"
	assert {k: v for k, v in loaded.items() if k != "source"} == summary


def test_loader_translates_a_legacy_campaign_json(tmp_path: Path) -> None:
	legacy = {
		"queries": 6000,
		"proof_requests": 0,
		"received": 1215,
		"blocked": 4785,
		"failed": 0,
		"deadline": 0,
		"challenge": 0,
		"pending": 0,
		"released_output_tokens": 118872,
		"levels": {"L4": {"J": 1216, "Q": 1215, "reached": True, "censored": False}},
	}
	(tmp_path / "campaign.json").write_text(json.dumps(legacy))
	loaded = load_campaign_summary(tmp_path)
	assert loaded is not None and loaded["source"] == "legacy campaign.json"
	assert (loaded["attempts"], loaded["query_attempts"], loaded["received"]) == (6000, 6000, 1215)
	assert loaded["released_tokens"] == 118872 and loaded["distinct_examples"] is None
	assert loaded["levels"]["L4"]["J"] == 1216 and loaded["levels"]["L4"]["seconds"] is None
	assert load_campaign_summary(tmp_path / "missing") is None


def test_old_events_keep_the_v1_summary_and_stored_v2_keeps_its_fields(tmp_path: Path) -> None:
	assert summarize_campaign(EVENTS)["summary_version"] == "campaign-summary-v1"
	assert "restricted" not in summarize_campaign(EVENTS)
	stored = {
		"summary_version": "campaign-summary-v2",
		"restricted": 2,
		"campaign": {"throttles": 2},
	}
	(tmp_path / "campaign.json").write_text(json.dumps(stored))
	loaded = load_campaign_summary(tmp_path)
	assert loaded is not None and loaded["restricted"] == 2
	assert loaded["campaign"] == {"throttles": 2} and loaded["source"] == "campaign.json"
