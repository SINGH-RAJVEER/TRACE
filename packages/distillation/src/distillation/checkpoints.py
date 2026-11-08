"""Export verified chronological snapshots of completed gateway campaigns."""

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from distillation.campaign import (
	CAMPAIGN_CONFIG_FILE,
	DECISIONS_SNAPSHOT_FILE,
	verify_experiment_binding,
)
from distillation.schemas import LedgerRecord

# Attributed families report one cause; the first three stay the cause-agnostic observations.
GENERAL_INTERVENTIONS = ("first_alert", "first_restriction", "first_refusal")
ATTRIBUTED_INTERVENTIONS = (
	"first_campaign_alert",
	"first_campaign_restriction",
	"first_entitlement_restriction",
)
OUTCOME_STATUSES = ("received", "challenge", "blocked", "failed", "deadline")
ENTITLEMENT_BLOCK_CODES = ("organization_required",)


class CheckpointError(ValueError):
	"""An artifact cannot safely be exported or verified."""


def _hash(data: bytes) -> str:
	return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
	return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _time(value: Any) -> float:
	if (
		isinstance(value, bool)
		or not isinstance(value, int | float)
		or not math.isfinite(value)
		or value < 0
	):
		raise CheckpointError("invalid event timestamp")
	return float(value)


def _read(path: Path) -> bytes:
	try:
		return path.read_bytes()
	except OSError as error:
		raise CheckpointError(f"cannot read {path}: {error}") from error


def _json(data: bytes) -> Any:
	try:
		return json.loads(data)
	except (ValueError, UnicodeError) as error:
		raise CheckpointError(f"invalid JSON: {error}") from error


def _lines(data: bytes) -> list[Any]:
	return [_json(line) for line in data.splitlines() if line.strip()]


def _outcome_cause(outcome: dict[str, Any]) -> str:
	restriction = outcome.get("restriction")
	if isinstance(restriction, dict):
		return str(restriction.get("kind"))
	return "entitlement" if outcome.get("code") in ENTITLEMENT_BLOCK_CODES else "policy"


def _outcome_matches(name: str, outcome: dict[str, Any]) -> bool:
	"""Whether an attributed client outcome qualifies; alerts are never client visible."""
	status, cause = outcome["status"], _outcome_cause(outcome)
	restricted = status == "restricted"
	if name == "first_restriction":
		return restricted or status in ("challenge", "blocked")
	if name == "first_refusal":
		return bool(
			status == "blocked"
			or (restricted and outcome["restriction"].get("decision") == "block")
		)
	if name == "first_campaign_restriction":
		return restricted and cause == "campaign"
	if name == "first_entitlement_restriction":
		return (restricted or status == "blocked") and cause == "entitlement"
	return False


def _policy_matches(name: str, policy: dict[str, Any]) -> bool:
	"""Whether a joined decision or throttle row qualifies, without inventing detector levels."""
	throttle = bool(policy["event"] == "campaign_throttle")
	evidence = policy.get("campaign")
	alert = isinstance(evidence, dict) and evidence.get("status") == "alert"
	level, action = policy.get("level"), policy.get("action")
	detector_alert = type(level) is int and level >= 1
	detector_restriction = action in ("challenge", "refuse") or (type(level) is int and level >= 2)
	if name == "first_alert":
		return alert or detector_alert
	if name == "first_campaign_alert":
		return alert
	if name == "first_restriction":
		return throttle or detector_restriction or policy.get("entitlement") is not None
	if name == "first_refusal":
		return bool(action == "refuse")
	if name == "first_campaign_restriction":
		return throttle
	if name == "first_entitlement_restriction":
		return policy.get("entitlement") is not None
	return False


def _policy_cause(policy: dict[str, Any]) -> str:
	if policy["event"] == "campaign_throttle":
		return "campaign"
	if policy.get("entitlement") is not None:
		return "entitlement"
	evidence = policy.get("campaign")
	level = policy.get("level")
	if isinstance(evidence, dict) and evidence.get("status") == "alert" and not level:
		return "campaign"
	return "rate" if policy.get("reason") == "rate" else "policy"


def export_campaign_checkpoints(
	campaign: Path, output: Path, *, response_counts: Sequence[int] = ()
) -> dict[str, Any]:
	"""Freeze receipt-time prefixes and intervention bounds in a new directory."""
	if any(type(n) is not int or n < 0 for n in response_counts) or len(
		set(response_counts)
	) != len(response_counts):
		raise CheckpointError("response counts must be unique nonnegative integers")
	campaign = campaign.resolve()
	names = {
		"contract": "contract.json",
		"summary": "campaign.json",
		"events": "campaign.events.jsonl",
		"ledger": "ledger.jsonl",
	}
	data = {key: _read(campaign / name) for key, name in names.items()}
	contract, summary = _json(data["contract"]), _json(data["summary"])
	if (
		not isinstance(contract, dict)
		or not isinstance(summary, dict)
		or contract.get("version") not in ("gateway-execution-v2", "gateway-execution-v3")
	):
		raise CheckpointError("unsupported gateway contract or summary")
	attributed = contract["version"] == "gateway-execution-v3"
	contract_hash = _hash(json.dumps(contract, sort_keys=True).encode())
	if summary.get("contract") != contract_hash or summary.get("sha256") != {
		"ledger": _hash(data["ledger"]),
		"events": _hash(data["events"]),
	}:
		raise CheckpointError("campaign contract or artifact digest mismatch")
	binding = None
	if attributed:
		try:
			binding = verify_experiment_binding(campaign, contract, summary)
		except (ValueError, OSError, KeyError, TypeError) as error:
			raise CheckpointError(f"experiment binding refused: {error}") from error
		names["decisions"] = DECISIONS_SNAPSHOT_FILE
		data["decisions"] = _read(campaign / DECISIONS_SNAPSHOT_FILE)
		if contract["experiment"]["campaign"] is not None:
			names["campaign_config"] = CAMPAIGN_CONFIG_FILE
			data["campaign_config"] = _read(campaign / CAMPAIGN_CONFIG_FILE)
	variants = contract.get("variants") or {}
	if not isinstance(variants, dict) or any(
		not isinstance(variants.get(key, {}), dict) for key in ("batching", "session_replay")
	):
		raise CheckpointError("invalid campaign variants")
	if (
		variants.get("batching", {}).get("members")
		or variants.get("session_replay", {}).get("request_ids")
		or contract.get("response_selection")
	):
		raise CheckpointError("batching, replay, and response selection are unsupported")
	events = _lines(data["events"])
	attempts: dict[str, dict[str, Any]] = {}
	outcomes: dict[str, dict[str, Any]] = {}
	policies: list[dict[str, Any]] = []
	waits: list[dict[str, Any]] = []
	indices: set[int] = set()
	for event in events:
		if not isinstance(event, dict):
			raise CheckpointError("invalid event")
		kind = event.get("event")
		if kind in ("attempt", "outcome"):
			ident = event.get("id")
			if not isinstance(ident, str) or not ident:
				raise CheckpointError("missing event ID")
			target = attempts if kind == "attempt" else outcomes
			if ident in target:
				raise CheckpointError("duplicate attempt or outcome ID")
			target[ident] = event
			_time(event.get("at_seconds"))
			if kind == "attempt":
				index = event.get("index")
				if (
					type(index) is not int
					or index < 1
					or index in indices
					or event.get("kind") not in ("query", "proof")
					or event.get("contract") != contract_hash
				):
					raise CheckpointError("invalid attempt index, kind, or contract")
				indices.add(index)
			elif event.get("status") not in OUTCOME_STATUSES and not (
				attributed
				and event.get("status") == "restricted"
				and isinstance(event.get("restriction"), dict)
			):
				raise CheckpointError("invalid outcome status")
		elif kind == "policy" or (attributed and kind == "campaign_throttle"):
			policies.append(event)
		elif attributed and kind == "attribution_ambiguous":
			raise CheckpointError("ambiguous policy attribution")
		elif attributed and kind == "restriction_wait":
			_time(event.get("at_seconds"))
			_time(event.get("seconds"))
			waits.append(event)
		elif attributed and kind == "restriction_give_up":
			_time(event.get("at_seconds"))
		elif kind not in ("solve", "selection_round"):
			raise CheckpointError("unknown campaign event")
	if set(attempts) != set(outcomes) or indices != set(range(1, len(indices) + 1)):
		raise CheckpointError("incomplete or unmatched campaign attempts")
	for ident, attempt in attempts.items():
		if _time(outcomes[ident]["at_seconds"]) < _time(attempt["at_seconds"]):
			raise CheckpointError("outcome precedes attempt")
		root_id = attempt.get("policy_request_id")
		root = attempts.get(root_id) if isinstance(root_id, str) else None
		if (
			root is None
			or root["kind"] != "query"
			or (attempt["kind"] == "query" and root is not attempt)
			or (
				attempt["kind"] == "proof"
				and (
					root["index"] >= attempt["index"]
					or outcomes[root["id"]]["status"] != "challenge"
					or _time(outcomes[root["id"]]["at_seconds"]) > _time(attempt["at_seconds"])
					or any(
						root.get(k) != attempt.get(k) for k in ("example_id", "account", "domain")
					)
				)
			)
		):
			raise CheckpointError("invalid query or proof attribution")
		tokens = outcomes[ident].get("output_tokens")
		if tokens is not None and (type(tokens) is not int or tokens < 0):
			raise CheckpointError("invalid output token count")
	policy_ids: set[str] = set()
	for policy in policies:
		ident = policy.get("id")
		if (
			not isinstance(ident, str)
			or ident not in attempts
			or ident in policy_ids
			or any(
				policy.get(k) != attempts[ident].get(k)
				for k in (
					"policy_request_id",
					"at_seconds",
					"contract",
					"example_id",
					"account",
					"domain",
				)
			)
		):
			raise CheckpointError("ambiguous policy attribution")
		if attributed and (
			policy.get("submission_id") != ident
			or policy.get("root_id") != policy["policy_request_id"]
			or (policy["event"] == "policy" and attempts[ident]["kind"] != "query")
			or (policy["event"] == "campaign_throttle" and outcomes[ident]["status"] == "received")
		):
			raise CheckpointError("ambiguous policy attribution")
		policy_ids.add(ident)
	if attributed and summary.get("attribution_complete") is not True:
		raise CheckpointError("incomplete policy attribution")
	records: dict[str, LedgerRecord] = {}
	try:
		for row in _lines(data["ledger"]):
			record = LedgerRecord.model_validate(row)
			if record.attempt_id in records or any(
				r.example_id == record.example_id for r in records.values()
			):
				raise CheckpointError("duplicate ledger attempt or example")
			records[record.attempt_id] = record
	except ValueError as error:
		raise CheckpointError(f"invalid ledger: {error}") from error
	received = {ident for ident, outcome in outcomes.items() if outcome["status"] == "received"}
	if set(records) != received or any(
		attempts[i].get("example_id") != records[i].example_id for i in received
	):
		raise CheckpointError("successful receipts and ledger records do not map uniquely")
	if summary.get("ledger_records") != len(records):
		raise CheckpointError("ledger count mismatch")
	ordered = sorted(received, key=lambda i: (_time(outcomes[i]["at_seconds"]), i))
	interventions: dict[str, Any] = {}
	for name in (*GENERAL_INTERVENTIONS, *(ATTRIBUTED_INTERVENTIONS if attributed else ())):
		candidates: list[tuple[float, float, str, str]] = []
		for ident, outcome in outcomes.items():
			if (
				_outcome_matches(name, outcome)
				if attributed
				else (name == "first_restriction" and outcome["status"] in ("challenge", "blocked"))
				or (name == "first_refusal" and outcome["status"] == "blocked")
			):
				candidates.append(
					(
						_time(attempts[ident]["at_seconds"]),
						_time(outcome["at_seconds"]),
						"client outcome",
						_outcome_cause(outcome),
					)
				)
		for policy in policies:
			level, action = policy.get("level"), policy.get("action")
			match = (
				_policy_matches(name, policy)
				if attributed
				else (name == "first_alert" and type(level) is int and level >= 1)
				or (
					name == "first_restriction"
					and (action in ("challenge", "refuse") or (type(level) is int and level >= 2))
				)
				or (name == "first_refusal" and action == "refuse")
			)
			if match:
				ident = policy["id"]
				candidates.append(
					(
						_time(attempts[ident]["at_seconds"]),
						_time(outcomes[ident]["at_seconds"]),
						"policy attribution and client outcome",
						_policy_cause(policy) if attributed else "policy",
					)
				)
		if candidates:
			_, upper, source, _ = min(candidates, key=lambda c: c[1])
			lower = min(c[0] for c in candidates)
			interventions[name] = {
				# Every cause whose submission could be the first inside this interval.
				**(
					{"causes": sorted({c[3] for c in candidates if c[0] <= upper})}
					if attributed
					else {}
				),
				"status": "reached",
				"lower_seconds": lower,
				"observed_seconds": upper,
				"source": source,
				"uncertainty": (
					"earliest qualifying attempt start to earliest qualifying client outcome; "
					"bounds cover attributed events only; "
					"earlier unobserved enforcement is possible"
				),
			}
		else:
			interventions[name] = {
				"status": "unknown",
				"lower_seconds": None,
				"observed_seconds": None,
				"source": "available events",
				"uncertainty": "policy attribution may be incomplete",
			}
	try:
		output.mkdir(parents=True, exist_ok=False)
	except OSError as error:
		raise CheckpointError(f"cannot create new output: {error}") from error
	checkpoints: list[dict[str, Any]] = []

	def snapshot(
		label: str, kind: str, cutoff: float | None, reached: bool, chosen: list[str]
	) -> None:
		path = f"{label}.jsonl" if reached else None
		content = b"".join(_canonical(records[i].model_dump(mode="json")) + b"\n" for i in chosen)
		if path is not None:
			(output / path).write_bytes(content)
		checkpoints.append(
			{
				"id": label,
				"kind": kind,
				"reached": reached,
				"cutoff_seconds": cutoff,
				"ledger_file": path,
				"ledger_sha256": _hash(content) if reached else None,
				"example_ids": [records[i].example_id for i in chosen],
				"record_count": len(chosen),
				"response_count": len(chosen) if reached else None,
				"cost": cost(cutoff) if reached and cutoff is not None else None,
			}
		)

	def cost(cutoff: float) -> dict[str, Any]:
		started = [a for a in attempts.values() if _time(a["at_seconds"]) <= cutoff]
		completed = [o for o in outcomes.values() if _time(o["at_seconds"]) <= cutoff]
		released = [o for o in completed if o["status"] == "received"]
		attributed_cost = (
			{
				"restriction_wait_seconds": sum(
					_time(w["seconds"]) for w in waits if _time(w["at_seconds"]) <= cutoff
				)
			}
			if attributed
			else {}
		)
		return {
			**attributed_cost,
			"attempts_started": len(started),
			"query_attempts_started": sum(a["kind"] == "query" for a in started),
			"proof_attempts_started": sum(a["kind"] == "proof" for a in started),
			"attempts_completed": len(completed),
			"outcomes": {
				status: sum(o["status"] == status for o in completed)
				for status in (*OUTCOME_STATUSES, *(("restricted",) if attributed else ()))
			},
			"known_released_output_tokens": sum(o.get("output_tokens") or 0 for o in released),
			"unknown_output_token_receipts": sum(o.get("output_tokens") is None for o in released),
		}

	for n in response_counts:
		reached = n <= len(ordered)
		cutoff = (
			0.0 if n == 0 else _time(outcomes[ordered[n - 1]]["at_seconds"]) if reached else None
		)
		chosen = [
			i
			for i in ordered
			if cutoff is not None and n > 0 and _time(outcomes[i]["at_seconds"]) <= cutoff
		]
		snapshot(f"response-{n}", "response_count", cutoff, reached, chosen)
	for name, intervention in interventions.items():
		if intervention["status"] == "reached":
			# An attributed family with the bounds of a general one needs no second student.
			same = next(
				(
					general
					for general in GENERAL_INTERVENTIONS
					if name in ATTRIBUTED_INTERVENTIONS
					and interventions[general]["lower_seconds"] == intervention["lower_seconds"]
					and interventions[general]["observed_seconds"]
					== intervention["observed_seconds"]
				),
				None,
			)
			if same is not None:
				intervention["checkpoints_from"] = same
				continue
			for bound, key in (("lower", "lower_seconds"), ("observed", "observed_seconds")):
				cutoff = intervention[key]
				snapshot(
					f"{name}-{bound}",
					f"intervention_{bound}",
					cutoff,
					True,
					[i for i in ordered if _time(outcomes[i]["at_seconds"]) < cutoff]
					if bound == "lower"
					else [i for i in ordered if _time(outcomes[i]["at_seconds"]) <= cutoff],
				)
	snapshot(
		"end",
		"end",
		max((_time(o["at_seconds"]) for o in outcomes.values()), default=0.0),
		True,
		ordered,
	)
	manifest: dict[str, Any] = {
		"version": "campaign-checkpoints-v2" if attributed else "campaign-checkpoints-v1",
		"inputs": {
			key: {"path": str(campaign / names[key]), "sha256": _hash(value)}
			for key, value in data.items()
		},
		"checkpoints": checkpoints,
		"interventions": interventions,
		"limitations": [
			"Receipt times are client observed; intervention times are intervals, "
			"not server decision timestamps.",
			"End includes late in-flight receipts; token usage may be unknown.",
		],
		"successful_responses": len(ordered),
		"late_inflight_responses": {
			name: sum(
				_time(attempts[i]["at_seconds"])
				< value["observed_seconds"]
				< _time(outcomes[i]["at_seconds"])
				for i in ordered
			)
			if value["status"] == "reached"
			else None
			for name, value in interventions.items()
		},
	}
	if attributed:
		manifest["experiment"] = contract["experiment"]
		manifest["policy_binding"] = binding
		manifest["limitations"].extend(
			[
				"Attributed causes come from harness response codes and decision log rows; "
				"server times are retained in events but are not used as interval bounds.",
				"A campaign restriction is temporary: it does not recall admitted work, and "
				"its cooldown can expire and recur.",
			]
		)
	manifest["digest"] = _hash(_canonical(manifest))
	(output / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
	return manifest


def load_checkpoint_export(output: Path) -> dict[str, Any]:
	"""Verify manifest, source artifacts, and each exported ledger before use."""
	manifest = _json(_read(output / "manifest.json"))
	if not isinstance(manifest, dict) or manifest.get("version") not in (
		"campaign-checkpoints-v1",
		"campaign-checkpoints-v2",
	):
		raise CheckpointError("invalid checkpoint manifest")
	body = dict(manifest)
	digest = body.pop("digest", None)
	if digest != _hash(_canonical(body)):
		raise CheckpointError("manifest digest mismatch")
	try:
		required = {"contract", "summary", "events", "ledger"}
		if manifest["version"] == "campaign-checkpoints-v2":
			required.add("decisions")
			if manifest["experiment"]["campaign"] is not None:
				required.add("campaign_config")
		if not isinstance(manifest["inputs"], dict) or set(manifest["inputs"]) != required:
			raise CheckpointError("incomplete checkpoint sources")
		for item in manifest["inputs"].values():
			if _hash(_read(Path(item["path"]))) != item["sha256"]:
				raise CheckpointError("source artifact digest mismatch")
		seen: set[str] = set()
		if not manifest["checkpoints"]:
			raise CheckpointError("missing checkpoints")
		for checkpoint in manifest["checkpoints"]:
			ident = checkpoint["id"]
			if (
				not isinstance(ident, str)
				or not ident
				or ident in seen
				or not all(c.isalnum() or c in "-_" for c in ident)
			):
				raise CheckpointError("invalid checkpoint ID")
			seen.add(ident)
			if (
				type(checkpoint["reached"]) is not bool
				or type(checkpoint["record_count"]) is not int
				or checkpoint["record_count"] < 0
			):
				raise CheckpointError("invalid checkpoint reachability or count")
			path = checkpoint["ledger_file"]
			if checkpoint["reached"]:
				_time(checkpoint["cutoff_seconds"])
				if (
					not isinstance(path, str)
					or Path(path).is_absolute()
					or not (output / path).resolve().is_relative_to(output.resolve())
				):
					raise CheckpointError("checkpoint path escapes export")
				content = _read(output / path)
				if _hash(content) != checkpoint["ledger_sha256"]:
					raise CheckpointError("checkpoint ledger digest mismatch")
				rows = [LedgerRecord.model_validate(row) for row in _lines(content)]
				if [r.example_id for r in rows] != checkpoint["example_ids"] or len(
					rows
				) != checkpoint["record_count"]:
					raise CheckpointError("checkpoint ledger contents mismatch")
			elif (
				path is not None
				or checkpoint["cutoff_seconds"] is not None
				or checkpoint["ledger_sha256"] is not None
				or checkpoint["record_count"] != 0
				or checkpoint["example_ids"]
			):
				raise CheckpointError("invalid unreached checkpoint")
	except (KeyError, TypeError, ValueError) as error:
		if isinstance(error, CheckpointError):
			raise
		raise CheckpointError(f"invalid checkpoint manifest: {error}") from error
	return manifest
