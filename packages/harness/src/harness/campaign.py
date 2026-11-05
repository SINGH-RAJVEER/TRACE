"""Experimental, unqualified per-principal campaign evidence."""

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from numbers import Real
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.store import Transaction


class CampaignError(ValueError):
	pass


class CampaignConfig(BaseModel):
	model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)
	version: Literal["campaign-v1"] = "campaign-v1"
	namespace: str = Field(min_length=1, max_length=256)
	window_size: int = Field(default=128, ge=1, le=512)
	min_samples: int = Field(default=16, ge=1, le=512)
	max_text_chars: int = Field(default=4096, ge=1, le=65536)
	window_seconds: float = Field(default=86400, gt=0, le=31536000)
	cooldown_seconds: float = Field(default=300, gt=0, le=31536000)
	threshold: float = Field(default=0.8, ge=0, le=1)
	mode: Literal["observe", "throttle"] = "observe"

	@model_validator(mode="after")
	def validate_bounds(self) -> Self:
		if self.min_samples > self.window_size or not self.namespace.strip():
			raise ValueError("invalid campaign bounds or namespace")
		return self

	@property
	def digest(self) -> str:
		return hashlib.sha256(
			json.dumps(
				self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
			).encode()
		).hexdigest()


def load_campaign_config(path: Path) -> CampaignConfig:
	try:
		return CampaignConfig.model_validate_json(path.read_text())
	except (ValueError, OSError) as exc:
		raise CampaignError(str(exc)) from exc


def campaign_key(config: CampaignConfig, principal: str) -> str:
	"""Organization metadata cannot reset an authenticated principal's history."""
	return hashlib.sha256(json.dumps([config.namespace, principal]).encode()).hexdigest()


def campaign_cooldown(
	tx: Transaction, config: CampaignConfig, principal: str, now: float
) -> float | None:
	"""Read the authoritative cooldown without advancing or duplicating state."""
	state = tx.campaign(campaign_key(config, principal))
	if state is None or state["until"] is None:
		return None
	return float(state["until"]) if max(now, state["last"]) < state["until"] else None


def observe_campaign(
	tx: Transaction,
	config: CampaignConfig,
	*,
	principal: str,
	organization: str | None,
	request_id: str,
	text: str,
	now: float,
	domain: str,
	embedding: Sequence[float] | None = None,
) -> dict[str, Any]:
	if any(
		not isinstance(v, str) or not v.strip() or len(v) > 1024
		for v in (principal, request_id, domain)
	) or (
		organization is not None
		and (
			not isinstance(organization, str)
			or not organization.strip()
			or len(organization) > 1024
		)
	):
		raise CampaignError("invalid identity or domain")
	if (
		not isinstance(text, str)
		or not text.strip()
		or len(text) > config.max_text_chars
		or isinstance(now, bool)
		or not isinstance(now, (int, float))
		or not math.isfinite(now)
	):
		raise CampaignError("invalid text or time")
	vector = None
	if embedding is not None:
		try:
			if (
				isinstance(embedding, str | bytes)
				or not 1 <= len(embedding) <= 2048
				or any(isinstance(x, bool) or not isinstance(x, Real) for x in embedding)
			):
				raise CampaignError("invalid embedding")
			vector = [float(x) for x in embedding]
		except TypeError, ValueError:
			raise CampaignError("invalid embedding") from None
		if not 1 <= len(vector) <= 2048 or not all(math.isfinite(x) for x in vector):
			raise CampaignError("invalid embedding")
		magnitude = max(abs(x) for x in vector)
		if magnitude == 0:
			raise CampaignError("zero embedding has no semantic direction")
		scaled = [x / magnitude for x in vector]
		norm = math.sqrt(sum(x * x for x in scaled))
		vector = [x / norm for x in scaled]
	key = campaign_key(config, principal)
	state = tx.campaign(key) or {"rows": [], "receipts": [], "count": 0, "last": now, "until": None}
	if now < 0:
		raise CampaignError("negative timestamp")
	received_at = now
	# Encoder awaits may reorder concurrent arrivals. Serialize event time at commit
	# without dropping observations or letting a clock rollback revive expired state.
	now = max(now, state["last"])
	fingerprint = hashlib.sha256(
		json.dumps([text, domain, vector], separators=(",", ":"), ensure_ascii=False).encode()
	).hexdigest()
	active = state["until"] is not None and now < state["until"]
	for receipt in state["receipts"]:
		if receipt["id"] == request_id:
			if receipt["fingerprint"] != fingerprint:
				raise CampaignError("request id reused with different content")
			result = dict(receipt["result"])
			result["should_throttle"] = config.mode == "throttle" and active
			result["throttle_until"] = state["until"] if active else None
			if result["should_throttle"]:
				result["status"] = "alert"
				result["reasons"] = ["cooldown_active"]
			elif receipt["result"]["should_throttle"]:
				result["status"] = "observed"
				result["score"] = None
				result["reasons"] = ["expired_retry_no_new_evidence"]
			# A retained retry observes time without adding training evidence. Persist
			# expiry so a subsequently reordered retry cannot resurrect this cooldown.
			state["last"] = now
			if state["until"] is not None and not active:
				state["rows"] = []
				state["until"] = None
			result["evaluated_count"] = state["count"]
			result["retained_count"] = len(state["rows"])
			tx.put_campaign(key, state)
			return result
	# A completed cooldown starts a fresh evidence window. Blocked probing during
	# cooldown cannot perpetually extend it, and old evidence cannot instantly rearm.
	expired = state["until"] is not None and not active
	rows = [] if expired else [r for r in state["rows"] if r["time"] > now - config.window_seconds]
	if vector is not None and any(
		r["vector"] is not None and len(r["vector"]) != len(vector) for r in rows
	):
		raise CampaignError("embedding dimension mismatch")
	words = re.findall(r"\w+", text.lower())[:256]
	template = hashlib.sha256(re.sub(r"\d+", "#", " ".join(words)).encode()).hexdigest()
	tokens = {hashlib.sha256(re.sub(r"\d+", "#", w).encode()).hexdigest()[:16] for w in words}
	followup = bool(re.match(r"\s*(and|also|why|how about|what about|continue)\b", text, re.I))
	row = {
		"time": now,
		"template": template,
		"tokens": sorted(tokens),
		"followup": followup,
		"domain": hashlib.sha256(domain.encode()).hexdigest(),
		"vector": vector,
	}
	rows = (rows + [row])[-config.window_size :]
	n = len(rows)
	counts = Counter(r["template"] for r in rows)
	repetition = (max(counts.values()) - 1) / max(1, n - 1)
	lexical = max(
		(
			len(tokens & set(r["tokens"])) / max(1, len(tokens | set(r["tokens"])))
			for r in rows[:-1]
		),
		default=0.0,
	)
	semantic = None
	if vector is not None:
		similarities = []
		for prior in rows[:-1]:
			other = prior["vector"]
			if other is None:
				continue
			den = math.sqrt(sum(x * x for x in vector)) * math.sqrt(sum(x * x for x in other))
			similarities.append(
				max(0.0, min(1.0, sum(x * y for x, y in zip(vector, other, strict=True)) / den))
				if den
				else 0.0
			)
		if similarities:
			semantic = max(similarities)
	independent = 1 - sum(r["followup"] for r in rows) / n
	# Template repetition and lexical overlap must agree. Semantic similarity,
	# domain breadth and volume are context only, not calibrated intent evidence.
	score = repetition * lexical * independent if n >= config.min_samples else None
	alert = (
		score is not None
		and score >= config.threshold
		and repetition > 0
		and lexical > 0
		and not followup
	)
	until = state["until"] if active else None
	if alert and until is None and config.mode == "throttle":
		until = now + config.cooldown_seconds
	result = {
		"version": config.version,
		"config_sha256": config.digest,
		"qualified": False,
		"status": "warming"
		if score is None
		else ("alert" if alert or until is not None else "observed"),
		"score": score,
		"reasons": (
			["insufficient_samples"]
			if score is None
			else (["repeated_independent_templates_and_lexical_overlap"] if alert else [])
		),
		"components": {
			"received_seconds": received_at,
			"ordering_adjusted": received_at != now,
			"template_repetition": repetition,
			"lexical_similarity": lexical,
			"semantic_similarity": semantic,
			"followup_fraction": 1 - independent,
			"domain_count": len({r["domain"] for r in rows}),
			"window_seconds": config.window_seconds,
			"window_size": config.window_size,
			"first_observed_seconds": rows[0]["time"],
			"last_observed_seconds": now,
		},
		"evaluated_count": state["count"] + 1,
		"retained_count": n,
		"throttle_until": until,
		"should_throttle": config.mode == "throttle" and until is not None,
	}
	receipts = (
		state["receipts"] + [{"id": request_id, "fingerprint": fingerprint, "result": result}]
	)[-config.window_size :]
	tx.put_campaign(
		key,
		{
			**state,
			"rows": rows,
			"receipts": receipts,
			"count": state["count"] + 1,
			"last": now,
			"until": until,
		},
	)
	return result
