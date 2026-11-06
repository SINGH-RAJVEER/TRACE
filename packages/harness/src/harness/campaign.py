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

from harness.corpus import shingles
from harness.session_features import answer_only, followup_language
from harness.store import Transaction


class CampaignError(ValueError):
	pass


class SessionConfig(BaseModel):
	"""Frozen uncalibrated combination, explicitly separate from campaign-v2."""

	model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)
	version: Literal["semantic-session-v1"] = "semantic-session-v1"
	encoder_identity: dict[str, Any] | None = None
	semantic_threshold: float = Field(default=0.9, gt=0, le=1)

	@model_validator(mode="after")
	def stable_identity(self) -> Self:
		if self.encoder_identity is not None and (
			not self.encoder_identity or {"encoded", "truncated"} & self.encoder_identity.keys()
		):
			raise ValueError("encoder identity must be a nonempty stable representation")
		return self


class CampaignConfig(BaseModel):
	model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)
	version: Literal["campaign-v2"] = "campaign-v2"
	session: SessionConfig | None = None
	namespace: str = Field(min_length=1, max_length=256)
	window_size: int = Field(default=128, ge=1, le=512)
	min_samples: int = Field(default=16, ge=1, le=512)
	max_text_chars: int = Field(default=4096, ge=1, le=65536)
	window_seconds: float = Field(default=86400, gt=0, le=31536000)
	cooldown_seconds: float = Field(default=300, gt=0, le=31536000)
	threshold: float = Field(default=0.8, ge=0, le=1)
	min_variant_fraction: float = Field(default=0.5, ge=0, lt=1)
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
				self.model_dump(
					mode="json", exclude={"session"} if self.session is None else set()
				),
				sort_keys=True,
				separators=(",", ":"),
				allow_nan=False,
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


def text_identity(text: str) -> tuple[str, str]:
	"""Hash a whole prompt into its template and exact form.

	Prompt length is unbounded, so callers compute this before opening the write transaction.
	"""
	normalized = " ".join(re.findall(r"\w+", text.lower()))
	return (
		hashlib.sha256(re.sub(r"\d+", "#", normalized).encode()).hexdigest(),
		hashlib.sha256(normalized.encode()).hexdigest(),
	)


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
	identity: tuple[str, str] | None = None,
	resampling: bool = False,
) -> dict[str, Any]:
	"""Score bounded `text`; `identity` separates prompts differing after that bound.

	`resampling` marks a request that can return a new output for identical input, so an
	exact repeat of it is new collection rather than a repeated deterministic call.
	"""
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
	if (
		config.session is not None
		and config.session.encoder_identity is None
		and embedding is not None
	):
		raise CampaignError("session embeddings require a bound encoder identity")
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
	# A completed cooldown starts a fresh evidence window. Clearing receipts with it stops
	# requests refused during the cooldown from being replayed later without being scored.
	# Admitted retries stay protected by the API's durable idempotency record.
	if state["until"] is not None and now >= state["until"]:
		state = {**state, "rows": [], "receipts": [], "until": None}
	template, exact = identity if identity is not None else text_identity(text)
	fingerprint = hashlib.sha256(
		json.dumps(
			[text, template, exact, domain, vector, resampling],
			separators=(",", ":"),
			ensure_ascii=False,
		).encode()
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
			# A retained retry observes time without adding training evidence.
			state["last"] = now
			result["evaluated_count"] = state["count"]
			result["retained_count"] = len(state["rows"])
			tx.put_campaign(key, state)
			return result
	# Blocked probing during cooldown cannot perpetually extend it, and old evidence cannot
	# instantly rearm.
	rows = [r for r in state["rows"] if r["time"] > now - config.window_seconds]
	if vector is not None and any(
		r["vector"] is not None and len(r["vector"]) != len(vector) for r in rows
	):
		raise CampaignError("embedding dimension mismatch")
	words = re.findall(r"\w+", text.lower())[:256]
	variant = hashlib.sha256(
		json.dumps([exact, request_id if resampling else None]).encode()
	).hexdigest()
	tokens = {hashlib.sha256(re.sub(r"\d+", "#", w).encode()).hexdigest()[:16] for w in words}
	followup = bool(re.match(r"\s*(and|also|why|how about|what about|continue)\b", text, re.I))
	row: dict[str, Any] = {
		"time": now,
		"template": template,
		"variant": variant,
		"tokens": sorted(tokens),
		"followup": followup,
		"domain": hashlib.sha256(domain.encode()).hexdigest(),
		"vector": vector,
	}
	if config.session is not None:
		row["truncated"] = identity is not None and identity != text_identity(text)
		parts = shingles(text)
		row["shingles"] = sorted(hashlib.sha256(p.encode()).hexdigest()[:16] for p in parts)
		row["answer_only"] = answer_only(text)
		row["contextual"] = followup_language(text)
		row["workflow"] = bool(
			re.search(
				r"\b(edit|revise|rewrite|debug|explain|teach|hint|feedback|proofread|clarify)\b",
				text,
				re.I,
			)
		)
	rows = (rows + [row])[-config.window_size :]
	n = len(rows)
	counts = Counter(r["template"] for r in rows)
	repetition = (max(counts.values()) - 1) / max(1, n - 1)
	modal_template, modal_count = counts.most_common(1)[0]
	variant_fraction = len({r["variant"] for r in rows if r["template"] == modal_template}) / (
		modal_count
	)
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
	# Exact deterministic repeats, such as a health check, stay visible as alerts but add no
	# new supervision, so they cannot start a cooldown alone.
	enforceable = alert and variant_fraction > config.min_variant_fraction
	extra_components: dict[str, Any] = {}
	extra_reasons: list[str] = []
	if config.session is not None:
		# Current-anchor family: O(window * embedding width), no transitive clustering.
		# Missing vectors contribute zero against the full retained-window denominator.
		family = [row]
		semantic_matches = 0
		available = 0
		for prior in rows[:-1]:
			other = prior["vector"]
			if (
				vector is not None
				and other is not None
				and not row["truncated"]
				and not prior["truncated"]
			):
				available += 1
				cosine = sum(x * y for x, y in zip(vector, other, strict=True))
				if cosine >= config.session.semantic_threshold:
					semantic_matches += 1
					family.append(prior)
		semantic_repetition = semantic_matches / max(1, n - 1)
		parts_now = set(row["shingles"])
		overlaps = [
			len(parts_now & set(r["shingles"])) / max(1, len(parts_now | set(r["shingles"])))
			for r in rows[:-1]
		]
		# Bounded current-to-history analogues of offline raw_signals.
		rephrasing = max(overlaps, default=0.0)
		contextual_share = sum(r["contextual"] for r in rows) / n
		workflow_share = sum(r["workflow"] for r in rows) / n
		answer_share = sum(r["answer_only"] for r in rows) / n
		control = 1 - max(contextual_share, workflow_share)
		lexical_evidence = repetition * lexical
		semantic_evidence = semantic_repetition * answer_share
		combined = max(lexical_evidence, semantic_evidence) * control
		score = combined if n >= config.min_samples else None
		alert = score is not None and score >= config.threshold and combined > 0
		semantic_route = semantic_evidence > lexical_evidence
		selected = (
			family if semantic_route else [r for r in rows if r["template"] == modal_template]
		)
		variant_fraction = len({r["variant"] for r in selected}) / len(selected)
		enforceable = alert and variant_fraction > config.min_variant_fraction
		extra_components = {
			"session_version": config.session.version,
			"semantic_repetition": semantic_repetition,
			"truncated_count": sum(r["truncated"] for r in rows),
			"semantic_pairs_available": available,
			"semantic_pairs_missing": n - 1 - available,
			"semantic_missingness": (
				"current_embedding_missing"
				if vector is None
				else "partial_history"
				if available < n - 1
				else "no_prior_samples"
				if n == 1
				else None
			),
			"rephrasing_similarity": rephrasing,
			"template_diversity": len(counts) / n,
			"answer_only_share": answer_share,
			"contextual_language_share": contextual_share,
			"workflow_share": workflow_share,
			"topic_switch_rate": sum(
				a["domain"] != b["domain"] for a, b in zip(rows[:-1], rows[1:], strict=True)
			)
			/ max(1, n - 1),
			"lexical_contribution": lexical_evidence * control,
			"semantic_contribution": semantic_evidence * control,
			"workflow_multiplier": control,
			"selected_route": "semantic" if semantic_route else "lexical",
			"selected_variant_fraction": variant_fraction,
		}
		if available < n - 1 or n == 1:
			extra_reasons.append("semantic_evidence_incomplete")
		if control < 1:
			extra_reasons.append("legitimate_workflow_control")
	until = state["until"] if active else None
	if enforceable and until is None and config.mode == "throttle":
		until = now + config.cooldown_seconds
	result = {
		"version": config.version,
		"config_sha256": config.digest,
		"qualified": False,
		"status": (
			"alert" if alert or until is not None else ("warming" if score is None else "observed")
		),
		"score": score,
		"reasons": (
			(["insufficient_samples"] if score is None else [])
			+ (
				[
					"semantic_session_combination"
					if config.session is not None
					else "repeated_independent_templates_and_lexical_overlap"
				]
				if alert
				else []
			)
			+ (["exact_repeats_not_enforced"] if alert and not enforceable else [])
			+ (["cooldown_active"] if until is not None else [])
			+ extra_reasons
		),
		"components": {
			**extra_components,
			"received_seconds": received_at,
			"ordering_adjusted": received_at != now,
			"template_repetition": repetition,
			"variant_fraction": variant_fraction,
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
