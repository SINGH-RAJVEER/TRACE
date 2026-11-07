"""Request-path account linkage: inferred clusters of principals that share an allowance.

Each request contributes signal values for its principal:

- `declared`: operator-supplied identity attributes from the accounts file, such as a payment
	or device fingerprint. These are the signals that can link rotated or Sybil accounts.
- `network`: the client's network prefix. Only meaningful when the service sees real client
	addresses; loopback addresses are ignored, so local fixtures never link through it.
- `content`: an exact repeat of substantial task content by another principal, such as a
	refused prompt retried from a second account.

Two principals that share at least `min_shared[kind]` distinct values of a kind gain that
kind's weight. A pair's strength is the noisy-OR of its kinds' weights, and pairs at or
above `threshold` join one cluster. With the defaults, behavior alone never links accounts:
content and network together stay below the threshold, and a declared attribute is needed.
Strengths are heuristics, not probabilities, and no false-link rate has been measured.

Inferred clusters are separate from declared organizations. A value shared by more than
`max_value_principals` principals is a hub, such as a reseller's payment instrument or a
public benchmark question, and stops linking anyone new. Clusters stop growing at
`max_cluster_size`. Operators can sever a pair; severing rebuilds the affected cluster from
the remaining evidence. Each value record and cluster is bounded, but the number of value
and pair records grows with distinct long prompts and linked pairs, and no retention or
expiry is built yet. State never holds raw attribute values or prompt text, only
namespaced digests.
"""

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.store import Transaction

Kind = Literal["declared", "network", "content"]
KINDS: tuple[Kind, ...] = ("declared", "network", "content")
# Bounds the per-principal list of recorded values of one kind; older values drop first.
MAX_VALUES_PER_KIND = 512
# Unqualified defaults: behavior alone (content and network) stays below the threshold.
DEFAULT_WEIGHTS: dict[Kind, float] = {"declared": 0.9, "network": 0.4, "content": 0.4}
DEFAULT_MIN_SHARED: dict[Kind, int] = {"declared": 1, "network": 1, "content": 3}


class LinkageError(ValueError):
	pass


class LinkageConfig(BaseModel):
	model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)
	version: Literal["linkage-v1"] = "linkage-v1"
	namespace: str = Field(min_length=1, max_length=256)
	weights: dict[Kind, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
	min_shared: dict[Kind, int] = Field(default_factory=lambda: dict(DEFAULT_MIN_SHARED))
	threshold: float = Field(default=0.8, gt=0, le=1)
	max_value_principals: int = Field(default=32, ge=2, le=1024)
	max_cluster_size: int = Field(default=64, ge=2, le=1024)
	min_content_chars: int = Field(default=40, ge=1, le=65536)
	network_ipv4_prefix: int = Field(default=24, ge=8, le=32)
	network_ipv6_prefix: int = Field(default=48, ge=16, le=128)

	@model_validator(mode="after")
	def validate_signals(self) -> Self:
		if set(self.weights) != set(KINDS) or set(self.min_shared) != set(KINDS):
			raise ValueError("weights and min_shared must name every signal kind")
		if any(not 0 <= w <= 1 for w in self.weights.values()):
			raise ValueError("weights must lie in [0, 1]")
		if any(n < 1 for n in self.min_shared.values()):
			raise ValueError("min_shared must be positive")
		if not self.namespace.strip():
			raise ValueError("namespace must not be blank")
		return self

	@property
	def digest(self) -> str:
		return hashlib.sha256(
			json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
		).hexdigest()


def load_linkage_config(path: Path) -> LinkageConfig:
	try:
		return LinkageConfig.model_validate_json(path.read_text())
	except (ValueError, OSError) as exc:
		raise LinkageError(str(exc)) from exc


@dataclass(frozen=True)
class Signal:
	kind: Kind
	value: str


def declared_signals(attributes: dict[str, str]) -> list[Signal]:
	return [Signal("declared", f"{name}={value}") for name, value in sorted(attributes.items())]


def network_signal(config: LinkageConfig, host: str | None) -> Signal | None:
	"""The client's network prefix, or None for unknown or loopback addresses."""
	if not host or config.weights["network"] == 0:
		return None
	try:
		address = ipaddress.ip_address(host)
	except ValueError:
		return None
	if address.is_loopback or address.is_unspecified:
		return None
	bits = config.network_ipv4_prefix if address.version == 4 else config.network_ipv6_prefix
	return Signal("network", str(ipaddress.ip_network(f"{address}/{bits}", strict=False)))


def content_signal(config: LinkageConfig, text: str) -> Signal | None:
	"""Normalized task content, when long enough that a repeat is unlikely to be chance."""
	if config.weights["content"] == 0:
		return None
	normalized = " ".join(re.findall(r"\w+", text.lower()))
	if len(normalized) < config.min_content_chars:
		return None
	return Signal("content", normalized)


def _value_key(config: LinkageConfig, signal: Signal) -> str:
	encoded = json.dumps([config.namespace, signal.kind, signal.value])
	return "v:" + hashlib.sha256(encoded.encode()).hexdigest()


def _pair(a: str, b: str) -> str:
	left, right = sorted((a, b))
	return json.dumps([left, right], separators=(",", ":"))


def strength(config: LinkageConfig, evidence: dict[str, int]) -> float:
	miss = 1.0
	for kind in KINDS:
		if evidence.get(kind, 0) >= config.min_shared[kind]:
			miss *= 1 - config.weights[kind]
	return 1 - miss


def _principal(tx: Transaction, principal: str) -> dict[str, Any]:
	return tx.linkage("p:" + principal) or {
		"values": {kind: [] for kind in KINDS},
		"cluster": principal,
	}


def members(tx: Transaction, principal: str) -> list[str]:
	"""The principal's inferred cluster, including itself."""
	root = _principal(tx, principal)["cluster"]
	cluster = tx.linkage("c:" + root)
	return list(cluster["members"]) if cluster else [principal]


def _severed(tx: Transaction, a: str, b: str) -> bool:
	return tx.linkage("s:" + _pair(a, b)) is not None


def _union(tx: Transaction, config: LinkageConfig, a: str, b: str) -> str:
	"""Join the clusters of a and b. Returns merged, same or capped."""
	left, right = members(tx, a), members(tx, b)
	if b in left:
		return "same"
	if len(left) + len(right) > config.max_cluster_size:
		return "capped"
	keep, move = (left, right) if len(left) >= len(right) else (right, left)
	root = _principal(tx, keep[0])["cluster"]
	joined = sorted(set(keep) | set(move))
	for name in move:
		record = _principal(tx, name)
		old = record["cluster"]
		record["cluster"] = root
		tx.put_linkage("p:" + name, record)
		if old != root:
			tx.put_linkage("c:" + old, {"members": [], "merged_into": root})
	tx.put_linkage("c:" + root, {"members": joined})
	return "merged"


def observe(
	tx: Transaction, config: LinkageConfig, principal: str, signals: list[Signal], now: float
) -> dict[str, Any]:
	"""Record this request's signal values and link the principal to any qualifying peers.

	Runs inside the admission transaction, so a rolled-back admission leaves no evidence.
	"""
	record = _principal(tx, principal)
	events: list[dict[str, Any]] = []
	for signal in signals:
		key = _value_key(config, signal)
		seen = record["values"][signal.kind]
		if key in seen:
			continue
		value = tx.linkage(key) or {"kind": signal.kind, "principals": [], "hub": False}
		if value["hub"]:
			continue
		if len(value["principals"]) >= config.max_value_principals:
			value["hub"] = True
			tx.put_linkage(key, value)
			events.append({"event": "hub", "kind": signal.kind})
			continue
		peers = [p for p in value["principals"] if p != principal]
		value["principals"].append(principal)
		tx.put_linkage(key, value)
		record["values"][signal.kind] = (seen + [key])[-MAX_VALUES_PER_KIND:]
		tx.put_linkage("p:" + principal, record)
		for peer in peers:
			pair = _pair(principal, peer)
			evidence = tx.linkage("e:" + pair) or {}
			evidence[signal.kind] = evidence.get(signal.kind, 0) + 1
			tx.put_linkage("e:" + pair, evidence)
			if strength(config, evidence) >= config.threshold and not _severed(tx, principal, peer):
				outcome = _union(tx, config, principal, peer)
				if outcome != "same":
					events.append(
						{
							"event": outcome,
							"kinds": sorted(k for k in evidence if evidence[k]),
							"strength": strength(config, evidence),
						}
					)
		record = _principal(tx, principal)
	tx.put_linkage("p:" + principal, record)
	cluster = members(tx, principal)
	return {
		"cluster": cluster_id(config, cluster),
		"size": len(cluster),
		"events": events,
		"time": now,
		"qualified": False,
	}


def cluster_id(config: LinkageConfig, cluster: list[str]) -> str:
	"""A stable public identifier for a cluster's current membership."""
	return hashlib.sha256(json.dumps([config.namespace, sorted(cluster)]).encode()).hexdigest()[:16]


def sever(
	tx: Transaction,
	config: LinkageConfig,
	a: str,
	b: str,
	*,
	operator: str,
	reason: str,
	now: float,
) -> list[list[str]]:
	"""Record an operator's decision that a and b are unrelated, then rebuild their cluster.

	Returns the resulting clusters. Evidence is kept; only the link is suppressed.
	"""
	if a == b or not operator.strip() or not reason.strip():
		raise LinkageError("sever needs two principals, an operator and a reason")
	tx.put_linkage("s:" + _pair(a, b), {"operator": operator, "reason": reason, "time": now})
	component = members(tx, a)
	if b not in component:
		return [component]
	old_root = _principal(tx, a)["cluster"]
	remaining = set(component)
	groups: list[list[str]] = []
	while remaining:
		start = min(remaining)
		group = {start}
		frontier = [start]
		while frontier:
			current = frontier.pop()
			for other in sorted(remaining - group):
				evidence = tx.linkage("e:" + _pair(current, other)) or {}
				if strength(config, evidence) >= config.threshold and not _severed(
					tx, current, other
				):
					group.add(other)
					frontier.append(other)
		remaining -= group
		groups.append(sorted(group))
	tx.put_linkage("c:" + old_root, {"members": [], "severed": True})
	for cluster in groups:
		for name in cluster:
			record = _principal(tx, name)
			record["cluster"] = cluster[0]
			tx.put_linkage("p:" + name, record)
		tx.put_linkage("c:" + cluster[0], {"members": cluster})
	return groups


@dataclass(frozen=True)
class ClusterBudget:
	"""What an inferred cluster may acquire together in any span of `window_seconds`.

	A single unlinked principal is left to the per-principal budgets.
	"""

	window_seconds: float
	requests: int | None = None
	tokens: int | None = None

	def exceeded(self, tx: Transaction, principal: str, reserved_tokens: int, now: float) -> bool:
		cluster = members(tx, principal)
		if len(cluster) < 2:
			return False
		since = now - self.window_seconds
		requests = tokens = 0
		for name in cluster:
			used = tx.rolling_exposure(name, since, "account")
			requests += used.requests
			tokens += used.tokens
		return (self.requests is not None and requests >= self.requests) or (
			self.tokens is not None and tokens + reserved_tokens > self.tokens
		)
