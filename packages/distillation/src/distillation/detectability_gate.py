"""Pre-campaign detectability gate: replay a frozen attack offline before a protected arm runs.

A protected campaign is worth running only if some declared MMD arm escalates the frozen
attacker schedule beyond held-out benign accounts. The gate replays both through the same
calibration, detector serialization and policy code the gateway serves, on CPU, and never
writes into the calibration, the schedule or a run directory. Passing it is a go/no-go check,
not evidence that the defense is effective; it never selects or tunes thresholds.
"""

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from harness.calibration import BenignRequest, Calibration, load_calibration, requests_sha256
from harness.encoder import Encoder
from harness.mmd import Array, WindowScorer
from harness.replay import replay
from harness.serialization import gateway_turn

from distillation.io import read_jsonl
from distillation.schemas import GatewayRequest

GATE_VERSION = "gateway-detectability-v1"
DEFAULT_ACCUMULATORS = ("paper", "instantaneous")
# Outside the default calibration directory, artifacts/harness, which must stay frozen.
DEFAULT_EMBEDDING_CACHE = Path("artifacts/embedding-cache")
# Encoder counters change with every call and cache hit; they never identify the embedding.
_COUNTERS = {"encoded", "truncated"}


class UndetectedError(ValueError):
	"""No declared MMD arm escalates the attacker beyond held-out benign accounts."""

	def __init__(self, message: str, report: dict[str, Any]) -> None:
		super().__init__(message)
		self.report = report


def _sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


class CachedEncoder:
	"""Content-addressed embeddings on disk, keyed by encoder identity and the exact texts.

	The wrapped encoder is built only on a miss, so a fully cached gate never loads a model.
	Files are written once through a temporary name and never overwritten.
	"""

	def __init__(
		self, directory: Path, identity: dict[str, Any], load: Callable[[], Encoder]
	) -> None:
		self.directory = directory
		self._identity = {k: v for k, v in identity.items() if k not in _COUNTERS}
		self._load = load
		self._encoder: Encoder | None = None
		self.hits = 0
		self.misses = 0

	def _path(self, texts: list[str]) -> Path:
		digest = hashlib.sha256(json.dumps(self._identity, sort_keys=True).encode())
		for text in texts:
			digest.update(hashlib.sha256(text.encode()).digest())
		return self.directory / f"{digest.hexdigest()}.npy"

	def encode(self, texts: list[str]) -> Array:
		path = self._path(texts)
		if path.exists():
			self.hits += 1
			rows: Array = np.load(path)
			return rows
		self.misses += 1
		if self._encoder is None:
			self._encoder = self._load()
		rows = self._encoder.encode(texts)
		self.directory.mkdir(parents=True, exist_ok=True)
		handle, temporary = tempfile.mkstemp(dir=self.directory, suffix=".npy.tmp")
		try:
			with os.fdopen(handle, "wb") as target:
				np.save(target, rows)
			os.replace(temporary, path)
		except BaseException:
			Path(temporary).unlink(missing_ok=True)
			raise
		return rows

	def identity(self) -> dict[str, Any]:
		return dict(self._identity)


def bge_cache(directory: Path, calibration: Calibration) -> CachedEncoder:
	"""The calibration's own BGE model and revision, on CPU, behind the embedding cache."""
	recorded = calibration.encoder
	identity = {
		"model": recorded.get("model"),
		"revision": recorded.get("revision"),
		"max_tokens": recorded.get("max_tokens", 512),
		"pooling": recorded.get("pooling", "cls"),
		"normalized": recorded.get("normalized", True),
		"device": "cpu",
	}

	def load() -> Encoder:
		from harness.encoder import BGEEncoder

		return BGEEncoder(
			model=identity["model"],
			revision=identity["revision"],
			max_tokens=identity["max_tokens"],
			device="cpu",
		)

	return CachedEncoder(directory, identity, load)


def schedule_requests(input_dir: Path) -> tuple[list[BenignRequest], dict[str, Any]]:
	"""The schedule as detector input, in the order and within the budget collect uses."""
	manifest = json.loads((input_dir / "manifest.json").read_text())
	path = input_dir / "requests.jsonl"
	if _sha256(path) != manifest["sha256"][path.name]:
		raise ValueError("gateway schedule hash does not match its manifest")
	# Collect sends each account's rows in time order and stops at the global budget.
	rows = sorted(read_jsonl(path, GatewayRequest), key=lambda row: row.time)
	rows = rows[: manifest["attempt_budget"]]
	return [
		BenignRequest(
			account=row.account,
			text=gateway_turn(
				row.domain,
				next(m.content for m in reversed(row.prompt.prompt) if m.role == "user"),
			),
			time=row.time,
		)
		for row in rows
	], manifest


def _read_requests(path: Path) -> list[BenignRequest]:
	with path.open(encoding="utf-8") as source:
		return [BenignRequest.model_validate_json(line) for line in source if line.strip()]


def _horizons(decisions: Path) -> list[tuple[str, int, int]]:
	"""Each account horizon's account, peak level and attempts."""
	horizons: list[tuple[str, int, int]] = []
	current: dict[str, int] = {}
	with decisions.open(encoding="utf-8") as source:
		for line in source:
			record = json.loads(line)
			if record["attempt"] == 1:
				current[record["account"]] = len(horizons)
				horizons.append((record["account"], 0, 0))
			index = current[record["account"]]
			account, peak, _ = horizons[index]
			horizons[index] = (account, max(peak, record["level"]), record["attempt"])
	return horizons


def _peak_levels(peaks: list[int]) -> dict[str, int]:
	return {f"L{level}": peaks.count(level) for level in range(5)}


def detectability(
	*,
	calibration_dir: Path,
	schedule_dir: Path,
	benign_test: Path,
	accumulators: tuple[str, ...] = DEFAULT_ACCUMULATORS,
	encoder: Encoder | None = None,
	embedding_cache: Path = DEFAULT_EMBEDDING_CACHE,
) -> dict[str, Any]:
	"""Replay the attacker and held-out benign accounts under each declared MMD arm.

	An arm escalates the attacker when the attacker's highest horizon peak is above L0 and
	fewer than `level_targets[0]` of the scored benign horizons reach that level. Every
	defense is compared on scored horizons, the population its thresholds were fitted on.
	Decisions are written to a temporary directory and discarded. The returned report is
	deterministic for the same inputs, so a resumed campaign reproduces its contract.
	"""
	if not accumulators:
		raise ValueError("declare at least one MMD accumulator for the detectability gate")
	cache = embedding_cache.resolve()
	if cache.is_relative_to(calibration_dir.resolve()) or cache.is_relative_to(
		schedule_dir.resolve()
	):
		raise ValueError("the embedding cache must not live inside a frozen artifact")
	calibration, scorer = load_calibration(calibration_dir)
	attack, _ = schedule_requests(schedule_dir)
	benign = _read_requests(benign_test)
	if not benign:
		raise ValueError("the held-out benign file has no requests")
	fitted = {
		role for role, value in calibration.sha256.items() if value == requests_sha256(benign)
	}
	if fitted:
		raise ValueError(f"the held-out benign file is the calibration's {', '.join(fitted)} role")
	if encoder is None:
		cached = bge_cache(embedding_cache, calibration)
	else:
		inner = encoder
		cached = CachedEncoder(embedding_cache, inner.identity(), lambda: inner)
	size = calibration.config.window_size
	target = calibration.config.level_targets[0]
	arms = {}
	with tempfile.TemporaryDirectory(prefix="detectability-") as scratch:
		for accumulator in accumulators:
			arms[accumulator] = _arm(
				calibration, scorer, cached, attack, benign, accumulator, Path(scratch)
			)
	for result in arms.values():
		peak = result["attacker_peak"]
		scored = result["benign_scored_peaks"]
		rate = sum(level >= peak for level in scored) / len(scored) if peak and scored else 1.0
		result["benign_rate_at_attacker_peak"] = rate
		result["escalates_beyond_benign"] = bool(peak) and bool(scored) and rate < target
		result["benign"] = {
			"horizons": result.pop("benign_horizons"),
			"scored_horizons": len(scored),
			"scored_peak_levels": _peak_levels(scored),
		}
		del result["benign_scored_peaks"]
	detected = any(result["escalates_beyond_benign"] for result in arms.values())
	return {
		"version": GATE_VERSION,
		"purpose": "go/no-go check before a protected campaign; not evidence of effectiveness",
		"calibration_sha256": _sha256(calibration_dir / "calibration.json"),
		# The form the gateway logs with every decision, for matching against a decision log.
		"policy_calibration_sha256": hashlib.sha256(
			calibration.model_dump_json().encode()
		).hexdigest(),
		"schedule_sha256": _sha256(schedule_dir / "requests.jsonl"),
		"benign_test_sha256": _sha256(benign_test),
		"encoder": cached.identity(),
		"thresholds": "calibrated",
		"actions": "graduated",
		"window_size": size,
		"level_target": target,
		"attacker_requests": len(attack),
		"benign_requests": len(benign),
		"arms": arms,
		"detected": detected,
	}


def _arm(
	calibration: Calibration,
	scorer: WindowScorer,
	encoder: Encoder,
	attack: list[BenignRequest],
	benign: list[BenignRequest],
	accumulator: str,
	scratch: Path,
) -> dict[str, Any]:
	peaks: dict[str, dict[str, Any]] = {}
	for name, requests in (("attacker", attack), ("benign", benign)):
		output = scratch / f"{accumulator}.{name}.jsonl"
		replay(
			calibration=calibration,
			scorer=scorer,
			requests=requests,
			output=output,
			policy="mmd",
			accumulator=accumulator,
			encoder=encoder,
		)
		peaks[name] = {"horizons": _horizons(output)}
	by_account: dict[str, int] = defaultdict(int)
	for account, peak, _ in peaks["attacker"]["horizons"]:
		by_account[account] = max(by_account[account], peak)
	size = calibration.config.window_size
	benign_horizons = peaks["benign"]["horizons"]
	return {
		"attacker_peak": max(by_account.values(), default=0),
		"attacker_peak_by_account": dict(sorted(by_account.items())),
		"attacker_horizons": len(peaks["attacker"]["horizons"]),
		"benign_horizons": len(benign_horizons),
		"benign_scored_peaks": [peak for _, peak, attempts in benign_horizons if attempts >= size],
	}


def gate(report: dict[str, Any], *, allow_undetected: bool) -> dict[str, Any]:
	"""Refuse an undetected campaign, or record the override and read it as a quota comparison."""
	record = {
		**report,
		"allow_undetected": allow_undetected,
		"reading": "detection" if report["detected"] else "quota comparison",
	}
	if not report["detected"] and not allow_undetected:
		arms = ", ".join(
			f"{name} peak L{arm['attacker_peak']}" for name, arm in report["arms"].items()
		)
		raise UndetectedError(
			"detectability gate: no MMD arm escalates the attacker beyond held-out benign "
			f"accounts ({arms}); the protected campaign would measure a quota comparison. "
			"Pass --allow-undetected to run it and record it as one.",
			record,
		)
	return record


def contract_record(record: dict[str, Any]) -> dict[str, Any]:
	"""The deterministic part of a gate record that binds a campaign contract."""
	return {
		"version": record["version"],
		"calibration_sha256": record["calibration_sha256"],
		"policy_calibration_sha256": record["policy_calibration_sha256"],
		"schedule_sha256": record["schedule_sha256"],
		"benign_test_sha256": record["benign_test_sha256"],
		"accumulators": sorted(record["arms"]),
		"attacker_peak": {name: arm["attacker_peak"] for name, arm in record["arms"].items()},
		"benign_rate_at_attacker_peak": {
			name: arm["benign_rate_at_attacker_peak"] for name, arm in record["arms"].items()
		},
		"detected": record["detected"],
		"allow_undetected": record["allow_undetected"],
		"reading": record["reading"],
	}
