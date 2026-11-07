"""Finite, domain/prefix-matched coverage monitoring, separate from legacy path policy."""

import hashlib
import math
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.logdet import CoverageSpec, LogDetCoverage
from harness.serialization import gateway_turn


class CoverageMonitor(BaseModel):
	model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
	version: Literal["campaign-coverage-v1"] = "campaign-coverage-v1"
	spec: CoverageSpec
	encoder_identity: dict[str, Any]
	serialization: Literal["raw_turn", "gateway_turn"]
	domain: str = Field(min_length=1)
	max_text_chars: int = Field(default=4096, ge=1, le=65536)
	looks: tuple[int, ...] = Field(min_length=1, max_length=512)
	scales: tuple[int, ...] = (8, 32)

	@model_validator(mode="after")
	def check(self) -> Self:
		if self.spec.input_dim > 2048:
			raise ValueError("coverage input width exceeds campaign embedding bound")
		if tuple(sorted(set(self.looks))) != self.looks or self.looks[0] < 1:
			raise ValueError("looks must be unique increasing positive prefixes")
		if not self.scales or len(self.scales) > 8 or any(s < 1 for s in self.scales):
			raise ValueError("declare one to eight positive EWMA scales")
		if not self.encoder_identity or {"encoded", "truncated"} & self.encoder_identity.keys():
			raise ValueError("coverage requires a stable encoder identity")
		return self

	def serialize(self, turn: str) -> str:
		"""The encoder input, identical in calibration and serving."""
		return gateway_turn(self.domain, turn) if self.serialization == "gateway_turn" else turn

	@property
	def digest(self) -> str:
		return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class CoverageCalibration(BaseModel):
	model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
	version: Literal["campaign-coverage-max-v1"] = "campaign-coverage-max-v1"
	monitor: CoverageMonitor
	alpha: float = Field(gt=0, lt=1)
	maxima: tuple[float, ...]

	@model_validator(mode="after")
	def check(self) -> Self:
		if tuple(sorted(self.maxima)) != self.maxima or any(
			not math.isfinite(x) or x < 0 for x in self.maxima
		):
			raise ValueError("maxima must be sorted finite nonnegative values")
		return self

	@property
	def digest(self) -> str:
		return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class CoverageConfig(BaseModel):
	model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
	monitor: CoverageMonitor
	calibration: CoverageCalibration | None = None
	calibration_sha256: str | None = None

	@model_validator(mode="after")
	def check(self) -> Self:
		if self.calibration is None:
			if self.calibration_sha256 is not None:
				raise ValueError("calibration digest without artifact")
		elif (
			self.calibration.monitor.digest != self.monitor.digest
			or self.calibration.digest != self.calibration_sha256
		):
			raise ValueError("coverage calibration digest/monitor mismatch")
		return self


def advance_coverage(
	monitor: CoverageMonitor,
	state: dict[str, Any],
	embedding: Sequence[float] | None,
	complete: bool,
) -> dict[str, Any]:
	"""Advance one domain prefix with bounded state; any missing row poisons this path."""
	count = state.get("count", 0) + 1
	if count > monitor.looks[-1]:
		return {**state, "expired": True}
	state = {**state, "count": count}
	if embedding is None or not complete:
		return {**state, "missing": True}
	if state.get("missing"):
		return state
	tracker = (
		LogDetCoverage.restore(state["sketch"], trusted=True)
		if "sketch" in state
		else LogDetCoverage(monitor.spec.dim, monitor.spec.ridge)
	)
	x = np.asarray([embedding], dtype=np.float64)
	if not np.any(x):
		raise ValueError("zero embedding has no coverage direction")
	gain = tracker.update(monitor.spec.project(x)[0])
	means = [
		gain if count == 1 else old + (gain - old) / scale
		for old, scale in zip(
			state.get("means", [0.0] * len(monitor.scales)), monitor.scales, strict=True
		)
	]
	maximum = state.get("maximum")
	if count in monitor.looks:
		maximum = max([*means, maximum] if maximum is not None else means)
	return {"count": count, "sketch": tracker.snapshot(), "means": means, "maximum": maximum}


def coverage_evidence(config: CoverageConfig, state: dict[str, Any], domain: str) -> dict[str, Any]:
	cal = config.calibration
	reason = (
		"unsupported_domain"
		if domain != config.monitor.domain
		else "missing_embedding_prefix"
		if state.get("missing")
		else "horizon_expired"
		if state.get("expired")
		else "calibration_missing"
		if cal is None
		else "insufficient_campaigns"
		if 1 / (len(cal.maxima) + 1) > cal.alpha
		else "no_scheduled_look"
		if state.get("maximum") is None
		else None
	)
	p = None
	if reason is None and cal is not None:
		p = (1 + len(cal.maxima) - bisect_left(cal.maxima, state["maximum"])) / (
			len(cal.maxima) + 1
		)
	return {
		"version": "campaign-coverage-v1",
		"qualified": False,
		"domain": domain,
		"prefix": state.get("count", 0),
		"running_max": state.get("maximum"),
		"p_value": p,
		"abstention_reason": reason,
		"calibration_sha256": config.calibration_sha256,
		"alert": p is not None and cal is not None and p <= cal.alpha,
	}


def fit_coverage_calibration(
	monitor: CoverageMonitor, paths: Iterable[Iterable[Sequence[float]]], alpha: float
) -> CoverageCalibration:
	"""Require complete benign paths at every declared look and include every scale."""
	maxima = []
	for path in paths:
		state: dict[str, Any] = {}
		for vector in path:
			state = advance_coverage(monitor, state, vector, True)
		if state.get("count") != monitor.looks[-1] or state.get("expired"):
			raise ValueError("benign campaign must contain exactly the declared horizon")
		maxima.append(state["maximum"])
	return CoverageCalibration(monitor=monitor, alpha=alpha, maxima=tuple(sorted(maxima)))
