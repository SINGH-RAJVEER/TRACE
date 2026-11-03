"""Finite-horizon calibration of whole-campaign maximum scores."""

import hashlib
import json
import math
from bisect import bisect_left
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.calibration import binomial_upper


class PathCalibration(BaseModel):
	"""Frozen calibration for one declared campaign stratum and horizon."""

	model_config = ConfigDict(frozen=True, extra="forbid")

	version: Literal["path-max-v1"] = "path-max-v1"
	alpha: float = Field(gt=0, lt=1, allow_inf_nan=False)
	horizon: int = Field(gt=0, strict=True)
	maxima: tuple[float, ...]
	n: int = Field(ge=0, strict=True)
	stratum: str = Field(min_length=1)
	input_hash: str
	abstained: bool
	abstention_reason: str | None

	@model_validator(mode="after")
	def validate_artifact(self) -> PathCalibration:
		if len(self.maxima) != self.n or any(not math.isfinite(x) for x in self.maxima):
			raise ValueError("maxima must contain n finite scores")
		if tuple(sorted(self.maxima)) != self.maxima:
			raise ValueError("maxima must be sorted")
		if not self.stratum.strip():
			raise ValueError("stratum must not be blank")
		if self.abstained != (1 / (self.n + 1) > self.alpha):
			raise ValueError("abstention must match calibration resolution")
		if self.abstained != (self.abstention_reason is not None):
			raise ValueError("an abstention requires a reason")
		return self


def fit_path_calibration(
	maxima_by_campaign: Iterable[float], *, alpha: float, horizon: int, stratum: str
) -> PathCalibration:
	"""Fit one stratum from one maximum per benign calibration campaign."""
	maxima = tuple(sorted(float(value) for value in maxima_by_campaign))
	if any(not math.isfinite(value) for value in maxima):
		raise ValueError("campaign maxima must be finite")
	if not math.isfinite(alpha) or not 0 < alpha < 1:
		raise ValueError("alpha must be finite and in (0, 1)")
	if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
		raise ValueError("horizon must be a positive integer")
	if not isinstance(stratum, str) or not stratum.strip():
		raise ValueError("stratum must not be blank")
	n = len(maxima)
	abstained = 1 / (n + 1) > alpha
	payload = json.dumps(
		{"alpha": alpha, "horizon": horizon, "stratum": stratum, "maxima": maxima},
		sort_keys=True,
		separators=(",", ":"),
		allow_nan=False,
	).encode()
	return PathCalibration(
		alpha=alpha,
		horizon=horizon,
		maxima=maxima,
		n=n,
		stratum=stratum,
		input_hash=hashlib.sha256(payload).hexdigest(),
		abstained=abstained,
		abstention_reason="insufficient campaigns for alpha" if abstained else None,
	)


def path_maximum(
	scores: Iterable[Sequence[float]],
	*,
	transform: Callable[[Sequence[float]], float] | None = None,
) -> float:
	"""Take the maximum over looks after a fixed per-look transform, defaulting to max."""
	maximum = -math.inf
	for look in scores:
		if len(look) == 0 or any(not math.isfinite(value) for value in look):
			raise ValueError("each look must contain finite scores")
		value = float(transform(look) if transform is not None else max(look))
		if not math.isfinite(value):
			raise ValueError("transformed score must be finite")
		maximum = max(maximum, value)
	if maximum == -math.inf:
		raise ValueError("a campaign must contain at least one look")
	return maximum


@dataclass(frozen=True)
class PathDecision:
	"""Result of one observation or an attempt after the horizon."""

	decision: Literal["continue", "stop", "abstain", "horizon expired"]
	p_value: float | None
	stopped: bool
	look_count: int
	stop_look: int | None


@dataclass
class PathState:
	"""State for one campaign, permanently bound to its calibration."""

	_calibration: PathCalibration
	running_max: float | None = field(default=None, init=False)
	look_count: int = field(default=0, init=False)
	stopped: bool = field(default=False, init=False)
	stop_look: int | None = field(default=None, init=False)
	_p_value: float | None = field(default=None, init=False, repr=False)

	@property
	def calibration(self) -> PathCalibration:
		return self._calibration

	@property
	def abstained(self) -> bool:
		return self._calibration.abstained

	def observe(self, score: float) -> PathDecision:
		"""Score a look, preserving a stop and rejecting looks beyond the horizon."""
		if self.look_count >= self.calibration.horizon:
			return self._result("horizon expired")
		if self.stopped:
			return self._result("stop")
		if not math.isfinite(score):
			raise ValueError("score must be finite")
		self.look_count += 1
		self.running_max = score if self.running_max is None else max(self.running_max, score)
		if self.abstained:
			return self._result("abstain")
		maxima = self.calibration.maxima
		self._p_value = (1 + len(maxima) - bisect_left(maxima, self.running_max)) / (
			len(maxima) + 1
		)
		if self._p_value <= self.calibration.alpha:
			self.stopped = True
			self.stop_look = self.look_count
			return self._result("stop")
		return self._result("continue")

	def _result(
		self, decision: Literal["continue", "stop", "abstain", "horizon expired"]
	) -> PathDecision:
		return PathDecision(decision, self._p_value, self.stopped, self.look_count, self.stop_look)


def period_alpha(alpha_total: float, j: int) -> float:
	"""Allocate a lifetime error budget to one-based period j."""
	if not math.isfinite(alpha_total) or not 0 < alpha_total < 1:
		raise ValueError("alpha_total must be finite and in (0, 1)")
	if isinstance(j, bool) or not isinstance(j, int) or j < 1:
		raise ValueError("period must be a positive integer")
	return alpha_total * (6 / math.pi**2) / j / j


def audit_upper_bound(false_stops: int, campaigns: int, confidence: float = 0.95) -> float:
	"""Return the one-sided Clopper-Pearson upper bound for an independent audit."""
	if not math.isfinite(confidence) or not 0 < confidence < 1:
		raise ValueError("confidence must be finite and in (0, 1)")
	return binomial_upper(false_stops, campaigns, confidence)
