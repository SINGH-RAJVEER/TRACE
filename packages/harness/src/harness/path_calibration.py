"""Finite-horizon calibration of whole-campaign maximum scores."""

import hashlib
import json
import math
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.calibration import binomial_upper

Look = Sequence[float]


class PathScoring(BaseModel):
	"""The frozen per-look score shared by calibration and runtime.

	A look supplies one score per declared stream, in stream order. Its path score is the
	maximum of the fixed positive weight times each stream's score. Fitting and runtime both
	score through this object, so a calibration cannot be applied to differently scored looks.
	"""

	model_config = ConfigDict(frozen=True, extra="forbid")

	name: str = Field(min_length=1)
	streams: tuple[str, ...] = Field(min_length=1)
	weights: tuple[float, ...] | None = None

	@model_validator(mode="after")
	def validate_scoring(self) -> PathScoring:
		if not self.name.strip() or any(not stream.strip() for stream in self.streams):
			raise ValueError("scoring name and streams must not be blank")
		if len(set(self.streams)) != len(self.streams):
			raise ValueError("streams must be unique")
		if self.weights is not None and (
			len(self.weights) != len(self.streams)
			or any(not math.isfinite(w) or w <= 0 for w in self.weights)
		):
			raise ValueError("weights must be one finite positive value per stream")
		return self

	def look(self, scores: Look) -> float:
		"""Score one look; every declared stream must be present and finite."""
		if len(scores) != len(self.streams):
			raise ValueError(f"a look needs one score per stream: {', '.join(self.streams)}")
		if any(not math.isfinite(value) for value in scores):
			raise ValueError("scores must be finite")
		weights = self.weights or (1.0,) * len(self.streams)
		# Adding zero folds -0.0 into 0.0 so equal scores serialize identically.
		score = max(w * float(v) for w, v in zip(weights, scores, strict=True)) + 0.0
		if not math.isfinite(score):
			raise ValueError("weighted score must be finite")
		return score

	def path_maximum(self, path: Iterable[Look], horizon: int) -> float:
		"""The maximum look score over a whole permitted path of at most `horizon` looks."""
		values = [self.look(look) for look in path]
		if not values:
			raise ValueError("a campaign must contain at least one look")
		if len(values) > horizon:
			raise ValueError("a calibration path is longer than the horizon")
		return max(values)


def _input_hash(
	alpha: float, horizon: int, stratum: str, scoring: PathScoring, maxima: Sequence[float]
) -> str:
	payload = json.dumps(
		{
			"alpha": alpha,
			"horizon": horizon,
			"stratum": stratum,
			"scoring": scoring.model_dump(),
			"maxima": list(maxima),
		},
		sort_keys=True,
		separators=(",", ":"),
		allow_nan=False,
	).encode()
	return hashlib.sha256(payload).hexdigest()


class PathCalibration(BaseModel):
	"""Frozen calibration for one declared campaign stratum, scoring and horizon.

	Construction and loading recompute `input_hash`, so an artifact whose scoring, maxima or
	settings were edited after fitting is rejected. The hash cannot show that hand-built
	maxima came from real benign paths; produce calibrations only with fit_path_calibration.
	"""

	model_config = ConfigDict(frozen=True, extra="forbid")

	version: Literal["path-max-v2"] = "path-max-v2"
	scoring: PathScoring
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
		expected = _input_hash(self.alpha, self.horizon, self.stratum, self.scoring, self.maxima)
		if self.input_hash != expected:
			raise ValueError("input hash differs from the calibration's scoring and maxima")
		return self


def fit_path_calibration(
	paths: Iterable[Iterable[Look]],
	*,
	scoring: PathScoring,
	alpha: float,
	horizon: int,
	stratum: str,
) -> PathCalibration:
	"""Fit one stratum from the complete look paths of independent benign campaigns."""
	if not math.isfinite(alpha) or not 0 < alpha < 1:
		raise ValueError("alpha must be finite and in (0, 1)")
	if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
		raise ValueError("horizon must be a positive integer")
	if not isinstance(stratum, str) or not stratum.strip():
		raise ValueError("stratum must not be blank")
	maxima = tuple(sorted(scoring.path_maximum(path, horizon) for path in paths))
	n = len(maxima)
	abstained = 1 / (n + 1) > alpha
	return PathCalibration(
		scoring=scoring,
		alpha=alpha,
		horizon=horizon,
		maxima=maxima,
		n=n,
		stratum=stratum,
		input_hash=_input_hash(alpha, horizon, stratum, scoring, maxima),
		abstained=abstained,
		abstention_reason="insufficient campaigns for alpha" if abstained else None,
	)


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

	def observe(self, scores: Look) -> PathDecision:
		"""Score a look with the calibration's scoring, keeping a stop and the horizon."""
		if self.look_count >= self.calibration.horizon:
			return self._result("horizon expired")
		if self.stopped:
			return self._result("stop")
		score = self.calibration.scoring.look(scores)
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
	for count in (false_stops, campaigns):
		if isinstance(count, bool) or not isinstance(count, int):
			raise ValueError("audit counts must be integers")
	if campaigns < 1 or not 0 <= false_stops <= campaigns:
		raise ValueError("audit needs campaigns >= 1 and 0 <= false_stops <= campaigns")
	if not math.isfinite(confidence) or not 0 < confidence < 1:
		raise ValueError("confidence must be finite and in (0, 1)")
	return binomial_upper(false_stops, campaigns, confidence)
