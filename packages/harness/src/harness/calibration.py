"""Benign-only calibration of reference windows, window p-values, and level thresholds.

Three disjoint benign sets play separate roles (paper Section III-C): fitting sets the kernel
bandwidth and draws the reference windows, score calibration gives the null distribution of
window scores, and policy calibration sets each level's threshold from the maximum statistic
over complete benign horizons. No attack example sets any value here.
"""

import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from harness.accumulators import DEFAULT_ACCUMULATORS, AccumulatorSpec
from harness.encoder import Encoder
from harness.mmd import SCALES, Array, WindowScorer, median_bandwidth
from harness.serialization import SERIALIZATION, Task, detector_text

LEVELS = 4
HORIZON_POLICY = "time-reset-attempt-cap-v1"
# Pooled-mixture windows ranking below the per-account null median at this share or more mean
# the upper-tail score cannot flag traffic that resembles the benign mixture.
MIXTURE_WARNING_SHARE = 0.9
MIXTURE_WINDOWS = 50


class HarnessConfig(BaseModel):
	"""Protocol settings. Defaults are the paper's proposed values, not tuned results."""

	model_config = ConfigDict(extra="forbid")

	# Selects the detector serialization; "text" embeds whole user turns.
	task: Task = "text"
	window_size: int = Field(default=100, ge=2)
	reference_windows: int = Field(default=20, ge=1)
	# Score-calibration windows start every `null_stride` requests within an account, so the
	# null resolves smaller p-values from a pool that bounds the number of benign accounts.
	# Overlapping windows are dependent; counts record the independent windows too. None uses
	# a quarter window; window_size reproduces non-overlapping null windows.
	null_stride: int | None = Field(default=None, ge=1)
	# Score each per-account window over its distinct prompts, collapsing exact repeats, so a
	# client that resends prompts is not scored as concentrated traffic. Calibrations saved
	# before this setting existed load with False, matching how their null was built. The
	# aggregate monitor always scores every request.
	distinct_prompts: bool = True
	bandwidth_sample: int = Field(default=2000, ge=2)
	scales: tuple[float, ...] = SCALES
	# Nominal benign targets for reaching L1 to L4 at least once in a horizon.
	level_targets: tuple[float, float, float, float] = (0.10, 0.05, 0.02, 0.01)
	horizon_seconds: float = Field(default=86_400, gt=0)
	horizon_attempts: int = Field(default=10_000, ge=1)
	accumulators: dict[str, AccumulatorSpec] = Field(
		default_factory=lambda: dict(DEFAULT_ACCUMULATORS)
	)
	# The aggregate monitor scores all traffic in windows of this size and only raises alerts.
	aggregate_window_size: int = Field(default=1500, ge=2)
	aggregate_reference_windows: int = Field(default=20, ge=1)
	# Share of benign aggregate windows allowed to alert, as the reproduction's 95th percentile.
	aggregate_alert_rate: float = Field(default=0.05, gt=0, lt=1)
	seed: int = 42
	aggregate_enabled: bool = True
	mixture_control_required: bool = False

	@field_validator("level_targets")
	@classmethod
	def valid_targets(
		cls, values: tuple[float, float, float, float]
	) -> tuple[float, float, float, float]:
		if any(not 0 < value < 1 for value in values) or any(
			a < b for a, b in zip(values, values[1:], strict=False)
		):
			raise ValueError("level targets must be in (0, 1) and nonincreasing")
		return values


class BenignRequest(BaseModel):
	"""One benign request. Requests of an account appear in order; time is in seconds."""

	model_config = ConfigDict(extra="forbid")

	account: str = Field(min_length=1)
	text: str
	time: float | None = None
	example_id: str | None = None
	source_group: str | None = None
	client_type: str | None = None
	workload_version: str | None = None


class LevelThresholds(BaseModel):
	thresholds: list[float]
	achieved: list[float]
	# Horizons fitted: those with at least one complete window, for every defense.
	units: int
	# One-sided 95% Clopper-Pearson upper bounds on each achieved rate over `units`.
	upper95: list[float] = Field(default_factory=list)
	# Absent in legacy artifacts. These counts distinguish detector evidence from quota.
	level_results: list[dict[str, Any]] | None = None


class AggregateCalibration(BaseModel):
	"""The aggregate monitor's alert threshold, fitted on interleaved benign windows."""

	window_size: int
	threshold: float
	achieved: float
	null_windows: int
	interleaving: Literal["time", "round_robin"]


class MixtureControl(BaseModel):
	"""Score-calibration requests shuffled across accounts, scored as one broad client.

	`share_below_null_median` is the share of these pooled windows whose upper-tail p-value
	exceeds 0.5, so they score below the median benign account window. Exchangeable benign
	accounts give about half; a share near one means a broad client looks more ordinary than
	ordinary users, and the upper-tail test cannot flag same-distribution extraction.
	"""

	windows: int
	median_p_value: float
	min_p_value: float
	share_below_null_median: float


class Calibration(BaseModel):
	"""The frozen calibration artifact. Reference embeddings are saved beside it."""

	config: HarnessConfig
	encoder: dict[str, Any]
	# Detector serialization version; None in artifacts from before it was recorded.
	serialization: str | None = None
	# Missing in legacy artifacts, which must not be reused under new reset semantics.
	horizon_policy: str | None = None
	sigma: float
	calibration_scores: list[float]
	thresholds: dict[str, LevelThresholds]
	# Theoretical log(1/target) thresholds; valid only for e_process under its assumptions.
	ville_thresholds: list[float]
	# Block-only baselines fitted to the L4 target on the same scored horizons: "quota" in
	# requests per horizon and, when every request has a time, "rate_per_minute".
	baselines: dict[str, LevelThresholds] = Field(default_factory=dict)
	counts: dict[str, int]
	sha256: dict[str, str] = Field(default_factory=dict)
	aggregate: AggregateCalibration | None = None
	aggregate_skipped: str | None = None
	# Missing in artifacts calibrated before the control existed; they report no warning.
	mixture_control: MixtureControl | None = None
	input_validation: dict[str, Any] | None = None
	_aggregate_references: Array | None = PrivateAttr(default=None)

	def _level_report(
		self, level: int, accumulators: tuple[str, ...], scored: int, null_windows: int
	) -> dict[str, Any]:
		"""One level's target, the sample sizes that resolve it, and its fitted evidence."""
		target = self.config.level_targets[level - 1]
		minimum_policy = math.ceil(1 / target)
		results: dict[str, dict[str, Any] | None] = {}
		for name in accumulators:
			fitted = self.thresholds.get(name)
			level_results = fitted.level_results if fitted is not None else None
			results[name] = (
				level_results[level - 1]
				if level_results is not None and len(level_results) >= level
				else None
			)
		legacy = any(
			name not in self.thresholds or self.thresholds[name].level_results is None
			for name in accumulators
		)
		return {
			"target": target,
			"minimum_policy_horizons": minimum_policy,
			"minimum_score_windows": minimum_policy - 1,
			"resolution_sufficient": scored >= minimum_policy
			and (not accumulators or null_windows >= minimum_policy - 1),
			"evidence_version": "legacy-inferred" if legacy else "level-results-v1",
			"results": results,
		}

	def adequacy(
		self,
		*,
		accumulators: tuple[str, ...] = (),
		baselines: tuple[str, ...] = (),
		allow_resolution_limited: bool = False,
		levels: tuple[int, ...] = (1, 2, 3, 4),
	) -> dict[str, Any]:
		"""Minimum resolution for selected study defenses, not a false-positive guarantee.

		Overlapping score windows do not count toward the minimum null size. Scored policy
		horizons must permit at least one exceedance at the smallest target, and selected MMD
		levels must remain distinct. An explicit resolution exception turns sample-size limits
		and small-sample level ties into warnings; invalid fits still fail closed. Legacy
		artifacts without scored counts fail closed.
		"""
		if not levels or any(level not in range(1, LEVELS + 1) for level in levels):
			raise ValueError("levels must select L1 through L4")
		target = min(self.config.level_targets[level - 1] for level in levels)
		minimum_policy = math.ceil(1 / target)
		minimum_score = minimum_policy - 1
		null_windows = self.counts.get("score_calibration_independent_windows", 0)
		scored = self.counts.get("policy_calibration_scored_horizons", 0)
		reasons: list[str] = []
		if self.input_validation is not None and not self.input_validation.get("passed"):
			reasons.append("workload input validation failed or was overridden")
		resolution_reasons: list[str] = []
		if accumulators:
			if null_windows < minimum_score:
				resolution_reasons.append(
					f"{null_windows} non-overlapping score windows; need {minimum_score}"
				)
			if scored < minimum_policy:
				resolution_reasons.append(f"{scored} scored policy horizons; need {minimum_policy}")
			for name in accumulators:
				fitted = self.thresholds.get(name)
				if fitted is None or fitted.units != scored:
					reasons.append(f"{name}: missing thresholds or incompatible scored denominator")
				elif (
					len(fitted.thresholds) != LEVELS
					or any(not math.isfinite(value) for value in fitted.thresholds)
					or any(
						a > b
						for a, b in zip(fitted.thresholds, fitted.thresholds[1:], strict=False)
					)
				):
					reasons.append(f"{name}: level thresholds are collapsed or invalid")
				elif len({fitted.thresholds[level - 1] for level in levels}) != len(set(levels)):
					if scored < minimum_policy or null_windows < minimum_score:
						resolution_reasons.append(f"{name}: level thresholds are collapsed")
					else:
						reasons.append(f"{name}: level thresholds are collapsed or invalid")
				if fitted is not None and fitted.level_results is not None:
					for level in levels:
						if len(fitted.level_results) < level:
							reasons.append(f"{name}: L{level} evidence missing")
						elif fitted.level_results[level - 1].get("scored") != scored:
							reasons.append(
								f"{name}: L{level} evidence has incompatible denominator"
							)
						else:
							result = fitted.level_results[level - 1]
							if (
								len(fitted.thresholds) < level
								or result.get("target") != self.config.level_targets[level - 1]
								or result.get("threshold") != fitted.thresholds[level - 1]
								or any(
									key not in result
									for key in ("reached", "capped", "upper95", "censored")
								)
							):
								reasons.append(f"{name}: L{level} evidence missing or incompatible")
							elif result.get("achieved", 1.0) > self.config.level_targets[level - 1]:
								reasons.append(
									f"{name}: L{level} target exceeded by mandatory caps"
								)
		for name in baselines:
			fitted = self.baselines.get(name)
			if fitted is None:
				reasons.append(f"{name}: missing baseline thresholds")
			elif fitted.units != scored:
				reasons.append(f"{name}: baseline not fitted on the scored horizons")
			elif fitted.units < minimum_policy:
				resolution_reasons.append(f"{name}: need {minimum_policy} baseline policy horizons")
		if (accumulators or baselines) and scored <= 0:
			reasons.append("no scored policy horizons")
		if accumulators and null_windows <= 0:
			reasons.append("no non-overlapping score windows")
		warnings = resolution_reasons if allow_resolution_limited else []
		if not allow_resolution_limited:
			reasons = resolution_reasons + reasons
		detectability: list[str] = []
		control = self.mixture_control
		if accumulators and control and control.share_below_null_median >= MIXTURE_WARNING_SHARE:
			detectability.append(
				f"{control.share_below_null_median:.0%} of {control.windows} pooled benign-mixture "
				f"windows score below the median benign account window (median p "
				f"{control.median_p_value:.3f}); the upper-tail score cannot flag traffic that "
				"resembles the benign mixture"
			)
		if accumulators and self.config.mixture_control_required:
			if control is None or control.windows <= 0:
				reasons.append("required mixture control is missing or has no windows")
			elif detectability:
				reasons.append("required mixture control failed: " + detectability[0])
		return {
			"rule": "calibration-resolution-v2",
			"status": "invalid" if reasons else "resolution-limited" if warnings else "adequate",
			"warnings": warnings,
			"allow_resolution_limited": allow_resolution_limited,
			"passed": not reasons,
			"reasons": reasons,
			"accumulators": list(accumulators),
			"baselines": list(baselines),
			"minimum_score_windows": minimum_score,
			"minimum_policy_horizons": minimum_policy,
			"nonoverlapping_score_windows": null_windows,
			"detectability_warnings": detectability,
			"mixture_control_required": self.config.mixture_control_required,
			"empirical_p_value_floor": 1 / (len(self.calibration_scores) + 1),
			"scored_policy_horizons": scored,
			"levels": {
				f"L{level}": self._level_report(level, accumulators, scored, null_windows)
				for level in levels
			},
			"policy_horizons": self.counts.get("policy_calibration_horizons", 0),
			"interpretation": "minimum resolution only; not evidence of population error control",
		}

	def require_current_policy(self) -> None:
		if self.horizon_policy != HORIZON_POLICY:
			raise ValueError(
				f"calibration uses horizon policy {self.horizon_policy}, not {HORIZON_POLICY}; "
				"recalibrate in a new directory"
			)

	def p_value(self, score: float) -> float:
		"""Paper Eq. 3: the upper-tail rank of a score among benign calibration scores."""
		m = len(self.calibration_scores)
		below = int(np.searchsorted(self.calibration_scores, score, side="left"))
		return (1 + m - below) / (m + 1)

	def level(
		self, accumulator: str, statistic: float, mode: Literal["calibrated", "ville"]
	) -> int:
		"""Highest level whose threshold the statistic strictly exceeds."""
		values = (
			self.ville_thresholds if mode == "ville" else self.thresholds[accumulator].thresholds
		)
		return sum(statistic > threshold for threshold in values)


def pooled_accounts(
	rows: Iterable[tuple[str, str]], prefix: str, size: int, seed: int
) -> list[BenignRequest]:
	"""Accounts of `size` consecutive requests over a seeded shuffle of (key, text) rows.

	Task pools keep source order, which groups SQuAD questions by article. Slicing that order
	gives single-topic accounts and windows that score above a broad extractor, so rows are
	ordered by a hash of their key first. These accounts remain pooled smoke fixtures.
	"""
	ordered = sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row[0]}".encode()).digest())
	return [
		BenignRequest(account=f"{prefix}-{index // size:05d}", text=text)
		for index, (_, text) in enumerate(ordered)
	]


def account_streams(requests: Iterable[BenignRequest]) -> dict[str, list[BenignRequest]]:
	streams: dict[str, list[BenignRequest]] = {}
	for index, request in enumerate(requests):
		# Replay uses the input row index as time when no timestamp is supplied.
		if request.time is None:
			request = request.model_copy(update={"time": float(index)})
		streams.setdefault(request.account, []).append(request)
	return streams


def horizons(stream: list[BenignRequest], seconds: float) -> list[list[BenignRequest]]:
	"""Split by elapsed time only; exceeding the attempt cap never starts a fresh horizon."""
	parts: list[list[BenignRequest]] = []
	start: float | None = None
	for request in stream:
		expired = bool(parts) and (
			request.time is not None and start is not None and request.time - start >= seconds
		)
		if not parts or expired:
			parts.append([])
			start = request.time
		parts[-1].append(request)
	return parts


def peak_per_minute(times: list[float]) -> int:
	"""The most requests in any 60-second interval (t - 60, t], as RatePolicy counts them."""
	peak, start = 0, 0
	for end, time in enumerate(times):
		while times[start] <= time - 60:
			start += 1
		peak = max(peak, end - start + 1)
	return peak


def interleave(
	requests: list[BenignRequest],
) -> tuple[list[int], Literal["time", "round_robin"]]:
	"""Indices in arrival order: by time when every request has one, else round-robin."""
	if all(request.time is not None for request in requests):
		return sorted(range(len(requests)), key=lambda i: requests[i].time or 0.0), "time"
	queues: dict[str, list[int]] = {}
	for index, request in enumerate(requests):
		queues.setdefault(request.account, []).append(index)
	order: list[int] = []
	position = 0
	while len(order) < len(requests):
		for queue in queues.values():
			if position < len(queue):
				order.append(queue[position])
		position += 1
	return order, "round_robin"


def complete_windows(count: int, size: int, stride: int | None = None) -> list[range]:
	return [range(start, start + size) for start in range(0, count - size + 1, stride or size)]


def binomial_upper(successes: int, trials: int, confidence: float = 0.95) -> float:
	"""One-sided Clopper-Pearson upper bound for a binomial rate, by bisection."""
	if trials < 1 or not 0 <= successes <= trials:
		raise ValueError("binomial bound needs 0 <= successes <= trials and trials >= 1")
	if successes == trials:
		return 1.0
	alpha = 1 - confidence

	def cdf(rate: float) -> float:
		return sum(
			math.comb(trials, k) * rate**k * (1 - rate) ** (trials - k)
			for k in range(successes + 1)
		)

	low, high = successes / trials, 1.0
	for _ in range(60):
		middle = (low + high) / 2
		low, high = (middle, high) if cdf(middle) > alpha else (low, middle)
	return high


def fit_threshold(values: list[float], target: float) -> tuple[float, float]:
	"""The smallest threshold that at most `target` of values strictly exceed.

	Returns the threshold and the achieved exceedance rate. Ties stay at or below it, so a
	small sample yields a conservative threshold rather than a fabricated rate.
	"""
	if not values:
		raise ValueError("threshold calibration needs at least one benign unit")
	ordered = sorted(values, reverse=True)
	allowed = math.floor(target * len(ordered) + 1e-9)
	threshold = ordered[min(allowed, len(ordered) - 1)]
	return threshold, sum(value > threshold for value in ordered) / len(ordered)


def nested_thresholds(
	values: list[float], targets: Iterable[float], forced: list[bool] | None = None
) -> LevelThresholds:
	"""Fit the remaining escalation allowance after mandatory attempt-cap refusals."""
	if not values:
		raise ValueError("threshold calibration needs at least one benign unit")
	targets = list(targets)
	forced = [False] * len(values) if forced is None else forced
	if len(forced) != len(values):
		raise ValueError("forced refusals must match calibration units")
	ordered = sorted((v for v, cap in zip(values, forced, strict=True) if not cap), reverse=True)
	thresholds: list[float] = []
	for target in targets:
		allowed = max(0, math.floor(target * len(values) + 1e-9) - sum(forced))
		threshold = ordered[min(allowed, len(ordered) - 1)] if ordered else max(values)
		thresholds.append(max([threshold, *thresholds[-1:]]))
	exceeded = [
		sum(cap or value > threshold for value, cap in zip(values, forced, strict=True))
		for threshold in thresholds
	]
	return LevelThresholds(
		thresholds=thresholds,
		achieved=[count / len(values) for count in exceeded],
		units=len(values),
		upper95=[binomial_upper(count, len(values)) for count in exceeded],
		level_results=[
			{
				"target": target,
				"threshold": threshold,
				"scored": len(values),
				"reached": count,
				"detector_reached": sum(value > threshold for value in values),
				"capped": sum(forced),
				"achieved": count / len(values),
				"upper95": binomial_upper(count, len(values)),
			}
			for target, threshold, count in zip(targets, thresholds, exceeded, strict=True)
		],
	)


def mixture_control(
	calibration: Calibration,
	scorer: WindowScorer,
	requests: list[BenignRequest],
	embedded: Array,
	windows: int = MIXTURE_WINDOWS,
) -> MixtureControl | None:
	"""Rank pooled-mixture windows of benign requests in the per-account null.

	A seeded shuffle across accounts stands for a client that samples the whole benign task
	mixture, as multiplexed extraction does. `embedded` holds one row per request, in order.
	None when the requests cannot fill one window.
	"""
	size = calibration.config.window_size
	order = random.Random(calibration.config.seed + 2).sample(range(len(requests)), len(requests))
	spans = complete_windows(len(order), size)[:windows]
	if not spans:
		return None
	p_values = sorted(
		calibration.p_value(
			scorer.score(
				embedded[[order[i] for i in span]],
				[requests[order[i]].text for i in span]
				if calibration.config.distinct_prompts
				else None,
			)
		)
		for span in spans
	)
	return MixtureControl(
		windows=len(p_values),
		median_p_value=float(np.median(p_values)),
		min_p_value=p_values[0],
		share_below_null_median=sum(p > 0.5 for p in p_values) / len(p_values),
	)


def requests_sha256(requests: list[BenignRequest]) -> str:
	payload = "\n".join(
		request.model_dump_json(
			exclude={"workload_version"} if request.workload_version is None else set()
		)
		for request in requests
	)
	return hashlib.sha256(payload.encode()).hexdigest()


def validate_separation(
	groups: dict[str, list[BenignRequest]], *, require_identity: bool = True
) -> None:
	"""Reject shared accounts, source documents, and exact task content across roles."""
	for field in ("account", "source_group", "text"):
		seen: set[str] = set()
		for role, rows in groups.items():
			values = {getattr(row, field) for row in rows}
			if require_identity and (None in values or "" in values):
				raise ValueError(f"{role} has missing {field}")
			values.discard(None)
			if seen.intersection(values):
				raise ValueError(f"{role} overlaps another role on {field}")
			seen.update(values)


def calibrate(
	*,
	config: HarnessConfig,
	encoder: Encoder,
	fitting: list[BenignRequest],
	score_calibration: list[BenignRequest],
	policy_calibration: list[BenignRequest],
	log: Callable[[str], None] | None = None,
	input_validation: dict[str, Any] | None = None,
) -> tuple[Calibration, Array]:
	"""Build the calibration artifact and the reference windows it scores against."""

	if (
		any(
			row.workload_version == "gateway-clients-v5"
			for role in (fitting, score_calibration, policy_calibration)
			for row in role
		)
		and input_validation is None
	):
		raise ValueError("v5 workload requires input validation")

	def report(message: str) -> None:
		if log:
			log(message)

	size = config.window_size
	validate_separation(
		{
			"fitting": fitting,
			"score_calibration": score_calibration,
			"policy_calibration": policy_calibration,
		},
		require_identity=config.task != "text",
	)
	rng = random.Random(config.seed)
	truncated: dict[str, int] = {}

	def embed(requests: list[BenignRequest], role: str) -> Array:
		"""Embed the detector serialization and count inputs the encoder truncated."""
		before = getattr(encoder, "truncated", None)
		rows = encoder.encode([detector_text(config.task, request.text) for request in requests])
		after = getattr(encoder, "truncated", None)
		if isinstance(before, int) and isinstance(after, int):
			truncated[f"{role}_truncated"] = after - before
		return rows

	report(f"embedding {len(fitting):,} fitting requests")
	fitted = embed(fitting, "fitting")
	needed = size * config.reference_windows
	if len(fitted) < needed:
		raise ValueError(f"fitting needs at least {needed} requests for the reference windows")
	order = rng.sample(range(len(fitted)), len(fitted))
	references = np.stack(
		[fitted[order[i * size : (i + 1) * size]] for i in range(config.reference_windows)]
	)
	bandwidth_rows = (
		order[needed : needed + config.bandwidth_sample] or order[: config.bandwidth_sample]
	)
	sigma = median_bandwidth(fitted[sorted(bandwidth_rows)])
	scorer = WindowScorer(references, sigma, config.scales)

	def stream_scores(
		requests: list[BenignRequest],
		label: str,
		embedded: Array | None = None,
		stride: int | None = None,
	) -> list[list[list[float]]]:
		"""Window scores per account, then per horizon, in order."""
		streams = account_streams(requests)
		if embedded is None:
			report(f"embedding {len(requests):,} {label} requests from {len(streams):,} accounts")
			embedded = embed(requests, label.replace("-", "_"))
		rows: dict[str, list[int]] = {}
		for index, request in enumerate(requests):
			rows.setdefault(request.account, []).append(index)
		result: list[list[list[float]]] = []
		for account, stream in streams.items():
			indices = iter(rows[account])
			per_horizon: list[list[float]] = []
			for horizon in horizons(stream, config.horizon_seconds):
				members = [next(indices) for _ in horizon]
				# Later requests are hard refusals, so they cannot form detector windows.
				members = members[: config.horizon_attempts]
				per_horizon.append(
					[
						scorer.score(
							embedded[[members[i] for i in window]],
							[horizon[i].text for i in window] if config.distinct_prompts else None,
						)
						for window in complete_windows(len(members), size, stride)
					]
				)
			result.append(per_horizon)
		return result

	report(f"embedding {len(score_calibration):,} score-calibration requests")
	null_embedded = embed(score_calibration, "score_calibration")
	stride = config.null_stride or max(1, size // 4)
	null_streams = stream_scores(score_calibration, "score-calibration", null_embedded, stride)
	null_scores = sorted(
		score for account in null_streams for horizon in account for score in horizon
	)
	independent_null = sum(
		min(len(horizon), config.horizon_attempts) // size
		for stream in account_streams(score_calibration).values()
		for horizon in horizons(stream, config.horizon_seconds)
	)
	# Null windows by client type, so the benign mix behind the null can be disclosed.
	null_by_type: dict[str, int] = {}
	for stream in account_streams(score_calibration).values():
		for horizon in horizons(stream, config.horizon_seconds):
			key = f"score_calibration_windows_{horizon[0].client_type or 'unknown'}"
			null_by_type[key] = null_by_type.get(key, 0) + len(
				complete_windows(min(len(horizon), config.horizon_attempts), size, stride)
			)
	if not null_scores:
		raise ValueError("score calibration produced no complete windows")
	calibration = Calibration(
		config=config,
		encoder=encoder.identity(),
		serialization=SERIALIZATION,
		horizon_policy=HORIZON_POLICY,
		input_validation=input_validation,
		sigma=sigma,
		calibration_scores=null_scores,
		thresholds={},
		ville_thresholds=[math.log(1 / target) for target in config.level_targets],
		counts={},
	)
	units = [
		horizon
		for account in stream_scores(policy_calibration, "policy-calibration")
		for horizon in account
	]
	# Thresholds are fitted on scored horizons only: an unscored horizon has statistic 0 and
	# can never escalate, so counting it would loosen every level's rate among scorable
	# accounts. The all-horizon rate is then at most each target times the scored fraction.
	scored_units = [scores for scores in units if scores]
	if not scored_units:
		raise ValueError("policy calibration has no horizon with a complete window")
	maxima: dict[str, list[float]] = {name: [] for name in config.accumulators}
	for scores in scored_units:
		p_values = [calibration.p_value(score) for score in scores]
		for name, spec in config.accumulators.items():
			state, peak = 0.0, 0.0
			for p in p_values:
				state = spec.update(state, p)
				peak = max(peak, state)
			maxima[name].append(peak)
	policy_horizons = [
		horizon
		for stream in account_streams(policy_calibration).values()
		for horizon in horizons(stream, config.horizon_seconds)
	]
	capped = [len(horizon) > config.horizon_attempts for horizon in policy_horizons]
	scored_capped = [cap for cap, scores in zip(capped, units, strict=True) if scores]
	calibration.thresholds = {
		name: nested_thresholds(values, config.level_targets, scored_capped)
		for name, values in maxima.items()
	}
	for level_fit in calibration.thresholds.values():
		assert level_fit.level_results is not None
		for result in level_fit.level_results:
			result["horizons"] = len(units)
			result["censored"] = len(units) - len(scored_units)
	calibration.counts = {
		"fitting_requests": len(fitting),
		"score_calibration_requests": len(score_calibration),
		"score_calibration_windows": len(null_scores),
		"score_calibration_independent_windows": independent_null,
		"score_calibration_stride": stride,
		**null_by_type,
		"policy_calibration_requests": len(policy_calibration),
		"policy_calibration_accounts": len(account_streams(policy_calibration)),
		"policy_calibration_horizons": len(units),
		"policy_calibration_scored_horizons": len(scored_units),
		"policy_calibration_unscored_horizons": len(units) - len(scored_units),
		"policy_calibration_capped_horizons": sum(capped),
		**truncated,
	}
	# Baselines are fitted on the same scored horizons as the accumulators, so every defense
	# is held to its target among the accounts long enough to be scored.
	policy_horizons = [horizon for horizon in policy_horizons if len(horizon) >= size]
	block_target = [config.level_targets[-1]]
	calibration.baselines["quota"] = nested_thresholds(
		[float(len(horizon)) for horizon in policy_horizons], block_target, scored_capped
	)
	if all(request.time is not None for request in policy_calibration):
		calibration.baselines["rate_per_minute"] = nested_thresholds(
			[
				float(peak_per_minute([request.time or 0.0 for request in horizon]))
				for horizon in policy_horizons
			],
			block_target,
			scored_capped,
		)
	calibration.mixture_control = mixture_control(
		calibration, scorer, score_calibration, null_embedded
	)
	_calibrate_aggregate(calibration, fitted, sigma, score_calibration, null_embedded, report)
	calibration.encoder = encoder.identity()
	calibration.sha256 = {
		"fitting": requests_sha256(fitting),
		"score_calibration": requests_sha256(score_calibration),
		"policy_calibration": requests_sha256(policy_calibration),
	}
	return calibration, references


def _calibrate_aggregate(
	calibration: Calibration,
	fitted: Array,
	sigma: float,
	requests: list[BenignRequest],
	embedded: Array,
	report: Callable[[str], None],
) -> None:
	"""Fit the aggregate monitor, or record why the benign data cannot support it."""
	config = calibration.config
	if not config.aggregate_enabled:
		calibration.aggregate_skipped = "not run in the declared task study"
		return
	size = config.aggregate_window_size
	needed = size * config.aggregate_reference_windows
	if len(fitted) < needed:
		calibration.aggregate_skipped = (
			f"fitting has {len(fitted):,} requests; {needed:,} are needed for "
			f"{config.aggregate_reference_windows} aggregate reference windows of {size:,}"
		)
		return
	rows = random.Random(config.seed + 1).sample(range(len(fitted)), needed)
	references = np.stack(
		[fitted[rows[i * size : (i + 1) * size]] for i in range(len(rows) // size)]
	)
	scorer = WindowScorer(references, sigma, config.scales)
	order, interleaving = interleave(requests)
	windows = complete_windows(len(order), size)
	if not windows:
		calibration.aggregate_skipped = (
			f"score calibration has {len(order):,} requests, under one aggregate window of {size:,}"
		)
		return
	report(f"scoring {len(windows)} aggregate windows")
	scores = [scorer.score(embedded[[order[i] for i in window]]) for window in windows]
	threshold, achieved = fit_threshold(scores, config.aggregate_alert_rate)
	calibration.aggregate = AggregateCalibration(
		window_size=size,
		threshold=threshold,
		achieved=achieved,
		null_windows=len(scores),
		interleaving=interleaving,
	)
	calibration._aggregate_references = references


def save_calibration(directory: Path, calibration: Calibration, references: Array) -> None:
	directory.mkdir(parents=True, exist_ok=True)
	if (directory / "calibration.json").exists():
		raise ValueError(f"{directory} already holds a calibration; choose a new directory")
	np.save(directory / "references.npy", references)
	calibration.sha256["references"] = hashlib.sha256(
		(directory / "references.npy").read_bytes()
	).hexdigest()
	if calibration._aggregate_references is not None:
		np.save(directory / "aggregate_references.npy", calibration._aggregate_references)
		calibration.sha256["aggregate_references"] = hashlib.sha256(
			(directory / "aggregate_references.npy").read_bytes()
		).hexdigest()
	(directory / "calibration.json").write_text(
		json.dumps(calibration.model_dump(mode="json"), indent="\t") + "\n", encoding="utf-8"
	)


def load_calibration(directory: Path) -> tuple[Calibration, WindowScorer]:
	raw = json.loads((directory / "calibration.json").read_text())
	# Older artifacts scored every request in a window; keep that for their frozen null.
	raw["config"].setdefault("distinct_prompts", False)
	calibration = Calibration.model_validate(raw)
	calibration.require_current_policy()
	if calibration.config.task != "text" and calibration.serialization != SERIALIZATION:
		raise ValueError(
			f"calibration uses detector serialization {calibration.serialization}, "
			f"not {SERIALIZATION}; recalibrate"
		)
	path = directory / "references.npy"
	if hashlib.sha256(path.read_bytes()).hexdigest() != calibration.sha256.get("references"):
		raise ValueError("reference windows differ from the calibration artifact")
	references = np.load(path)
	return calibration, WindowScorer(references, calibration.sigma, calibration.config.scales)


def load_aggregate(directory: Path, calibration: Calibration) -> WindowScorer | None:
	"""The aggregate monitor's scorer, or None when calibration skipped it."""
	if calibration.aggregate is None:
		return None
	path = directory / "aggregate_references.npy"
	if hashlib.sha256(path.read_bytes()).hexdigest() != calibration.sha256.get(
		"aggregate_references"
	):
		raise ValueError("aggregate reference windows differ from the calibration artifact")
	return WindowScorer(np.load(path), calibration.sigma, calibration.config.scales)
