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
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from harness.accumulators import DEFAULT_ACCUMULATORS, AccumulatorSpec
from harness.encoder import Encoder
from harness.mmd import SCALES, Array, WindowScorer, median_bandwidth
from harness.serialization import SERIALIZATION, Task, detector_text

LEVELS = 4


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


class BenignRequest(BaseModel):
	"""One benign request. Requests of an account appear in order; time is in seconds."""

	model_config = ConfigDict(extra="forbid")

	account: str = Field(min_length=1)
	text: str
	time: float | None = None
	example_id: str | None = None
	source_group: str | None = None
	client_type: str | None = None


class LevelThresholds(BaseModel):
	thresholds: list[float]
	achieved: list[float]
	# Horizons fitted: scored horizons for accumulators, all horizons for block baselines.
	units: int
	# One-sided 95% Clopper-Pearson upper bounds on each achieved rate over `units`.
	upper95: list[float] = Field(default_factory=list)


class AggregateCalibration(BaseModel):
	"""The aggregate monitor's alert threshold, fitted on interleaved benign windows."""

	window_size: int
	threshold: float
	achieved: float
	null_windows: int
	interleaving: Literal["time", "round_robin"]


class Calibration(BaseModel):
	"""The frozen calibration artifact. Reference embeddings are saved beside it."""

	config: HarnessConfig
	encoder: dict[str, Any]
	# Detector serialization version; None in artifacts from before it was recorded.
	serialization: str | None = None
	sigma: float
	calibration_scores: list[float]
	thresholds: dict[str, LevelThresholds]
	# Theoretical log(1/target) thresholds; valid only for e_process under its assumptions.
	ville_thresholds: list[float]
	# Block-only baselines fitted to the L4 target on the same benign horizons: "quota" in
	# requests per horizon and, when every request has a time, "rate_per_minute".
	baselines: dict[str, LevelThresholds] = Field(default_factory=dict)
	counts: dict[str, int]
	sha256: dict[str, str] = Field(default_factory=dict)
	aggregate: AggregateCalibration | None = None
	aggregate_skipped: str | None = None
	_aggregate_references: Array | None = PrivateAttr(default=None)

	def adequacy(
		self, *, accumulators: tuple[str, ...] = (), baselines: tuple[str, ...] = ()
	) -> dict[str, Any]:
		"""Minimum resolution for selected study defenses, not a false-positive guarantee.

		Overlapping score windows do not count toward the minimum null size. Scored policy
		horizons must permit at least one exceedance at the smallest target, and selected MMD
		levels must remain distinct. Legacy artifacts without scored counts fail closed.
		"""
		target = min(self.config.level_targets)
		minimum_policy = math.ceil(1 / target)
		minimum_score = minimum_policy - 1
		null_windows = self.counts.get("score_calibration_independent_windows", 0)
		scored = self.counts.get("policy_calibration_scored_horizons", 0)
		reasons = []
		if accumulators:
			if null_windows < minimum_score:
				reasons.append(
					f"{null_windows} non-overlapping score windows; need {minimum_score}"
				)
			if scored < minimum_policy:
				reasons.append(f"{scored} scored policy horizons; need {minimum_policy}")
			for name in accumulators:
				fitted = self.thresholds.get(name)
				if fitted is None or fitted.units != scored:
					reasons.append(f"{name}: missing thresholds or incompatible scored denominator")
				elif (
					len(fitted.thresholds) != LEVELS
					or any(not math.isfinite(value) for value in fitted.thresholds)
					or any(
						a >= b
						for a, b in zip(fitted.thresholds, fitted.thresholds[1:], strict=False)
					)
				):
					reasons.append(f"{name}: level thresholds are collapsed or invalid")
		for name in baselines:
			fitted = self.baselines.get(name)
			if fitted is None or fitted.units < minimum_policy:
				reasons.append(f"{name}: need {minimum_policy} baseline policy horizons")
		return {
			"rule": "calibration-resolution-v1",
			"passed": not reasons,
			"reasons": reasons,
			"accumulators": list(accumulators),
			"baselines": list(baselines),
			"minimum_score_windows": minimum_score,
			"minimum_policy_horizons": minimum_policy,
			"nonoverlapping_score_windows": null_windows,
			"empirical_p_value_floor": 1 / (len(self.calibration_scores) + 1),
			"scored_policy_horizons": scored,
			"policy_horizons": self.counts.get("policy_calibration_horizons", 0),
			"interpretation": "minimum resolution only; not evidence of population error control",
		}

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
	for request in requests:
		streams.setdefault(request.account, []).append(request)
	return streams


def horizons(
	stream: list[BenignRequest], seconds: float, attempts: int
) -> list[list[BenignRequest]]:
	"""Split one account's requests into policy horizons by elapsed time or attempt count."""
	parts: list[list[BenignRequest]] = []
	start: float | None = None
	for request in stream:
		expired = bool(parts) and (
			len(parts[-1]) >= attempts
			or (request.time is not None and start is not None and request.time - start >= seconds)
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


def nested_thresholds(values: list[float], targets: Iterable[float]) -> LevelThresholds:
	thresholds: list[float] = []
	for target in targets:
		threshold, _ = fit_threshold(values, target)
		thresholds.append(max([threshold, *thresholds[-1:]]))
	exceeded = [sum(value > threshold for value in values) for threshold in thresholds]
	return LevelThresholds(
		thresholds=thresholds,
		achieved=[count / len(values) for count in exceeded],
		units=len(values),
		upper95=[binomial_upper(count, len(values)) for count in exceeded],
	)


def requests_sha256(requests: list[BenignRequest]) -> str:
	payload = "\n".join(request.model_dump_json() for request in requests)
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
) -> tuple[Calibration, Array]:
	"""Build the calibration artifact and the reference windows it scores against."""

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
			for horizon in horizons(stream, config.horizon_seconds, config.horizon_attempts):
				members = [next(indices) for _ in horizon]
				per_horizon.append(
					[
						scorer.score(embedded[[members[i] for i in window]])
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
		len(horizon) // size
		for stream in account_streams(score_calibration).values()
		for horizon in horizons(stream, config.horizon_seconds, config.horizon_attempts)
	)
	if not null_scores:
		raise ValueError("score calibration produced no complete windows")
	calibration = Calibration(
		config=config,
		encoder=encoder.identity(),
		serialization=SERIALIZATION,
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
	calibration.thresholds = {
		name: nested_thresholds(values, config.level_targets) for name, values in maxima.items()
	}
	calibration.counts = {
		"fitting_requests": len(fitting),
		"score_calibration_requests": len(score_calibration),
		"score_calibration_windows": len(null_scores),
		"score_calibration_independent_windows": independent_null,
		"score_calibration_stride": stride,
		"policy_calibration_requests": len(policy_calibration),
		"policy_calibration_accounts": len(account_streams(policy_calibration)),
		"policy_calibration_horizons": len(units),
		"policy_calibration_scored_horizons": len(scored_units),
		"policy_calibration_unscored_horizons": len(units) - len(scored_units),
		**truncated,
	}
	policy_horizons = [
		horizon
		for stream in account_streams(policy_calibration).values()
		for horizon in horizons(stream, config.horizon_seconds, config.horizon_attempts)
	]
	block_target = [config.level_targets[-1]]
	calibration.baselines["quota"] = nested_thresholds(
		[float(len(horizon)) for horizon in policy_horizons], block_target
	)
	if all(request.time is not None for request in policy_calibration):
		calibration.baselines["rate_per_minute"] = nested_thresholds(
			[
				float(peak_per_minute([request.time or 0.0 for request in horizon]))
				for horizon in policy_horizons
			],
			block_target,
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
	calibration = Calibration.model_validate_json((directory / "calibration.json").read_text())
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
