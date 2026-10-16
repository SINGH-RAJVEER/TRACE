"""Replay a fixed request stream through one policy and record every decision."""

import json
from pathlib import Path
from typing import Any, Literal

from harness.calibration import BenignRequest, Calibration
from harness.encoder import Encoder
from harness.mmd import WindowScorer
from harness.policy import AllowPolicy, MMDPolicy, Policy, QuotaPolicy, RatePolicy
from harness.serialization import detector_text


def make_policy(
	name: str,
	calibration: Calibration,
	scorer: WindowScorer,
	*,
	accumulator: str = "paper",
	thresholds: Literal["calibrated", "ville"] = "calibrated",
	l4_action: Literal["refuse", "deceive"] = "refuse",
	deception_rate: float = 1.0,
	deception_key: bytes | None = None,
) -> Policy:
	config = calibration.config
	horizon = (config.horizon_seconds, config.horizon_attempts)
	if name == "allow":
		return AllowPolicy(*horizon)
	if name == "quota":
		return QuotaPolicy(calibration.baselines["quota"].thresholds[0], *horizon)
	if name == "rate":
		if "rate_per_minute" not in calibration.baselines:
			raise ValueError("the calibration has no rate cutoff; its requests had no times")
		return RatePolicy(calibration.baselines["rate_per_minute"].thresholds[0], *horizon)
	if name == "mmd":
		return MMDPolicy(
			calibration,
			scorer,
			accumulator=accumulator,
			thresholds=thresholds,
			l4_action=l4_action,
			deception_rate=deception_rate,
			deception_key=deception_key,
		)
	raise ValueError("policy must be allow, quota, rate, or mmd")


def replay(
	*,
	calibration: Calibration,
	scorer: WindowScorer,
	requests: list[BenignRequest],
	output: Path,
	policy: str,
	accumulator: str = "paper",
	thresholds: Literal["calibrated", "ville"] = "calibrated",
	encoder: Encoder | None = None,
	l4_action: Literal["refuse", "deceive"] = "refuse",
	deception_rate: float = 1.0,
	deception_key: bytes | None = None,
) -> dict[str, Any]:
	"""Process requests in file order. Requests without a time are spaced one second apart.

	The summary counts, per account horizon, the highest level reached. On benign traffic its
	rates are the account-level false-escalation rates FPR_1 to FPR_4.
	"""
	runner = make_policy(
		policy,
		calibration,
		scorer,
		accumulator=accumulator,
		thresholds=thresholds,
		l4_action=l4_action,
		deception_rate=deception_rate,
		deception_key=deception_key,
	)
	embeddings = None
	if policy == "mmd":
		if encoder is None:
			raise ValueError("the mmd policy needs an encoder")
		task = calibration.config.task
		embeddings = encoder.encode([detector_text(task, request.text) for request in requests])
	peaks: list[int] = []
	unit_of: dict[str, int] = {}
	scored: set[int] = set()
	windows = 0
	deceived = 0
	output.parent.mkdir(parents=True, exist_ok=True)
	with output.open("w", encoding="utf-8") as sink:
		for index, request in enumerate(requests):
			now = request.time if request.time is not None else float(index)
			decision = runner.admit(
				request.account,
				request.text,
				now,
				embedding=None if embeddings is None else embeddings[index],
			)
			if decision.attempt == 1:
				unit_of[request.account] = len(peaks)
				peaks.append(0)
			unit = unit_of[request.account]
			peaks[unit] = max(peaks[unit], decision.level)
			deceived += decision.deceive
			if decision.window_completed:
				windows += 1
				scored.add(unit)
			record = {
				"index": index,
				"account": decision.account,
				"attempt": decision.attempt,
				"level": decision.level,
				"action": decision.action,
				"release_at": decision.release_at,
				"window_completed": decision.window_completed,
				"p_value": decision.p_value,
				"statistic": decision.statistic,
				"deceive": decision.deceive,
			}
			sink.write(json.dumps(record) + "\n")
	units = len(peaks)
	return {
		"policy": policy,
		"accumulator": accumulator if policy == "mmd" else None,
		"thresholds": thresholds if policy == "mmd" else None,
		"requests": len(requests),
		"horizons": units,
		"unscored_horizons": units - len(scored) if policy == "mmd" else None,
		"windows": windows,
		"l4_action": l4_action if policy == "mmd" else None,
		"deceived": deceived,
		"reached": {f"L{level}": sum(peak >= level for peak in peaks) for level in range(1, 5)},
		"reached_rate": {
			f"L{level}": sum(peak >= level for peak in peaks) / units if units else None
			for level in range(1, 5)
		},
		"output": str(output),
	}
