"""Replay a fixed request stream through one policy and record every decision."""

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from harness.calibration import BenignRequest, Calibration, horizon_indices
from harness.encoder import Encoder
from harness.mmd import Array, WindowScorer
from harness.policy import (
	AllowPolicy,
	MMDPolicy,
	Policy,
	QuotaPolicy,
	RatePolicy,
	request_digest,
)
from harness.serialization import detector_text
from harness.shadow import ShadowMonitor, assign, request_domain
from harness.store import StateStore


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
	actions: Literal["graduated", "direct"] = "graduated",
	store: StateStore | None = None,
	request_quota: int | None = None,
) -> Policy:
	config = calibration.config
	calibration.require_current_policy()
	horizon = (config.horizon_seconds, config.horizon_attempts)
	if name == "allow":
		return AllowPolicy(*horizon, store=store)
	if name == "quota":
		return QuotaPolicy(
			request_quota
			if request_quota is not None
			else calibration.baselines["quota"].thresholds[0],
			*horizon,
			store=store,
		)
	if name == "rate":
		if "rate_per_minute" not in calibration.baselines:
			raise ValueError("the calibration has no rate cutoff; its requests had no times")
		return RatePolicy(
			calibration.baselines["rate_per_minute"].thresholds[0], *horizon, store=store
		)
	if name == "mmd":
		return MMDPolicy(
			calibration,
			scorer,
			accumulator=accumulator,
			thresholds=thresholds,
			l4_action=l4_action,
			deception_rate=deception_rate,
			deception_key=deception_key,
			actions=actions,
			store=store,
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
	actions: Literal["graduated", "direct"] = "graduated",
) -> dict[str, Any]:
	"""Process requests in file order. Requests without a time are spaced one second apart.

	The summary counts, per account horizon, the highest level reached. On benign traffic its
	rates are the account-level false-escalation rates FPR_1 to FPR_4 over all horizons;
	`reached_rate_scored` restricts them to horizons with at least a window's worth of
	attempts, the denominator every defense's thresholds were fitted on.
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
		actions=actions,
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
			# A horizon is scored once it holds a complete window's worth of attempts, the
			# population every defense's thresholds were fitted on.
			if decision.attempt >= calibration.config.window_size:
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
	levels = (
		{1, 2, 3, 4}
		if policy == "mmd" and actions == "graduated"
		else ({4} if policy != "allow" else set())
	)
	return {
		"horizon_policy": calibration.horizon_policy,
		"policy": policy,
		"accumulator": accumulator if policy == "mmd" else None,
		"actions": actions if policy == "mmd" else None,
		"thresholds": thresholds if policy == "mmd" else None,
		"requests": len(requests),
		"horizons": units,
		"unscored_horizons": units - len(scored) if policy != "allow" else None,
		"windows": windows,
		"l4_action": l4_action if policy == "mmd" else None,
		"deceived": deceived,
		"implemented_levels": sorted(levels),
		"reached": {
			f"L{level}": sum(peak >= level for peak in peaks) if level in levels else None
			for level in range(1, 5)
		},
		"reached_rate": {
			f"L{level}": sum(peak >= level for peak in peaks) / units
			if units and level in levels
			else None
			for level in range(1, 5)
		},
		# Over horizons with a complete window, the denominator the thresholds were fitted on.
		"reached_rate_scored": {
			f"L{level}": sum(peaks[unit] >= level for unit in scored) / len(scored)
			if scored and level in levels
			else None
			for level in range(1, 5)
		},
		"output": str(output),
	}


def shadow_check(
	*,
	calibration: Calibration,
	requests: list[BenignRequest],
	embeddings: Array,
	reorder: Literal["none", "topic"] = "none",
) -> dict[str, Any]:
	"""Replay requests through the shadow scorers only and summarize both scores per horizon.

	`embeddings` holds one row per request. With reorder="topic", each account horizon is sent
	grouped by domain and then by coverage cell, as the order-only evasion probes did, while
	keeping its original arrival times. Coverage must not change; conditional windows may.
	No policy runs, so no level or action is produced.
	"""
	shadow = calibration.shadow
	if shadow is None:
		raise ValueError("the calibration has no shadow scorers; recalibrate with --shadow")
	config = calibration.config
	monitor = ShadowMonitor(
		shadow,
		task=config.task,
		window_size=config.window_size,
		scales=config.scales,
		distinct_prompts=config.distinct_prompts,
	)
	cells = assign(embeddings, monitor.centroids)
	units: list[dict[str, Any]] = []
	for members in horizon_indices(requests, config.horizon_seconds, config.horizon_attempts):
		if reorder == "topic":
			members = sorted(
				members,
				key=lambda i: (request_domain(config.task, requests[i].text), int(cells[i]), i),
			)
		account = requests[members[0]].account
		horizon = float(len(units))
		report: dict[str, Any] = {}
		windows: dict[str, list[dict[str, Any]]] = {}
		for index in members:
			report = monitor.observe(
				account,
				horizon,
				requests[index].text,
				embeddings[index],
				request_digest(requests[index].text),
			)
			if report["conditional"] is not None:
				windows.setdefault(report["conditional"]["domain"], []).append(
					report["conditional"]
				)
		units.append(
			{
				"account": account,
				"requests": len(members),
				"coverage": report["coverage"],
				"conditional": {
					domain: _window_summary(rows) for domain, rows in sorted(windows.items())
				},
			}
		)
		monitor.accounts.pop(account, None)
	return {
		"version": shadow.version,
		"reorder": reorder,
		"requests": len(requests),
		"horizons": len(units),
		"minimum_null": shadow.minimum_null,
		"domains": {
			domain: {
				"null_windows": len(fitted.null_scores),
				"independent_windows": fitted.independent_windows,
				"abstain": fitted.abstain,
			}
			for domain, fitted in shadow.domains.items()
		},
		"units": units,
	}


def _window_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
	scored = [row for row in rows if row["abstain"] is None]
	p_values = sorted(row["p_value"] for row in scored)
	scores = sorted(row["score"] for row in scored)
	return {
		"windows": len(rows),
		"abstain": rows[0]["abstain"] if not scored else None,
		"p_value_min_median_max": [p_values[0], float(np.median(p_values)), p_values[-1]]
		if p_values
		else None,
		"score_median": float(np.median(scores)) if scores else None,
	}
