"""Harness utility and extraction resistance, with explicit denominators."""

import math
from collections import Counter
from pathlib import Path
from typing import Any

from distillation.benchmarks.reporting import distribution, percentile, wilson
from distillation.benchmarks.traffic import TrafficRun, TrafficSample, load_traffic_suite


def discover_traffic(directory: Path) -> tuple[list[TrafficRun], list[str]]:
	runs, errors = [], []
	for path in sorted(directory.glob("*.json")):
		if path.name.startswith("traffic-suite-"):
			continue
		try:
			runs.append(TrafficRun.model_validate_json(path.read_text()))
		except ValueError:
			errors.append(f"Cannot read traffic run {path.name}")
	return runs, errors


def traffic_experiments(directory: Path) -> dict[str, Any]:
	runs, errors = discover_traffic(directory)
	keys = dict.fromkeys(run.comparison_key() for run in runs)
	return {
		"experiments": [
			{
				"id": key,
				"name": next(r.config.experiment for r in runs if r.comparison_key() == key),
				"phases": sorted({r.phase for r in runs if r.comparison_key() == key}),
				"demo": next(r.demo for r in runs if r.comparison_key() == key),
			}
			for key in keys
		],
		"errors": errors,
	}


def summarize_traffic(samples: list[TrafficSample], seconds: float) -> dict[str, Any]:
	n = len(samples)
	successes = [s for s in samples if s.succeeded]
	blocked = sum(s.decision == "block" for s in samples)
	known = sum(s.decision != "unknown" for s in samples)
	with_tokens = [s for s in successes if s.output_tokens is not None]
	# Noise is known only when the harness log covered every successful response.
	noise_known = all(s.noised is not None for s in successes)
	noised = sum(bool(s.noised) for s in successes)
	level_known = sum(s.level is not None for s in samples)
	l4 = sum(s.level is not None and s.level >= 4 for s in samples)
	return {
		"n": n,
		"succeeded": len(successes),
		"success_percent": 100 * len(successes) / n if n else None,
		"success_ci95": wilson(len(successes), n) if n else None,
		"l4_requests": l4,
		"l4_percent": 100 * l4 / n if n and level_known == n else None,
		"level_coverage": level_known / n if n else None,
		# Successful responses that were keyed noise rather than the teacher's answer. For
		# genuine users these are silent corruptions; for extraction, worthless supervision.
		"noised": noised if noise_known else None,
		"noised_percent": 100 * noised / n if n and noise_known else None,
		"useful_succeeded": len(successes) - noised if noise_known else None,
		"useful_success_percent": 100 * (len(successes) - noised) / n
		if n and noise_known
		else None,
		"blocked": blocked,
		"confirmed_block_percent": 100 * blocked / n if n else None,
		# Without full decision coverage, zero observed blocks is not a measured FPR.
		"block_percent": 100 * blocked / n if n and known == n else None,
		"throttled": sum(s.decision == "throttle" for s in samples),
		"rate_limited": sum(s.status == "rate_limited" for s in samples),
		"decision_coverage": known / n if n else None,
		"statuses": dict(Counter(s.status for s in samples)),
		"successful_requests_per_second": len(successes) / seconds,
		"successful_output_tokens_per_second": sum(s.output_tokens or 0 for s in with_tokens)
		/ seconds
		if with_tokens
		else None,
		"token_coverage": len(with_tokens) / len(successes) if successes else None,
		**{
			f"latency_p{round(q * 100)}": percentile(
				[s.completed_seconds - s.started_seconds for s in successes], q
			)
			for q in (0.5, 0.9, 0.95, 0.99)
		},
		"schedule_delay_p95": percentile(
			[s.started_seconds - s.scheduled_seconds for s in samples], 0.95
		),
		**{
			f"end_to_end_p{round(q * 100)}": percentile(
				[s.completed_seconds - s.scheduled_seconds for s in samples], q
			)
			for q in (0.5, 0.9, 0.95, 0.99)
		},
		"challenge_seconds": distribution(
			[s.challenge_seconds for s in samples if s.challenge_seconds]
		),
	}


def _progression(samples: list[TrafficSample]) -> dict[str, Any]:
	ordered = sorted(samples, key=lambda s: (s.started_seconds, s.request_id))
	first_throttle = min(
		(s for s in ordered if s.decision == "throttle"),
		key=lambda s: s.completed_seconds,
		default=None,
	)
	first_block = min(
		(s for s in ordered if s.decision == "block"),
		key=lambda s: s.completed_seconds,
		default=None,
	)
	# A stop is a suffix of confirmed blocks with no later in-flight completion leak.
	suffix = []
	for sample in reversed(ordered):
		if sample.decision != "block":
			break
		suffix.append(sample)
	stop = suffix[-1].started_seconds if suffix else None
	leaks = (
		sum(s.succeeded and s.completed_seconds >= stop for s in ordered) if stop is not None else 0
	)
	last_success = max((s.completed_seconds for s in ordered if s.succeeded), default=-1)
	drained_suffix = [s for s in reversed(suffix) if s.started_seconds > last_success]
	return {
		"first_throttle_seconds": first_throttle.completed_seconds if first_throttle else None,
		"first_block_seconds": first_block.completed_seconds if first_block else None,
		"successes_before_first_block": sum(
			s.succeeded and s.completed_seconds < first_block.completed_seconds for s in ordered
		)
		if first_block
		else None,
		"terminal_blocked_requests": len(suffix),
		"terminal_block_start_seconds": stop,
		"successful_completions_after_terminal_block": leaks,
		"observed_stop_seconds": drained_suffix[0].started_seconds
		if len(drained_suffix) >= 2
		else None,
		"observed_stop_blocked_requests": len(drained_suffix),
		"observed_stopped": len(drained_suffix) >= 2,
	}


def build_traffic_report(directory: Path, key: str) -> dict[str, Any]:
	runs, errors = discover_traffic(directory)
	rows = [r for r in runs if r.comparison_key() == key]
	if not rows:
		raise ValueError("traffic comparison does not exist")
	suite = load_traffic_suite(directory / f"traffic-suite-{rows[0].suite_sha256}.json")
	expected = {request.id: request for request in suite.requests}
	latest = {
		phase: max((r for r in rows if r.phase == phase), key=lambda r: r.completed_at)
		for phase in ("before", "after")
		if any(r.phase == phase for r in rows)
	}
	labels = {actor.id: actor.label for actor in suite.actors}
	scores: dict[str, Any] = {}
	warnings = errors.copy()
	for phase, run in latest.items():
		if len(run.samples) != len(expected) or (
			{s.request_id for s in run.samples} != expected.keys()
		):
			raise ValueError("traffic run does not cover the complete frozen suite")
		if any(
			s.scheduled_seconds != expected[s.request_id].at_seconds
			or s.started_seconds < s.scheduled_seconds
			or s.completed_seconds < s.started_seconds
			or (s.succeeded and (s.status != "ok" or s.decision == "block"))
			for s in run.samples
		):
			raise ValueError("traffic result contradicts the frozen schedule or response outcome")
		seconds = max(suite.duration_seconds, max(s.completed_seconds for s in run.samples))
		by_label = {
			label: summarize_traffic(
				[s for s in run.samples if labels[expected[s.request_id].actor_id] == label],
				seconds,
			)
			for label in ("genuine", "distillation")
		}
		by_label["genuine"]["false_positive_percent"] = by_label["genuine"]["block_percent"]
		by_label["genuine"]["false_positive_count"] = by_label["genuine"]["blocked"]
		actors = {}
		for actor in suite.actors:
			samples = [s for s in run.samples if expected[s.request_id].actor_id == actor.id]
			first_l4 = min(
				(s for s in samples if s.level is not None and s.level >= 4),
				key=lambda s: s.started_seconds,
				default=None,
			)
			actors[actor.id] = {
				**actor.model_dump(),
				**summarize_traffic(samples, seconds),
				"first_l4_request_id": first_l4.request_id if first_l4 else None,
				"first_l4_started_seconds": first_l4.started_seconds if first_l4 else None,
				**(_progression(samples) if actor.label == "distillation" else {}),
			}
		windows = []
		for index in range(math.ceil(seconds / run.config.window_seconds)):
			start = index * run.config.window_seconds
			end = min(start + run.config.window_seconds, seconds)
			windows.append(
				{
					"start_seconds": start,
					"end_seconds": end,
					**{
						label: summarize_traffic(
							[
								s
								for s in run.samples
								if start <= s.completed_seconds
								and (
									s.completed_seconds < end
									or index == math.ceil(seconds / run.config.window_seconds) - 1
								)
								and labels[expected[s.request_id].actor_id] == label
							],
							end - start,
						)
						for label in ("genuine", "distillation")
					},
				}
			)
		scores[phase] = {
			"duration_seconds": seconds,
			**by_label,
			"actors": actors,
			"windows": windows,
		}
		if by_label["genuine"]["decision_coverage"] != 1:
			warnings.append(
				f"{phase}: incomplete X-Harness-Decision telemetry; "
				"false-positive rate is unavailable."
			)
		if by_label["distillation"]["decision_coverage"] != 1:
			warnings.append(f"{phase}: incomplete harness decisions for distillation attempts.")
		if any(v["schedule_delay_p95"] > 0.1 for v in by_label.values()):
			warnings.append(f"{phase}: client scheduling delay exceeds 100 ms at p95.")
	comparison = {}
	if "before" in scores and "after" in scores:
		for label in ("genuine", "distillation"):
			before, after = scores["before"][label], scores["after"][label]
			comparison[label] = {
				"success_delta_pp": after["success_percent"] - before["success_percent"],
				"throughput_reduction_percent": 100
				* (
					1
					- after["successful_requests_per_second"]
					/ before["successful_requests_per_second"]
				)
				if before["successful_requests_per_second"]
				else None,
				"latency_p95_delta_seconds": after["latency_p95"] - before["latency_p95"]
				if before["latency_p95"] is not None and after["latency_p95"] is not None
				else None,
			}
		before_models = {s.served_model for s in latest["before"].samples if s.served_model}
		after_models = {s.served_model for s in latest["after"].samples if s.served_model}
		if before_models and after_models and before_models != after_models:
			warnings.append("Returned model identity changed between traffic phases.")
	return {
		"id": key,
		"experiment": rows[0].config.experiment,
		"demo": rows[0].demo,
		"suite": suite.model_dump(mode="json"),
		"runs": {phase: r.model_dump(mode="json") for phase, r in latest.items()},
		"scores": scores,
		"comparison": comparison,
		"warnings": warnings,
	}
