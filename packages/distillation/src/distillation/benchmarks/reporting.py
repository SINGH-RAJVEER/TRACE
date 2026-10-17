"""Comparable cohorts, Wilson score intervals, and paired bootstrap gains."""

import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from distillation.benchmarks.models import SLOTS, EvaluationRun, SampleResult, load_suite


def wilson(correct: int, total: int) -> list[float]:
	if not total:
		return [0, 0]
	z = 1.959963984540054
	p = correct / total
	denominator = 1 + z * z / total
	center = (p + z * z / (2 * total)) / denominator
	radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
	return [100 * max(0, center - radius), 100 * min(1, center + radius)]


def percentile(values: list[float], fraction: float) -> float | None:
	if not values:
		return None
	ordered = sorted(values)
	position = (len(ordered) - 1) * fraction
	lower = math.floor(position)
	upper = math.ceil(position)
	return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean_ci(values: list[float]) -> list[float]:
	"""Normal-approximation 95% interval for a mean, in percent."""
	if len(values) < 2:
		return [100 * values[0]] * 2 if values else [0, 0]
	mean = statistics.fmean(values)
	radius = 1.959963984540054 * statistics.stdev(values) / math.sqrt(len(values))
	return [100 * (mean - radius), 100 * (mean + radius)]


def _f1_summary(samples: list[SampleResult]) -> dict[str, Any]:
	values = [sample.f1 for sample in samples if sample.f1 is not None]
	if not values or len(values) != len(samples):
		return {}
	return {"f1": 100 * statistics.fmean(values), "f1_ci95": mean_ci(values)}


def summarize(samples: list[SampleResult]) -> dict[str, Any]:
	correct = sum(sample.correct for sample in samples)
	latencies = [sample.latency_seconds for sample in samples if sample.status != "error"]
	known = [
		sample
		for sample in samples
		if sample.usage.output_tokens is not None and sample.status != "error"
	]
	seconds = sum(sample.latency_seconds for sample in known)
	tokens = sum(sample.usage.output_tokens or 0 for sample in known)
	return {
		"primary_metric": "token_f1"
		if samples and all(s.benchmark == "squad" for s in samples)
		else "pass_at_1"
		if samples and all(s.benchmark == "coding" for s in samples)
		else "accuracy",
		"primary_score": 100
		* statistics.fmean(
			(s.f1 or 0) if s.benchmark == "squad" else float(s.correct) for s in samples
		)
		if samples
		else None,
		"n": len(samples),
		"correct": correct,
		"score": 100 * correct / len(samples) if samples else None,
		"ci95": wilson(correct, len(samples)),
		"statuses": dict(Counter(s.status for s in samples)),
		"latency_p50": percentile(latencies, 0.5),
		"latency_p95": percentile(latencies, 0.95),
		"tokens_per_second": tokens / seconds if seconds else None,
		"token_coverage": len(known) / len(samples) if samples else 0,
		**_f1_summary(samples),
	}


def paired(before: list[SampleResult], after: list[SampleResult]) -> dict[str, Any]:
	"""Paired source-document bootstrap; invalid responses remain incorrect."""
	import numpy as np

	left = {sample.item_id: sample for sample in before}
	right = {sample.item_id: sample for sample in after}
	if left.keys() != right.keys() or not left:
		raise ValueError("paired scores require the same nonempty item set")
	groups: dict[str, list[float]] = defaultdict(list)
	binary = []
	for key, sample in left.items():
		other = right[key]
		if (sample.benchmark, sample.category) != (other.benchmark, other.category):
			raise ValueError("paired source metadata changed")
		binary.append(int(other.correct) - int(sample.correct))
		delta = ((other.f1 or 0) - (sample.f1 or 0)) if sample.benchmark == "squad" else binary[-1]
		group = sample.category if sample.benchmark == "squad" else sample.item_id
		groups[group].append(float(delta))
	sums = np.array([sum(values) for values in groups.values()])
	sizes = np.array([len(values) for values in groups.values()])
	rng = np.random.default_rng(42)
	bootstrap = []
	for _ in range(2000):
		indices = rng.integers(0, len(groups), size=len(groups))
		bootstrap.append(float(100 * sums[indices].sum() / sizes[indices].sum()))
	pairs = [
		(left[key].prediction, right[key].prediction)
		for key in left
		if left[key].benchmark != "coding"
		and left[key].status == right[key].status == "ok"
		and left[key].prediction is not None
		and right[key].prediction is not None
	]
	agreed = sum(a == b for a, b in pairs)
	primary = 100 * float(sums.sum()) / len(left)
	ci = [float(x) for x in np.percentile(bootstrap, [2.5, 97.5])]
	result = {
		"n": len(left),
		"delta_pp": 100 * statistics.fmean(binary),
		"primary_delta_pp": primary,
		"primary_ci95": ci,
		"ci95": ci,
		"resampling_unit": "source_document",
		"clusters": len(groups),
		"improved": binary.count(1),
		"regressed": binary.count(-1),
		"both_correct": sum(left[k].correct and right[k].correct for k in left),
		"both_wrong": sum(not left[k].correct and not right[k].correct for k in left),
		"answer_agreement": 100 * agreed / len(pairs) if pairs else None,
		"agreement_numerator": agreed,
		"agreement_n": len(pairs),
		"agreement_coverage": len(pairs) / len(left),
	}
	if any(s.benchmark == "coding" for s in before):
		result["functional_agreement"] = None
		result["functional_agreement_status"] = (
			"not_measured; assertion outcomes do not provide output traces"
		)
		result["pass_fail_agreement"] = (
			100 * sum(left[k].correct == right[k].correct for k in left) / len(left)
		)
	if all(s.benchmark == "squad" for s in before):
		result.update(f1_delta_pp=primary, f1_ci95=ci)
	return result


def discover_runs(directory: Path) -> tuple[list[EvaluationRun], list[str]]:
	runs = []
	errors = []
	for path in sorted(directory.glob("*.json")):
		if path.name.startswith("suite-"):
			continue
		try:
			runs.append(EvaluationRun.model_validate_json(path.read_text()))
		except ValueError:
			errors.append(f"Cannot read run {path.name}")
	return runs, errors


def experiments(directory: Path) -> dict[str, Any]:
	runs, errors = discover_runs(directory)
	groups: dict[str, list[EvaluationRun]] = defaultdict(list)
	for run in runs:
		groups[run.comparison_key()].append(run)
	return {
		"experiments": [
			{
				"id": key,
				"name": rows[0].experiment,
				"demo": rows[0].demo,
				"suite_sha256": rows[0].suite_sha256,
				"slots": sorted({row.slot for row in rows}),
				"updated_at": max(row.completed_at for row in rows).isoformat(),
			}
			for key, rows in sorted(
				groups.items(),
				key=lambda group: max(r.completed_at for r in group[1]),
				reverse=True,
			)
		],
		"errors": errors,
	}


def build_report(directory: Path, comparison_key: str) -> dict[str, Any]:
	runs, errors = discover_runs(directory)
	rows = [run for run in runs if run.comparison_key() == comparison_key]
	if not rows:
		raise ValueError("comparison does not exist")
	latest = {
		slot: max((run for run in rows if run.slot == slot), key=lambda r: r.completed_at)
		for slot in SLOTS
		if any(run.slot == slot for run in rows)
	}
	suite = load_suite(directory / f"suite-{rows[0].suite_sha256}.json")
	expected = {item.id: item for item in suite.items}
	for run in latest.values():
		if len(run.samples) != len(expected) or {s.item_id for s in run.samples} != expected.keys():
			raise ValueError("run does not cover the complete frozen suite")
		if any(
			s.benchmark != expected[s.item_id].benchmark
			or s.category != expected[s.item_id].category
			for s in run.samples
		):
			raise ValueError("run sample metadata does not match the frozen suite")
	benchmarks = sorted({item.benchmark for item in suite.items})
	scores = {
		slot: {
			benchmark: summarize([s for s in run.samples if s.benchmark == benchmark])
			for benchmark in benchmarks
		}
		for slot, run in latest.items()
	}
	slices: dict[str, Any] = {}
	for benchmark in benchmarks:
		for category in sorted({i.category for i in suite.items if i.benchmark == benchmark}):
			slices[f"{benchmark}/{category}"] = {
				slot: summarize(
					[s for s in run.samples if s.benchmark == benchmark and s.category == category]
				)
				for slot, run in latest.items()
			}
	comparisons: dict[str, Any] = {}
	for name, before, after in (
		("student_gain", "student_before", "student_after"),
		("teacher_drift", "teacher", "teacher_after"),
		("teacher_student_before", "teacher", "student_before"),
		("teacher_student_after", "teacher_after", "student_after"),
	):
		if before in latest and after in latest:
			comparisons[name] = {
				benchmark: paired(
					[s for s in latest[before].samples if s.benchmark == benchmark],
					[s for s in latest[after].samples if s.benchmark == benchmark],
				)
				for benchmark in benchmarks
			}
	warnings = errors.copy()
	if any(not run.training_sha256 for run in latest.values()):
		warnings.append(
			"Some runs have no supplied training corpus. Contamination is unchecked for those runs."
		)
	if "student_before" in latest and "student_after" in latest:
		a, b = latest["student_before"].model, latest["student_after"].model
		if a.backend == b.backend == "local" and a.model_dump(
			exclude={"adapter_path"}
		) != b.model_dump(exclude={"adapter_path"}):
			warnings.append(
				"Student base model or inference configuration changed. "
				"Gains cannot be attributed to distillation alone."
			)
	if "teacher" in latest and "teacher_after" in latest:
		a, b = latest["teacher"].model, latest["teacher_after"].model
		if (a.model, a.revision) != (b.model, b.revision):
			warnings.append(
				"Teacher identity changed between phases. Drift includes a model change."
			)
	indexed = {
		slot: {sample.item_id: sample for sample in run.samples} for slot, run in latest.items()
	}
	return {
		"id": comparison_key,
		"experiment": rows[0].experiment,
		"demo": rows[0].demo,
		"suite": suite.model_dump(mode="json"),
		"benchmarks": benchmarks,
		"runs": {
			slot: run.model_dump(mode="json", exclude={"samples"}) for slot, run in latest.items()
		},
		"scores": scores,
		"slices": slices,
		"comparisons": comparisons,
		"warnings": warnings,
		"samples": [
			{
				"item": item.model_dump(mode="json"),
				"results": {
					slot: indexed[slot][item.id].model_dump(mode="json")
					for slot, run in latest.items()
				},
			}
			for item in suite.items
		],
	}
