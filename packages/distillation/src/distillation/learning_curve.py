"""Frozen AG News development learning curves; no model imports during planning."""

import hashlib
import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from distillation.benchmarks.models import FrozenSuite
from distillation.io import read_jsonl
from distillation.schemas import LedgerRecord, SFTRecord

SIZES = (50, 100, 200, 300, 500, 1000, 1500)
SCOPE = (
	"One domain (AG News), one prompt source (the frozen AG News template). "
	"Development evidence only."
)
RECIPE = {
	"base_model": "Qwen/Qwen3.5-2B-Base",
	"revision": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
	"tokenizer_model": "Qwen/Qwen3.5-9B",
	"tokenizer_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
	"learning_rate": 1e-4,
	"epochs": 3,
	"max_length": 2048,
	"batch_size": 4,
	"gradient_accumulation_steps": 4,
	"warmup_ratio": 0.03,
	"lora_rank": 16,
	"lora_alpha": 32,
	"lora_dropout": 0.05,
	"target_modules": "all-linear",
	"precision": "bf16",
}
MATH_RECIPE: dict[str, Any] = {
	**RECIPE,
	"max_length": 2048,
	"evaluation_max_new_tokens": 512,
	"overlength_policy": "error_before_training",
}


def digest(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_suffix(path.suffix + ".tmp")
	temporary.write_text(json.dumps(value, indent="\t", allow_nan=False) + "\n")
	temporary.replace(path)


def candidates(path: Path, source: str, seed: int) -> list[dict[str, Any]]:
	rows = []
	seen: set[str] = set()
	line_numbers = [i for i, text in enumerate(path.read_text().splitlines(), 1) if text.strip()]
	for line, record in zip(line_numbers, read_jsonl(path, LedgerRecord), strict=True):
		if record.task != "ag_news":
			continue
		if record.example_id in seen:
			raise ValueError("duplicate AG News example ID in ledger")
		seen.add(record.example_id)
		if record.quality.status != "accepted" or record.finish_reason not in {
			"stop",
			"eos",
			"end_turn",
			"completed",
		}:
			continue
		if record.split != "train":
			raise ValueError("learning-curve responses must belong to the training split")
		sft = SFTRecord(prompt=record.prompt, completion=record.completion)
		rows.append(
			{
				"example_id": record.example_id,
				"source": source,
				"ledger": str(path.resolve()),
				"line": line,
				"record_sha256": hashlib.sha256(record.model_dump_json().encode()).hexdigest(),
				"sft": sft.model_dump(mode="json"),
			}
		)
	return sorted(
		rows, key=lambda r: hashlib.sha256(f"{seed}:{r['example_id']}".encode()).hexdigest()
	)


def plan_curve(
	output: Path,
	development: Path,
	a1: Path,
	suite_path: Path,
	*,
	sizes: tuple[int, ...] = SIZES,
	seeds: tuple[int, ...] = (42, 43, 44),
	subset_seed: int = 42,
	fraction: float = 0.8,
	gpu_hours: tuple[float, float] = (4, 7),
) -> dict[str, Any]:
	if output.exists():
		raise ValueError("plan output must be a new directory")
	if (
		not sizes
		or len(set(sizes)) != len(sizes)
		or min(sizes) < 1
		or 1500 not in sizes
		or len(seeds) != 3
		or len(set(seeds)) != 3
		or not 0 < fraction <= 1
		or not all(math.isfinite(h) for h in gpu_hours)
		or not 0 <= gpu_hours[0] <= gpu_hours[1]
	):
		raise ValueError(
			"need positive distinct sizes including 1500, three seeds and valid estimates"
		)
	suite = FrozenSuite.model_validate_json(suite_path.read_text())
	suite.verify()
	items = [i for i in suite.items if i.benchmark == "ag_news"]
	if len(items) != 1000 or any(i.metric != "label_accuracy" for i in items):
		raise ValueError("need the 1,000-item AG News development suite")
	dev = candidates(development, "development", subset_seed)
	extension = candidates(a1, "A1/s42/undefended", subset_seed)
	if len(dev) < 500:
		raise ValueError("need 500 accepted development responses")
	ordered = dev[:500]
	ids = {r["example_id"] for r in ordered}
	prompts = {json.dumps(r["sft"]["prompt"], sort_keys=True) for r in ordered}
	for row in extension:
		prompt = json.dumps(row["sft"]["prompt"], sort_keys=True)
		if row["example_id"] not in ids and prompt not in prompts:
			ordered.append(row)
			ids.add(row["example_id"])
			prompts.add(prompt)
	if len(ordered) < max(sizes):
		raise ValueError("not enough distinct accepted A1 responses")
	if len({json.dumps(r["sft"]["prompt"], sort_keys=True) for r in dev[:500]}) != 500:
		raise ValueError("duplicate development prompts")
	eval_prompts = {json.dumps([m.model_dump() for m in i.messages], sort_keys=True) for i in items}
	eval_ids = {i.id for i in items}
	if any(
		r["example_id"] in eval_ids
		or json.dumps(r["sft"]["prompt"], sort_keys=True) in eval_prompts
		for r in ordered[: max(sizes)]
	):
		raise ValueError("training/development evaluation overlap")
	output.mkdir(parents=True)
	write(output / "suite.json", [i.model_dump(mode="json") for i in items])
	files = {"suite.json": digest(output / "suite.json")}
	for n in sorted(sizes):
		write(output / f"subset-{n}.json", ordered[:n])
		files[f"subset-{n}.json"] = digest(output / f"subset-{n}.json")
	plan = {
		"version": "ag-news-learning-curve-v1",
		"scope": SCOPE,
		"sizes": sorted(sizes),
		"seeds": list(seeds),
		"subset_seed": subset_seed,
		"target_fraction": fraction,
		"recipe": RECIPE,
		"files": files,
		"inputs": {str(p.resolve()): digest(p) for p in (development, a1, suite_path)},
		"estimated_gpu_hours": [h * len(sizes) * len(seeds) / 21 for h in gpu_hours],
		"estimate_note": "Unmeasured projection for RTX 4000 Ada, scaled by cell count; "
		"excludes base evaluation and acquisition. Supply a revised 21-cell estimate if needed.",
		"cells": [{"n": n, "seed": s} for n in sorted(sizes) for s in seeds],
	}
	write(output / "plan.json", plan)
	return plan


def load_plan(output: Path) -> dict[str, Any]:
	plan: dict[str, Any] = json.loads((output / "plan.json").read_text())
	for name, expected in plan["files"].items():
		if digest(output / name) != expected:
			raise ValueError(f"plan artifact hash mismatch: {name}")
	return plan


def validate_scores(scores: dict[str, Any], ids: set[str]) -> None:
	if set(scores) != ids or any(
		type(v) not in (int, float) or v not in (0, 1) for v in scores.values()
	):
		raise ValueError(
			"scores must contain every suite item exactly once with binary correctness"
		)


def execute_curve(
	output: Path,
	runner: Callable[[Path, dict[str, Any], int | None, int], dict[str, Any]] | None = None,
	preflight: Callable[[Path, dict[str, Any], list[str]], None] | None = None,
) -> dict[str, int]:
	"""A committed score file marks completion; interrupted cells restart training.

	A math plan checks the largest teacher and gold subsets against the length limit before
	any cell trains, so an overlong example never surfaces after other cells finished. With a
	custom runner and no preflight, the check is skipped.
	"""
	if runner is None:
		from distillation.learning_curve_runner import preflight as check_lengths
		from distillation.learning_curve_runner import run_cell

		runner = run_cell
		preflight = preflight or check_lengths
	plan = load_plan(output)
	contract = digest(output / "plan.json")
	ids = {i["id"] for i in json.loads((output / "suite.json").read_text())}
	completed = skipped = 0
	if preflight is not None and plan["version"] == "math-learning-curve-v1":
		largest = max(plan["sizes"])
		files = [f"subset-{largest}.json"]
		if any(c.get("arm") == "gold" for c in plan["cells"]):
			files.append(f"gold-subset-{largest}.json")
		preflight(output, plan["recipe"], files)
	for cell in [{"n": None, "seed": plan["seeds"][0]}, *plan["cells"]]:
		n, seed = cell["n"], cell["seed"]
		arm = cell.get("arm", "teacher")
		name = "base" if n is None else f"{'gold-' if arm == 'gold' else ''}n{n}-s{seed}"
		path = output / "scores" / f"{name}.json"
		if path.exists():
			result = json.loads(path.read_text())
			if (
				result["contract"] != contract
				or result["n"] != n
				or result["seed"] != seed
				or result.get("arm", "teacher") != arm
			):
				raise ValueError("completed cell belongs to a different plan or cell")
			validate_scores(result["scores"], ids)
			skipped += 1
			continue
		recipe = {**plan["recipe"], "_arm": arm} if arm == "gold" else plan["recipe"]
		scores = runner(output, recipe, n, seed)
		validate_scores(scores, ids)
		result = {"contract": contract, "n": n, "seed": seed, "scores": scores}
		if arm == "gold":
			result["arm"] = arm
		write(path, result)
		completed += 1
	return {"completed": completed, "skipped": skipped}


def summarize_curve(output: Path, *, replicates: int = 2000, seed: int = 42) -> dict[str, Any]:
	"""Crossed bootstrap: resample training seeds and paired test items independently."""
	plan = load_plan(output)
	if replicates < 100:
		raise ValueError("need at least 100 bootstrap replicates")
	contract = digest(output / "plan.json")
	ids = [i["id"] for i in json.loads((output / "suite.json").read_text())]

	def scores(name: str, n: int | None, training_seed: int, arm: str = "teacher") -> list[float]:
		result = json.loads((output / "scores" / f"{name}.json").read_text())
		if (
			result["contract"] != contract
			or result["n"] != n
			or result["seed"] != training_seed
			or result.get("arm", "teacher") != arm
		):
			raise ValueError("score plan or cell mismatch")
		validate_scores(result["scores"], set(ids))
		return [float(result["scores"][i]) for i in ids]

	base = scores("base", None, plan["seeds"][0])
	rng = random.Random(seed)
	rows: list[dict[str, Any]] = []
	for arm in (
		["teacher", "gold"] if any(c.get("arm") == "gold" for c in plan["cells"]) else ["teacher"]
	):
		for n in plan["sizes"]:
			gains = [
				[
					v - b
					for v, b in zip(
						scores(f"{'gold-' if arm == 'gold' else ''}n{n}-s{s}", n, s, arm),
						base,
						strict=True,
					)
				]
				for s in plan["seeds"]
			]
			means = [100 * sum(g) / len(ids) for g in gains]
			samples = []
			for _ in range(replicates):
				selected = rng.choices(gains, k=len(gains))
				indices = rng.choices(range(len(ids)), k=len(ids))
				samples.append(
					100
					* sum(sum(g[i] for i in indices) for g in selected)
					/ (len(selected) * len(ids))
				)
			samples.sort()
			rows.append(
				{
					"n": n,
					**({"arm": arm} if plan["version"] == "math-learning-curve-v1" else {}),
					"seed_gains_pp": means,
					"mean_gain_pp": sum(means) / len(means),
					"ci95_pp": [samples[int(0.025 * replicates)], samples[int(0.975 * replicates)]],
				}
			)
	reference_n = 1500 if plan["version"] == "ag-news-learning-curve-v1" else max(plan["sizes"])
	reference = next(
		r["mean_gain_pp"]
		for r in rows
		if r["n"] == reference_n and r.get("arm", "teacher") == "teacher"
	)
	target = reference * plan["target_fraction"] if reference > 0 else None
	result = {
		"scope": plan["scope"],
		"results": [r for r in rows if r.get("arm", "teacher") == "teacher"],
		**(
			{"gold_results": [r for r in rows if r.get("arm") == "gold"]}
			if any(r.get("arm") == "gold" for r in rows)
			else {}
		),
		"target_fraction": plan["target_fraction"],
		"reference_gain_pp": reference,
		"target_gain_pp": target,
		"smallest_n": next(
			(
				r["n"]
				for r in rows
				if r.get("arm", "teacher") == "teacher"
				and target is not None
				and r["mean_gain_pp"] >= target
			),
			None,
		),
		"bootstrap": {
			"replicates": replicates,
			"seed": seed,
			"method": "percentile, crossed seeds and paired test items",
		},
		"warning": (
			"Three seeds give unstable intervals. "
			if len(plan["seeds"]) == 3
			else "Training-seed intervals depend on the selected seed panel. "
		)
		+ "Conditional on these nested subsets; "
		"not independent campaigns. Threshold uses point means, not a confidence guarantee. "
		f"Nonpositive {reference_n}-response gain leaves the threshold undefined.",
	}
	write(output / "summary.json", result)
	return result


def math_family(question: str) -> str:
	"""Reproduce the GSM8K builder's normalized-question family key."""
	from distillation.tasks import GSM8K_EVAL_SUFFIX, normalized

	return hashlib.sha256(
		normalized(question.removesuffix(GSM8K_EVAL_SUFFIX)).encode()
	).hexdigest()[:24]


def plan_math_curve(
	output: Path,
	ledger: Path,
	suite_path: Path,
	*,
	gold_path: Path | None = None,
	sizes: tuple[int, ...] = (300, 1000),
	seeds: tuple[int, ...] = (42, 43, 44),
	subset_seed: int = 42,
	fraction: float = 0.8,
	include_pending: bool = False,
) -> dict[str, Any]:
	"""Freeze nested GSM8K subsets and optional matched gold completions."""
	from distillation.tasks import GSM8K_EVAL_SUFFIX, GoldRecord

	if output.exists():
		raise ValueError("plan output must be a new directory")
	if (
		not sizes
		or min(sizes) < 1
		or len(set(sizes)) != len(sizes)
		or not seeds
		or len(set(seeds)) != len(seeds)
		or not 0 < fraction <= 1
	):
		raise ValueError("need positive distinct sizes, distinct seeds and a valid fraction")
	suite = FrozenSuite.model_validate_json(suite_path.read_text())
	suite.verify()
	items = suite.items
	if any(i.benchmark != "gsm8k" or i.metric != "numeric_exact_match" for i in items):
		raise ValueError("need a GSM8K numeric exact match development suite")
	eval_ids = {i.id for i in items} | {i.id.replace(":", "-") for i in items}
	eval_prompts = {json.dumps([m.model_dump() for m in i.messages], sort_keys=True) for i in items}
	eval_families = {math_family(i.question) for i in items}
	rows: list[dict[str, Any]] = []
	seen: set[str] = set()
	eligible = {"accepted", "pending"} if include_pending else {"accepted"}
	line_numbers = [i for i, text in enumerate(ledger.read_text().splitlines(), 1) if text.strip()]
	for line, record in zip(line_numbers, read_jsonl(ledger, LedgerRecord), strict=True):
		if record.task != "gsm8k":
			continue
		if record.example_id in seen:
			raise ValueError("duplicate math example ID in ledger")
		seen.add(record.example_id)
		if (
			record.split != "train"
			or record.quality.status not in eligible
			or record.finish_reason not in {"stop", "eos", "end_turn", "completed"}
		):
			continue
		prompt = [m.model_dump(mode="json") for m in record.prompt]
		question = next((m.content for m in record.prompt if m.role == "user"), "")
		# Reconstructing the builder key is exact only for the builder's own template; another
		# wrapper around an evaluation question would hash differently and slip through.
		if not question.endswith(GSM8K_EVAL_SUFFIX):
			raise ValueError(
				f"math prompt {record.example_id} does not use the task builder template, "
				"so its question key cannot be checked against the evaluation suite"
			)
		family = record.source.group_id or math_family(question)
		if (
			record.example_id in eval_ids
			or json.dumps(prompt, sort_keys=True) in eval_prompts
			or family in eval_families
			or math_family(question) in eval_families
		):
			raise ValueError(
				"training/development evaluation overlap by ID, prompt or source family"
			)
		if record.source.group_id and family != math_family(question):
			raise ValueError("math source family does not match the task builder question key")
		rows.append(
			{
				"example_id": record.example_id,
				"source": "teacher",
				"family": family,
				"ledger": str(ledger.resolve()),
				"line": line,
				"record_sha256": hashlib.sha256(record.model_dump_json().encode()).hexdigest(),
				"sft": SFTRecord(prompt=record.prompt, completion=record.completion).model_dump(
					mode="json"
				),
			}
		)
	rows.sort(key=lambda r: hashlib.sha256(f"{subset_seed}:{r['example_id']}".encode()).hexdigest())
	if len(rows) < max(sizes):
		raise ValueError("not enough eligible math responses")
	selected = rows[: max(sizes)]
	if len({json.dumps(r["sft"]["prompt"], sort_keys=True) for r in selected}) != len(selected):
		raise ValueError("duplicate training prompts")
	gold: dict[str, str] = {}
	if gold_path is not None:
		for gold_record in read_jsonl(gold_path, GoldRecord):
			if gold_record.example_id in gold:
				raise ValueError("duplicate gold example ID")
			gold[gold_record.example_id] = gold_record.completion
		if any(r["example_id"] not in gold for r in selected):
			raise ValueError("missing gold completion for selected example")
	output.mkdir(parents=True)
	write(output / "suite.json", [i.model_dump(mode="json") for i in items])
	files = {"suite.json": digest(output / "suite.json")}
	for n in sorted(sizes):
		write(output / f"subset-{n}.json", selected[:n])
		files[f"subset-{n}.json"] = digest(output / f"subset-{n}.json")
		if gold_path is not None:
			gold_rows = [
				{
					**r,
					"sft": {
						**r["sft"],
						"completion": [{"role": "assistant", "content": gold[r["example_id"]]}],
					},
				}
				for r in selected[:n]
			]
			write(output / f"gold-subset-{n}.json", gold_rows)
			files[f"gold-subset-{n}.json"] = digest(output / f"gold-subset-{n}.json")
	inputs = (ledger, suite_path) if gold_path is None else (ledger, suite_path, gold_path)
	plan = {
		"version": "math-learning-curve-v1",
		"scope": "GSM8K development numeric reasoning.",
		"include_pending": include_pending,
		"sizes": sorted(sizes),
		"seeds": list(seeds),
		"subset_seed": subset_seed,
		"target_fraction": fraction,
		"recipe": MATH_RECIPE,
		"files": files,
		"inputs": {str(p.resolve()): digest(p) for p in inputs},
		"training_ledgers": [str(ledger.resolve())],
		"cells": [
			{"n": n, "seed": seed, "arm": arm}
			for arm in (["teacher", "gold"] if gold_path else ["teacher"])
			for n in sorted(sizes)
			for seed in seeds
		],
	}
	write(output / "plan.json", plan)
	return plan
