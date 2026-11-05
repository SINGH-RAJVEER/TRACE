"""Chronological checkpoint student learning: plan, execute and summarize GSM8K cells.

A checkpoint export freezes what an acquirer held at each point in time. This module trains
one student per reached checkpoint and seed on exactly that snapshot's eligible rows, then
reports where a preregistered accuracy target is first observed. Planning and summary need no
GPU dependencies; execution lazy-loads the existing pinned runner unless one is injected.
"""

import hashlib
import importlib
import json
import math
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from distillation.benchmarks.models import FrozenSuite
from distillation.io import canonical_json, read_jsonl
from distillation.learning_curve import (
	MATH_RECIPE,
	digest,
	math_family,
	validate_scores,
	write,
)
from distillation.schemas import LedgerRecord, SFTRecord

VERSION = "checkpoint-learning-v1"
SCOPE = "GSM8K numeric reasoning on chronological acquisition checkpoints. Development evidence."
SUPPORTED_TASKS = ("gsm8k",)
BASE = "base"
TERMINAL_FINISH = {"stop", "eos", "end_turn", "completed"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
KNOWN_ENTRY_KEYS = {
	"id",
	"kind",
	"reached",
	"cutoff_seconds",
	"ledger_file",
	"ledger_sha256",
	"example_ids",
	"record_count",
}
CLAIM_BOUNDARY = (
	"Point estimates from a fixed seed panel on one development suite. Reaching the target is "
	"an observation about these students, not a confidence statement and not evidence that "
	"any defense prevents distillation in general. Unreached checkpoints are censored "
	"unknowns, never zero-accuracy failures."
)

Loader = Callable[[Path], dict[str, Any]]
Runner = Callable[[Path, dict[str, Any], int | None, int], dict[str, Any]]
Preflight = Callable[[Path, dict[str, Any], list[str]], None]


def load_export(export: Path) -> dict[str, Any]:
	"""Validate an export through the exporter's own loader, resolved lazily."""
	module = importlib.import_module("distillation.checkpoints")
	manifest: dict[str, Any] = module.load_checkpoint_export(export)
	return manifest


def run_export(campaign: Path, output: Path, response_counts: Sequence[int]) -> dict[str, Any]:
	module = importlib.import_module("distillation.checkpoints")
	manifest: dict[str, Any] = module.export_campaign_checkpoints(
		campaign, output, response_counts=tuple(response_counts)
	)
	return manifest


def _within(root: Path, path: Path) -> bool:
	return path == root or root in path.parents


def _line_numbers(path: Path) -> list[int]:
	return [i for i, text in enumerate(path.read_text().splitlines(), 1) if text.strip()]


def _ledger_path(root: Path, entry: Mapping[str, Any]) -> Path:
	name = entry["ledger_file"]
	relative = Path(name)
	if (
		not isinstance(name, str)
		or "\\" in name
		or relative.is_absolute()
		or ".." in relative.parts
		or not relative.parts
	):
		raise ValueError(f"checkpoint {entry['id']}: ledger path must be relative and contained")
	path = (root / relative).resolve()
	if not _within(root, path):
		raise ValueError(f"checkpoint {entry['id']}: ledger path escapes the export root")
	if not path.is_file():
		raise ValueError(f"checkpoint {entry['id']}: ledger file is missing")
	return path


def _validate_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
	entries = manifest.get("checkpoints")
	if not isinstance(entries, list) or not entries:
		raise ValueError("export manifest has no checkpoints")
	seen: set[str] = set()
	for entry in entries:
		if not isinstance(entry, dict) or not set(entry) >= KNOWN_ENTRY_KEYS:
			raise ValueError("checkpoint entry is missing required keys")
		name = entry["id"]
		if not isinstance(name, str) or not SAFE_ID.match(name) or name == BASE:
			raise ValueError(f"checkpoint id {name!r} is not a unique filesystem-safe label")
		if name in seen:
			raise ValueError(f"duplicate checkpoint id {name}")
		seen.add(name)
		ids = entry["example_ids"]
		cutoff = entry["cutoff_seconds"]
		if (
			not isinstance(entry["kind"], str)
			or not isinstance(entry["reached"], bool)
			or not isinstance(ids, list)
			or not all(isinstance(i, str) for i in ids)
			or len(set(ids)) != len(ids)
			or type(entry["record_count"]) is not int
			or entry["record_count"] != len(ids)
			or (
				cutoff is not None
				and (
					isinstance(cutoff, bool)
					or not isinstance(cutoff, int | float)
					or not math.isfinite(cutoff)
				)
			)
		):
			raise ValueError(f"checkpoint {name}: malformed fields")
		if entry["reached"]:
			if not isinstance(entry["ledger_file"], str) or not isinstance(
				entry["ledger_sha256"], str
			):
				raise ValueError(f"checkpoint {name}: reached checkpoint needs a ledger")
		elif entry["ledger_file"] is not None or entry["ledger_sha256"] is not None or ids:
			raise ValueError(f"checkpoint {name}: unreached checkpoint must have no ledger")
	return list(entries)


def _chronology(entries: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
	"""Reached checkpoints with a cutoff, ordered; each id set must contain the previous."""
	timeline = sorted(
		(e for e in entries if e["reached"] and e["cutoff_seconds"] is not None),
		key=lambda e: (e["cutoff_seconds"], e["record_count"], e["id"]),
	)
	for earlier, later in zip(timeline, timeline[1:], strict=False):
		if not set(earlier["example_ids"]) <= set(later["example_ids"]):
			raise ValueError(
				f"checkpoint {later['id']} does not contain every example of the earlier "
				f"checkpoint {earlier['id']}; membership is not chronological"
			)
	return timeline


def _evaluation_view(suite: FrozenSuite) -> tuple[set[str], set[str], set[str]]:
	items = suite.items
	if any(i.benchmark not in SUPPORTED_TASKS for i in items) or any(
		i.metric != "numeric_exact_match" for i in items
	):
		raise ValueError("need a single-task GSM8K numeric exact match development suite")
	ids = {i.id for i in items} | {i.id.replace(":", "-") for i in items}
	prompts = {json.dumps([m.model_dump() for m in i.messages], sort_keys=True) for i in items}
	families = {math_family(i.question) for i in items}
	return ids, prompts, families


def _training_rows(
	record: LedgerRecord,
	line: int,
	ledger_name: str,
	evaluation: tuple[set[str], set[str], set[str]],
	eligible_status: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
	"""Return a trainable row, or the single reason the record is filtered."""
	from distillation.tasks import GSM8K_EVAL_SUFFIX

	if record.task not in SUPPORTED_TASKS:
		return None, "other_task"
	if record.split != "train":
		return None, "held_out_split"
	if record.quality.status == "rejected":
		return None, "rejected"
	if record.quality.status not in eligible_status:
		return None, "pending_excluded"
	if record.finish_reason not in TERMINAL_FINISH:
		return None, "nonterminal_finish"
	eval_ids, eval_prompts, eval_families = evaluation
	prompt = [m.model_dump(mode="json") for m in record.prompt]
	question = next((m.content for m in record.prompt if m.role == "user"), "")
	if not question.endswith(GSM8K_EVAL_SUFFIX):
		raise ValueError(
			f"math prompt {record.example_id} does not use the task builder template, "
			"so its question key cannot be checked against the evaluation suite"
		)
	key = math_family(question)
	family = record.source.group_id or key
	if (
		record.example_id in eval_ids
		or json.dumps(prompt, sort_keys=True) in eval_prompts
		or family in eval_families
		or key in eval_families
	):
		raise ValueError("training/development evaluation overlap by ID, prompt or source family")
	if record.source.group_id and family != key:
		raise ValueError("math source family does not match the task builder question key")
	return {
		"example_id": record.example_id,
		"source": "teacher",
		"family": family,
		"ledger": ledger_name,
		"line": line,
		"record_sha256": hashlib.sha256(record.model_dump_json().encode()).hexdigest(),
		"sft": SFTRecord(prompt=record.prompt, completion=record.completion).model_dump(
			mode="json"
		),
	}, None


def plan_checkpoint_curve(
	output: Path,
	export: Path,
	suite_path: Path,
	*,
	target_accuracy: float,
	task: str = "gsm8k",
	seeds: tuple[int, ...] = (42, 43, 44),
	include_pending: bool = False,
	recipe: Mapping[str, Any] | None = None,
	loader: Loader | None = None,
) -> dict[str, Any]:
	"""Freeze per-checkpoint training sets, the suite, seeds, target and every input hash."""
	if task not in SUPPORTED_TASKS:
		raise ValueError(f"unsupported task {task!r}; supported: {', '.join(SUPPORTED_TASKS)}")
	if output.exists():
		raise ValueError("plan output must be a new directory")
	root = export.resolve()
	if _within(root, output.resolve()):
		raise ValueError("plan output must not sit inside the export")
	if (
		isinstance(target_accuracy, bool)
		or not math.isfinite(target_accuracy)
		or not 0 < target_accuracy <= 1
	):
		raise ValueError("a preregistered target_accuracy in (0, 1] is required")
	if (
		not seeds
		or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
		or len(set(seeds)) != len(seeds)
	):
		raise ValueError("need unique integer seeds in [0, 2**32)")
	manifest = (loader or load_export)(export)
	entries = _validate_entries(manifest)
	timeline = _chronology(entries)
	manifest_path = root / "manifest.json"
	inputs = {str(p.resolve()): digest(p) for p in (manifest_path, suite_path)}
	for source in manifest.get("inputs", {}).values():
		path = Path(source["path"]).resolve()
		if not path.is_file() or digest(path) != source["sha256"]:
			raise ValueError(f"export source artifact hash mismatch: {path}")
		inputs[str(path)] = source["sha256"]
	suite = FrozenSuite.model_validate_json(suite_path.read_text())
	suite.verify()
	evaluation = _evaluation_view(suite)
	eligible_status = {"accepted", "pending"} if include_pending else {"accepted"}

	records_seen: dict[str, str] = {}
	ledger_hashes: dict[str, str] = {}
	checkpoints: list[dict[str, Any]] = []
	stages: dict[str, list[dict[str, Any]]] = {}
	for entry in entries:
		info: dict[str, Any] = {
			"id": entry["id"],
			"kind": entry["kind"],
			"reached": entry["reached"],
			"cutoff_seconds": entry["cutoff_seconds"],
			"acquired_count": entry["record_count"],
			"ledger_file": entry["ledger_file"],
			"ledger_sha256": entry["ledger_sha256"],
			"evidence": {k: v for k, v in entry.items() if k not in KNOWN_ENTRY_KEYS},
		}
		if not entry["reached"]:
			checkpoints.append(
				{**info, "status": "censored", "trainable_count": None, "filtered": None}
			)
			continue
		path = _ledger_path(root, entry)
		if digest(path) != entry["ledger_sha256"]:
			raise ValueError(f"checkpoint {entry['id']}: ledger hash mismatch")
		ledger_hashes[entry["ledger_file"]] = entry["ledger_sha256"]
		records = read_jsonl(path, LedgerRecord)
		if len(records) != entry["record_count"] or [r.example_id for r in records] != list(
			entry["example_ids"]
		):
			raise ValueError(
				f"checkpoint {entry['id']}: ledger records do not match the manifest examples"
			)
		rows: list[dict[str, Any]] = []
		filtered: dict[str, int] = {}
		for line, record in zip(_line_numbers(path), records, strict=True):
			fingerprint = hashlib.sha256(record.model_dump_json().encode()).hexdigest()
			if records_seen.setdefault(record.example_id, fingerprint) != fingerprint:
				raise ValueError(f"example {record.example_id} differs between checkpoint ledgers")
			row, reason = _training_rows(
				record, line, entry["ledger_file"], evaluation, eligible_status
			)
			if row is not None:
				rows.append(row)
			elif reason is not None:
				filtered[reason] = filtered.get(reason, 0) + 1
		if len({json.dumps(r["sft"]["prompt"], sort_keys=True) for r in rows}) != len(rows):
			raise ValueError(f"checkpoint {entry['id']}: duplicate training prompts")
		checkpoints.append(
			{
				**info,
				"status": "trained" if rows else "base",
				"trainable_count": len(rows),
				"filtered": dict(sorted(filtered.items())),
				"training_ids_sha256": hashlib.sha256(
					canonical_json([r["example_id"] for r in rows]).encode()
				).hexdigest(),
			}
		)
		if rows:
			stages[entry["id"]] = rows
	if not any(c["status"] != "censored" for c in checkpoints):
		raise ValueError("no checkpoint was reached; nothing to plan")

	output.mkdir(parents=True)
	items = [i.model_dump(mode="json") for i in suite.items]
	write(output / "suite.json", items)
	files = {"suite.json": digest(output / "suite.json")}
	write(output / "stages" / BASE / "suite.json", items)
	files[f"stages/{BASE}/suite.json"] = digest(output / "stages" / BASE / "suite.json")
	for name, rows in stages.items():
		stage = output / "stages" / name
		write(stage / "suite.json", items)
		write(stage / f"subset-{len(rows)}.json", rows)
		for file in ("suite.json", f"subset-{len(rows)}.json"):
			files[f"stages/{name}/{file}"] = digest(stage / file)
	for info in checkpoints:
		if info["id"] in stages:
			info["stage"] = f"stages/{info['id']}"
			info["n"] = info["trainable_count"]
	plan = {
		"version": VERSION,
		"scope": SCOPE,
		"task": task,
		"include_pending": include_pending,
		"target_accuracy": float(target_accuracy),
		"seeds": list(seeds),
		"recipe": dict(recipe) if recipe is not None else MATH_RECIPE,
		"export": {
			"path": str(root),
			"manifest_sha256": digest(manifest_path),
			"manifest_digest": manifest.get("digest"),
			"manifest_version": manifest.get("version"),
			"ledgers": ledger_hashes,
		},
		"suite": {"path": str(suite_path.resolve()), "frozen_sha256": suite.sha256},
		"inputs": inputs,
		"files": files,
		"timeline": [e["id"] for e in timeline],
		"unordered_checkpoints": [
			e["id"] for e in entries if e["reached"] and e["cutoff_seconds"] is None
		],
		"interventions": manifest.get("interventions"),
		"limitations": manifest.get("limitations"),
		"checkpoints": checkpoints,
		"cells": [
			{"checkpoint": name, "n": len(rows), "seed": seed}
			for name, rows in stages.items()
			for seed in seeds
		],
	}
	write(output / "plan.json", plan)
	return plan


def load_checkpoint_plan(output: Path) -> dict[str, Any]:
	"""Load a plan, refusing changed stage artifacts and changed or tampered inputs."""
	path = output / "plan.json"
	try:
		plan: dict[str, Any] = json.loads(path.read_text())
	except (OSError, ValueError) as error:
		raise ValueError("plan.json is missing or corrupt") from error
	if plan.get("version") != VERSION:
		raise ValueError("not a checkpoint learning plan")
	for name, expected in plan["files"].items():
		target = output / name
		if not target.is_file() or digest(target) != expected:
			raise ValueError(f"plan artifact hash mismatch: {name}")
	for name, expected in plan["inputs"].items():
		source = Path(name)
		if not source.is_file():
			raise ValueError(f"plan input is missing: {name}")
		if digest(source) != expected:
			raise ValueError(f"plan input changed since planning: {name}")
	export = Path(plan["export"]["path"])
	for name, expected in plan["export"]["ledgers"].items():
		source = export / name
		if not source.is_file():
			raise ValueError(f"export ledger is missing: {name}")
		if digest(source) != expected:
			raise ValueError(f"export ledger changed since planning: {name}")
	return plan


def _suite_ids(output: Path) -> set[str]:
	return {i["id"] for i in json.loads((output / "suite.json").read_text())}


def _score_name(checkpoint: str | None, seed: int) -> str:
	return BASE if checkpoint is None else f"{checkpoint}-s{seed}"


def _scores_digest(scores: Mapping[str, Any]) -> str:
	return hashlib.sha256(canonical_json(scores).encode()).hexdigest()


def _load_cell(
	output: Path,
	contract: str,
	suite_sha: str,
	ids: set[str],
	*,
	checkpoint: str | None,
	n: int | None,
	seed: int,
) -> dict[str, Any] | None:
	path = output / "scores" / f"{_score_name(checkpoint, seed)}.json"
	if not path.exists():
		return None
	try:
		result: dict[str, Any] = json.loads(path.read_text())
		bound = (
			result["contract"] == contract
			and result["suite_sha256"] == suite_sha
			and result["checkpoint"] == checkpoint
			and result["n"] == n
			and result["seed"] == seed
		)
		if not bound:
			raise ValueError("completed cell belongs to a different plan or cell")
		validate_scores(result["scores"], ids)
		if result["scores_sha256"] != _scores_digest(result["scores"]):
			raise ValueError("score file content hash mismatch")
	except (KeyError, TypeError, ValueError) as error:
		raise ValueError(f"unusable score artifact {path.name}: {error}") from error
	return result


def execute_checkpoint_curve(
	output: Path,
	runner: Runner | None = None,
	preflight: Preflight | None = None,
) -> dict[str, int]:
	"""Run missing cells; a verified score file marks a cell complete.

	An interrupted cell has no score file and restarts from the pinned base. With the default
	runner every training row is length-checked before any model work; with an injected runner
	and no preflight that check is skipped.
	"""
	if runner is None:
		from distillation.learning_curve_runner import preflight as check_lengths
		from distillation.learning_curve_runner import run_cell

		runner = run_cell
		preflight = preflight or check_lengths
	plan = load_checkpoint_plan(output)
	contract = digest(output / "plan.json")
	suite_sha = plan["files"]["suite.json"]
	ids = _suite_ids(output)
	if preflight is not None:
		subsets = [
			f"{c['stage']}/subset-{c['n']}.json" for c in plan["checkpoints"] if "stage" in c
		]
		if subsets:
			preflight(output, plan["recipe"], subsets)
	cells: list[dict[str, Any]] = [
		{"checkpoint": None, "n": None, "seed": plan["seeds"][0]},
		*plan["cells"],
	]
	completed = skipped = 0
	for cell in cells:
		name, n, seed = cell["checkpoint"], cell["n"], cell["seed"]
		if _load_cell(output, contract, suite_sha, ids, checkpoint=name, n=n, seed=seed):
			skipped += 1
			continue
		stage = f"stages/{BASE if name is None else name}"
		for file, expected in plan["files"].items():
			if file.startswith(f"{stage}/") and digest(output / file) != expected:
				raise ValueError(f"plan artifact hash mismatch: {file}")
		scores = runner(output / stage, plan["recipe"], n, seed)
		validate_scores(scores, ids)
		write(
			output / "scores" / f"{_score_name(name, seed)}.json",
			{
				"contract": contract,
				"suite_sha256": suite_sha,
				"checkpoint": name,
				"n": n,
				"seed": seed,
				"scores_sha256": _scores_digest(scores),
				"scores": scores,
			},
		)
		completed += 1
	return {"completed": completed, "skipped": skipped}


def _accuracy(scores: Sequence[float]) -> float:
	return sum(scores) / len(scores)


def summarize_checkpoint_curve(output: Path) -> dict[str, Any]:
	"""Per-checkpoint, per-seed accuracy and paired base gain, bound to plan, export and suite."""
	plan = load_checkpoint_plan(output)
	contract = digest(output / "plan.json")
	suite_sha = plan["files"]["suite.json"]
	ids = sorted(_suite_ids(output))
	id_set = set(ids)
	seeds = plan["seeds"]
	target = plan["target_accuracy"]
	score_hashes: dict[str, str] = {}

	def scores(checkpoint: str | None, n: int | None, seed: int) -> list[float]:
		result = _load_cell(
			output, contract, suite_sha, id_set, checkpoint=checkpoint, n=n, seed=seed
		)
		if result is None:
			raise ValueError(f"missing score for {_score_name(checkpoint, seed)}; run execute")
		name = _score_name(checkpoint, seed)
		score_hashes[name] = digest(output / "scores" / f"{name}.json")
		return [float(result["scores"][i]) for i in ids]

	base = scores(None, None, seeds[0])
	base_accuracy = _accuracy(base)
	rows: list[dict[str, Any]] = []
	for cp in plan["checkpoints"]:
		row: dict[str, Any] = {
			"id": cp["id"],
			"kind": cp["kind"],
			"status": cp["status"],
			"cutoff_seconds": cp["cutoff_seconds"],
			"acquired_count": cp["acquired_count"],
			"trainable_count": cp["trainable_count"],
			"filtered": cp["filtered"],
			"evidence": cp["evidence"],
		}
		if cp["status"] == "censored":
			rows.append({**row, "seeds": None, "mean_accuracy": None, "note": "unreached"})
			continue
		per_seed = []
		for seed in seeds:
			values = base if cp["status"] == "base" else scores(cp["id"], cp["n"], seed)
			per_seed.append(
				{
					"seed": seed,
					"accuracy": _accuracy(values),
					"gain_pp": 100 * (_accuracy(values) - base_accuracy),
					"improved": sum(v > b for v, b in zip(values, base, strict=True)),
					"regressed": sum(v < b for v, b in zip(values, base, strict=True)),
				}
			)
		accuracies = [s["accuracy"] for s in per_seed]
		rows.append(
			{
				**row,
				"seeds": per_seed,
				"source": "base_model" if cp["status"] == "base" else "trained_students",
				"mean_accuracy": statistics.fmean(accuracies),
				"min_accuracy": min(accuracies),
				"max_accuracy": max(accuracies),
				"seed_sd": statistics.stdev(accuracies) if len(accuracies) > 1 else None,
				"mean_gain_pp": statistics.fmean(s["gain_pp"] for s in per_seed),
				"seeds_reaching_target": sum(a >= target for a in accuracies),
				"mean_reaches_target": statistics.fmean(accuracies) >= target,
			}
		)
	by_id = {r["id"]: r for r in rows}
	first = next(
		(by_id[i] for i in plan["timeline"] if by_id[i]["mean_reaches_target"]),
		None,
	)
	censored = [r["id"] for r in rows if r["status"] == "censored"]
	result = {
		"version": VERSION,
		"scope": plan["scope"],
		"task": plan["task"],
		"target_accuracy": target,
		"base_accuracy": base_accuracy,
		"base_meets_target": base_accuracy >= target,
		"seeds": seeds,
		"checkpoints": rows,
		"first_reaching_target": (
			{
				"id": first["id"],
				"kind": first["kind"],
				"cutoff_seconds": first["cutoff_seconds"],
				"acquired_count": first["acquired_count"],
				"trainable_count": first["trainable_count"],
				"evidence": first["evidence"],
				"mean_accuracy": first["mean_accuracy"],
				"seeds_reaching_target": first["seeds_reaching_target"],
				"seed_count": len(seeds),
			}
			if first
			else None
		),
		"target_status": "observed" if first else "not_observed_among_reached_checkpoints",
		"censored_checkpoints": censored,
		"unordered_checkpoints": plan["unordered_checkpoints"],
		"interventions": plan["interventions"],
		"limitations": plan["limitations"],
		"claim_boundary": CLAIM_BOUNDARY,
		"bindings": {
			"plan_sha256": contract,
			"export_manifest_sha256": plan["export"]["manifest_sha256"],
			"export_manifest_digest": plan["export"]["manifest_digest"],
			"suite_sha256": suite_sha,
			"suite_frozen_sha256": plan["suite"]["frozen_sha256"],
			"scores": dict(sorted(score_hashes.items())),
		},
	}
	write(output / "summary.json", result)
	return result
