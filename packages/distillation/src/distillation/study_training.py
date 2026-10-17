"""Development selection and response-token controls for the declared task study."""

import hashlib
import json
import random
import tomllib
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from distillation.benchmarks.models import save_json
from distillation.benchmarks.reporting import discover_runs
from distillation.io import read_jsonl, write_jsonl
from distillation.launcher import LaunchError, _toml, _trained
from distillation.schemas import LedgerRecord, SFTRecord

if TYPE_CHECKING:
	from distillation.study import Study


def primary_score(samples: list[Any], task: str) -> float:
	if not samples:
		raise ValueError("capability scoring requires a nonempty suite")
	return sum((s.f1 or 0) if task == "squad" else float(s.correct) for s in samples) / len(samples)


def select_recipe(study: Study, task: str) -> dict[str, Any]:
	"""Six seed-42, 2000-query candidates; only development scores select the recipe."""
	from distillation.study import freeze

	pipeline = study.pipeline
	path = study.root / task / "recipe.json"
	if int(pipeline.manifest(task)["roles"]["attacker"]) < 2000 or 42 not in pipeline.seeds or max(pipeline.plan[task]) < 2000:
		raise LaunchError("recipe selection requires 2000 attacker prompts")
	base = tomllib.loads(pipeline.config.read_text())
	contract = {
		"training": base,
		"development_sha256": hashlib.sha256(
			(pipeline.task_dir(task) / "development-suite.json").read_bytes()
		).hexdigest(),
		"ledger_sha256": hashlib.sha256(pipeline.run_ledgers[(task, 42)].read_bytes()).hexdigest(),
		"budget": 2000,
		"seed": 42,
	}
	freeze(path.with_name("recipe-protocol.json"), contract)
	if path.exists():
		return dict(json.loads(path.read_text()))
	trials = []
	for rate in (1e-4, 2e-4):
		for epochs in (1, 2, 3):
			name = f"lr{rate}-e{epochs}"
			candidate = replace(
				pipeline, gold=False, artifact_root=study.root / task / "development" / name
			)
			candidate.train_run(base | {"learning_rate": rate, "epochs": epochs}, task, 42, 2000)
			adapter = candidate.artifact_root / task / "s42-b2000" / "attacker"
			config = study.evaluation(
				task, 42, 2000, name, development=True, adapter_override=adapter
			)
			study.benchmark(config, "student_before")
			study.benchmark(config, "student_after")
			from distillation.benchmarks.models import load_config

			runs, errors = discover_runs(load_config(config).output_dir)
			if errors:
				raise LaunchError("invalid development evaluation artifacts")
			latest = {
				slot: max((r for r in runs if r.slot == slot), key=lambda r: r.completed_at)
				for slot in ("student_before", "student_after")
			}
			trials.append(
				{
					"learning_rate": rate,
					"epochs": epochs,
					"base_score": primary_score(latest["student_before"].samples, task),
					"score": primary_score(latest["student_after"].samples, task),
					"adapter": str(adapter),
					"evaluation_ids": {k: v.id for k, v in latest.items()},
				}
			)
	# Stable tie break: fewer epochs, then lower learning rate.
	winner = max(trials, key=lambda row: (row["score"], -row["epochs"], -row["learning_rate"]))
	result = {
		"selected": {"learning_rate": winner["learning_rate"], "epochs": winner["epochs"]},
		"development_gain": winner["score"] - winner["base_score"],
		"transfer_gate_passed": winner["score"] > winner["base_score"],
		"trials": trials,
		"selection": "highest development primary score; "
		"ties use fewer epochs then lower learning rate",
	}
	save_json(path, result)
	return result


def token_controls(
	records: list[LedgerRecord], target: int, seed: int
) -> dict[str, tuple[list[LedgerRecord], int]]:
	"""Uniformly permute or retain order, stopping at the nearest whole-response token count."""
	if target < 0 or any(r.usage.output_tokens is None for r in records):
		raise ValueError("controls require measured released-response token counts")
	result = {}
	for name in ("matched-volume", "matched-prefix"):
		ordered = list(records)
		if name == "matched-volume":
			random.Random(seed).shuffle(ordered)
		selected, total = [], 0
		for record in ordered:
			next_total = total + int(record.usage.output_tokens or 0)
			if next_total > target:
				if abs(next_total - target) < abs(total - target):
					selected.append(record)
					total = next_total
				break
			selected.append(record)
			total = next_total
			if total == target:
				break
		result[name] = selected, total
	return result


def train_controls(study: Study, task: str, seed: int, budget: int, arm: str) -> None:
	"""Use released tokens, then apply the same gold-blind preparation to both controls."""
	from distillation.preparation import prepare_sft_dataset

	defended = study.protected_dir(task, seed, budget, arm)
	rows = read_jsonl(defended / "budget-ledger.jsonl", LedgerRecord)
	if any(r.usage.output_tokens is None for r in rows):
		raise LaunchError(
			"defended responses lack output token usage; token controls cannot be matched"
		)
	target = sum(int(r.usage.output_tokens or 0) for r in rows)
	baseline = study.root / task / f"s{seed}" / "undefended" / "ledger.jsonl"
	all_rows = read_jsonl(baseline, LedgerRecord)
	prompts = study.pipeline.run_dir(task, seed, budget) / "prompts.jsonl"
	from distillation.schemas import PromptRecord

	wanted = {p.example_id for p in read_jsonl(prompts, PromptRecord)}
	all_rows = [r for r in all_rows if r.example_id in wanted]
	for name, (selected, tokens) in token_controls(all_rows, target, seed).items():
		directory = defended / name
		ledger = directory / "ledger.jsonl"
		write_jsonl(ledger, selected)
		base = tomllib.loads(study.pipeline.config.read_text()) | study.pipeline.recipes[task]
		base.pop("validation_dataset_path", None)
		dataset = directory / "train.jsonl"
		adapter = directory / "adapter"
		config = directory / "train.toml"
		config.write_text(
			_toml(base | {"dataset_path": str(dataset), "output_dir": str(adapter), "seed": seed})
		)
		reviews = directory / "reviews.jsonl"
		study.pipeline.launcher.distill(
			"review", "--ledger", str(ledger), "--output", str(reviews), "--config", str(config)
		)
		prepare_sft_dataset(
			input_path=ledger, output_path=dataset, reviews_path=reviews, prompts_path=prompts
		)
		count = len(read_jsonl(dataset, SFTRecord))
		save_json(
			directory / "control.json",
			{
				"target_released_tokens": target,
				"achieved_released_tokens": tokens,
				"residual_tokens": tokens - target,
				"released_responses": len(selected),
				"training_examples": count,
				"seed": seed,
				"selection": name,
			},
		)
		if count:
			if not _trained(adapter, dataset, config):
				study.pipeline.launcher.distill("train", "--config", str(config))
			config = study.evaluation(task, seed, budget, f"{arm}-{name}", adapter_override=adapter)
			study.benchmark(config, "student_before")
			study.benchmark(config, "student_after")
