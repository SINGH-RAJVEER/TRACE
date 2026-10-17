"""Deterministic task clients with disclosed synthetic arrival times and source identities."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from harness.calibration import BenignRequest

from distillation.benchmarks.models import BenchmarkItem, save_json
from distillation.io import read_jsonl
from distillation.schemas import PromptRecord

WORKLOAD_VERSION = "task-clients-v1"
CLIENT_TYPES = ("specialist", "repetitive", "batch", "bursty", "short")


def task_references(directory: Path) -> dict[str, BenchmarkItem]:
	"""Recover scoring references for existing builds without rewriting pools or ledgers."""
	from distillation.tasks import CONVERTERS, LEGITIMATE_ROLES, TASKS, _benchmark_item, _load_rows

	path = directory / "legitimate-references.json"
	if path.exists():
		return {
			key: BenchmarkItem.model_validate(value)
			for key, value in json.loads(path.read_text()).items()
		}
	manifest = json.loads((directory / "manifest.json").read_text())
	task = manifest["task"]
	spec = TASKS[task]
	if manifest["dataset"]["revision"] != spec.dataset.revision:
		raise ValueError("task references require the build's pinned dataset revision")
	prompts = {
		p.example_id: p
		for role in LEGITIMATE_ROLES
		for p in read_jsonl(directory / "legitimate" / f"{role}.jsonl", PromptRecord)
	}
	items = {}
	for index, row in enumerate(_load_rows(spec.dataset, spec.pool_split)):
		example = CONVERTERS[task](index, row)
		key = f"{task}-{spec.pool_split}-{example.key}"
		if key in prompts:
			item = _benchmark_item(task, spec.pool_split, example)
			if (
				item.messages != prompts[key].prompt
				or example.group != prompts[key].source.group_id
			):
				raise ValueError(f"reference does not match frozen prompt {key}")
			items[key] = item
	if items.keys() != prompts.keys():
		raise ValueError("missing legitimate scoring references")
	save_json(path, {key: item.model_dump(mode="json") for key, item in items.items()})
	return items


def clients(
	prompts: list[PromptRecord],
	role: str,
	seed: int = 42,
	references: dict[str, BenchmarkItem] | None = None,
) -> list[BenignRequest]:
	"""Partition source groups between five client types, then form accounts within each type.

	Repetition reuses a prompt only within its original account. Document sharing between
	accounts is recorded and must be clustered in uncertainty estimates. No natural client
	identity or real-world arrival distribution is inferred from these synthetic accounts.
	"""
	groups: dict[str, list[PromptRecord]] = defaultdict(list)
	for prompt in prompts:
		if not prompt.source.group_id:
			raise ValueError(f"missing source group for {prompt.example_id}")
		groups[prompt.source.group_id].append(prompt)
	ordered = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest())
	buckets: dict[str, list[PromptRecord]] = {kind: [] for kind in CLIENT_TYPES}
	for index, group in enumerate(ordered):
		buckets[CLIENT_TYPES[index % len(CLIENT_TYPES)]].extend(groups[group])
	requests = []
	for kind, pool in buckets.items():
		pool.sort(key=lambda p: hashlib.sha256(f"{seed}:{p.example_id}".encode()).digest())
		if kind == "specialist":
			pool.sort(
				key=lambda p: (
					references[p.example_id].category if references else p.source.group_id or ""
				)
			)
		size = {"specialist": 200, "repetitive": 100, "batch": 500, "bursty": 300, "short": 20}[
			kind
		]
		for offset in range(0, len(pool), size):
			chunk = pool[offset : offset + size]
			if kind == "repetitive":
				chunk = [p for p in chunk for _ in range(3)]
			account = f"{role}-{kind}-{offset // size:05d}"
			for index, prompt in enumerate(chunk):
				at = index * 0.1 if kind == "batch" else index * 2.0
				if kind == "bursty":
					at = (index // 25) * 60 + (index % 25) * 0.1
				requests.append(
					BenignRequest(
						account=account,
						text=next(m.content for m in reversed(prompt.prompt) if m.role == "user"),
						time=at,
						example_id=prompt.example_id,
						source_group=prompt.source.group_id,
						client_type=kind,
					)
				)
	return requests


def workload_summary(rows: list[BenignRequest]) -> dict[str, Any]:
	lengths = Counter(row.account for row in rows)
	return {
		"version": WORKLOAD_VERSION,
		"requests": len(rows),
		"accounts": len(lengths),
		"scored_accounts": sum(n >= 100 for n in lengths.values()),
		"short_accounts": sum(n < 100 for n in lengths.values()),
		"source_documents": len({r.source_group for r in rows}),
		"unique_prompts": len({r.example_id for r in rows}),
		"account_lengths": dict(lengths),
		"arrival_times": "synthetic",
		"client_types": dict(Counter(r.client_type for r in rows)),
	}
