"""Qualify the complete coding source and export disjoint, gold-blind study pools."""

import hashlib
import heapq
import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from coding.dataset import CodingExample, file_hash, prepare_corpus, specification_hash
from coding.scoring import CodingItem, ensure_runtime, pass_at_1

from distillation.benchmarks.models import BenchmarkItem, save_json
from distillation.io import canonical_json
from distillation.schemas import Message, PromptRecord, PromptSource
from distillation.tasks import (
	ROLES,
	TASKS,
	DatasetSpec,
	_role,
	_suite,
)

EVALUATION = DatasetSpec(
	"evalplus/humanevalplus",
	"d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
	"apache-2.0",
	"default",
)


def _stream(spec: DatasetSpec, split: str) -> Iterable[dict[str, Any]]:
	from datasets import load_dataset

	dataset = load_dataset(
		spec.repository, spec.configuration, split=split, revision=spec.revision, streaming=True
	)
	return (dict(row) for row in dataset)


def _evaluation(row: dict[str, Any]) -> BenchmarkItem:
	entry = row["entry_point"]
	if not isinstance(entry, str) or not entry.isidentifier():
		raise ValueError("invalid coding evaluation entry point")
	prompt = row["prompt"]
	solution = prompt + row["canonical_solution"]
	tests = row["test"] + f"\nassert check({entry}) is None\n"
	item = BenchmarkItem(
		id=f"coding:test:{row['task_id']}",
		benchmark="coding",
		category=row["task_id"],
		question=prompt,
		messages=[
			Message(
				role="user",
				content=(
					"Complete the Python function below. Return the complete implementation, "
					"including needed imports, in a Python code block.\n\n" + prompt
				),
			)
		],
		metric="pass_at_1",
		answer=solution,
		test_code=tests,
		entry_point=entry,
	)
	if not pass_at_1(CodingItem(tests), solution).correct:
		raise ValueError(f"coding evaluation reference failed: {item.id}")
	return item


def _item(example: CodingExample) -> BenchmarkItem:
	return BenchmarkItem(
		id=f"coding:train:{example.key}",
		benchmark="coding",
		category=example.category,
		question=example.specification,
		messages=[Message(role="user", content=example.user)],
		metric="pass_at_1",
		answer=example.solution,
		test_code=example.tests,
		entry_point=example.entry_point,
	)


def build_coding_task(
	*,
	output_dir: Path,
	seed: int = 42,
	development_limit: int = 1000,
	rows: dict[str, list[dict[str, Any]]] | None = None,
	log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
	"""Stream and qualify every source row. Only the development suite has a size limit.

	An incomplete build is retained for inspection and never silently accepted as a task.
	Reference solutions and assertion suites remain in preparation/scoring artifacts only.
	"""
	if output_dir.exists():
		raise ValueError(f"{output_dir} already exists; choose a new coding build directory")
	if development_limit < 0:
		raise ValueError("development_limit must be nonnegative")
	ensure_runtime()
	output_dir.mkdir(parents=True)
	save_json(output_dir / "build-status.json", {"complete": False})
	if log:
		log("coding: validating pinned external evaluation references in the sandbox")
	tests = [
		_evaluation(row)
		for row in (rows["test"] if rows is not None else _stream(EVALUATION, "test"))
	]
	if not tests or len({item.id for item in tests}) != len(tests):
		raise ValueError("coding evaluation must have distinct nonempty items")
	test_suite = _suite("coding", "test", tests, len(tests), seed, dataset=EVALUATION)
	save_json(output_dir / "test-suite.json", test_suite)
	if log:
		log("coding: qualifying the complete OpenCodeInstruct stream; no row cap")
	digest = hashlib.sha256()

	def source() -> Iterator[dict[str, Any]]:
		for row in rows["train"] if rows is not None else _stream(TASKS["coding"].dataset, "train"):
			digest.update(canonical_json(row).encode() + b"\n")
			yield row

	qualified = output_dir / "qualified"
	qualification = prepare_corpus(
		source(),
		[
			{"id": item.id, "specification": item.question, "solution": item.answer}
			for item in tests
		],
		qualified,
		progress=(lambda counts: log(f"coding qualification: {counts}")) if log else None,
	)
	qualification.update(
		input_sha256=digest.hexdigest(),
		input_hash_format="canonical JSON rows in source order",
		source_identity="supplied fixture" if rows is not None else "revision-pinned Hub stream",
	)
	save_json(qualified / "manifest.json", qualification)
	counts: Counter[str] = Counter()
	development: list[tuple[int, str, BenchmarkItem]] = []
	spec = TASKS["coding"]
	with ExitStack() as stack:
		outputs = {}
		for role in ROLES:
			if role == "development":
				continue
			path = output_dir / (
				"attacker.jsonl" if role == "attacker" else f"legitimate/{role}.jsonl"
			)
			path.parent.mkdir(parents=True, exist_ok=True)
			outputs[role] = stack.enter_context(path.open("w", encoding="utf-8"))
		gold = stack.enter_context((output_dir / "gold-answers.jsonl").open("w", encoding="utf-8"))
		references = stack.enter_context(
			(output_dir / "legitimate-references.json").open("w", encoding="utf-8")
		)
		references.write("{")
		separator = ""
		eligible = stack.enter_context((qualified / "eligible.jsonl").open(encoding="utf-8"))
		for line in eligible:
			example = CodingExample(**json.loads(line))
			group = specification_hash(example.specification)
			role = _role(seed, "coding", group)
			counts[role] += 1
			if role == "development":
				priority = -int(hashlib.sha256(f"{seed}:{example.key}".encode()).hexdigest(), 16)
				candidate = (priority, example.key, _item(example))
				if not development_limit or len(development) < development_limit:
					heapq.heappush(development, candidate)
				elif priority > development[0][0]:
					heapq.heapreplace(development, candidate)
				continue
			identifier = f"coding-train-{example.key}"
			messages = [Message(role="user", content=example.user)]
			prompt = PromptRecord(
				example_id=identifier,
				task="coding",
				split="train" if role == "attacker" else "test",
				prompt=messages,
				source=PromptSource(
					corpus=spec.dataset.repository,
					source_id=f"train:{example.key}",
					license=spec.dataset.license,
					group_id=group,
					content_sha256=hashlib.sha256(
						canonical_json([m.model_dump() for m in messages]).encode()
					).hexdigest(),
				),
			)
			outputs[role].write(prompt.model_dump_json() + "\n")
			if role == "attacker":
				gold.write(
					json.dumps(
						{
							"example_id": identifier,
							"completion": f"```python\n{example.solution}\n```",
						}
					)
					+ "\n"
				)
			else:
				references.write(
					separator + json.dumps(identifier) + ":" + _item(example).model_dump_json()
				)
				separator = ",\n"
		references.write("}\n")
	if any(not counts[role] for role in ROLES):
		raise ValueError(
			"coding corpus cannot supply every study role; inspect qualification exclusions"
		)
	development_suite = _suite(
		"coding",
		"train",
		sorted([row[2] for row in development], key=lambda item: item.id),
		counts["development"],
		seed,
	)
	save_json(output_dir / "development-suite.json", development_suite)
	qualification["role_assignment"] = {role: counts[role] for role in ROLES}
	save_json(qualified / "manifest.json", qualification)
	manifest: dict[str, Any] = {
		"task": "coding",
		"seed": seed,
		"complete": True,
		"max_output_tokens": spec.max_output_tokens,
		"output_cap_status": "provisional",
		"metric": "pass_at_1",
		"scorer": "sandboxed-pinned-assertions-v1",
		"dataset": {
			"repository": spec.dataset.repository,
			"revision": spec.dataset.revision,
			"configuration": spec.dataset.configuration,
			"license": spec.dataset.license,
			"pool_split": "train",
			"test_split": "test",
		},
		"evaluation_dataset": test_suite.sources[0].model_dump(),
		"available": {"train": qualification["counts"]["raw"], "test": len(tests)},
		"roles": {role: counts[role] for role in ROLES},
		"groups": {role: counts[role] for role in ROLES},
		"filtered": {
			key: value
			for key, value in qualification["counts"].items()
			if key not in {"raw", "eligible"}
		},
		"truncated": {"train": 0, "test": 0},
		"development_suite_items": len(development_suite.items),
		"test_suite_items": len(tests),
		"qualification": "qualified/manifest.json",
		"corpus_limit": None,
		"sha256": {
			str(path.relative_to(output_dir)): file_hash(path)
			for path in sorted(output_dir.rglob("*"))
			if path.is_file() and path.name not in {"seen.sqlite", "build-status.json"}
		},
	}
	save_json(output_dir / "manifest.json", manifest)
	save_json(output_dir / "build-status.json", {"complete": True})
	if log:
		log(
			f"coding: {sum(counts.values())} qualified examples; "
			f"{counts['attacker']} attacker prompts"
		)
	return manifest
