"""Qualify the complete coding source and export disjoint, gold-blind study pools."""

import ast
import copy
import hashlib
import heapq
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coding.dataset import PROMPT, CodingExample, file_hash, prepare_corpus, specification_hash
from coding.sandbox import runtime_protocol
from coding.scoring import CodingItem, ensure_runtime, pass_at_1

from distillation.benchmarks.models import BenchmarkItem, DatasetSource, FrozenSuite, save_json
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
# HumanEval+ alone gives a pass@1 interval near +-7.7 points at p=0.5, too wide for the
# study's two-point gain rule. MBPP+ adds 378 executable tasks to the same frozen suite.
EVALUATION_MBPP = DatasetSpec(
	"evalplus/mbppplus",
	"b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
	"apache-2.0",
	"default",
)

# This immutable export has a malformed polynomial oracle: it expands a scalar
# candidate return value as positional arguments. Declare the exclusion before any
# model evaluation; never silently convert it into a quality failure or repair gold.
EVALUATION_EXCLUSIONS = {
	"HumanEval/32": "pinned export calls _poly(*candidate(*inp), inp), expanding a float",
	# Found by running every pinned MBPP+ reference in the sandbox on 2026-10-20.
	"Mbpp/255": "reference exhausts the sandbox memory limit on the pinned plus inputs",
	"Mbpp/596": "expected sys.getsizeof values depend on the interpreter version",
}


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
	functions = [
		node
		for node in ast.parse(solution).body
		if isinstance(node, ast.FunctionDef) and node.name == entry
	]
	if len(functions) != 1:
		raise ValueError("external coding reference must declare one matching entry point")
	declaration = copy.deepcopy(functions[0])
	declaration.body = [ast.Pass()]
	declaration.decorator_list = []
	signature = ast.unparse(declaration).splitlines()[0]
	tests = row["test"] + f"\nassert check({entry}) is None\n"
	item = BenchmarkItem(
		id=f"coding:test:{row['task_id']}",
		benchmark="coding",
		category=row["task_id"],
		question=prompt,
		messages=[
			Message(
				role="user",
				content=PROMPT.format(specification=prompt, signature=signature),
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


def _mbpp_evaluation(row: dict[str, Any]) -> BenchmarkItem:
	"""An MBPP+ task: its one-line description with the first plain assertion as an example."""
	solution = row["code"].strip() + "\n"
	called = re.findall(r"assertion\((\w+)\(", row["test"])
	functions = {
		node.name: node for node in ast.parse(solution).body if isinstance(node, ast.FunctionDef)
	}
	entry = called[-1] if called else next(iter(functions), "")
	if entry not in functions:
		raise ValueError(f"MBPP+ reference lacks its entry point: {row['task_id']}")
	declaration = copy.deepcopy(functions[entry])
	declaration.body = [ast.Pass()]
	declaration.decorator_list = []
	signature = ast.unparse(declaration).splitlines()[0]
	examples = row["test_list"]
	if isinstance(examples, str):
		examples = ast.literal_eval(examples)
	imports = row["test_imports"]
	if isinstance(imports, str):
		imports = ast.literal_eval(imports)
	specification = row["prompt"].strip() + (f"\n\nExample: {examples[0]}" if examples else "")
	# The pinned harness asserts inside helpers; the final check makes it a scorable item.
	tests = "\n".join([*imports, row["test"], f"assert callable({entry})", ""])
	item = BenchmarkItem(
		id=f"coding:test:Mbpp/{row['task_id']}",
		benchmark="coding",
		category=f"Mbpp/{row['task_id']}",
		question=specification,
		messages=[
			Message(
				role="user",
				content=PROMPT.format(specification=specification, signature=signature),
			)
		],
		metric="pass_at_1",
		answer=solution,
		test_code=tests,
		entry_point=entry,
	)
	if not pass_at_1(CodingItem(tests), f"```python\n{solution}```").correct:
		raise ValueError(f"coding evaluation reference failed: {item.id}")
	return item


def _coding_suite(
	parts: list[tuple[DatasetSpec, list[BenchmarkItem], int]], seed: int
) -> FrozenSuite:
	sources = [
		DatasetSource(
			benchmark="coding",
			repository=spec.repository,
			revision=spec.revision,
			configuration=spec.configuration or "default",
			split="test",
			license=spec.license,
			url=f"https://huggingface.co/datasets/{spec.repository}/tree/{spec.revision}",
			available=available,
			selected=len(items),
		)
		for spec, items, available in parts
		if items
	]
	items = sorted((item for _, part, _ in parts for item in part), key=lambda item: item.id)
	suite = FrozenSuite(seed=seed, created_at=datetime.now(UTC), sources=sources, items=items)
	suite.sha256 = suite.digest()
	suite.verify()
	return suite


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
	tests = []
	external_count = 0
	exclusions = []
	for row in rows["test"] if rows is not None else _stream(EVALUATION, "test"):
		external_count += 1
		if row["task_id"] in EVALUATION_EXCLUSIONS:
			exclusions.append(
				{
					"id": row["task_id"],
					"reason": EVALUATION_EXCLUSIONS[row["task_id"]],
					"source_sha256": hashlib.sha256(canonical_json(row).encode()).hexdigest(),
				}
			)
			continue
		tests.append(_evaluation(row))
	if not tests or len({item.id for item in tests}) != len(tests):
		raise ValueError("coding evaluation must have distinct nonempty items")
	# Fixture builds that omit MBPP+ rows keep a HumanEval+-only suite; the manifest says so.
	mbpp_rows = rows.get("test_mbpp", []) if rows is not None else _stream(EVALUATION_MBPP, "test")
	mbpp, mbpp_count = [], 0
	for row in mbpp_rows:
		mbpp_count += 1
		identifier = f"Mbpp/{row['task_id']}"
		if identifier in EVALUATION_EXCLUSIONS:
			exclusions.append(
				{
					"id": identifier,
					"reason": EVALUATION_EXCLUSIONS[identifier],
					"source_sha256": hashlib.sha256(canonical_json(row).encode()).hexdigest(),
				}
			)
			continue
		mbpp.append(_mbpp_evaluation(row))
	test_suite = _coding_suite(
		[(EVALUATION, tests, external_count), (EVALUATION_MBPP, mbpp, mbpp_count)], seed
	)
	tests = [*tests, *mbpp]
	save_json(output_dir / "evaluation-exclusions.json", {"exclusions": exclusions})
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
		"scorer": "sandboxed-pinned-assertions-qualified-v2",
		"evaluation_exclusions": exclusions,
		"coding_runtime": runtime_protocol(),
		"dataset": {
			"repository": spec.dataset.repository,
			"revision": spec.dataset.revision,
			"configuration": spec.dataset.configuration,
			"license": spec.dataset.license,
			"pool_split": "train",
			"test_split": "test",
		},
		"evaluation_dataset": test_suite.sources[0].model_dump(),
		"evaluation_datasets": [source.model_dump() for source in test_suite.sources],
		"available": {
			"train": qualification["counts"]["raw"],
			"test": external_count + mbpp_count,
		},
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
