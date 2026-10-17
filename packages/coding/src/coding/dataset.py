"""Full-corpus conversion and conservative cross-source decontamination.

This staging format contains gold code and tests. It is never an attacker input file.
The study builder consumes this staging format to export gold-blind role pools.
"""

import argparse
import ast
import copy
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coding.extract import extract_code
from coding.sandbox import Limits
from coding.scoring import CodingItem, pass_at_1

if TYPE_CHECKING:
	from distillation.tasks import TaskExample

SOURCE = "nvidia/OpenCodeInstruct"
REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
ALGORITHM = "coding-decontamination-v1"
PROMPT = (
	"Implement the Python function described below. Return only a Python code block."
	"\n\nSpecification: {specification}\n\nSignature: {signature}"
)
MAX_SOURCE_CHARS = 128_000


class Ineligible(ValueError):
	"""A source row cannot be used under the narrow function contract."""


@dataclass(frozen=True)
class CodingExample:
	key: str
	specification: str
	signature: str
	entry_point: str
	solution: str
	tests: str
	category: str

	@property
	def user(self) -> str:
		return PROMPT.format(specification=self.specification, signature=self.signature)

	def to_task_example(self) -> TaskExample:
		"""Lazy core import keeps the coding distribution independently installable."""
		from distillation.tasks import TaskExample

		return TaskExample(
			key=self.key,
			group=specification_hash(self.specification),
			user=self.user,
			answer=self.solution,
			category=self.category,
			question=self.specification,
			gold_completion=f"```python\n{self.solution}\n```",
		)


def specification_hash(text: str) -> str:
	return hashlib.sha256(" ".join(re.findall(r"\w+", text.casefold())).encode()).hexdigest()


class _CanonicalAST(ast.NodeTransformer):
	"""Ignore names and docstrings, but preserve literals, operators and attributes."""

	def __init__(self) -> None:
		self.names: dict[str, str] = {}

	def _bound_name(self, name: str) -> str:
		if name not in self.names:
			self.names[name] = f"bound_{len(self.names)}"
		return self.names[name]

	def visit_Name(self, node: ast.Name) -> ast.Name:
		if isinstance(node.ctx, ast.Store):
			self._bound_name(node.id)
		return ast.Name(id=self.names.get(node.id, node.id), ctx=node.ctx)

	def visit_arg(self, node: ast.arg) -> ast.arg:
		return ast.arg(arg=self._bound_name(node.arg), annotation=None)

	def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
		node.name = self._bound_name(node.name)
		if (
			node.body
			and isinstance(node.body[0], ast.Expr)
			and isinstance(node.body[0].value, ast.Constant)
			and isinstance(node.body[0].value.value, str)
		):
			node.body.pop(0)
		self.generic_visit(node)
		return node


def ast_hash(code: str) -> str:
	tree = _CanonicalAST().visit(ast.parse(code))
	return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def _ngrams(text: str, width: int = 13) -> set[tuple[str, ...]]:
	tokens = re.findall(r"\w+", text.casefold())
	return {tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


class Decontaminator:
	"""Index only the small external evaluation set, never the entire source corpus."""

	def __init__(self, evaluation: Iterable[Mapping[str, str]]) -> None:
		self.exact: dict[str, str] = {}
		self.code: dict[str, str] = {}
		self.grams: dict[tuple[str, ...], set[str]] = defaultdict(set)
		self.counts: dict[str, int] = {}
		self.sha256 = hashlib.sha256()
		for row in evaluation:
			self.sha256.update(json.dumps(dict(row), sort_keys=True).encode() + b"\n")
			key = row["id"]
			if key in self.counts:
				raise ValueError(f"duplicate evaluation ID: {key}")
			self.exact[specification_hash(row["specification"])] = key
			self.code[ast_hash(row["solution"])] = key
			grams = _ngrams(row["specification"])
			self.counts[key] = len(grams)
			for gram in grams:
				self.grams[gram].add(key)
		if not self.counts:
			raise ValueError("a nonempty frozen external evaluation set is required")

	def overlap(self, example: CodingExample) -> str | None:
		if match := self.exact.get(specification_hash(example.specification)):
			return match
		if match := self.code.get(ast_hash(example.solution)):
			return match
		grams = _ngrams(example.specification)
		shared: Counter[str] = Counter()
		for gram in grams:
			shared.update(self.grams.get(gram, ()))
		# Containment catches long copied problems embedded in extra instructions.
		for key, count in sorted(shared.items()):
			if count / min(len(grams), self.counts[key]) >= 0.7:
				return key
		return None


def convert(row: Mapping[str, Any]) -> CodingExample:
	"""Convert one OpenCodeInstruct row without executing its source or test literals."""
	for name in ("id", "input", "output", "unit_tests"):
		if name not in row:
			raise Ineligible(f"missing {name}")
	if any(len(str(row[name])) > MAX_SOURCE_CHARS for name in ("input", "output", "unit_tests")):
		raise Ineligible("oversized source")
	specification = str(row["input"]).strip()
	solution = extract_code(str(row["output"]))
	if not specification or solution is None:
		raise Ineligible("missing specification or invalid reference")
	try:
		tests = (
			ast.literal_eval(row["unit_tests"])
			if isinstance(row["unit_tests"], str)
			else row["unit_tests"]
		)
		if (
			not isinstance(tests, list)
			or not tests
			or not all(isinstance(test, str) for test in tests)
		):
			raise Ineligible("tests must be a nonempty list of strings")
		test_source = "\n".join(tests)
		CodingItem(test_source)
		calls = {
			node.func.id
			for node in ast.walk(ast.parse(test_source))
			if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
		}
		functions = [
			node
			for node in ast.parse(solution).body
			if isinstance(node, ast.FunctionDef) and node.name in calls
		]
		if len(functions) != 1:
			raise Ineligible("tests must target one unambiguous top-level function")
		function = copy.deepcopy(functions[0])
		function.body = [ast.Pass()]
		function.decorator_list = []
		signature = ast.unparse(function).split("\n", 1)[0]
	except (SyntaxError, ValueError, TypeError, RecursionError) as error:
		raise Ineligible(str(error)) from error
	return CodingExample(
		key=str(row["id"]),
		specification=specification,
		signature=signature,
		entry_point=function.name,
		solution=solution,
		tests=test_source,
		category=str(row.get("domain") or "unspecified"),
	)


def prepare_corpus(
	rows: Iterable[Mapping[str, Any]],
	evaluation: Iterable[Mapping[str, str]],
	output: Path,
	*,
	limits: Limits | None = None,
	input_sha256: str = "unrecorded",
	progress: Callable[[dict[str, int]], None] | None = None,
) -> dict[str, Any]:
	"""Consume ALL input rows, without sampling or a budget cap, and validate references.

	Disk-backed exact deduplication keeps memory independent of pool size. The first
	eligible occurrence in immutable source order survives. An interrupted output retains
	complete=false and is never silently reused as a completed corpus.
	"""
	decontaminator = Decontaminator(evaluation)
	output.mkdir(parents=True, exist_ok=False)
	manifest: dict[str, Any] = {
		"complete": False,
		"source": SOURCE,
		"revision": REVISION,
		"source_identity": "operator-declared pinned export; not remotely attested",
		"input_sha256": input_sha256,
		"evaluation_sha256": decontaminator.sha256.hexdigest(),
		"algorithm": ALGORITHM,
		"ngram_width": 13,
		"containment_threshold": 0.7,
		"limit": None,
		"counts": {},
		"role_assignment": "pending",
	}
	manifest_path = output / "manifest.json"
	manifest_path.write_text(json.dumps(manifest, indent=4) + "\n")
	counts: Counter[str] = Counter()
	with (
		sqlite3.connect(output / "seen.sqlite") as seen,
		(output / "eligible.jsonl").open("w") as eligible,
		(output / "exclusions.jsonl").open("w") as exclusions,
	):
		seen.execute("CREATE TABLE seen (kind TEXT, hash TEXT, PRIMARY KEY (kind, hash))")
		for row in rows:
			counts["raw"] += 1
			reason = ""
			detail = ""
			match = None
			try:
				example = convert(row)
				fingerprints = [
					("id", example.key),
					("specification", specification_hash(example.specification)),
					("ast", ast_hash(example.solution)),
				]
				if match := decontaminator.overlap(example):
					reason = "test_overlap"
				elif any(
					seen.execute("SELECT 1 FROM seen WHERE kind=? AND hash=?", pair).fetchone()
					for pair in fingerprints
				):
					reason = "duplicate"
				else:
					score = pass_at_1(CodingItem(example.tests), example.solution, limits=limits)
					if not score.correct:
						reason = "reference_" + (score.status if score.status != "ok" else "failed")
					else:
						seen.executemany("INSERT INTO seen VALUES (?, ?)", fingerprints)
						eligible.write(json.dumps(asdict(example), sort_keys=True) + "\n")
						counts["eligible"] += 1
			except Ineligible as error:
				reason = "malformed_or_unsupported"
				detail = str(error)[:256]
			if reason:
				counts[reason] += 1
				exclusions.write(
					json.dumps(
						{
							"id": row.get("id"),
							"reason": reason,
							"detail": detail,
							"matched_test": match,
						}
					)
					+ "\n"
				)
			if counts["raw"] % 1000 == 0:
				seen.commit()
				manifest["counts"] = dict(counts)
				manifest_path.write_text(json.dumps(manifest, indent="\t") + "\n")
				if progress:
					progress(dict(counts))
	manifest.update(complete=True, counts=dict(counts))
	manifest["eligible_sha256"] = file_hash(output / "eligible.jsonl")
	manifest_path.write_text(json.dumps(manifest, indent=4) + "\n")
	return manifest


def file_hash(path: Path) -> str:
	with path.open("rb") as source:
		return hashlib.file_digest(source, "sha256").hexdigest()


def _rows(path: Path) -> Iterator[dict[str, Any]]:
	with path.open() as source:
		for line in source:
			if line.strip():
				yield json.loads(line)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--input", required=True, type=Path, help="Pinned OpenCodeInstruct JSONL export"
	)
	parser.add_argument(
		"--evaluation", required=True, type=Path, help="Frozen id/specification/solution JSONL"
	)
	parser.add_argument("--output", type=Path, default=Path("data/tasks/coding/qualified"))
	arguments = parser.parse_args()
	manifest = prepare_corpus(
		_rows(arguments.input),
		_rows(arguments.evaluation),
		arguments.output,
		input_sha256=file_hash(arguments.input),
	)
	print(json.dumps(manifest, indent=4))


if __name__ == "__main__":
	main()
