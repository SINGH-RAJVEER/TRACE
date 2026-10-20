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
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coding import DEFAULT_POOL_SIZE, DEFAULT_QUALIFICATION_WORKERS
from coding.extract import extract_code
from coding.sandbox import Limits
from coding.scoring import CodingItem, Score, pass_at_1

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


@dataclass
class Pending:
	"""A row read ahead of its in-order decision, with its reference run if one started."""

	row: Mapping[str, Any]
	example: CodingExample | None = None
	fingerprints: list[tuple[str, str]] = field(default_factory=list)
	match: str | None = None
	score: Future[Score] | None = None
	error: Ineligible | None = None


def prepare_corpus(
	rows: Iterable[Mapping[str, Any]],
	evaluation: Iterable[Mapping[str, str]],
	output: Path,
	*,
	limits: Limits | None = None,
	input_sha256: str = "unrecorded",
	progress: Callable[[dict[str, int]], None] | None = None,
	workers: int = DEFAULT_QUALIFICATION_WORKERS,
	eligible_limit: int | None = None,
) -> dict[str, Any]:
	"""Qualify input rows, optionally stopping at an explicit eligible-subset target.

	Disk-backed exact deduplication keeps memory independent of pool size. The first
	eligible occurrence in immutable source order survives. An interrupted output retains
	complete=false and is never silently reused as a completed corpus. With workers > 1,
	reference runs execute concurrently but rows are decided in source order, so the
	output matches a sequential build.
	"""
	if workers < 1:
		raise ValueError("workers must be at least 1")
	if eligible_limit is not None and eligible_limit < 1:
		raise ValueError("eligible_limit must be positive or None")
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
		"limit": eligible_limit,
		"scope": "full-source" if eligible_limit is None else "eligible-subset",
		"counts": {},
		"role_assignment": "pending",
	}
	manifest_path = output / "manifest.json"
	manifest_path.write_text(json.dumps(manifest, indent=4) + "\n")
	counts: Counter[str] = Counter()
	decided_hash = hashlib.sha256()
	with (
		sqlite3.connect(output / "seen.sqlite") as seen,
		(output / "eligible.jsonl").open("w") as eligible,
		(output / "exclusions.jsonl").open("w") as exclusions,
		(output / "source-ids.jsonl").open("w") as source_ids,
	):
		seen.execute("CREATE TABLE seen (kind TEXT, hash TEXT, PRIMARY KEY (kind, hash))")

		def duplicate(fingerprints: list[tuple[str, str]]) -> bool:
			return any(
				seen.execute("SELECT 1 FROM seen WHERE kind=? AND hash=?", pair).fetchone()
				for pair in fingerprints
			)

		def admit(row: Mapping[str, Any]) -> Pending:
			"""Converts a row and, with workers, starts its reference run ahead of its commit."""
			try:
				example = convert(row)
			except Ineligible as error:
				return Pending(row, error=error)
			fingerprints = [
				("id", example.key),
				("specification", specification_hash(example.specification)),
				("ast", ast_hash(example.solution)),
			]
			match = decontaminator.overlap(example)
			score = None
			if executor and not match and not duplicate(fingerprints):
				score = executor.submit(
					pass_at_1, CodingItem(example.tests), example.solution, limits=limits
				)
			return Pending(row, example, fingerprints, match, score)

		def commit(pending: Pending) -> None:
			"""Decides one row in source order, so the first eligible occurrence survives."""
			counts["raw"] += 1
			decided_hash.update(
				json.dumps(
					dict(pending.row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
				).encode()
				+ b"\n"
			)
			source_ids.write(json.dumps(str(pending.row.get("id"))) + "\n")
			reason = ""
			detail = ""
			match = pending.match
			try:
				if pending.error:
					raise pending.error
				assert pending.example is not None
				if match:
					reason = "test_overlap"
				elif duplicate(pending.fingerprints):
					reason = "duplicate"
					if pending.score:
						pending.score.cancel()
				else:
					score = (
						pending.score.result()
						if pending.score
						else pass_at_1(
							CodingItem(pending.example.tests),
							pending.example.solution,
							limits=limits,
						)
					)
					if not score.correct:
						reason = "reference_" + (score.status if score.status != "ok" else "failed")
					else:
						seen.executemany("INSERT INTO seen VALUES (?, ?)", pending.fingerprints)
						eligible.write(json.dumps(asdict(pending.example), sort_keys=True) + "\n")
						counts["eligible"] += 1
			except Ineligible as error:
				reason = "malformed_or_unsupported"
				detail = str(error)[:256]
			if reason:
				counts[reason] += 1
				exclusions.write(
					json.dumps(
						{
							"id": pending.row.get("id"),
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

		# Dispatch against unfinished runs, rather than distance from the commit point.
		# Completed out-of-order rows have a separate bound to cap retained source data.
		executor = ThreadPoolExecutor(workers) if workers > 1 else None
		window: deque[Pending] = deque()
		in_flight: set[Future[Score]] = set()

		def reached_target() -> bool:
			return eligible_limit is not None and counts["eligible"] >= eligible_limit

		def decidable(pending: Pending) -> bool:
			"""A row whose fingerprint was claimed after dispatch need not await its run."""
			return pending.score is None or pending.score.done() or duplicate(pending.fingerprints)

		try:
			source = iter(rows)
			exhausted = False
			while window or not exhausted:
				while window and decidable(window[0]):
					commit(window.popleft())
					if reached_target():
						break
				if reached_target():
					break
				in_flight = {future for future in in_flight if not future.done()}
				if not exhausted and len(window) < 32 * workers and len(in_flight) < 2 * workers:
					try:
						pending = admit(next(source))
					except StopIteration:
						exhausted = True
					else:
						window.append(pending)
						if pending.score is not None:
							in_flight.add(pending.score)
					continue
				if window:
					# At the row bound only the oldest decision can release space.
					oldest = window[0].score
					waiting = (
						{oldest}
						if oldest is not None and (exhausted or len(window) >= 32 * workers)
						else in_flight
					)
					if waiting:
						wait(waiting, return_when=FIRST_COMPLETED)
		finally:
			if executor:
				executor.shutdown(cancel_futures=True)
	manifest.update(
		complete=eligible_limit is None or counts["eligible"] >= eligible_limit,
		counts=dict(counts),
		source_rows_sha256=decided_hash.hexdigest(),
		source_ids_sha256=file_hash(output / "source-ids.jsonl"),
	)
	manifest["eligible_sha256"] = file_hash(output / "eligible.jsonl")
	manifest_path.write_text(json.dumps(manifest, indent=4) + "\n")
	if not manifest["complete"]:
		raise ValueError(
			f"eligible target {eligible_limit} not reached: {counts['eligible']} qualified"
		)
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
	parser.add_argument(
		"--eligible-limit",
		type=int,
		default=DEFAULT_POOL_SIZE,
		help="Eligible target; 0 scans all supplied rows. Declare input selection separately.",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=DEFAULT_QUALIFICATION_WORKERS,
		help="Parallel sandbox workers; use 1 for sequential qualification",
	)
	arguments = parser.parse_args()
	manifest = prepare_corpus(
		_rows(arguments.input),
		_rows(arguments.evaluation),
		arguments.output,
		input_sha256=file_hash(arguments.input),
		workers=arguments.workers,
		eligible_limit=arguments.eligible_limit or None,
	)
	print(json.dumps(manifest, indent=4))


if __name__ == "__main__":
	main()
