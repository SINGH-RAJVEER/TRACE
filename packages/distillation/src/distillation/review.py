"""Automatic task validation that turns a teacher ledger into review decisions."""

import ast
import asyncio
import json
import re
import tomllib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillation.io import read_jsonl, write_jsonl
from distillation.schemas import (
	LedgerRecord,
	Message,
	OutputConstraints,
	ReferenceRecord,
	ReviewRecord,
)

REVIEWER = "auto-review-v1"
FENCE = re.compile(r"```[ \t]*(?:python|py|json)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
REASONED_CHOICE = re.compile(r"^\W*(?:[Ff]inal\s+)?[Aa]nswer\W*:\W*([A-Z])\b", re.MULTILINE)
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")
WORD = re.compile(r"\b\w[\w'-]*\b")


@dataclass(frozen=True)
class Decision:
	status: str
	checks: list[str]
	reason: str | None = None


def _accept(*checks: str) -> Decision:
	return Decision("accepted", list(checks))


def _reject(check: str, reason: str) -> Decision:
	return Decision("rejected", [check], reason)


def _normalized(text: str) -> str:
	text = re.sub(r"\s+", " ", text.strip().strip("`").strip()).casefold()
	text = re.sub(r"\s*,\s*", ", ", text)
	return text.rstrip(".").strip("\"'")


def code_block(text: str) -> str:
	"""The first fenced block, or the whole response when it has no fence."""
	match = FENCE.search(text)
	return (match[1] if match else text).strip("\n")


def _numeric(text: str) -> str | None:
	from decimal import Decimal, InvalidOperation

	if "####" not in text:
		return None
	answer = text.rsplit("####", 1)[1].strip().rstrip(".").removeprefix("$").strip()
	if not re.fullmatch(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", answer):
		return None
	try:
		return str(Decimal(answer.replace(",", "")).normalize())
	except InvalidOperation:
		return None


def check_constraints(text: str, constraints: OutputConstraints) -> Decision | None:
	words = len(WORD.findall(text))
	if constraints.max_words is not None and words > constraints.max_words:
		return _reject("constraints", f"{words} words exceeds {constraints.max_words}")
	if constraints.min_words is not None and words < constraints.min_words:
		return _reject("constraints", f"{words} words is under {constraints.min_words}")
	if constraints.bullet_count is not None:
		lines = [line for line in text.splitlines() if line.strip()]
		bullets = sum(bool(BULLET.match(line)) for line in lines)
		if bullets != constraints.bullet_count or bullets != len(lines):
			return _reject("constraints", f"expected {constraints.bullet_count} bullets only")
	lowered = text.casefold()
	for phrase in constraints.must_include:
		if phrase.casefold() not in lowered:
			return _reject("constraints", f"missing {phrase!r}")
	for phrase in constraints.must_exclude:
		if re.search(rf"\b{re.escape(phrase.casefold())}\b", lowered):
			return _reject("constraints", f"contains {phrase!r}")
	return None


def check_response(text: str, reference: ReferenceRecord) -> Decision:
	"""Validate one response. Code tests run separately, in a sandbox, when enabled."""
	if reference.constraints and (failure := check_constraints(text, reference.constraints)):
		return failure
	kind = reference.kind
	if kind == "numeric":
		expected = _numeric(f"#### {reference.answer}")
		actual = _numeric(text)
		if actual is None:
			return _reject("numeric", "no #### answer")
		return _accept("numeric") if actual == expected else _reject("numeric", "wrong answer")
	if kind == "choice":
		actual = text.strip().strip(".").strip()
		if not re.fullmatch(r"[A-Z]", actual):
			return _reject("choice", "not a single letter")
		return (
			_accept("choice") if actual == reference.answer else _reject("choice", "wrong answer")
		)
	if kind == "choice_reasoning":
		matches = REASONED_CHOICE.findall(text)
		if not matches:
			return _reject("choice", "no final answer line")
		if matches[-1].upper() != reference.answer:
			return _reject("choice", "wrong answer")
		return _accept("choice")
	if kind in {"python_tests", "python_code"}:
		source = code_block(text)
		try:
			ast.parse(source)
		except SyntaxError:
			return _reject("python_syntax", "code does not parse")
		if reference.entry_point and not re.search(
			rf"\bdef\s+{re.escape(reference.entry_point)}\s*\(", source
		):
			return _reject("python_syntax", f"missing def {reference.entry_point}")
		return _accept("python_syntax")
	if kind in {"json_equals", "json_valid"}:
		try:
			value = json.loads(code_block(text))
		except ValueError:
			return _reject("json", "invalid JSON")
		if kind == "json_valid":
			return _accept("json")
		expected = json.loads(reference.answer or "null")
		return (
			_accept("json", "json_equals") if value == expected else _reject("json", "wrong fields")
		)
	if kind == "exact":
		if _normalized(text) == _normalized(reference.answer or ""):
			return _accept("exact")
		return _reject("exact", "wrong answer")
	return _accept("constraints" if reference.constraints else "nonempty")


class LengthChecker:
	"""Rejects records the training config would refuse, using the student tokenizer."""

	def __init__(self, config_path: Path) -> None:
		from distillation.training import TrainingConfig, _load_tokenizer

		with config_path.open("rb") as source:
			data: dict[str, Any] = tomllib.load(source)
		# The prepared dataset does not exist yet, so skip path validation.
		defaults = {
			name: field.get_default(call_default_factory=True)
			for name, field in TrainingConfig.model_fields.items()
			if not field.is_required()
		}
		config = TrainingConfig.model_construct(**{**defaults, **data})
		self.max_length = int(config.max_length)
		self.kwargs = dict(config.chat_template_kwargs)
		self.tokenizer = _load_tokenizer(config)

	def length(self, prompt: list[Message], completion: list[Message]) -> int:
		conversation = [message.model_dump(mode="json") for message in [*prompt, *completion]]
		token_ids = self.tokenizer.apply_chat_template(
			conversation,
			tokenize=True,
			add_generation_prompt=False,
			return_dict=False,
			**self.kwargs,
		)
		return len(token_ids)


async def _run_code_tests(
	pending: list[tuple[LedgerRecord, ReferenceRecord]],
	code_image: str,
	timeout_seconds: int,
	concurrency: int,
	progress: Callable[[int, int], None],
) -> dict[str, bool]:
	from distillation.benchmarks.models import BenchmarkItem
	from distillation.benchmarks.scoring import resolve_code_image, score_code

	image_id = resolve_code_image(code_image)
	semaphore = asyncio.Semaphore(concurrency)
	finished = 0

	async def run(record: LedgerRecord, reference: ReferenceRecord) -> tuple[str, bool]:
		item = BenchmarkItem(
			id=record.example_id,
			benchmark="review",
			category=record.task,
			question=record.example_id,
			messages=record.prompt,
			metric="pass_at_1",
			answer="passes tests",
			test_code=reference.tests,
			entry_point=reference.entry_point,
		)
		async with semaphore:
			passed, _ = await score_code(
				item, code_block(record.completion[0].content), image_id, timeout_seconds
			)
		nonlocal finished
		finished += 1
		progress(finished, len(pending))
		return record.example_id, passed

	results = await asyncio.gather(*(run(record, reference) for record, reference in pending))
	return dict(results)


def review_ledger(
	*,
	ledger_path: Path,
	references_path: Path | None,
	output_path: Path,
	overrides_path: Path | None = None,
	config_path: Path | None = None,
	code_image: str | None = None,
	code_timeout_seconds: int = 10,
	code_concurrency: int = 4,
	progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
	"""Write one decision per ledger record. Human overrides always win.

	progress receives a phase description, the records finished, and the phase total.
	"""

	def report(phase: str, done: int, total: int) -> None:
		if progress:
			progress(phase, done, total)

	for path in (ledger_path, references_path, overrides_path, config_path):
		if path and output_path.resolve() == path.resolve():
			raise ValueError("review output must not overwrite an input artifact")
	ledger = read_jsonl(ledger_path, LedgerRecord)
	references: dict[str, ReferenceRecord] = {}
	if references_path:
		for reference in read_jsonl(references_path, ReferenceRecord):
			if reference.example_id in references:
				raise ValueError(f"duplicate reference for example_id: {reference.example_id}")
			references[reference.example_id] = reference
	overrides = (
		{review.example_id: review for review in read_jsonl(overrides_path, ReviewRecord)}
		if overrides_path
		else {}
	)
	lengths = LengthChecker(config_path) if config_path else None

	decisions: dict[str, Decision] = {}
	code_pending: list[tuple[LedgerRecord, ReferenceRecord]] = []
	kinds: Counter[str] = Counter()
	report("Checking answers", 0, len(ledger))
	for done, record in enumerate(ledger, 1):
		reference = references.get(record.example_id) or ReferenceRecord(
			example_id=record.example_id, kind="general"
		)
		kinds[reference.kind] += 1
		if record.quality.status == "rejected":
			decision = _reject("collector", record.quality.rejection_reason or "collector rejected")
		else:
			decision = check_response(record.completion[0].content, reference)
		if decision.status == "accepted" and lengths:
			length = lengths.length(record.prompt, record.completion)
			if length > lengths.max_length:
				decision = _reject("length", f"{length} tokens exceeds {lengths.max_length}")
			else:
				decision = Decision("accepted", [*decision.checks, "length"])
		if decision.status == "accepted" and reference.kind == "python_tests" and code_image:
			code_pending.append((record, reference))
		decisions[record.example_id] = decision
		report("Checking answers", done, len(ledger))

	if code_pending and code_image:
		report("Running code tests", 0, len(code_pending))
		passed = asyncio.run(
			_run_code_tests(
				code_pending,
				code_image,
				code_timeout_seconds,
				code_concurrency,
				lambda done, total: report("Running code tests", done, total),
			)
		)
		for example_id, ok in passed.items():
			previous = decisions[example_id]
			decisions[example_id] = (
				Decision("accepted", [*previous.checks, "python_tests"])
				if ok
				else _reject("python_tests", "tests failed")
			)

	reviews: list[ReviewRecord] = []
	statuses: Counter[str] = Counter()
	reasons: Counter[str] = Counter()
	accepted_by_task: Counter[str] = Counter()
	for record in ledger:
		override = overrides.get(record.example_id)
		if override is not None:
			review = override
		else:
			decision = decisions[record.example_id]
			review = ReviewRecord(
				example_id=record.example_id,
				status="accepted" if decision.status == "accepted" else "rejected",
				checks=decision.checks,
				reviewer=REVIEWER,
				reason=decision.reason,
			)
			if decision.reason:
				reasons[f"{decision.checks[0]}: {re.sub(r'\d+', 'N', decision.reason)}"] += 1
		reviews.append(review)
		statuses[review.status] += 1
		if review.status == "accepted":
			accepted_by_task[record.task] += 1
	write_jsonl(output_path, reviews)
	summary = {
		"reviewer": REVIEWER,
		"records": len(ledger),
		"status": dict(statuses),
		"overrides": sum(record.example_id in overrides for record in ledger),
		"reference_kinds": dict(kinds),
		"rejection_reasons": dict(reasons.most_common()),
		"accepted_by_task": dict(sorted(accepted_by_task.items())),
		"code_tests_executed": bool(code_image),
		"length_checked": lengths is not None,
	}
	output_path.with_suffix(".summary.json").write_text(
		json.dumps(summary, indent="\t", sort_keys=True) + "\n", encoding="utf-8"
	)
	return summary
