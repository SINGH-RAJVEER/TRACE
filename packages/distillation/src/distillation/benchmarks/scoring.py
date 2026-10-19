"""Deterministic scoring for AG News, SQuAD, and GSM8K."""

import asyncio
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

from distillation.benchmarks.models import BenchmarkItem


@dataclass(frozen=True)
class AnswerScore:
	prediction: str | None
	correct: bool
	status: Literal["ok", "invalid", "truncated", "error", "timeout"]
	f1: float | None = None
	functional_trace: dict[str, str | int | bool] | None = None


async def evaluate_answer(item: BenchmarkItem, text: str, finish_reason: str | None) -> AnswerScore:
	"""Score the same response contract in capability and benign utility evaluations.

	Sandbox infrastructure errors propagate so a missing execution boundary cannot be
	reported as a model failure. Candidate failures and timeouts remain in the denominator.
	"""
	if finish_reason in {"length", "max_tokens"}:
		return AnswerScore(None, False, "truncated", 0.0 if item.metric == "squad" else None)
	if (
		finish_reason not in {"stop", "eos", "end_turn", "completed"}
		or "<think>" in text
		or "</think>" in text
	):
		return AnswerScore(None, False, "invalid", 0.0 if item.metric == "squad" else None)
	if item.metric == "pass_at_1":
		from coding.extract import extract_code
		from coding.scoring import CodingItem, pass_at_1
		from coding.tracing import execution_trace

		result = await asyncio.to_thread(pass_at_1, CodingItem(item.test_code), text)
		trace = None
		if result.status == "ok":
			observed = await asyncio.to_thread(
				execution_trace, text, item.test_code, item.entry_point
			)
			reference = await asyncio.to_thread(
				execution_trace, item.answer, item.test_code, item.entry_point
			)
			trace = asdict(observed)
			trace["valid"] = (
				observed.valid
				and reference.valid
				and (
					observed.inputs_sha256 == reference.inputs_sha256
					and observed.calls == reference.calls
				)
			)
		return AnswerScore(
			extract_code(text), result.correct, result.status, functional_trace=trace
		)
	prediction, correct = score_answer(item, text)
	status: Literal["ok", "invalid"] = "ok" if prediction is not None else "invalid"
	return AnswerScore(
		prediction,
		correct,
		status,
		(squad_f1(item, text) if status == "ok" else 0.0) if item.metric == "squad" else None,
	)


def squad_normalize(text: str) -> str:
	"""The official SQuAD v1.1 normalization: lowercase, no punctuation or articles."""
	text = "".join(char for char in text.lower() if char not in set(string.punctuation))
	return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def squad_f1(item: BenchmarkItem, response: str) -> float:
	"""Token F1 against the best-matching gold answer, as in the official script."""
	predicted = squad_normalize(response).split()
	best = 0.0
	for answer in item.answers:
		gold = squad_normalize(answer).split()
		common = sum((Counter(predicted) & Counter(gold)).values())
		if common:
			precision, recall = common / len(predicted), common / len(gold)
			best = max(best, 2 * precision * recall / (precision + recall))
	return best


def extract_answer(item: BenchmarkItem, response: str) -> str | None:
	if item.metric == "label_accuracy":
		answer = response.strip().strip("\"'").rstrip(".").strip().casefold()
		return next((label for label in item.choices if label.casefold() == answer), None)
	if item.metric == "squad":
		return squad_normalize(response) or None
	if item.metric == "numeric_exact_match":
		# A bare numeric answer is valid too; never guess using the last number in a rationale.
		answer = response.rsplit("####", 1)[-1].strip()
		if not re.fullmatch(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", answer):
			return None
		try:
			return str(Decimal(answer.replace(",", "")).normalize())
		except InvalidOperation:
			return None
	raise ValueError("coding responses require the sandboxed evaluate_answer path")


def score_answer(item: BenchmarkItem, response: str) -> tuple[str | None, bool]:
	prediction = extract_answer(item, response)
	if item.metric == "squad":
		accepted = {squad_normalize(answer) for answer in item.answers}
		return prediction, prediction is not None and prediction in accepted
	if item.metric == "numeric_exact_match":
		expected = extract_answer(item, item.answer)
	else:
		expected = item.answer
	return prediction, prediction is not None and prediction == expected
