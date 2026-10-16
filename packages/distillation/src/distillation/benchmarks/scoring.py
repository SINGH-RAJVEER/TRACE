"""Deterministic scoring for AG News, SQuAD, and GSM8K."""

import re
import string
from collections import Counter
from decimal import Decimal, InvalidOperation

from distillation.benchmarks.models import BenchmarkItem


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
