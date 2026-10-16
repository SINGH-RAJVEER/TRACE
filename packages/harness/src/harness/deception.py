"""Keyed noise for accounts at L4, and the watermark test it enables.

Instead of refusing a flagged account, the harness answers with cheap noise in the task's
output format. No teacher call is made, so serving noise costs microseconds of CPU. Noise is
pseudo-random but keyed: it is derived from HMAC-SHA256(secret, request), so repeated and
Sybil queries for the same request receive the same answer and the attacker cannot predict
it. This adapts adaptive misinformation (Kariyappa and Qureshi, CVPR 2020) and keyed response
changes (DAWN, Szyller et al., ACM MM 2021) to a text API. The same key later tests whether a
suspect model learned the noise.
"""

import hashlib
import hmac
import random
import re
import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

AG_NEWS_LABELS = ("World", "Sports", "Business", "Sci/Tech")
SQUAD_PASSAGE = re.compile(r"Passage: (.*?)\n\nQuestion: ", re.DOTALL)
NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
SENTENCE = re.compile(r"(?<=[.!?;])\s+")
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
FILLER = ("the", "a", "of", "and", "to", "in", "is", "that", "for", "with", "as", "this")
TASKS = ("ag_news", "squad", "gsm8k", "text")


def keyed_digest(key: bytes, *parts: str) -> bytes:
	return hmac.new(key, "\x1f".join(parts).encode(), hashlib.sha256).digest()


def keyed_unit(key: bytes, *parts: str) -> float:
	"""A keyed, uniform value in [0, 1) that the attacker cannot predict."""
	return int.from_bytes(keyed_digest(key, *parts)[:8]) / 2**64


def keyed_rng(key: bytes, *parts: str) -> random.Random:
	return random.Random(int.from_bytes(keyed_digest(key, *parts)[:16]))


def squad_normalize(text: str) -> str:
	"""The official SQuAD normalization."""
	text = "".join(char for char in text.lower() if char not in set(string.punctuation))
	return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def _decimal(text: str) -> Decimal | None:
	try:
		return Decimal(text.replace(",", "").strip())
	except InvalidOperation:
		return None


def _format(value: Decimal) -> str:
	return format(value.normalize(), "f")


def label_noise(rng: random.Random, request: str, max_tokens: int) -> str:
	return rng.choice(AG_NEWS_LABELS)


def span_noise(rng: random.Random, request: str, max_tokens: int) -> str:
	"""A short span from one sentence of the passage, or of the request without one."""
	match = SQUAD_PASSAGE.search(request)
	source = match.group(1) if match else request
	sentences = [s.split() for s in SENTENCE.split(source) if s.split()] or [["unknown"]]
	words = rng.choice(sentences)
	size = min(rng.choice((1, 1, 2, 2, 3, 4)), len(words), max(1, max_tokens))
	start = rng.randrange(len(words) - size + 1)
	return " ".join(words[start : start + size]).strip(string.punctuation) or words[start]


def arithmetic_noise(rng: random.Random, request: str, max_tokens: int) -> str:
	"""A short worked solution chaining the problem's numbers, ending in #### N.

	Each step combines the running total with a number from the problem, never goes negative,
	and never subtracts a number from itself. The answer depends only on the key and request;
	the token budget only drops early lines.
	"""
	given = [v for v in (_decimal(n) for n in NUMBER.findall(request)) if v is not None and v > 0]
	while len(given) < 2:
		given.append(Decimal(rng.randint(2, 12)))
	total = rng.choice(given)
	lines: list[str] = []
	for step in range(rng.randint(2, 4)):
		value = rng.choice([v for v in given if v != total] or given)
		operator = rng.choice("+-*")
		if operator == "*" and total * value > 1_000_000:
			operator = "+"
		if operator == "-" and value >= total:
			operator = "+"
		result = (
			total + value
			if operator == "+"
			else total - value
			if operator == "-"
			else total * value
		)
		opener = "First, " if step == 0 else rng.choice(("Then ", "Next, ", "So ", ""))
		lines.append(f"{opener}{_format(total)} {operator} {_format(value)} = {_format(result)}.")
		total = result
	# About twelve tokens per line and four for the answer line.
	keep = max(1, (max_tokens - 4) // 12)
	return "\n".join([*lines[-keep:], f"#### {_format(total)}"])


def text_noise(rng: random.Random, request: str, max_tokens: int) -> str:
	"""Sentence-shaped word salad from the request's own vocabulary."""
	vocabulary = WORD.findall(request) or list(FILLER)
	budget = max(3, int(max_tokens * 0.7))
	sentences: list[str] = []
	used = 0
	for _ in range(rng.randint(1, 3)):
		length = min(rng.randint(6, 14), budget - used)
		if length < 3:
			break
		words = [rng.choice(vocabulary if rng.random() < 0.6 else FILLER) for _ in range(length)]
		sentences.append(" ".join(words).capitalize() + ".")
		used += length
	return " ".join(sentences) or "No."


GENERATORS: dict[str, Callable[[random.Random, str, int], str]] = {
	"ag_news": label_noise,
	"squad": span_noise,
	"gsm8k": arithmetic_noise,
	"text": text_noise,
}


def noise(task: str, key: bytes, request: str, max_tokens: int) -> str:
	"""The keyed noise response for this request, in the task's output format."""
	if task not in GENERATORS:
		raise ValueError(f"task must be one of {TASKS}")
	return GENERATORS[task](keyed_rng(key, "noise", task, request), request, max_tokens)


def extract(task: str, response: str) -> str | None:
	"""The answer a response gives, normalized for watermark comparison."""
	if task == "ag_news":
		answer = response.strip().strip("\"'").rstrip(".").strip().casefold()
		return next((label for label in AG_NEWS_LABELS if label.casefold() == answer), None)
	if task == "squad":
		return squad_normalize(response) or None
	if task == "gsm8k":
		if "####" not in response:
			return None
		match = NUMBER.search(response.rsplit("####", 1)[1])
		value = _decimal(match.group(0)) if match else None
		return None if value is None else _format(value)
	raise ValueError("watermark tests need an ag_news, squad, or gsm8k task")


@dataclass(frozen=True)
class Probe:
	"""One watermark probe: a request and the suspect model's response to it."""

	request: str
	suspect_response: str


def control_key(index: int) -> bytes:
	return hashlib.sha256(f"harness-watermark-control:{index}".encode()).digest()


def watermark_agreement(task: str, key: bytes, probes: Sequence[Probe]) -> float:
	"""The share of probes whose suspect answer equals the keyed noise answer."""
	hits = 0
	for probe in probes:
		expected = extract(task, noise(task, key, probe.request, 256))
		hits += expected is not None and extract(task, probe.suspect_response) == expected
	return hits / len(probes) if probes else 0.0


def watermark_test(
	task: str, key: bytes, probes: Sequence[Probe], controls: int = 999
) -> dict[str, float]:
	"""Compare agreement under the secret key with agreement under unrelated control keys.

	The p-value is the upper-tail rank among control keys, so it assumes nothing about how
	often a clean model happens to give a particular answer.
	"""
	observed = watermark_agreement(task, key, probes)
	null = [watermark_agreement(task, control_key(i), probes) for i in range(controls)]
	exceed = sum(value >= observed for value in null)
	return {
		"agreement": observed,
		"control_mean": sum(null) / len(null) if null else 0.0,
		"p_value": (1 + exceed) / (controls + 1),
		"probes": float(len(probes)),
	}
