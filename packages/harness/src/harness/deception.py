"""Keyed deceptive responses for accounts at L4, and the watermark test they enable.

A separately reported arm replaces refusal at L4 with plausible wrong answers, adapting
adaptive misinformation (Kariyappa and Qureshi, CVPR 2020) and keyed response changes
(DAWN, Szyller et al., ACM MM 2021) to a text API. Every wrong answer is derived from
HMAC-SHA256(secret, request), so repeated and Sybil queries receive the same answer and the
attacker cannot predict it. The same key later tests whether a suspect model learned those
answers. GSM8K traces can be rewritten by a model call, after Ma et al. (ACL 2026).
"""

import hashlib
import hmac
import re
import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Protocol

AG_NEWS_LABELS = ("World", "Sports", "Business", "Sci/Tech")
SQUAD_PASSAGE = re.compile(r"Passage: (.*?)\n\nQuestion: ", re.DOTALL)
NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
SENTENCE = re.compile(r"(?<=[.!?;])\s+")
CAPITALIZED = re.compile(r"\b[A-Z][\w'-]*(?:\s+(?:of|the|de|and|for|in)?\s*[A-Z][\w'-]*)*")
GSM8K_REWRITE_PROMPT = (
	"Rewrite the solution below so it reads naturally and keeps its structure, but contains "
	"one plausible arithmetic mistake that leads to the final answer {target}. End with a "
	"line of the form #### {target}. Return only the rewritten solution.\n\n{solution}"
)


def keyed_digest(key: bytes, *parts: str) -> bytes:
	return hmac.new(key, "\x1f".join(parts).encode(), hashlib.sha256).digest()


def keyed_unit(key: bytes, *parts: str) -> float:
	"""A keyed, uniform value in [0, 1) that the attacker cannot predict."""
	return int.from_bytes(keyed_digest(key, *parts)[:8]) / 2**64


def keyed_choice[T](key: bytes, options: Sequence[T], *parts: str) -> T:
	return options[int.from_bytes(keyed_digest(key, *parts)[:8]) % len(options)]


def squad_normalize(text: str) -> str:
	"""The official SQuAD normalization, so the deceptive span never matches the answer."""
	text = "".join(char for char in text.lower() if char not in set(string.punctuation))
	return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def _number(text: str) -> Decimal | None:
	try:
		return Decimal(text.replace(",", "").strip())
	except InvalidOperation:
		return None


class TaskDeceiver(Protocol):
	def target(self, key: bytes, request: str, response: str) -> str | None:
		"""The keyed wrong answer for this request, or None when none can be formed."""
		...

	def render(self, key: bytes, request: str, response: str, target: str) -> str:
		"""The full response text that carries the wrong answer."""
		...

	def extract(self, response: str) -> str | None:
		"""The answer a response gives, normalized for comparison with a target."""
		...

	def answer(self, target: str) -> str | None:
		"""The normalized answer a response carrying this target gives."""
		...


@dataclass(frozen=True)
class LabelDeceiver:
	"""AG News: one of the other labels, keyed by the request."""

	labels: tuple[str, ...] = AG_NEWS_LABELS

	def extract(self, response: str) -> str | None:
		answer = response.strip().strip("\"'").rstrip(".").strip().casefold()
		return next((label for label in self.labels if label.casefold() == answer), None)

	def target(self, key: bytes, request: str, response: str) -> str | None:
		given = self.extract(response)
		options = [label for label in self.labels if label != given]
		return keyed_choice(key, options, "label", request)

	def render(self, key: bytes, request: str, response: str, target: str) -> str:
		return target

	def answer(self, target: str) -> str | None:
		return target


@dataclass(frozen=True)
class SpanDeceiver:
	"""SQuAD: a span of the same kind from the same passage that shares no answer token."""

	max_words: int = 6

	def extract(self, response: str) -> str | None:
		return squad_normalize(response) or None

	def candidates(self, passage: str, response: str) -> list[str]:
		"""Spans of the answer's kind when it has one, otherwise word windows of its length."""
		answer = response.strip()
		head = re.sub(r"^(?:the|a|an)\s+", "", answer, flags=re.IGNORECASE)
		if NUMBER.search(answer):
			found = NUMBER.findall(passage)
		elif head[:1].isupper():
			found = [match.group(0) for match in CAPITALIZED.finditer(passage)]
		else:
			found = []
		excluded = set(squad_normalize(answer).split())

		def usable(spans: list[str]) -> list[str]:
			kept: list[str] = []
			for span in spans:
				tokens = set(squad_normalize(span).split())
				if tokens and not tokens & excluded and span not in kept:
					kept.append(span)
			return kept

		if kept := usable(found):
			return kept
		size = max(1, min(len(answer.split()), self.max_words))
		windows: list[str] = []
		# Windows stay inside one sentence so the span reads like an answer.
		for sentence in SENTENCE.split(passage):
			words = sentence.split()
			windows += [
				" ".join(words[i : i + size]).strip(string.punctuation)
				for i in range(len(words) - size + 1)
			]
		return usable(windows)

	def target(self, key: bytes, request: str, response: str) -> str | None:
		match = SQUAD_PASSAGE.search(request)
		options = self.candidates(match.group(1), response) if match else []
		return keyed_choice(key, options, "span", request) if options else None

	def render(self, key: bytes, request: str, response: str, target: str) -> str:
		return target

	def answer(self, target: str) -> str | None:
		return squad_normalize(target) or None


@dataclass(frozen=True)
class NumericTraceDeceiver:
	"""GSM8K: a keyed wrong final number, carried by a rewritten trace.

	With a rewriter, a model call rewrites the trace to reach that number, after Ma et al.
	Without one, the final line and the last occurrence of the original number before it are
	replaced. That fallback is cheap but easier to spot.
	"""

	rewriter: Callable[[str], str] | None = None
	offsets: tuple[int, ...] = (-10, -5, -2, -1, 1, 2, 5, 10)

	def extract(self, response: str) -> str | None:
		if "####" not in response:
			return None
		match = NUMBER.search(response.rsplit("####", 1)[1])
		value = _number(match.group(0)) if match else None
		return None if value is None else str(value.normalize())

	def target(self, key: bytes, request: str, response: str) -> str | None:
		given = self.extract(response)
		if given is None:
			return None
		value = Decimal(given)
		offset = keyed_choice(key, self.offsets, "number", request)
		changed = value + offset
		if value >= 0 > changed:
			changed = value - offset
		return str(changed.normalize())

	def answer(self, target: str) -> str | None:
		value = _number(target)
		return None if value is None else str(value.normalize())

	def render(self, key: bytes, request: str, response: str, target: str) -> str:
		if self.rewriter is not None:
			text = self.rewriter(GSM8K_REWRITE_PROMPT.format(target=target, solution=response))
		else:
			body, final = response.rsplit("####", 1)
			original = NUMBER.search(final)
			if original and original.group(0) in body:
				head, _, tail = body.rpartition(original.group(0))
				body = head + target + tail
			text = body
		text = text.rsplit("####", 1)[0].rstrip()
		return f"{text}\n#### {target}"


DECEIVERS: dict[str, Callable[[], TaskDeceiver]] = {
	"ag_news": LabelDeceiver,
	"squad": SpanDeceiver,
	"gsm8k": NumericTraceDeceiver,
}


def deceive(deceiver: TaskDeceiver, key: bytes, request: str, response: str) -> str | None:
	"""The deceptive response text, or None when the deceiver cannot form one."""
	target = deceiver.target(key, request, response)
	return None if target is None else deceiver.render(key, request, response, target)


def released_text(
	deceptive: bool,
	refused: bool,
	deceiver: TaskDeceiver | None,
	key: bytes | None,
	request: str,
	response: str,
) -> str | None:
	"""What the client receives. A deceiver that cannot form an answer falls back to refusal."""
	if refused:
		return None
	if not deceptive:
		return response
	if deceiver is None or key is None:
		raise ValueError("a deceptive decision needs a deceiver and a key")
	return deceive(deceiver, key, request, response)


@dataclass(frozen=True)
class Probe:
	"""One watermark probe: a request, the victim's true response, and the suspect's answer."""

	request: str
	victim_response: str
	suspect_response: str


def control_key(index: int) -> bytes:
	return hashlib.sha256(f"harness-watermark-control:{index}".encode()).digest()


def watermark_agreement(deceiver: TaskDeceiver, key: bytes, probes: Sequence[Probe]) -> float:
	"""The share of probes whose suspect answer equals the keyed wrong answer."""
	hits = total = 0
	for probe in probes:
		target = deceiver.target(key, probe.request, probe.victim_response)
		if target is None:
			continue
		total += 1
		expected = deceiver.answer(target)
		hits += expected is not None and deceiver.extract(probe.suspect_response) == expected
	return hits / total if total else 0.0


def watermark_test(
	deceiver: TaskDeceiver, key: bytes, probes: Sequence[Probe], controls: int = 999
) -> dict[str, float]:
	"""Compare agreement under the secret key with agreement under unrelated control keys.

	The p-value is the upper-tail rank among control keys, so it assumes nothing about how
	often a clean model happens to give a particular wrong answer.
	"""
	observed = watermark_agreement(deceiver, key, probes)
	null = [watermark_agreement(deceiver, control_key(i), probes) for i in range(controls)]
	exceed = sum(value >= observed for value in null)
	return {
		"agreement": observed,
		"control_mean": sum(null) / len(null) if null else 0.0,
		"p_value": (1 + exceed) / (controls + 1),
		"probes": float(len(probes)),
	}
