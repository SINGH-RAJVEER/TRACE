"""Opt-in release-boundary output policy: what the gateway releases instead of a clean draft.

The teacher's draft is buffered, never streamed, so it can be replaced before release. The draft
stays in this process: clients, logs and stored records get its SHA-256 at most. Rewrite failures
go to the declared fallback and never to the draft. Each task in ANSWER_FORMATS has its own locked
answer; other tasks follow `unsupported_task` and are recorded as unprotected.

This has never run on a real teacher or student, so it establishes no suppression. A correct
released answer is still training data, and the policy changes the product for every client.
"""

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from distillation.benchmarks.scoring import numeric_answer, squad_normalize
from distillation.io import canonical_json
from distillation.output_sets import INSTRUCTIONS, MAX_COPIED_FRACTION, copied_fraction
from distillation.schemas import Message, TeacherGeneration, TeacherGenerationRequest
from pydantic import BaseModel, ConfigDict, Field, model_validator

OUTPUT_POLICY_VERSION = "output-policy-v1"
RECEIPT_HEADER = "X-Harness-Output-Receipt"
REWRITE_MODES = ("compact_verified", "paraphrase", "dense_locked")
ANSWER_MARKER = "####"
ANSWER_ONLY_TASKS = {"ag_news", "squad"}
AG_NEWS_LABELS = ("World", "Sports", "Business", "Sci/Tech")
# A development value: longer SQuAD output may carry reasoning that cannot be separated from it.
SQUAD_MAX_ANSWER_WORDS = 32
_CODE_FENCE = "```"
_PYTHON_FENCE = re.compile(r"```(?:python|py)\s*\n.*?```", re.DOTALL | re.IGNORECASE)
_TRUNCATED = {"length", "max_tokens"}


class OutputPolicy(BaseModel):
	"""The HARNESS_OUTPUT_POLICY JSON. Each field changes the policy revision and state scope."""

	model_config = ConfigDict(extra="forbid", frozen=True)

	version: Literal["output-policy-v1"]
	mode: Literal["final_answer_only", "compact_verified", "paraphrase", "dense_locked"]
	# Used when a rewrite cannot be released. There is deliberately no default.
	fallback: Literal["withhold", "final_answer"]
	tasks: tuple[Literal["gsm8k", "coding", "ag_news", "squad"], ...] = Field(min_length=1)
	# Requests for another task: refuse before admission, or release unmodified and say so.
	unsupported_task: Literal["refuse", "pass_through"]
	rewrite_max_output_tokens: int | None = Field(default=None, ge=1, le=32768)
	rewrite_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
	# A rewrite sharing at least this fraction of its word 5-grams with the draft is a copy.
	max_copied_fraction: float = Field(default=MAX_COPIED_FRACTION, gt=0, le=1, allow_inf_nan=False)

	@model_validator(mode="after")
	def require_rewrite_bounds(self) -> OutputPolicy:
		bounds = (self.rewrite_max_output_tokens, self.rewrite_timeout_seconds)
		if self.mode in REWRITE_MODES and None in bounds:
			raise ValueError(
				f"{self.mode} requires rewrite_max_output_tokens and rewrite_timeout_seconds"
			)
		if self.mode not in REWRITE_MODES and bounds != (None, None):
			raise ValueError(f"{self.mode} makes no rewrite call, so it takes no rewrite bounds")
		answer_only = set(self.tasks) & ANSWER_ONLY_TASKS
		if self.mode in REWRITE_MODES and self.fallback == "withhold" and answer_only:
			raise ValueError(
				f"{sorted(answer_only)} have no explanation to rewrite, so {self.mode} with "
				"fallback withhold would withhold every response"
			)
		return self

	def binding(self) -> dict[str, Any]:
		"""Bound into the policy revision and state scope, with the instruction text's hash."""
		instruction = INSTRUCTIONS.get(self.mode)
		return {
			**self.model_dump(mode="json"),
			"instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest()
			if instruction
			else None,
		}

	@property
	def digest(self) -> str:
		return hashlib.sha256(canonical_json(self.binding()).encode()).hexdigest()

	def event_fields(self) -> dict[str, Any]:
		"""What a decision event records before there is an outcome."""
		return {
			"output_policy": {
				"version": self.version,
				"mode": self.mode,
				"digest": self.digest,
				"qualified": False,
			}
		}


@dataclass(frozen=True)
class LockedAnswer:
	"""A draft split at its one answer marker. `answer` is immutable once extracted."""

	explanation: str
	answer: str
	value: str


def parse_gsm8k(text: str) -> LockedAnswer | None:
	"""The explanation and final answer of text ending in one `#### NUMBER`, else None.

	`value` uses the benchmark scorer's normalization, so equality here is what a later student
	evaluation scores as the same answer.
	"""
	if text.count(ANSWER_MARKER) != 1:
		return None
	explanation, tail = text.split(ANSWER_MARKER)
	value = numeric_answer(tail)
	return None if value is None else LockedAnswer(explanation.strip(), tail.strip(), value)


def final_answer_text(locked: LockedAnswer) -> str:
	return f"{ANSWER_MARKER} {locked.answer}"


def parse_coding(text: str) -> LockedAnswer | None:
	"""The prose around one Python fence, and the fence itself, unmodified, as the answer."""
	from coding.extract import extract_code

	matches = _PYTHON_FENCE.findall(text)
	if len(matches) != 1 or text.count(_CODE_FENCE) != 2:
		return None
	code = extract_code(matches[0])
	if code is None:
		return None
	before, after = text.split(matches[0])
	return LockedAnswer(f"{before.strip()}\n{after.strip()}".strip(), matches[0], code)


def parse_ag_news(text: str) -> LockedAnswer | None:
	"""One topic label and nothing else, read the way the benchmark's label scorer reads it."""
	answer = text.strip().strip("\"'").rstrip(".").strip().casefold()
	label = next((name for name in AG_NEWS_LABELS if name.casefold() == answer), None)
	return None if label is None else LockedAnswer("", label, label)


def parse_squad(text: str) -> LockedAnswer | None:
	"""One short line. SQuAD has no marker, so length is the only sign of added reasoning."""
	answer = text.strip()
	if "\n" in answer or len(answer.split()) > SQUAD_MAX_ANSWER_WORDS:
		return None
	value = squad_normalize(answer)
	return LockedAnswer("", answer, value) if value else None


@dataclass(frozen=True)
class AnswerFormat:
	"""How a task's draft splits into an explanation and an answer the server locks."""

	parse: Callable[[str], LockedAnswer | None]
	final: Callable[[LockedAnswer], str]
	# A rewrite holding this would be a second, unchecked answer.
	marker: str | None = None
	separator: str = "\n"


def _answer(locked: LockedAnswer) -> str:
	return locked.answer


ANSWER_FORMATS = {
	"gsm8k": AnswerFormat(parse_gsm8k, final_answer_text, ANSWER_MARKER),
	"coding": AnswerFormat(parse_coding, _answer, _CODE_FENCE, "\n\n"),
	"ag_news": AnswerFormat(parse_ag_news, _answer),
	"squad": AnswerFormat(parse_squad, _answer),
}


def _sha256(text: str) -> str:
	return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class ProviderCall:
	"""One teacher call in the provider-cost ledger. Unknown counts stay None."""

	# "interrupted" remains if the request was cancelled while this call was in flight.
	status: Literal["ok", "error", "timeout", "interrupted"]
	input_tokens: int | None = None
	output_tokens: int | None = None
	max_output_tokens: int | None = None
	error: str | None = None


def _provider_call(
	generation: TeacherGeneration, count: Callable[[str], int], max_output_tokens: int
) -> ProviderCall:
	reported = generation.usage.output_tokens
	return ProviderCall(
		"ok",
		generation.usage.input_tokens,
		count(generation.message.content) if reported is None else reported,
		max_output_tokens,
	)


@dataclass
class OutputAttempt:
	"""The record of one admission under the policy, filled in as far as it got."""

	policy: OutputPolicy
	domain: str
	outcome: str = "no_draft"
	failure: str | None = None
	fallback_used: str | None = None
	draft_sha256: str | None = None
	released_sha256: str | None = None
	released_tokens: int | None = None
	calls: dict[str, ProviderCall] = field(default_factory=dict)

	@property
	def protected(self) -> bool:
		return self.domain in self.policy.tasks

	def release(self, text: str, tokens: int, outcome: str) -> None:
		self.outcome = outcome
		self.released_sha256 = _sha256(text)
		self.released_tokens = tokens

	def withhold(self, failure: str) -> None:
		self.outcome = "withheld"
		self.fallback_used = "withhold"
		self.failure = self.failure or failure

	def settle(self, status: str) -> None:
		"""Align the record with the ledger: only a settled release counts as released."""
		if status == "released":
			return
		if self.released_sha256 is not None:
			self.failure = self.failure or "not_settled_as_released"
			self.fallback_used = "withhold"
		self.released_sha256 = None
		self.released_tokens = None
		if status == "cancelled":
			self.outcome = "cancelled"
		elif self.draft_sha256 is not None:
			self.outcome = "withheld"

	def receipt(self) -> dict[str, Any]:
		"""The operator's record. It never holds draft text, only the draft's hash."""
		return {
			**self.client_receipt(),
			"digest": self.policy.digest,
			"domain": self.domain,
			"failure": self.failure,
			"draft_sha256": self.draft_sha256,
			# Provider cost is separate from the released tokens that entitlements charge.
			"provider": {name: vars(call) for name, call in self.calls.items()},
		}

	def client_receipt(self) -> dict[str, Any]:
		"""Sent with a response. Provider counts and the draft hash would reveal the draft's
		length or confirm a guess at its text, so they stay out."""
		return {
			"version": self.policy.version,
			"mode": self.policy.mode,
			"outcome": self.outcome,
			"protected": self.protected,
			"fallback": self.fallback_used,
			"released_sha256": self.released_sha256,
			"released_tokens": self.released_tokens,
			"qualified": False,
		}

	def receipt_header(self) -> str:
		return json.dumps(self.client_receipt(), sort_keys=True, separators=(",", ":"))


def rewrite_request(
	policy: OutputPolicy, explanation: str, request: TeacherGenerationRequest
) -> TeacherGenerationRequest:
	"""The second call: the explanation alone, so the model cannot alter the locked answer.

	Bounded by the declared cap and by the client's own maximum, which the reservation holds.
	"""
	assert policy.rewrite_max_output_tokens is not None
	return TeacherGenerationRequest(
		messages=[Message(role="user", content=f"{INSTRUCTIONS[policy.mode]}\n\n{explanation}")],
		temperature=0.0,
		max_output_tokens=min(policy.rewrite_max_output_tokens, request.max_output_tokens),
		seed=request.seed,
		enable_thinking=False,
	)


async def _rewrite(
	attempt: OutputAttempt,
	locked: LockedAnswer,
	request: TeacherGenerationRequest,
	call: Callable[[TeacherGenerationRequest], Awaitable[TeacherGeneration]],
	count: Callable[[str], int],
) -> str | None:
	"""A validated rewrite with the locked answer attached, or None with `failure` set.

	Teacher errors may echo the prompt, which is draft text, so only the error's type is kept.
	Cancellation is not caught: the client left and nothing is released.
	"""
	policy = attempt.policy
	answer_format = ANSWER_FORMATS[attempt.domain]
	assert policy.rewrite_timeout_seconds is not None
	prompt = rewrite_request(policy, locked.explanation, request)
	cap = prompt.max_output_tokens
	attempt.calls["rewrite"] = ProviderCall("interrupted", max_output_tokens=cap)
	try:
		generation = await asyncio.wait_for(call(prompt), policy.rewrite_timeout_seconds)
	except TimeoutError:
		attempt.calls["rewrite"] = ProviderCall("timeout", max_output_tokens=cap)
		attempt.failure = "rewrite_timeout"
		return None
	except Exception as error:
		attempt.calls["rewrite"] = ProviderCall(
			"error", max_output_tokens=cap, error=type(error).__name__
		)
		attempt.failure = "rewrite_error"
		return None
	attempt.calls["rewrite"] = _provider_call(generation, count, cap)
	text = generation.message.content.strip()
	if generation.finish_reason in _TRUNCATED:
		attempt.failure = "rewrite_truncated"
	elif not text:
		attempt.failure = "rewrite_empty"
	elif answer_format.marker is not None and answer_format.marker in text:
		# The server attaches the answer, so a marker here would be a second, unchecked answer.
		attempt.failure = "rewrite_answer_marker"
	elif copied_fraction(text, locked.explanation) >= policy.max_copied_fraction:
		attempt.failure = "rewrite_copies_draft"
	else:
		released = f"{text}{answer_format.separator}{answer_format.final(locked)}"
		parsed = answer_format.parse(released)
		if parsed is None or not parsed.explanation:
			attempt.failure = "task_invalid"
		elif parsed.value != locked.value:
			attempt.failure = "answer_mismatch"
		else:
			return released
	return None


async def apply_policy(
	attempt: OutputAttempt,
	request: TeacherGenerationRequest,
	draft: TeacherGeneration,
	call: Callable[[TeacherGenerationRequest], Awaitable[TeacherGeneration]],
	count: Callable[[str], int],
) -> str | None:
	"""The text to release, or None to withhold. `attempt` records what happened either way."""
	policy = attempt.policy
	text = draft.message.content
	attempt.draft_sha256 = _sha256(text)
	attempt.calls["draft"] = _provider_call(draft, count, request.max_output_tokens)
	if not attempt.protected:
		# Declared by unsupported_task and recorded as unprotected. Never described as protected.
		attempt.release(text, count(text), "out_of_scope_passthrough")
		return text
	# A truncated draft's last number may be cut short, so it is not an answer to lock.
	if draft.finish_reason in _TRUNCATED:
		attempt.withhold("draft_truncated")
		return None
	answer_format = ANSWER_FORMATS[attempt.domain]
	locked = answer_format.parse(text)
	if locked is None:
		attempt.withhold("draft_answer_missing")
		return None
	limit = request.max_output_tokens
	if policy.mode in REWRITE_MODES:
		if not locked.explanation:
			attempt.failure = "draft_no_explanation"
		else:
			rewritten = await _rewrite(attempt, locked, request, call, count)
			if rewritten is not None:
				tokens = count(rewritten)
				if tokens <= limit:
					attempt.release(rewritten, tokens, "rewritten")
					return rewritten
				attempt.failure = "released_over_limit"
		if policy.fallback == "withhold":
			attempt.withhold(attempt.failure or "rewrite_failed")
			return None
	answer = answer_format.final(locked)
	tokens = count(answer)
	if tokens > limit:
		attempt.withhold("released_over_limit")
		return None
	if policy.mode in REWRITE_MODES:
		attempt.fallback_used = "final_answer"
		attempt.release(answer, tokens, "fallback_final_answer")
	else:
		attempt.release(answer, tokens, "final_answer")
	return answer
