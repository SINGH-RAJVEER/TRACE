"""Concurrent, resumable collection of hosted-teacher sequences."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from distillation.io import append_jsonl, read_jsonl
from distillation.schemas import (
	CollectionAttempt,
	CollectionAttemptLog,
	LedgerRecord,
	PromptRecord,
	QualityRecord,
	TeacherGenerationRequest,
	TeacherMetadata,
)
from distillation.teacher import Teacher, TeacherUpstreamError


@dataclass(frozen=True)
class CollectionOptions:
	terms_review_id: str
	temperature: float = 0.0
	top_p: float | None = None
	top_k: int | None = None
	max_output_tokens: int = 1024
	seed: int | None = None
	enable_thinking: bool | None = None
	concurrency: int = 4
	max_attempts: int = 3
	include_test: bool = False

	def __post_init__(self) -> None:
		if not self.terms_review_id.strip():
			raise ValueError("terms_review_id is required")
		if self.concurrency < 1:
			raise ValueError("concurrency must be at least 1")
		if self.max_attempts < 1:
			raise ValueError("max_attempts must be at least 1")
		if self.top_p is not None and not 0 < self.top_p <= 1:
			raise ValueError("top_p must be greater than 0 and at most 1")
		if self.top_k is not None and self.top_k < 1:
			raise ValueError("top_k must be at least 1")


@dataclass(frozen=True)
class CollectionProgress:
	"""Counts after each finished prompt. done includes records resumed from the ledger."""

	done: int
	total: int
	collected: int
	rejected: int
	retries: int


async def collect(
	*,
	input_path: Path,
	output_path: Path,
	teacher: Teacher,
	options: CollectionOptions,
	progress: Callable[[CollectionProgress], None] | None = None,
) -> tuple[int, int]:
	"""Collect missing prompt records and append as workers finish."""

	try:
		prompts = read_jsonl(input_path, PromptRecord)
		example_ids = [prompt.example_id for prompt in prompts]
		if len(example_ids) != len(set(example_ids)):
			raise ValueError("input contains duplicate example_id values")
		existing_ids = _existing_example_ids(output_path, prompts, options)
	except Exception:
		await teacher.close()
		raise
	pending = [
		prompt
		for prompt in prompts
		if prompt.example_id not in existing_ids
		and (options.include_test or prompt.split != "test")
	]
	semaphore = asyncio.Semaphore(options.concurrency)
	write_lock = asyncio.Lock()
	attempt_log_path = _attempt_log_path(output_path)
	total = sum(options.include_test or prompt.split != "test" for prompt in prompts)
	counts = {"collected": 0, "rejected": 0, "retries": 0}

	def report() -> None:
		if progress:
			progress(
				CollectionProgress(
					done=total - len(pending) + counts["collected"],
					total=total,
					collected=counts["collected"],
					rejected=counts["rejected"],
					retries=counts["retries"],
				)
			)

	def retried() -> None:
		counts["retries"] += 1
		report()

	async def run(prompt: PromptRecord) -> LedgerRecord:
		async with semaphore:
			record = await _collect_one(prompt, teacher, options, attempt_log_path, retried)
		async with write_lock:
			append_jsonl(output_path, [record])
		counts["collected"] += 1
		counts["rejected"] += record.quality.status == "rejected"
		report()
		return record

	report()
	tasks = [asyncio.create_task(run(prompt)) for prompt in pending]
	try:
		results = await asyncio.gather(*tasks, return_exceptions=True)
	finally:
		await teacher.close()
	errors = [result for result in results if isinstance(result, Exception)]
	if errors:
		raise errors[0]
	collected = sum(isinstance(result, LedgerRecord) for result in results)
	return collected, len(prompts) - len(pending)


async def _collect_one(
	prompt: PromptRecord,
	teacher: Teacher,
	options: CollectionOptions,
	attempt_log_path: Path,
	retried: Callable[[], None],
) -> LedgerRecord:
	request = TeacherGenerationRequest(
		messages=prompt.prompt,
		temperature=options.temperature,
		top_p=options.top_p,
		top_k=options.top_k,
		max_output_tokens=options.max_output_tokens,
		seed=options.seed,
		enable_thinking=options.enable_thinking,
	)
	last_error: TeacherUpstreamError | None = None
	attempts: list[CollectionAttempt] = []
	successful_attempt_id = ""
	for attempt_number in range(1, options.max_attempts + 1):
		attempt_id = f"att_{uuid4().hex}"
		attempted_at = datetime.now(UTC)
		try:
			generation = await teacher.generate(request)
			attempts.append(
				CollectionAttempt(
					attempt_id=attempt_id,
					attempted_at=attempted_at,
					status="succeeded",
				)
			)
			append_jsonl(
				attempt_log_path,
				[CollectionAttemptLog(example_id=prompt.example_id, attempt=attempts[-1])],
			)
			successful_attempt_id = attempt_id
			break
		except TeacherUpstreamError as error:
			last_error = error
			attempts.append(
				CollectionAttempt(
					attempt_id=attempt_id,
					attempted_at=attempted_at,
					status="failed",
					error_code="teacher_upstream_error",
				)
			)
			append_jsonl(
				attempt_log_path,
				[CollectionAttemptLog(example_id=prompt.example_id, attempt=attempts[-1])],
			)
			if attempt_number == options.max_attempts:
				raise
			retried()
			await asyncio.sleep(2 ** (attempt_number - 1))
	else:
		raise RuntimeError("teacher retry loop exited unexpectedly") from last_error

	accepted_finish_reasons = {"stop", "end_turn", "completed"}
	accepted = generation.finish_reason in accepted_finish_reasons
	rejection_reason = None if accepted else f"finish_reason:{generation.finish_reason}"
	if options.enable_thinking is False and any(
		marker in generation.message.content for marker in ("<think>", "</think>")
	):
		accepted = False
		rejection_reason = "unexpected_thinking_trace"
	return LedgerRecord(
		example_id=prompt.example_id,
		attempt_id=successful_attempt_id,
		split=prompt.split,
		task=prompt.task,
		source=prompt.source,
		prompt=prompt.prompt,
		teacher=TeacherMetadata(
			provider=generation.provider,
			model=generation.model,
			endpoint=generation.endpoint,
			request_id=generation.request_id or generation.id,
			generated_at=generation.generated_at,
			parameters=_generation_parameters(options),
			terms_review_id=options.terms_review_id,
		),
		completion=[generation.message],
		finish_reason=generation.finish_reason,
		usage=generation.usage,
		quality=QualityRecord(
			status="pending" if accepted else "rejected",
			checks=["non_empty", "finish_reason"]
			+ (["non_thinking"] if options.enable_thinking is False else []),
			reviewer="collector-v1",
			rejection_reason=rejection_reason,
		),
		attempts=attempts,
	)


def _generation_parameters(
	options: CollectionOptions,
) -> dict[str, bool | int | float | str | None]:
	parameters: dict[str, bool | int | float | str | None] = {
		"temperature": options.temperature,
		"max_output_tokens": options.max_output_tokens,
		"seed": options.seed,
	}
	if options.enable_thinking is not None:
		parameters["enable_thinking"] = options.enable_thinking
	if options.top_p is not None:
		parameters["top_p"] = options.top_p
	if options.top_k is not None:
		parameters["top_k"] = options.top_k
	return parameters


def _existing_example_ids(
	path: Path,
	prompts: list[PromptRecord],
	options: CollectionOptions,
) -> set[str]:
	if not path.exists():
		return set()
	by_id = {prompt.example_id: prompt for prompt in prompts}
	seen: set[str] = set()
	for record in read_jsonl(path, LedgerRecord):
		if record.example_id in seen:
			raise ValueError(f"duplicate ledger example_id: {record.example_id}")
		seen.add(record.example_id)
		prompt = by_id.get(record.example_id)
		if prompt is None:
			continue
		if (
			any(
				getattr(prompt, key) != getattr(record, key)
				for key in ("prompt", "split", "task", "source")
			)
			or record.teacher.parameters != _generation_parameters(options)
			or record.teacher.terms_review_id != options.terms_review_id
		):
			raise ValueError(f"resume mismatch for {record.example_id}; use a new ledger")
	return seen


def _attempt_log_path(path: Path) -> Path:
	return path.with_name(f"{path.stem}.attempts{path.suffix}")
