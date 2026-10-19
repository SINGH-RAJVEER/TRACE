"""Evaluate every frozen item and keep failed attempts in the denominator."""

import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from distillation.benchmarks.inference import make_model, validate_adapter
from distillation.benchmarks.models import (
	EvaluationConfig,
	EvaluationRun,
	SampleResult,
	Slot,
	load_suite,
	save_json,
)
from distillation.benchmarks.scoring import evaluate_answer
from distillation.io import append_jsonl
from distillation.schemas import TeacherGenerationRequest
from distillation.teacher import Teacher
from distillation.training import _sha256


def _normalized(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip().casefold()


def audit_training(suite_path: Path, paths: list[Path]) -> dict[str, str]:
	"""Reject exact questions even when a training prompt appends answer instructions."""
	suite = load_suite(suite_path)
	questions = {_normalized(item.question) for item in suite.items}
	hashes: dict[str, str] = {}
	for path in dict.fromkeys(paths):
		hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
		# Split on newlines only; str.splitlines also breaks on separators inside JSON strings.
		for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
			if not line.strip():
				continue
			row = json.loads(line)
			messages = row.get("prompt", row.get("messages", []))
			if not messages:
				raise ValueError(
					f"{path}:{line_number}: cannot audit a row without prompt/messages"
				)
			for message in messages:
				content = _normalized(message["content"])
				if any(question in content for question in questions):
					raise ValueError(f"held-out benchmark question found in {path}:{line_number}")
	return hashes


async def run_evaluation(
	config: EvaluationConfig,
	slot: Slot,
	*,
	factory: Callable[..., Teacher] = make_model,
	progress: Callable[[int, int], None] | None = None,
) -> EvaluationRun:
	suite = load_suite(config.suite_path)
	coding_runtime: dict[str, str | int | float] = {}
	if any(item.metric == "pass_at_1" for item in suite.items):
		from coding.sandbox import runtime_protocol
		from coding.scoring import ensure_runtime

		await asyncio.to_thread(ensure_runtime)
		coding_runtime = runtime_protocol()
	model_config = config.models[slot]
	training_paths = config.training_paths + validate_adapter(model_config)
	training_hashes = audit_training(config.suite_path, training_paths)
	artifact_paths: list[Path] = []
	if model_config.adapter_path:
		artifact_paths.extend(p for p in model_config.adapter_path.iterdir() if p.is_file())
	if model_config.chat_template_path:
		artifact_paths.append(model_config.chat_template_path)
	artifact_hashes = {str(path): _sha256(path) for path in artifact_paths}
	if model_config.backend == "local" and not re.fullmatch(r"[0-9a-f]{40}", model_config.revision):
		raise ValueError("local evaluation requires an immutable 40-character model revision")
	if (
		model_config.backend == "local"
		and model_config.tokenizer_revision
		and not re.fullmatch(r"[0-9a-f]{40}", model_config.tokenizer_revision)
	):
		raise ValueError("local evaluation requires an immutable tokenizer revision")
	if model_config.adapter_path and slot != "student_after":
		raise ValueError("only student_after may load a distillation adapter")
	started_at = datetime.now(UTC)
	run_id = uuid.uuid4().hex
	model = factory(model_config)
	samples: list[SampleResult] = []
	try:
		for item in suite.items:
			start = time.perf_counter()
			try:
				generation = await model.generate(
					TeacherGenerationRequest(
						messages=item.messages,
						max_output_tokens=config.generation.max_output_tokens,
						temperature=0,
						seed=config.generation.seed,
						enable_thinking=False,
					)
				)
			except Exception:
				# No provider bodies or keys go into result artifacts.
				result = SampleResult(
					item_id=item.id,
					benchmark=item.benchmark,
					category=item.category,
					response="",
					prediction=None,
					correct=False,
					f1=0.0 if item.metric == "squad" else None,
					status="error",
					latency_seconds=time.perf_counter() - start,
				)
			else:
				latency = time.perf_counter() - start  # Excludes scoring.
				text = generation.message.content
				score = await evaluate_answer(item, text, generation.finish_reason)
				result = SampleResult(
					item_id=item.id,
					benchmark=item.benchmark,
					category=item.category,
					response=text,
					prediction=score.prediction,
					correct=score.correct,
					f1=score.f1,
					status=score.status,
					latency_seconds=latency,
					usage=generation.usage,
					finish_reason=generation.finish_reason,
					request_id=generation.request_id,
					served_model=generation.model,
					functional_trace=score.functional_trace,
				)
			samples.append(result)
			append_jsonl(config.output_dir / f"{run_id}.samples.jsonl", [result])
			if progress:
				progress(len(samples), len(suite.items))
	finally:
		await model.close()
	packages = {}
	for name in (
		"distillation-core",
		"coding-task",
		"numpy",
		"httpx",
		"torch",
		"transformers",
		"peft",
		"datasets",
	):
		with contextlib.suppress(importlib.metadata.PackageNotFoundError):
			packages[name] = importlib.metadata.version(name)
	try:
		code_revision = (
			await asyncio.to_thread(
				subprocess.check_output,
				["jj", "log", "-r", "@", "--no-graph", "-T", "commit_id"],
				text=True,
			)
		).strip()
	except OSError, subprocess.CalledProcessError:
		code_revision = "unknown"
	run = EvaluationRun(
		id=run_id,
		experiment=config.experiment,
		slot=slot,
		started_at=started_at,
		completed_at=datetime.now(UTC),
		suite_sha256=suite.sha256,
		generation=config.generation,
		dimensions=config.dimensions,
		model=model_config,
		training_sha256=training_hashes,
		artifact_sha256=artifact_hashes,
		packages=packages,
		platform=platform.platform(),
		code_revision=code_revision,
		coding_runtime=coding_runtime,
		samples=samples,
	)
	config.output_dir.mkdir(parents=True, exist_ok=True)
	suite_copy = config.output_dir / f"suite-{suite.sha256}.json"
	if not suite_copy.exists():
		save_json(suite_copy, suite)
	save_json(config.output_dir / f"{run_id}.json", run)
	return run
