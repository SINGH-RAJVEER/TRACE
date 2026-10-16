"""Explicitly simulated artifacts for checking the dashboard without a GPU."""

import random
from datetime import UTC, datetime
from pathlib import Path

from distillation.benchmarks.models import (
	SLOTS,
	BenchmarkItem,
	DatasetSource,
	EvaluationRun,
	FrozenSuite,
	GenerationConfig,
	ModelConfig,
	SampleResult,
	save_json,
)
from distillation.schemas import Message, TokenUsage
from distillation.tasks import AG_NEWS_LABELS, TASKS


def create_demo(directory: Path, *, task: str = "ag_news") -> None:
	if task not in TASKS:
		raise ValueError("task must be ag_news, squad, or gsm8k")
	spec = TASKS[task]
	if list(directory.glob("*.json")):
		raise ValueError("demo directory contains artifacts; choose an empty directory")
	now = datetime.now(UTC)
	items = []
	for i in range(24):
		question = f"Simulated {task} question {i + 1}. This is a dashboard fixture."
		answer = (
			AG_NEWS_LABELS[i % 4]
			if task == "ag_news"
			else "fixture span"
			if task == "squad"
			else "42"
		)
		items.append(
			BenchmarkItem(
				id=f"demo:{task}:{i}",
				benchmark=task,
				category=task,
				question=question,
				messages=[Message(role="user", content=question)],
				metric=spec.metric,
				answer=answer,
				choices=list(AG_NEWS_LABELS) if task == "ag_news" else [],
				answers=[answer] if task == "squad" else [],
			)
		)
	suite = FrozenSuite(
		seed=42,
		created_at=now,
		items=items,
		sources=[
			DatasetSource(
				benchmark=task,
				repository="synthetic/dashboard-fixture",
				revision="synthetic-v1",
				configuration="demo",
				split="demo",
				license="Apache-2.0",
				url="https://example.invalid/simulated",
				available=24,
				selected=24,
			)
		],
	)
	suite.sha256 = suite.digest()
	save_json(directory / f"suite-{suite.sha256}.json", suite)
	rng = random.Random(7)
	for slot in SLOTS:
		teacher = slot.startswith("teacher")
		after = slot.endswith("after")
		samples = []
		for index, item in enumerate(items):
			threshold = 19 if teacher else 16 if after else 11
			correct = (index * 7 + (1 if after else 0)) % 24 < threshold
			samples.append(
				SampleResult(
					item_id=item.id,
					benchmark=item.benchmark,
					category=item.category,
					response="Simulated passing answer" if correct else "Simulated failing answer",
					prediction=item.answer if correct else "wrong",
					f1=float(correct) if task == "squad" else None,
					correct=correct,
					status="ok",
					latency_seconds=rng.uniform(1.1, 3.8) if teacher else rng.uniform(0.3, 1.5),
					usage=TokenUsage(input_tokens=120, output_tokens=rng.randint(40, 200)),
					finish_reason="stop",
					served_model="simulated-teacher" if teacher else "simulated-student",
				)
			)
		run = EvaluationRun(
			id=f"demo-{slot}",
			experiment=f"Dashboard preview: {task}",
			slot=slot,
			started_at=now,
			completed_at=now,
			suite_sha256=suite.sha256,
			generation=GenerationConfig(max_output_tokens=spec.max_output_tokens),
			model=ModelConfig(
				backend="openai",
				model="simulated-teacher" if teacher else "simulated-student",
				revision="synthetic-v1",
				base_url="https://example.invalid",
			),
			demo=True,
			samples=samples,
		)
		save_json(directory / f"{run.id}.json", run)
