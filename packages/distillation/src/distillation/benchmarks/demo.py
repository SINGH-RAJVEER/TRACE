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
from distillation.benchmarks.suites import CATALOG
from distillation.schemas import Message, TokenUsage


def create_demo(directory: Path) -> None:
	if list(directory.glob("*.json")):
		raise ValueError("demo directory contains artifacts; choose an empty directory")
	now = datetime.now(UTC)
	items = []
	for benchmark in CATALOG:
		for i in range(24):
			coding = benchmark in {"humaneval", "mbpp"}
			question = (
				f"Simulated {benchmark} question {i + 1}. "
				"This is a dashboard fixture, not a public benchmark question."
			)
			items.append(
				BenchmarkItem(
					id=f"demo:{benchmark}:{i}",
					benchmark=benchmark,
					category=["algebra", "history", "computer_science"][i % 3]
					if benchmark == "mmlu"
					else benchmark,
					question=question,
					messages=[Message(role="user", content=question)],
					metric="pass_at_1" if coding else "choice_accuracy",
					answer="passes tests" if coding else "A",
					choices=[]
					if coding
					else ["Correct fixture answer", "Incorrect fixture answer"],
				)
			)
	suite = FrozenSuite(
		seed=42,
		created_at=now,
		items=items,
		sources=[
			DatasetSource(
				benchmark=b,
				repository="synthetic/dashboard-fixture",
				revision="synthetic-v1",
				configuration="demo",
				split="demo",
				license="Apache-2.0",
				url="https://example.invalid/simulated",
				available=24,
				selected=24,
			)
			for b in CATALOG
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
					prediction=("passed" if item.metric == "pass_at_1" else "A")
					if correct
					else "B",
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
			experiment="Dashboard preview",
			slot=slot,
			started_at=now,
			completed_at=now,
			suite_sha256=suite.sha256,
			generation=GenerationConfig(),
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
