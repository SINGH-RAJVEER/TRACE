"""Import and freeze publisher benchmark questions without teacher answers."""

import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from distillation.benchmarks.models import BenchmarkItem, DatasetSource, FrozenSuite, save_json
from distillation.schemas import Message

# TruthfulQA uses the authors' binary task added in 2025, rather than legacy MC1/MC2.
CATALOG = {
	"gsm8k": ("openai/gsm8k", "main", "test", "mit"),
	"arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", "cc-by-sa-4.0"),
	"mmlu": ("cais/mmlu", "all", "test", "mit"),
	"truthfulqa": ("truthfulqa/truthful_qa", "generation", "validation", "apache-2.0"),
	"humaneval": ("openai/openai_humaneval", "openai_humaneval", "test", "mit"),
	"mbpp": ("google-research-datasets/mbpp", "sanitized", "test", "cc-by-4.0"),
}
TRUTHFULQA_REVISION = "d71c110897f5d31c5d7f309e7bc316c152f6f031"


def _choice_item(
	benchmark: str,
	key: str,
	question: str,
	choices: list[str],
	answer: int,
	category: str,
) -> BenchmarkItem:
	labels = [chr(65 + i) for i in range(len(choices))]
	if not 0 <= answer < len(choices) or len(choices) > 26:
		raise ValueError("invalid multiple-choice reference")
	options = "\n".join(f"{label}. {text}" for label, text in zip(labels, choices, strict=True))
	return BenchmarkItem(
		id=f"{benchmark}:{key}",
		benchmark=benchmark,
		category=category,
		question=question,
		messages=[
			Message(
				role="user",
				content=f"{question}\n\n{options}\n\nReturn only the letter of the correct answer.",
			)
		],
		metric="choice_accuracy",
		answer=labels[answer],
		choices=choices,
	)


def convert_row(benchmark: str, index: int, row: dict[str, Any], seed: int) -> BenchmarkItem:
	if benchmark == "gsm8k":
		return BenchmarkItem(
			id=f"gsm8k:{index}",
			benchmark=benchmark,
			category="math",
			question=row["question"],
			messages=[
				Message(
					role="user",
					content=row["question"]
					+ "\nSolve step by step. End with #### followed by the final numeric answer.",
				)
			],
			metric="numeric_exact_match",
			answer=row["answer"].rsplit("####", 1)[1].strip(),
		)
	if benchmark == "arc_challenge":
		choices = row["choices"]
		return _choice_item(
			benchmark,
			row["id"],
			row["question"],
			choices["text"],
			choices["label"].index(row["answerKey"]),
			"science",
		)
	if benchmark == "mmlu":
		return _choice_item(
			benchmark,
			str(index),
			row["question"],
			row["choices"],
			int(row["answer"]),
			row["subject"],
		)
	if benchmark == "truthfulqa":
		# The Hub generation dataset lacks Best Incorrect Answer. The freezer below loads
		# the authors' pinned CSV with the two exact reference answers instead.
		choices = [row["Best Answer"], row["Best Incorrect Answer"]]
		order = [0, 1]
		random.Random(f"{seed}:truthfulqa:{index}").shuffle(order)
		return _choice_item(
			benchmark,
			str(index),
			row["Question"],
			[choices[i] for i in order],
			order.index(0),
			row["Category"],
		)
	if benchmark == "humaneval":
		prompt = row["prompt"]
		return BenchmarkItem(
			id=f"humaneval:{row['task_id']}",
			benchmark=benchmark,
			category="python",
			question=prompt,
			code_prompt=prompt,
			entry_point=row["entry_point"],
			messages=[
				Message(
					role="user",
					content="Complete this Python function. Return only the full Python code, "
					"including the function signature.\n\n" + prompt,
				)
			],
			metric="pass_at_1",
			answer="passes official tests",
			test_code=row["test"] + f"\ncheck({row['entry_point']})\n",
		)
	if benchmark == "mbpp":
		# Public assertion tests are part of the MBPP task prompt. Challenge tests stay hidden.
		visible = "\n".join(row["test_list"])
		return BenchmarkItem(
			id=f"mbpp:{row['task_id']}",
			benchmark=benchmark,
			category="python",
			question=row["prompt"],
			metric="pass_at_1",
			answer="passes official tests",
			messages=[
				Message(
					role="user",
					content="Write Python code for this task. Return only Python code.\n\n"
					f"{row['prompt']}\n\nYour code must satisfy:\n{visible}",
				)
			],
			test_code="\n".join(
				row.get("test_imports", []) + row["test_list"] + row.get("challenge_test_list", [])
			),
		)
	raise ValueError(f"unsupported benchmark: {benchmark}")


def _select(items: list[BenchmarkItem], limit: int, seed: int) -> list[BenchmarkItem]:
	"""Round-robin shuffled categories, so small MMLU runs cover subjects."""
	groups: dict[str, list[BenchmarkItem]] = defaultdict(list)
	for item in items:
		groups[item.category].append(item)
	rng = random.Random(seed)
	for group in groups.values():
		rng.shuffle(group)
	categories = sorted(groups)
	rng.shuffle(categories)
	selected: list[BenchmarkItem] = []
	while any(groups.values()) and (limit == 0 or len(selected) < limit):
		for category in categories:
			if groups[category] and (limit == 0 or len(selected) < limit):
				selected.append(groups[category].pop())
	return sorted(selected, key=lambda item: item.id)


def load_benchmark_rows(benchmark: str) -> tuple[list[dict[str, Any]], DatasetSource]:
	"""Download every held-out row of one benchmark at a resolved immutable revision."""
	if benchmark not in CATALOG:
		raise ValueError(f"unknown benchmark: {benchmark}")
	repository, configuration, split, license_name = CATALOG[benchmark]
	if benchmark == "truthfulqa":
		import csv
		import io

		import httpx

		repository = "sylinrl/TruthfulQA"
		revision = TRUTHFULQA_REVISION
		with httpx.Client(timeout=60) as client:
			response = client.get(
				f"https://raw.githubusercontent.com/{repository}/{revision}/TruthfulQA.csv"
			)
			response.raise_for_status()
		rows: list[dict[str, Any]] = list(csv.DictReader(io.StringIO(response.text)))
		configuration, split = "binary-2025", "all"
		url = f"https://github.com/{repository}/blob/{revision}/TruthfulQA.csv"
	else:
		try:
			from datasets import load_dataset
			from huggingface_hub import HfApi
		except ImportError as error:
			raise RuntimeError("run `uv sync --all-packages --extra eval`") from error
		resolved_revision = HfApi().dataset_info(repository).sha
		if not resolved_revision:
			raise ValueError(f"cannot resolve immutable revision for {repository}")
		revision = resolved_revision
		dataset = load_dataset(repository, configuration, split=split, revision=revision)
		rows = [dict(row) for row in dataset]
		url = f"https://huggingface.co/datasets/{repository}/tree/{revision}"
	source = DatasetSource(
		benchmark=benchmark,
		repository=repository,
		revision=revision,
		configuration=configuration,
		split=split,
		license=license_name,
		url=url,
		available=len(rows),
		selected=len(rows),
	)
	return rows, source


def freeze_suite(
	output: Path,
	benchmarks: list[str],
	limit: int = 200,
	seed: int = 42,
) -> FrozenSuite:
	if output.exists():
		raise ValueError("suite already exists; choose a new path to preserve the frozen questions")
	if limit < 0 or not benchmarks or len(set(benchmarks)) != len(benchmarks):
		raise ValueError("use unique benchmarks and a nonnegative limit")
	if unknown := set(benchmarks) - CATALOG.keys():
		raise ValueError(f"unknown benchmarks: {sorted(unknown)}")
	items: list[BenchmarkItem] = []
	sources: list[DatasetSource] = []
	for benchmark in benchmarks:
		rows, source = load_benchmark_rows(benchmark)
		converted = [convert_row(benchmark, i, row, seed) for i, row in enumerate(rows)]
		selected = _select(converted, limit, seed)
		items.extend(selected)
		sources.append(source.model_copy(update={"selected": len(selected)}))
	suite = FrozenSuite(seed=seed, created_at=datetime.now(UTC), sources=sources, items=items)
	suite.sha256 = suite.digest()
	suite.verify()
	save_json(output, suite)
	return suite
