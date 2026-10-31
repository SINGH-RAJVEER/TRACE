"""Small task builds with topic references, shared by the workload and role tests."""

import json
import random
from pathlib import Path

from coding.dataset import PROMPT as CODING_PROMPT
from distillation.benchmarks.models import BenchmarkItem
from distillation.io import write_jsonl
from distillation.schemas import Message, PromptRecord, PromptSource
from distillation.tasks import (
	AG_NEWS_LABELS,
	AG_NEWS_PROMPT,
	GSM8K_EVAL_SUFFIX,
	LEGITIMATE_ROLES,
	SQUAD_PROMPT,
	task_directory,
)

CAPS4 = {"ag_news": 16, "squad": 64, "gsm8k": 256, "coding": 512}
VOCABULARY = [f"w{index}" for index in range(400)]


def words(*seed: object, count: int = 24) -> str:
	return " ".join(random.Random(":".join(map(str, seed))).choices(VOCABULARY, k=count))


def prompt_record(domain: str, role: str, index: int) -> tuple[PromptRecord, str]:
	"""One prompt in its task template, and its topic category."""
	if domain == "ag_news":
		category = AG_NEWS_LABELS[index % 4]
		user = AG_NEWS_PROMPT.format(text=f"{category} {words(domain, role, index)}")
		group = f"{role}-story-{index}"
	elif domain == "squad":
		category = f"{role}-title-{index // 10}"
		user = SQUAD_PROMPT.format(
			context=words(domain, role, index // 10, count=60),
			question=words(domain, role, index, count=8),
		)
		group = category
	elif domain == "gsm8k":
		category = "math"
		user = words(domain, role, index) + GSM8K_EVAL_SUFFIX
		group = f"{role}-problem-{index}"
	else:
		category = ("algorithmic", "generic")[index % 2]
		user = CODING_PROMPT.format(
			specification=words(domain, role, index), signature=f"def f{index}():"
		)
		group = f"{role}-spec-{index}"
	example_id = f"{domain}-train-{role}-{index}"
	return (
		PromptRecord(
			example_id=example_id,
			split="test",
			task=domain,
			source=PromptSource(
				corpus=f"fixture-{domain}",
				source_id=f"train:{role}-{index}",
				license="test",
				group_id=group,
			),
			prompt=[Message(role="user", content=user)],
		),
		category,
	)


def task_tree(
	root: Path, count: int, domains: tuple[str, ...] = tuple(CAPS4), roles: tuple[str, ...] = ()
) -> None:
	"""Write task builds with `count` prompts in each legitimate role and the attacker pool."""
	for domain in domains:
		directory = task_directory(domain, root=root)
		references = {}
		for role in roles or (*LEGITIMATE_ROLES, "attacker"):
			records = []
			for index in range(count):
				record, category = prompt_record(domain, role, index)
				if role == "attacker":
					records.append(record.model_copy(update={"split": "train"}))
					continue
				records.append(record)
				references[record.example_id] = BenchmarkItem(
					id=record.example_id,
					benchmark=domain,
					category=category,
					question="fixture",
					messages=record.prompt,
					metric="label_accuracy",
					answer="fixture",
				).model_dump(mode="json")
			path = (
				directory / "attacker.jsonl"
				if role == "attacker"
				else directory / "legitimate" / f"{role}.jsonl"
			)
			write_jsonl(path, records)
		(directory / "legitimate-references.json").write_text(json.dumps(references))
		(directory / "manifest.json").write_text(
			json.dumps({"task": domain, "max_output_tokens": CAPS4[domain]})
		)
