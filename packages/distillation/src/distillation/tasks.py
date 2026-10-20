"""Paper task corpora: AG News, SQuAD 1.1, and GSM8K under the paper's split protocol.

Each task gets its own directory. Seeded source-group hashes preserve 10% development
and 9% benign-test assignments. AG News and SQuAD reserve more training data for benign
calibration; GSM8K and coding retain their original allocations. The official evaluation
split is used only for the final test suite. Gold answers stay in gold-answers.jsonl and
the frozen suites; attacker-side files never include them.
"""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from coding import DEFAULT_POOL_SIZE, DEFAULT_QUALIFICATION_WORKERS
from pydantic import BaseModel, ConfigDict, Field

from distillation.benchmarks.models import BenchmarkItem, DatasetSource, FrozenSuite, save_json
from distillation.io import canonical_json, read_jsonl, write_jsonl
from distillation.schemas import LedgerRecord, Message, PromptRecord, PromptSource, SFTRecord


@dataclass(frozen=True)
class DatasetSpec:
	"""A public dataset pinned to an immutable Hub revision."""

	repository: str
	revision: str
	license: str
	configuration: str | None = None


def normalized(text: str) -> str:
	"""Matches the evaluator's audit normalization so build and eval checks agree."""
	return re.sub(r"\s+", " ", text).strip().casefold()


def _select(items: list[BenchmarkItem], limit: int, seed: int) -> list[BenchmarkItem]:
	"""Round-robin shuffled categories for the development suite."""
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


GSM8K_EVAL_SUFFIX = "\nSolve step by step. End with #### followed by the final numeric answer."


TaskName = Literal["ag_news", "squad", "gsm8k", "coding"]
Role = Literal[
	"development",
	"attacker",
	"fitting",
	"score_calibration",
	"policy_calibration",
	"benign_test",
]
LEGITIMATE_ROLES: tuple[Role, ...] = (
	"fitting",
	"score_calibration",
	"policy_calibration",
	"benign_test",
)
ROLES: tuple[Role, ...] = ("development", "attacker", *LEGITIMATE_ROLES)

AG_NEWS_LABELS = ("World", "Sports", "Business", "Sci/Tech")
AG_NEWS_PROMPT = (
	"Classify the news article into one of these topics: World, Sports, Business, Sci/Tech. "
	"Answer with only the topic name.\n\nArticle: {text}"
)
SQUAD_PROMPT = (
	"Answer the question with a short span copied from the passage. Return only the answer."
	"\n\nPassage: {context}\n\nQuestion: {question}"
)
# Applied identically before querying, training, and evaluation, and counted in the manifest.
SQUAD_MAX_CONTEXT_CHARS = 6000


@dataclass(frozen=True)
class TaskSpec:
	dataset: DatasetSpec
	pool_split: str
	test_split: str
	max_output_tokens: int
	metric: Literal["label_accuracy", "squad", "numeric_exact_match", "pass_at_1"]


TASKS: dict[str, TaskSpec] = {
	"ag_news": TaskSpec(
		DatasetSpec("fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4", "unknown"),
		"train",
		"test",
		16,
		"label_accuracy",
	),
	"squad": TaskSpec(
		DatasetSpec(
			"rajpurkar/squad",
			"7b6d24c440a36b6815f21b70d25016731768db1f",
			"cc-by-sa-4.0",
			"plain_text",
		),
		"train",
		"validation",
		64,
		"squad",
	),
	"gsm8k": TaskSpec(
		DatasetSpec("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "mit", "main"),
		"train",
		"test",
		256,
		"numeric_exact_match",
	),
	"coding": TaskSpec(
		DatasetSpec(
			"nvidia/OpenCodeInstruct",
			"8f3ba5bafe4d6e8db46082cf7ae6741bc370604d",
			"cc-by-4.0",
			"train",
		),
		"train",
		"test",
		512,
		"pass_at_1",
	),
}


@dataclass(frozen=True)
class TaskExample:
	"""One source row in the task's fixed serialization."""

	key: str
	group: str
	user: str
	answer: str
	category: str
	question: str
	answers: list[str] = field(default_factory=list)
	gold_completion: str = ""
	truncated: bool = False


class GoldRecord(BaseModel):
	"""Gold completion for one attacker-pool prompt, for the gold-label reference student."""

	model_config = ConfigDict(extra="forbid")

	example_id: str = Field(min_length=1)
	completion: str = Field(min_length=1)


def _digest(*parts: object) -> int:
	return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16)


# Absolute training-source shares, in ROLES order. Boundaries at 10% and 91% retain
# the old development and final benign-test groups. Attacker pools are nested subsets.
TASK_SPLIT_VERSION = "task-source-split-v2"
ROLE_SHARES: dict[str, tuple[float, ...]] = {
	"ag_news": (0.10, 0.30, 0.05, 0.12, 0.34, 0.09),
	"squad": (0.10, 0.10, 0.06, 0.15, 0.50, 0.09),
	"gsm8k": (0.10, 0.45, 0.18, 0.09, 0.09, 0.09),
	"coding": (0.10, 0.45, 0.18, 0.09, 0.09, 0.09),
}


def _role(seed: int, task: str, group: str) -> Role:
	"""Assign each complete source group once under the task's frozen role shares."""
	value = _digest(seed, "task-split", task, group) % 1_000_000
	if task in {"gsm8k", "coding"}:
		# Preserve legacy floating-point boundary assignments in existing frozen builds.
		u = value / 1_000_000
		if u < 0.10:
			return "development"
		v = (u - 0.10) / 0.90
		if v < 0.5:
			return "attacker"
		w = (v - 0.5) / 0.5
		if w < 0.4:
			return "fitting"
		if w < 0.6:
			return "score_calibration"
		if w < 0.8:
			return "policy_calibration"
		return "benign_test"
	boundary = 0
	for role, share in zip(ROLES, ROLE_SHARES[task], strict=True):
		boundary += round(share * 1_000_000)
		if value < boundary:
			return role
	raise AssertionError("role shares must cover the hash range")


def _truncate(text: str, limit: int) -> tuple[str, bool]:
	if len(text) <= limit:
		return text, False
	cut = text[:limit]
	space = cut.rfind(" ")
	return (cut[:space] if space > limit // 2 else cut).rstrip(), True


def ag_news_example(index: int, row: dict[str, Any]) -> TaskExample:
	text = row["text"].strip()
	label = AG_NEWS_LABELS[int(row["label"])]
	return TaskExample(
		key=str(index),
		group=hashlib.sha256(normalized(text).encode()).hexdigest()[:24],
		user=AG_NEWS_PROMPT.format(text=text),
		answer=label,
		category=label,
		question=text,
		gold_completion=label,
	)


def squad_example(index: int, row: dict[str, Any]) -> TaskExample:
	context, truncated = _truncate(row["context"].strip(), SQUAD_MAX_CONTEXT_CHARS)
	answers = [answer.strip() for answer in row["answers"]["text"] if answer.strip()]
	user = SQUAD_PROMPT.format(context=context, question=row["question"].strip())
	return TaskExample(
		key=str(row["id"]),
		group=row["title"],
		user=user,
		answer=answers[0] if answers else "",
		category=row["title"],
		# The whole serialized request, so the training audit matches only exact reuse.
		question=user,
		answers=answers,
		gold_completion=answers[0] if answers else "",
		truncated=truncated,
	)


def gsm8k_example(index: int, row: dict[str, Any]) -> TaskExample:
	rationale, number = row["answer"].rsplit("####", 1)
	number = number.strip()
	# Drop the dataset's calculator annotations, which the prompt never asks for.
	rationale = re.sub(r"<<[^>]*>>", "", rationale).strip()
	question = row["question"]
	return TaskExample(
		key=str(index),
		group=hashlib.sha256(normalized(question).encode()).hexdigest()[:24],
		user=question + GSM8K_EVAL_SUFFIX,
		answer=number,
		category="math",
		question=question,
		gold_completion=f"{rationale}\n#### {number}",
	)


CONVERTERS: dict[str, Callable[[int, dict[str, Any]], TaskExample]] = {
	"ag_news": ag_news_example,
	"squad": squad_example,
	"gsm8k": gsm8k_example,
}


def _load_rows(spec: DatasetSpec, split: str) -> list[dict[str, Any]]:
	from datasets import load_dataset

	dataset = load_dataset(spec.repository, spec.configuration, split=split, revision=spec.revision)
	return [dict(row) for row in dataset]


def _benchmark_item(task: str, split: str, example: TaskExample) -> BenchmarkItem:
	metric = TASKS[task].metric
	return BenchmarkItem(
		id=f"{task}:{split}:{example.key}",
		benchmark=task,
		category=example.category,
		question=example.question,
		messages=[Message(role="user", content=example.user)],
		metric=metric,
		answer=example.answer,
		choices=list(AG_NEWS_LABELS) if task == "ag_news" else [],
		answers=example.answers,
	)


def _suite(
	task: str,
	split: str,
	items: list[BenchmarkItem],
	available: int,
	seed: int,
	*,
	dataset: DatasetSpec | None = None,
) -> FrozenSuite:
	spec = dataset or TASKS[task].dataset
	source = DatasetSource(
		benchmark=task,
		repository=spec.repository,
		revision=spec.revision,
		configuration=spec.configuration or "default",
		split=split,
		license=spec.license,
		url=f"https://huggingface.co/datasets/{spec.repository}/tree/{spec.revision}",
		available=available,
		selected=len(items),
	)
	suite = FrozenSuite(seed=seed, created_at=datetime.now(UTC), sources=[source], items=items)
	suite.sha256 = suite.digest()
	suite.verify()
	return suite


def task_directory(
	task: str,
	*,
	coding_pool_size: int = DEFAULT_POOL_SIZE,
	seed: int = 42,
	root: Path = Path("data/tasks"),
) -> Path:
	name = (
		f"coding-subset-{coding_pool_size}-s{seed}"
		if task == "coding" and coding_pool_size
		else task
	)
	if task in {"ag_news", "squad"}:
		name = f"{task}-split-v2-s{seed}"
	return root / name


def build_task(
	*,
	task: str,
	output_dir: Path,
	seed: int = 42,
	development_limit: int = 1000,
	rows: dict[str, list[dict[str, Any]]] | None = None,
	log: Callable[[str], None] | None = None,
	qualification_workers: int = DEFAULT_QUALIFICATION_WORKERS,
	coding_pool_size: int = DEFAULT_POOL_SIZE,
) -> dict[str, Any]:
	"""Write the task's prompt pools, gold answers, frozen suites, and manifest."""
	if task == "coding":
		from distillation.coding_task import build_coding_task

		return build_coding_task(
			output_dir=output_dir,
			seed=seed,
			development_limit=development_limit,
			rows=rows,
			log=log,
			qualification_workers=qualification_workers,
			coding_pool_size=coding_pool_size,
		)
	if task not in TASKS:
		raise ValueError(f"task must be one of {sorted(TASKS)}")
	if (output_dir / "manifest.json").exists():
		raise ValueError(f"{output_dir} already holds a task build; choose a new directory")
	spec = TASKS[task]
	convert = CONVERTERS[task]
	if rows is None:
		rows = {}
		for split in (spec.pool_split, spec.test_split):
			if log:
				log(f"downloading {spec.dataset.repository} {split}")
			rows[split] = _load_rows(spec.dataset, split)

	tests = [convert(index, row) for index, row in enumerate(rows[spec.test_split])]
	test_prompts = {normalized(example.question) for example in tests}
	test_groups = {example.group for example in tests}
	filtered: Counter[str] = Counter()
	seen: set[str] = set()
	by_role: dict[Role, list[TaskExample]] = {}
	truncated = Counter({spec.test_split: sum(example.truncated for example in tests)})
	for index, row in enumerate(rows[spec.pool_split]):
		example = convert(index, row)
		fingerprint = normalized(example.user)
		if not example.answer:
			filtered["no_answer"] += 1
		elif example.group in test_groups or normalized(example.question) in test_prompts:
			filtered["test_overlap"] += 1
		elif fingerprint in seen:
			filtered["duplicate_prompt"] += 1
		else:
			seen.add(fingerprint)
			truncated[spec.pool_split] += example.truncated
			by_role.setdefault(_role(seed, task, example.group), []).append(example)

	corpus = spec.dataset.repository

	def record(example: TaskExample, split: Literal["train", "test"]) -> PromptRecord:
		messages = [Message(role="user", content=example.user)]
		payload = [message.model_dump(mode="json") for message in messages]
		return PromptRecord(
			example_id=f"{task}-{spec.pool_split}-{example.key}",
			split=split,
			task=task,
			source=PromptSource(
				corpus=corpus,
				source_id=f"{spec.pool_split}:{example.key}",
				license=spec.dataset.license,
				group_id=example.group,
				content_sha256=hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
			),
			prompt=messages,
		)

	output_dir.mkdir(parents=True, exist_ok=True)
	attacker = by_role.get("attacker", [])
	write_jsonl(output_dir / "attacker.jsonl", [record(example, "train") for example in attacker])
	write_jsonl(
		output_dir / "gold-answers.jsonl",
		[
			GoldRecord(
				example_id=f"{task}-{spec.pool_split}-{example.key}",
				completion=example.gold_completion,
			)
			for example in attacker
		],
	)
	# Legitimate pools are marked test so the collector skips them unless told otherwise.
	for role in LEGITIMATE_ROLES:
		write_jsonl(
			output_dir / "legitimate" / f"{role}.jsonl",
			[record(example, "test") for example in by_role.get(role, [])],
		)
	development = [
		_benchmark_item(task, spec.pool_split, example)
		for example in by_role.get("development", [])
	]
	save_json(
		output_dir / "legitimate-references.json",
		{
			f"{task}-{spec.pool_split}-{example.key}": _benchmark_item(
				task, spec.pool_split, example
			).model_dump(mode="json")
			for role in LEGITIMATE_ROLES
			for example in by_role.get(role, [])
		},
	)
	development_suite = _suite(
		task,
		spec.pool_split,
		_select(development, development_limit, seed),
		len(development),
		seed,
	)
	test_suite = _suite(
		task,
		spec.test_split,
		sorted(
			(_benchmark_item(task, spec.test_split, example) for example in tests),
			key=lambda item: item.id,
		),
		len(tests),
		seed,
	)
	save_json(output_dir / "development-suite.json", development_suite)
	save_json(output_dir / "test-suite.json", test_suite)

	files = sorted(path for path in output_dir.rglob("*") if path.is_file())
	manifest: dict[str, Any] = {
		"task": task,
		"source_split": {
			"version": TASK_SPLIT_VERSION,
			"shares": dict(zip(ROLES, ROLE_SHARES[task], strict=True)),
		},
		"seed": seed,
		"max_output_tokens": spec.max_output_tokens,
		"metric": spec.metric,
		"dataset": {
			"repository": spec.dataset.repository,
			"configuration": spec.dataset.configuration,
			"revision": spec.dataset.revision,
			"license": spec.dataset.license,
			"pool_split": spec.pool_split,
			"test_split": spec.test_split,
		},
		"available": {split: len(values) for split, values in rows.items()},
		"filtered": dict(filtered),
		"roles": {role: len(by_role.get(role, [])) for role in ROLES},
		"groups": {
			role: len({example.group for example in by_role.get(role, [])}) for role in ROLES
		},
		"development_suite_items": len(development_suite.items),
		"test_suite_items": len(test_suite.items),
		"truncated": dict(truncated),
		"sha256": {
			str(path.relative_to(output_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
			for path in files
		},
	}
	(output_dir / "manifest.json").write_text(
		json.dumps(manifest, indent="\t", sort_keys=True) + "\n", encoding="utf-8"
	)
	return manifest


def budget_run(
	*,
	task_dir: Path,
	budget: int,
	seed: int,
	output_dir: Path,
	ledger_path: Path | None = None,
) -> dict[str, Any]:
	"""Take the first `budget` attacker prompts in a seeded order, so budgets nest per seed.

	Writes prompts.jsonl and the matching gold-train.jsonl. With a ledger, also writes the
	subset of its records for these prompts, in prompt order, counts prompts it lacks, and
	refuses records collected with a different output limit from the task's.
	"""
	pool = read_jsonl(task_dir / "attacker.jsonl", PromptRecord)
	if not 1 <= budget <= len(pool):
		raise ValueError(f"budget must be between 1 and the attacker pool size, {len(pool)}")
	ordered = sorted(pool, key=lambda prompt: _digest(seed, "order", prompt.example_id))
	selected = ordered[:budget]
	gold = {
		record.example_id: record
		for record in read_jsonl(task_dir / "gold-answers.jsonl", GoldRecord)
	}
	write_jsonl(output_dir / "prompts.jsonl", selected)
	write_jsonl(
		output_dir / "gold-train.jsonl",
		[
			SFTRecord(
				prompt=prompt.prompt,
				completion=[Message(role="assistant", content=gold[prompt.example_id].completion)],
			)
			for prompt in selected
		],
	)
	summary: dict[str, Any] = {
		"task_dir": str(task_dir),
		"budget": budget,
		"seed": seed,
		"pool": len(pool),
	}
	if ledger_path:
		records = {record.example_id: record for record in read_jsonl(ledger_path, LedgerRecord)}
		subset: list[LedgerRecord] = [
			records[prompt.example_id] for prompt in selected if prompt.example_id in records
		]
		limit = json.loads((task_dir / "manifest.json").read_text())["max_output_tokens"]
		for record in subset:
			if record.teacher.parameters.get("max_output_tokens") != limit:
				raise ValueError(
					f"{record.example_id} was collected with a different max_output_tokens; "
					f"collect this task with --max-output-tokens {limit}"
				)
		write_jsonl(output_dir / "ledger.jsonl", subset)
		summary["ledger"] = str(ledger_path)
		summary["collected"] = len(subset)
		summary["missing"] = budget - len(subset)
		# The collector rejects nonterminal finishes, such as hitting the output limit, and
		# preparation drops them, so these responses were received but cannot train.
		summary["rejected_by_collector"] = sum(r.quality.status == "rejected" for r in subset)
		summary["finish_reasons"] = dict(Counter(str(r.finish_reason) for r in subset))
	summary["sha256"] = {
		path.name: hashlib.sha256(path.read_bytes()).hexdigest()
		for path in sorted(output_dir.glob("*.jsonl"))
	}
	(output_dir / "budget.json").write_text(
		json.dumps(summary, indent="\t", sort_keys=True) + "\n", encoding="utf-8"
	)
	return summary
