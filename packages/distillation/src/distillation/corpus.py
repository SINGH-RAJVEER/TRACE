"""Build the transfer prompt corpus from public datasets and project-authored prompts.

The builder never includes a benchmark's held-out split, drops any prompt that overlaps a
held-out benchmark question, assigns train and validation splits by source group before any
teacher call, and writes an answer key beside the prompts for automatic review.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from distillation.io import canonical_json, write_jsonl
from distillation.schemas import Message, PromptRecord, PromptSource, ReferenceRecord
from distillation.synthetic import (
	GSM8K_SUFFIX,
	MBPP_PREFIX,
	Generated,
	constrained_writing,
	json_extraction,
	python_functions,
	text_operations,
	word_problems,
)

CHOICE_SUFFIX = "Return only the letter of the correct answer."
REASONING_SUFFIX = (
	"Reason through the problem briefly, then end with a final line of the form "
	'"Answer: X", where X is the letter of the correct choice.'
)
AUTHORED_CORPUS = "project-authored-v1"
GENERATED_CORPUS = "project-generated-v1"
PROJECT_LICENSE = "Apache-2.0"


@dataclass(frozen=True)
class DatasetSpec:
	"""A public dataset pinned to an immutable Hub revision."""

	repository: str
	revision: str
	license: str
	configuration: str | None = None


DATASETS = {
	"gsm8k": DatasetSpec("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "mit", "main"),
	"arc": DatasetSpec(
		"allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "cc-by-sa-4.0"
	),
	"mbpp": DatasetSpec(
		"google-research-datasets/mbpp",
		"4bb6404fdc6cacfda99d4ac4205087b89d32030c",
		"cc-by-4.0",
		"full",
	),
	"aqua_rat": DatasetSpec(
		"deepmind/aqua_rat", "33301c6a050c96af81f63cad5562cb5363e88971", "apache-2.0", "raw"
	),
	"commonsense_qa": DatasetSpec(
		"tau/commonsense_qa", "94630fe30dad47192a8546eb75f094926d47e155", "mit"
	),
	"self_oss_instruct": DatasetSpec(
		"bigcode/self-oss-instruct-sc2-exec-filter-50k",
		"356bb069eee815daa6e23e9a282eeefe1490ad44",
		"odc-by",
	),
	"dolly": DatasetSpec(
		"databricks/databricks-dolly-15k",
		"bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
		"cc-by-sa-3.0",
	),
	"oasst2": DatasetSpec(
		"OpenAssistant/oasst2", "179dd21fc55192153d94adb0e0ce8f69e222bf75", "apache-2.0"
	),
}

# Per-family caps after filtering. None takes every eligible prompt.
PROFILES: dict[str, dict[str, int | None]] = {
	"quick": {
		"gsm8k": 1500,
		"aqua_rat": 500,
		"arc_challenge": 400,
		"arc_easy": 400,
		"commonsense_qa": 500,
		"mbpp": None,
		"self_oss_instruct": 500,
		"dolly": 1000,
		"oasst2": 800,
		"authored": None,
		"python_functions": None,
		"word_problems": 300,
		"json_extraction": 150,
		"text_operations": 150,
		"constrained_writing": 200,
	},
	"standard": {
		"gsm8k": None,
		"aqua_rat": 3000,
		"arc_challenge": None,
		"arc_easy": None,
		"commonsense_qa": 3000,
		"mbpp": None,
		"self_oss_instruct": 4000,
		"dolly": 6000,
		"oasst2": 5000,
		"authored": None,
		"python_functions": None,
		"word_problems": 1200,
		"json_extraction": 500,
		"text_operations": 500,
		"constrained_writing": None,
	},
	"full": {
		"gsm8k": None,
		"aqua_rat": 12000,
		"arc_challenge": None,
		"arc_easy": None,
		"commonsense_qa": None,
		"mbpp": None,
		"self_oss_instruct": 15000,
		"dolly": None,
		"oasst2": None,
		"authored": None,
		"python_functions": None,
		"word_problems": 3000,
		"json_extraction": 1200,
		"text_operations": 1200,
		"constrained_writing": None,
	},
}


@dataclass(frozen=True)
class Candidate:
	family: str
	example_id: str
	corpus: str
	source_id: str
	group_id: str
	license: str
	task: str
	messages: list[Message]
	reference: dict[str, Any]


def _digest(*parts: object) -> int:
	return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def normalized(text: str) -> str:
	"""Matches the evaluator's audit normalization so build and eval checks agree."""
	return re.sub(r"\s+", " ", text).strip().casefold()


class ContaminationIndex:
	"""Finds prompts sharing a 13-word span with, or containing, a held-out question."""

	def __init__(self, questions: Iterable[str], ngram: int = 13, minimum_words: int = 4) -> None:
		self.ngram = ngram
		self.spans: set[tuple[str, ...]] = set()
		self.short: dict[int, set[tuple[str, ...]]] = defaultdict(set)
		self.size = 0
		for question in questions:
			words = normalized(question).split()
			if len(words) >= ngram:
				self.spans.update(
					tuple(words[i : i + ngram]) for i in range(len(words) - ngram + 1)
				)
			elif len(words) >= minimum_words:
				self.short[len(words)].add(tuple(words))
			self.size += 1

	def contains(self, text: str) -> bool:
		words = normalized(text).split()
		for i in range(len(words) - self.ngram + 1):
			if tuple(words[i : i + self.ngram]) in self.spans:
				return True
		for length, questions in self.short.items():
			for i in range(len(words) - length + 1):
				if tuple(words[i : i + length]) in questions:
					return True
		return False


def benchmark_index() -> tuple[ContaminationIndex, list[dict[str, Any]], set[str]]:
	"""Index every held-out benchmark question, not only the sampled frozen suite."""
	from distillation.benchmarks.suites import CATALOG, convert_row, load_benchmark_rows

	questions: list[str] = []
	sources: list[dict[str, Any]] = []
	mbpp_test_ids: set[str] = set()
	for benchmark in CATALOG:
		rows, source = load_benchmark_rows(benchmark)
		for index, row in enumerate(rows):
			questions.append(convert_row(benchmark, index, row, 0).question)
			if benchmark == "mbpp":
				mbpp_test_ids.add(str(row["task_id"]))
		sources.append(source.model_dump(mode="json", exclude={"selected"}))
	return ContaminationIndex(questions), sources, mbpp_test_ids


def _load(key: str, split: str) -> list[dict[str, Any]]:
	from datasets import load_dataset

	spec = DATASETS[key]
	return [
		dict(row)
		for row in load_dataset(
			spec.repository, spec.configuration, split=split, revision=spec.revision
		)
	]


def _choice_prompt(question: str, choices: list[str], reasoning: bool) -> str:
	options = "\n".join(f"{chr(65 + i)}. {text}" for i, text in enumerate(choices))
	return f"{question.strip()}\n\n{options}\n\n{REASONING_SUFFIX if reasoning else CHOICE_SUFFIX}"


def _choice_candidate(
	family: str,
	key: str,
	identifier: str,
	split: str,
	question: str,
	choices: list[str],
	answer: int,
	task: str,
	reasoning_share: float,
	seed: int,
) -> Candidate:
	spec = DATASETS[key]
	reasoning = _digest(seed, "variant", family, identifier) % 1000 < reasoning_share * 1000
	return Candidate(
		family=family,
		example_id=f"{family}-{split}-{identifier}",
		corpus=spec.repository,
		source_id=f"{split}:{identifier}",
		group_id=f"{split}:{identifier}",
		license=spec.license,
		task=f"{task}-{'reasoning' if reasoning else 'letter'}",
		messages=[Message(role="user", content=_choice_prompt(question, choices, reasoning))],
		reference={
			"kind": "choice_reasoning" if reasoning else "choice",
			"answer": chr(65 + answer),
		},
	)


def gsm8k(seed: int) -> Iterator[Candidate]:
	spec = DATASETS["gsm8k"]
	for index, row in enumerate(_load("gsm8k", "train")):
		answer = row["answer"].rsplit("####", 1)[1].strip().replace(",", "")
		yield Candidate(
			family="gsm8k",
			example_id=f"gsm8k-train-{index:05d}",
			corpus=spec.repository,
			source_id=f"train:{index}",
			group_id=f"train:{index}",
			license=spec.license,
			task="math-word-problem",
			messages=[Message(role="user", content=row["question"] + GSM8K_SUFFIX)],
			reference={"kind": "numeric", "answer": answer},
		)


def arc(level: str, seed: int) -> Iterator[Candidate]:
	family = f"arc_{level.lower()}"
	for split in ("train", "validation"):
		for row in _load_arc(level, split):
			labels = row["choices"]["label"]
			if row["answerKey"] not in labels:
				continue
			yield _choice_candidate(
				family,
				"arc",
				row["id"],
				split,
				row["question"],
				row["choices"]["text"],
				labels.index(row["answerKey"]),
				"science-choice",
				0.5,
				seed,
			)


def _load_arc(level: str, split: str) -> list[dict[str, Any]]:
	from datasets import load_dataset

	spec = DATASETS["arc"]
	dataset = load_dataset(spec.repository, f"ARC-{level}", split=split, revision=spec.revision)
	return [dict(row) for row in dataset]


def commonsense_qa(seed: int) -> Iterator[Candidate]:
	for split in ("train", "validation"):
		for row in _load("commonsense_qa", split):
			labels = row["choices"]["label"]
			if row["answerKey"] not in labels:
				continue
			yield _choice_candidate(
				"commonsense_qa",
				"commonsense_qa",
				row["id"],
				split,
				row["question"],
				row["choices"]["text"],
				labels.index(row["answerKey"]),
				"commonsense-choice",
				0.5,
				seed,
			)


def aqua_rat(seed: int) -> Iterator[Candidate]:
	for split in ("train", "validation"):
		for index, row in enumerate(_load("aqua_rat", split)):
			parsed = [
				re.match(r"^\s*([A-E])\s*\)\s*(.*\S)\s*$", option) for option in row["options"]
			]
			if not parsed or any(match is None for match in parsed):
				continue
			letters = [match[1] for match in parsed if match]
			if letters != [chr(65 + i) for i in range(len(letters))]:
				continue
			if row["correct"] not in letters:
				continue
			yield _choice_candidate(
				"aqua_rat",
				"aqua_rat",
				str(index),
				split,
				row["question"],
				[match[2] for match in parsed if match],
				letters.index(row["correct"]),
				"math-choice",
				0.85,
				seed,
			)


def mbpp(excluded_ids: set[str]) -> Iterator[Candidate]:
	spec = DATASETS["mbpp"]
	for split in ("train", "validation", "prompt"):
		for row in _load("mbpp", split):
			task_id = str(row["task_id"])
			if task_id in excluded_ids:
				raise ValueError(f"MBPP task {task_id} is in the held-out sanitized test split")
			entry = re.match(
				r"\s*assert\s+(?:not\s+)?\(?\s*(?:set\()?([A-Za-z_]\w*)\s*\(", row["test_list"][0]
			)
			visible = "\n".join(row["test_list"])
			tests = "\n".join(
				part
				for part in [row["test_setup_code"], *row["test_list"], *row["challenge_test_list"]]
				if part.strip()
			)
			yield Candidate(
				family="mbpp",
				example_id=f"mbpp-{task_id}",
				corpus=spec.repository,
				source_id=f"{split}:{task_id}",
				group_id=f"task:{task_id}",
				license=spec.license,
				task="python-function",
				messages=[
					Message(
						role="user",
						content=f"{MBPP_PREFIX}{row['text']}\n\nYour code must satisfy:\n{visible}",
					)
				],
				reference={
					"kind": "python_tests",
					"tests": tests,
					"entry_point": entry[1] if entry else "",
				},
			)


def self_oss_instruct() -> Iterator[Candidate]:
	spec = DATASETS["self_oss_instruct"]
	for row in _load("self_oss_instruct", "train"):
		identifier = str(row["id"])
		yield Candidate(
			family="self_oss_instruct",
			example_id=f"oss-{identifier}",
			corpus=spec.repository,
			source_id=f"train:{identifier}",
			group_id=f"train:{identifier}",
			license=spec.license,
			task="python-instruction",
			messages=[Message(role="user", content=row["instruction"].strip())],
			reference={"kind": "python_code"},
		)


def dolly() -> Iterator[Candidate]:
	spec = DATASETS["dolly"]
	for index, row in enumerate(_load("dolly", "train")):
		instruction, context = row["instruction"].strip(), (row["context"] or "").strip()
		if not instruction:
			continue
		content = f"{instruction}\n\nReference text:\n{context}" if context else instruction
		yield Candidate(
			family="dolly",
			example_id=f"dolly-{index:05d}",
			corpus=spec.repository,
			source_id=f"train:{index}",
			group_id=f"train:{index}",
			license=spec.license,
			task=f"dolly-{row['category']}",
			messages=[Message(role="user", content=content)],
			reference={"kind": "general"},
		)


BLOCKED_LABELS = {"spam", "pii", "not_appropriate", "hate_speech", "sexual_content", "toxicity"}


def _flagged(row: dict[str, Any]) -> bool:
	labels = row.get("labels") or {}
	for name, value in zip(labels.get("name", []), labels.get("value", []), strict=False):
		if name in BLOCKED_LABELS and value is not None and value >= 0.5:
			return True
	detoxify = row.get("detoxify") or {}
	return bool((detoxify.get("toxicity") or 0) >= 0.2)


def oasst2() -> Iterator[Candidate]:
	"""English opening prompts of reviewed conversation trees, one per tree."""
	spec = DATASETS["oasst2"]
	for split in ("train", "validation"):
		for row in _load("oasst2", split):
			if (
				row["parent_id"] is not None
				or row["role"] != "prompter"
				or row["lang"] != "en"
				or row["deleted"]
				or row["synthetic"]
				or row["review_result"] is not True
				or _flagged(row)
			):
				continue
			yield Candidate(
				family="oasst2",
				example_id=f"oasst2-{row['message_id']}",
				corpus=spec.repository,
				source_id=f"{split}:{row['message_id']}",
				group_id=f"tree:{row['message_tree_id']}",
				license=spec.license,
				task="open-conversation",
				messages=[Message(role="user", content=row["text"].strip())],
				reference={"kind": "general"},
			)


def authored(path: Path) -> Iterator[Candidate]:
	"""Hand-written prompts. Each row: id, task, user, optional system, group, reference."""
	with path.open(encoding="utf-8") as source:
		for line_number, line in enumerate(source, 1):
			if not line.strip():
				continue
			row = json.loads(line)
			unknown = set(row) - {"id", "task", "user", "messages", "system", "group", "reference"}
			if unknown or ("user" in row) == ("messages" in row):
				raise ValueError(f"{path}:{line_number}: use known fields and one of user/messages")
			if "messages" in row:
				messages = [Message.model_validate(message) for message in row["messages"]]
				if messages[-1].role != "user":
					raise ValueError(
						f"{path}:{line_number}: the last message must be from the user"
					)
			else:
				messages = [Message(role="user", content=row["user"])]
			if row.get("system"):
				messages.insert(0, Message(role="system", content=row["system"]))
			yield Candidate(
				family="authored",
				example_id=f"authored-{row['id']}",
				corpus=AUTHORED_CORPUS,
				source_id=row["id"],
				group_id=row.get("group", row["id"]),
				license=PROJECT_LICENSE,
				task=row["task"],
				messages=messages,
				reference=row.get("reference", {"kind": "general"}),
			)


def _generated(family: str, items: Iterable[Generated]) -> Iterator[Candidate]:
	for item in items:
		messages = [Message(role="user", content=item.user)]
		if item.system:
			messages.insert(0, Message(role="system", content=item.system))
		yield Candidate(
			family=family,
			example_id=f"generated-{item.key}",
			corpus=GENERATED_CORPUS,
			source_id=item.key,
			group_id=item.group,
			license=PROJECT_LICENSE,
			task=item.task,
			messages=messages,
			reference=item.reference,
		)


def _prompt_text(candidate: Candidate) -> str:
	return "\n".join(message.content for message in candidate.messages)


def build_corpus(
	*,
	output_dir: Path,
	authored_path: Path,
	profile: str = "standard",
	seed: int = 42,
	validation_fraction: float = 0.04,
	max_prompt_chars: int = 3000,
	families: list[str] | None = None,
	index: ContaminationIndex | None = None,
	loaders: dict[str, Callable[[], Iterable[Candidate]]] | None = None,
) -> dict[str, Any]:
	"""Write all-splits.jsonl, references.jsonl, and manifest.json into output_dir."""
	if profile not in PROFILES:
		raise ValueError(f"profile must be one of {sorted(PROFILES)}")
	if not 0 <= validation_fraction < 0.5:
		raise ValueError("validation_fraction must be in [0, 0.5)")
	caps = PROFILES[profile]
	selected_families = families or list(caps)
	if unknown := set(selected_families) - caps.keys():
		raise ValueError(f"unknown families: {sorted(unknown)}")

	benchmark_sources: list[dict[str, Any]] = []
	mbpp_test_ids: set[str] = set()
	if index is None:
		index, benchmark_sources, mbpp_test_ids = benchmark_index()
	if loaders is None:
		loaders = {
			"gsm8k": lambda: gsm8k(seed),
			"aqua_rat": lambda: aqua_rat(seed),
			"arc_challenge": lambda: arc("Challenge", seed),
			"arc_easy": lambda: arc("Easy", seed),
			"commonsense_qa": lambda: commonsense_qa(seed),
			"mbpp": lambda: mbpp(mbpp_test_ids),
			"self_oss_instruct": self_oss_instruct,
			"dolly": dolly,
			"oasst2": oasst2,
			"authored": lambda: authored(authored_path),
			"python_functions": lambda: _generated("python_functions", python_functions()),
			"word_problems": lambda: _generated(
				"word_problems", word_problems(caps["word_problems"] or 3000, seed)
			),
			"json_extraction": lambda: _generated(
				"json_extraction", json_extraction(caps["json_extraction"] or 1200, seed)
			),
			"text_operations": lambda: _generated(
				"text_operations", text_operations(caps["text_operations"] or 1200, seed)
			),
			"constrained_writing": lambda: _generated(
				"constrained_writing",
				constrained_writing(caps["constrained_writing"] or 10_000, seed),
			),
		}

	seen_prompts: set[str] = set()
	family_by_id: dict[str, str] = {}
	prompts: list[PromptRecord] = []
	references: list[ReferenceRecord] = []
	filtered: dict[str, Counter[str]] = defaultdict(Counter)
	available: Counter[str] = Counter()
	for family in selected_families:
		eligible: list[Candidate] = []
		for candidate in loaders[family]():
			available[family] += 1
			text = _prompt_text(candidate)
			if len(text) > max_prompt_chars:
				filtered[family]["too_long"] += 1
			elif index.contains(text):
				filtered[family]["benchmark_overlap"] += 1
			else:
				eligible.append(candidate)
		# A stable hash order makes the capped sample reproducible and source-order independent.
		eligible.sort(key=lambda c: _digest(seed, "sample", c.example_id))
		cap = caps[family]
		kept = 0
		for candidate in eligible:
			if cap is not None and kept >= cap:
				break
			fingerprint = normalized(_prompt_text(candidate))
			if fingerprint in seen_prompts:
				filtered[family]["duplicate_prompt"] += 1
				continue
			if candidate.example_id in family_by_id:
				raise ValueError(f"duplicate example_id: {candidate.example_id}")
			seen_prompts.add(fingerprint)
			family_by_id[candidate.example_id] = family
			kept += 1
			split: Literal["train", "validation"] = (
				"validation"
				if _digest(seed, "split", candidate.corpus, candidate.group_id) % 10_000
				< validation_fraction * 10_000
				else "train"
			)
			payload = [message.model_dump(mode="json") for message in candidate.messages]
			prompts.append(
				PromptRecord(
					example_id=candidate.example_id,
					split=split,
					task=candidate.task,
					source=PromptSource(
						corpus=candidate.corpus,
						source_id=candidate.source_id,
						license=candidate.license,
						group_id=candidate.group_id,
						content_sha256=hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
					),
					prompt=candidate.messages,
				)
			)
			references.append(
				ReferenceRecord.model_validate(
					{"example_id": candidate.example_id, **candidate.reference}
				)
			)

	order = sorted(range(len(prompts)), key=lambda i: prompts[i].example_id)
	prompts = [prompts[i] for i in order]
	references = [references[i] for i in order]
	prompts_path = output_dir / "all-splits.jsonl"
	references_path = output_dir / "references.jsonl"
	write_jsonl(prompts_path, prompts)
	write_jsonl(references_path, references)

	by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "validation": 0})
	for prompt in prompts:
		by_family[family_by_id[prompt.example_id]][prompt.split] += 1
	manifest = {
		"profile": profile,
		"seed": seed,
		"validation_fraction": validation_fraction,
		"max_prompt_chars": max_prompt_chars,
		"total": len(prompts),
		"splits": dict(Counter(p.split for p in prompts)),
		"families": {
			family: {
				"available": available[family],
				"cap": caps[family],
				"filtered": dict(filtered[family]),
				**by_family.get(family, {"train": 0, "validation": 0}),
			}
			for family in selected_families
		},
		"tasks": dict(sorted(Counter(p.task for p in prompts).items())),
		"reference_kinds": dict(sorted(Counter(r.kind for r in references).items())),
		"datasets": {
			key: {
				"repository": spec.repository,
				"configuration": spec.configuration,
				"revision": spec.revision,
				"license": spec.license,
				"url": f"https://huggingface.co/datasets/{spec.repository}/tree/{spec.revision}",
			}
			for key, spec in DATASETS.items()
		},
		"contamination": {
			"method": "drop prompts that share a 13-word span with a held-out question",
			"held_out_questions": index.size,
			"benchmarks": benchmark_sources,
		},
		"sha256": {
			"prompts": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
			"references": hashlib.sha256(references_path.read_bytes()).hexdigest(),
			"authored": hashlib.sha256(authored_path.read_bytes()).hexdigest()
			if authored_path.is_file()
			else None,
		},
	}
	(output_dir / "manifest.json").write_text(
		json.dumps(manifest, indent="\t", sort_keys=True) + "\n", encoding="utf-8"
	)
	return manifest
