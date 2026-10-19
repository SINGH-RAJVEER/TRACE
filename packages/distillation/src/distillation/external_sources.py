"""Independently pinned attacker prompt sources for the A7 campaign and its positive control.

The task pools split one training corpus between the attacker and legitimate users, so a
uniform extractor matches the benign references in expectation. A realistic attacker lacks
the victim's own data. These sources serve each domain's fixed template with prompts from a
separately pinned corpus, so A7 tests the detector on a shift it is designed to find. The
pools carry no gold answers: they train attacker students only from teacher responses.
"""

import ast
import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillation.benchmarks.models import load_suite
from distillation.io import canonical_json, write_jsonl
from distillation.schemas import Message, PromptRecord, PromptSource
from distillation.tasks import (
	AG_NEWS_PROMPT,
	GSM8K_EVAL_SUFFIX,
	SQUAD_MAX_CONTEXT_CHARS,
	SQUAD_PROMPT,
	DatasetSpec,
	_truncate,
	normalized,
)

EXTERNAL_VERSION = "external-sources-v1"
CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class ExternalSource:
	dataset: DatasetSpec
	splits: tuple[str, ...]
	description: str


EXTERNAL_SOURCES: dict[str, ExternalSource] = {
	"ag_news": ExternalSource(
		DatasetSpec("SetFit/bbc-news", "9c31f7087a8c225be8b4c420d70216ef1f121eb0", "unknown"),
		("train", "test"),
		"BBC News articles, lower-cased, with five BBC categories rather than AG News topics",
	),
	"squad": ExternalSource(
		DatasetSpec("lucadiliello/newsqa", "728e52920b8e4ffcfaad93fa47556f26a1d82546", "unknown"),
		("train",),
		"NewsQA questions over CNN articles rather than Wikipedia passages",
	),
	"gsm8k": ExternalSource(
		DatasetSpec(
			"MU-NLPC/Calc-math_qa", "102358bfc5de2bf527778bee2f875538c977872a", "apache-2.0"
		),
		("train",),
		"MathQA GRE/GMAT-style problems with answer options removed",
	),
	"coding": ExternalSource(
		DatasetSpec(
			"bigcode/self-oss-instruct-sc2-exec-filter-50k",
			"356bb069eee815daa6e23e9a282eeefe1490ad44",
			"odc-by",
		),
		("train",),
		"StarCoder2 self-generated function instructions with one top-level function response",
	),
}


class Ineligible(ValueError):
	"""A source row cannot be served through the domain's fixed template."""


def _sha(text: str) -> str:
	return hashlib.sha256(normalized(text).encode()).hexdigest()[:24]


def ag_news_external(row: dict[str, Any]) -> tuple[str, str, str]:
	text = " ".join(str(row["text"]).split())
	if not text:
		raise Ineligible("empty article")
	return _sha(text), AG_NEWS_PROMPT.format(text=text), text


def squad_external(row: dict[str, Any]) -> tuple[str, str, str]:
	context, _ = _truncate(str(row["context"]).strip(), SQUAD_MAX_CONTEXT_CHARS)
	question = " ".join(str(row["question"]).split())
	if not context or not question:
		raise Ineligible("empty passage or question")
	# Questions about one article stay in one source group.
	return _sha(context), SQUAD_PROMPT.format(context=context, question=question), question


def gsm8k_external(row: dict[str, Any]) -> tuple[str, str, str]:
	problem = " ".join(str(row["question_without_options"]).split())
	if not problem:
		raise Ineligible("empty problem")
	return _sha(problem), problem + GSM8K_EVAL_SUFFIX, problem


def coding_external(row: dict[str, Any]) -> tuple[str, str, str]:
	from coding.dataset import PROMPT

	specification = str(row["instruction"]).strip()
	blocks = CODE_BLOCK.findall(str(row["response"]))
	if not specification or not blocks:
		raise Ineligible("missing instruction or code block")
	try:
		module = ast.parse(blocks[0])
	except SyntaxError as error:
		raise Ineligible("response code does not parse") from error
	functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
	if len(functions) != 1:
		raise Ineligible("response must define exactly one top-level function")
	declaration = copy.deepcopy(functions[0])
	declaration.body = [ast.Pass()]
	declaration.decorator_list = []
	signature = ast.unparse(declaration).splitlines()[0]
	user = PROMPT.format(specification=specification, signature=signature)
	return _sha(specification), user, specification


CONVERTERS: dict[str, Callable[[dict[str, Any]], tuple[str, str, str]]] = {
	"ag_news": ag_news_external,
	"squad": squad_external,
	"gsm8k": gsm8k_external,
	"coding": coding_external,
}


def _load(source: ExternalSource) -> dict[str, list[dict[str, Any]]]:
	from datasets import load_dataset

	return {
		split: [
			dict(row)
			for row in load_dataset(
				source.dataset.repository,
				source.dataset.configuration,
				split=split,
				revision=source.dataset.revision,
			)
		]
		for split in source.splits
	}


def _evaluation_texts(task_dir: Path) -> set[str]:
	"""Normalized user turns of the build's frozen suites, for exact-overlap removal."""
	texts: set[str] = set()
	for name in ("development-suite.json", "test-suite.json"):
		path = task_dir / name
		if path.exists():
			for item in load_suite(path).items:
				texts.update(normalized(m.content) for m in item.messages if m.role == "user")
				texts.add(normalized(item.question))
	return texts


def build_external(
	*,
	task: str,
	task_dir: Path,
	rows: dict[str, Iterable[dict[str, Any]]] | None = None,
	log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
	"""Write TASK_DIR/external/attacker.jsonl and its manifest from the pinned source.

	Prompts follow the domain's frozen template and output cap. Exact duplicates and exact
	overlaps with the build's development and test suites are dropped and counted; paraphrase
	and pretraining overlap are not detected.
	"""
	if task not in EXTERNAL_SOURCES:
		raise ValueError(f"no external source for {task}")
	output = task_dir / "external"
	if (output / "manifest.json").exists():
		raise ValueError(f"{output} already holds an external pool; choose a new task build")
	source = EXTERNAL_SOURCES[task]
	if rows is None and log:
		log(f"downloading {source.dataset.repository}")
	sources: dict[str, Iterable[dict[str, Any]]] = dict(_load(source) if rows is None else rows)
	convert = CONVERTERS[task]
	evaluation = _evaluation_texts(task_dir)
	filtered: Counter[str] = Counter()
	seen: set[str] = set()
	records = []
	available = {}
	for split, split_rows in sources.items():
		count = 0
		for index, row in enumerate(split_rows):
			count += 1
			try:
				group, user, content = convert(row)
			except Ineligible:
				filtered["ineligible"] += 1
				continue
			fingerprint = normalized(user)
			if fingerprint in seen:
				filtered["duplicate_prompt"] += 1
				continue
			if fingerprint in evaluation or normalized(content) in evaluation:
				filtered["evaluation_overlap"] += 1
				continue
			seen.add(fingerprint)
			messages = [Message(role="user", content=user)]
			payload = [message.model_dump(mode="json") for message in messages]
			records.append(
				PromptRecord(
					example_id=f"{task}-external-{split}-{index}",
					split="train",
					task=task,
					source=PromptSource(
						corpus=source.dataset.repository,
						source_id=f"{split}:{index}",
						license=source.dataset.license,
						group_id=f"external:{group}",
						content_sha256=hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
					),
					prompt=messages,
				)
			)
		available[split] = count
	if not records:
		raise ValueError(f"external source for {task} produced no eligible prompts")
	output.mkdir(parents=True, exist_ok=True)
	path = output / "attacker.jsonl"
	write_jsonl(path, records)
	manifest = {
		"version": EXTERNAL_VERSION,
		"task": task,
		"dataset": {
			"repository": source.dataset.repository,
			"configuration": source.dataset.configuration,
			"revision": source.dataset.revision,
			"license": source.dataset.license,
			"splits": list(source.splits),
		},
		"description": source.description,
		"available": available,
		"filtered": dict(filtered),
		"prompts": len(records),
		"groups": len({r.source.group_id for r in records}),
		"overlap_check": "exact normalized match against development and test suites",
		"sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
	}
	(output / "manifest.json").write_text(
		json.dumps(manifest, indent="\t", sort_keys=True) + "\n", encoding="utf-8"
	)
	return manifest
