"""Frozen benchmark suites, generation settings and JSON artifact helpers."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from distillation.io import canonical_json
from distillation.schemas import Message

SCORER_VERSION = "generated-answer-v1"


class StrictModel(BaseModel):
	model_config = ConfigDict(extra="forbid")


class BenchmarkItem(StrictModel):
	id: str
	benchmark: str
	category: str
	question: str
	messages: list[Message] = Field(min_length=1)
	metric: Literal["numeric_exact_match", "label_accuracy", "squad", "pass_at_1"]
	answer: str
	choices: list[str] = Field(default_factory=list)
	# Empty legacy fields preserve the hashes of already frozen v1 paper-task suites.
	test_code: str = ""
	code_prompt: str = ""
	entry_point: str = ""
	# Every accepted gold answer for SQuAD; empty for other metrics.
	answers: list[str] = Field(default_factory=list)

	@model_validator(mode="after")
	def validate_code(self) -> BenchmarkItem:
		if self.metric == "pass_at_1" and (
			not self.test_code or not self.entry_point.isidentifier()
		):
			raise ValueError("coding items require executable tests and a function entry point")
		return self


class DatasetSource(StrictModel):
	benchmark: str
	repository: str
	revision: str
	configuration: str
	split: str
	license: str
	url: str
	available: int = Field(ge=1)
	selected: int = Field(ge=1)


class FrozenSuite(StrictModel):
	version: Literal[1] = 1
	scorer_version: str = SCORER_VERSION
	seed: int
	created_at: datetime
	sources: list[DatasetSource]
	items: list[BenchmarkItem] = Field(min_length=1)
	sha256: str = ""

	def digest(self) -> str:
		payload = self.model_dump(mode="json", exclude={"sha256", "created_at"})
		# Suites frozen before the answers field existed keep their hash.
		for item in payload["items"]:
			if not item["answers"]:
				del item["answers"]
		return hashlib.sha256(canonical_json(payload).encode()).hexdigest()

	def verify(self) -> None:
		if self.scorer_version != SCORER_VERSION:
			raise ValueError("suite scorer version is unsupported; freeze a new suite")
		if self.sha256 != self.digest():
			raise ValueError("frozen suite hash mismatch")
		if len({item.id for item in self.items}) != len(self.items):
			raise ValueError("duplicate benchmark item IDs")
		# A benchmark may draw from several pinned sources, such as HumanEval+ and MBPP+.
		counts = {
			s.benchmark: sum(i.benchmark == s.benchmark for i in self.items) for s in self.sources
		}
		selected = {
			name: sum(s.selected for s in self.sources if s.benchmark == name) for name in counts
		}
		if counts != selected or len({(s.benchmark, s.repository) for s in self.sources}) != len(
			self.sources
		):
			raise ValueError("source counts do not match frozen items")
		if {item.benchmark for item in self.items} != set(counts):
			raise ValueError("benchmark items have missing provenance")


class GenerationConfig(StrictModel):
	max_output_tokens: int = Field(default=1024, ge=1, le=32768)
	seed: int = 42
	temperature: Literal[0] = 0
	enable_thinking: Literal[False] = False


def save_json(path: Path, value: BaseModel | dict[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
	temporary = path.with_suffix(".tmp")
	temporary.write_text(json.dumps(payload, indent="\t", ensure_ascii=False) + "\n")
	temporary.replace(path)


def load_suite(path: Path) -> FrozenSuite:
	suite = FrozenSuite.model_validate_json(path.read_text())
	suite.verify()
	return suite
