"""Versioned evaluation artifacts and configuration."""

import hashlib
import json
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from distillation.io import canonical_json
from distillation.schemas import Message, TokenUsage

# The initial teacher remains the gap-closure reference; reevaluation measures drift.
Slot = Literal["teacher", "student_before", "teacher_after", "student_after"]
SLOTS: tuple[Slot, ...] = ("teacher", "student_before", "teacher_after", "student_after")
STAGE_SLOTS: dict[str, tuple[Slot, ...]] = {
	"before": ("teacher", "student_before"),
	"after": ("teacher_after", "student_after"),
	"all": SLOTS,
}
SCORER_VERSION = "generated-answer-v1"


class StrictModel(BaseModel):
	model_config = ConfigDict(extra="forbid")


class BenchmarkItem(StrictModel):
	id: str
	benchmark: str
	category: str
	question: str
	messages: list[Message] = Field(min_length=1)
	metric: Literal["numeric_exact_match", "choice_accuracy", "pass_at_1"]
	answer: str
	choices: list[str] = Field(default_factory=list)
	test_code: str = ""
	code_prompt: str = ""
	entry_point: str = ""


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
		return hashlib.sha256(canonical_json(payload).encode()).hexdigest()

	def verify(self) -> None:
		if self.scorer_version != SCORER_VERSION:
			raise ValueError("suite scorer version is unsupported; freeze a new suite")
		if self.sha256 != self.digest():
			raise ValueError("frozen suite hash mismatch")
		if len({item.id for item in self.items}) != len(self.items):
			raise ValueError("duplicate benchmark item IDs")
		counts = {
			s.benchmark: sum(i.benchmark == s.benchmark for i in self.items) for s in self.sources
		}
		if len(counts) != len(self.sources) or any(
			counts[s.benchmark] != s.selected for s in self.sources
		):
			raise ValueError("source counts do not match frozen items")
		if {item.benchmark for item in self.items} != set(counts):
			raise ValueError("benchmark items have missing provenance")


class GenerationConfig(StrictModel):
	max_output_tokens: int = Field(default=1024, ge=1, le=32768)
	seed: int = 42
	temperature: Literal[0] = 0
	enable_thinking: Literal[False] = False


class ModelConfig(StrictModel):
	backend: Literal["openai", "teacher_api", "local"]
	model: str = Field(min_length=1)
	revision: str = Field(min_length=1)
	base_url: str = ""
	api_key_env: str = ""
	adapter_path: Path | None = None
	tokenizer_model: str | None = None
	tokenizer_revision: str | None = None
	chat_template_path: Path | None = None
	eos_token: str | None = None
	precision: Literal["bf16", "fp16", "fp32"] = "bf16"
	load_in_4bit: bool = False
	device: str = "auto"
	timeout_seconds: float = Field(default=120, gt=0)

	@model_validator(mode="after")
	def validate_backend(self) -> ModelConfig:
		if self.backend != "local" and not self.base_url:
			raise ValueError("remote models require base_url")
		if self.backend != "local" and self.adapter_path:
			raise ValueError("serve adapters under a distinct remote model ID")
		if self.tokenizer_model and not self.tokenizer_revision:
			raise ValueError("tokenizer_model requires tokenizer_revision")
		return self


class EvaluationConfig(StrictModel):
	experiment: str = Field(min_length=1)
	suite_path: Path
	output_dir: Path = Path("outputs/benchmarks")
	generation: GenerationConfig = Field(default_factory=GenerationConfig)
	models: dict[Slot, ModelConfig]
	training_paths: list[Path] = Field(default_factory=list)
	code_image: str = "python:3.12-slim"
	code_timeout_seconds: int = Field(default=10, ge=1, le=120)


def load_config(path: Path) -> EvaluationConfig:
	with path.open("rb") as source:
		return EvaluationConfig.model_validate(tomllib.load(source))


class SampleResult(StrictModel):
	item_id: str
	benchmark: str
	category: str
	response: str
	prediction: str | None
	correct: bool
	status: Literal["ok", "invalid", "truncated", "error", "test_failed", "timeout"]
	latency_seconds: float = Field(ge=0)
	usage: TokenUsage = Field(default_factory=TokenUsage)
	finish_reason: str | None = None
	request_id: str | None = None
	served_model: str | None = None


class EvaluationRun(StrictModel):
	version: Literal[1] = 1
	id: str
	experiment: str
	slot: Slot
	started_at: datetime
	completed_at: datetime
	suite_sha256: str
	scorer_version: str = SCORER_VERSION
	generation: GenerationConfig
	model: ModelConfig
	code_image_id: str | None = None
	code_timeout_seconds: int = 10
	training_sha256: dict[str, str] = Field(default_factory=dict)
	artifact_sha256: dict[str, str] = Field(default_factory=dict)
	packages: dict[str, str] = Field(default_factory=dict)
	platform: str = ""
	code_revision: str = ""
	demo: bool = False
	samples: list[SampleResult]

	def comparison_key(self) -> str:
		payload = [
			self.experiment,
			self.suite_sha256,
			self.scorer_version,
			self.generation.model_dump(),
			self.code_image_id,
			self.code_timeout_seconds,
			self.demo,
		]
		return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]


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
