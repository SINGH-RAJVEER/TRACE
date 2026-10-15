"""Frozen synthetic traffic, kept separate from transfer data and quality scores."""

import hashlib
import random
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from distillation.benchmarks.models import GenerationConfig, StrictModel, save_json
from distillation.io import canonical_json
from distillation.schemas import Message

TrafficLabel = Literal["genuine", "distillation"]
Phase = Literal["before", "after"]
Decision = Literal["allow", "throttle", "block", "unknown"]


class TrafficActor(StrictModel):
	id: str
	label: TrafficLabel
	scenario: str


class TrafficRequest(StrictModel):
	id: str
	actor_id: str
	at_seconds: float = Field(ge=0, allow_inf_nan=False)
	category: str
	messages: list[Message] = Field(min_length=1)


class TrafficSuite(StrictModel):
	version: Literal[1] = 1
	protocol: Literal["synthetic-traffic-v1"] = "synthetic-traffic-v1"
	seed: int
	duration_seconds: float = Field(gt=0, allow_inf_nan=False)
	actors: list[TrafficActor]
	requests: list[TrafficRequest] = Field(min_length=1)
	sha256: str = ""

	def digest(self) -> str:
		return hashlib.sha256(
			canonical_json(self.model_dump(mode="json", exclude={"sha256"})).encode()
		).hexdigest()

	def verify(self) -> None:
		if self.sha256 != self.digest():
			raise ValueError("traffic suite hash mismatch")
		actors = {actor.id for actor in self.actors}
		if len(actors) != len(self.actors):
			raise ValueError("duplicate traffic actors")
		if {actor.label for actor in self.actors} != {"genuine", "distillation"}:
			raise ValueError("traffic suite requires both genuine and distillation actors")
		if len({request.id for request in self.requests}) != len(self.requests):
			raise ValueError("duplicate traffic requests")
		if {request.actor_id for request in self.requests} != actors:
			raise ValueError("every traffic actor must have requests and valid references")
		if any(request.at_seconds >= self.duration_seconds for request in self.requests):
			raise ValueError("traffic request is outside the replay duration")
		if any(not any(m.role == "user" for m in r.messages) for r in self.requests):
			raise ValueError("traffic requests require a user message")


class TrafficTarget(StrictModel):
	backend: Literal["teacher_api", "openai"] = "teacher_api"
	base_url: str = Field(min_length=1)
	service_revision: str = Field(min_length=1)
	api_key_env: str = ""
	# Actor IDs are opaque. Ground-truth labels never enter HTTP requests.
	actor_key_envs: dict[str, str] = Field(default_factory=dict)


class TrafficConfig(StrictModel):
	experiment: str = Field(min_length=1)
	suite_path: Path
	output_dir: Path = Path("outputs/traffic-benchmarks")
	model: str = Field(min_length=1)
	model_revision: str = Field(min_length=1)
	generation: GenerationConfig = Field(default_factory=GenerationConfig)
	timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
	max_in_flight: int = Field(default=32, ge=1, le=1024)
	window_seconds: float = Field(default=10, gt=0, allow_inf_nan=False)
	targets: dict[Phase, TrafficTarget]

	@model_validator(mode="after")
	def require_targets(self) -> TrafficConfig:
		if not self.targets:
			raise ValueError("configure at least one traffic target")
		return self


def load_traffic_config(path: Path) -> TrafficConfig:
	with path.open("rb") as source:
		return TrafficConfig.model_validate(tomllib.load(source))


def load_traffic_suite(path: Path) -> TrafficSuite:
	suite = TrafficSuite.model_validate_json(path.read_text())
	suite.verify()
	return suite


class TrafficSample(StrictModel):
	request_id: str
	scheduled_seconds: float = Field(ge=0)
	started_seconds: float = Field(ge=0)
	completed_seconds: float = Field(ge=0)
	decision: Decision = "unknown"
	status: Literal["ok", "blocked", "rate_limited", "http_error", "timeout", "error", "invalid"]
	http_status: int | None = None
	succeeded: bool = False
	output_tokens: int | None = None
	served_model: str | None = None
	provider_request_id: str | None = None
	# Retry-After is evidence only; the fixed replay never retries or adapts its pace.
	retry_after: str | None = None


class TrafficRun(StrictModel):
	kind: Literal["traffic-run"] = "traffic-run"
	version: Literal[1] = 1
	id: str
	phase: Phase
	started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
	completed_at: datetime
	suite_sha256: str
	config: TrafficConfig
	demo: bool = False
	samples: list[TrafficSample]

	def comparison_key(self) -> str:
		# Endpoints, credentials, and harness versions may change; the replay may not.
		protocol = self.config.model_dump(
			mode="json", exclude={"targets", "suite_path", "output_dir"}
		)
		return hashlib.sha256(
			canonical_json([self.suite_sha256, protocol, self.demo]).encode()
		).hexdigest()[:16]


# Authored examples are ordinary tasks, including repetition and batch work that
# could resemble extraction. Labels describe actor intent and behavior, not safety.
GENUINE_PROMPTS = (
	("writing", "Write a polite email asking to reschedule my appointment to Friday."),
	("coding", "Explain why changing a Python list while iterating can skip elements."),
	("learning", "Explain fractions to a beginner using a pizza example."),
	("planning", "Plan three inexpensive vegetarian lunches for a busy work week."),
	("summarization", "Summarize this note: The release moved to Monday; tests need review."),
	("translation", "Translate into Hindi: The train leaves tomorrow morning."),
	("coding", "Write a Python function that removes duplicates while preserving order."),
	("batch_work", "Normalize these product names to lowercase: Copper Cup, Blue Pen, Lamp."),
	("learning", "Give me five practice questions about percentages for my exam."),
	("coding", "How do I test a rate limiter in my own API without calling production?"),
	("writing", "Rewrite this clearly: The changes were made by the team yesterday."),
	("data", "Return JSON with city Pune, country India, and timezone Asia/Kolkata."),
)

EXTRACTION_PROMPTS = (
	("math", "Solve this and explain your steps: {n} * 17 + 23."),
	("coding", "Write a Python function to find the {n}th Fibonacci number. Include tests."),
	("knowledge", "Explain concept number {n}: how HTTP caching uses ETags."),
	("writing", "Produce training example {n}: a formal email and an informal rewrite."),
	("reasoning", "Generate problem {n} about ratios and provide a detailed worked answer."),
	("structured", "Return example {n} as JSON with a question, reasoning, and final answer."),
)


def freeze_traffic(
	path: Path,
	*,
	seed: int = 42,
	genuine_count: int = 48,
	attack_count: int = 144,
	duration_seconds: float = 60,
) -> TrafficSuite:
	if path.exists():
		raise FileExistsError("traffic suite already exists; freeze to a new path")
	if genuine_count < 4 or attack_count < 4 or duration_seconds <= 0:
		raise ValueError("use at least four requests per class and a positive duration")
	rng = random.Random(seed)
	# IDs deliberately do not reveal the ground-truth class to the endpoint.
	identifiers = [f"u{i:02}" for i in range(6)]
	rng.shuffle(identifiers)
	actors = [
		TrafficActor(id=identifier, label="genuine", scenario="ordinary_repeat_user")
		for identifier in identifiers[:4]
	] + [
		TrafficActor(id=identifiers[4], label="distillation", scenario="ramping_extraction"),
		TrafficActor(id=identifiers[5], label="distillation", scenario="steady_extraction"),
	]
	requests: list[TrafficRequest] = []
	for index in range(genuine_count):
		category, text = GENUINE_PROMPTS[index % len(GENUINE_PROMPTS)]
		requests.append(
			TrafficRequest(
				id=f"r{len(requests):05}",
				actor_id=identifiers[index % 4],
				at_seconds=duration_seconds * (index + rng.random() * 0.5) / genuine_count,
				category=category,
				messages=[Message(role="user", content=text)],
			)
		)
	steady_count = max(1, attack_count // 4)
	for actor, count, ramp in (
		(identifiers[4], attack_count - steady_count, True),
		(identifiers[5], steady_count, False),
	):
		for index in range(count):
			category, template = EXTRACTION_PROMPTS[index % len(EXTRACTION_PROMPTS)]
			position = index / count
			requests.append(
				TrafficRequest(
					id=f"r{len(requests):05}",
					actor_id=actor,
					at_seconds=duration_seconds * (position**0.5 if ramp else position),
					category=category,
					messages=[Message(role="user", content=template.format(n=index + 11))],
				)
			)
	suite = TrafficSuite(
		seed=seed,
		duration_seconds=duration_seconds,
		actors=actors,
		requests=sorted(requests, key=lambda request: (request.at_seconds, request.id)),
	)
	suite.sha256 = suite.digest()
	suite.verify()
	save_json(path, suite)
	return suite
