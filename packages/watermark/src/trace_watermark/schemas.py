"""Evidence records for a watermark run. Fields that could not be measured are None."""

import json
import math
import re
from dataclasses import asdict, dataclass, fields
from types import NoneType, UnionType
from typing import Any, ClassVar, Self, get_args, get_origin, get_type_hints

SCHEMA_VERSION = 1
_MAX_DEPTH = 32

_WORD_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SEPARATORS = re.compile(r"[^a-z0-9]+")
_SECRET_FRAGMENTS = (
	"secret",
	"password",
	"passwd",
	"passphrase",
	"credential",
	"authorization",
	"bearer",
	"apikey",
	"accesskey",
	"privatekey",
	"apitoken",
	"authtoken",
	"accesstoken",
	"refreshtoken",
	"sessiontoken",
	"hftoken",
)
# Only these words may follow `key` or `token`. Any other suffix may name the value itself.
_METADATA_WORDS = frozenset(
	{
		"id",
		"ids",
		"epoch",
		"version",
		"versions",
		"count",
		"counts",
		"budget",
		"limit",
		"usage",
		"latency",
	}
)
_SECRET_WORDS = frozenset({"pem", "jwt", "cookie", "cookies"})
# A `<role>_token` is a vocabulary entry. Any other `*_token` is treated as a credential.
_VOCABULARY_ROLES = frozenset(
	{
		"added",
		"bos",
		"cls",
		"current",
		"decoder",
		"end",
		"eos",
		"first",
		"generated",
		"input",
		"last",
		"mask",
		"next",
		"output",
		"pad",
		"per",
		"previous",
		"prompt",
		"sampled",
		"scored",
		"sep",
		"special",
		"start",
		"stop",
		"unk",
	}
)


def looks_like_secret(name: str) -> bool:
	"""Whether a field name is credential-shaped. Unknown `*_token` names fail closed."""
	words = [word for word in _SEPARATORS.split(_WORD_BOUNDARY.sub("_", name).lower()) if word]
	stems = [word.rstrip("0123456789") or word for word in words]
	compact = "".join(stems)
	if any(fragment in compact for fragment in _SECRET_FRAGMENTS):
		return True
	for index, stem in enumerate(stems):
		following = words[index + 1 :]
		names_value = not following or following[0] not in _METADATA_WORDS
		if stem == "auth" or stem in _SECRET_WORDS or (stem in {"key", "keys"} and names_value):
			return True
		role = stems[index - 1] if index else None
		if stem == "token" and names_value and role not in _VOCABULARY_ROLES:
			return True
	return False


def reject_secret_names(value: Any, path: str = "$") -> None:
	"""Raise ValueError if any nested mapping key looks like secret material.

	This reads names only, so a secret inside a string value, such as an error, gets through.
	Writers should copy known parameters into evidence rather than whole request or config objects.
	"""
	if isinstance(value, dict):
		for name, item in value.items():
			location = f"{path}.{name}"
			if looks_like_secret(str(name)):
				raise ValueError(f"secret material is forbidden in evidence: {location}")
			reject_secret_names(item, location)
	elif isinstance(value, list | tuple):
		for index, item in enumerate(value):
			reject_secret_names(item, f"{path}[{index}]")


def _check_depth(value: Any, depth: int = 0) -> None:
	"""Bound nesting up front, so later recursive checks cannot exhaust the stack."""
	if depth > _MAX_DEPTH:
		raise ValueError(f"evidence is nested deeper than {_MAX_DEPTH} levels")
	if isinstance(value, dict):
		for item in value.values():
			_check_depth(item, depth + 1)
	elif isinstance(value, list | tuple):
		for item in value:
			_check_depth(item, depth + 1)


def _is_finite_number(value: Any) -> bool:
	if not isinstance(value, int | float) or isinstance(value, bool):
		return False
	# Python integers can exceed the float range, and those are finite by definition.
	return isinstance(value, int) or math.isfinite(value)


def _is_json(value: Any) -> bool:
	"""Whether a value survives a JSON round trip unchanged, with finite numbers throughout."""
	if value is None or isinstance(value, bool | str):
		return True
	if isinstance(value, list):
		return all(_is_json(item) for item in value)
	if isinstance(value, dict):
		return all(isinstance(name, str) and _is_json(item) for name, item in value.items())
	return _is_finite_number(value)


def _matches(value: Any, hint: Any) -> bool:
	origin = get_origin(hint)
	if hint is Any:
		return _is_json(value)
	if origin is UnionType:
		return any(_matches(value, option) for option in get_args(hint))
	if origin is list:
		return isinstance(value, list) and all(_matches(item, get_args(hint)[0]) for item in value)
	if origin is dict:
		key_hint, value_hint = get_args(hint)
		return isinstance(value, dict) and all(
			_matches(key, key_hint) and _matches(item, value_hint) for key, item in value.items()
		)
	if hint is NoneType:
		return value is None
	if hint is int:
		return isinstance(value, int) and not isinstance(value, bool)
	if hint is float:
		return _is_finite_number(value)
	return isinstance(value, hint)


def _reject_constant(name: str) -> None:
	raise ValueError(f"evidence JSON cannot contain {name}; use null for an unmeasured field")


class _Record:
	_indent: ClassVar[str | None] = None

	def __post_init__(self) -> None:
		hints = get_type_hints(type(self))
		for item in fields(self):  # type: ignore[arg-type]
			# Field values sit one level inside the record, as they do when a record is parsed.
			_check_depth(getattr(self, item.name), depth=1)
			if not _matches(getattr(self, item.name), hints[item.name]):
				raise ValueError(f"{type(self).__name__}.{item.name} has an invalid value")

	def to_dict(self) -> dict[str, Any]:
		payload: dict[str, Any] = asdict(self)  # type: ignore[call-overload]
		reject_secret_names(payload)
		return payload

	def to_json(self) -> str:
		return json.dumps(self.to_dict(), indent=self._indent, allow_nan=False)

	@classmethod
	def from_dict(cls, payload: Any) -> Self:
		if not isinstance(payload, dict):
			raise ValueError(f"{cls.__name__} must be a JSON object")
		_check_depth(payload)
		reject_secret_names(payload)
		names = {item.name for item in fields(cls)}  # type: ignore[arg-type]
		if unknown := sorted(payload.keys() - names):
			raise ValueError(f"{cls.__name__} has unknown fields: {', '.join(unknown)}")
		if missing := sorted(names - payload.keys()):
			raise ValueError(f"{cls.__name__} is missing fields: {', '.join(missing)}")
		return cls(**payload)

	@classmethod
	def from_json(cls, text: str) -> Self:
		try:
			payload = json.loads(text, parse_constant=_reject_constant)
		except RecursionError:
			raise ValueError("evidence JSON is nested too deeply") from None
		return cls.from_dict(payload)


@dataclass(frozen=True, kw_only=True)
class RunManifest(_Record):
	"""One `manifest.json`. Written with tab indentation. Never holds secret material."""

	_indent: ClassVar[str | None] = "\t"

	schema_version: int = SCHEMA_VERSION
	run_id: str
	created_at: str
	workspace_base: str | None = None
	serving_lock_sha256: str | None = None
	model_id: str | None = None
	model_revision: str | None = None
	tokenizer_revision: str | None = None
	tokenizer_files_sha256: str | None = None
	chat_template_sha256: str | None = None
	special_token_ids: list[int] | None = None
	thinking: bool | None = None
	effective_generation_config: dict[str, Any] | None = None
	vllm_version: str | None = None
	source_commit: str | None = None
	runner: str | None = None
	torch_version: str | None = None
	transformers_version: str | None = None
	gpu_model: str | None = None
	driver: str | None = None
	cuda_version: str | None = None
	attention_backend: str | None = None
	dtype: str | None = None
	quantization: str | None = None
	algorithm_id: str | None = None
	algorithm_version: int | None = None
	processor_digest: str | None = None
	prf_version: str | None = None
	key_id: str | None = None
	key_epoch: int | None = None
	context_width: int | None = None
	dedup_policy: str | None = None
	gamma: float | None = None
	delta: float | None = None
	fixture_manifest_sha256: str | None = None
	split_hashes: dict[str, str] | None = None
	seed_schedule: Any = None
	concurrency: Any = None
	batch_invariance: Any = None

	def __post_init__(self) -> None:
		super().__post_init__()
		if self.schema_version != SCHEMA_VERSION:
			raise ValueError(f"unsupported manifest schema version: {self.schema_version}")


@dataclass(frozen=True, kw_only=True)
class GenerationRow(_Record):
	"""One line of `generations.jsonl`. Serialized as a single line."""

	request_id: str
	prompt_id: str
	split: str
	task: str
	profile_id: str | None = None
	key_id: str | None = None
	seed: int | None = None
	retry_id: int | None = None
	actual_parameters: dict[str, Any] | None = None
	prompt_token_ids: list[int] | None = None
	generated_token_ids: list[int] | None = None
	exact_text_sha256: str | None = None
	text_only_roundtrip_match: bool | None = None
	finish_reason: str | None = None
	output_tokens: int | None = None
	scored_tokens: int | None = None
	watermark_application_status: str | None = None
	skip_reason: str | None = None
	stream_mode: str | None = None
	batch_condition: str | None = None
	ttft_ms: float | None = None
	inter_token_latency_ms: list[float] | None = None
	latency_ms: float | None = None
	tokens_per_second: float | None = None
	peak_memory_bytes: int | None = None
	utility_score: float | None = None
	exact_answer_match: bool | None = None
	code_pass_at_1: bool | None = None
	json_valid: bool | None = None
	error: str | None = None
