from dataclasses import asdict, fields

import pytest

from watermark_probe.core import WatermarkKey
from watermark_probe.schemas import GenerationRow, RunManifest, looks_like_secret

MANIFEST_FIELDS = [
	"schema_version",
	"run_id",
	"created_at",
	"workspace_base",
	"serving_lock_sha256",
	"model_id",
	"model_revision",
	"tokenizer_revision",
	"tokenizer_files_sha256",
	"chat_template_sha256",
	"special_token_ids",
	"thinking",
	"effective_generation_config",
	"vllm_version",
	"source_commit",
	"runner",
	"torch_version",
	"transformers_version",
	"gpu_model",
	"driver",
	"cuda_version",
	"attention_backend",
	"dtype",
	"quantization",
	"algorithm_id",
	"algorithm_version",
	"processor_digest",
	"prf_version",
	"key_id",
	"key_epoch",
	"context_width",
	"dedup_policy",
	"gamma",
	"delta",
	"fixture_manifest_sha256",
	"split_hashes",
	"seed_schedule",
	"concurrency",
	"batch_invariance",
]
ROW_FIELDS = [
	"request_id",
	"prompt_id",
	"split",
	"task",
	"profile_id",
	"key_id",
	"seed",
	"retry_id",
	"actual_parameters",
	"prompt_token_ids",
	"generated_token_ids",
	"exact_text_sha256",
	"text_only_roundtrip_match",
	"finish_reason",
	"output_tokens",
	"scored_tokens",
	"watermark_application_status",
	"skip_reason",
	"stream_mode",
	"batch_condition",
	"ttft_ms",
	"inter_token_latency_ms",
	"latency_ms",
	"tokens_per_second",
	"peak_memory_bytes",
	"utility_score",
	"exact_answer_match",
	"code_pass_at_1",
	"json_valid",
	"error",
]


def manifest(**overrides: object) -> RunManifest:
	return RunManifest(run_id="run-1", created_at="2026-11-06T00:00:00Z", **overrides)


def row(**overrides: object) -> GenerationRow:
	return GenerationRow(request_id="r1", prompt_id="p1", split="test", task="squad", **overrides)


def test_field_names_are_exactly_the_reports() -> None:
	assert [item.name for item in fields(RunManifest)] == MANIFEST_FIELDS
	assert [item.name for item in fields(GenerationRow)] == ROW_FIELDS


def test_manifest_round_trip_uses_tabs_and_keeps_nulls() -> None:
	original = manifest(
		special_token_ids=[1, 2],
		effective_generation_config={"temperature": 0.7, "max_tokens": 64},
		key_id="public-fixture",
		key_epoch=3,
		gamma=0.25,
		split_hashes={"test": "abc"},
	)
	text = original.to_json()
	assert '\n\t"run_id"' in text
	assert '"model_id": null' in text
	assert RunManifest.from_json(text) == original


def test_generation_row_round_trip_is_one_line() -> None:
	original = row(
		actual_parameters={"temperature": 1, "max_tokens": 16},
		prompt_token_ids=[5, 6],
		generated_token_ids=[7],
		output_tokens=1,
		tokens_per_second=12.5,
		inter_token_latency_ms=[3.5],
	)
	text = original.to_json()
	assert "\n" not in text
	assert GenerationRow.from_json(text) == original


def test_unmeasured_fields_default_to_none() -> None:
	assert row().latency_ms is None
	assert manifest().gpu_model is None


def test_parsing_rejects_missing_unknown_and_not_a_number() -> None:
	payload = row().to_dict()
	del payload["error"]
	with pytest.raises(ValueError, match="missing fields: error"):
		GenerationRow.from_dict(payload)
	with pytest.raises(ValueError, match="unknown fields: extra"):
		GenerationRow.from_dict({**row().to_dict(), "extra": 1})
	with pytest.raises(ValueError, match="NaN"):
		GenerationRow.from_json(row().to_json().replace('"latency_ms": null', '"latency_ms": NaN'))
	with pytest.raises(ValueError, match="latency_ms"):
		row(latency_ms=float("nan"))
	with pytest.raises(ValueError, match="output_tokens"):
		row(output_tokens="12")


@pytest.mark.parametrize(
	"name",
	[
		"secret",
		"secret_material",
		"client_secret",
		"api_key",
		"apiKey",
		"X-Api-Key",
		"OPENAI_API_KEY",
		"key",
		"key_hex",
		"watermark_key",
		"private_key",
		"token",
		"hf_token",
		"HF_TOKEN",
		"access_token",
		"accessToken",
		"session_token",
		"authorization",
		"password",
	],
)
def test_guard_rejects_credential_names(name: str) -> None:
	assert looks_like_secret(name)
	with pytest.raises(ValueError, match="secret material"):
		manifest(effective_generation_config={"outer": [{name: "hidden-value"}]}).to_json()
	with pytest.raises(ValueError, match="secret material"):
		RunManifest.from_dict({**manifest().to_dict(), name: "hidden-value"})


@pytest.mark.parametrize(
	"name",
	[
		*MANIFEST_FIELDS,
		*ROW_FIELDS,
		"max_tokens",
		"prompt_tokens",
		"skip_special_tokens",
		"pad_token",
		"eos_token_id",
		"time_to_first_token_ms",
		"candidate_key_id",
	],
)
def test_guard_accepts_legitimate_names(name: str) -> None:
	assert not looks_like_secret(name)


def test_guard_catches_a_serialized_key_and_never_echoes_its_value() -> None:
	key = WatermarkKey("fixture", 0, b"public-test-material-only")
	with pytest.raises(ValueError, match="effective_generation_config") as raw_bytes:
		manifest(effective_generation_config=asdict(key))
	assert "public-test" not in str(raw_bytes.value)
	leaked = manifest(effective_generation_config={"secret": key.secret.hex()})
	with pytest.raises(ValueError, match="effective_generation_config.secret") as caught:
		leaked.to_json()
	assert key.secret.hex() not in str(caught.value)


def test_guard_also_covers_generation_parameters() -> None:
	with pytest.raises(ValueError, match="actual_parameters.api_key"):
		row(actual_parameters={"api_key": "hidden-value"}).to_json()


@pytest.mark.parametrize(
	"name",
	[
		"model",
		"messages",
		"role",
		"content",
		"temperature",
		"max_tokens",
		"seed",
		"chat_template_kwargs",
		"enable_thinking",
		"top_p",
		"top_k",
		"skip_special_tokens",
		"stop_token_ids",
		"min_tokens",
		"logit_bias",
		"n_probs",
		"cache_prompt",
		"add_special_tokens",
	],
)
def test_real_request_parameters_are_not_mistaken_for_secrets(name: str) -> None:
	assert not looks_like_secret(name)


@pytest.mark.parametrize("name", ["key_data", "token_data", "signing_key_pem", "jwt", "client_pem"])
def test_unknown_suffixes_on_credential_words_are_rejected(name: str) -> None:
	assert looks_like_secret(name)
	with pytest.raises(ValueError, match=name):
		manifest(effective_generation_config={name: "private-material"}).to_json()


@pytest.mark.parametrize(
	"text",
	['{"temperature": 1e999}', '{"nested": [{"temperature": -1e999}]}'],
)
def test_nested_overflow_is_rejected_on_parse(text: str) -> None:
	payload = row().to_json().replace('"actual_parameters": null', f'"actual_parameters": {text}')
	with pytest.raises(ValueError, match="actual_parameters"):
		GenerationRow.from_json(payload)


@pytest.mark.parametrize("value", [float("inf"), (1, 2), {1: "x"}, b"raw", {"a": [float("nan")]}])
def test_nested_values_must_survive_json(value: object) -> None:
	with pytest.raises(ValueError, match="actual_parameters"):
		row(actual_parameters={"value": value})


def test_huge_integers_stay_valid_and_every_accepted_record_round_trips() -> None:
	original = row(actual_parameters={"seed": 10**400, "nested": [{"top_p": 0.9}, None, True]})
	assert GenerationRow.from_json(original.to_json()) == original
	assert row(latency_ms=10**400).latency_ms == 10**400


def test_deep_nesting_is_a_value_error() -> None:
	with pytest.raises(ValueError, match="nested"):
		GenerationRow.from_json("[" * 200_000)


def test_complete_records_with_deep_nesting_are_value_errors_on_every_path() -> None:
	deep: object = 0
	for _ in range(600):
		deep = [deep]
	with pytest.raises(ValueError, match="nested"):
		row(actual_parameters={"x": deep})
	payload = {**row().to_dict(), "actual_parameters": {"x": deep}}
	with pytest.raises(ValueError, match="nested"):
		GenerationRow.from_dict(payload)
	nested_text = "[" * 600 + "0" + "]" * 600
	text = (
		row()
		.to_json()
		.replace('"actual_parameters": null', f'"actual_parameters": {{"x": {nested_text}}}')
	)
	with pytest.raises(ValueError, match="nested"):
		GenerationRow.from_json(text)


def nested_lists(levels: int) -> object:
	value: object = 0
	for _ in range(levels):
		value = [value]
	return value


def test_depth_limit_is_the_same_for_construction_and_parsing() -> None:
	# The record is level 0, the field level 1, its "x" value level 2.
	deepest = nested_lists(30)
	record = row(actual_parameters={"x": deepest})
	assert GenerationRow.from_json(record.to_json()) == record
	assert GenerationRow.from_dict(record.to_dict()) == record
	config = manifest(effective_generation_config={"x": deepest})
	assert RunManifest.from_json(config.to_json()) == config
	assert RunManifest.from_dict(config.to_dict()) == config

	too_deep = nested_lists(31)
	with pytest.raises(ValueError, match="nested"):
		row(actual_parameters={"x": too_deep})
	with pytest.raises(ValueError, match="nested"):
		manifest(effective_generation_config={"x": too_deep})
	with pytest.raises(ValueError, match="nested"):
		GenerationRow.from_dict({**row().to_dict(), "actual_parameters": {"x": too_deep}})
