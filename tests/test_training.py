from pathlib import Path

import pytest
from distillation.io import JSONLValidationError, write_jsonl
from distillation.schemas import Message, SFTRecord
from distillation.training import (
	TrainingConfig,
	_record_offloading,
	_reject_overlong_records,
	_resume_point,
	_validate_dataset_splits,
)


class FakeTokenizer:
	chat_template: str | None = "fake"
	eos_token_id: int | None = None

	def apply_chat_template(
		self,
		conversation: list[dict[str, str]],
		*,
		tokenize: bool,
		add_generation_prompt: bool,
		**kwargs: bool | str | int | float,
	) -> list[int]:
		assert tokenize
		return list(range(sum(len(message["content"]) for message in conversation)))


def test_length_preflight_rejects_truncated_teacher_targets() -> None:
	records = [
		{
			"prompt": [{"role": "user", "content": "1234"}],
			"completion": [{"role": "assistant", "content": "5678"}],
		}
	]

	with pytest.raises(ValueError, match="1 records over max_length=7; longest=8"):
		_reject_overlong_records(records, FakeTokenizer(), 7, "training")

	_reject_overlong_records(records, FakeTokenizer(), 8, "training")


@pytest.mark.parametrize("failure", ["prefix", "eos", "masked"])
def test_preflight_rejects_incorrect_completion_mask(failure: str) -> None:
	class BrokenTokenizer(FakeTokenizer):
		eos_token_id = 99 if failure == "eos" else None

		def apply_chat_template(
			self,
			conversation: list[dict[str, str]],
			*,
			tokenize: bool,
			add_generation_prompt: bool,
			**kwargs: bool | str | int | float,
		) -> list[int]:
			if add_generation_prompt:
				return [9] if failure == "prefix" else [1]
			return [1] if failure == "masked" else [1, 2]

	records = [
		{
			"prompt": [{"role": "user", "content": "question"}],
			"completion": [{"role": "assistant", "content": "answer"}],
		}
	]
	expected = {"prefix": "prompt prefix", "eos": "no EOS", "masked": "no supervised"}
	with pytest.raises(ValueError, match=expected[failure]):
		_reject_overlong_records(records, BrokenTokenizer(), 128, "training")


def test_training_admission_rejects_empty_or_overlapping_validation(tmp_path: Path) -> None:
	training, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
	row = SFTRecord(
		prompt=[Message(role="user", content="Question")],
		completion=[Message(role="assistant", content="Answer")],
	)
	write_jsonl(training, [row])
	write_jsonl(validation, [row])
	config = TrainingConfig(
		base_model="local",
		dataset_path=training,
		validation_dataset_path=validation,
		output_dir=tmp_path / "adapter",
	)
	with pytest.raises(ValueError, match="overlapping prompts"):
		_validate_dataset_splits(config)
	write_jsonl(validation, [])
	with pytest.raises(ValueError, match="validation dataset is empty"):
		_validate_dataset_splits(config)
	training.write_text(row.model_dump_json()[:-1] + ',"input_ids":[1],"labels":[1]}\n')
	with pytest.raises(JSONLValidationError, match="invalid record"):
		_validate_dataset_splits(config)


def test_preflight_cli_never_calls_training(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	from distillation import cli
	from typer.testing import CliRunner

	dataset = tmp_path / "train.jsonl"
	write_jsonl(dataset, [])
	config = tmp_path / "train.toml"
	config.write_text(f'base_model="local"\ndataset_path="{dataset}"\noutput_dir="{tmp_path}"\n')

	def forbidden_training(config: TrainingConfig) -> Path:
		pytest.fail("preflight invoked training")

	monkeypatch.setattr(cli, "train", forbidden_training)
	monkeypatch.setattr(cli, "preflight_training", lambda config: {"checked": True})
	result = CliRunner().invoke(cli.app, ["train", "--config", str(config), "--preflight-only"])
	assert result.exit_code == 0, result.output
	assert '"checked": true' in result.output


def test_resume_uses_newest_complete_checkpoint_of_the_same_run(tmp_path: Path) -> None:
	dataset = tmp_path / "train.jsonl"
	write_jsonl(dataset, [])
	config = TrainingConfig(base_model="local", dataset_path=dataset, output_dir=tmp_path / "out")
	fingerprint = {"config": {"learning_rate": 1e-4}, "dataset_sha256": {"training": "a"}}
	assert _resume_point(config, fingerprint) == (None, False)
	for step, complete in ((10, True), (20, True), (30, False)):
		checkpoint = config.output_dir / f"checkpoint-{step}"
		checkpoint.mkdir()
		if complete:
			(checkpoint / "trainer_state.json").write_text("{}")
	_record_offloading(config)
	assert _resume_point(config, fingerprint) == (config.output_dir / "checkpoint-20", True)
	with pytest.raises(ValueError, match="different training config"):
		_resume_point(config, {**fingerprint, "dataset_sha256": {"training": "b"}})
