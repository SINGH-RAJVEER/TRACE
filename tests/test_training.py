import pytest
from distillation.training import _reject_overlong_records


class FakeTokenizer:
	chat_template: str | None = "fake"

	def apply_chat_template(
		self,
		conversation: list[dict[str, str]],
		*,
		tokenize: bool,
		add_generation_prompt: bool,
	) -> list[int]:
		assert tokenize
		assert not add_generation_prompt
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
