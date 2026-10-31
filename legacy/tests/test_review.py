from datetime import UTC, datetime
from pathlib import Path

import pytest
from distillation.io import read_jsonl, write_jsonl
from distillation.review import review_ledger
from distillation.schemas import (
	CollectionAttempt,
	LedgerRecord,
	Message,
	PromptSource,
	QualityRecord,
	ReviewRecord,
	TeacherMetadata,
	TokenUsage,
)


def _ledger(example_id: str, completion: str, status: str = "pending") -> LedgerRecord:
	now = datetime.now(UTC)
	return LedgerRecord(
		example_id=example_id,
		attempt_id="attempt",
		split="train",
		task="test",
		source=PromptSource(corpus="test", source_id=example_id, license="Apache-2.0"),
		prompt=[Message(role="user", content="question")],
		teacher=TeacherMetadata(
			provider="test",
			model="teacher",
			endpoint="http://teacher",
			generated_at=now,
			parameters={},
			terms_review_id="approval",
		),
		completion=[Message(role="assistant", content=completion)],
		usage=TokenUsage(),
		quality=QualityRecord(status=status, checks=["basic"], reviewer="collector"),  # type: ignore[arg-type]
		attempts=[CollectionAttempt(attempt_id="attempt", attempted_at=now, status="succeeded")],
	)


def test_task_review_keeps_wrong_and_malformed_answers(tmp_path: Path) -> None:
	ledger, output = tmp_path / "ledger.jsonl", tmp_path / "reviews.jsonl"
	write_jsonl(
		ledger,
		[
			_ledger("wrong", "This is not an AG News label"),
			_ledger("collector", "World", "rejected"),
		],
	)
	summary = review_ledger(ledger_path=ledger, output_path=output)
	reviews = {r.example_id: r for r in read_jsonl(output, ReviewRecord)}
	assert {key: r.status for key, r in reviews.items()} == {
		"wrong": "accepted",
		"collector": "rejected",
	}
	assert summary["status"] == {"accepted": 1, "rejected": 1}
	assert summary["length_checked"] is False


def test_task_review_checks_training_length(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	import distillation.review as review

	class Lengths:
		max_length = 10

		def __init__(self, config: Path) -> None:
			pass

		def length(self, prompt: list[Message], completion: list[Message]) -> int:
			return len(completion[0].content)

	monkeypatch.setattr(review, "LengthChecker", Lengths)
	ledger, output = tmp_path / "ledger.jsonl", tmp_path / "reviews.jsonl"
	write_jsonl(ledger, [_ledger("short", "World"), _ledger("long", "x" * 11)])
	summary = review_ledger(
		ledger_path=ledger, output_path=output, config_path=tmp_path / "config.toml"
	)
	assert summary["status"] == {"accepted": 1, "rejected": 1}
	assert read_jsonl(output, ReviewRecord)[1].checks == ["length"]
