import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from distillation.corpus import Candidate, ContaminationIndex, build_corpus
from distillation.io import read_jsonl, write_jsonl
from distillation.review import check_response, review_ledger
from distillation.schemas import (
	CollectionAttempt,
	LedgerRecord,
	Message,
	PromptRecord,
	PromptSource,
	QualityRecord,
	ReferenceRecord,
	ReviewRecord,
	TeacherMetadata,
	TokenUsage,
)
from distillation.synthetic import (
	constrained_writing,
	json_extraction,
	python_functions,
	text_operations,
	word_problems,
)

HELD_OUT = "Janet's ducks lay 16 eggs per day and she eats three for breakfast every morning"


def test_contamination_index_matches_spans_and_short_questions() -> None:
	index = ContaminationIndex([HELD_OUT, "Which gas do plants absorb?"])
	assert index.contains("janet's  ducks lay 16 eggs per day and SHE eats three for breakfast")
	assert index.contains("Q: which gas do plants absorb? Return only the letter.")
	assert not index.contains("Janet's ducks lay eggs. Which gas do plants release?")


def _candidates(count: int, group_size: int) -> Iterable[Candidate]:
	for i in range(count):
		yield Candidate(
			family="authored",
			example_id=f"authored-{i}",
			corpus="test",
			source_id=str(i),
			group_id=f"group-{i // group_size}",
			license="Apache-2.0",
			task="test",
			messages=[Message(role="user", content=f"Prompt number {i}")],
			reference={"kind": "exact", "answer": str(i)},
		)


def test_build_corpus_filters_and_keeps_groups_in_one_split(tmp_path: Path) -> None:
	def loader() -> Iterable[Candidate]:
		yield from _candidates(400, 4)
		yield Candidate(
			family="authored",
			example_id="authored-long",
			corpus="test",
			source_id="long",
			group_id="long",
			license="Apache-2.0",
			task="test",
			messages=[Message(role="user", content="x" * 500)],
			reference={"kind": "general"},
		)
		yield Candidate(
			family="authored",
			example_id="authored-leak",
			corpus="test",
			source_id="leak",
			group_id="leak",
			license="Apache-2.0",
			task="test",
			messages=[Message(role="user", content=HELD_OUT + "?")],
			reference={"kind": "general"},
		)
		yield Candidate(
			family="authored",
			example_id="authored-duplicate",
			corpus="test",
			source_id="duplicate",
			group_id="duplicate",
			license="Apache-2.0",
			task="test",
			messages=[Message(role="user", content="prompt NUMBER 7")],
			reference={"kind": "general"},
		)

	manifest = build_corpus(
		output_dir=tmp_path,
		authored_path=tmp_path / "missing.jsonl",
		profile="quick",
		families=["authored"],
		validation_fraction=0.2,
		max_prompt_chars=200,
		index=ContaminationIndex([HELD_OUT]),
		loaders={"authored": loader},
	)
	prompts = read_jsonl(tmp_path / "all-splits.jsonl", PromptRecord)
	references = read_jsonl(tmp_path / "references.jsonl", ReferenceRecord)
	assert [p.example_id for p in prompts] == [r.example_id for r in references]
	assert len(prompts) == 400
	assert manifest["families"]["authored"]["filtered"] == {
		"too_long": 1,
		"benchmark_overlap": 1,
		"duplicate_prompt": 1,
	}
	splits_by_group: dict[str | None, set[str]] = {}
	for prompt in prompts:
		splits_by_group.setdefault(prompt.source.group_id, set()).add(prompt.split)
	assert all(len(splits) == 1 for splits in splits_by_group.values())
	assert 0 < manifest["splits"]["validation"] < 160


def test_generated_families_have_valid_unique_references() -> None:
	items = [
		*python_functions(),
		*word_problems(200, 1),
		*json_extraction(60, 1),
		*text_operations(80, 1),
		*constrained_writing(100, 1),
	]
	assert len({item.key for item in items}) == len(items)
	for item in items:
		reference = ReferenceRecord.model_validate({"example_id": item.key, **item.reference})
		if reference.kind == "numeric":
			assert float(reference.answer or "-1") >= 0


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


def test_check_response_validates_each_reference_kind() -> None:
	def ok(text: str, **reference: object) -> bool:
		record = ReferenceRecord.model_validate({"example_id": "x", **reference})
		return check_response(text, record).status == "accepted"

	assert ok("3 + 4 = 7\n#### 7", kind="numeric", answer="7.0")
	assert not ok("The answer is 7", kind="numeric", answer="7")
	assert ok("B", kind="choice", answer="B")
	assert not ok("B because", kind="choice", answer="B")
	assert ok("Water boils first.\nAnswer: **C**", kind="choice_reasoning", answer="C")
	assert not ok("Answer: A", kind="choice_reasoning", answer="C")
	assert ok("```python\ndef f(x):\n    return x\n```", kind="python_tests", entry_point="f")
	assert not ok("def g(x): return x", kind="python_tests", entry_point="f")
	assert ok('{"b": 2, "a": 1}', kind="json_equals", answer='{"a": 1, "b": 2}')
	assert not ok("{'a': 1}", kind="json_valid")
	assert ok("Apple, Pear.", kind="exact", answer="apple,pear")
	assert ok("- one\n- two", kind="general", constraints={"bullet_count": 2})
	assert not ok("Intro\n- one\n- two", kind="general", constraints={"bullet_count": 2})
	assert not ok("It is very hot", kind="general", constraints={"must_exclude": ["very"]})


def test_review_ledger_writes_one_decision_per_record_with_overrides(tmp_path: Path) -> None:
	ledger = tmp_path / "ledger.jsonl"
	write_jsonl(
		ledger,
		[
			_ledger("right", "#### 4"),
			_ledger("wrong", "#### 5"),
			_ledger("unkeyed", "Free text"),
			_ledger("overridden", "#### 9"),
			_ledger("collector", "x", "rejected"),
		],
	)
	references = tmp_path / "references.jsonl"
	write_jsonl(
		references,
		[
			ReferenceRecord(example_id=name, kind="numeric", answer="4")
			for name in ("right", "wrong", "overridden", "collector")
		],
	)
	overrides = tmp_path / "human.jsonl"
	write_jsonl(
		overrides,
		[ReviewRecord(example_id="overridden", status="accepted", checks=["human"], reviewer="r1")],
	)
	output = tmp_path / "reviews.jsonl"
	summary = review_ledger(
		ledger_path=ledger,
		references_path=references,
		output_path=output,
		overrides_path=overrides,
	)
	reviews = {review.example_id: review for review in read_jsonl(output, ReviewRecord)}
	assert {name: review.status for name, review in reviews.items()} == {
		"right": "accepted",
		"wrong": "rejected",
		"unkeyed": "accepted",
		"overridden": "accepted",
		"collector": "rejected",
	}
	assert reviews["overridden"].reviewer == "r1"
	assert summary["status"] == {"accepted": 3, "rejected": 2}
	assert json.loads(output.with_suffix(".summary.json").read_text())["overrides"] == 1
