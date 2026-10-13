"""Project reviewed ledger records into TRL prompt-completion data."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from distillation.io import canonical_json, read_jsonl, write_jsonl
from distillation.schemas import LedgerRecord, PromptRecord, ReviewRecord, SFTRecord


def prepare_sft_dataset(
	*,
	input_path: Path,
	output_path: Path,
	split: str = "train",
	reviews_path: Path | None = None,
	prompts_path: Path | None = None,
	accept_pending: bool = False,
) -> tuple[int, int]:
	if split == "test":
		raise ValueError("the frozen test split cannot be prepared for training")
	if split not in {"train", "validation"}:
		raise ValueError("split must be train or validation")
	for path in (input_path, reviews_path, prompts_path):
		if path and output_path.resolve() == path.resolve():
			raise ValueError("prepared output must not overwrite an input artifact")
	ledger = read_jsonl(input_path, LedgerRecord)
	if len({record.example_id for record in ledger}) != len(ledger):
		raise ValueError("ledger contains duplicate example_id values")
	reviews = _load_reviews(reviews_path)
	prepared: list[SFTRecord] = []
	prompts = read_jsonl(prompts_path, PromptRecord) if prompts_path else []
	if prompts_path:
		by_id = {record.example_id: record for record in prompts}
		if len(by_id) != len(prompts):
			raise ValueError("prompt set contains duplicate example_id values")
		for record in ledger:
			original = by_id.get(record.example_id)
			if original is None or any(
				getattr(original, key) != getattr(record, key)
				for key in ("split", "task", "source", "prompt")
			):
				raise ValueError(f"ledger does not match prompt set: {record.example_id}")
	all_prompts: list[LedgerRecord | PromptRecord] = [*ledger, *prompts]
	other_split_prompts = {
		_prompt_fingerprint(record) for record in all_prompts if record.split != split
	}
	other_split_groups = {_source_group(record) for record in all_prompts if record.split != split}
	seen_prompts: set[str] = set(other_split_prompts)
	skipped = 0
	reasons: Counter[str] = Counter()
	tasks: Counter[str] = Counter()
	corpora: Counter[str] = Counter()

	for record in ledger:
		review = reviews.get(record.example_id)
		accepted_by_review = review is not None and review.status == "accepted"
		accepted_without_review = (
			accept_pending and review is None and record.quality.status == "pending"
		)
		if (
			record.split != split
			or record.quality.status == "rejected"
			or not (accepted_by_review or accepted_without_review)
		):
			skipped += 1
			reasons["other_split" if record.split != split else "quality_or_review"] += 1
			continue
		fingerprint = _prompt_fingerprint(record)
		if fingerprint in seen_prompts or _source_group(record) in other_split_groups:
			skipped += 1
			reasons["duplicate_or_split_overlap"] += 1
			continue
		seen_prompts.add(fingerprint)
		prepared.append(SFTRecord(prompt=record.prompt, completion=record.completion))
		tasks[record.task] += 1
		corpora[record.source.corpus] += 1

	write_jsonl(output_path, prepared)
	manifest = {
		"split": split,
		"accept_pending": accept_pending,
		"written": len(prepared),
		"skipped": dict(reasons),
		"tasks": dict(tasks),
		"corpora": dict(corpora),
		"sha256": {
			name: hashlib.sha256(path.read_bytes()).hexdigest()
			for name, path in (
				("ledger", input_path),
				("reviews", reviews_path),
				("prompts", prompts_path),
				("dataset", output_path),
			)
			if path is not None
		},
	}
	output_path.with_suffix(".manifest.json").write_text(
		json.dumps(manifest, indent="\t", sort_keys=True) + "\n", encoding="utf-8"
	)
	return len(prepared), skipped


def _load_reviews(path: Path | None) -> dict[str, ReviewRecord]:
	if path is None:
		return {}
	reviews = read_jsonl(path, ReviewRecord)
	by_id: dict[str, ReviewRecord] = {}
	for review in reviews:
		if review.example_id in by_id:
			raise ValueError(f"duplicate review for example_id: {review.example_id}")
		by_id[review.example_id] = review
	return by_id


def _source_group(record: LedgerRecord | PromptRecord) -> tuple[str, str]:
	return record.source.corpus, record.source.group_id or record.source.source_id


def _prompt_fingerprint(record: LedgerRecord | PromptRecord) -> str:
	payload = [message.model_dump(mode="json") for message in record.prompt]
	return hashlib.sha256(canonical_json(payload).encode()).hexdigest()
