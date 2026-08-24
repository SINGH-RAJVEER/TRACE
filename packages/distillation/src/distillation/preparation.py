"""Project reviewed ledger records into TRL prompt-completion data."""

import hashlib
from pathlib import Path

from distillation.io import canonical_json, read_jsonl, write_jsonl
from distillation.schemas import LedgerRecord, ReviewRecord, SFTRecord


def prepare_sft_dataset(
	*,
	input_path: Path,
	output_path: Path,
	split: str = "train",
	reviews_path: Path | None = None,
	accept_pending: bool = False,
) -> tuple[int, int]:
	if split == "test":
		raise ValueError("the frozen test split cannot be prepared for training")
	ledger = read_jsonl(input_path, LedgerRecord)
	reviews = _load_reviews(reviews_path)
	prepared: list[SFTRecord] = []
	other_split_prompts = {
		_prompt_fingerprint(record) for record in ledger if record.split != split
	}
	seen_prompts: set[str] = set(other_split_prompts)
	skipped = 0

	for record in ledger:
		review = reviews.get(record.example_id)
		accepted_by_review = review is not None and review.status == "accepted"
		accepted_without_review = accept_pending and record.quality.status == "pending"
		if (
			record.split != split
			or record.quality.status == "rejected"
			or not (accepted_by_review or accepted_without_review)
		):
			skipped += 1
			continue
		fingerprint = _prompt_fingerprint(record)
		if fingerprint in seen_prompts:
			skipped += 1
			continue
		seen_prompts.add(fingerprint)
		prepared.append(SFTRecord(prompt=record.prompt, completion=record.completion))

	write_jsonl(output_path, prepared)
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


def _prompt_fingerprint(record: LedgerRecord) -> str:
	payload = [message.model_dump(mode="json") for message in record.prompt]
	return hashlib.sha256(canonical_json(payload).encode()).hexdigest()
