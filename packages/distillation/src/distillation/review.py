"""Review attacker responses for usability without consulting gold answers."""

import json
import re
import tomllib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillation.io import read_jsonl, write_jsonl
from distillation.schemas import LedgerRecord, Message, ReviewRecord

REVIEWER = "task-usability-v1"


@dataclass(frozen=True)
class Decision:
	status: str
	checks: list[str]
	reason: str | None = None


class LengthChecker:
	"""Rejects records the training config would refuse, using the student tokenizer."""

	def __init__(self, config_path: Path) -> None:
		from distillation.training import TrainingConfig, _load_tokenizer

		with config_path.open("rb") as source:
			data: dict[str, Any] = tomllib.load(source)
		# The prepared dataset does not exist yet, so skip path validation.
		defaults = {
			name: field.get_default(call_default_factory=True)
			for name, field in TrainingConfig.model_fields.items()
			if not field.is_required()
		}
		config = TrainingConfig.model_construct(**{**defaults, **data})
		self.max_length = int(config.max_length)
		self.kwargs = dict(config.chat_template_kwargs)
		self.tokenizer = _load_tokenizer(config)

	def length(self, prompt: list[Message], completion: list[Message]) -> int:
		conversation = [message.model_dump(mode="json") for message in [*prompt, *completion]]
		token_ids = self.tokenizer.apply_chat_template(
			conversation,
			tokenize=True,
			add_generation_prompt=False,
			return_dict=False,
			**self.kwargs,
		)
		return len(token_ids)


def review_ledger(
	*,
	ledger_path: Path,
	output_path: Path,
	config_path: Path | None = None,
	progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
	"""Keep wrong and malformed answers; reject collector failures and overlong records."""
	for path in (ledger_path, config_path):
		if path and output_path.resolve() == path.resolve():
			raise ValueError("review output must not overwrite an input artifact")
	ledger = read_jsonl(ledger_path, LedgerRecord)
	lengths = LengthChecker(config_path) if config_path else None
	decisions: dict[str, Decision] = {}
	for done, record in enumerate(ledger, 1):
		if record.quality.status == "rejected":
			decision = Decision(
				"rejected", ["collector"], record.quality.rejection_reason or "collector rejected"
			)
		else:
			decision = Decision("accepted", ["nonempty"])
		if decision.status == "accepted" and lengths:
			length = lengths.length(record.prompt, record.completion)
			if length > lengths.max_length:
				decision = Decision(
					"rejected", ["length"], f"{length} tokens exceeds {lengths.max_length}"
				)
			else:
				decision = Decision("accepted", [*decision.checks, "length"])
		decisions[record.example_id] = decision
		if progress:
			progress("Checking response usability", done, len(ledger))

	reviews: list[ReviewRecord] = []
	statuses: Counter[str] = Counter()
	reasons: Counter[str] = Counter()
	accepted_by_task: Counter[str] = Counter()
	for record in ledger:
		decision = decisions[record.example_id]
		review = ReviewRecord(
			example_id=record.example_id,
			status="accepted" if decision.status == "accepted" else "rejected",
			checks=decision.checks,
			reviewer=REVIEWER,
			reason=decision.reason,
		)
		if decision.reason:
			reasons[f"{decision.checks[0]}: {re.sub(r'\d+', 'N', decision.reason)}"] += 1
		reviews.append(review)
		statuses[review.status] += 1
		if review.status == "accepted":
			accepted_by_task[record.task] += 1
	write_jsonl(output_path, reviews)
	summary = {
		"reviewer": REVIEWER,
		"records": len(ledger),
		"status": dict(statuses),
		"rejection_reasons": dict(reasons.most_common()),
		"accepted_by_task": dict(sorted(accepted_by_task.items())),
		"length_checked": lengths is not None,
	}
	output_path.with_suffix(".summary.json").write_text(
		json.dumps(summary, indent="\t", sort_keys=True) + "\n", encoding="utf-8"
	)
	return summary
