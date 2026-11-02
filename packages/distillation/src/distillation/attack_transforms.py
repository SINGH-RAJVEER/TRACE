"""Opt-in, reproducible transformations of frozen gateway inventories."""

import random
import re
from collections.abc import Callable
from typing import Any

from distillation.schemas import GatewayRequest, Message

Rewriter = Callable[[str], str]


def offline_rewrite(text: str) -> str:
	"""Apply small deterministic paraphrases without a model or network access."""
	for old, new in (
		("What is", "Determine"),
		("How many", "Find the number of"),
		("Please", "Kindly"),
	):
		text = re.sub(r"\b" + re.escape(old) + r"\b", new, text)
	return text


def transform_inventory(
	rows: list[GatewayRequest],
	*,
	rewriter: Rewriter | None = None,
	rewriter_id: str | None = None,
	user_prefix: str | None = None,
	assistant_prefill: str | None = None,
	bursts: bool = False,
	seed: int = 42,
) -> tuple[list[GatewayRequest], dict[str, Any]]:
	if bool(rewriter) != bool(rewriter_id):
		raise ValueError("rewriter and its identity must be supplied together")
	if user_prefix is not None and not user_prefix.strip():
		raise ValueError("user prefix must not be blank")
	if assistant_prefill is not None and not assistant_prefill.strip():
		raise ValueError("assistant prefill must not be blank")
	rng = random.Random(seed)
	time = 0.0
	result = []
	for index, row in enumerate(rows):
		messages = []
		for message in row.prompt.prompt:
			content = message.content
			if message.role == "user":
				if rewriter:
					content = rewriter(content)
				if user_prefix:
					content = user_prefix + "\n\n" + content
			messages.append(Message(role=message.role, content=content))
		if assistant_prefill:
			messages.append(Message(role="assistant", content=assistant_prefill))
		if bursts and index:
			# Sessions of six requests, with a daily overnight gap.
			if index % 24 == 0:
				time = index // 24 * 86400 + rng.uniform(0, 600)
			elif index % 6 == 0:
				time += rng.uniform(600, 1800)
			else:
				time += rng.uniform(2, 12)
		prompt = row.prompt.model_copy(update={"prompt": messages})
		result.append(
			row.model_copy(update={"prompt": prompt, "time": time if bursts else row.time})
		)
	settings: dict[str, Any] = {}
	if rewriter:
		settings["rewriter"] = {
			"identity": rewriter_id,
			"offline_compute": "deterministic CPU text rewriting",
		}
	if user_prefix:
		settings["user_prefix"] = user_prefix
	if assistant_prefill:
		settings["assistant_prefill"] = assistant_prefill
	if bursts:
		settings["arrival_pattern"] = {
			"method": "seeded sessions and long idle gaps v1",
			"seed": seed,
		}
	return result, settings


def replay_sessions(
	rows: list[GatewayRequest], benign: list[GatewayRequest], budget: int
) -> tuple[list[GatewayRequest], list[str]]:
	"""Interleave whole recorded accounts, preserving their relative arrival times."""
	if budget < 1 or not benign:
		raise ValueError("replay requires a positive budget and a nonempty benign workload")
	groups: dict[str, list[GatewayRequest]] = {}
	for row in benign:
		groups.setdefault(row.account, []).append(row)
	selected: list[GatewayRequest] = []
	for account in sorted(groups):
		session = sorted(groups[account], key=lambda row: row.time)
		if len(selected) + len(session) <= budget:
			selected.extend(session)
	if not selected:
		raise ValueError("replay budget cannot fit a complete benign session")
	attack = rows[: max(0, len(rows) - len(selected))]
	if len({r.request_id for r in attack + selected}) != len(attack) + len(selected):
		raise ValueError("replayed and attack request IDs must be distinct")
	if len({r.prompt.example_id for r in attack + selected}) != len(attack) + len(selected):
		raise ValueError("replayed and attack example IDs must be distinct")
	return sorted(attack + selected, key=lambda row: (row.time, row.request_id)), [
		r.request_id for r in selected
	]


def batch_inventory(
	rows: list[GatewayRequest], size: int
) -> tuple[list[GatewayRequest], dict[str, list[dict[str, Any]]]]:
	"""Batch same-account and domain single-user templates into JSON-array requests."""
	import json

	if not 2 <= size <= 4:
		raise ValueError("batch size must be between 2 and 4")
	result: list[GatewayRequest] = []
	members: dict[str, list[dict[str, Any]]] = {}
	index = 0
	while index < len(rows):
		first = rows[index]
		group = [first]
		while index + len(group) < len(rows) and len(group) < size:
			candidate = rows[index + len(group)]
			if candidate.account != first.account or candidate.domain != first.domain:
				break
			group.append(candidate)
		if any(len(row.prompt.prompt) != 1 or row.prompt.prompt[0].role != "user" for row in group):
			raise ValueError("batching requires single-user-message templates")
		if len(group) == 1:
			result.append(first)
		else:
			payload = [row.prompt.prompt[0].content for row in group]
			message = Message(
				role="user",
				content="Answer each item in order. Return only a JSON array of answer strings.\n"
				+ json.dumps(payload, ensure_ascii=False),
			)
			prompt = first.prompt.model_copy(update={"prompt": [message]})
			result.append(
				first.model_copy(
					update={
						"prompt": prompt,
						"max_output_tokens": min(row.max_output_tokens for row in group),
					}
				)
			)
			members[first.request_id] = [row.model_dump(mode="json") for row in group]
		index += len(group)
	return result, members
