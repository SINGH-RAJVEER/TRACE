"""Development-only output-cap pilot shared by coding and GSM8K."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from anyio import Path as AsyncPath

from distillation.benchmarks.models import load_suite, save_json
from distillation.schemas import TeacherGenerationRequest
from distillation.tasks import _select
from distillation.teacher import Teacher


def _append(path: Path, row: dict[str, Any]) -> None:
	with path.open("a", encoding="utf-8") as destination:
		destination.write(json.dumps(row) + "\n")


async def run_cap_pilot(
	*,
	task_dir: Path,
	output_dir: Path,
	teacher: Teacher,
	max_output_tokens: int,
	limit: int = 200,
	seed: int = 42,
) -> dict[str, Any]:
	"""Make one request per selected development item, without retries or training exports."""
	try:
		if limit < 1 or max_output_tokens < 1:
			raise ValueError("pilot limit and output cap must be positive")
		suite = load_suite(task_dir / "development-suite.json")
		items = _select(suite.items, limit, seed)
		await AsyncPath(output_dir).mkdir(parents=True, exist_ok=False)
		contract = {
			"complete": False,
			"development_suite_sha256": suite.sha256,
			"task": items[0].benchmark,
			"seed": seed,
			"max_output_tokens": max_output_tokens,
			"n": len(items),
			"temperature": 0,
			"enable_thinking": False,
			"retry_policy": "none",
			"purpose": "development_output_cap_only",
		}
		save_json(output_dir / "pilot.json", contract)
		statuses: Counter[str] = Counter()
		models: set[str] = set()
		for item in items:
			try:
				generation = await teacher.generate(
					TeacherGenerationRequest(
						messages=item.messages,
						max_output_tokens=max_output_tokens,
						seed=seed,
						temperature=0,
						enable_thinking=False,
					)
				)
			except Exception:
				statuses["error"] += 1
				_append(output_dir / "responses.jsonl", {"item_id": item.id, "status": "error"})
			else:
				status = generation.finish_reason or "missing"
				statuses[status] += 1
				models.add(generation.model)
				_append(
					output_dir / "responses.jsonl",
					{
						"item_id": item.id,
						"status": status,
						"response": generation.message.content,
						"served_model": generation.model,
						"usage": generation.usage.model_dump(),
						"request_id": generation.request_id,
					},
				)
		errors = statuses.get("error", 0)
		responses = len(items) - errors
		length = statuses["length"] + statuses["max_tokens"]
		fraction = length / responses if responses else None
		terminal = {"stop", "eos", "end_turn", "completed", "length", "max_tokens"}
		conclusive = not errors and len(models) == 1 and not (statuses.keys() - terminal)
		decision = (
			"inconclusive"
			if not conclusive
			else "raise_cap"
			if fraction is not None and fraction > 0.02
			else "cap_passed"
		)
		summary = contract | {
			"complete": True,
			"statuses": dict(statuses),
			"responses": responses,
			"length": length,
			"length_fraction": fraction,
			"length_denominator": "successful generation responses, including length finishes",
			"served_models": sorted(models),
			"decision": decision,
			"threshold": 0.02,
		}
		save_json(output_dir / "pilot.json", summary)
		return summary
	finally:
		await teacher.close()
