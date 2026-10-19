"""Join shared-student capability, supervision, campaign costs, controls, and utility."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from distillation.benchmarks.models import save_json
from distillation.io import canonical_json, read_jsonl
from distillation.schemas import GatewayRequest, LedgerRecord, SFTRecord


def supervision(directory: Path, domains: list[str], schedule: Path, *, gold: bool) -> None:
	"""Retain example lineage after usability and duplicate exclusions, with measured tokens."""
	prepared = read_jsonl(directory / "train.jsonl", SFTRecord)
	keys = {canonical_json([m.model_dump() for m in row.prompt]) for row in prepared}
	rows = read_jsonl(schedule, GatewayRequest)
	selected = [
		r for r in rows if canonical_json([m.model_dump() for m in r.prompt.prompt]) in keys
	]
	selected_ids = {r.prompt.example_id for r in selected}
	records = [] if gold else read_jsonl(directory / "ledger.jsonl", LedgerRecord)
	counts: dict[str, Any] = {}

	def tokens(records: list[LedgerRecord]) -> int | None:
		if gold or any(r.usage.output_tokens is None for r in records):
			return None
		return sum(int(r.usage.output_tokens or 0) for r in records)

	for domain in domains:
		received = [r for r in records if r.task == domain]
		usable = [r for r in received if r.example_id in selected_ids]
		counts[domain] = {
			"released_responses": None if gold else len(received),
			"released_output_tokens": tokens(received),
			"prepared_examples": sum(r.domain == domain for r in selected),
			"prepared_output_tokens": tokens(usable),
		}
	save_json(
		directory / "supervision.json",
		{
			"domains": counts,
			"lineage": [
				{
					"example_id": r.prompt.example_id,
					"domain": r.domain,
					"source": r.prompt.source.model_dump(mode="json"),
				}
				for r in selected
			],
			"token_unit": "teacher-reported output tokens; "
			"student-tokenizer totals are in training preflight",
		},
	)


def profile_rates(rows: list[GatewayRequest], decisions: Path, levels: list[int]) -> dict[str, Any]:
	"""Account-horizon escalation denominators, retaining unscored short horizons."""
	profiles = {r.account: r.domain_profile for r in rows}
	horizons: dict[str, int] = defaultdict(int)
	peaks: dict[tuple[str, int], int] = {}
	scored: set[tuple[str, int]] = set()
	for line in decisions.read_text().splitlines():
		decision = json.loads(line)
		account = decision["account"]
		if decision["attempt"] == 1:
			horizons[account] += 1
		key = account, horizons[account]
		peaks[key] = max(peaks.get(key, 0), decision["level"])
		if decision["window_completed"]:
			scored.add(key)
	return {
		str(profile): {
			"horizons": len(keys),
			"scored_horizons": len(set(keys) & scored),
			"reached_rate": {
				f"L{level}": sum(peaks[k] >= level for k in keys) / len(keys)
				if level in levels
				else None
				for level in range(1, 5)
			},
		}
		for profile in set(profiles.values())
		if (keys := [k for k in peaks if profiles[k[0]] == profile])
	}


def campaign_results(
	directory: Path, seed: int, budget: int, arms: tuple[str, ...], domains: list[str]
) -> list[dict[str, Any]]:
	def load(path: Path) -> Any:
		return json.loads(path.read_text())

	baseline = load(directory / "undefended/scores.json")
	teacher = load(directory / "teacher/teacher-scores.json")
	teacher_after = load(directory / "teacher/teacher-after-scores.json")
	results = []
	for arm in arms:
		campaign = directory / arm / "campaign"
		scores = load(campaign / "scores.json")
		events = load(campaign / "campaign.json")
		prepared = load(campaign / "supervision.json")
		for domain in domains:
			base = baseline[domain]["student_before"]
			gain = baseline[domain]["student_after"] - base
			protected_gain = scores[domain]["student_after"] - scores[domain]["student_before"]
			results.append(
				{
					"campaign": str(directory),
					"seed": seed,
					"budget": budget,
					"arm": arm,
					"domain": domain,
					"base_score": base,
					"undefended_score": baseline[domain]["student_after"],
					"defended_score": scores[domain]["student_after"],
					"undefended_gain": gain,
					"defended_gain": protected_gain,
					"retained_gain_fraction": protected_gain / gain if gain > 0 else None,
					"gain_removed_fraction": 1 - protected_gain / gain if gain > 0 else None,
					"evaluation_ids": scores[domain]["evaluation_ids"],
					"undefended_evaluation_ids": baseline[domain]["evaluation_ids"],
					"teacher_score": teacher[domain]["teacher"],
					"teacher_after_score": teacher_after[domain]["teacher_after"],
					"http": events["domains"][domain],
					"supervision": prepared["domains"][domain],
					"utility": str(directory / arm / "utility.json"),
					"controls": {
						name: {
							"tokens": load(campaign / name / "control.json")[domain],
							"scores": load(campaign / name / "scores.json")[domain],
						}
						for name in ("matched-volume", "matched-prefix")
					},
				}
			)
	return results
