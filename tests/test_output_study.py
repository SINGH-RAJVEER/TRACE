"""CPU fixtures for output arms and quota comparisons; no measured teacher/student results."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from distillation.acquisition_manifest import build_manifest, write_manifest
from distillation.output_sets import (
	ADAPTIVE_ARMS,
	CONTROL_ARMS,
	DENSE_INSTRUCTION,
	Adapters,
	Recipe,
	Rewrite,
	build_arm,
	loss_values,
	select_training_set,
	tensor_loss,
	write_training_set,
)
from distillation.quota_comparison import comparison_report
from distillation.quota_matching import match_quotas
from distillation.schemas import LedgerRecord
from test_acquisition_manifest import ledger, record


class CharacterCodec:
	def encode(self, text: str) -> Sequence[int]:
		return [ord(c) for c in text]

	def decode(self, tokens: Sequence[int]) -> str:
		return "".join(chr(t) for t in tokens)


def answer(record: LedgerRecord, text: str) -> str | None:
	return "#### " + text.rsplit("#### ", 1)[1].strip() if "#### " in text else None


def valid(record: LedgerRecord, text: str) -> bool:
	return text.endswith("#### 42")


@pytest.fixture
def cache(tmp_path: Path) -> Path:
	rows = [record(f"e{i}", task="gsm8k", tokens=25, group_id=f"family{i}") for i in range(6)]
	for row in rows:
		row["completion"][0]["content"] = "A clean explanation. #### 42"
	rows.append(record("heldout", split="test"))
	path = tmp_path / "manifest.json"
	write_manifest(
		build_manifest([ledger(tmp_path / "cache.jsonl", rows)], response_policy="clean"), path
	)
	return path


@pytest.fixture
def adapters() -> Adapters:
	return Adapters(
		CharacterCodec(),
		answer,
		valid,
		{
			"codec": "fixture-characters-v1",
			"answer_extractor": "fixture-v1",
			"validator": "fixture-v1",
			"rewriter": "fake-same-teacher-v1",
		},
		lambda r, text, instruction: Rewrite("Formal proof. #### 42", 1, 0.0, 0.01),
	)


def development() -> dict[str, Any]:
	return {
		"split": "development",
		"population_id": "fixture-clients",
		"schedule_sha256": "fixture-schedule",
		"tolerance": 0.02,
		"candidate": {"arm": "dense_locked", "loss_by_stratum": {"casual": 0.1, "specialist": 0.2}},
		"settings": [
			{
				"policy": "quota",
				"request_quota": 4,
				"loss_by_stratum": {"casual": 0.1, "specialist": 0.2},
			},
			{
				"policy": "quota_tokens",
				"request_quota": 4,
				"output_token_budget": 200,
				"loss_by_stratum": {"casual": 0.1, "specialist": 0.2},
			},
		],
	}


def test_matching_requires_one_observed_setting_for_every_stratum() -> None:
	dev = development()
	dev["settings"][0]["loss_by_stratum"]["specialist"] = 0.3
	dev["settings"].append(
		{
			"policy": "quota",
			"request_quota": 7,
			"loss_by_stratum": {"casual": 0.2, "specialist": 0.2},
		}
	)
	result = match_quotas(dev)
	assert result["baselines"]["quota"]["status"] == "not matched"
	assert result["baselines"]["quota"]["selected"] is None
	assert result["baselines"]["quota_tokens"]["status"] == "matched"
	assert not result["interpolated"]
	assert (
		result["baselines"]["quota"]["trials"][0]["strata"]["specialist"]["status"] == "not matched"
	)
	dev["split"] = "test"
	with pytest.raises(ValueError, match="development"):
		match_quotas(dev)


def test_answer_lock_records_fallback_and_never_releases_clean_draft(
	cache: Path, adapters: Adapters
) -> None:
	seen = []

	def wrong(r: LedgerRecord, text: str, instruction: str) -> Rewrite:
		seen.append(instruction)
		return Rewrite("Formal proof. #### 99", 1, 0.05, 0.2)

	bad = replace(adapters, rewrite=wrong)
	withheld = build_arm(cache, Recipe("dense_locked"), bad)
	assert seen == [DENSE_INSTRUCTION] * 6
	assert withheld["fallbacks"] == 6
	assert withheld["released_tokens"] == 0
	assert all(row["sft"] is None and row["fallback"] == "withhold" for row in withheld["rows"])
	assert all(row["failure"].startswith("answer mismatch") for row in withheld["rows"])
	assert withheld["rows"][0]["rewriter_cost"] == {"queries": 1, "money": 0.05, "seconds": 0.2}
	final = build_arm(cache, Recipe("dense_locked", fallback="final_answer"), bad)
	assert all(row["released_text"] == "#### 42" and row["answer_equal"] for row in final["rows"])
	assert all(row["released_tokens"] == 7 for row in final["rows"])
	with pytest.raises(ValueError, match="pluggable rewriter"):
		build_arm(cache, Recipe("dense_locked"), replace(adapters, rewrite=None))


def test_validity_failure_uses_recorded_fallback(cache: Path, adapters: Adapters) -> None:
	bad = replace(
		adapters,
		rewrite=lambda r, t, i: Rewrite("Invalid proof. #### 42", 1, 0.0, 0.01),
		valid=lambda r, t: t == "#### 42",
	)
	arm = build_arm(cache, Recipe("compact_verified", fallback="final_answer"), bad)
	assert all(
		r["failure"] == "task validity failed" and r["fallback"] == "final_answer"
		for r in arm["rows"]
	)
	assert all(r["task_valid"] and r["released_text"] == "#### 42" for r in arm["rows"])


def test_equal_token_selection_and_all_controls(cache: Path, adapters: Adapters) -> None:
	clean = build_arm(cache, Recipe("clean"), adapters)
	short = build_arm(cache, Recipe("final_answer_only"), adapters)
	clean_set = select_training_set(clean, by="tokens", budget=56, seed=42)
	short_set = select_training_set(short, by="tokens", budget=56, seed=42)
	assert clean_set["selection"]["count"] == 2
	assert short_set["selection"]["count"] == 6
	assert short_set["selection"]["output_tokens"] == 42
	assert short_set["selection"]["exhausted"]
	assert all(r["example_id"] != "heldout" for r in clean["rows"])
	queries = select_training_set(short, by="queries", budget=2, seed=42)
	assert queries["selection"]["example_ids"] == clean_set["selection"]["example_ids"]
	for name in CONTROL_ARMS:
		control = build_arm(
			cache, Recipe(name, max_tokens=10 if name == "clean_truncated" else None), adapters
		)
		selected = select_training_set(control, by="queries", budget=2, seed=1)
		assert selected["trainable_rows"] == 2
		assert selected["selection"]["output_tokens"] == (20 if name == "clean_truncated" else 56)
	for name in ADAPTIVE_ARMS:
		adaptive = build_arm(
			cache,
			Recipe(name, loss="filter", loss_threshold=2)
			if name == "protected_loss"
			else Recipe(name),
			adapters,
			protected=clean,
		)
		assert adaptive["parent_digest"] == clean["digest"]
		assert adaptive["released_tokens"] == clean["released_tokens"]
		assert all(r["answer_equal"] for r in adaptive["rows"])


def test_loss_recipes_and_tensor_gradients() -> None:
	assert loss_values([1, 3], Recipe("clean", loss="filter", loss_threshold=2)) == [1, 0]
	assert loss_values([1, 3], Recipe("clean", loss="clip", loss_threshold=2)) == [1, 2]
	with pytest.raises(ValueError, match="threshold"):
		loss_values([1], Recipe("clean", loss="filter"))
	torch = pytest.importorskip("torch")
	values = torch.tensor([1.0, 3.0], requires_grad=True)
	loss = tensor_loss(values, Recipe("clean", loss="clip", loss_threshold=2))
	loss.backward()
	assert values.grad.tolist() == [0.5, 0.0]


def comparison_fixture(
	tmp_path: Path, cache: Path, adapters: Adapters
) -> tuple[dict[str, Any], Path]:
	manifest_digest = json.loads(cache.read_text())["digest"]
	gate = tmp_path / "gate.json"
	gate.write_text(
		json.dumps(
			{
				"version": "clean-baseline-gate-v1",
				"status": "pass",
				"manifest_digest": manifest_digest,
			}
		)
	)
	gate_sha = hashlib.sha256(gate.read_bytes()).hexdigest()
	decl: dict[str, Any] = {
		"development": development(),
		"manifest_digest": manifest_digest,
		"evidence_kind": "fixture",
		"adaptive_arms": ["answer_attack"],
		"seeds": [1, 2, 3],
		"reference_queries": 2,
		"views": {},
	}
	for view, by in (("equal_query", "queries"), ("equal_token", "tokens")):
		entries = {}
		for label in ["dense_locked", "quota", "quota_tokens", "answer_attack", *CONTROL_ARMS]:
			admissions = None
			if label in {"quota", "quota_tokens"}:
				admissions = {
					"manifest_digest": manifest_digest,
					"policy": label,
					"accepted_example_ids": [f"e{i}" for i in range(6)],
					"evidence_path": str(cache),
					"evidence_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
					"settings": {
						"request_quota": 4,
						**({"output_token_budget": 200} if label == "quota_tokens" else {}),
					},
				}
			arm_name = "final_answer_extraction" if label == "answer_attack" else label
			arm = build_arm(
				cache,
				Recipe(arm_name, max_tokens=10 if label == "clean_truncated" else None),
				adapters,
				admissions=admissions,
				protected=build_arm(cache, Recipe("dense_locked"), adapters)
				if label == "answer_attack"
				else None,
			)
			selected = select_training_set(
				arm,
				by="queries" if by == "queries" else "tokens",
				budget=2 if by == "queries" else 28,
				seed=42,
			)
			training = tmp_path / f"{view}-{label}.json"
			write_training_set(selected, training)
			students = {}
			for seed in decl["seeds"]:
				student = tmp_path / f"student-{view}-{label}-{seed}.json"
				interval = [0.6, 0.7] if label in {"quota", "quota_tokens"} else [0.1, 0.2]
				student.write_text(
					json.dumps(
						{
							"seed": seed,
							"training_set_digest": selected["digest"],
							"gate_report_sha256": gate_sha,
							"transfer": {
								"normalized_transfer": {
									"defended": {"value": sum(interval) / 2, "interval": interval}
								},
								"headroom": {"usable": True},
								"clean_gain": {"usable": True},
							},
						}
					)
				)
				students[str(seed)] = str(student)
			entries[label] = {
				"training_set": str(training),
				"students": students,
				"short_output_cost": {
					"extra_queries": max(0, selected["selection"]["count"] - 2),
					"money": 0,
					"seconds": 0,
				},
			}
		decl["views"][view] = entries
	return decl, gate


def test_comparison_refuses_single_seed_missing_students_gate_and_unmatched(
	tmp_path: Path,
	cache: Path,
	adapters: Adapters,
) -> None:
	decl, gate = comparison_fixture(tmp_path, cache, adapters)
	decl["seeds"] = [1]
	report = comparison_report(decl)
	assert report["verdict"] == "no result"
	assert any("screening only" in reason for reason in report["refusal_reasons"])
	assert any("missing clean-baseline gate" in reason for reason in report["refusal_reasons"])
	assert report["views"]["equal_query"]["arms"]["dense_locked"]["status"] == "no result"
	del decl["views"]["equal_query"]["quota"]["students"]["1"]
	decl["development"]["settings"][0]["loss_by_stratum"]["casual"] = 0.5
	report = comparison_report(decl, gate_report=gate)
	assert report["views"]["equal_query"]["arms"]["quota"]["status"] == "no result"
	assert "quota: not matched" in report["refusal_reasons"]
	del decl["views"]["equal_token"]
	assert (
		"missing equal_token view" in comparison_report(decl, gate_report=gate)["refusal_reasons"]
	)


def test_comparison_uses_worst_adaptive_interval_and_refuses_misjoined_data(
	tmp_path: Path,
	cache: Path,
	adapters: Adapters,
) -> None:
	decl, gate = comparison_fixture(tmp_path, cache, adapters)
	student_path = Path(decl["views"]["equal_query"]["answer_attack"]["students"]["2"])
	student = json.loads(student_path.read_text())
	student["transfer"]["normalized_transfer"]["defended"]["interval"] = [0.4, 0.8]
	student_path.write_text(json.dumps(student))
	report = comparison_report(decl, gate_report=gate)
	comparison = report["views"]["equal_query"]["per_seed_comparison"]["2"]
	assert comparison["strongest_declared_attacker_arm"] == "answer_attack"
	assert not comparison["clears"]["quota"]
	assert report["verdict"] == "no result"
	student["training_set_digest"] = "unrelated"
	student_path.write_text(json.dumps(student))
	report = comparison_report(decl, gate_report=gate)
	assert report["views"]["equal_query"]["arms"]["answer_attack"]["status"] == "no result"
	gate.write_text(
		json.dumps(
			{"version": "clean-baseline-gate-v1", "status": "pass", "manifest_digest": "other"}
		)
	)
	assert comparison_report(decl, gate_report=gate)["gate"] is None


def test_complete_fixture_report_exercises_success_rule_without_measurement_claim(
	tmp_path: Path,
	cache: Path,
	adapters: Adapters,
) -> None:
	# Equal-length defense output makes exact equal-token prefixes feasible in this fixture.
	adapters = replace(adapters, rewrite=lambda r, t, i: Rewrite(t, 1, 0, 0.01))
	decl, gate = comparison_fixture(tmp_path, cache, adapters)
	capacity = tmp_path / "capacity-fixture.json"
	schedule = tmp_path / "benign-fixture.jsonl"
	schedule.write_text("fixture only\n")
	capacity.write_text(
		json.dumps(
			{
				"version": "teacher-capacity-v1",
				"passed": True,
				"schedule_sha256": hashlib.sha256(schedule.read_bytes()).hexdigest(),
				"slo": {},
				"checks": {
					k: True
					for k in (
						"offered_throughput",
						"completed_throughput",
						"latency_p95",
						"errors",
						"timeouts",
						"queue_end",
						"queue_growth",
						"undefended",
					)
				},
			}
		)
	)
	decl.update(capacity_evidence=str(capacity), benign_schedule=str(schedule))
	decl["development"]["schedule_sha256"] = hashlib.sha256(schedule.read_bytes()).hexdigest()
	report = comparison_report(decl, gate_report=gate)
	assert report["refusal_reasons"] == [
		"fixture or undeclared evidence cannot support a measurement verdict"
	]
	assert report["verdict"] == "no result"
	assert all(result["comparable"] for result in report["utility_by_stratum"].values())
	assert all(
		all(seed["clears"].values())
		for view in report["views"].values()
		for seed in view["per_seed_comparison"].values()
	)


def test_matching_decimal_tolerance_boundary() -> None:
	dev = development()
	dev["settings"][0]["loss_by_stratum"]["casual"] = 0.08
	assert match_quotas(dev)["baselines"]["quota"]["status"] == "matched"


def test_adaptive_callback_receives_protected_draft_and_keeps_acquisition_tokens(
	cache: Path,
	adapters: Adapters,
) -> None:
	protected = build_arm(cache, Recipe("final_answer_only"), adapters)
	seen = []

	def cleanup(r: LedgerRecord, text: str, instruction: str) -> Rewrite:
		seen.append((r.completion[0].content, text))
		return Rewrite("Longer reconstructed explanation. #### 42", 1, 0, 0.01)

	adaptive = build_arm(
		cache,
		Recipe("cleaned_explanations"),
		replace(adapters, rewrite=cleanup),
		protected=protected,
	)
	assert seen == [("#### 42", "#### 42")] * 6
	assert adaptive["released_tokens"] == 42
	assert all(row["training_tokens"] > row["released_tokens"] for row in adaptive["rows"])
