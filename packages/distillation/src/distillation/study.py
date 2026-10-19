"""End-to-end task extraction, including the protected teacher and second students."""

import hashlib
import json
import os
import secrets
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from distillation.benchmarks.models import (
	EvaluationConfig,
	EvaluationRun,
	GenerationConfig,
	ModelConfig,
	Slot,
	load_config,
	load_suite,
	save_json,
)
from distillation.io import read_jsonl, write_jsonl
from distillation.launcher import (
	DEFAULT_TEACHER_MODEL,
	DEFAULT_TEACHER_REVISION,
	LOG_DIR,
	LaunchError,
	TaskPipeline,
	_toml,
	_trained,
)
from distillation.schemas import LedgerRecord, PromptRecord, SFTRecord
from distillation.workloads import DEFAULT_CONCENTRATED_SHARE

# The defended services of Table VI and the primary contrast: harness policy and accumulator.
DEFENSES: dict[str, tuple[str, str]] = {
	"full": ("mmd", "paper"),
	"instantaneous": ("mmd", "instantaneous"),
	"quota": ("quota", "paper"),
}
# Resumed passes over an undefended campaign that lost prompts to teacher failures.
UNDEFENDED_PASSES = 3
# Each task's success target recovers this fraction of the selected recipe's development gain.
TARGET_GAIN_FRACTION = 0.5
# The accumulators the paper reports: the full policy, its baseline, and the memory ablations.
# The e-value accumulators stay in the library but out of study calibrations and reports.
DECLARED_ACCUMULATORS = ("paper", "instantaneous", "paper_no_memory", "paper_no_decay")


def freeze(path: Path, value: dict[str, Any]) -> None:
	"""Prevent an existing run directory from silently changing its protocol."""
	if path.exists():
		if json.loads(path.read_text()) != value:
			raise LaunchError(f"{path} changed; select a new --run-dir")
	else:
		save_json(path, value)


def capability_target(selection: dict[str, Any]) -> dict[str, Any]:
	"""A per-task success target on the task's own score scale, from development scores only.

	A single fixed score cannot serve every task: base and teacher scores differ by task, and
	a target above what undefended transfer reaches makes every cost-to-target "not reached".
	"""
	winner = next(
		trial
		for trial in selection["trials"]
		if {"learning_rate": trial["learning_rate"], "epochs": trial["epochs"]}
		== selection["selected"]
	)
	evidence = selection.get("qualification", winner)
	base, score = evidence["base_score"], evidence["score"]
	return {
		"target_score": base + TARGET_GAIN_FRACTION * (score - base),
		"gain_fraction": TARGET_GAIN_FRACTION,
		"development_base_score": base,
		"development_score": score,
		"source": "selected recipe on untouched development qualification suite"
		if "qualification" in selection
		else "legacy selected development scores",
	}


def _replayed_horizons(
	decisions: Path, client_of: list[str]
) -> dict[str, tuple[list[int], list[int], list[float]]]:
	"""Per client type: each horizon's peak level and completed windows, and window p-values."""
	groups: dict[str, tuple[list[int], list[int], list[float]]] = defaultdict(lambda: ([], [], []))
	unit: dict[str, int] = {}
	with decisions.open(encoding="utf-8") as source:
		for line in source:
			record = json.loads(line)
			peaks, windows, p_values = groups[client_of[record["index"]]]
			if record["attempt"] == 1:
				unit[record["account"]] = len(peaks)
				peaks.append(0)
				windows.append(0)
			index = unit[record["account"]]
			peaks[index] = max(peaks[index], record["level"])
			if record["window_completed"]:
				windows[index] += 1
				if record["p_value"] is not None:
					p_values.append(record["p_value"])
	return dict(groups)


def _describe(peaks: list[int], windows: list[int], p_values: list[float]) -> dict[str, Any]:
	ordered = sorted(p_values)
	return {
		"horizons": len(peaks),
		"scored_horizons": sum(count > 0 for count in windows),
		"peak_levels": {f"L{level}": peaks.count(level) for level in range(5)},
		"windows": len(ordered),
		"p_value_mean": sum(ordered) / len(ordered) if ordered else None,
		"p_value_median": ordered[len(ordered) // 2] if ordered else None,
	}


class _MemoEncoder:
	"""Embeds each distinct text once, so several replays of one pool share the encoder cost."""

	def __init__(self, encoder: Any) -> None:
		self.encoder = encoder
		self.rows: dict[str, Any] = {}

	def encode(self, texts: list[str]) -> Any:
		import numpy as np

		missing = [text for text in dict.fromkeys(texts) if text not in self.rows]
		if missing:
			self.rows.update(zip(missing, self.encoder.encode(missing), strict=True))
		return np.stack([self.rows[text] for text in texts])

	def identity(self) -> dict[str, Any]:
		return dict(self.encoder.identity())


@dataclass
class Study:
	pipeline: TaskPipeline
	root: Path
	teacher_url: str
	token: str
	port: int
	upstream_url: str
	harness_port: int
	calibration_root: Path | None
	l4_action: Literal["refuse", "deceive"]
	references: dict[str, EvaluationRun] = field(default_factory=dict)
	arms: tuple[str, ...] = tuple(DEFENSES)
	# Share of benign source groups sent by specialist and repetitive clients; see workloads.
	concentrated_share: float = DEFAULT_CONCENTRATED_SHARE
	# Continue when no MMD defense separates the extractor from benign accounts, reporting
	# the study as a quota comparison rather than a detection result.
	allow_undetected: bool = False
	minimum_gain: dict[str, float] = field(default_factory=dict)

	@property
	def protected_url(self) -> str:
		return f"http://127.0.0.1:{self.harness_port}"

	@property
	def url(self) -> str:
		return self.teacher_url.rstrip("/") or f"http://127.0.0.1:{self.port}"

	def ensure_teacher(self) -> None:
		launcher = self.pipeline.launcher
		if self.teacher_url:
			launcher.wait_for_http(f"{self.url}/healthz", '"ready"', 60, "Waiting for teacher")
			launcher.smoke_test(self.url, self.token)
		else:
			launcher.start_teacher_stack(
				"127.0.0.1", self.port, self.upstream_url, LOG_DIR / "start-teacher.log"
			)

	def run_dir(self, task: str, seed: int, budget: int) -> Path:
		return self.root / task / f"s{seed}-b{budget}"

	def protected_dir(self, task: str, seed: int, budget: int, arm: str) -> Path:
		return self.run_dir(task, seed, budget) / "protected" / arm

	def development_suites(self, task: str) -> dict[str, Path]:
		from distillation.transfer import development_suites

		return development_suites(
			self.pipeline.task_dir(task) / "development-suite.json",
			self.root / task / "development-suites",
		)

	def evaluation(
		self,
		task: str,
		seed: int,
		budget: int,
		kind: str,
		*,
		arm: str | None = None,
		development: bool = False,
		development_split: str = "selection",
		adapter_override: Path | None = None,
		untrained: bool = False,
	) -> Path:
		with self.pipeline.config.open("rb") as source:
			training = tomllib.load(source)
		student = ModelConfig(
			backend="local",
			model=training["base_model"],
			revision=training["base_revision"],
			precision=training.get("precision", "bf16"),
			load_in_4bit=training.get("qlora", False),
			eos_token=training.get("eos_token"),
			chat_template_path=training.get("chat_template_path"),
		)
		directory = self.run_dir(task, seed, budget)
		adapter = (
			self.protected_dir(task, seed, budget, arm) / "adapter"
			if arm
			else self.pipeline.artifact_root / task / f"s{seed}-b{budget}" / kind
		)
		teacher = ModelConfig(
			backend="teacher_api",
			model=os.environ.get("TEACHER_UPSTREAM_MODEL") or DEFAULT_TEACHER_MODEL,
			revision=os.environ.get("TEACHER_MODEL_REVISION") or DEFAULT_TEACHER_REVISION,
			base_url=self.url,
			api_key_env="TEACHER_API_TOKEN" if self.token else "",
		)
		condition = f"protected-{arm}" if arm else kind
		if development:
			condition = f"development-{development_split}-{kind}"
		if adapter_override:
			adapter = adapter_override
		config = EvaluationConfig(
			dimensions={
				"task": task,
				"query_seed": seed,
				"training_seed": self.pipeline.training_seed_for(seed),
				"budget": budget,
				"condition": condition,
				"schedule": "uniform_without_replacement",
				"split": development_split if development else "test",
			},
			experiment=f"start-{task}-s{seed}-b{budget}-{condition}",
			suite_path=(
				self.development_suites(task)[development_split]
				if development
				else self.pipeline.task_dir(task) / "test-suite.json"
			),
			output_dir=directory / "benchmarks" / condition,
			generation=GenerationConfig(
				max_output_tokens=self.pipeline.manifest(task)["max_output_tokens"]
			),
			models={
				"teacher": teacher,
				"teacher_after": teacher,
				"student_before": student,
				"student_after": student.model_copy(
					update={"adapter_path": None if untrained else adapter}
				),
			},
		)
		path = directory / f"eval.{condition}.toml"
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(_toml(config.model_dump(mode="json")), encoding="utf-8")
		return path

	def benchmark(self, config: Path, slot: Slot) -> None:
		from distillation.benchmarks.reporting import discover_runs

		settings = load_config(config)
		suite = load_suite(settings.suite_path)
		coding_runtime: dict[str, str | int | float] = {}
		if any(item.metric == "pass_at_1" for item in suite.items):
			from coding.sandbox import runtime_protocol

			coding_runtime = runtime_protocol()
		runs, errors = discover_runs(settings.output_dir)
		if errors:
			raise LaunchError(f"invalid evaluation artifacts in {settings.output_dir}")
		if any(
			run.slot == slot
			and run.suite_sha256 == suite.sha256
			and run.model == settings.models[slot]
			and run.generation == settings.generation
			and run.dimensions == settings.dimensions
			and run.coding_runtime == coding_runtime
			for run in runs
		):
			return
		key = json.dumps(
			[
				slot,
				suite.sha256,
				settings.generation.model_dump(),
				settings.models[slot].model_dump(mode="json"),
				coding_runtime,
			],
			sort_keys=True,
		)
		if slot != "student_after" and key in self.references:
			# Identical reference inference is shared across budgets and conditions. Keep
			# the original ID, timestamps, and provenance, while joining the new comparison.
			run = self.references[key].model_copy(
				update={"experiment": settings.experiment, "dimensions": settings.dimensions}
			)
			save_json(settings.output_dir / f"suite-{suite.sha256}.json", suite)
			save_json(settings.output_dir / f"{run.id}.json", run)
			return
		self.pipeline.launcher.distill("benchmark", "run", "--config", str(config), "--slot", slot)
		if slot != "student_after":
			runs, _ = discover_runs(settings.output_dir)
			self.references[key] = max(
				(run for run in runs if run.slot == slot), key=lambda run: run.completed_at
			)

	def baseline_evaluations(self, slots: tuple[Slot, ...]) -> None:
		for task, budgets in self.pipeline.plan.items():
			for seed in self.pipeline.seeds:
				for budget in budgets:
					for kind in ["attacker", "gold"] if self.pipeline.gold else ["attacker"]:
						config = self.evaluation(task, seed, budget, kind)
						for slot in slots:
							self.benchmark(config, slot)
					# The same unprotected victim is the reference for both students. Protected
					# endpoint utility is measured separately by the paired traffic replay.
					for slot in slots:
						if slot in {"teacher", "teacher_after"}:
							for arm in self.arms:
								self.benchmark(
									self.evaluation(task, seed, budget, "attacker", arm=arm), slot
								)

	def pooled(self, task: str, role: str) -> list[Any]:
		"""One role's reproducible task clients, shared by calibration and utility scoring."""
		from distillation.workloads import clients, task_references, workload_summary

		prompts = read_jsonl(
			self.pipeline.task_dir(task) / "legitimate" / f"{role}.jsonl", PromptRecord
		)
		rows = clients(
			prompts,
			role,
			references=task_references(self.pipeline.task_dir(task)),
			concentrated=self.concentrated_share,
		)
		if role == "fitting":
			# Fit references on unique prompts; repeated clients belong only to calibration/test.
			rows = list({row.example_id: row for row in rows}.values())
		save_json(
			self.root / task / "workloads" / f"{role}.json",
			{
				"summary": workload_summary(rows, self.concentrated_share),
				"requests": [r.model_dump() for r in rows],
			},
		)
		return rows

	def calibrate(self, task: str) -> Path:
		from harness.calibration import (
			HarnessConfig,
			calibrate,
			load_calibration,
			requests_sha256,
			save_calibration,
			validate_separation,
		)
		from harness.encoder import BGEEncoder

		groups = {
			role: self.pooled(task, role)
			for role in ("fitting", "score_calibration", "policy_calibration")
		}
		validate_separation({**groups, "benign_test": self.pooled(task, "benign_test")})
		path = (
			self.calibration_root / task
			if self.calibration_root
			else self.root / task / "calibration"
		)
		if (path / "calibration.json").exists():
			existing, _ = load_calibration(path)
			# Older calibrations sliced article-ordered SQuAD pools or embedded whole turns.
			if (
				existing.config.task != task
				or set(existing.config.accumulators) != set(DECLARED_ACCUMULATORS)
				or any(
					existing.sha256.get(role) != requests_sha256(rows)
					for role, rows in groups.items()
				)
				or existing.config.aggregate_enabled
			):
				raise LaunchError(
					f"{path} was calibrated on different benign accounts or detector input; "
					"start a fresh --run-dir or remove that calibration"
				)
			return path
		self.pipeline.launcher.log(
			f"{task}: calibrating on synthetic task clients; rates apply to this workload"
		)
		# Small task pools cannot supply twenty 100-request references. Keep the window
		# fixed and record the supported reference count in the calibration artifact.
		count = min(20, len(groups["fitting"]) // 100)
		if not count:
			raise LaunchError(f"{task}: calibration needs at least 100 fitting prompts")
		self.pipeline.launcher.log(f"{task}: calibration window=100, reference_windows={count}")
		defaults = HarnessConfig().accumulators
		calibration, references = calibrate(
			config=HarnessConfig.model_validate(
				{
					"task": task,
					"reference_windows": count,
					"aggregate_enabled": False,
					"accumulators": {name: defaults[name] for name in DECLARED_ACCUMULATORS},
				}
			),
			encoder=BGEEncoder(device="cpu"),
			fitting=groups["fitting"],
			score_calibration=groups["score_calibration"],
			policy_calibration=groups["policy_calibration"],
			log=self.pipeline.launcher.log,
		)
		save_calibration(path, calibration, references)
		return path

	def detectability(self, task: str, path: Path, budget: int = 2000) -> dict[str, Any]:
		"""Development diagnostic only; never tune thresholds using attacker replay.

		Compares one extraction account with the held-out benign accounts, per client type,
		under each defense. MMD arms compare against benign horizons that complete a window.
		`mmd_signal` is True when some MMD defense escalates the extractor to a level that
		fewer than `level_targets[0]` of those benign horizons reach.
		"""
		from harness.calibration import BenignRequest, load_calibration
		from harness.encoder import BGEEncoder
		from harness.replay import replay

		budget = min(budget, int(self.pipeline.manifest(task)["roles"]["attacker"]))
		self.pipeline.budget(task, 42, budget)
		prompts = read_jsonl(
			self.pipeline.run_dir(task, 42, budget) / "prompts.jsonl", PromptRecord
		)
		requests = [
			BenignRequest(
				account="extractor",
				text=next(m.content for m in reversed(p.prompt) if m.role == "user"),
				time=float(i),
			)
			for i, p in enumerate(prompts)
		]
		self.benign_holdout(task, path)
		benign_types = [row.client_type or "unknown" for row in self.pooled(task, "benign_test")]
		calibration, scorer = load_calibration(path)
		target = calibration.config.level_targets[0]
		encoder = _MemoEncoder(BGEEncoder(device="cpu"))
		results = {}
		for arm in self.arms:
			policy, accumulator = DEFENSES[arm]
			output = self.root / task / "detectability" / f"{arm}.jsonl"
			summary = replay(
				calibration=calibration,
				scorer=scorer,
				requests=requests,
				output=output,
				policy=policy,
				accumulator=accumulator,
				encoder=encoder if policy == "mmd" else None,
			)
			benign = _replayed_horizons(
				self.root / task / "benign-test" / f"{arm}.decisions.jsonl", benign_types
			)
			extractor = _replayed_horizons(output, ["extractor"] * len(requests))["extractor"]
			peak = max(extractor[0])
			# MMD can only escalate horizons that complete a window; short accounts never do,
			# so they would dilute the benign comparison.
			benign_peaks = [
				level
				for peaks, windows, _ in benign.values()
				for level, count in zip(peaks, windows, strict=True)
				if count or policy != "mmd"
			]
			rate = (
				sum(level >= peak for level in benign_peaks) / len(benign_peaks)
				if peak and benign_peaks
				else 1.0
			)
			results[arm] = {
				**summary,
				"extractor": _describe(*extractor),
				"benign_by_client_type": {
					kind: _describe(*benign[kind]) for kind in sorted(benign)
				},
				"benign_rate_at_extractor_peak": rate,
				"escalates_beyond_benign": bool(peak) and rate < target,
			}
		mmd = [
			results[arm]["escalates_beyond_benign"]
			for arm in self.arms
			if DEFENSES[arm][0] == "mmd"
		]
		result = {
			"budget": budget,
			"seed": 42,
			"single_account": True,
			"purpose": "development diagnostic, not a power estimate",
			"concentrated_share": self.concentrated_share,
			"level_target": target,
			"calibration_sha256": hashlib.sha256(
				(path / "calibration.json").read_bytes()
			).hexdigest(),
			"mmd_signal": any(mmd) if mmd else None,
			"results": results,
		}
		save_json(self.root / task / "detectability.json", result)
		return result

	def require_signal(self, signals: dict[str, dict[str, Any]]) -> None:
		"""Stop before the matrix when no MMD defense separates the extractor from benign users."""
		missing = sorted(task for task, result in signals.items() if result["mmd_signal"] is False)
		save_json(
			self.root / "detectability-gate.json",
			{
				"undetected_tasks": missing,
				"allow_undetected": self.allow_undetected,
				"reading": "quota comparison" if missing else "detection",
			},
		)
		if missing and not self.allow_undetected:
			save_json(
				self.root / "status.json",
				{"status": "detectability_gate_failed", "tasks": missing},
			)
			raise LaunchError(
				f"{', '.join(missing)}: no MMD defense escalates the extraction account beyond "
				"held-out benign accounts; see detectability.json, or pass --allow-undetected "
				"to run the study as a quota comparison"
			)

	def calibration_report(self, task: str, path: Path) -> None:
		"""Record the per-task calibration settings the paper discloses in Section III-B."""
		from harness.calibration import load_calibration

		calibration, _ = load_calibration(path)
		save_json(
			self.root / task / "calibration-report.json",
			{
				"calibration": str(path),
				"serialization": calibration.serialization,
				"window_size": calibration.config.window_size,
				"reference_windows": calibration.config.reference_windows,
				"sigma": calibration.sigma,
				"counts": calibration.counts,
				"thresholds": {
					name: value.model_dump() for name, value in calibration.thresholds.items()
				},
				"baselines": {
					name: value.model_dump() for name, value in calibration.baselines.items()
				},
				"aggregate_monitor": "skipped" if calibration.aggregate is None else "fitted",
				"aggregate_skipped": calibration.aggregate_skipped,
			},
		)

	def benign_holdout(self, task: str, path: Path) -> None:
		"""Replay the held-out benign pool through every frozen defense: account FPR_1 to FPR_4.

		The accounts are pooled like the calibration accounts, so these rates check the
		pipeline on unseen requests and do not establish real-user false-positive rates.
		"""
		from harness.calibration import load_calibration
		from harness.encoder import BGEEncoder
		from harness.replay import replay

		directory = self.root / task / "benign-test"
		if all((directory / f"{arm}.json").exists() for arm in self.arms):
			return
		calibration, scorer = load_calibration(path)
		requests = self.pooled(task, "benign_test")
		encoder = _MemoEncoder(BGEEncoder(device="cpu"))
		for arm in self.arms:
			policy, accumulator = DEFENSES[arm]
			summary = replay(
				calibration=calibration,
				scorer=scorer,
				requests=requests,
				output=directory / f"{arm}.decisions.jsonl",
				policy=policy,
				accumulator=accumulator,
				encoder=encoder if policy == "mmd" else None,
			)
			save_json(directory / f"{arm}.json", summary)

	def start_harness(
		self, task: str, seed: int, calibration: Path, arm: str = "full"
	) -> dict[str, str]:
		actors = [
			"extractor",
			"warmup",
			*sorted({row.account for row in self.pooled(task, "benign_test")}),
		]
		credentials = self.root / task / f"s{seed}" / "credentials.json"
		if not credentials.exists():
			credentials.parent.mkdir(parents=True, exist_ok=True)
			# Create credentials with restricted permissions before writing their values.
			with open(
				credentials, "x", encoding="utf-8", opener=lambda p, f: os.open(p, f, 0o600)
			) as f:
				json.dump({actor: secrets.token_hex(32) for actor in actors}, f)
		keys: dict[str, str] = json.loads(credentials.read_text())
		accounts = credentials.with_name("accounts.json")
		save_json(
			accounts,
			{hashlib.sha256(key.encode()).hexdigest(): actor for actor, key in keys.items()},
		)
		policy, accumulator = DEFENSES[arm]
		decisions = self.root / task / f"s{seed}" / arm / "decisions.jsonl"
		decisions.parent.mkdir(parents=True, exist_ok=True)
		env = {
			"HARNESS_HOST": "127.0.0.1",
			"HARNESS_PORT": str(self.harness_port),
			"HARNESS_TEACHER_URL": self.url,
			"HARNESS_TEACHER_TOKEN": self.token,
			"HARNESS_CALIBRATION_DIR": str(calibration),
			"HARNESS_ACCOUNTS_FILE": str(accounts),
			"HARNESS_TRUST_ACTOR_HEADER": "false",
			"HARNESS_TEST_CLOCK": "false",
			"HARNESS_POLICY": policy,
			"HARNESS_ACCUMULATOR": accumulator,
			"HARNESS_THRESHOLDS": "calibrated",
			"HARNESS_L4_ACTION": self.l4_action,
			"HARNESS_TASK": task,
			"HARNESS_ENCODER_DEVICE": "cpu",
			"HARNESS_DECISION_LOG": str(decisions),
			"HARNESS_DECEPTION_KEY_FILE": str(self.root / "deception.key"),
		}
		log = LOG_DIR / f"start-harness-{task}-s{seed}-{arm}.log"
		process = self.pipeline.launcher.start(
			"harness-api", [sys.executable, "-m", "harness_api"], log, env
		)
		self.pipeline.launcher.wait_for_http(
			f"{self.protected_url}/healthz", '"ready"', 300, "Starting harness", process, log
		)
		return keys

	def traffic(
		self,
		task: str,
		seed: int,
		phase: str,
		keys: dict[str, str] | None = None,
		arm: str = "full",
	) -> None:
		import asyncio

		from distillation.utility import measure_utility
		from distillation.workloads import task_references

		directory = self.root / task / f"s{seed}" / arm
		asyncio.run(
			measure_utility(
				requests=self.pooled(task, "benign_test"),
				references=task_references(self.pipeline.task_dir(task)),
				url=self.url if phase == "before" else self.protected_url,
				keys={"default": self.token} if phase == "before" else keys or {},
				max_output_tokens=int(self.pipeline.manifest(task)["max_output_tokens"]),
				output=directory / f"utility-{phase}.json",
			)
		)

	def collect_undefended(self) -> None:
		"""Collect each seed's largest budget without a defense, under the campaign envelope.

		The undefended teacher holds no policy state, so an interrupted or failed campaign
		resumes on its original clock and retries prompts that lack a response. Training
		requires a response for every planned prompt; anything less stops the study.
		"""
		for task, budgets in self.pipeline.plan.items():
			for seed in self.pipeline.seeds:
				directory = self.root / task / f"s{seed}" / "undefended"
				prompts = self.pipeline.run_dir(task, seed, budgets[-1]) / "prompts.jsonl"
				for _ in range(UNDEFENDED_PASSES):
					self.pipeline.launcher.distill(
						"collect-protected",
						"--input",
						str(prompts),
						"--output",
						str(directory / "ledger.jsonl"),
						"--summary",
						str(directory / "campaign.json"),
						"--teacher-url",
						self.url,
						"--terms-review-id",
						self.pipeline.terms_review_id,
						"--max-output-tokens",
						str(self.pipeline.manifest(task)["max_output_tokens"]),
						"--max-requests",
						str(budgets[-1]),
						"--resumable",
						env={"TEACHER_API_TOKEN": self.token},
					)
					problem = self.undefended_gap(prompts, directory)
					# A deadline leaves no time, and a policy block means the wrong endpoint.
					if (
						problem is None
						or problem["termination"] == "deadline"
						or problem["blocked"]
					):
						break
				if problem is not None:
					save_json(
						self.root / "status.json",
						{"status": "undefended_baseline_incomplete", "task": task, "seed": seed}
						| problem,
					)
					if problem["blocked"]:
						remedy = "a policy refused requests; the baseline must use the bare teacher"
					elif problem["termination"] == "deadline":
						remedy = (
							"the declared deadline cannot carry this budget, and defended "
							"campaigns at the same budget would be cut off too; pass smaller "
							"--budgets"
						)
					else:
						remedy = "check the teacher, then rerun with the same --run-dir to resume"
					raise LaunchError(
						f"{task} seed {seed}: undefended baseline received "
						f"{problem['received']} of {problem['expected']} prompts "
						f"({problem['termination']}); {remedy}"
					)
				self.pipeline.run_ledgers[(task, seed)] = directory / "ledger.jsonl"

	@staticmethod
	def undefended_gap(prompts: Path, directory: Path) -> dict[str, Any] | None:
		"""None when every planned prompt has an undefended response, else what is missing."""
		summary = json.loads((directory / "campaign.json").read_text())
		wanted = {p.example_id for p in read_jsonl(prompts, PromptRecord)}
		have = {
			json.loads(line)["example_id"]
			for line in (directory / "ledger.jsonl").read_text().splitlines()
		}
		if (
			summary["termination"] == "prompts_exhausted"
			and wanted <= have
			and not summary["blocked"]
		):
			return None
		return {
			"termination": summary["termination"],
			"received": len(wanted & have),
			"expected": len(wanted),
			"blocked": summary["blocked"],
		}

	def collect_protected(self, task: str, seed: int, budget: int, token: str, arm: str) -> None:
		directory = self.protected_dir(task, seed, budget, arm)
		directory.mkdir(parents=True, exist_ok=True)
		self.pipeline.launcher.distill(
			"collect-protected",
			"--input",
			str(self.pipeline.run_dir(task, seed, budget) / "prompts.jsonl"),
			"--output",
			str(directory / "ledger.jsonl"),
			"--summary",
			str(directory / "campaign.json"),
			"--teacher-url",
			self.protected_url,
			"--terms-review-id",
			self.pipeline.terms_review_id,
			"--max-output-tokens",
			str(self.pipeline.manifest(task)["max_output_tokens"]),
			"--max-requests",
			str(budget),
			"--decision-log",
			str(self.root / task / f"s{seed}" / arm / "decisions.jsonl"),
			*(["--block-only"] if arm == "quota" else []),
			env={"TEACHER_API_TOKEN": token},
		)

	def train_protected(
		self, task: str, seed: int, budget: int, largest: int, arm: str = "full"
	) -> bool:
		from distillation.preparation import prepare_sft_dataset

		directory = self.protected_dir(task, seed, budget, arm)
		directory.mkdir(parents=True, exist_ok=True)
		prompts = self.pipeline.run_dir(task, seed, budget) / "prompts.jsonl"
		campaign = self.protected_dir(task, seed, largest, arm)
		events = [
			json.loads(line)
			for line in (campaign / "campaign.events.jsonl").read_text().splitlines()
		]
		wanted = {
			event["id"]
			for event in events
			if event["event"] == "attempt" and event["index"] <= budget
		}
		from distillation.campaign import event_summary

		save_json(
			directory / "budget-campaign.json",
			event_summary(events, (4,) if arm == "quota" else (1, 2, 3, 4), budget=budget),
		)
		ledger = directory / "budget-ledger.jsonl"
		write_jsonl(
			ledger,
			[
				r
				for r in read_jsonl(campaign / "ledger.jsonl", LedgerRecord)
				if r.attempt_id in wanted
			],
		)
		dataset = directory / "train.jsonl"
		with self.pipeline.config.open("rb") as source:
			training = tomllib.load(source)
		training.update(self.pipeline.recipes.get(task, {}))
		training.pop("validation_dataset_path", None)
		training |= {
			"dataset_path": str(dataset),
			"output_dir": str(directory / "adapter"),
			"seed": seed,
		}
		config = directory / "train.toml"
		config.write_text(_toml(training), encoding="utf-8")
		reviews = directory / "reviews.jsonl"
		self.pipeline.launcher.distill(
			"review", "--ledger", str(ledger), "--output", str(reviews), "--config", str(config)
		)
		prepare_sft_dataset(
			input_path=ledger, output_path=dataset, reviews_path=reviews, prompts_path=prompts
		)
		count = len(read_jsonl(dataset, SFTRecord))
		if not count:
			save_json(
				directory / "training-status.json",
				{"status": "skipped", "reason": "no_usable_responses"},
			)
			self.pipeline.launcher.log(
				f"{task} s{seed}-b{budget} {arm}: no usable protected outputs; skipping training"
			)
			return False
		if not _trained(directory / "adapter", dataset, config):
			self.pipeline.launcher.distill("train", "--config", str(config), "--preflight-only")
			self.pipeline.launcher.distill("train", "--config", str(config))
		save_json(directory / "training-status.json", {"status": "trained", "examples": count})
		return True

	def reports(self) -> None:
		from distillation.benchmarks.reporting import build_report, experiments

		for directory in self.root.glob("*/s*-b*/benchmarks/*"):
			for experiment in experiments(directory)["experiments"]:
				save_json(
					directory.parent.parent / f"report.{directory.name}.json",
					build_report(directory, experiment["id"]),
				)

	def run(self) -> None:
		from distillation.study_training import select_recipe, train_controls

		launcher = self.pipeline.launcher
		self.pipeline.validate()
		if "coding" in self.pipeline.tasks and self.l4_action != "refuse":
			raise LaunchError(
				"coding supports L4 refusal; select --l4-action refuse or omit coding"
			)
		if self.harness_port in {self.port, int(os.environ.get("VLLM_PORT") or 8001)}:
			raise LaunchError("--harness-port must differ from the teacher and vLLM ports")
		if self.teacher_url and self.upstream_url:
			raise LaunchError("choose either --teacher-url or --upstream-url")
		self.pipeline.artifact_root = self.root / "adapters"
		if not self.arms or set(self.arms) - set(DEFENSES):
			raise LaunchError(f"--arms must name defenses from {', '.join(DEFENSES)}")
		from distillation.transfer import analysis_protocol

		freeze(
			self.root / "protocol.json",
			{
				"tasks": self.pipeline.tasks,
				"seeds": self.pipeline.seeds,
				"budgets": "full" if self.pipeline.budgets is None else self.pipeline.budgets,
				"training": self.pipeline.config.read_text(),
				"gold": self.pipeline.gold,
				"teacher_url": self.url,
				"upstream_url": self.upstream_url,
				"teacher_model": os.environ.get("TEACHER_UPSTREAM_MODEL") or DEFAULT_TEACHER_MODEL,
				"teacher_revision": os.environ.get("TEACHER_MODEL_REVISION")
				or DEFAULT_TEACHER_REVISION,
				"terms_review_id": self.pipeline.terms_review_id,
				"l4_action": self.l4_action,
				"arms": list(self.arms),
				"protocol_version": "task-study-v5",
				"transfer_analysis": analysis_protocol(self.minimum_gain, self.pipeline.tasks),
				"study_kind": "separate_task",
				"budget_plan": "budgets.json",
				"capability_target_rule": (
					f"development base score plus {TARGET_GAIN_FRACTION:g} of the selected "
					"recipe's development gain, frozen in TASK/target.json before test evaluation"
				),
				"training_seed": self.pipeline.training_seed,
				"attempt_budget_unit": "HTTP submissions, including proof redemption",
				"deadline_seconds": 86400,
				"schedule": "uniform_without_replacement",
				"controls": ["matched-volume", "matched-prefix"],
				"aggregate_monitor": "not_run",
				"calibration_root": str(self.calibration_root),
			},
		)
		os.environ["TEACHER_TERMS_REVIEW_ID"] = self.pipeline.terms_review_id
		os.environ["TEACHER_API_TOKEN"] = self.token
		save_json(self.root / "status.json", {"status": "running"})
		launcher.install("train")
		self.pipeline.build_tasks()
		freeze(self.root / "budgets.json", self.pipeline.plan)
		self.pipeline.validate_ledgers()
		freeze(
			self.root / "tasks.json",
			{task: self.pipeline.manifest(task) for task in self.pipeline.tasks},
		)
		calibrations = {task: self.calibrate(task) for task in self.pipeline.tasks}
		from harness.calibration import load_calibration

		freeze(
			self.root / "policy-horizons.json",
			{
				task: {
					"attempts": calibration.config.horizon_attempts,
					"seconds": calibration.config.horizon_seconds,
					"campaign_budget_replenished_on_reset": False,
				}
				for task, path in calibrations.items()
				for calibration, _ in [load_calibration(path)]
			},
		)
		signals = {}
		for task, path in calibrations.items():
			self.calibration_report(task, path)
			self.benign_holdout(task, path)
			signals[task] = self.detectability(task, path)
		self.require_signal(signals)
		self.ensure_teacher()
		self.collect_undefended()
		launcher.stop_background()
		for task in self.pipeline.tasks:
			selection = select_recipe(self, task)
			self.pipeline.recipes[task] = selection["selected"]
			if not selection["transfer_gate_passed"]:
				save_json(
					self.root / "status.json",
					{
						"status": "transfer_gate_failed",
						"task": task,
						"development_gain": selection["development_gain"],
					},
				)
				raise LaunchError(
					f"{task}: undefended qualification did not establish meaningful transfer; "
					"final defense testing stopped"
				)
			freeze(self.root / task / "target.json", capability_target(selection))
		self.ensure_teacher()
		self.baseline_evaluations(("teacher",))
		for task in self.pipeline.tasks:
			for seed in self.pipeline.seeds:
				for arm in self.arms:
					self.traffic(task, seed, "before", arm=arm)
		launcher.stop_background()
		self.baseline_evaluations(("student_before",))
		self.pipeline.train_all()
		self.baseline_evaluations(("student_after",))
		self.ensure_teacher()
		self.baseline_evaluations(("teacher_after",))
		# Restart only the protected service between campaigns. The teacher remains available,
		# and each defense and seed gets fresh per-account and aggregate state.
		for task, budgets in self.pipeline.plan.items():
			for seed in self.pipeline.seeds:
				for arm in self.arms:
					teacher_processes = len(launcher.background)
					keys = self.start_harness(task, seed, calibrations[task], arm)
					try:
						self.collect_protected(task, seed, budgets[-1], keys["extractor"], arm)
						self.traffic(task, seed, "after", keys, arm)
					finally:
						launcher.stop_background(keep=teacher_processes)
		launcher.stop_background()
		for task, budgets in self.pipeline.plan.items():
			for seed in self.pipeline.seeds:
				for budget in budgets:
					for arm in self.arms:
						trained = self.train_protected(task, seed, budget, budgets[-1], arm)
						config = self.evaluation(
							task, seed, budget, "attacker", arm=arm, untrained=not trained
						)
						self.benchmark(config, "student_before")
						self.benchmark(config, "student_after")
						train_controls(self, task, seed, budget, arm)
		self.reports()
		from distillation.study_reporting import write_study_results

		write_study_results(self.root)
		save_json(self.root / "status.json", {"status": "complete", "tasks": self.pipeline.plan})
		launcher.log(
			f"done; protected ledgers, adapters, benchmarks, and reports are in {self.root}"
		)
