"""End-to-end task extraction, including the protected teacher and second students."""

import hashlib
import json
import os
import secrets
import sys
import tomllib
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


def freeze(path: Path, value: dict[str, Any]) -> None:
	"""Prevent an existing run directory from silently changing its protocol."""
	if path.exists():
		if json.loads(path.read_text()) != value:
			raise LaunchError(f"{path} changed; select a new --run-dir")
	else:
		save_json(path, value)


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

	def evaluation(
		self, task: str, seed: int, budget: int, kind: str, *, protected: bool = False
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
			directory / "protected" / "adapter"
			if protected
			else self.pipeline.artifact_root / task / f"s{seed}-b{budget}" / kind
		)
		teacher = ModelConfig(
			backend="teacher_api",
			model=os.environ.get("TEACHER_UPSTREAM_MODEL") or DEFAULT_TEACHER_MODEL,
			revision=os.environ.get("TEACHER_MODEL_REVISION") or DEFAULT_TEACHER_REVISION,
			base_url=self.url,
			api_key_env="TEACHER_API_TOKEN" if self.token else "",
		)
		condition = "protected" if protected else kind
		config = EvaluationConfig(
			experiment=f"start-{task}-s{seed}-b{budget}-{condition}",
			suite_path=self.pipeline.task_dir(task) / "test-suite.json",
			output_dir=directory / "benchmarks" / condition,
			generation=GenerationConfig(
				max_output_tokens=self.pipeline.manifest(task)["max_output_tokens"]
			),
			models={
				"teacher": teacher,
				"teacher_after": teacher,
				"student_before": student,
				"student_after": student.model_copy(update={"adapter_path": adapter}),
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
		key = json.dumps(
			[
				slot,
				suite.sha256,
				settings.generation.model_dump(),
				settings.models[slot].model_dump(mode="json"),
			],
			sort_keys=True,
		)
		if slot != "student_after" and key in self.references:
			# Identical reference inference is shared across budgets and conditions. Keep
			# the original ID, timestamps, and provenance, while joining the new comparison.
			run = self.references[key].model_copy(update={"experiment": settings.experiment})
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
							self.benchmark(
								self.evaluation(task, seed, budget, "attacker", protected=True),
								slot,
							)

	def calibrate(self, task: str) -> Path:
		from harness.calibration import (
			HarnessConfig,
			calibrate,
			load_calibration,
			pooled_accounts,
			requests_sha256,
			save_calibration,
		)
		from harness.encoder import BGEEncoder

		if self.calibration_root:
			path = self.calibration_root / task
			load_calibration(path)
			return path
		groups = {}
		for role in ("fitting", "score_calibration", "policy_calibration"):
			prompts = read_jsonl(
				self.pipeline.task_dir(task) / "legitimate" / f"{role}.jsonl", PromptRecord
			)
			groups[role] = pooled_accounts(
				(
					(
						prompt.example_id,
						next(m.content for m in reversed(prompt.prompt) if m.role == "user"),
					)
					for prompt in prompts
				),
				role,
				500,
				HarnessConfig().seed,
			)
		path = self.root / task / "calibration"
		if (path / "calibration.json").exists():
			existing, _ = load_calibration(path)
			# Older calibrations sliced article-ordered SQuAD pools or embedded whole turns.
			if existing.config.task != task or existing.sha256 != {
				role: requests_sha256(rows) for role, rows in groups.items()
			}:
				raise LaunchError(
					f"{path} was calibrated on different benign accounts or detector input; "
					"start a fresh --run-dir or remove that calibration"
				)
			return path
		self.pipeline.launcher.log(
			f"{task}: calibrating on pooled benign accounts; this does not validate real-user FPR"
		)
		# Small task pools cannot supply twenty 100-request references. Keep the window
		# fixed and record the supported reference count in the calibration artifact.
		count = min(20, len(groups["fitting"]) // 100)
		if not count:
			raise LaunchError(f"{task}: calibration needs at least 100 fitting prompts")
		self.pipeline.launcher.log(f"{task}: calibration window=100, reference_windows={count}")
		calibration, references = calibrate(
			config=HarnessConfig.model_validate({"task": task, "reference_windows": count}),
			encoder=BGEEncoder(device="cpu"),
			fitting=groups["fitting"],
			score_calibration=groups["score_calibration"],
			policy_calibration=groups["policy_calibration"],
			log=self.pipeline.launcher.log,
		)
		save_calibration(path, calibration, references)
		return path

	def start_harness(self, task: str, seed: int, calibration: Path) -> dict[str, str]:
		from distillation.benchmarks.traffic import load_traffic_suite

		actors = [
			"extractor",
			*(a.id for a in load_traffic_suite(self.root / "traffic.json").actors),
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
		env = {
			"HARNESS_HOST": "127.0.0.1",
			"HARNESS_PORT": str(self.harness_port),
			"HARNESS_TEACHER_URL": self.url,
			"HARNESS_TEACHER_TOKEN": self.token,
			"HARNESS_CALIBRATION_DIR": str(calibration),
			"HARNESS_ACCOUNTS_FILE": str(accounts),
			"HARNESS_TRUST_ACTOR_HEADER": "false",
			"HARNESS_TEST_CLOCK": "false",
			"HARNESS_POLICY": "mmd",
			"HARNESS_ACCUMULATOR": "paper",
			"HARNESS_THRESHOLDS": "calibrated",
			"HARNESS_L4_ACTION": self.l4_action,
			"HARNESS_TASK": task,
			"HARNESS_ENCODER_DEVICE": "cpu",
			"HARNESS_DECISION_LOG": str(credentials.with_name("decisions.jsonl")),
			"HARNESS_DECEPTION_KEY_FILE": str(self.root / "deception.key"),
		}
		log = LOG_DIR / f"start-harness-{task}-s{seed}.log"
		process = self.pipeline.launcher.start(
			"harness-api", [sys.executable, "-m", "harness_api"], log, env
		)
		self.pipeline.launcher.wait_for_http(
			f"{self.protected_url}/healthz", '"ready"', 300, "Starting harness", process, log
		)
		return keys

	def traffic(self, task: str, seed: int, phase: str, keys: dict[str, str] | None = None) -> None:
		from distillation.benchmarks.traffic import TrafficConfig, TrafficTarget, load_traffic_suite

		key_envs = {
			a.id: f"START_TRAFFIC_{a.id}"
			for a in load_traffic_suite(self.root / "traffic.json").actors
		}
		values = TrafficConfig(
			experiment=f"start-{task}-s{seed}-traffic",
			suite_path=self.root / "traffic.json",
			output_dir=self.root / task / f"s{seed}" / "traffic-results",
			model=os.environ.get("TEACHER_UPSTREAM_MODEL") or DEFAULT_TEACHER_MODEL,
			model_revision=os.environ.get("TEACHER_MODEL_REVISION") or DEFAULT_TEACHER_REVISION,
			targets={
				"before": TrafficTarget(
					base_url=self.url,
					service_revision="unprotected",
					api_key_env="TEACHER_API_TOKEN" if self.token else "",
				),
				"after": TrafficTarget(
					base_url=self.protected_url,
					service_revision=f"mmd-paper-{self.l4_action}",
					actor_key_envs=key_envs,
				),
			},
		)
		config = self.root / task / f"s{seed}" / "traffic.toml"
		config.parent.mkdir(parents=True, exist_ok=True)
		config.write_text(_toml(values.model_dump(mode="json")), encoding="utf-8")
		self.pipeline.launcher.distill(
			"benchmark",
			"traffic",
			"run",
			"--config",
			str(config),
			"--phase",
			phase,
			env={name: keys[actor] for actor, name in key_envs.items()} if keys else None,
		)

	def collect_protected(self, task: str, seed: int, budget: int, token: str) -> None:
		directory = self.run_dir(task, seed, budget) / "protected"
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
			env={"TEACHER_API_TOKEN": token},
		)

	def train_protected(self, task: str, seed: int, budget: int, largest: int) -> bool:
		from distillation.preparation import prepare_sft_dataset

		directory = self.run_dir(task, seed, budget) / "protected"
		directory.mkdir(parents=True, exist_ok=True)
		prompts = self.pipeline.run_dir(task, seed, budget) / "prompts.jsonl"
		wanted = {p.example_id for p in read_jsonl(prompts, PromptRecord)}
		campaign = self.run_dir(task, seed, largest) / "protected"
		ledger = directory / "budget-ledger.jsonl"
		write_jsonl(
			ledger,
			[
				r
				for r in read_jsonl(campaign / "ledger.jsonl", LedgerRecord)
				if r.example_id in wanted
			],
		)
		dataset = directory / "train.jsonl"
		with self.pipeline.config.open("rb") as source:
			training = tomllib.load(source)
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
				f"{task} s{seed}-b{budget}: no usable protected outputs; skipping training"
			)
			return False
		if not _trained(directory / "adapter", dataset, config):
			self.pipeline.launcher.distill("train", "--config", str(config), "--preflight-only")
			self.pipeline.launcher.distill("train", "--config", str(config))
		save_json(directory / "training-status.json", {"status": "trained", "examples": count})
		return True

	def reports(self) -> None:
		from distillation.benchmarks.reporting import build_report, experiments
		from distillation.benchmarks.traffic_reporting import (
			build_traffic_report,
			traffic_experiments,
		)

		for directory in self.root.glob("*/s*-b*/benchmarks/*"):
			for experiment in experiments(directory)["experiments"]:
				save_json(
					directory.parent.parent / f"report.{directory.name}.json",
					build_report(directory, experiment["id"]),
				)
		for directory in self.root.glob("*/s*/traffic-results"):
			for experiment in traffic_experiments(directory)["experiments"]:
				save_json(
					directory.parent / "traffic-report.json",
					build_traffic_report(directory, experiment["id"]),
				)

	def run(self) -> None:
		from distillation.benchmarks.traffic import freeze_traffic

		launcher = self.pipeline.launcher
		self.pipeline.validate()
		if self.harness_port in {self.port, int(os.environ.get("VLLM_PORT") or 8001)}:
			raise LaunchError("--harness-port must differ from the teacher and vLLM ports")
		if self.teacher_url and self.upstream_url:
			raise LaunchError("choose either --teacher-url or --upstream-url")
		if any(budget < 1 for budget in self.pipeline.budgets):
			raise LaunchError("query budgets must be positive")
		freeze(
			self.root / "protocol.json",
			{
				"tasks": self.pipeline.tasks,
				"seeds": self.pipeline.seeds,
				"budgets": self.pipeline.budgets,
				"training": self.pipeline.config.read_text(),
				"gold": self.pipeline.gold,
				"teacher_url": self.url,
				"upstream_url": self.upstream_url,
				"teacher_model": os.environ.get("TEACHER_UPSTREAM_MODEL") or DEFAULT_TEACHER_MODEL,
				"teacher_revision": os.environ.get("TEACHER_MODEL_REVISION")
				or DEFAULT_TEACHER_REVISION,
				"terms_review_id": self.pipeline.terms_review_id,
				"l4_action": self.l4_action,
				"calibration_root": str(self.calibration_root),
			},
		)
		os.environ["TEACHER_TERMS_REVIEW_ID"] = self.pipeline.terms_review_id
		os.environ["TEACHER_API_TOKEN"] = self.token
		save_json(self.root / "status.json", {"status": "running"})
		launcher.install("train")
		self.pipeline.build_tasks()
		self.pipeline.validate_ledgers()
		freeze(
			self.root / "tasks.json",
			{task: self.pipeline.manifest(task) for task in self.pipeline.tasks},
		)
		if not (self.root / "traffic.json").exists():
			freeze_traffic(self.root / "traffic.json")
		self.ensure_teacher()
		self.pipeline.collect(self.url, self.token)
		self.baseline_evaluations(("teacher",))
		for task in self.pipeline.tasks:
			for seed in self.pipeline.seeds:
				self.traffic(task, seed, "before")
		launcher.stop_background()
		self.baseline_evaluations(("student_before",))
		self.pipeline.train_all()
		self.baseline_evaluations(("student_after",))
		calibrations = {task: self.calibrate(task) for task in self.pipeline.tasks}
		self.ensure_teacher()
		self.baseline_evaluations(("teacher_after",))
		# Restart only the protected service between seed campaigns. The teacher remains
		# available, and each campaign gets fresh per-account and aggregate state.
		for task, budgets in self.pipeline.plan.items():
			for seed in self.pipeline.seeds:
				teacher_processes = len(launcher.background)
				keys = self.start_harness(task, seed, calibrations[task])
				try:
					self.collect_protected(task, seed, budgets[-1], keys["extractor"])
					self.traffic(task, seed, "after", keys)
				finally:
					launcher.stop_background(keep=teacher_processes)
		launcher.stop_background()
		for task, budgets in self.pipeline.plan.items():
			for seed in self.pipeline.seeds:
				for budget in budgets:
					if self.train_protected(task, seed, budget, budgets[-1]):
						config = self.evaluation(task, seed, budget, "attacker", protected=True)
						self.benchmark(config, "student_before")
						self.benchmark(config, "student_after")
		self.reports()
		save_json(self.root / "status.json", {"status": "complete", "tasks": self.pipeline.plan})
		launcher.log(
			f"done; protected ledgers, adapters, benchmarks, and reports are in {self.root}"
		)
