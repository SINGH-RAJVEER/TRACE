"""One shared gateway, campaign ledger, and student adapter across all selected domains."""

import asyncio
import hashlib
import json
import os
import secrets
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.calibration import (
	HORIZON_POLICY,
	BenignRequest,
	HarnessConfig,
	calibrate,
	load_calibration,
	requests_sha256,
	save_calibration,
	validate_separation,
)
from harness.encoder import BGEEncoder
from harness.replay import replay
from harness.serialization import gateway_turn

from distillation.benchmarks.models import Slot, load_config, load_suite, save_json
from distillation.benchmarks.reporting import discover_runs
from distillation.campaign import service_failures
from distillation.collection import CollectionOptions
from distillation.gateway_campaign import collect_gateway, digest
from distillation.gateway_reporting import campaign_results, profile_rates, supervision
from distillation.io import read_jsonl, write_jsonl
from distillation.launcher import (
	DEFAULT_TEACHER_MODEL,
	DEFAULT_TEACHER_REVISION,
	LOG_DIR,
	LaunchError,
	_toml,
	_trained,
)
from distillation.preparation import prepare_sft_dataset
from distillation.schedules import ATTACKS, EXTERNAL_ATTACKS, Attack, campaign_schedule
from distillation.schemas import GatewayRequest, LedgerRecord, Message, PromptRecord, SFTRecord
from distillation.study import DECLARED_ACCUMULATORS, DEFENSES, Study, _MemoEncoder, freeze
from distillation.study_training import primary_score, token_controls
from distillation.tasks import LEGITIMATE_ROLES, GoldRecord
from distillation.utility import measure_utility
from distillation.workloads import mixed_clients, task_references


def benign_requests(rows: list[GatewayRequest], *, serialized: bool = True) -> list[BenignRequest]:
	return [
		BenignRequest(
			account=row.account,
			text=gateway_turn(
				row.domain, next(m.content for m in reversed(row.prompt.prompt) if m.role == "user")
			)
			if serialized
			else next(m.content for m in reversed(row.prompt.prompt) if m.role == "user"),
			time=row.time,
			example_id=row.prompt.example_id,
			source_group=f"{row.domain}:{row.prompt.source.group_id}",
			client_type=row.client_type,
		)
		for row in rows
	]


def rank_trials(trials: list[dict[str, Any]], domains: list[str]) -> dict[str, Any]:
	"""Choose the recipe with the lowest mean per-domain rank of development gain.

	Domain metrics have different scales and variances, so a raw cross-domain mean would let
	the noisiest domain choose. Ties share their average rank; equal mean ranks favor fewer
	epochs, then the lower learning rate.
	"""
	for domain in domains:
		gains = [trial["gains"][domain] for trial in trials]
		for trial in trials:
			higher = sum(g > trial["gains"][domain] for g in gains)
			equal = sum(g == trial["gains"][domain] for g in gains)
			trial.setdefault("ranks", {})[domain] = higher + (equal + 1) / 2
	for trial in trials:
		trial["mean_rank"] = sum(trial["ranks"].values()) / len(trial["ranks"])
	return min(
		trials, key=lambda t: (t["mean_rank"], t["recipe"]["epochs"], t["recipe"]["learning_rate"])
	)


@dataclass
class GatewayStudy:
	study: Study
	# A7 uses independently pinned prompts, the positive control for a distribution shift.
	attacks: tuple[Attack, ...] = ("A1", "A2", "A5", "A6", "A7")
	interval_seconds: float = 1
	# A4's declared synthetic spacing; the datasets carry no natural request timing.
	slow_interval_seconds: float | None = None
	deadline_seconds: float = 86400
	benign_accounts: int = 125
	requests_per_account: int = 100
	development_budget: int = 2000
	pools: dict[str, list[PromptRecord]] = field(default_factory=dict)
	external_pools: dict[str, list[PromptRecord]] = field(default_factory=dict)
	caps: dict[str, int] = field(default_factory=dict)
	benign: dict[str, list[GatewayRequest]] = field(default_factory=dict)
	selected: dict[str, Any] = field(default_factory=dict)
	_active: bool = False

	@property
	def root(self) -> Path:
		return self.study.root

	def build(self, *, benign: bool = True) -> None:
		pipeline = self.study.pipeline
		pipeline.build_missing_tasks()
		for domain in pipeline.tasks:
			directory = pipeline.task_dir(domain)
			self.pools[domain] = read_jsonl(directory / "attacker.jsonl", PromptRecord)
			self.caps[domain] = int(pipeline.manifest(domain)["max_output_tokens"])
			if set(self.attacks) & EXTERNAL_ATTACKS:
				if not (directory / "external" / "manifest.json").exists():
					pipeline.launcher.distill(
						"gateway",
						"external",
						"--task",
						domain,
						"--tasks-dir",
						str(directory.parent),
					)
				self.external_pools[domain] = read_jsonl(
					directory / "external" / "attacker.jsonl", PromptRecord
				)
		freeze(self.root / "tasks.json", {d: pipeline.manifest(d) for d in pipeline.tasks})
		if self.external_pools:
			freeze(
				self.root / "external.json",
				{
					d: json.loads((pipeline.task_dir(d) / "external" / "manifest.json").read_text())
					for d in pipeline.tasks
				},
			)
		if not benign:
			return
		references = {
			k: v for d in pipeline.tasks for k, v in task_references(pipeline.task_dir(d)).items()
		}
		for role in LEGITIMATE_ROLES:
			pools = {
				d: read_jsonl(pipeline.task_dir(d) / "legitimate" / f"{role}.jsonl", PromptRecord)
				for d in pipeline.tasks
			}
			rows, manifest = mixed_clients(
				pools=pools,
				caps=self.caps,
				role=role,
				accounts=self.benign_accounts,
				requests_per_account=self.requests_per_account,
				seed=42,
				references=references,
				concentrated=self.study.concentrated_share,
			)
			self.benign[role] = rows
			directory = self.root / "benign" / role
			freeze(directory / "manifest.json", manifest)
			self.write_requests(directory / "requests.jsonl", rows)
		# The held-out role must be disjoint from every calibration role, too.
		validate_separation({role: benign_requests(rows) for role, rows in self.benign.items()})

	def write_requests(self, path: Path, rows: list[GatewayRequest]) -> None:
		if path.exists():
			if read_jsonl(path, GatewayRequest) != rows:
				raise LaunchError(f"gateway inventory changed: {path}")
		else:
			path.parent.mkdir(parents=True, exist_ok=True)
			write_jsonl(path, rows)

	def calibration(self) -> Path:
		path = self.study.calibration_root or self.root / "calibration"
		groups = {
			r: benign_requests(self.benign[r]) for r in LEGITIMATE_ROLES if r != "benign_test"
		}
		if (path / "calibration.json").exists():
			calibration, _ = load_calibration(path)
			if calibration.config.task != "gateway" or any(
				calibration.sha256[r] != requests_sha256(rows) for r, rows in groups.items()
			):
				raise LaunchError("gateway calibration does not match this mixed benign workload")
			return path
		if len(groups["fitting"]) < 100:
			raise LaunchError("mixed calibration needs at least 100 fitting requests")
		defaults = HarnessConfig().accumulators
		config = HarnessConfig(
			task="gateway",
			reference_windows=min(20, len(groups["fitting"]) // 100),
			aggregate_enabled=False,
			accumulators={name: defaults[name] for name in DECLARED_ACCUMULATORS},
		)
		calibration, references = calibrate(
			config=config,
			encoder=BGEEncoder(device="cpu"),
			fitting=groups["fitting"],
			score_calibration=groups["score_calibration"],
			policy_calibration=groups["policy_calibration"],
			log=self.study.pipeline.launcher.log,
		)
		save_calibration(path, calibration, references)
		return path

	def inventory(
		self, directory: Path, attack: Attack, seed: int, budget: int, *, full: bool = False
	) -> Path:
		external = attack in EXTERNAL_ATTACKS
		pools = self.external_pools if external else self.pools
		# Full external inventories match the task pools wherever the source is large enough,
		# and the campaign allowance is that inventory's own total. Full A3 adds an equal
		# number of cover prompts per domain, which the task pools always hold.
		allocation = (
			{d: min(len(p), len(pools[d])) for d, p in self.pools.items()} if full else None
		)
		if allocation is not None:
			budget = sum(allocation.values()) * (2 if attack == "A3" else 1)
		rows, manifest = campaign_schedule(
			pools=pools,
			caps=self.caps,
			attack=attack,
			seed=seed,
			budget=budget,
			campaign_id=f"{attack}-s{seed}-b{budget}",
			interval_seconds=(
				self.slow_interval_seconds
				if attack == "A4" and self.slow_interval_seconds is not None
				else self.interval_seconds
			),
			deadline_seconds=self.deadline_seconds,
			allocation=allocation,
			cover_pools=self.pools if attack == "A3" else None,
			cover_allocation=allocation if attack == "A3" else None,
		)
		freeze(directory / "inventory.json", manifest)
		path = directory / "requests.jsonl"
		self.write_requests(path, rows)
		return path

	def collect(
		self,
		schedule: Path,
		directory: Path,
		budget: int,
		seed: int,
		*,
		keys: dict[str, str] | None = None,
		decisions: Path | None = None,
	) -> dict[str, Any]:
		summary = asyncio.run(
			collect_gateway(
				schedule=schedule,
				output=directory,
				url=self.study.protected_url if keys else self.study.url,
				keys=keys if keys else {"default": self.study.token},
				options=CollectionOptions(
					terms_review_id=self.study.pipeline.terms_review_id,
					seed=seed,
					enable_thinking=False,
					concurrency=self.study.pipeline.concurrency,
				),
				attempt_budget=budget,
				deadline_seconds=self.deadline_seconds,
				decision_log=decisions,
			)
		)
		if failed := service_failures(summary):
			raise LaunchError(
				f"{directory}: teacher service failed for {', '.join(failed)}; a transport "
				"failure is not a harness refusal. Fix the service and use a fresh --run-dir"
			)
		return summary

	def start_harness(
		self, directory: Path, schedule: Path, calibration: Path, arm: str
	) -> tuple[dict[str, str], Path]:
		accounts = sorted(
			{r.account for r in read_jsonl(schedule, GatewayRequest)}
			| {r.account for r in self.benign["benign_test"]}
			| {"warmup"}
		)
		credentials = directory / "credentials.json"
		if not credentials.exists():
			credentials.parent.mkdir(parents=True, exist_ok=True)
			with open(credentials, "x", opener=lambda p, f: os.open(p, f, 0o600)) as stream:
				json.dump({a: secrets.token_hex(32) for a in accounts}, stream)
		keys: dict[str, str] = json.loads(credentials.read_text())
		accounts_file = directory / "accounts.json"
		save_json(
			accounts_file, {hashlib.sha256(k.encode()).hexdigest(): a for a, k in keys.items()}
		)
		decisions = directory / "decisions.jsonl"
		policy, accumulator = DEFENSES[arm]
		env = {
			"HARNESS_HOST": "127.0.0.1",
			"HARNESS_PORT": str(self.study.harness_port),
			"HARNESS_TEACHER_URL": self.study.url,
			"HARNESS_TEACHER_TOKEN": self.study.token,
			"HARNESS_TASK": "gateway",
			"HARNESS_GATEWAY_CAPS": json.dumps(self.caps),
			"HARNESS_CALIBRATION_DIR": str(calibration),
			"HARNESS_ACCOUNTS_FILE": str(accounts_file),
			"HARNESS_TRUST_ACTOR_HEADER": "false",
			"HARNESS_TEST_CLOCK": "false",
			"HARNESS_POLICY": policy,
			"HARNESS_ACCUMULATOR": accumulator,
			"HARNESS_L4_ACTION": "refuse",
			"HARNESS_THRESHOLDS": "calibrated",
			"HARNESS_ENCODER_DEVICE": "cpu",
			"HARNESS_DECISION_LOG": str(decisions),
			"HARNESS_DECEPTION_KEY_FILE": str(self.root / "deception.key"),
		}
		launcher = self.study.pipeline.launcher
		log = LOG_DIR / "gateway-harness.log"
		process = launcher.start("gateway", [sys.executable, "-m", "harness_api"], log, env)
		launcher.wait_for_http(
			f"{self.study.protected_url}/healthz",
			'"ready"',
			300,
			"Starting shared gateway",
			process,
			log,
		)
		return keys, decisions

	def train(
		self,
		directory: Path,
		schedule: Path,
		seed: int,
		recipe: dict[str, Any],
		*,
		gold: bool = False,
	) -> Path | None:
		pipeline = self.study.pipeline
		base = tomllib.loads(pipeline.config.read_text()) | recipe
		base.pop("validation_dataset_path", None)
		dataset, adapter, config = (
			directory / "train.jsonl",
			directory / "adapter",
			directory / "train.toml",
		)
		freeze(
			directory / "training-protocol.json",
			base
			| {
				"seed": pipeline.training_seed_for(seed),
				"input_sha256": {
					d: digest(pipeline.task_dir(d) / "gold-answers.jsonl") for d in self.pools
				}
				if gold
				else {"ledger": digest(directory / "ledger.jsonl")},
				"schedule_sha256": digest(schedule),
			},
		)
		config.write_text(
			_toml(
				base
				| {
					"seed": pipeline.training_seed_for(seed),
					"dataset_path": str(dataset),
					"output_dir": str(adapter),
				}
			)
		)
		if gold:
			answers = {
				r.example_id: r.completion
				for d in self.pools
				for r in read_jsonl(pipeline.task_dir(d) / "gold-answers.jsonl", GoldRecord)
			}
			write_jsonl(
				dataset,
				[
					SFTRecord(
						prompt=r.prompt.prompt,
						completion=[
							Message(role="assistant", content=answers[r.prompt.example_id])
						],
					)
					for r in read_jsonl(schedule, GatewayRequest)
				],
			)
			count = len(read_jsonl(dataset, SFTRecord))
		else:
			reviews = directory / "reviews.jsonl"
			pipeline.launcher.distill(
				"review",
				"--ledger",
				str(directory / "ledger.jsonl"),
				"--output",
				str(reviews),
				"--config",
				str(config),
			)
			prompts = directory / "prompts.jsonl"
			write_jsonl(prompts, [r.prompt for r in read_jsonl(schedule, GatewayRequest)])
			count, _ = prepare_sft_dataset(
				input_path=directory / "ledger.jsonl",
				output_path=dataset,
				reviews_path=reviews,
				prompts_path=prompts,
			)
		supervision(directory, pipeline.tasks, schedule, gold=gold)
		if count and not _trained(adapter, dataset, config):
			pipeline.launcher.distill("train", "--config", str(config), "--preflight-only")
			pipeline.launcher.distill("train", "--config", str(config))
		save_json(
			directory / "training-status.json",
			{
				"status": "trained" if count else "no_usable_responses",
				"examples": count,
				"domains": self.study.pipeline.tasks,
				"shared_adapter": str(adapter) if count else None,
			},
		)
		return adapter if count else None

	def evaluate(
		self,
		directory: Path,
		adapter: Path | None,
		seed: int,
		budget: int,
		*,
		development: bool = False,
		development_split: str = "selection",
		teacher: bool = False,
		teacher_after: bool = False,
	) -> dict[str, Any]:
		results: dict[str, Any] = {}
		slots: tuple[Slot, ...] = (
			(("teacher_after",) if teacher_after else ("teacher",))
			if teacher
			else ("student_before", "student_after")
		)
		# The untrained student is the same model for every final-test condition, so it is
		# benchmarked once per seed and budget. Development trials keep their own baselines.
		base = self.root / "base-student" / f"s{seed}-b{budget}"
		for domain in self.study.pipeline.tasks:
			source = self.study.evaluation(
				domain,
				seed,
				budget,
				"gateway",
				adapter_override=adapter,
				development=development,
				development_split=development_split,
				untrained=adapter is None,
			)
			latest = {}
			for slot in slots:
				target = (
					base
					if slot == "student_before" and not development and adapter is not None
					else directory
				)
				config = load_config(source)
				config.output_dir = target / "benchmarks" / domain
				config.experiment = str(target)
				config.dimensions |= {
					"study_kind": "shared_gateway",
					"condition": str(target),
					"schedule": "mixed_domain",
				}
				config_path = target / f"eval.{domain}.toml"
				config_path.parent.mkdir(parents=True, exist_ok=True)
				config_path.write_text(_toml(config.model_dump(mode="json")))
				self.study.benchmark(config_path, slot)
				runs, errors = discover_runs(config.output_dir)
				if errors:
					raise LaunchError("invalid shared student evaluation artifacts")
				latest[slot] = max(
					(
						r
						for r in runs
						if r.slot == slot
						and r.model == config.models[slot]
						and r.generation == config.generation
						and r.dimensions == config.dimensions
						and r.suite_sha256 == load_suite(config.suite_path).sha256
					),
					key=lambda r: r.completed_at,
				)
			results[domain] = {
				slot: primary_score(run.samples, domain) for slot, run in latest.items()
			}
			results[domain]["evaluation_ids"] = {slot: run.id for slot, run in latest.items()}
			results[domain]["suite_sha256"] = load_suite(config.suite_path).sha256
			results[domain]["samples"] = {
				slot: [sample.model_dump(mode="json") for sample in run.samples]
				for slot, run in latest.items()
			}
		save_json(
			directory
			/ (
				"teacher-after-scores.json"
				if teacher_after
				else "teacher-scores.json"
				if teacher
				else "scores.json"
			),
			results,
		)
		return results

	def select_recipe(self, schedule: Path, campaign: Path) -> None:
		from distillation.benchmarks.models import SampleResult
		from distillation.transfer import analysis_protocol, qualification

		analysis = analysis_protocol(self.study.minimum_gain, self.study.pipeline.tasks)
		# Materialize both source-disjoint halves before any candidate evaluation.
		suites = {d: self.study.development_suites(d) for d in self.study.pipeline.tasks}
		freeze(
			self.root / "development/protocol.json",
			{
				"analysis": analysis,
				"suites": {
					d: {k: load_suite(p).sha256 for k, p in paths.items()}
					for d, paths in suites.items()
				},
			},
		)
		trials = []
		for rate in (1e-4, 2e-4):
			for epochs in (1, 2, 3):
				directory = self.root / "development" / f"lr{rate}-e{epochs}"
				write_jsonl(
					directory / "ledger.jsonl", read_jsonl(campaign / "ledger.jsonl", LedgerRecord)
				)
				recipe = {"learning_rate": rate, "epochs": epochs}
				adapter = self.train(directory, schedule, 42, recipe)
				if adapter is None:
					raise LaunchError("balanced development campaign produced no usable responses")
				scores = self.evaluate(
					directory, adapter, 42, self.development_budget, development=True
				)
				trials.append(
					{
						"recipe": recipe,
						"scores": scores,
						"gains": {
							d: s["student_after"] - s["student_before"] for d, s in scores.items()
						},
					}
				)
		winner = rank_trials(trials, self.study.pipeline.tasks)
		self.selected = winner["recipe"]
		freeze(self.root / "development/selected-recipe.json", winner)
		directory = (
			self.root
			/ "development"
			/ f"lr{self.selected['learning_rate']}-e{self.selected['epochs']}"
		)
		qualified_scores = self.evaluate(
			self.root / "development/qualification",
			directory / "adapter",
			42,
			self.development_budget,
			development=True,
			development_split="qualification",
		)
		qualifications = {
			d: qualification(
				[SampleResult.model_validate(s) for s in scores["samples"]["student_before"]],
				[SampleResult.model_validate(s) for s in scores["samples"]["student_after"]],
				analysis["minimum_gain_pp"][d],
			)
			| {"evaluation_ids": scores["evaluation_ids"], "suite_sha256": scores["suite_sha256"]}
			for d, scores in qualified_scores.items()
		}
		passed = all(q["passed"] for q in qualifications.values())
		# A domain without qualified development transfer is uninformative about the defense:
		# it stays in every report with that label. The study stops only when none qualifies.
		informative = sorted(d for d, q in qualifications.items() if q["passed"])
		freeze(
			self.root / "recipe.json",
			{
				"selected": self.selected,
				"trials": trials,
				"qualification": qualifications,
				"analysis_version": analysis["version"],
				"transfer_gate_passed": passed,
				"informative_domains": informative,
				"uninformative_domains": sorted(set(qualifications) - set(informative)),
				"selection": "lowest mean per-domain rank of development gain; "
				"ties favor fewer epochs then lower rate",
				"inventory_sha256": digest(schedule),
			},
		)
		if not informative:
			raise LaunchError(
				"shared-student qualification did not establish meaningful transfer in any domain"
			)

	def utility(self, directory: Path, *, keys: dict[str, str] | None = None) -> None:
		references = {
			k: v
			for d in self.pools
			for k, v in task_references(self.study.pipeline.task_dir(d)).items()
		}
		asyncio.run(
			measure_utility(
				requests=benign_requests(self.benign["benign_test"], serialized=False),
				references=references,
				url=self.study.protected_url if keys else self.study.url,
				keys=keys if keys else {"default": self.study.token},
				max_output_tokens=self.caps,
				account_profiles={
					r.account: r.domain_profile or "unknown" for r in self.benign["benign_test"]
				},
				output=directory / "utility.json",
				concurrency=self.study.pipeline.concurrency,
			)
		)

	def controls(
		self, baseline: Path, defended: Path, schedule: Path, seed: int, budget: int
	) -> None:
		before = read_jsonl(baseline / "ledger.jsonl", LedgerRecord)
		after = read_jsonl(defended / "ledger.jsonl", LedgerRecord)
		selections: dict[str, list[LedgerRecord]] = {"matched-volume": [], "matched-prefix": []}
		matches: dict[str, dict[str, Any]] = {name: {} for name in selections}
		for domain in self.pools:
			rows = [r for r in after if r.task == domain]
			if any(r.usage.output_tokens is None for r in rows):
				raise LaunchError("token controls require measured released tokens")
			target = sum(int(r.usage.output_tokens or 0) for r in rows)
			for name, (selected, tokens) in token_controls(
				[r for r in before if r.task == domain], target, seed
			).items():
				selections[name].extend(selected)
				matches[name][domain] = {
					"target_tokens": target,
					"achieved_tokens": tokens,
					"residual_tokens": tokens - target,
				}
		for name, rows in selections.items():
			directory = defended / name
			write_jsonl(directory / "ledger.jsonl", rows)
			freeze(directory / "control.json", matches[name])
			adapter = self.train(directory, schedule, seed, self.selected)
			self.evaluate(directory, adapter, seed, budget)

	def run(self, *, collect_only: bool = False) -> None:
		self._active = False
		try:
			self._run(collect_only=collect_only)
		except BaseException as error:
			if self._active:
				save_json(
					self.root / "status.json",
					{
						"status": "interrupted"
						if isinstance(error, KeyboardInterrupt)
						else "failed",
						"reason": str(error),
						"error_type": type(error).__name__,
					},
				)
			if isinstance(error, (OSError, ValueError, KeyError)):
				raise LaunchError(str(error)) from error
			raise
		finally:
			self.study.pipeline.launcher.stop_background()

	def _run(self, *, collect_only: bool = False) -> None:
		from distillation.transfer import analysis_protocol

		pipeline, launcher = self.study.pipeline, self.study.pipeline.launcher
		pipeline.validate()
		if len(pipeline.tasks) < 2 or len(set(pipeline.tasks)) != len(pipeline.tasks):
			raise LaunchError("shared gateway requires at least two distinct domains")
		if self.study.l4_action != "refuse":
			raise LaunchError("shared gateway currently supports L4 refusal")
		if (
			not self.attacks
			or set(self.attacks) - set(ATTACKS)
			or len(set(self.attacks)) != len(self.attacks)
			or not self.study.arms
			or set(self.study.arms) - DEFENSES.keys()
		):
			raise LaunchError("unknown gateway attack or defense")
		if "A4" in self.attacks and (
			self.slow_interval_seconds is None
			or self.slow_interval_seconds <= self.interval_seconds
		):
			raise LaunchError(
				"A4 needs --slow-interval-seconds above --interval-seconds: the datasets have "
				"no natural request timing, so declare the synthetic slow rate"
			)
		if self.study.harness_port in {self.study.port, int(os.environ.get("VLLM_PORT") or 8001)}:
			raise LaunchError("harness port must differ from teacher ports")
		if self.study.teacher_url and self.study.upstream_url:
			raise LaunchError("choose teacher-url or upstream-url")
		os.environ["TEACHER_TERMS_REVIEW_ID"] = pipeline.terms_review_id
		os.environ["TEACHER_API_TOKEN"] = self.study.token
		freeze(
			self.root / "protocol.json",
			{
				"version": "shared-gateway-v4-bounded",
				"calibration_gate": None if collect_only else "calibration-resolution-v2",
				"transfer_analysis": analysis_protocol(self.study.minimum_gain, pipeline.tasks)
				if not collect_only
				else None,
				"study_kind": "shared_gateway",
				"domains": pipeline.tasks,
				"coding_selection": {
					"eligible_target": pipeline.coding_pool_size or None,
					"source_seed": 42,
				}
				if "coding" in pipeline.tasks
				else None,
				"seeds": pipeline.seeds,
				"budgets": pipeline.budgets or "full",
				"attacks": list(self.attacks),
				"arms": list(self.study.arms),
				"training": pipeline.config.read_text(),
				"terms_review_id": pipeline.terms_review_id,
				"deadline_seconds": self.deadline_seconds,
				"interval_seconds": self.interval_seconds,
				"benign_accounts": self.benign_accounts,
				"requests_per_account": self.requests_per_account,
				"development_budget": self.development_budget,
				"collect_only": collect_only,
				"gold": pipeline.gold,
				# Recorded only when set, so existing paired-seed run directories still match.
				**(
					{"training_seed": pipeline.training_seed}
					if pipeline.training_seed is not None
					else {}
				),
				**(
					{"slow_interval_seconds": self.slow_interval_seconds}
					if self.slow_interval_seconds is not None
					else {}
				),
				"resource_unit": "all HTTP submissions including proofs",
				"aggregate_monitor": "disabled",
				"teacher_url": self.study.url,
				"teacher_model": os.environ.get("TEACHER_UPSTREAM_MODEL") or DEFAULT_TEACHER_MODEL,
				"teacher_revision": os.environ.get("TEACHER_MODEL_REVISION")
				or DEFAULT_TEACHER_REVISION,
				"upstream_url": self.study.upstream_url,
			},
		)
		save_json(self.root / "status.json", {"status": "running"})
		self._active = True
		launcher.install("train")
		self.build(benign=not collect_only)
		budgets = sorted(pipeline.budgets or [sum(map(len, self.pools.values()))])
		calibration_path = None if collect_only else self.calibration()
		if calibration_path:
			self.study.require_calibration("gateway", calibration_path)
			calibration, scorer = load_calibration(calibration_path)
			freeze(
				self.root / "policy-horizons.json",
				{
					"attempts": calibration.config.horizon_attempts,
					"seconds": calibration.config.horizon_seconds,
					"campaign_budget_replenished_on_reset": False,
					"horizon_policy": HORIZON_POLICY,
					"reset": "elapsed_time",
					"attempt_limit_action": "refuse_until_time_reset",
					"allow_policy_capped": False,
				},
			)
			encoder = _MemoEncoder(BGEEncoder(device="cpu"))
			for arm in self.study.arms:
				policy, accumulator = DEFENSES[arm]
				result = replay(
					calibration=calibration,
					scorer=scorer,
					requests=benign_requests(self.benign["benign_test"]),
					output=self.root / "benign-test" / f"{arm}.jsonl",
					policy=policy,
					accumulator=accumulator,
					encoder=encoder if policy == "mmd" else None,
				)
				result["profiles"] = profile_rates(
					self.benign["benign_test"],
					self.root / "benign-test" / f"{arm}.jsonl",
					result["implemented_levels"],
				)
				save_json(self.root / "benign-test" / f"{arm}.json", result)
		jobs = []
		for seed in pipeline.seeds:
			for budget in budgets:
				for attack in self.attacks:
					directory = self.root / attack / f"s{seed}-b{budget}"
					schedule = self.inventory(
						directory, attack, seed, budget, full=pipeline.budgets is None
					)
					# Full external inventories can be smaller than the task-pool total.
					allowance = json.loads((directory / "inventory.json").read_text())
					jobs.append((directory, schedule, seed, allowance["attempt_budget"]))
		development = self.root / "development"
		devschedule = (
			None if collect_only else self.inventory(development, "A1", 42, self.development_budget)
		)
		self.study.ensure_teacher()
		for directory, schedule, seed, budget in jobs:
			self.collect(schedule, directory / "undefended", budget, seed)
		if collect_only:
			launcher.stop_background()
			for directory, schedule, seed, _ in jobs:
				self.train(directory / "undefended", schedule, seed, {})
				if pipeline.gold and directory.parent.name not in EXTERNAL_ATTACKS:
					self.train(directory / "gold", schedule, seed, {}, gold=True)
			save_json(
				self.root / "status.json", {"status": "complete", "mode": "shared_distillation"}
			)
			return
		assert devschedule is not None
		self.collect(devschedule, development / "campaign", self.development_budget, 42)
		self.utility(self.root / "benign-test" / "undefended")
		launcher.stop_background()
		self.select_recipe(devschedule, development / "campaign")
		self.study.ensure_teacher()
		for directory, schedule, seed, budget in jobs:
			for arm in self.study.arms:
				defended = directory / arm
				# A completed collection is reused without restarting a previously spent policy.
				if (defended / "campaign" / "campaign.json").exists():
					keys = json.loads((defended / "credentials.json").read_text())
					self.collect(
						schedule,
						defended / "campaign",
						budget,
						seed,
						keys=keys,
						decisions=defended / "decisions.jsonl",
					)
					if not (defended / "utility.json").exists():
						raise LaunchError(
							"interrupted protected utility: use a fresh run directory"
						)
					continue
				if (defended / "campaign").exists():
					raise LaunchError("interrupted protected campaign: use a fresh run directory")
				keep = len(launcher.background)
				assert calibration_path is not None
				keys, decisions = self.start_harness(defended, schedule, calibration_path, arm)
				try:
					self.collect(
						schedule,
						defended / "campaign",
						budget,
						seed,
						keys=keys,
						decisions=decisions,
					)
					self.utility(defended, keys=keys)
				finally:
					launcher.stop_background(keep=keep)
		# The bare teacher is the same model for every attack, so each seed and budget is
		# benchmarked once and its scores are copied to the other campaigns.
		measured: dict[tuple[int, int], Path] = {}
		for directory, _, seed, budget in jobs:
			source = measured.setdefault((seed, budget), directory / "teacher")
			if source == directory / "teacher":
				self.evaluate(source, None, seed, budget, teacher=True)
				self.evaluate(source, None, seed, budget, teacher=True, teacher_after=True)
				continue
			for name in ("teacher-scores.json", "teacher-after-scores.json"):
				(directory / "teacher").mkdir(parents=True, exist_ok=True)
				shutil.copyfile(source / name, directory / "teacher" / name)
		launcher.stop_background()
		results = []
		for directory, schedule, seed, budget in jobs:
			base_adapter = self.train(directory / "undefended", schedule, seed, self.selected)
			self.evaluate(directory / "undefended", base_adapter, seed, budget)
			# Independent sources carry no gold answers, so A3 and A7 have no gold reference.
			if pipeline.gold and directory.parent.name not in EXTERNAL_ATTACKS:
				gold_adapter = self.train(
					directory / "gold", schedule, seed, self.selected, gold=True
				)
				self.evaluate(directory / "gold", gold_adapter, seed, budget)
			for arm in self.study.arms:
				campaign = directory / arm / "campaign"
				adapter = self.train(campaign, schedule, seed, self.selected)
				self.evaluate(campaign, adapter, seed, budget)
				self.controls(directory / "undefended", campaign, schedule, seed, budget)
			results.extend(
				campaign_results(
					directory,
					seed,
					budget,
					self.study.arms,
					pipeline.tasks,
					qualification=json.loads((self.root / "recipe.json").read_text())[
						"qualification"
					],
				)
			)
		save_json(
			self.root / "results.json",
			{
				"study_kind": "shared_gateway",
				"units": "scores and gains in percentage points; fractions are dimensionless",
				"results": results,
				"uncertainty": "schedules and ledgers retain source identities; "
				"no across-seed independence assumed",
			},
		)
		save_json(
			self.root / "status.json",
			{"status": "complete", "campaigns": len(jobs), "shared_student": True},
		)
