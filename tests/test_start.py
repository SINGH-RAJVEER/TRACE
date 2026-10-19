"""Whole-study sequencing with real datasets, preparation, scoring, and HTTP adapters."""

import asyncio
import hashlib
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import httpx
import pytest
from distillation.benchmarks.models import Slot, load_config
from distillation.benchmarks.runner import run_evaluation
from distillation.benchmarks.traffic import load_traffic_config
from distillation.benchmarks.traffic_runner import run_traffic
from distillation.campaign import ProtectedTeacher, collect_campaign
from distillation.cli import app
from distillation.collection import CollectionOptions, collect
from distillation.io import read_jsonl
from distillation.launcher import Launcher, LaunchError, TaskPipeline, run_app
from distillation.schemas import (
	Message,
	PromptRecord,
	SFTRecord,
	TeacherGeneration,
	TeacherGenerationRequest,
	TokenUsage,
)
from distillation.study import Study
from distillation.tasks import build_task
from distillation.training import TrainingConfig
from typer.testing import CliRunner


class Teacher:
	async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
		return TeacherGeneration(
			id="fixture",
			provider="fixture",
			model="fixture",
			generated_at=datetime.now(UTC),
			message=Message(role="assistant", content="World"),
			finish_reason="stop",
			endpoint="fixture://teacher",
			usage=TokenUsage(output_tokens=1),
		)

	async def close(self) -> None:
		pass


@pytest.mark.parametrize("scenario", ["start", "distill"])
@pytest.mark.parametrize("tasks", [None, "ag_news,squad"])
@pytest.mark.parametrize("budgets", [None, "full", "500,2000,8000"])
def test_launchers_default_tasks_and_supported_budgets(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	scenario: str,
	tasks: str | None,
	budgets: str | None,
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setenv("TEACHER_TERMS_REVIEW_ID", "fixture")
	Path("configs").mkdir()
	Path("configs/train.task.toml").write_text('base_model="fixture"\n')
	for task, pool in (("ag_news", 53893), ("squad", 40356), ("gsm8k", 3389), ("coding", 12345)):
		directory = Path("data/tasks") / task
		directory.mkdir(parents=True)
		(directory / "manifest.json").write_text(json.dumps({"roles": {"attacker": pool}}))
	monkeypatch.setattr(Launcher, "install", lambda self, component: None)
	monkeypatch.setattr(TaskPipeline, "budget", lambda self, task, seed, budget: None)
	monkeypatch.setattr(TaskPipeline, "validate_ledgers", lambda self: None)
	monkeypatch.setattr(TaskPipeline, "uncollected", lambda self: 0)
	plans: list[dict[str, list[int]]] = []

	def record(pipeline: TaskPipeline) -> None:
		plans.append(pipeline.plan.copy())

	def plan(study: Study) -> None:
		study.pipeline.validate()
		study.pipeline.build_tasks()
		record(study.pipeline)

	# Exercise CLI task selection and real budget planning without collecting or training.
	monkeypatch.setattr(TaskPipeline, "train_all", record)
	monkeypatch.setattr(Study, "run", plan)
	command = [scenario, "--mode", "tasks"]
	if tasks is not None:
		command += ["--tasks", tasks]
	if budgets is not None:
		command += ["--budgets", budgets]
	result = CliRunner().invoke(run_app, command)
	assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
	explicit = budgets == "500,2000,8000"
	expected = {
		"ag_news": [500, 2000, 8000] if explicit else [53893],
		"squad": [500, 2000, 8000] if explicit else [40356],
	}
	if tasks is None:
		expected["gsm8k"] = [500, 2000] if explicit else [3389]
		expected["coding"] = [500, 2000, 8000] if explicit else [12345]
	assert plans == [expected]


@pytest.mark.parametrize("task", ["ag_news", "gsm8k"])
@pytest.mark.parametrize("budgets", [None, "1"])
def test_start_checks_corrupt_ledgers_before_loading_the_teacher(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	task: Literal["ag_news", "gsm8k"],
	budgets: str | None,
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setenv("TEACHER_TERMS_REVIEW_ID", "fixture")
	build_task(
		task=task,
		output_dir=Path("data/tasks") / task,
		rows={
			"train": [{"text": f"News story {i}", "label": i % 4} for i in range(500)],
			"test": [{"text": "Held-out news article", "label": 0}],
		}
		if task == "ag_news"
		else {
			"train": [
				{"question": f"What is {i} plus 1?", "answer": f"#### {i + 1}"} for i in range(500)
			],
			"test": [{"question": "What is 600 plus 1?", "answer": "#### 601"}],
		},
	)
	Path("configs").mkdir()
	Path("configs/train.task.toml").write_text('base_model="fixture"\n')
	ledger = Path("data/tasks") / task / "ledger.jsonl"
	ledger.write_bytes(b"\x00" * 64 + b"\n")
	monkeypatch.setattr(Launcher, "install", lambda self, component: None)
	monkeypatch.setattr(
		Study, "ensure_teacher", lambda self: pytest.fail("teacher loaded before checking ledger")
	)

	def distill(self: Launcher, *args: str, env: dict[str, str] | None = None) -> None:
		result = CliRunner().invoke(app, list(args))
		assert result.exit_code == 0, result.output

	monkeypatch.setattr(Launcher, "distill", distill)
	command = ["start", "--mode", "tasks", "--tasks", task, "--seeds", "42", "--no-gold"]
	if budgets is not None:
		command += ["--budgets", budgets]
	result = CliRunner().invoke(run_app, command)
	assert result.exit_code == 1
	assert "ledger.jsonl:1: invalid record" in result.output
	protocol = json.loads(Path("artifacts/start/protocol.json").read_text())
	assert protocol["budgets"] == ("full" if budgets is None else [1])
	assert json.loads(Path("artifacts/start/budgets.json").read_text()) == {
		task: [
			json.loads((Path("data/tasks") / task / "manifest.json").read_text())["roles"][
				"attacker"
			]
			if budgets is None
			else 1
		]
	}
	assert "ledger repair" in result.output
	assert ledger.read_bytes() == b"\x00" * 64 + b"\n"


def test_start_runs_both_students_and_reports_with_services_stopped_before_training(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.chdir(tmp_path)
	monkeypatch.setenv("TEACHER_TERMS_REVIEW_ID", "fixture")
	build_task(
		task="ag_news",
		output_dir=Path("data/tasks/ag_news"),
		rows={
			"train": [{"text": f"News story {i}", "label": i % 4} for i in range(500)],
			"test": [{"text": "Held-out news article", "label": 0}],
		},
	)
	Path("configs").mkdir()
	Path("configs/train.task.toml").write_text(
		'base_model="fixture"\nbase_revision="' + "a" * 40 + '"\n'
		'dataset_path="unused"\noutput_dir="unused"\nqlora=true\n'
	)
	steps: list[str] = []
	services: list[str] = []
	monkeypatch.setattr(Launcher, "install", lambda self, component: None)
	monkeypatch.setattr(Study, "calibrate", lambda self, task: Path("calibration"))
	monkeypatch.setattr(Study, "calibration_report", lambda self, task, path: None)
	monkeypatch.setattr(Study, "benign_holdout", lambda self, task, path: None)
	monkeypatch.setattr(Study, "detectability", lambda self, task, path: {"mmd_signal": True})
	from types import SimpleNamespace

	monkeypatch.setattr(
		"harness.calibration.load_calibration",
		lambda path: (
			SimpleNamespace(config=SimpleNamespace(horizon_attempts=10000, horizon_seconds=86400)),
			None,
		),
	)
	# Model selection itself needs GPU evaluations; this test checks sequencing only.
	monkeypatch.setattr(
		"distillation.study_training.select_recipe",
		lambda study, task: {
			"selected": {"learning_rate": 2e-4, "epochs": 1},
			"transfer_gate_passed": True,
			"trials": [{"learning_rate": 2e-4, "epochs": 1, "base_score": 20.0, "score": 60.0}],
		},
	)
	monkeypatch.setattr(
		"distillation.training.training_identity", lambda config: config.model_dump(mode="json")
	)
	import distillation.benchmarks.traffic as traffic

	freeze_traffic = traffic.freeze_traffic
	monkeypatch.setattr(
		traffic, "freeze_traffic", lambda path: freeze_traffic(path, duration_seconds=0.01)
	)

	def serve(self: Study) -> None:
		services.append("teacher")
		steps.append("serve")

	def harness(self: Study, task: str, seed: int, calibration: Path, arm: str) -> dict[str, str]:
		services.append("harness")
		steps.append(f"harness {arm}")
		return {"extractor": "fixture", **{f"u{i:02}": "fixture" for i in range(6)}}

	def stop(self: Launcher, *, keep: int = 0) -> None:
		services[:] = services[:keep]
		steps.append("stop")

	async def response(request: httpx.Request) -> httpx.Response:
		payload = json.loads(request.content)
		if "v1/teacher/generate" in str(request.url):
			answer = await Teacher().generate(TeacherGenerationRequest.model_validate(payload))
			return httpx.Response(
				200, json=answer.model_dump(mode="json"), headers={"X-Harness-Decision": "allow"}
			)
		raise AssertionError(request.url)

	import distillation.utility as utility

	measure_utility = utility.measure_utility

	async def measured_utility(**kwargs: Any) -> dict[str, Any]:
		# Keep the real utility scorer with immediate arrivals and mocked inference.
		kwargs["requests"] = [row.model_copy(update={"time": 0.0}) for row in kwargs["requests"]]
		return await measure_utility(**kwargs, transport=httpx.MockTransport(response))

	monkeypatch.setattr(utility, "measure_utility", measured_utility)

	def distill(self: Launcher, *args: str, env: dict[str, str] | None = None) -> None:
		def option(name: str) -> str:
			return args[args.index(name) + 1]

		if args[0] == "collect":
			assert services == ["teacher"]
			steps.append("collect")
			asyncio.run(
				collect(
					input_path=Path(option("--input")),
					output_path=Path(option("--output")),
					teacher=Teacher(),
					options=CollectionOptions(
						terms_review_id="fixture", max_output_tokens=16, enable_thinking=False
					),
				)
			)
		elif args[0] == "collect-protected":
			protected = "protected" in option("--output")
			assert ("harness" in services) == protected
			steps.append("protected-collect" if protected else "collect")
			asyncio.run(
				collect_campaign(
					prompts_path=Path(option("--input")),
					ledger_path=Path(option("--output")),
					summary_path=Path(option("--summary")),
					teacher=ProtectedTeacher(
						"http://protected", "fixture", transport=httpx.MockTransport(response)
					),
					options=CollectionOptions(
						terms_review_id="fixture",
						max_output_tokens=16,
						enable_thinking=False,
						concurrency=1,
						max_attempts=1,
					),
				)
			)
		elif args[:2] == ("benchmark", "run"):
			config = load_config(Path(option("--config")))
			slot = option("--slot")
			if slot.startswith("student"):
				assert not services
			else:
				assert "teacher" in services
			assert config.models["student_before"].load_in_4bit
			steps.append(f"benchmark {slot}")
			# Keep adapter provenance and the real scoring path; only inference is mocked.
			asyncio.run(run_evaluation(config, cast(Slot, slot), factory=lambda config: Teacher()))
		elif args[:2] == ("benchmark", "traffic"):
			settings = load_traffic_config(Path(option("--config")))
			with monkeypatch.context() as context:
				for name in settings.targets["after"].actor_key_envs.values():
					context.setenv(name, "fixture")
				asyncio.run(
					run_traffic(
						settings,
						cast(Literal["before", "after"], option("--phase")),
						transport=httpx.MockTransport(response),
					)
				)
			steps.append(f"traffic {option('--phase')}")
		elif args[0] == "train":
			assert not services
			if "--preflight-only" in args:
				return
			values = tomllib.loads(Path(option("--config")).read_text())
			output = Path(values["output_dir"])
			output.mkdir(parents=True)
			(output / "training-manifest.json").write_text(
				json.dumps(
					{
						"dataset_sha256": hashlib.sha256(
							Path(values["dataset_path"]).read_bytes()
						).hexdigest(),
						"config": values,
						"fingerprint": TrainingConfig.model_validate(values).model_dump(
							mode="json"
						),
					}
				)
			)
			(output / "adapter_config.json").write_text("{}")
			(output / "adapter_model.safetensors").write_bytes(b"fixture")
			steps.append("train protected" if "protected" in str(output) else "train baseline")
		else:
			if args[0] == "review":
				args = args[: args.index("--config")]
			result = CliRunner().invoke(app, list(args))
			assert result.exit_code == 0, result.output

	monkeypatch.setattr(Study, "ensure_teacher", serve)
	monkeypatch.setattr(Study, "start_harness", harness)
	monkeypatch.setattr(Launcher, "stop_background", stop)
	monkeypatch.setattr(Launcher, "distill", distill)
	command = [
		"start",
		"--mode",
		"tasks",
		"--tasks",
		"ag_news",
		"--seeds",
		"42",
		"--budgets",
		"4",
		"--no-gold",
	]
	result = CliRunner().invoke(run_app, command)
	assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
	harnesses = [step for step in steps if step.startswith("harness")]
	assert harnesses == ["harness full", "harness instantaneous", "harness quota"]
	assert steps.index("train baseline") < steps.index(harnesses[0])
	assert steps.index(harnesses[-1]) < steps.index("train protected")
	assert steps.index("benchmark student_before") < steps.index("train baseline")
	root = Path("artifacts/start/ag_news/s42-b4")
	assert (root / "report.attacker.json").is_file()
	for arm in ("full", "instantaneous", "quota"):
		assert len(read_jsonl(root / f"protected/{arm}/train.jsonl", SFTRecord)) == 4
		assert (root / f"report.protected-{arm}.json").is_file()
		assert (root.parent / f"s42/{arm}/utility-after.json").is_file()
	assert json.loads(Path("artifacts/start/status.json").read_text())["status"] == "complete"
	baseline = (root.parent / "s42/undefended/ledger.jsonl").read_bytes()
	steps.clear()
	result = CliRunner().invoke(run_app, command)
	assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
	assert "train baseline" not in steps and "train protected" not in steps
	assert (root.parent / "s42/undefended/ledger.jsonl").read_bytes() == baseline


@pytest.mark.asyncio
async def test_campaign_records_refusals_and_preserves_interrupted_budget(tmp_path: Path) -> None:
	from distillation.io import write_jsonl
	from distillation.schemas import LedgerRecord, PromptSource

	prompts = tmp_path / "prompts.jsonl"
	write_jsonl(
		prompts,
		[
			PromptRecord(
				example_id=str(i),
				split="train",
				task="fixture",
				source=PromptSource(corpus="fixture", source_id=str(i), license="mit"),
				prompt=[Message(role="user", content=str(i))],
			)
			for i in range(3)
		],
	)
	options = CollectionOptions(
		terms_review_id="fixture",
		max_output_tokens=16,
		enable_thinking=False,
		concurrency=1,
		max_attempts=1,
	)
	broken, failing = True, False

	async def response(request: httpx.Request) -> httpx.Response:
		payload = TeacherGenerationRequest.model_validate_json(request.content)
		question = payload.messages[-1].content
		if question == "1":
			return httpx.Response(403, headers={"X-Harness-Decision": "block"})
		if question == "2" and broken:
			return httpx.Response(401)
		if question == "2" and failing:
			return httpx.Response(502)
		return httpx.Response(200, json=(await Teacher().generate(payload)).model_dump(mode="json"))

	def teacher() -> ProtectedTeacher:
		return ProtectedTeacher("http://fixture", "secret", transport=httpx.MockTransport(response))

	ledger, summary = tmp_path / "ledger.jsonl", tmp_path / "campaign.json"
	with pytest.raises(httpx.HTTPStatusError):
		await collect_campaign(
			prompts_path=prompts,
			ledger_path=ledger,
			summary_path=summary,
			teacher=teacher(),
			options=options,
		)
	original = ledger.read_bytes()
	assert len(read_jsonl(ledger, LedgerRecord)) == 1 and not summary.exists()
	broken = False
	with pytest.raises(ValueError, match="retained its spent budget"):
		await collect_campaign(
			prompts_path=prompts,
			ledger_path=ledger,
			summary_path=summary,
			teacher=teacher(),
			options=options,
		)
	assert ledger.read_bytes() == original
	# Restart only in a separately declared campaign with fresh policy state.
	ledger, summary = tmp_path / "fresh/ledger.jsonl", tmp_path / "fresh/campaign.json"
	result = await collect_campaign(
		prompts_path=prompts,
		ledger_path=ledger,
		summary_path=summary,
		teacher=teacher(),
		options=options,
	)
	assert result["queries"] == 3 and result["blocked"] == 1 and result["ledger_records"] == 2
	assert all(r.example_id != "1" for r in read_jsonl(ledger, LedgerRecord))
	before = ledger.read_bytes()
	assert (
		await collect_campaign(
			prompts_path=prompts,
			ledger_path=ledger,
			summary_path=summary,
			teacher=teacher(),
			options=options,
		)
		== result
	)
	assert ledger.read_bytes() == before
	# A server error is charged to the budget as a failed attempt rather than replaced.
	failing = True
	charged = tmp_path / "charged"
	result = await collect_campaign(
		prompts_path=prompts,
		ledger_path=charged / "ledger.jsonl",
		summary_path=charged / "campaign.json",
		teacher=teacher(),
		options=options,
	)
	assert (result["queries"], result["blocked"], result["failed"]) == (3, 1, 1)
	assert result["received"] == result["ledger_records"] == 1
	events = (charged / "campaign.events.jsonl").read_text().splitlines()
	assert any(json.loads(event).get("status") == "failed" for event in events)
	# A generic 403 must not be counted as a confirmed policy refusal.
	gateway = ProtectedTeacher(
		"http://fixture", "bad", transport=httpx.MockTransport(lambda request: httpx.Response(403))
	)
	try:
		with pytest.raises(httpx.HTTPStatusError):
			await gateway.generate(
				TeacherGenerationRequest(messages=[Message(role="user", content="x")])
			)
	finally:
		await gateway.close()


def test_detectability_gate_stops_unless_quota_comparison_is_allowed(tmp_path: Path) -> None:
	pipeline = TaskPipeline(Launcher("test"), "fixture", ["ag_news"], [42], [1], tmp_path, 1, False)
	study = Study(pipeline, tmp_path / "study", "", "", 8000, "", 8003, None, "refuse")
	signals = {"ag_news": {"mmd_signal": False}, "squad": {"mmd_signal": True}}
	with pytest.raises(LaunchError, match="ag_news: no MMD defense"):
		study.require_signal(signals)
	assert json.loads((study.root / "status.json").read_text())["tasks"] == ["ag_news"]
	study.allow_undetected = True
	study.require_signal(signals)
	gate = json.loads((study.root / "detectability-gate.json").read_text())
	assert gate == {
		"undetected_tasks": ["ag_news"],
		"allow_undetected": True,
		"reading": "quota comparison",
	}


def test_empty_protected_campaign_skips_training(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	from distillation.io import write_jsonl
	from distillation.launcher import TaskPipeline
	from distillation.schemas import PromptSource

	monkeypatch.chdir(tmp_path)
	config = tmp_path / "train.toml"
	config.write_text('base_model="fixture"\n')
	pipeline = TaskPipeline(Launcher("test"), "fixture", ["ag_news"], [42], [1], config, 1, False)
	study = Study(pipeline, tmp_path / "study", "", "", 8000, "", 8003, None, "refuse")
	write_jsonl(
		pipeline.run_dir("ag_news", 42, 1) / "prompts.jsonl",
		[
			PromptRecord(
				example_id="one",
				split="train",
				task="ag_news",
				source=PromptSource(corpus="fixture", source_id="one", license="mit"),
				prompt=[Message(role="user", content="One")],
			)
		],
	)
	write_jsonl(study.protected_dir("ag_news", 42, 1, "full") / "ledger.jsonl", [])
	(study.protected_dir("ag_news", 42, 1, "full") / "campaign.events.jsonl").write_text("")

	def review(self: Launcher, *args: str, env: dict[str, str] | None = None) -> None:
		assert args[0] == "review"
		result = CliRunner().invoke(app, list(args[: args.index("--config")]))
		assert result.exit_code == 0, result.output

	monkeypatch.setattr(Launcher, "distill", review)
	assert not study.train_protected("ag_news", 42, 1, 1)
	assert not (study.protected_dir("ag_news", 42, 1, "full") / "adapter").exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_study_report_uses_resolved_task_budgets(tmp_path: Path, legacy: bool) -> None:
	from distillation.study_reporting import study_report

	protocol = {"tasks": ["ag_news", "gsm8k"], "seeds": [42], "budgets": "full", "arms": ["full"]}
	(tmp_path / "protocol.json").write_text(json.dumps(protocol))
	if legacy:
		(tmp_path / "tasks.json").write_text(
			json.dumps(
				{
					"ag_news": {"roles": {"attacker": 53893}},
					"gsm8k": {"roles": {"attacker": 3389}},
				}
			)
		)
	else:
		(tmp_path / "budgets.json").write_text(json.dumps({"ag_news": [53893], "gsm8k": [3389]}))
	report = study_report(tmp_path)
	assert [(row["task"], row["budget"]) for row in report["missing_pairs"]] == [
		("ag_news", 53893),
		("gsm8k", 3389),
	]
	assert {row["status"] for row in report["cost_to_target"]} == {"target_not_declared"}
	assert report["seed_sd_sources"].startswith("query order and training")


def test_capability_target_comes_from_selected_development_scores(tmp_path: Path) -> None:
	from distillation.study import capability_target
	from distillation.study_reporting import study_report

	selection = {
		"selected": {"learning_rate": 2e-4, "epochs": 2},
		"trials": [
			{"learning_rate": 1e-4, "epochs": 2, "base_score": 10.0, "score": 90.0},
			{"learning_rate": 2e-4, "epochs": 2, "base_score": 10.0, "score": 30.0},
		],
	}
	target = capability_target(selection)
	assert target["target_score"] == 20.0
	protocol = {
		"tasks": ["gsm8k"],
		"seeds": [42, 1],
		"budgets": [2000],
		"arms": ["full"],
		"training_seed": 7,
	}
	(tmp_path / "protocol.json").write_text(json.dumps(protocol))
	(tmp_path / "budgets.json").write_text(json.dumps({"gsm8k": [2000]}))
	(tmp_path / "gsm8k").mkdir()
	(tmp_path / "gsm8k" / "target.json").write_text(json.dumps(target))
	report = study_report(tmp_path)
	assert {row["training_seed"] for row in report["missing_pairs"]} == {7}
	assert {row["target_score"] for row in report["cost_to_target"]} == {20.0}
	assert report["seed_sd_sources"] == "query order only (training seed fixed at 7)"


def test_study_report_uses_only_budgets_that_fit_each_task(tmp_path: Path) -> None:
	from distillation.study_reporting import study_report

	(tmp_path / "protocol.json").write_text(
		json.dumps(
			{
				"tasks": ["ag_news", "gsm8k"],
				"seeds": [42],
				"budgets": [2000, 8000],
				"arms": ["full"],
			}
		)
	)
	(tmp_path / "budgets.json").write_text(json.dumps({"ag_news": [2000, 8000], "gsm8k": [2000]}))
	assert [(row["task"], row["budget"]) for row in study_report(tmp_path)["missing_pairs"]] == [
		("ag_news", 2000),
		("ag_news", 8000),
		("gsm8k", 2000),
	]


@pytest.mark.asyncio
async def test_policy_resets_do_not_replenish_campaign_attempt_budget(tmp_path: Path) -> None:
	from distillation.io import write_jsonl
	from distillation.schemas import PromptSource
	from harness.policy import QuotaPolicy

	policy = QuotaPolicy(quota=1, horizon_seconds=86400, horizon_attempts=3)
	prompts = tmp_path / "prompts.jsonl"
	write_jsonl(
		prompts,
		[
			PromptRecord(
				example_id=str(i),
				split="train",
				task="fixture",
				source=PromptSource(corpus="fixture", source_id=str(i), license="mit"),
				prompt=[Message(role="user", content=str(i))],
			)
			for i in range(10)
		],
	)

	async def response(request: httpx.Request) -> httpx.Response:
		payload = TeacherGenerationRequest.model_validate_json(request.content)
		decision = policy.admit("extractor", payload.messages[-1].content, 0)
		if decision.action == "refuse":
			return httpx.Response(403, headers={"X-Harness-Decision": "block"})
		return httpx.Response(200, json=(await Teacher().generate(payload)).model_dump(mode="json"))

	result = await collect_campaign(
		prompts_path=prompts,
		ledger_path=tmp_path / "ledger.jsonl",
		summary_path=tmp_path / "campaign.json",
		teacher=ProtectedTeacher(
			"http://fixture", "fixture", transport=httpx.MockTransport(response)
		),
		options=CollectionOptions(
			terms_review_id="fixture", max_output_tokens=16, enable_thinking=False
		),
		max_requests=7,
	)
	assert result["queries"] == result["attempt_budget"] == 7
	assert result["received"] == 3
	assert result["blocked"] == 4
	assert result["termination"] == "attempt_budget"
	events = [
		json.loads(line) for line in (tmp_path / "campaign.events.jsonl").read_text().splitlines()
	]
	assert [e["index"] for e in events if e["event"] == "attempt"] == list(range(1, 8))


def test_full_pool_report_joins_numeric_dimensions_and_reports_horizons(tmp_path: Path) -> None:
	from distillation.benchmarks.models import SampleResult, save_json
	from distillation.study_reporting import write_study_results

	budget = 53893
	save_json(
		tmp_path / "protocol.json",
		{
			"tasks": ["ag_news"],
			"seeds": [42],
			"budgets": "full",
			"arms": ["full"],
		},
	)
	save_json(tmp_path / "budgets.json", {"ag_news": [budget]})
	horizons: dict[str, object] = {
		"ag_news": {
			"attempts": 10000,
			"seconds": 86400,
			"campaign_budget_replenished_on_reset": False,
		}
	}
	save_json(tmp_path / "policy-horizons.json", horizons)
	save_json(tmp_path / "ag_news" / "recipe.json", {"transfer_gate_passed": True})
	sample = SampleResult(
		item_id="held-out",
		benchmark="ag_news",
		category="World",
		response="World",
		prediction="World",
		correct=True,
		status="ok",
		latency_seconds=0,
	)
	for arm, score in (("attacker", 90), ("protected-full", 60)):
		save_json(
			tmp_path / "ag_news" / f"s42-b{budget}" / f"report.{arm}.json",
			{
				"suite": {"sha256": "frozen"},
				"scores": {
					"student_before": {"ag_news": {"primary_score": 25}},
					"student_after": {"ag_news": {"primary_score": score}},
				},
				"runs": {
					"student_after": {
						"dimensions": {
							"task": "ag_news",
							"query_seed": 42,
							"training_seed": 42,
							"budget": budget,
						},
						"model": {"model": "fixture", "adapter_path": arm},
					}
				},
				"samples": [{"results": {"student_after": sample.model_dump(mode="json")}}],
			},
		)
	write_study_results(tmp_path)
	report = json.loads((tmp_path / "study-report.json").read_text())
	assert report["missing_pairs"] == []
	assert report["per_seed"][0]["budget"] == budget
	assert report["per_seed"][0]["Delta_D"] == 30
	assert report["study_kind"] == "separate_task"
	assert report["policy_horizons"] == horizons
	assert json.loads((tmp_path / "study-manifest.json").read_text())["complete"]
