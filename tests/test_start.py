"""Whole-study sequencing with real datasets, preparation, scoring, and HTTP adapters."""

import asyncio
import hashlib
import json
import os
import time
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
from distillation.proof import ProofSolver, SolveMetrics
from distillation.schemas import (
	Message,
	PromptRecord,
	SFTRecord,
	TeacherGeneration,
	TeacherGenerationRequest,
	TokenUsage,
)
from distillation.study import Study
from distillation.tasks import build_task, task_directory
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


def test_build_only_bootstrap_needs_datasets_without_gpu_or_vllm(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	import importlib.util

	path = Path(__file__).resolve().parents[1] / "scripts/pipeline.py"
	spec = importlib.util.spec_from_file_location("pipeline_bootstrap", path)
	assert spec and spec.loader
	bootstrap = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(bootstrap)
	commands: list[list[str]] = []
	monkeypatch.setattr(bootstrap, "run", commands.append)
	monkeypatch.setattr(bootstrap, "resolve_runtime", lambda: "uv")
	assert bootstrap.prepare(["eval"]) == 0
	assert commands == [["uv", "sync", "--all-packages", "--inexact", "--extra", "eval"]]
	monkeypatch.setattr(bootstrap, "load_dotenv", lambda: None)
	monkeypatch.setattr(bootstrap, "enter_devenv_if_nixos", lambda argv: None)
	monkeypatch.setenv("DISTILL_RUNTIME", "uv")
	needs: list[tuple[bool, bool]] = []

	def select(needs_gpu: bool, needs_vllm: bool) -> str:
		needs.append((needs_gpu, needs_vllm))
		return "uv"

	monkeypatch.setattr(bootstrap, "select_runtime", select)
	monkeypatch.setattr(bootstrap, "hand_off", lambda command: 0)
	assert bootstrap.main(["start", "--build-only"]) == 0
	assert needs == [(False, False)]


def test_start_build_only_parallelizes_coding_without_starting_models(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	import threading

	monkeypatch.chdir(tmp_path)
	monkeypatch.setenv("TEACHER_TERMS_REVIEW_ID", "fixture")
	Path("configs").mkdir()
	Path("configs/train.task.toml").write_text('base_model="fixture"\n')
	installed: list[str] = []
	builds: dict[str, tuple[str, ...]] = {}
	barrier = threading.Barrier(2, timeout=5)
	monkeypatch.setattr(Launcher, "install", lambda self, component: installed.append(component))
	monkeypatch.setattr(TaskPipeline, "budget", lambda self, task, seed, budget: None)
	monkeypatch.setattr(Study, "run", lambda self: pytest.fail("started study during build-only"))

	def distill(self: Launcher, *args: str, env: dict[str, str] | None = None) -> None:
		assert args[:2] == ("tasks", "build")
		task = args[args.index("--task") + 1]
		builds[task] = args
		barrier.wait()
		directory = Path(args[args.index("--output-dir") + 1])
		directory.mkdir(parents=True)
		(directory / "manifest.json").write_text(json.dumps({"roles": {"attacker": 10}}))

	monkeypatch.setattr(Launcher, "distill", distill)
	result = CliRunner().invoke(
		run_app,
		["start", "--build-only", "--tasks", "coding,gsm8k", "--coding-workers", "4"],
	)
	assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
	assert installed == ["eval"]
	assert builds["coding"][builds["coding"].index("--qualification-workers") + 1] == "4"
	assert "--qualification-workers" not in builds["gsm8k"]
	assert "--coding-workers" in CliRunner().invoke(run_app, ["distill", "--help"]).output


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
		directory = task_directory(task)
		directory.mkdir(parents=True)
		(directory / "manifest.json").write_text(
			json.dumps(
				{"roles": {"attacker": pool}, "corpus_limit": 50000 if task == "coding" else None}
			)
		)
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
		output_dir=task_directory(task),
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
	ledger = task_directory(task) / "ledger.jsonl"
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
			json.loads((task_directory(task) / "manifest.json").read_text())["roles"]["attacker"]
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
		output_dir=task_directory("ag_news"),
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
	monkeypatch.setattr(
		Study, "require_calibration", lambda self, task, path: steps.append("calibration gate")
	)
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
					resumable="--resumable" in args,
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
		"--arms",
		"full,instantaneous,quota",
	]
	result = CliRunner().invoke(run_app, command)
	assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
	harnesses = [step for step in steps if step.startswith("harness")]
	assert steps.index("calibration gate") < steps.index("serve")
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
async def test_resumable_campaign_keeps_its_clock_and_retries_missing_prompts(
	tmp_path: Path,
) -> None:
	from distillation.io import write_jsonl
	from distillation.schemas import PromptSource

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
	status = 502

	async def response(request: httpx.Request) -> httpx.Response:
		payload = TeacherGenerationRequest.model_validate_json(request.content)
		if payload.messages[-1].content == "1" and status != 200:
			return httpx.Response(status)
		return httpx.Response(200, json=(await Teacher().generate(payload)).model_dump(mode="json"))

	async def run(directory: str, deadline: float = 100) -> dict[str, Any]:
		return await collect_campaign(
			prompts_path=prompts,
			ledger_path=tmp_path / directory / "ledger.jsonl",
			summary_path=tmp_path / directory / "campaign.json",
			teacher=ProtectedTeacher(
				"http://fixture", "secret", transport=httpx.MockTransport(response)
			),
			options=options,
			deadline_seconds=deadline,
			resumable=True,
		)

	# A failed prompt is charged, then retried by the next pass on the same clock.
	first = await run("failed")
	assert (first["termination"], first["ledger_records"], first["failed"]) == (
		"prompts_exhausted",
		2,
		1,
	)
	status = 200
	second = await run("failed")
	assert (second["ledger_records"], second["queries"]) == (3, 4)
	assert (second["resumes"], second["retried_queries"]) == (1, 1)
	assert await run("failed") == second
	# An interruption resumes from its last recorded time, so the deadline still binds.
	status = 401
	with pytest.raises(httpx.HTTPStatusError):
		await run("interrupted")
	events = tmp_path / "interrupted/campaign.events.jsonl"
	lines = [json.loads(line) for line in events.read_text().splitlines()]
	lines[-1]["at_seconds"] = 150.0
	events.write_text("".join(json.dumps(line) + "\n" for line in lines) + '{"event": "att')
	status = 200
	resumed = await run("interrupted")
	assert resumed["termination"] == "deadline" and resumed["ledger_records"] == 1
	assert await run("interrupted") == resumed


def test_undefended_baseline_stops_when_any_prompt_lacks_a_response(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	from distillation.io import write_jsonl
	from distillation.schemas import PromptSource

	monkeypatch.chdir(tmp_path)
	pipeline = TaskPipeline(Launcher("test"), "fixture", ["ag_news"], [42], [2], tmp_path, 1, False)
	pipeline.plan = {"ag_news": [2]}
	study = Study(pipeline, tmp_path / "study", "", "", 8000, "", 8003, None, "refuse")
	prompts = pipeline.run_dir("ag_news", 42, 2) / "prompts.jsonl"
	write_jsonl(
		prompts,
		[
			PromptRecord(
				example_id=str(i),
				split="train",
				task="ag_news",
				source=PromptSource(corpus="fixture", source_id=str(i), license="mit"),
				prompt=[Message(role="user", content=str(i))],
			)
			for i in range(2)
		],
	)
	monkeypatch.setattr(TaskPipeline, "manifest", lambda self, task: {"max_output_tokens": 16})
	calls: list[tuple[str, ...]] = []

	def distill(self: Launcher, *args: str, env: dict[str, str] | None = None) -> None:
		calls.append(args)
		directory = Path(args[args.index("--output") + 1]).parent
		directory.mkdir(parents=True, exist_ok=True)
		(directory / "ledger.jsonl").write_text(json.dumps({"example_id": "0"}) + "\n")
		(directory / "campaign.json").write_text(
			json.dumps({"termination": "deadline", "blocked": 0})
		)

	monkeypatch.setattr(Launcher, "distill", distill)
	with pytest.raises(LaunchError, match="received 1 of 2 prompts \\(deadline\\)"):
		study.collect_undefended()
	# A deadline is not retried: no time is left in the declared envelope.
	assert len(calls) == 1 and "--resumable" in calls[0]
	status = json.loads((study.root / "status.json").read_text())
	assert status["status"] == "undefended_baseline_incomplete"
	assert (status["received"], status["expected"]) == (1, 2)


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
	qualified_target = capability_target(
		selection
		| {
			"qualification": {"base_score": 40.0, "score": 60.0},
		}
	)
	assert qualified_target["target_score"] == 50.0
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
@pytest.mark.parametrize("time_reset", [False, True])
async def test_refusals_and_time_resets_preserve_campaign_attempt_budget(
	tmp_path: Path, time_reset: bool
) -> None:
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
		now = (int(payload.messages[-1].content) // 3) * 86400 if time_reset else 0
		decision = policy.admit("extractor", payload.messages[-1].content, now)
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
	assert result["received"] == (3 if time_reset else 1)
	assert result["blocked"] == (4 if time_reset else 6)
	assert result["termination"] == "attempt_budget"
	events = [
		json.loads(line) for line in (tmp_path / "campaign.events.jsonl").read_text().splitlines()
	]
	assert [e["index"] for e in events if e["event"] == "attempt"] == list(range(1, 8))


@pytest.mark.parametrize("qualified", [False, True])
def test_full_pool_report_joins_numeric_dimensions_and_reports_horizons(
	tmp_path: Path,
	qualified: bool,
) -> None:
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
	save_json(
		tmp_path / "ag_news" / "recipe.json",
		{
			"analysis_version": "transfer-v2",
			"transfer_gate_passed": qualified,
			"qualification": {"passed": qualified, "minimum_gain_pp": 2},
		},
	)
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
				"samples": [
					{
						"results": {
							"student_before": sample.model_copy(
								update={"item_id": str(i), "correct": i < 5}
							).model_dump(mode="json"),
							"student_after": sample.model_copy(
								update={"item_id": str(i), "correct": i < score / 5}
							).model_dump(mode="json"),
						}
					}
					for i in range(20)
				],
			},
		)
	run = tmp_path / "ag_news" / f"s42-b{budget}"
	control = json.loads((run / "report.protected-full.json").read_text())
	save_json(run / "report.full-matched-volume.json", control)
	save_json(
		run / "protected" / "full" / "matched-prefix" / "control.json", {"training_examples": 0}
	)
	levels = {"L4": {"J": 7, "Q": 6, "reached": True, "censored": False}}
	save_json(run / "protected" / "full" / "budget-campaign.json", {"levels": levels})
	for phase, utility in (("before", 0.8), ("after", 0.6)):
		save_json(
			tmp_path / "ag_news" / "s42" / "full" / f"utility-{phase}.json",
			{"unconditional_utility": utility, "served_only_quality": 0.8}
			| dict.fromkeys(
				("submitted", "served", "completion_rate", "latency_p50", "latency_p95", "strata")
			),
		)
	save_json(
		tmp_path / "ag_news" / "benign-test" / "full.json",
		{
			"horizons": 10,
			"unscored_horizons": 2,
			"implemented_levels": [1, 2, 3, 4],
			"reached": {"L1": 1},
			"reached_rate": {"L1": 0.1},
		},
	)
	write_study_results(tmp_path)
	report = json.loads((tmp_path / "study-report.json").read_text())
	controls = report["per_seed"][0]["controls"]
	assert controls["matched-volume"]["score"] == 60
	assert controls["matched-volume"]["defended_minus_control"]["primary_delta_pp"] == 0
	assert controls["matched-prefix"]["status"] == "no_usable_examples"
	assert report["per_seed"][0]["escalation"]["levels"] == levels
	assert report["benign_utility"][0]["unconditional_utility_change"] == pytest.approx(-0.2)
	assert report["benign_utility"][0]["served_only_quality_change"] == 0
	assert report["benign_false_escalation"][0]["false_escalation_rate"] == {"L1": 0.1}
	assert report["missing_pairs"] == []
	assert report["per_seed"][0]["budget"] == budget
	assert report["per_seed"][0]["Delta_D"] == 30
	row = report["per_seed"][0]
	assert row["ratio_valid"] is qualified
	assert row["gain_removed"] == 30
	assert len(row["gain_removed_ci95_pp"]) == 2
	if qualified:
		assert row["R_D"] == row["retained_gain_fraction"] == pytest.approx(35 / 65)
		assert row["gain_removed_fraction"] == pytest.approx(30 / 65)
	else:
		assert row["R_D"] is row["gain_removed_fraction"] is None
		assert row["ratio_invalid_reason"] == "qualification_failed"
	assert report["study_kind"] == "separate_task"
	assert report["policy_horizons"] == horizons
	assert json.loads((tmp_path / "study-manifest.json").read_text())["complete"]


async def test_concurrent_proofs_use_processes_and_preserve_http_responsiveness() -> None:
	from harness.pow import leading_zero_bits

	solver = ProofSolver(2)
	clients: list[ProtectedTeacher] = []
	request = TeacherGenerationRequest(messages=[Message(role="user", content="proof")])
	proofs: list[int] = []

	async def response(http_request: httpx.Request) -> httpx.Response:
		proof = http_request.headers.get("X-Harness-Proof")
		if not proof:
			return httpx.Response(
				428,
				json={
					"challenge": {
						"prefix": "measurement-probe-",
						"nonce": "probe",
						"difficulty": 16,
						"expires_at": time.time() + 30,
					}
				},
			)
		counter = int(proof.split(":")[1])
		assert (
			leading_zero_bits(hashlib.sha256(f"measurement-probe-{counter}".encode()).digest())
			>= 16
		)
		proofs.append(counter)
		return httpx.Response(200, json=(await Teacher().generate(request)).model_dump(mode="json"))

	try:
		await solver.warmup()
		for _ in range(32):
			clients.append(
				ProtectedTeacher(
					"http://fixture", "", solver=solver, transport=httpx.MockTransport(response)
				)
			)
		tasks = [asyncio.create_task(client.generate(request)) for client in clients]
		await asyncio.sleep(0)
		# An unrelated HTTP client must remain runnable while proofs are outstanding.
		ordinary = ProtectedTeacher(
			"http://fixture",
			"",
			solver=solver,
			transport=httpx.MockTransport(
				lambda _: httpx.Response(403, headers={"X-Harness-Decision": "block"})
			),
		)
		try:
			from distillation.campaign import PolicyRefusal

			with pytest.raises(PolicyRefusal):
				await ordinary.generate(request)
			assert not all(task.done() for task in tasks)
		finally:
			await ordinary.close()
		await asyncio.gather(*tasks)
		assert len(proofs) == 32
		assert all(client.hashes == proofs[index] + 1 for index, client in enumerate(clients))
		for client in clients:
			event = next(e for e in client.events if e["event"] == "solve")
			assert event["status"] == "solved"
			assert event["queue_seconds"] >= 0
			assert event["cpu_seconds"] > 0
			assert event["worker_pids"] and os.getpid() not in event["worker_pids"]
	finally:
		await asyncio.gather(*(client.close() for client in clients))
		await solver.close()


async def test_proof_cancellation_drains_only_current_chunk_and_pool_remains_usable() -> None:
	solver = ProofSolver(1)
	try:
		await solver.warmup()
		metrics = SolveMetrics()
		task = asyncio.create_task(solver.solve("cancel", 256, time.monotonic() + 30, metrics))
		for _ in range(1000):
			if metrics.hashes:
				break
			await asyncio.sleep(0.001)
		assert metrics.hashes
		before = metrics.hashes
		task.cancel()
		with pytest.raises(asyncio.CancelledError):
			await task
		from distillation.proof import CHUNK_HASHES

		assert before <= metrics.hashes <= before + CHUNK_HASHES
		assert metrics.status == "cancelled"
		assert await solver.solve("next", 0, time.monotonic() + 5, SolveMetrics()) == 0
	finally:
		await solver.close()


@pytest.mark.parametrize("expiry", [True, False])
async def test_proof_stops_at_challenge_expiry_or_request_deadline(expiry: bool) -> None:
	from distillation.campaign import AttemptFailed, BudgetExhausted

	attempts = 0

	async def response(request: httpx.Request) -> httpx.Response:
		nonlocal attempts
		attempts += 1
		assert "X-Harness-Proof" not in request.headers
		return httpx.Response(
			428,
			json={
				"challenge": {
					"prefix": "expires",
					"nonce": "probe",
					"difficulty": 256,
					"expires_at": time.time() + (0.05 if expiry else 30),
				}
			},
		)

	teacher = ProtectedTeacher("http://fixture", "", transport=httpx.MockTransport(response))
	try:
		await teacher.solver.warmup()
		teacher.deadline = time.monotonic() + (30 if expiry else 0.05)
		with pytest.raises(AttemptFailed if expiry else BudgetExhausted):
			await teacher.generate(
				TeacherGenerationRequest(messages=[Message(role="user", content="x")])
			)
		assert attempts == 1
		event = next(e for e in teacher.events if e["event"] == "solve")
		assert event["status"] == "expired"
		assert event["hashes"] == teacher.hashes > 0
	finally:
		await teacher.close()


async def test_utility_scores_after_replay_and_freezes_solver_capacity(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	import distillation.utility as utility
	from distillation.benchmarks.models import BenchmarkItem
	from distillation.utility import measure_utility
	from harness.calibration import BenignRequest

	item = BenchmarkItem(
		id="one",
		benchmark="ag_news",
		category="World",
		question="q",
		messages=[Message(role="user", content="q")],
		metric="label_accuracy",
		answer="World",
		choices=["World", "Sports", "Business", "Sci/Tech"],
	)
	calls = 0
	from distillation.benchmarks.scoring import evaluate_answer as original_score

	async def response(request: httpx.Request) -> httpx.Response:
		nonlocal calls
		calls += 1
		payload = TeacherGenerationRequest.model_validate_json(request.content)
		return httpx.Response(200, json=(await Teacher().generate(payload)).model_dump(mode="json"))

	async def score_after_replay(item: BenchmarkItem, text: str, finish: str | None) -> Any:
		assert calls == 7  # Five warmup requests and both timed requests finish before scoring.
		await asyncio.sleep(0.01)
		return await original_score(item, text, finish)

	monkeypatch.setattr(utility, "evaluate_answer", score_after_replay)
	kwargs: dict[str, Any] = dict(
		requests=[
			BenignRequest(account="a", text="q", time=0, example_id="one"),
			BenignRequest(account="a", text="q", time=0, example_id="one"),
		],
		references={"one": item},
		url="http://fixture",
		keys={},
		max_output_tokens=16,
		output=tmp_path / "utility.json",
		solver_workers=1,
		transport=httpx.MockTransport(response),
	)
	result = await measure_utility(**kwargs)
	assert result["served"] == 2
	assert result["unconditional_utility"] == 1
	assert result["solver"]["workers"] == 1
	assert result["scoring_seconds"] >= 0.02
	for row in result["results"]:
		assert row["latency_seconds"] == pytest.approx(
			row["arrival_delay_seconds"] + row["service_and_queue_seconds"]
		)
		assert row["proof_seconds"] == row["proof_queue_seconds"] == row["hashes"] == 0
	assert await measure_utility(**kwargs) == result
	kwargs["solver_workers"] = 2
	with pytest.raises(ValueError, match="configuration changed"):
		await measure_utility(**kwargs)


@pytest.mark.parametrize("cached", [True, False])
def test_vllm_serves_the_pinned_snapshot_offline(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached: bool
) -> None:
	import distillation.launcher as launcher

	monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
	monkeypatch.setattr(launcher, "_find_vllm", lambda: "vllm")
	monkeypatch.setattr(launcher, "_cached_snapshot", lambda model, revision: cached)
	downloads: list[tuple[str, ...]] = []
	monkeypatch.setattr(Launcher, "distill", lambda self, *args, **kwargs: downloads.append(args))
	started: dict[str, dict[str, str]] = {}
	monkeypatch.setattr(
		Launcher, "start", lambda self, name, command, log, env: started.setdefault(name, env)
	)
	monkeypatch.setattr(Launcher, "wait_for_http", lambda *args: None)
	monkeypatch.setattr(Launcher, "smoke_test", lambda *args: None)
	Launcher("test").start_teacher_stack("127.0.0.1", 8000, "", tmp_path / "api.log")
	# vLLM's startup file listing must not need the network once the snapshot is local.
	assert started["vllm"]["HF_HUB_OFFLINE"] == "1"
	assert "HF_HUB_OFFLINE" not in started["teacher-api"]
	assert [args[0] for args in downloads] == ([] if cached else ["download-model"])
