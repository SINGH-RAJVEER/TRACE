"""Whole-study sequencing with real datasets, preparation, scoring, and HTTP adapters."""

import asyncio
import hashlib
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

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
from distillation.launcher import Launcher, run_app
from distillation.schemas import (
	Message,
	PromptRecord,
	SFTRecord,
	TeacherGeneration,
	TeacherGenerationRequest,
)
from distillation.study import Study
from distillation.tasks import build_task
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
		)

	async def close(self) -> None:
		pass


def test_start_checks_corrupt_ledgers_before_loading_the_teacher(
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
	Path("configs/train.task.toml").write_text('base_model="fixture"\n')
	ledger = Path("data/tasks/ag_news/ledger.jsonl")
	ledger.write_bytes(b"\x00" * 64 + b"\n")
	monkeypatch.setattr(Launcher, "install", lambda self, component: None)
	monkeypatch.setattr(
		Study, "ensure_teacher", lambda self: pytest.fail("teacher loaded before checking ledger")
	)

	def distill(self: Launcher, *args: str, env: dict[str, str] | None = None) -> None:
		result = CliRunner().invoke(app, list(args))
		assert result.exit_code == 0, result.output

	monkeypatch.setattr(Launcher, "distill", distill)
	result = CliRunner().invoke(
		run_app, ["start", "--tasks", "ag_news", "--seeds", "42", "--budgets", "1", "--no-gold"]
	)
	assert result.exit_code == 1
	assert "ledger.jsonl:1: invalid record" in result.output
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
	import distillation.benchmarks.traffic as traffic

	freeze_traffic = traffic.freeze_traffic
	monkeypatch.setattr(
		traffic, "freeze_traffic", lambda path: freeze_traffic(path, duration_seconds=0.01)
	)

	def serve(self: Study) -> None:
		services.append("teacher")
		steps.append("serve")

	def harness(self: Study, task: str, seed: int, calibration: Path) -> dict[str, str]:
		services.append("harness")
		steps.append("harness")
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
			assert "harness" in services
			steps.append("protected-collect")
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
			# The production runner validates adapters separately. This fixture uses its
			# real suite/scoring/report path while replacing model weights and optimization.
			config.models["student_after"].adapter_path = None
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
					}
				)
			)
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
	command = ["start", "--tasks", "ag_news", "--seeds", "42", "--budgets", "4", "--no-gold"]
	result = CliRunner().invoke(run_app, command)
	assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
	assert steps.index("train baseline") < steps.index("harness") < steps.index("train protected")
	assert steps.index("benchmark student_before") < steps.index("train baseline")
	root = Path("artifacts/start/ag_news/s42-b4")
	assert len(read_jsonl(root / "protected/train.jsonl", SFTRecord)) == 4
	assert (root / "report.attacker.json").is_file()
	assert (root / "report.protected.json").is_file()
	assert (root.parent / "s42/traffic-report.json").is_file()
	assert json.loads(Path("artifacts/start/status.json").read_text())["status"] == "complete"
	baseline = Path("data/tasks/ag_news/ledger.jsonl").read_bytes()
	steps.clear()
	result = CliRunner().invoke(run_app, command)
	assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
	assert "train baseline" not in steps and "train protected" not in steps
	assert Path("data/tasks/ag_news/ledger.jsonl").read_bytes() == baseline


@pytest.mark.asyncio
async def test_campaign_records_refusals_and_archives_interrupted_attempts(tmp_path: Path) -> None:
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
	broken = True

	async def response(request: httpx.Request) -> httpx.Response:
		payload = TeacherGenerationRequest.model_validate_json(request.content)
		question = payload.messages[-1].content
		if question == "1":
			return httpx.Response(403, headers={"X-Harness-Decision": "block"})
		if question == "2" and broken:
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
	result = await collect_campaign(
		prompts_path=prompts,
		ledger_path=ledger,
		summary_path=summary,
		teacher=teacher(),
		options=options,
	)
	assert result["queries"] == 3 and result["blocked"] == 1 and result["ledger_records"] == 2
	assert result["interrupted_attempts"] == 1
	archived = await asyncio.to_thread(
		lambda: next(tmp_path.glob("interrupted/*/ledger.jsonl")).read_bytes()
	)
	assert archived == original
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
	write_jsonl(study.run_dir("ag_news", 42, 1) / "protected/ledger.jsonl", [])

	def review(self: Launcher, *args: str, env: dict[str, str] | None = None) -> None:
		assert args[0] == "review"
		result = CliRunner().invoke(app, list(args[: args.index("--config")]))
		assert result.exit_code == 0, result.output

	monkeypatch.setattr(Launcher, "distill", review)
	assert not study.train_protected("ag_news", 42, 1, 1)
	assert not (study.run_dir("ag_news", 42, 1) / "protected/adapter").exists()
