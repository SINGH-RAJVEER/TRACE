"""Shared execution uses native HTTP, policy, preparation, scoring, and artifact joins."""

import asyncio
import hashlib
import json
import socket
import threading
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
import uvicorn
from distillation.benchmarks.models import (
	BenchmarkItem,
	DatasetSource,
	FrozenSuite,
	Slot,
	load_config,
	save_json,
)
from distillation.benchmarks.reporting import discover_runs
from distillation.benchmarks.runner import run_evaluation
from distillation.external_sources import EXTERNAL_SOURCES, build_external
from distillation.gateway_study import GatewayStudy, benign_requests
from distillation.io import read_jsonl, write_jsonl
from distillation.launcher import (
	DEFAULT_CODING_WORKERS,
	Launcher,
	LaunchError,
	TaskPipeline,
	run_app,
)
from distillation.review import review_ledger
from distillation.schemas import (
	GatewayRequest,
	Message,
	PromptRecord,
	PromptSource,
	SFTRecord,
	TeacherGeneration,
	TeacherGenerationRequest,
	TokenUsage,
)
from distillation.study import Study
from distillation.tasks import (
	AG_NEWS_PROMPT,
	GSM8K_EVAL_SUFFIX,
	LEGITIMATE_ROLES,
	GoldRecord,
	task_directory,
)
from distillation.teacher import TeacherAPIGateway
from distillation.training import TrainingConfig
from fastapi import FastAPI
from harness.calibration import HarnessConfig, calibrate, load_calibration, save_calibration
from harness_api.app import create_app
from harness_api.settings import Settings
from test_harness_api import Encoder
from typer.testing import CliRunner


class Server:
	def __init__(self, app: FastAPI) -> None:
		self.socket = socket.socket()
		self.socket.bind(("127.0.0.1", 0))
		self.port = self.socket.getsockname()[1]
		self.server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
		self.thread = threading.Thread(
			target=self.server.run, kwargs={"sockets": [self.socket]}, daemon=True
		)
		self.thread.start()
		deadline = time.monotonic() + 5
		while not self.server.started:
			if not self.thread.is_alive() or time.monotonic() > deadline:
				raise RuntimeError("test HTTP server did not start")
			time.sleep(0.01)

	def close(self) -> None:
		self.server.should_exit = True
		self.thread.join(timeout=5)
		self.socket.close()
		assert not self.thread.is_alive()


class Inference:
	def __init__(self, correct: bool = True) -> None:
		self.correct = correct

	async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
		text = request.messages[-1].content
		answer = "World" if "ag_news" in text else "#### 7"
		return TeacherGeneration(
			id="fixture",
			provider="fixture",
			model="fixture",
			generated_at=datetime.now(UTC),
			endpoint="fixture",
			message=Message(role="assistant", content=answer if self.correct else "wrong"),
			finish_reason="stop",
			usage=TokenUsage(output_tokens=2, input_tokens=5),
		)

	async def close(self) -> None:
		pass


@pytest.mark.parametrize("command", ["start", "distill"])
@pytest.mark.parametrize("workers", [None, 1, 3])
def test_launchers_default_to_shared_students_with_math_and_coding(
	command: str, workers: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
	seen = []

	def run(self: GatewayStudy, *, collect_only: bool = False) -> None:
		assert self.study.pipeline.coding_pool_size == 50000
		assert self.study.pipeline.task_dir("coding") == Path("data/tasks/coding-subset-50000-s42")
		assert self.study.root == Path(
			"artifacts/gateway-distill-full-v1"
			if command == "distill"
			else "artifacts/gateway-full-v1"
		)
		assert self.study.pipeline.coding_workers == (
			DEFAULT_CODING_WORKERS if workers is None else workers
		)
		seen.append(
			(
				self.study.pipeline.tasks,
				collect_only,
				self.study.pipeline.budgets,
				self.study.pipeline.seeds,
				self.study.arms,
				self.attacks,
			)
		)

	monkeypatch.setattr(GatewayStudy, "run", run)
	args = [command] + (["--coding-workers", str(workers)] if workers is not None else [])
	result = CliRunner().invoke(run_app, args)
	assert result.exit_code == 0, result.exception
	assert seen == [
		(
			["ag_news", "squad", "gsm8k", "coding"],
			command == "distill",
			None,
			[42],
			("full",),
			("A1",) if command == "distill" else ("A1", "A2", "A5", "A6", "A7"),
		)
	]


@pytest.mark.parametrize("collect_only,qualifies", [(False, True), (False, False), (True, True)])
def test_shared_study_runs_http_campaigns_and_trains_one_adapter_for_every_domain(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collect_only: bool, qualifies: bool
) -> None:
	monkeypatch.chdir(tmp_path)
	# This small fixture verifies HTTP and training sequencing, not calibration adequacy.
	calibration_checks: list[str] = []
	monkeypatch.setattr(
		Study, "require_calibration", lambda self, task, path: calibration_checks.append(task)
	)
	config = tmp_path / "train.toml"
	config.write_text('base_model="fixture"\nbase_revision="' + "a" * 40 + '"\n')
	for domain, cap in (("ag_news", 16), ("gsm8k", 256)):
		directory = task_directory(domain, root=tmp_path / "data/tasks")
		manifest = {"task": domain, "max_output_tokens": cap, "roles": {"attacker": 20}}
		save_json(directory / "manifest.json", manifest)
		references = {}
		for role in ("attacker", *LEGITIMATE_ROLES):
			rows = [
				PromptRecord(
					example_id=f"{domain}-{role}-{i}",
					task=domain,
					split="train",
					source=PromptSource(
						corpus=domain,
						source_id=f"{role}-{i}",
						group_id=f"{role}-{i}",
						license="fixture",
					),
					prompt=[
						Message(
							role="user",
							content=AG_NEWS_PROMPT.format(text=f"{domain} {role} input {i}")
							if domain == "ag_news"
							else f"{domain} {role} input {i}{GSM8K_EVAL_SUFFIX}",
						)
					],
				)
				for i in range(20)
			]
			write_jsonl(
				directory / "attacker.jsonl"
				if role == "attacker"
				else directory / "legitimate" / f"{role}.jsonl",
				rows,
			)
			if role == "attacker":
				write_jsonl(
					directory / "gold-answers.jsonl",
					[
						GoldRecord(
							example_id=r.example_id,
							completion="World" if domain == "ag_news" else "#### 7",
						)
						for r in rows
					],
				)
			else:
				for row in rows:
					references[row.example_id] = BenchmarkItem(
						id=row.example_id,
						benchmark=domain,
						category="topic",
						question=row.prompt[0].content,
						messages=row.prompt,
						metric="label_accuracy" if domain == "ag_news" else "numeric_exact_match",
						answer="World" if domain == "ag_news" else "7",
						choices=["World", "Sports"] if domain == "ag_news" else [],
					)
		save_json(
			directory / "legitimate-references.json",
			{k: v.model_dump(mode="json") for k, v in references.items()},
		)
		for split in ("development", "test"):
			question = f"{domain} held out {split}"
			suite = FrozenSuite(
				seed=42,
				created_at=datetime.now(UTC),
				sources=[
					DatasetSource(
						benchmark=domain,
						repository="fixture",
						revision="a" * 40,
						configuration="default",
						split=split,
						license="fixture",
						url="https://example.com",
						available=4,
						selected=4,
					)
				],
				items=[
					BenchmarkItem(
						id=f"{domain}-{split}-{index}",
						benchmark=domain,
						category="topic",
						question=question,
						messages=[
							Message(
								role="user",
								content=AG_NEWS_PROMPT.format(text=question)
								if domain == "ag_news"
								else f"{question}{GSM8K_EVAL_SUFFIX}",
							)
						],
						metric="label_accuracy" if domain == "ag_news" else "numeric_exact_match",
						answer="World" if domain == "ag_news" else "7",
						choices=["World", "Sports"] if domain == "ag_news" else [],
					)
					for index in range(4)
				],
			)
			suite.sha256 = suite.digest()
			save_json(directory / f"{split}-suite.json", suite)
	services: list[Server] = []
	all_servers: list[Server] = []
	training_datasets: list[Path] = []

	def ensure_teacher(self: Study) -> None:
		if services:
			return
		app = FastAPI()

		@app.post("/v1/teacher/generate")
		async def generate(request: TeacherGenerationRequest) -> TeacherGeneration:
			return await Inference().generate(request)

		server = Server(app)
		services.append(server)
		all_servers.append(server)
		self.teacher_url = f"http://127.0.0.1:{server.port}"
		self.pipeline.launcher.background.append(cast(Any, server))

	def stop(self: Launcher, keep: int = 0) -> None:
		for server in reversed(services[keep:]):
			server.close()
		del services[keep:]
		del self.background[keep:]

	def start(
		self: Launcher, name: str, command: list[str], log_path: Path | None, env: dict[str, str]
	) -> Any:
		calibration, scorer = load_calibration(Path(env["HARNESS_CALIBRATION_DIR"]))
		with monkeypatch.context() as context:
			for key, value in env.items():
				context.setenv(key, value)
			settings = Settings(_env_file=None)  # type: ignore[call-arg]
		app = create_app(
			settings=settings,
			teacher=TeacherAPIGateway(base_url=env["HARNESS_TEACHER_URL"], token=""),
			encoder=Encoder(),
			calibration=calibration,
			scorer=scorer,
		)
		server = Server(app)
		study.harness_port = server.port
		services.append(server)
		all_servers.append(server)
		self.background.append(cast(Any, server))
		return server

	def small_calibration(self: GatewayStudy) -> Path:
		path = self.root / "calibration"
		if (path / "calibration.json").exists():
			return path
		groups = {
			r: benign_requests(self.benign[r]) for r in LEGITIMATE_ROLES if r != "benign_test"
		}
		calibration, refs = calibrate(
			config=HarnessConfig(
				task="gateway", window_size=2, reference_windows=2, aggregate_enabled=False
			),
			encoder=Encoder(),
			fitting=groups["fitting"],
			score_calibration=groups["score_calibration"],
			policy_calibration=groups["policy_calibration"],
		)
		calibration.baselines["quota"].thresholds = [2.0]
		save_calibration(path, calibration, refs)
		return path

	def command(self: Launcher, *args: str, env: dict[str, str] | None = None) -> None:
		def option(name: str) -> Path:
			return Path(args[args.index(name) + 1])

		if args[:2] == ("gateway", "external"):
			domain = option("--task")
			build_external(
				task=str(domain),
				task_dir=task_directory(str(domain), root=option("--tasks-dir")),
				rows={
					"train": [
						{"text": f"bbc ag_news story {i}"}
						if str(domain) == "ag_news"
						else {"question_without_options": f"gsm8k external problem {i}"}
						for i in range(20)
					]
				},
			)
		elif args[0] == "review":
			review_ledger(ledger_path=option("--ledger"), output_path=option("--output"))
		elif args[0] == "train":
			assert not services, "inference services must stop before training"
			if "--preflight-only" in args:
				return
			values = tomllib.loads(option("--config").read_text())
			dataset = Path(values["dataset_path"])
			training_datasets.append(dataset)
			adapter = Path(values["output_dir"])
			save_json(
				adapter / "training-manifest.json",
				{
					"config": values,
					"dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
					"fingerprint": TrainingConfig.model_validate(values).model_dump(mode="json"),
				},
			)
			(adapter / "adapter_config.json").write_text("{}")
			(adapter / "adapter_model.safetensors").write_bytes(b"fixture weights")
		elif args[:2] == ("benchmark", "run"):
			settings = load_config(option("--config"))
			slot = cast(Slot, args[args.index("--slot") + 1])
			assert bool(services) == (slot in {"teacher", "teacher_after"})
			asyncio.run(
				run_evaluation(
					settings,
					slot,
					factory=lambda model: Inference(
						(bool(model.adapter_path) or model.backend == "teacher_api")
						and (qualifies or "qualification" not in settings.experiment)
					),
				)
			)
		else:
			raise AssertionError(args)

	monkeypatch.setattr(Launcher, "install", lambda *args: None)
	monkeypatch.setattr(Launcher, "distill", command)
	monkeypatch.setattr(Launcher, "start", start)
	monkeypatch.setattr(Launcher, "stop_background", stop)
	monkeypatch.setattr(Launcher, "wait_for_http", lambda *args: None)
	monkeypatch.setattr(Study, "ensure_teacher", ensure_teacher)
	monkeypatch.setattr(GatewayStudy, "calibration", small_calibration)
	monkeypatch.setattr("distillation.gateway_study.BGEEncoder", lambda **kwargs: Encoder())
	monkeypatch.setattr(
		"distillation.training.training_identity", lambda c: c.model_dump(mode="json")
	)
	pipeline = TaskPipeline(
		Launcher("fixture"), "fixture", ["ag_news", "gsm8k"], [42], [8], config, 2, True
	)
	study = Study(
		pipeline, tmp_path / "run", "", "", 8000, "", 8003, None, "refuse", arms=("quota",)
	)
	gateway = GatewayStudy(
		study,
		attacks=("A1", "A5", "A7"),
		interval_seconds=0,
		benign_accounts=4,
		requests_per_account=2,
		development_budget=8,
	)
	try:
		if qualifies:
			gateway.run(collect_only=collect_only)
		else:
			with pytest.raises(LaunchError, match="qualification"):
				gateway.run()
	finally:
		stop(pipeline.launcher)
	assert calibration_checks == ([] if collect_only else ["gateway"])
	if not qualifies:
		assert json.loads((gateway.root / "status.json").read_text())["status"] == "failed"
		assert len(training_datasets) == 6
		recipe = json.loads((gateway.root / "recipe.json").read_text())
		assert recipe["transfer_gate_passed"] is False
		assert recipe["informative_domains"] == []
		assert set(recipe["uninformative_domains"]) == set(pipeline.tasks)
		assert all(set(t["ranks"]) == set(pipeline.tasks) for t in recipe["trials"])
		assert all(q["gain_pp"] == 0 for q in recipe["qualification"].values())
		assert not (gateway.root / "results.json").exists()
		return
	assert json.loads((gateway.root / "status.json").read_text())["status"] == "complete"
	assert not services
	assert all(not s.thread.is_alive() for s in all_servers)
	if collect_only:
		# A1 and A5 train attacker and gold students; A7 has no gold answers.
		assert len(training_datasets) == 5
		for dataset in training_datasets:
			prepared = read_jsonl(dataset, SFTRecord)
			assert len(prepared) == 8
			assert all(any(d in r.prompt[0].content for r in prepared) for d in pipeline.tasks)
		assert not (gateway.root / "calibration").exists()
		return
	results = json.loads((gateway.root / "results.json").read_text())["results"]
	assert len(results) == 6
	assert all(r["ratio_valid"] for r in results)
	assert all(r["undefended_gain_ci95_pp"] == [100.0, 100.0] for r in results)
	recipe = json.loads((gateway.root / "recipe.json").read_text())
	assert all(q["passed"] for q in recipe["qualification"].values())
	for domain in pipeline.tasks:
		selection = load_config(gateway.root / "development/lr0.0001-e1" / f"eval.{domain}.toml")
		qualification = load_config(
			gateway.root / "development/qualification" / f"eval.{domain}.toml"
		)
		assert selection.suite_path != qualification.suite_path
		assert recipe["qualification"][domain]["minimum_gain_pp"] == 2.0
	assert all(set(r["controls"]) == {"matched-volume", "matched-prefix"} for r in results)
	assert all(r["http"]["queries"] >= 1 for r in results)
	assert all(set(r["first_intervention"]) == {"L1", "L2", "L3", "L4"} for r in results)
	assert all(Path(r["timing"]).exists() for r in results)
	timing = json.loads(Path(results[0]["timing"]).read_text())
	assert timing["latency_seconds"]["received"]["n"] >= 1
	assert {r["domain"] for r in results} == {"ag_news", "gsm8k"}
	# The untrained student is benchmarked once per seed and budget for every final condition.
	for domain in pipeline.tasks:
		before = [
			r
			for r in discover_runs(gateway.root / "base-student/s42-b8/benchmarks" / domain)[0]
			if r.slot == "student_before"
		]
		assert len(before) == 1
		ids = {
			json.loads((d / "scores.json").read_text())[domain]["evaluation_ids"]["student_before"]
			for d in (gateway.root / "A1/s42-b8/undefended", gateway.root / "A5/s42-b8/gold")
		}
		assert ids == {before[0].id}
	# The bare teacher is benchmarked once and its scores are shared by every campaign.
	assert not (gateway.root / "A5/s42-b8/teacher/benchmarks").exists()
	for attack in ("A5", "A7"):
		for name in ("teacher-scores.json", "teacher-after-scores.json"):
			assert (gateway.root / attack / "s42-b8/teacher" / name).read_bytes() == (
				gateway.root / "A1/s42-b8/teacher" / name
			).read_bytes()
	for attack in ("A1", "A5"):
		directory = gateway.root / attack / "s42-b8"
		assert len(read_jsonl(directory / "undefended/train.jsonl", SFTRecord)) == 8
		assert len(read_jsonl(directory / "gold/train.jsonl", SFTRecord)) == 8
		for condition in ("undefended", "quota/campaign"):
			models = [
				load_config(directory / condition / f"eval.{d}.toml")
				.models["student_after"]
				.adapter_path
				for d in pipeline.tasks
			]
			assert models[0] == models[1] == directory / condition / "adapter"
			prepared = read_jsonl(directory / condition / "train.jsonl", SFTRecord)
			assert {
				d for d in pipeline.tasks if any(d in r.prompt[0].content for r in prepared)
			} == set(pipeline.tasks)
	assert len([p for p in training_datasets if "development" in str(p)]) == 6
	independent = gateway.root / "A7" / "s42-b8"
	assert not (independent / "gold").exists()
	inventory = json.loads((independent / "inventory.json").read_text())
	assert inventory["prompt_sources"] == "independent external"
	assert all(
		r.prompt.source.corpus == EXTERNAL_SOURCES[r.domain].dataset.repository
		for r in read_jsonl(independent / "requests.jsonl", GatewayRequest)
	)


def test_shared_study_rejects_campaigns_stopped_by_service_failures(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	from types import SimpleNamespace

	async def collect(**kwargs: Any) -> dict[str, Any]:
		return {"termination_by_account": {"a": "prompts_exhausted", "b": "service_failures"}}

	monkeypatch.setattr("distillation.gateway_study.collect_gateway", collect)
	study = SimpleNamespace(
		protected_url="http://harness",
		url="http://teacher",
		token="fixture",
		pipeline=SimpleNamespace(terms_review_id="fixture", concurrency=1),
	)
	owner = cast(GatewayStudy, SimpleNamespace(study=study, deadline_seconds=1.0))
	with pytest.raises(LaunchError, match="teacher service failed for b"):
		GatewayStudy.collect(owner, tmp_path / "requests.jsonl", tmp_path / "campaign", 1, 42)


def test_recipe_selection_ranks_domains_instead_of_averaging_raw_scores() -> None:
	from distillation.gateway_study import rank_trials

	def trial(rate: float, epochs: int, *gains: float) -> dict[str, Any]:
		return {
			"recipe": {"learning_rate": rate, "epochs": epochs},
			"gains": dict(zip(("ag_news", "squad", "coding"), gains, strict=True)),
		}

	domains = ["ag_news", "squad", "coding"]
	# A raw mean picks the large coding swing (14 > 7); ranks favor the recipe that is better
	# in two of three domains.
	trials = [trial(1e-4, 1, 10.0, 10.0, 1.0), trial(2e-4, 1, 1.0, 1.0, 40.0)]
	winner = rank_trials(trials, domains)
	assert winner["recipe"] == {"learning_rate": 1e-4, "epochs": 1}
	assert winner["ranks"] == {"ag_news": 1.0, "squad": 1.0, "coding": 2.0}
	tied = [trial(2e-4, 3, 1.0, 1.0, 1.0), trial(1e-4, 3, 1.0, 1.0, 1.0)]
	assert rank_trials(tied, domains)["recipe"]["learning_rate"] == 1e-4
	assert all(t["mean_rank"] == 1.5 for t in tied)
