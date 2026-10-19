"""Whole-pipeline scenarios: serve the teacher, collect, review, prepare, and train.

scripts/pipeline.py loads .env, installs the base Python environment, and then runs these
commands, which call it back to install the training stack and vLLM once the options are
valid. Each long step runs as a `distill` subprocess that draws its own progress, and services
started here run in their own process group so they can be stopped as a whole.
"""

import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Annotated, Any, Literal

import httpx
import typer
from rich.text import Text

from distillation.progress import console

DEFAULT_TEACHER_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_TEACHER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TASK_CONFIG = Path("configs/train.task.toml")
TASK_ROOT = Path("data/tasks")
TASK_ARTIFACTS = Path("artifacts/tasks")
# The keys of distillation.tasks.TASKS, which imports the dataset stack.
TASK_NAMES = ("ag_news", "squad", "gsm8k", "coding")
DEFAULT_TASKS = ",".join(TASK_NAMES)
LOG_DIR = Path("artifacts/logs")
PIPELINE_SCRIPT = Path("scripts/pipeline.py")
STOP_SECONDS = 30
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# vLLM writes " ERROR ", uvicorn "ERROR:", and tracebacks end in "SomeError: message".
ERROR_LINE = re.compile(r"\bERROR\b|(Error|Exception)(:|$)")

run_app = typer.Typer(no_args_is_help=True)


class LaunchError(Exception):
	"""A step failed; the message says what to fix."""


class Launcher:
	"""Logs, foreground steps, and background services for one scenario."""

	def __init__(self, name: str) -> None:
		self.name = name
		self.background: list[tuple[str, subprocess.Popen[bytes]]] = []

	def log(self, message: str, style: str = "") -> None:
		console.print(Text.assemble((f"[{self.name}] ", "bold cyan"), (message, style)))

	def run(self, name: str, command: list[str], env: dict[str, str] | None = None) -> None:
		"""Runs a command in the foreground with the terminal attached.

		Ctrl+C reaches the child too. This waits for it to exit, so training can save a
		checkpoint, and then stops the scenario.
		"""
		process = subprocess.Popen(command, env={**os.environ, **(env or {})})
		interrupted = False
		while True:
			try:
				code = process.wait()
				break
			except KeyboardInterrupt:
				interrupted = True
		if interrupted or code == 130:
			raise KeyboardInterrupt
		if code:
			raise LaunchError(f"{name} exited with code {code}")

	def distill(self, *args: str, env: dict[str, str] | None = None) -> None:
		self.run(f"distill {args[0]}", [sys.executable, "-m", "distillation.cli", *args], env)

	def install(self, component: str) -> None:
		"""Installs the training stack (train) or vLLM (vllm) with scripts/pipeline.py."""
		if not PIPELINE_SCRIPT.is_file():
			raise LaunchError(f"{PIPELINE_SCRIPT} not found; run from the repository root")
		self.run(
			"scripts/pipeline.py", [sys.executable, str(PIPELINE_SCRIPT), "--prepare", component]
		)

	def start(
		self, name: str, command: list[str], log_path: Path | None, env: dict[str, str]
	) -> subprocess.Popen[bytes]:
		"""Starts a service in its own process group, logging to a file or the terminal."""
		options: dict[str, Any] = {"env": {**os.environ, **env}}
		if sys.platform == "win32":
			options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
		else:
			# A new session keeps Ctrl+C away from the service and makes vLLM's engine
			# processes one group that stop_background can signal together.
			options["start_new_session"] = True
		if log_path:
			LOG_DIR.mkdir(parents=True, exist_ok=True)
			with log_path.open("wb") as output:
				process = subprocess.Popen(
					command, stdout=output, stderr=subprocess.STDOUT, **options
				)
		else:
			process = subprocess.Popen(command, **options)
		self.background.append((name, process))
		self.log(f"started {name} (pid {process.pid})")
		return process

	def stop_background(self, *, keep: int = 0) -> None:
		"""Stops services newest first: terminate, then kill after 30 seconds."""
		while len(self.background) > keep:
			name, process = self.background.pop()
			if process.poll() is not None:
				continue
			self.log(f"stopping {name} (pid {process.pid})")
			_signal_group(process, kill=False)
			try:
				process.wait(STOP_SECONDS)
			except subprocess.TimeoutExpired:
				_signal_group(process, kill=True)
				process.wait()

	def wait_for_http(
		self,
		url: str,
		expected: str,
		timeout: float,
		label: str,
		process: subprocess.Popen[bytes] | None = None,
		log_path: Path | None = None,
	) -> None:
		"""Waits for a 2xx response containing expected, showing the service's latest log line."""
		self.log(f"waiting for {url}")
		started = time.monotonic()
		next_check = started
		next_note = started + 30
		with _status() as status:
			while (elapsed := time.monotonic() - started) < timeout:
				if time.monotonic() >= next_check:
					next_check = time.monotonic() + 3
					try:
						response = httpx.get(url, timeout=5)
						if response.is_success and expected in response.text:
							return
					except httpx.HTTPError:
						pass
				if process is not None and process.poll() is not None:
					status(None)
					if log_path:
						self._show_failure(log_path)
					raise LaunchError(f"process for {url} exited during startup")
				line = _last_line(log_path) if log_path else ""
				if console.is_terminal:
					status(
						Text.assemble(
							(label, "bold"),
							f" {_duration(elapsed)}  ",
							(line, "dim"),
							no_wrap=True,
							overflow="ellipsis",
						)
					)
				elif time.monotonic() >= next_note:
					next_note += 30
					self.log(f"{label} {_duration(elapsed)}  {line}".rstrip())
				time.sleep(0.5)
		raise LaunchError(f"timed out after {timeout:.0f}s waiting for {url}")

	def _show_failure(self, log_path: Path) -> None:
		lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
		console.print("\n".join(lines[-40:]), markup=False, highlight=False)
		# Child processes such as the vLLM engine core log the root cause well above the
		# tail, so repeat the error lines from the whole log.
		errors = [line for line in lines if ERROR_LINE.search(line)]
		if errors:
			self.log(f"error lines from {log_path}:")
			console.print("\n".join(errors[-20:]), markup=False, highlight=False)

	def smoke_test(self, teacher_url: str, token: str) -> None:
		"""Sends one small generation through the teacher API to check auth and non-thinking."""
		headers = {"Authorization": f"Bearer {token}"} if token else {}
		with console.status("Sending a test generation through the teacher API"):
			try:
				response = httpx.post(
					f"{teacher_url}/v1/teacher/generate",
					headers=headers,
					json={
						"messages": [
							{"role": "user", "content": "Reply with the single word: ready"}
						],
						"temperature": 0,
						"max_output_tokens": 32,
					},
					timeout=300,
				)
			except httpx.HTTPError as error:
				raise LaunchError(f"teacher smoke test failed: {error}") from error
		if response.status_code != 200:
			console.print(response.text, markup=False)
			raise LaunchError(f"teacher smoke test failed with HTTP {response.status_code}")
		if "<think>" in response.text:
			raise LaunchError(
				"teacher response contains a <think> trace; disable Qwen thinking mode upstream"
			)
		self.log("teacher smoke test passed")

	def start_teacher_stack(
		self, host: str, port: int, upstream_url: str, api_log: Path | None
	) -> None:
		"""Starts vLLM (unless given an upstream URL) and the teacher API, then checks both."""
		model = os.environ.get("TEACHER_UPSTREAM_MODEL") or DEFAULT_TEACHER_MODEL
		env = {"TEACHER_UPSTREAM_MODEL": model}
		if not upstream_url:
			vllm = _find_vllm()
			if not vllm:
				self.install("vllm")
				vllm = _find_vllm()
			if not vllm:
				raise LaunchError("vLLM installation did not produce .venv-vllm/bin/vllm")
			revision = os.environ.get("TEACHER_MODEL_REVISION") or DEFAULT_TEACHER_REVISION
			if Path(model).exists() or os.environ.get("HF_HUB_OFFLINE", "") in {"1", "true"}:
				self.log(f"using {model} without downloading")
			else:
				self.log(f"downloading {model} at {revision[:12]} into the Hugging Face cache")
				self.distill("download-model", model, "--revision", revision)
			vllm_port = os.environ.get("VLLM_PORT") or "8001"
			vllm_log = LOG_DIR / "vllm.log"
			# A CPU buffer for KV blocks evicted from the GPU, so repeated prefixes such as
			# shared system prompts are reloaded from RAM instead of recomputed.
			offload = []
			if offload_size := os.environ.get("VLLM_KV_OFFLOADING_SIZE"):
				offload = [
					"--kv-offloading-size",
					offload_size,
					"--kv-offloading-backend",
					os.environ.get("VLLM_KV_OFFLOADING_BACKEND") or "native",
				]
			# Qwen3.5 checkpoints include a vision encoder; text-only serving skips it and its
			# profiling, leaving that memory for the KV cache. No reasoning parser: it would
			# move <think> text out of the content that the smoke test and collector check.
			process = self.start(
				"vllm",
				[
					vllm,
					"serve",
					model,
					"--revision",
					revision,
					"--served-model-name",
					model,
					"--host",
					"127.0.0.1",
					"--port",
					vllm_port,
					"--max-model-len",
					os.environ.get("VLLM_MAX_MODEL_LEN") or "8192",
					"--gpu-memory-utilization",
					os.environ.get("VLLM_GPU_MEMORY_UTILIZATION") or "0.90",
					"--default-chat-template-kwargs",
					'{"enable_thinking": false}',
					"--language-model-only",
					*offload,
					*shlex.split(os.environ.get("VLLM_EXTRA_ARGS", "")),
				],
				vllm_log,
				_vllm_env(),
			)
			self.log(f"vLLM log: {vllm_log}")
			self.wait_for_http(
				f"http://127.0.0.1:{vllm_port}/health",
				"",
				float(os.environ.get("VLLM_STARTUP_TIMEOUT") or 1800),
				"Starting vLLM",
				process,
				vllm_log,
			)
			upstream_url = f"http://127.0.0.1:{vllm_port}/v1"
			env["TEACHER_PROVIDER_NAME"] = (
				os.environ.get("TEACHER_PROVIDER_NAME") or "self-hosted-vllm"
			)
		env |= {
			"TEACHER_UPSTREAM_BASE_URL": upstream_url,
			"TEACHER_HOST": host,
			"TEACHER_PORT": str(port),
		}
		process = self.start("teacher-api", [sys.executable, "-m", "teacher_api"], api_log, env)
		self.wait_for_http(
			f"http://127.0.0.1:{port}/healthz",
			'"ready"',
			120,
			"Starting the teacher API",
			process,
			api_log,
		)
		self.smoke_test(f"http://127.0.0.1:{port}", os.environ.get("TEACHER_API_TOKEN", ""))

	def supervise(self) -> None:
		"""Blocks until a service exits, then fails."""
		while True:
			for name, process in self.background:
				if process.poll() is not None:
					raise LaunchError(f"{name} exited unexpectedly")
			time.sleep(2)


@dataclass
class TaskPipeline:
	"""The paper's task extraction study.

	Each task has its own prompt pools, ledger, and adapters under data/tasks/TASK and
	artifacts/tasks/TASK. Collection queries the largest budget of every seed once; smaller
	budgets are prefixes of it. Review never uses gold labels: it only drops empty responses
	and responses longer than the training limit.
	"""

	launcher: Launcher
	terms_review_id: str
	tasks: list[str]
	seeds: list[int]
	budgets: list[int] | None
	config: Path
	concurrency: int
	gold: bool
	artifact_root: Path = TASK_ARTIFACTS
	plan: dict[str, list[int]] = field(default_factory=dict)
	recipes: dict[str, dict[str, Any]] = field(default_factory=dict)
	run_ledgers: dict[tuple[str, int], Path] = field(default_factory=dict)

	def validate(self) -> None:
		if not self.terms_review_id:
			raise LaunchError(
				"set --terms-review-id or TEACHER_TERMS_REVIEW_ID "
				"(see docs/distillation-pipeline.md)"
			)
		unknown = [task for task in self.tasks if task not in TASK_NAMES]
		if unknown or not self.tasks:
			raise LaunchError(f"--tasks must name tasks from {', '.join(TASK_NAMES)}")
		if not self.seeds or self.budgets == []:
			raise LaunchError("--seeds and --budgets must each name at least one value")
		if self.budgets is not None and any(budget < 1 for budget in self.budgets):
			raise LaunchError("query budgets must be positive")
		if not self.config.is_file():
			raise LaunchError(f"training config not found: {self.config}")

	def task_dir(self, task: str) -> Path:
		return TASK_ROOT / task

	def manifest(self, task: str) -> dict[str, Any]:
		path = self.task_dir(task) / "manifest.json"
		return dict(json.loads(path.read_text(encoding="utf-8")))

	def ledger(self, task: str) -> Path:
		return self.task_dir(task) / "ledger.jsonl"

	def run_dir(self, task: str, seed: int, budget: int) -> Path:
		return self.task_dir(task) / "runs" / f"s{seed}-b{budget}"

	def build_tasks(self) -> None:
		"""Builds missing tasks and resolves full-pool or explicit query budgets."""
		for task in self.tasks:
			directory = self.task_dir(task)
			if (directory / "manifest.json").is_file():
				self.launcher.log(f"using the existing {task} build in {directory}")
			else:
				self.launcher.log(f"building {task} into {directory} (downloads the dataset)")
				self.launcher.distill(
					"tasks", "build", "--task", task, "--output-dir", str(directory)
				)
			pool = int(self.manifest(task)["roles"]["attacker"])
			requested = {pool} if self.budgets is None else set(self.budgets)
			budgets = sorted(budget for budget in requested if budget <= pool)
			if skipped := sorted(requested - set(budgets)):
				self.launcher.log(
					f"{task}: skipping budgets {', '.join(map(str, skipped))}, "
					f"larger than its attacker pool of {pool}",
					"yellow",
				)
			if not budgets:
				raise LaunchError(f"{task}: no budget fits its attacker pool of {pool}")
			self.plan[task] = budgets
			self.launcher.log(
				f"{task}: query budgets {', '.join(map(str, budgets))} (pool: {pool})"
			)
			# The largest budget's prompts are what collection queries.
			for seed in self.seeds:
				self.budget(task, seed, budgets[-1])

	def budget(self, task: str, seed: int, budget: int, *, with_ledger: bool = False) -> None:
		args = [
			"tasks",
			"budget",
			"--task-dir",
			str(self.task_dir(task)),
			"--budget",
			str(budget),
			"--seed",
			str(seed),
			"--output-dir",
			str(self.run_dir(task, seed, budget)),
		]
		if with_ledger:
			args += ["--ledger", str(self.run_ledgers.get((task, seed), self.ledger(task)))]
		self.launcher.distill(*args)

	def uncollected(self) -> int:
		"""Prompts of the planned budgets that the task ledgers do not hold yet."""
		missing = 0
		for task, budgets in self.plan.items():
			have = _example_ids(self.ledger(task))
			for seed in self.seeds:
				wanted = _example_ids(self.run_dir(task, seed, budgets[-1]) / "prompts.jsonl")
				missing += len(wanted - have)
		return missing

	def validate_ledgers(self) -> None:
		"""Check every resume ledger before loading a teacher or sending requests."""
		from distillation.collection import CollectionOptions, _existing_example_ids
		from distillation.io import JSONLValidationError, read_jsonl
		from distillation.schemas import PromptRecord

		for task, budgets in self.plan.items():
			path = self.ledger(task)
			prompts = {}
			for seed in self.seeds:
				for prompt in read_jsonl(
					self.run_dir(task, seed, budgets[-1]) / "prompts.jsonl", PromptRecord
				):
					prompts[prompt.example_id] = prompt
			try:
				_existing_example_ids(
					path,
					list(prompts.values()),
					CollectionOptions(
						terms_review_id=self.terms_review_id,
						max_output_tokens=int(self.manifest(task)["max_output_tokens"]),
						enable_thinking=False,
						concurrency=self.concurrency,
					),
				)
			except (JSONLValidationError, UnicodeError) as error:
				raise LaunchError(
					f"{error}; teacher not started. Inspect recovery with "
					f"uv run distill ledger repair --input {shlex.quote(str(path))}, "
					"then add --apply to back up and repair while collection is stopped."
				) from error
			except ValueError as error:
				raise LaunchError(f"cannot resume {path}: {error}") from error
			self.launcher.log(f"{task}: collection ledger validated")

	def collect(self, teacher_url: str, token: str) -> None:
		for task, budgets in self.plan.items():
			# The output limit is part of the task contract, so it comes from the build, never
			# from an environment override.
			limit = str(self.manifest(task)["max_output_tokens"])
			for seed in self.seeds:
				prompts = self.run_dir(task, seed, budgets[-1]) / "prompts.jsonl"
				self.launcher.log(f"collecting {task} seed {seed} into {self.ledger(task)}")
				self.launcher.distill(
					"collect",
					"--input",
					str(prompts),
					"--output",
					str(self.ledger(task)),
					"--teacher-url",
					teacher_url,
					"--terms-review-id",
					self.terms_review_id,
					"--concurrency",
					str(self.concurrency),
					"--max-output-tokens",
					limit,
					"--no-thinking",
					env={"TEACHER_API_TOKEN": token},
				)

	def train_all(self) -> None:
		with self.config.open("rb") as source:
			base = tomllib.load(source)
		for task, budgets in self.plan.items():
			for seed in self.seeds:
				for budget in budgets:
					self.train_run(base | self.recipes.get(task, {}), task, seed, budget)

	def train_run(self, base: dict[str, Any], task: str, seed: int, budget: int) -> None:
		"""Trains the attacker student, and with --gold the gold-label reference, for one run."""
		directory = self.run_dir(task, seed, budget)
		name = f"{task} s{seed}-b{budget}"
		self.budget(task, seed, budget, with_ledger=True)
		kinds = [("attacker", directory / "train.jsonl")]
		if self.gold:
			kinds.append(("gold", directory / "gold-train.jsonl"))
		for kind, dataset in kinds:
			output = self.artifact_root / task / f"s{seed}-b{budget}" / kind
			config = directory / f"train.{kind}.toml"
			values = {key: value for key, value in base.items() if key != "validation_dataset_path"}
			values |= {"dataset_path": str(dataset), "output_dir": str(output), "seed": seed}
			config.write_text(_toml(values), encoding="utf-8")
			if kind == "attacker":
				reviews = directory / "reviews.jsonl"
				self.launcher.log(f"{name}: reviewing responses without gold labels")
				self.launcher.distill(
					"review",
					"--ledger",
					str(directory / "ledger.jsonl"),
					"--output",
					str(reviews),
					"--config",
					str(config),
				)
				self.launcher.distill(
					"prepare",
					"--input",
					str(directory / "ledger.jsonl"),
					"--output",
					str(dataset),
					"--prompts",
					str(directory / "prompts.jsonl"),
					"--reviews",
					str(reviews),
				)
			if _trained(output, dataset, config):
				self.launcher.log(f"{name}: the {kind} adapter in {output} is already trained")
				continue
			self.launcher.log(f"{name}: training the {kind} student into {output}")
			self.launcher.distill("train", "--config", str(config))


def _example_ids(path: Path) -> set[str]:
	if not path.is_file():
		return set()
	with path.open(encoding="utf-8") as source:
		return {json.loads(line)["example_id"] for line in source if line.strip()}


def _trained(output: Path, dataset: Path, config: Path | None = None) -> bool:
	"""Whether output holds a finished adapter trained on dataset's current contents."""
	manifest = output / "training-manifest.json"
	if not manifest.is_file():
		return False
	recorded = json.loads(manifest.read_text(encoding="utf-8"))
	if not config or "fingerprint" not in recorded:
		raise LaunchError(
			f"{output} has no verified training fingerprint; use a new output directory"
		)
	from distillation.training import load_training_config, training_identity

	if recorded["fingerprint"] != training_identity(load_training_config(config)):
		raise LaunchError(
			f"{output} has a different model, tokenizer, template, data, or training configuration"
		)
	if not (output / "adapter_config.json").is_file() or not list(
		output.glob("adapter_model*.safetensors")
	):
		raise LaunchError(f"{output} is missing its trained adapter")
	if recorded.get("dataset_sha256") != hashlib.sha256(dataset.read_bytes()).hexdigest():
		raise LaunchError(f"{output} was trained on different data; choose a new output directory")
	if config:
		with config.open("rb") as source:
			values = tomllib.load(source)
		if any(recorded.get("config", {}).get(key) != value for key, value in values.items()):
			raise LaunchError(
				f"{output} has a different training recipe; choose a new output directory"
			)
	return True


def _toml(values: dict[str, Any]) -> str:
	"""Serializes a flat training config. JSON strings, numbers, and arrays are valid TOML."""

	def value(item: Any) -> str:
		if isinstance(item, dict):
			entries = (
				f"{key} = {value(entry)}" for key, entry in item.items() if entry is not None
			)
			return "{ " + ", ".join(entries) + " }"
		return json.dumps(item)

	return "".join(f"{key} = {value(item)}\n" for key, item in values.items() if item is not None)


def _numbers(text: str, option: str) -> list[int]:
	try:
		return [int(part) for part in text.split(",") if part.strip()]
	except ValueError:
		raise LaunchError(f"{option} must be comma-separated integers") from None


def _budgets(text: str) -> list[int] | None:
	if text.strip().lower() == "full":
		return None
	return list(dict.fromkeys(_numbers(text, "--budgets")))


def scenario(name: str, body: Callable[[Launcher], None]) -> None:
	"""Runs a scenario, always stopping its services, and exits 130 after Ctrl+C."""
	launcher = Launcher(name)

	def terminate(signum: int, frame: FrameType | None) -> None:
		raise KeyboardInterrupt

	previous = signal.signal(signal.SIGTERM, terminate)
	code = 0
	try:
		body(launcher)
	except KeyboardInterrupt:
		launcher.log("interrupted", "yellow")
		code = 130
	except LaunchError as error:
		launcher.log(f"error: {error}", "bold red")
		code = 1
	finally:
		launcher.stop_background()
		signal.signal(signal.SIGTERM, previous)
	if code:
		raise typer.Exit(code)


TermsReviewIdOption = Annotated[
	str,
	typer.Option(
		envvar="TEACHER_TERMS_REVIEW_ID",
		help="Approval reference written into every ledger record.",
	),
]
ConcurrencyOption = Annotated[
	int, typer.Option(envvar="DISTILL_CONCURRENCY", min=1, help="Parallel teacher requests.")
]
PortOption = Annotated[
	int, typer.Option(envvar="TEACHER_PORT", min=1, max=65535, help="Teacher API port.")
]
UpstreamUrlOption = Annotated[
	str,
	typer.Option(help="Use an existing OpenAI-compatible server instead of starting vLLM."),
]


@run_app.command("teacher-host")
def teacher_host_command(
	host: Annotated[str, typer.Option(help="Teacher API bind address.")] = "0.0.0.0",
	port: PortOption = 8000,
	upstream_url: UpstreamUrlOption = "",
) -> None:
	"""Serve Qwen3.5-9B with vLLM on loopback and the teacher API on the network.

	Runs until interrupted. Requires TEACHER_TERMS_REVIEW_ID, and TEACHER_API_TOKEN when
	binding beyond loopback.
	"""

	def body(launcher: Launcher) -> None:
		if not os.environ.get("TEACHER_TERMS_REVIEW_ID"):
			raise LaunchError(
				"set TEACHER_TERMS_REVIEW_ID in .env; the teacher API refuses requests without it"
			)
		if host not in {"127.0.0.1", "localhost"} and not os.environ.get("TEACHER_API_TOKEN"):
			raise LaunchError(
				f"set TEACHER_API_TOKEN before binding to {host} "
				"(generate one with: openssl rand -hex 32)"
			)
		launcher.start_teacher_stack(host, port, upstream_url, None)
		launcher.log(f"teacher API ready on {host}:{port}", "bold green")
		launcher.log("on the student machine run:")
		launcher.log(f"  just distill --teacher-url http://{_address()}:{port}")
		launcher.log("with the same TEACHER_API_TOKEN set on that machine")
		launcher.log("press Ctrl+C to stop")
		launcher.supervise()

	scenario("teacher-host", body)


@run_app.command("distill")
def distill_command(
	mode: Annotated[
		Literal["gateway", "tasks"],
		typer.Option(help="Shared student or legacy separate-task students."),
	] = "gateway",
	run_dir: Path = Path("artifacts/gateway-distill"),
	attacks: str = "A1",
	tasks: Annotated[
		str,
		typer.Option(
			help="Comma-separated tasks: ag_news, squad, gsm8k, or coding. Defaults to all four. "
			"GSM8K supplies the math distillation corpus; its output cap is provisional."
		),
	] = DEFAULT_TASKS,
	seeds: Annotated[
		str, typer.Option(help="Comma-separated seeds; each orders its own attacker queries.")
	] = "42,1,20",
	budgets: Annotated[
		str,
		typer.Option(
			help="'full' uses all attacker pools. Gateway numeric budgets are shared totals. "
			"Legacy tasks mode applies budgets separately to each task."
		),
	] = "full",
	gold: Annotated[
		bool, typer.Option(help="Also train the gold-label reference student for every run.")
	] = True,
	teacher_url: Annotated[
		str,
		typer.Option(help="Collect from a running teacher API instead of starting one here."),
	] = "",
	token: Annotated[
		str,
		typer.Option(envvar="TEACHER_API_TOKEN", help="Teacher API bearer token."),
	] = "",
	port: PortOption = 8000,
	upstream_url: UpstreamUrlOption = "",
	terms_review_id: TermsReviewIdOption = "",
	config: Annotated[
		Path, typer.Option(dir_okay=False, help="Training recipe copied into every run.")
	] = TASK_CONFIG,
	concurrency: ConcurrencyOption = 4,
) -> None:
	"""Collect across all four domains and train one shared student by default.

	Gateway runs keep one combined ledger and adapter per campaign, seed, and budget,
	plus a gold reference unless --no-gold. Use --mode tasks for per-task ledgers and
	adapters. Managed teacher services stop before student training.
	"""

	def body(launcher: Launcher) -> None:
		pipeline = TaskPipeline(
			launcher=launcher,
			terms_review_id=terms_review_id,
			tasks=[task.strip() for task in tasks.split(",") if task.strip()],
			seeds=_numbers(seeds, "--seeds"),
			budgets=_budgets(budgets),
			config=config,
			concurrency=concurrency,
			gold=gold,
		)
		if mode == "gateway":
			from typing import cast

			from distillation.gateway_study import GatewayStudy
			from distillation.schedules import Attack
			from distillation.study import Study

			GatewayStudy(
				Study(
					pipeline, run_dir, teacher_url, token, port, upstream_url, 8003, None, "refuse"
				),
				attacks=tuple(cast(Attack, a.strip()) for a in attacks.split(",")),
			).run(collect_only=True)
			return
		pipeline.validate()
		os.environ["TEACHER_TERMS_REVIEW_ID"] = pipeline.terms_review_id
		launcher.install("train")
		pipeline.build_tasks()
		pipeline.validate_ledgers()
		if missing := pipeline.uncollected():
			launcher.log(f"{missing} prompts still need teacher responses")
			if teacher_url:
				url = teacher_url.rstrip("/")
				launcher.wait_for_http(f"{url}/healthz", '"ready"', 60, "Waiting for the teacher")
				launcher.smoke_test(url, token)
				pipeline.collect(url, token)
			else:
				LOG_DIR.mkdir(parents=True, exist_ok=True)
				launcher.start_teacher_stack(
					"127.0.0.1", port, upstream_url, LOG_DIR / "teacher-api.log"
				)
				pipeline.collect(f"http://127.0.0.1:{port}", token)
				launcher.log("stopping the teacher to free GPU memory for training")
				launcher.stop_background()
		else:
			launcher.log("every planned prompt already has a teacher response")
		pipeline.train_all()
		launcher.log(f"done; adapters are in {TASK_ARTIFACTS}/TASK/sSEED-bBUDGET")

	scenario("distill", body)


@run_app.command("start")
def start_command(
	mode: Annotated[
		Literal["gateway", "tasks"],
		typer.Option(help="Shared gateway is the default; tasks selects the legacy study."),
	] = "gateway",
	attacks: str = "A1,A2,A5",
	interval_seconds: Annotated[float, typer.Option(min=0)] = 1,
	deadline_seconds: Annotated[float, typer.Option(min=0)] = 86400,
	benign_accounts: Annotated[int, typer.Option(min=1)] = 20,
	requests_per_account: Annotated[int, typer.Option(min=1)] = 100,
	development_budget: Annotated[int, typer.Option(min=2)] = 2000,
	tasks: Annotated[
		str,
		typer.Option(
			help="Comma-separated tasks: ag_news, squad, gsm8k, or coding. Defaults to all four. "
			"GSM8K supplies the math distillation corpus; its output cap is provisional."
		),
	] = DEFAULT_TASKS,
	seeds: Annotated[str, typer.Option(help="Comma-separated campaign seeds.")] = "42,1,20",
	budgets: Annotated[
		str,
		typer.Option(
			help="'full' uses all attacker pools. Gateway numeric budgets are shared totals. "
			"Legacy tasks mode applies budgets separately to each task."
		),
	] = "full",
	config: Annotated[Path, typer.Option(help="Student training recipe.")] = TASK_CONFIG,
	run_dir: Annotated[
		Path | None,
		typer.Option(
			help="Run directory; defaults to artifacts/gateway or artifacts/start for tasks."
		),
	] = None,
	calibration_root: Annotated[
		Path | None,
		typer.Option(help="Existing per-task calibrations; otherwise fit pooled benign accounts."),
	] = None,
	l4_action: Annotated[Literal["refuse", "deceive"], typer.Option()] = "refuse",
	arms: Annotated[
		str, typer.Option(help="Comma-separated defenses: full, instantaneous, and quota.")
	] = "full,instantaneous,quota",
	gold: Annotated[
		bool, typer.Option(help="Also train and benchmark gold reference adapters.")
	] = True,
	teacher_url: Annotated[str, typer.Option(help="Use an existing repository teacher API.")] = "",
	upstream_url: UpstreamUrlOption = "",
	token: Annotated[str, typer.Option(envvar="TEACHER_API_TOKEN")] = "",
	port: PortOption = 8000,
	harness_port: Annotated[int, typer.Option(min=1, max=65535)] = 8003,
	terms_review_id: TermsReviewIdOption = "",
	concurrency: ConcurrencyOption = 4,
	concentrated_share: Annotated[
		float,
		typer.Option(
			min=0,
			max=0.8,
			help="Share of benign traffic from specialist and repetitive clients; 0.4 is v1.",
		),
	] = 0.1,
	allow_undetected: Annotated[
		bool,
		typer.Option(
			help="Tasks mode: continue as a quota comparison when no MMD defense separates "
			"the extraction account from held-out benign accounts."
		),
	] = False,
) -> None:
	"""Build corpora, collect, train, benchmark, deploy the harness, and repeat extraction.

	The protected student starts from the same base model. Services run locally for the
	experiment and stop before student training. No GPU experiment runs during --help.
	"""
	from distillation.study import Study

	def body(launcher: Launcher) -> None:
		pipeline = TaskPipeline(
			launcher=launcher,
			terms_review_id=terms_review_id,
			tasks=list(dict.fromkeys(t.strip() for t in tasks.split(",") if t.strip())),
			seeds=list(dict.fromkeys(_numbers(seeds, "--seeds"))),
			budgets=_budgets(budgets),
			config=config,
			concurrency=concurrency,
			gold=gold,
		)
		root = run_dir or Path("artifacts/gateway" if mode == "gateway" else "artifacts/start")
		pipeline.artifact_root = root / "baseline-adapters"
		study = Study(
			pipeline,
			root,
			teacher_url,
			token,
			port,
			upstream_url,
			harness_port,
			calibration_root,
			l4_action,
			arms=tuple(dict.fromkeys(a.strip() for a in arms.split(",") if a.strip())),
			concentrated_share=concentrated_share,
			allow_undetected=allow_undetected,
		)
		if mode == "gateway":
			from typing import cast

			from distillation.gateway_study import GatewayStudy
			from distillation.schedules import Attack

			GatewayStudy(
				study,
				attacks=tuple(cast(Attack, a.strip()) for a in attacks.split(",")),
				interval_seconds=interval_seconds,
				deadline_seconds=deadline_seconds,
				benign_accounts=benign_accounts,
				requests_per_account=requests_per_account,
				development_budget=development_budget,
			).run()
		else:
			study.run()

	scenario("start", body)


def _find_vllm() -> str:
	"""VLLM_BIN, vllm on PATH, or the .venv-vllm install; empty when there is none yet."""
	if explicit := os.environ.get("VLLM_BIN") or shutil.which("vllm"):
		return explicit
	if sys.platform == "win32":
		raise LaunchError(
			"vLLM does not run natively on Windows. Start an OpenAI-compatible server "
			"(vLLM under WSL2, llama.cpp, or LM Studio) and pass --upstream-url"
		)
	venv = Path(".venv-vllm/bin/vllm")
	return str(venv) if venv.is_file() else ""


def _vllm_env() -> dict[str, str]:
	"""Disables the FlashInfer sampler when no CUDA toolkit is installed.

	FlashInfer JIT-compiles its top-k/top-p sampling kernel with nvcc on first use, so a host
	with only the NVIDIA driver fails engine startup. vLLM's PyTorch sampler needs no compiler.
	An explicit VLLM_USE_FLASHINFER_SAMPLER setting is left alone.
	"""
	if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ:
		return {}
	cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
	if shutil.which("nvcc") or (Path(cuda_home) / "bin" / "nvcc").is_file():
		return {}
	return {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


def _signal_group(process: subprocess.Popen[bytes], *, kill: bool) -> None:
	if sys.platform == "win32":
		if kill:
			process.kill()
		else:
			process.terminate()
		return
	with suppress(ProcessLookupError):
		os.killpg(process.pid, signal.SIGKILL if kill else signal.SIGTERM)


@contextmanager
def _status() -> Iterator[Callable[[Text | None], None]]:
	"""A spinner line on a terminal. Calling the updater with None removes it."""
	if not console.is_terminal:
		yield lambda text: None
		return
	with console.status("") as live:

		def update(text: Text | None) -> None:
			if text is None:
				live.stop()
			else:
				live.update(text)

		yield update


def _last_line(path: Path) -> str:
	"""The newest non-empty line of a log, including progress bars redrawn with \\r."""
	try:
		with path.open("rb") as source:
			source.seek(max(0, source.seek(0, os.SEEK_END) - 8192))
			tail = source.read().decode("utf-8", errors="replace")
	except OSError:
		return ""
	lines = [line.strip() for line in re.split(r"[\r\n]", ANSI.sub("", tail)) if line.strip()]
	return lines[-1] if lines else ""


def _duration(seconds: float) -> str:
	minutes, seconds = divmod(int(seconds), 60)
	return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


def _address() -> str:
	"""This machine's outbound IP address; connecting a UDP socket sends nothing."""
	try:
		with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
			probe.connect(("10.255.255.255", 1))
			return str(probe.getsockname()[0])
	except OSError:
		return "<this-host>"
