"""Whole-pipeline scenarios: serve the teacher, collect, review, prepare, and train.

scripts/pipeline.py loads .env, installs the base Python environment, and then runs these
commands, which call it back to install the training stack and vLLM once the options are
valid. Each long step runs as a `distill` subprocess that draws its own progress, and services
started here run in their own process group so they can be stopped as a whole.
"""

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
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Annotated, Any

import httpx
import typer
from rich.text import Text

from distillation.progress import console

DEFAULT_TEACHER_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_TEACHER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DEFAULT_PROMPTS = Path("data/prompts/all-splits.jsonl")
DEFAULT_LEDGER = Path("data/teacher/ledger.jsonl")
DEFAULT_CONFIG = Path("configs/train.example.toml")
CORPUS_CONFIG = Path("configs/train.distill.toml")
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

	def stop_background(self) -> None:
		"""Stops services newest first: terminate, then kill after 30 seconds."""
		while self.background:
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
					*shlex.split(os.environ.get("VLLM_EXTRA_ARGS", "")),
				],
				vllm_log,
				{},
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
class Pipeline:
	"""Options shared by student-train and distill-local, and the steps that use them."""

	launcher: Launcher
	terms_review_id: str
	corpus: str
	references: Path | None
	reviews: Path | None
	accept_pending: bool
	prompts: Path
	ledger: Path
	config: Path
	concurrency: int
	max_output_tokens: int
	seed: int | None

	def validate(self) -> None:
		if not self.terms_review_id:
			raise LaunchError(
				"set --terms-review-id or TEACHER_TERMS_REVIEW_ID "
				"(see docs/distillation-pipeline.md)"
			)
		if not (
			self.corpus
			or self.references
			or self.reviews
			or self.accept_pending
			or self.prompts != DEFAULT_PROMPTS
		):
			# Without a data source, run the whole loop from the prompt corpus.
			self.corpus = os.environ.get("DISTILL_CORPUS") or "standard"
			self.launcher.log(f"no data source given; using the {self.corpus} prompt corpus")
		if self.corpus:
			if self.corpus not in {"quick", "standard", "full"}:
				raise LaunchError("--corpus must be quick, standard, or full")
			self.references = self.references or self.prompts.parent / "references.jsonl"
			if self.config == DEFAULT_CONFIG:
				self.config = CORPUS_CONFIG
		if not self.reviews and not self.references and not self.accept_pending:
			raise LaunchError(
				"pass --references PATH or --reviews PATH, or --accept-pending for a plumbing test"
			)
		if self.reviews and not self.reviews.is_file():
			raise LaunchError(f"reviews file not found: {self.reviews}")
		if not self.corpus:
			if self.references and not self.references.is_file():
				raise LaunchError(f"references file not found: {self.references}")
			if not self.prompts.is_file():
				raise LaunchError(
					f"prompts file not found: {self.prompts}; pass --corpus PROFILE or run "
					"just corpus"
				)
		if not self.config.is_file():
			raise LaunchError(f"training config not found: {self.config}")

	def build_corpus_if_needed(self) -> None:
		"""Builds the --corpus profile unless it is already in place.

		The build is deterministic, so a prompt keeps its example_id and text across rebuilds
		and profiles.
		"""
		if not self.corpus or not self.references:
			return
		directory = self.prompts.parent
		manifest = directory / "manifest.json"
		if manifest.is_file() and self.prompts.is_file() and self.references.is_file():
			current = json.loads(manifest.read_text(encoding="utf-8")).get("profile", "")
			if current == self.corpus:
				self.launcher.log(f"using the existing {self.corpus} prompt corpus in {directory}")
				return
		self.launcher.log(
			f"building the {self.corpus} prompt corpus into {directory} (downloads public datasets)"
		)
		self.launcher.distill(
			"corpus", "build", "--profile", self.corpus, "--output-dir", str(directory)
		)
		if not self.prompts.is_file() or not self.references.is_file():
			raise LaunchError(f"corpus build did not write {self.prompts} and {self.references}")

	def collect(self, teacher_url: str, token: str) -> None:
		args = [
			"collect",
			"--input",
			str(self.prompts),
			"--output",
			str(self.ledger),
			"--teacher-url",
			teacher_url,
			"--terms-review-id",
			self.terms_review_id,
			"--concurrency",
			str(self.concurrency),
			"--max-output-tokens",
			str(self.max_output_tokens),
			"--no-thinking",
		]
		if self.seed is not None:
			args += ["--seed", str(self.seed)]
		# The token travels in the environment so it never appears in the process list.
		self.launcher.log(f"collecting teacher responses into {self.ledger}")
		self.launcher.distill(*args, env={"TEACHER_API_TOKEN": token})

	def review(self) -> None:
		"""Checks the ledger against the answer key; the decisions become the reviews."""
		if not self.references:
			return
		output = self.ledger.with_name(f"{self.ledger.stem}.reviews.jsonl")
		args = [
			"review",
			"--ledger",
			str(self.ledger),
			"--references",
			str(self.references),
			"--output",
			str(output),
			"--config",
			str(self.config),
		]
		if self.reviews:
			args += ["--overrides", str(self.reviews)]
		if image := os.environ.get("REVIEW_CODE_IMAGE"):
			args += ["--code-image", image]
		self.launcher.log(f"reviewing teacher responses into {output}")
		self.launcher.distill(*args)
		self.reviews = output

	def prepare_and_train(self) -> None:
		self.review()
		with self.config.open("rb") as source:
			config = tomllib.load(source)
		datasets = [("train", str(config["dataset_path"]))]
		if validation := config.get("validation_dataset_path"):
			datasets.append(("validation", str(validation)))
		for split, path in datasets:
			args = [
				"prepare",
				"--input",
				str(self.ledger),
				"--output",
				path,
				"--split",
				split,
				"--prompts",
				str(self.prompts),
			]
			if self.reviews:
				args += ["--reviews", str(self.reviews)]
			if self.accept_pending:
				args.append("--accept-pending")
			self.launcher.log(f"preparing {split} data into {path}")
			self.launcher.distill(*args)
		self.launcher.log(f"training the student with {self.config}")
		self.launcher.distill("train", "--config", str(self.config))


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
CorpusOption = Annotated[
	str,
	typer.Option(
		help="quick, standard, or full. Builds or reuses the corpus in data/prompts, "
		"and uses its prompts, answer key, and "
		"configs/train.distill.toml. Without --references, --reviews, --accept-pending, "
		"or --prompts, defaults to DISTILL_CORPUS, or standard.",
	),
]
ReferencesOption = Annotated[
	Path | None,
	typer.Option(
		dir_okay=False,
		help="Answer key from distill corpus build; reviews the ledger before preparation.",
	),
]
ReviewsOption = Annotated[
	Path | None,
	typer.Option(
		dir_okay=False,
		help="Review decisions JSONL; with --references, they override the automatic ones.",
	),
]
AcceptPendingOption = Annotated[
	bool, typer.Option(help="Train on unreviewed records (plumbing tests only).")
]
PromptsOption = Annotated[Path, typer.Option(dir_okay=False, help="Prompt JSONL.")]
LedgerOption = Annotated[Path, typer.Option(dir_okay=False, help="Teacher ledger JSONL.")]
ConfigOption = Annotated[Path, typer.Option(dir_okay=False, help="Training TOML.")]
ConcurrencyOption = Annotated[
	int, typer.Option(envvar="DISTILL_CONCURRENCY", min=1, help="Parallel teacher requests.")
]
MaxOutputTokensOption = Annotated[
	int, typer.Option(envvar="DISTILL_MAX_OUTPUT_TOKENS", min=1, help="Teacher output limit.")
]
SeedOption = Annotated[
	int | None, typer.Option(envvar="DISTILL_SEED", help="Teacher sampling seed.")
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
		launcher.log(f"  just student-train --teacher-url http://{_address()}:{port}")
		launcher.log("with the same TEACHER_API_TOKEN set on that machine")
		launcher.log("press Ctrl+C to stop")
		launcher.supervise()

	scenario("teacher-host", body)


@run_app.command("student-train")
def student_train_command(
	teacher_url: Annotated[
		str,
		typer.Option(help="Teacher API server root, for example http://10.0.0.5:8000."),
	],
	token: Annotated[
		str,
		typer.Option(envvar="TEACHER_API_TOKEN", help="Teacher API bearer token."),
	] = "",
	terms_review_id: TermsReviewIdOption = "",
	corpus: CorpusOption = "",
	references: ReferencesOption = None,
	reviews: ReviewsOption = None,
	accept_pending: AcceptPendingOption = False,
	prompts: PromptsOption = DEFAULT_PROMPTS,
	ledger: LedgerOption = DEFAULT_LEDGER,
	config: ConfigOption = DEFAULT_CONFIG,
	concurrency: ConcurrencyOption = 4,
	max_output_tokens: MaxOutputTokensOption = 1024,
	seed: SeedOption = None,
) -> None:
	"""Collect from this repository's teacher API, then prepare the dataset and train."""

	def body(launcher: Launcher) -> None:
		url = teacher_url.rstrip("/")
		pipeline = Pipeline(
			launcher=launcher,
			terms_review_id=terms_review_id,
			corpus=corpus,
			references=references,
			reviews=reviews,
			accept_pending=accept_pending,
			prompts=prompts,
			ledger=ledger,
			config=config,
			concurrency=concurrency,
			max_output_tokens=max_output_tokens,
			seed=seed,
		)
		pipeline.validate()
		launcher.install("train")
		pipeline.build_corpus_if_needed()
		launcher.wait_for_http(f"{url}/healthz", '"ready"', 60, "Waiting for the teacher")
		launcher.smoke_test(url, token)
		pipeline.collect(url, token)
		pipeline.prepare_and_train()
		launcher.log("done; the adapter and training manifest are in the config's output_dir")

	scenario("student-train", body)


@run_app.command("distill-local")
def distill_local_command(
	port: PortOption = 8000,
	upstream_url: UpstreamUrlOption = "",
	terms_review_id: TermsReviewIdOption = "",
	corpus: CorpusOption = "",
	references: ReferencesOption = None,
	reviews: ReviewsOption = None,
	accept_pending: AcceptPendingOption = False,
	prompts: PromptsOption = DEFAULT_PROMPTS,
	ledger: LedgerOption = DEFAULT_LEDGER,
	config: ConfigOption = DEFAULT_CONFIG,
	concurrency: ConcurrencyOption = 4,
	max_output_tokens: MaxOutputTokensOption = 1024,
	seed: SeedOption = None,
) -> None:
	"""Run the whole distillation on this machine.

	Starts the teacher on 127.0.0.1, collects, stops the teacher to free GPU memory, then
	prepares the dataset and trains the student.
	"""

	def body(launcher: Launcher) -> None:
		pipeline = Pipeline(
			launcher=launcher,
			terms_review_id=terms_review_id,
			corpus=corpus,
			references=references,
			reviews=reviews,
			accept_pending=accept_pending,
			prompts=prompts,
			ledger=ledger,
			config=config,
			concurrency=concurrency,
			max_output_tokens=max_output_tokens,
			seed=seed,
		)
		pipeline.validate()
		# The teacher API refuses requests until it has an approval reference.
		os.environ["TEACHER_TERMS_REVIEW_ID"] = pipeline.terms_review_id
		launcher.install("train")
		pipeline.build_corpus_if_needed()
		LOG_DIR.mkdir(parents=True, exist_ok=True)
		launcher.start_teacher_stack("127.0.0.1", port, upstream_url, LOG_DIR / "teacher-api.log")
		pipeline.collect(f"http://127.0.0.1:{port}", os.environ.get("TEACHER_API_TOKEN", ""))
		launcher.log("stopping the teacher to free GPU memory for training")
		launcher.stop_background()
		pipeline.prepare_and_train()
		launcher.log("done; the adapter and training manifest are in the config's output_dir")

	scenario("distill-local", body)


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
