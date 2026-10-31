"""Collect only AG News and SQuAD using the existing Colab Q8 teacher."""

import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import httpx
from distillation.collection import CollectionOptions, collect
from distillation.io import read_jsonl, write_jsonl
from distillation.progress import LiveProgress
from distillation.schemas import LedgerRecord, PromptRecord
from distillation.tasks import TASKS, budget_run
from distillation.teacher import TeacherAPIGateway
from teacher_api.settings import Settings


root = Path("artifacts/paper-tasks-colab")
connection = Path("artifacts/colab-teacher/q8")
settings = Settings()
seeds, budgets = [42, 1, 20], [500, 2000, 8000]
locks = []
for path in (root / "collection.lock",):
	lock = path.open("a")
	fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
	locks.append(lock)


def save(name, value):
	path = root / name
	temporary = path.with_suffix(path.suffix + ".tmp")
	temporary.write_text(json.dumps(value, indent="\t") + "\n")
	temporary.replace(path)


plan = {}
for task in ("ag_news", "squad"):
	directory = Path("data/tasks") / task
	manifest = json.loads((directory / "manifest.json").read_text())
	for filename, digest in manifest["sha256"].items():
		assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == digest, filename
	wanted = {}
	for seed in seeds:
		run = directory / "runs" / f"s{seed}-b{max(budgets)}"
		budget_run(task_dir=directory, budget=max(budgets), seed=seed, output_dir=run)
		for prompt in read_jsonl(run / "prompts.jsonl", PromptRecord):
			if prompt.example_id in wanted:
				assert wanted[prompt.example_id] == prompt
			wanted[prompt.example_id] = prompt
	prompts = directory / "collection-prompts.jsonl"
	write_jsonl(prompts, wanted.values())
	plan[task] = {
		"prompts": str(prompts),
		"ledger": str(directory / "ledger.jsonl"),
		"unique_queries": len(wanted),
		"max_output_tokens": TASKS[task].max_output_tokens,
		"prompts_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
		"dataset": manifest["dataset"],
	}
	print(task, "planned unique queries", len(wanted), flush=True)
save(
	"plan.json",
	{
		"created_at": datetime.now(UTC).isoformat(),
		"tasks": plan,
		"seeds": seeds,
		"budgets": budgets,
		"teacher": "Qwen3.5-9B-Q8_0",
		"provider": "colab-llamacpp-q8_0",
		"teacher_url": "http://127.0.0.1:18010",
		"temperature": 0,
		"enable_thinking": False,
		"concurrency_per_task": 2,
		"deployment": json.loads((connection / "manifest.json").read_text()),
	},
)


class ConnectionGuard:
	def __init__(self):
		self.client = httpx.AsyncClient(timeout=5)
		self.lock = asyncio.Lock()
		self.last_ready = 0
		self.tunnel = None

	async def ready(self):
		if time.monotonic() - self.last_ready < 1:
			return
		async with self.lock:
			while True:
				try:
					response = await self.client.get("http://127.0.0.1:18011/health")
					if response.status_code == 200:
						self.last_ready = time.monotonic()
						save(
							"connection.json",
							{"status": "ready", "updated_at": datetime.now(UTC).isoformat()},
						)
						return
				except httpx.HTTPError:
					pass
				if self.tunnel is not None:
					self.tunnel.poll()
				try:
					os.kill(int((connection / "tunnel.pid").read_text()), 0)
				except ProcessLookupError:
					ssh = json.loads((connection / "ssh-command.json").read_text())
					args = [
						a.replace("ControlMaster=auto", "ControlMaster=yes").replace(
							"ControlPersist=600", "ControlPersist=no"
						)
						for a in ssh[:-1]
					]
					args += [
						"-N",
						"-o",
						"ExitOnForwardFailure=yes",
						"-L",
						"127.0.0.1:18011:127.0.0.1:8001",
						ssh[-1],
					]
					with (connection / "tunnel.log").open("ab") as log:
						self.tunnel = subprocess.Popen(
							args,
							stdin=subprocess.DEVNULL,
							stdout=log,
							stderr=subprocess.STDOUT,
							start_new_session=True,
						)
					(connection / "tunnel.pid").write_text(str(self.tunnel.pid) + "\n")
					print("Reconnecting Colab SSH tunnel", self.tunnel.pid, flush=True)
				save(
					"connection.json",
					{"status": "waiting_for_colab", "updated_at": datetime.now(UTC).isoformat()},
				)
				await asyncio.sleep(5)


class GuardedTeacher:
	def __init__(self, guard):
		self.guard = guard
		self.gateway = TeacherAPIGateway(
			base_url="http://127.0.0.1:18010", token=settings.api_token, timeout_seconds=600
		)

	async def generate(self, request):
		await self.guard.ready()
		return await self.gateway.generate(request)

	async def close(self):
		await self.gateway.close()


async def main():
	guard = ConnectionGuard()
	states = {}
	try:
		await guard.ready()
		models = await guard.client.get("http://127.0.0.1:18011/v1/models")
		models.raise_for_status()
		assert models.json()["data"][0]["id"] == "Qwen3.5-9B-Q8_0"
		with LiveProgress() as live:

			def report(task, state):
				states[task] = asdict(state)
				save(
					"progress.json",
					{
						"status": "collecting",
						"updated_at": datetime.now(UTC).isoformat(),
						"tasks": states,
						"done": sum(s["done"] for s in states.values()),
						"total": sum(p["unique_queries"] for p in plan.values()),
					},
				)
				live.update(
					task,
					state.done,
					state.total,
					f"new {state.collected:,} rejected {state.rejected:,} retries {state.retries:,}",
				)

			async def run(task):
				return await collect(
					input_path=Path(plan[task]["prompts"]),
					output_path=Path(plan[task]["ledger"]),
					teacher=GuardedTeacher(guard),
					options=CollectionOptions(
						terms_review_id=settings.terms_review_id,
						max_output_tokens=plan[task]["max_output_tokens"],
						temperature=0,
						enable_thinking=False,
						concurrency=2,
						max_attempts=3,
					),
					progress=lambda state: report(task, state),
				)

			results = await asyncio.gather(*(run(task) for task in plan), return_exceptions=True)
			for result in results:
				if isinstance(result, BaseException):
					raise result
		for task in plan:
			directory = Path("data/tasks") / task
			for seed in seeds:
				for budget in budgets:
					budget_run(
						task_dir=directory,
						budget=budget,
						seed=seed,
						output_dir=directory / "runs" / f"s{seed}-b{budget}",
						ledger_path=Path(plan[task]["ledger"]),
					)
		save(
			"exit.json",
			{"status": "complete", "finished_at": datetime.now(UTC).isoformat(), "tasks": states},
		)
	finally:
		await guard.client.aclose()


try:
	asyncio.run(main())
except BaseException as error:
	save(
		"exit.json",
		{
			"status": "stopped" if isinstance(error, KeyboardInterrupt) else "failed",
			"finished_at": datetime.now(UTC).isoformat(),
			"error_type": type(error).__name__,
		},
	)
	raise
