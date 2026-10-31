"""Force-kill the real server mid-request and check what survives the restart.

The server runs as its own process against a stand-in teacher, so these tests exercise
process death rather than a closed store. They measure enforcement only: no real model,
encoder, or detector runs here.
"""

import hashlib
import json
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from harness.calibration import (
	BenignRequest,
	Calibration,
	HarnessConfig,
	calibrate,
	save_calibration,
)
from harness.mmd import Array

KEY = "crash-test-key"
AUTHORIZED = {"Authorization": f"Bearer {KEY}"}
OUTPUT_TOKENS = 7


class Encoder:
	def encode(self, texts: list[str]) -> Array:
		rows = [
			np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).normal(
				size=4
			)
			for t in texts
		]
		return np.array([row / np.linalg.norm(row) for row in rows])

	def identity(self) -> dict[str, Any]:
		return {"model": "fake"}


class StubTeacher(ThreadingHTTPServer):
	"""Answers every request, and holds its answers while `open` is cleared."""

	daemon_threads = True

	def __init__(self) -> None:
		super().__init__(("127.0.0.1", 0), Handler)
		self.calls = 0
		self.count = threading.Lock()
		self.open = threading.Event()
		self.open.set()

	@property
	def url(self) -> str:
		return f"http://127.0.0.1:{self.server_address[1]}"

	def handle_error(self, request: Any, client_address: Any) -> None:
		"""A killed server drops held connections, which is the point of these tests."""


class Handler(BaseHTTPRequestHandler):
	server: StubTeacher

	def do_POST(self) -> None:  # noqa: N802
		self.rfile.read(int(self.headers["Content-Length"]))
		with self.server.count:
			self.server.calls += 1
		self.server.open.wait(timeout=60)
		body = json.dumps(
			{
				"id": "stub",
				"provider": "stub",
				"model": "stub",
				"generated_at": datetime.now(UTC).isoformat(),
				"message": {"role": "assistant", "content": "answer"},
				"finish_reason": "stop",
				"usage": {"input_tokens": 1, "output_tokens": OUTPUT_TOKENS},
				"endpoint": "stub",
			}
		).encode()
		self.send_response(200)
		self.send_header("Content-Type", "application/json")
		self.send_header("Content-Length", str(len(body)))
		self.end_headers()
		self.wfile.write(body)

	def log_message(self, format: str, *args: Any) -> None:
		pass


@pytest.fixture
def teacher() -> Iterator[StubTeacher]:
	stub = StubTeacher()
	thread = threading.Thread(target=stub.serve_forever, daemon=True)
	thread.start()
	yield stub
	stub.open.set()
	stub.shutdown()
	stub.server_close()


@pytest.fixture(scope="module")
def calibration_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
	requests = [BenignRequest(account=f"a{i // 20}", text=f"q {i}") for i in range(400)]
	calibration: Calibration
	calibration, references = calibrate(
		config=HarnessConfig(
			window_size=2, reference_windows=2, bandwidth_sample=20, aggregate_enabled=False
		),
		encoder=Encoder(),
		fitting=requests[:100],
		score_calibration=requests[100:200],
		policy_calibration=requests[200:],
	)
	calibration.baselines["quota"].thresholds = [3.0]
	directory = tmp_path_factory.mktemp("calibration")
	save_calibration(directory, calibration, references)
	return directory


class Server:
	"""The harness API in its own process, restartable on the same files."""

	def __init__(self, work: Path, calibration: Path, teacher: StubTeacher, **env: str) -> None:
		self.work = work
		self.log = work / "decisions.jsonl"
		accounts = work / "accounts.json"
		accounts.write_text(json.dumps({hashlib.sha256(KEY.encode()).hexdigest(): "attacker"}))
		with socket.socket() as probe:
			probe.bind(("127.0.0.1", 0))
			self.port = probe.getsockname()[1]
		self.base = f"http://127.0.0.1:{self.port}"
		self.env = {
			**os.environ,
			"HARNESS_CALIBRATION_DIR": str(calibration),
			"HARNESS_TEACHER_URL": teacher.url,
			"HARNESS_ACCOUNTS_FILE": str(accounts),
			"HARNESS_DECISION_LOG": str(self.log),
			"HARNESS_L4_ACTION": "refuse",
			"HARNESS_TASK": "text",
			"HARNESS_PORT": str(self.port),
			"HARNESS_LOG_LEVEL": "warning",
			**env,
		}
		self.process: subprocess.Popen[bytes] | None = None

	def start(self) -> None:
		self.process = subprocess.Popen(
			[sys.executable, "-m", "harness_api"], env=self.env, cwd=self.work
		)
		deadline = time.monotonic() + 60
		while time.monotonic() < deadline:
			try:
				httpx.get(f"{self.base}/healthz", timeout=1)
				return
			except httpx.TransportError:
				time.sleep(0.05)
		raise TimeoutError("the harness did not start")

	def crash(self) -> None:
		"""SIGKILL or TerminateProcess: no shutdown code runs."""
		assert self.process is not None
		self.process.kill()
		self.process.wait()
		self.process = None

	def ask(self, index: int) -> int | str:
		try:
			return httpx.post(
				f"{self.base}/v1/teacher/generate",
				json={"messages": [{"role": "user", "content": f"q {index}"}]},
				headers=AUTHORIZED,
				timeout=60,
			).status_code
		except httpx.TransportError as error:
			return type(error).__name__

	def events(self, kind: str) -> list[dict[str, Any]]:
		lines = self.log.read_text(encoding="utf-8").splitlines()
		return [e for e in map(json.loads, lines) if e["event"] == kind]


@pytest.fixture
def server(tmp_path: Path, calibration_dir: Path, teacher: StubTeacher) -> Iterator[Any]:
	created: list[Server] = []

	def build(**env: str) -> Server:
		created.append(Server(tmp_path, calibration_dir, teacher, **env))
		return created[-1]

	yield build
	for each in created:
		if each.process is not None:
			each.crash()


def test_a_crash_with_requests_in_flight_cannot_refill_a_budget(
	server: Any, teacher: StubTeacher, tmp_path: Path
) -> None:
	state = tmp_path / "state.sqlite3"
	harness: Server = server(
		HARNESS_POLICY="allow",
		HARNESS_STATE_DB=str(state),
		HARNESS_BUDGET_REQUESTS="12",
		HARNESS_BUDGET_CONCURRENCY="2",
	)
	harness.start()
	assert [harness.ask(i) for i in range(6)] == [200] * 6

	teacher.open.clear()
	with ThreadPoolExecutor(max_workers=2) as pool:
		flying = [pool.submit(harness.ask, 100 + i) for i in range(2)]
		deadline = time.monotonic() + 30
		while teacher.calls < 8 and time.monotonic() < deadline:
			time.sleep(0.01)
		assert teacher.calls == 8
		assert harness.ask(102) == 429
		harness.crash()
		teacher.open.set()
		assert all(isinstance(request.result(), str) for request in flying)

	with sqlite3.connect(state) as database:
		rows = database.execute(
			"SELECT COUNT(*), SUM(max_output_tokens) FROM reservations WHERE status = 'reserved'"
		).fetchone()
	assert rows == (2, 2 * 1024)

	harness.start()
	# 6 released and 2 left open by the crash leave 4 of the 12.
	codes = [harness.ask(200 + i) for i in range(10)]
	assert codes == [200] * 4 + [403] * 6
	assert [e["orphaned"] for e in harness.events("startup")] == [0, 2]
	releases = harness.events("release")
	assert [e["status"] for e in releases] == ["released"] * 10
	assert sum(e["output_tokens"] for e in releases) == 10 * OUTPUT_TOKENS
	assert {e["policy_revision"] for e in harness.events("startup")} == {
		e["policy_revision"] for e in releases
	}


def test_a_token_budget_survives_a_crash_at_the_full_reservation(
	server: Any, teacher: StubTeacher, tmp_path: Path
) -> None:
	harness: Server = server(
		HARNESS_POLICY="allow",
		HARNESS_STATE_DB=str(tmp_path / "state.sqlite3"),
		HARNESS_BUDGET_OUTPUT_TOKENS=str(1024 + 2 * OUTPUT_TOKENS + 1024),
	)
	harness.start()
	assert [harness.ask(i) for i in range(2)] == [200, 200]
	teacher.open.clear()
	with ThreadPoolExecutor(max_workers=1) as pool:
		flying = pool.submit(harness.ask, 100)
		deadline = time.monotonic() + 30
		while teacher.calls < 3 and time.monotonic() < deadline:
			time.sleep(0.01)
		harness.crash()
		teacher.open.set()
		assert isinstance(flying.result(), str)

	harness.start()
	# The open request stays charged its whole 1,024, leaving exactly room for one more.
	assert [harness.ask(200 + i) for i in range(2)] == [200, 403]


def test_without_a_state_file_a_crash_refills_a_quota(server: Any) -> None:
	"""The control: the failure the state file and budget exist to close."""
	harness: Server = server(HARNESS_POLICY="quota")
	harness.start()
	assert [harness.ask(i) for i in range(4)] == [200, 200, 200, 403]
	harness.crash()
	harness.start()
	assert [harness.ask(10 + i) for i in range(4)] == [200, 200, 200, 403]
