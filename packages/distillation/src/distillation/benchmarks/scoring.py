"""Deterministic answer scoring and isolated Python test execution."""

import ast
import asyncio
import re
import shutil
import subprocess
import uuid
from decimal import Decimal, InvalidOperation

from distillation.benchmarks.models import BenchmarkItem


def extract_answer(item: BenchmarkItem, response: str) -> str | None:
	if item.metric == "choice_accuracy":
		answer = response.strip()
		if re.fullmatch(r"[A-Z]", answer) and ord(answer) - 65 < len(item.choices):
			return answer
		return None
	if item.metric == "numeric_exact_match":
		# A bare numeric answer is valid too; never guess using the last number in a rationale.
		answer = response.rsplit("####", 1)[-1].strip()
		if not re.fullmatch(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", answer):
			return None
		try:
			return str(Decimal(answer.replace(",", "")).normalize())
		except InvalidOperation:
			return None
	return None


def score_answer(item: BenchmarkItem, response: str) -> tuple[str | None, bool]:
	prediction = extract_answer(item, response)
	if item.metric == "numeric_exact_match":
		expected = extract_answer(item, item.answer)
	else:
		expected = item.answer
	return prediction, prediction is not None and prediction == expected


def python_candidate(item: BenchmarkItem, response: str) -> str:
	candidate = response.strip("\r\n")
	fence = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", candidate.strip(), re.DOTALL)
	if fence:
		candidate = fence[1]
	if item.code_prompt and not re.search(
		rf"\bdef\s+{re.escape(item.entry_point)}\s*\(", candidate
	):
		candidate = item.code_prompt + "\n" + candidate
	ast.parse(candidate)
	return candidate


def resolve_code_image(image: str) -> str:
	if not shutil.which("docker"):
		raise RuntimeError(
			"coding benchmarks require Docker; install it and pull the configured code_image"
		)
	result = subprocess.run(
		["docker", "image", "inspect", "--format", "{{.Id}}", image],
		capture_output=True,
		text=True,
		check=False,
		timeout=30,
	)
	if result.returncode or not result.stdout.strip().startswith("sha256:"):
		raise RuntimeError(f"code image unavailable locally; run `docker pull {image}`")
	return result.stdout.strip()


def code_test_script(candidate: str, tests: str) -> str:
	"""A trusted wrapper counts a pass only after the assertions complete."""
	marker = uuid.uuid4().hex
	wrapper = (
		"import contextlib, os, sys\n"
		"scope = {'__name__': '__candidate__'}\n"
		"try:\n"
		"\twith open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):\n"
		f"\t\texec({candidate!r}, scope)\n"
		f"\t\texec({tests!r}, scope)\n"
		"except BaseException:\n\tsys.exit(1)\n"
		f"print({marker!r})\n"
	)
	return (
		"import subprocess, sys\n"
		f"result = subprocess.run([sys.executable, '-I', '-c', {wrapper!r}], "
		"capture_output=True)\n"
		f"passed = result.returncode == 0 and result.stdout.strip() == {marker.encode()!r}\n"
		"sys.exit(0 if passed else 1)\n"
	)


async def score_code(
	item: BenchmarkItem,
	response: str,
	image_id: str,
	wall_seconds: int,
) -> tuple[bool, str]:
	try:
		candidate = python_candidate(item, response)
	except SyntaxError, ValueError:
		return False, "invalid"
	name = "distill-eval-" + uuid.uuid4().hex
	script = code_test_script(candidate, item.test_code)
	process = await asyncio.create_subprocess_exec(
		"docker",
		"run",
		"--rm",
		"--name",
		name,
		"--network",
		"none",
		"--cpus",
		"1",
		"--memory",
		"256m",
		"--memory-swap",
		"256m",
		"--pids-limit",
		"64",
		"--read-only",
		"--tmpfs",
		"/tmp:rw,noexec,nosuid,size=16m",
		"--cap-drop",
		"ALL",
		"--security-opt",
		"no-new-privileges",
		"--user",
		"65534:65534",
		image_id,
		"python",
		"-I",
		"-",
		stdin=asyncio.subprocess.PIPE,
		stdout=asyncio.subprocess.DEVNULL,
		stderr=asyncio.subprocess.DEVNULL,
	)
	try:
		await asyncio.wait_for(process.communicate(script.encode()), timeout=wall_seconds)
		if process.returncode in {125, 126, 127}:
			raise RuntimeError("Docker could not execute the coding benchmark container")
		return process.returncode == 0, "ok" if process.returncode == 0 else "test_failed"
	except TimeoutError:
		return False, "timeout"
	finally:
		# Kill the container itself; killing the Docker client leaves submitted code running.
		cleanup = await asyncio.create_subprocess_exec(
			"docker",
			"rm",
			"--force",
			name,
			stdout=asyncio.subprocess.DEVNULL,
			stderr=asyncio.subprocess.DEVNULL,
		)
		await cleanup.wait()
		if process.returncode is None:
			process.kill()
			await process.wait()
