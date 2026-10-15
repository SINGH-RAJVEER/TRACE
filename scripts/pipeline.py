#!/usr/bin/env python3
"""Run the distillation pipeline with one command per machine, on Linux, macOS, or Windows.

Usage: python3 scripts/pipeline.py SCENARIO [options]   (py -3 on Windows)

  teacher-host   Serve vLLM and the teacher API until interrupted
  student-train  Collect from this repository's teacher API, then prepare and train
  distill-local  Serve the teacher, collect, stop it, then prepare and train

`python3 scripts/pipeline.py SCENARIO --help` lists the options. See docs/run-scripts.md.

This file uses only the standard library, so any Python 3.9 or newer can start it. It loads
.env, reports missing tools, installs the base Python 3.14 workspace environment, and hands
the options to `distill run SCENARIO` inside that environment. Once the options are valid,
`distill run` calls back with --prepare train or --prepare vllm to install the training stack
and vLLM, so a mistyped option fails before a multi-gigabyte download. Runtime selection, in
order:
  1. devenv  NixOS with devenv installed. The script re-runs itself inside `devenv shell`,
             which provides Python, uv, a compiler, and library paths.
  2. uv      uv on PATH.
  3. pip     Neither of the above. A Python 3.14 interpreter and pip build .venv-pip.
Set DISTILL_RUNTIME=devenv|uv|pip to force one.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ("teacher-host", "student-train", "distill-local")
WINDOWS = sys.platform == "win32"


def log(message: str) -> None:
	print(f"[pipeline] {message}", file=sys.stderr, flush=True)


def die(message: str) -> NoReturn:
	log(f"error: {message}")
	sys.exit(1)


def load_dotenv() -> None:
	"""Exports .env values that are not already set in the environment."""
	path = REPO_ROOT / ".env"
	if not path.is_file():
		return
	for line in path.read_text(encoding="utf-8").splitlines():
		if re.match(r"^\s*(#|$)", line) or "=" not in line:
			continue
		key, value = line.split("=", 1)
		key = re.sub(r"^\s*export\s+", "", key).strip()
		quoted = re.match(r"^\"(.*)\"$|^'(.*)'$", value)
		if quoted:
			value = quoted.group(1) if quoted.group(1) is not None else quoted.group(2)
		if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key) and key not in os.environ:
			os.environ[key] = value


def is_nixos() -> bool:
	if Path("/etc/NIXOS").exists():
		return True
	try:
		return "ID=nixos" in Path("/etc/os-release").read_text().splitlines()
	except OSError:
		return False


def enter_devenv_if_nixos(argv: list[str]) -> None:
	"""Re-runs this script inside devenv on NixOS, where the uv-managed Python cannot run."""
	requested = os.environ.get("DISTILL_RUNTIME") or "auto"
	if os.environ.get("DEVENV_ROOT"):
		if requested == "auto":
			os.environ["DISTILL_RUNTIME"] = "devenv"
		return
	if requested not in ("auto", "devenv") or (requested == "auto" and not is_nixos()):
		return
	devenv = shutil.which("devenv")
	if not devenv:
		if requested == "devenv":
			die("DISTILL_RUNTIME=devenv but devenv is not installed")
		log("NixOS detected but devenv is not installed; falling back to uv or pip")
		return
	log("NixOS detected; re-running inside devenv shell")
	os.chdir(REPO_ROOT)
	os.environ["DISTILL_RUNTIME"] = "devenv"
	command = [devenv, "shell", "--", "python3", str(Path(__file__).resolve()), *argv]
	os.execv(devenv, command)


def find_python(minimum: tuple[int, int], candidates: list[list[str]]) -> str | None:
	"""The path of the first interpreter that is at least minimum. A candidate may carry
	launcher arguments, such as ["py", "-3.14"]."""
	check = f"import sys; print(sys.executable); sys.exit(sys.version_info[:2] < {minimum!r})"
	for candidate in candidates:
		if not shutil.which(candidate[0]):
			continue
		result = subprocess.run(
			[*candidate, "-c", check], capture_output=True, text=True, check=False
		)
		if result.returncode == 0 and result.stdout.strip():
			return result.stdout.strip().splitlines()[-1]
	return None


def find_workspace_python() -> str | None:
	candidates = [["python3.14"], ["python3"], ["python"]]
	if WINDOWS:
		candidates.insert(1, ["py", "-3.14"])
	return find_python((3, 14), candidates)


def find_vllm_python() -> str | None:
	python = os.environ.get("VLLM_PYTHON") or "3.12"
	if re.match(r"^[0-9.]+$", python):
		python = f"python{python}"
	return find_python(
		(3, 10), [[python], ["python3.12"], ["python3.13"], ["python3.11"], ["python3"]]
	)


def venv_executable(venv: Path, name: str) -> Path:
	return venv / "Scripts" / f"{name}.exe" if WINDOWS else venv / "bin" / name


def vllm_installed() -> bool:
	return bool(
		os.environ.get("VLLM_BIN")
		or shutil.which("vllm")
		or venv_executable(REPO_ROOT / ".venv-vllm", "vllm").exists()
	)


def resolve_runtime() -> str:
	runtime = os.environ.get("DISTILL_RUNTIME") or "auto"
	if runtime == "auto":
		runtime = "uv" if shutil.which("uv") else "pip"
	return runtime


def select_runtime(needs_gpu: bool, needs_vllm: bool) -> str:
	"""Picks the runtime and reports every missing tool before anything is installed."""
	missing: list[str] = []
	warnings: list[str] = []
	runtime = resolve_runtime()
	if runtime in ("devenv", "uv"):
		if not shutil.which("uv"):
			missing.append("uv (https://docs.astral.sh/uv/)")
	elif runtime == "pip":
		warnings.append(
			"using pip, which does not read uv.lock; install uv for locked dependencies"
		)
		python = find_workspace_python()
		if not python:
			missing.append("Python 3.14 (python3.14 on PATH, or py -3.14 on Windows)")
		elif subprocess.run(
			[python, "-c", "import ensurepip, venv"], capture_output=True, check=False
		).returncode:
			missing.append(
				f"venv and ensurepip for {python} (python3.14-venv on Debian and Ubuntu)"
			)
		if needs_vllm and not WINDOWS and not vllm_installed() and not find_vllm_python():
			missing.append("Python 3.10 to 3.13 for vLLM (set VLLM_PYTHON)")
	else:
		die(f"unknown DISTILL_RUNTIME '{runtime}'; use devenv, uv, or pip")
	if needs_vllm and WINDOWS and not os.environ.get("VLLM_BIN"):
		missing.append(
			"vLLM, which does not run natively on Windows; start an OpenAI-compatible server "
			"and pass --upstream-url"
		)
	if needs_gpu:
		if not shutil.which("nvidia-smi"):
			warnings.append(
				"nvidia-smi not found; vLLM and QLoRA training need an NVIDIA GPU and driver"
			)
		if not WINDOWS:
			compiler = shutil.which("cc") or shutil.which("gcc")
			if not compiler:
				warnings.append(
					"no C compiler (cc or gcc); Triton kernels in vLLM and training compile "
					"with one"
				)
			elif subprocess.run(
				[compiler, "-x", "c", "-fsyntax-only", "-"],
				input=b"#include <stdlib.h>\n",
				capture_output=True,
				check=False,
			).returncode:
				# A compiler without libc headers passes the check above but fails inside Triton.
				warnings.append(
					f"{compiler} cannot find the C library headers; install build-essential "
					"(Debian and Ubuntu)"
				)

	log(f"runtime: {runtime}")
	for item in warnings:
		log(f"warning: {item}")
	if missing:
		for item in missing:
			log(f"missing: {item}")
		die("install the missing tools above and run again")
	return runtime


def run(command: list[str]) -> None:
	result = subprocess.run(command, check=False)
	if result.returncode:
		die(f"{command[0]} exited with code {result.returncode}")


def prepare_app_env(runtime: str, train: bool) -> Path:
	"""Installs the workspace packages and returns the environment's Python."""
	if runtime == "pip":
		venv = REPO_ROOT / ".venv-pip"
		python = venv_executable(venv, "python")
		if not python.exists():
			interpreter = find_workspace_python()
			if not interpreter:
				die("Python 3.14 not found")
			run([str(interpreter), "-m", "venv", str(venv)])
		log("installing workspace packages with pip")
		run([str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
		distillation = "./packages/distillation[train]" if train else "./packages/distillation"
		run([str(python), "-m", "pip", "install", "-e", distillation, "-e", "./apps/teacher-api"])
	else:
		venv = REPO_ROOT / ".venv"
		log("installing workspace packages with uv")
		run(["uv", "sync", "--all-packages", "--inexact", *(["--extra", "train"] if train else [])])
	return venv_executable(venv, "python")


def prepare_vllm(runtime: str) -> str:
	"""Locates vLLM or installs it into .venv-vllm, separate from the Python 3.14 workspace."""
	existing = os.environ.get("VLLM_BIN") or shutil.which("vllm")
	if existing:
		return existing
	venv = REPO_ROOT / ".venv-vllm"
	vllm = venv_executable(venv, "vllm")
	if vllm.exists():
		return str(vllm)
	spec = os.environ.get("VLLM_SPEC") or "vllm"
	log(f"installing {spec} into .venv-vllm (several GB)")
	python = venv_executable(venv, "python")
	if runtime == "pip":
		interpreter = find_vllm_python()
		if not interpreter:
			die("vLLM needs a Python interpreter between 3.10 and 3.13; set VLLM_PYTHON")
		run([str(interpreter), "-m", "venv", str(venv)])
		run([str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
		run([str(python), "-m", "pip", "install", spec])
	else:
		run(["uv", "venv", "--python", os.environ.get("VLLM_PYTHON") or "3.12", str(venv)])
		run(["uv", "pip", "install", "--python", str(python), spec])
	if not vllm.exists():
		die(f"vLLM executable not found at {vllm}")
	return str(vllm)


def prepare(components: list[str]) -> int:
	"""Installs what a running scenario needs next. distill run calls this after it has
	validated the options."""
	runtime = resolve_runtime()
	for component in components:
		if component == "train":
			prepare_app_env(runtime, train=True)
		elif component == "vllm":
			prepare_vllm(runtime)
		else:
			die(f"unknown component '{component}'; use train or vllm")
	return 0


def hand_off(command: list[str]) -> int:
	"""Replaces this process with the launcher. Windows cannot exec in place, so there the
	launcher runs as a child, which receives Ctrl+C itself while this process waits."""
	if not WINDOWS:
		os.execv(command[0], command)
	process = subprocess.Popen(command)
	while True:
		try:
			return process.wait()
		except KeyboardInterrupt:
			continue


def main(argv: list[str]) -> int:
	if not argv or argv[0] in ("-h", "--help"):
		print(__doc__.split("\n\nThis file")[0].strip())
		return 0
	os.chdir(REPO_ROOT)
	if argv[0] == "--prepare":
		return prepare(argv[1:])
	scenario, options = argv[0], argv[1:]
	if scenario not in SCENARIOS:
		die(f"unknown scenario '{scenario}'; use {', '.join(SCENARIOS)}")
	load_dotenv()
	enter_devenv_if_nixos(argv)
	help_only = "-h" in options or "--help" in options
	upstream = any(o == "--upstream-url" or o.startswith("--upstream-url=") for o in options)
	needs_vllm = not help_only and scenario != "student-train" and not upstream
	needs_gpu = not help_only and (scenario != "teacher-host" or not upstream)
	os.environ["DISTILL_RUNTIME"] = select_runtime(needs_gpu, needs_vllm)
	python = prepare_app_env(os.environ["DISTILL_RUNTIME"], train=False)
	# python -m rather than the distill executable, which Windows would lock against the
	# reinstall that --prepare train may do while the launcher runs.
	return hand_off([str(python), "-m", "distillation.cli", "run", scenario, *options])


if __name__ == "__main__":
	if sys.version_info < (3, 9):  # noqa: UP036 - the future import parses on 3.8
		sys.exit("scripts/pipeline.py needs Python 3.9 or newer")
	sys.exit(main(sys.argv[1:]))
