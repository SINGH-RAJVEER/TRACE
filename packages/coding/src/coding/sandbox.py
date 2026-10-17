"""Linux execution boundary for untrusted Python; fail closed on missing isolation."""

import errno
import math
import os
import platform
import shutil
import signal
import struct
import subprocess
import sys
import sysconfig
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


class SandboxUnavailable(RuntimeError):
	"""The host cannot provide the required isolation."""


@dataclass(frozen=True)
class Limits:
	wall_seconds: float = 3.0
	cpu_seconds: int = 2
	memory_bytes: int = 256 * 1024 * 1024
	file_bytes: int = 1024 * 1024

	def __post_init__(self) -> None:
		if not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0:
			raise ValueError("wall_seconds must be positive and finite")
		if min(self.cpu_seconds, self.memory_bytes, self.file_bytes) <= 0:
			raise ValueError("resource limits must be positive")


@dataclass(frozen=True)
class Execution:
	status: Literal["ok", "timeout", "error"]
	returncode: int | None
	stdout: str = ""
	stderr: str = ""


def _seccomp() -> bytes:
	"""Deny process creation, sockets, namespace changes and kernel escape helpers.

	The architecture check also rejects alternate syscall ABIs. This filter supplements
	user/mount/PID/network namespaces; it is not a general syscall allowlist.
	"""
	architecture = platform.machine()
	if architecture == "x86_64":
		audit_arch = 0xC000003E
		denied = [41, 53, 56, 57, 58, 101, 165, 166, 246, 272, 298, 308, 310, 311, 321]
	elif architecture in {"aarch64", "arm64"}:
		audit_arch = 0xC00000B7
		denied = [39, 40, 97, 117, 198, 199, 220, 241, 268, 270, 271, 280]
	else:
		raise SandboxUnavailable(f"unsupported seccomp architecture: {architecture}")
	denied += [425, 426, 427, 435]  # io_uring and clone3 on both architectures.
	# Classic BPF: load arch, reject mismatch, load syscall, reject x32, deny list.
	instructions = [
		(0x20, 0, 0, 4),
		(0x15, 1, 0, audit_arch),
		(0x06, 0, 0, 0x80000000),  # SECCOMP_RET_KILL_PROCESS
		(0x20, 0, 0, 0),
		(0x35, 0, 1, 0x40000000),
		(0x06, 0, 0, 0x80000000),
	]
	for number in denied:
		instructions.extend([(0x15, 0, 1, number), (0x06, 0, 0, 0x50000 | errno.EPERM)])
	instructions.append((0x06, 0, 0, 0x7FFF0000))  # SECCOMP_RET_ALLOW
	return b"".join(struct.pack("HBBI", *instruction) for instruction in instructions)


def _command(directory: Path, filter_fd: int, unsafe_local_exec: bool) -> list[str]:
	python = str(Path(sys.executable).resolve())
	if unsafe_local_exec:
		return [python, "-I", "-B", str(directory / "run.py")]
	bwrap = shutil.which("bwrap")
	if sys.platform != "linux" or not bwrap:
		raise SandboxUnavailable("bubblewrap is required; install it using devenv shell")
	command = [
		bwrap,
		"--unshare-all",
		"--unshare-user",
		"--die-with-parent",
		"--new-session",
		"--disable-userns",
		"--cap-drop",
		"ALL",
		"--clearenv",
		"--setenv",
		"PATH",
		"/nonexistent",
		"--setenv",
		"HOME",
		"/tmp",
	]
	# Nix interpreters need their store closure. FHS interpreters need stdlib and loaders.
	for runtime in ("/nix/store", "/usr", "/lib", "/lib64", "/bin"):
		if Path(runtime).exists():
			command += ["--ro-bind", runtime, runtime]
	command += ["--ro-bind", sysconfig.get_path("purelib"), "/runtime"]
	command += ["--setenv", "OPENBLAS_NUM_THREADS", "1", "--setenv", "OMP_NUM_THREADS", "1"]
	# Nix binary wheels need the compiler runtime. Only mounted immutable store paths
	# are forwarded; the host environment otherwise remains empty.
	libraries = [
		str(Path(path).resolve())
		for path in os.environ.get("LD_LIBRARY_PATH", "").split(":")
		if path and Path(path).is_dir() and str(Path(path).resolve()).startswith("/nix/store/")
	]
	if libraries:
		command += ["--setenv", "LD_LIBRARY_PATH", ":".join(libraries)]
	command += [
		"--proc",
		"/proc",
		"--dev",
		"/dev",
		"--size",
		"16777216",
		"--tmpfs",
		"/tmp",
		"--ro-bind",
		str(directory),
		"/work",
		"--chdir",
		"/tmp",
		"--remount-ro",
		"/dev",
		"--remount-ro",
		"/proc",
		"--remount-ro",
		"/",
		"--seccomp",
		str(filter_fd),
		python,
		"-I",
		"-B",
		"-c",
		"import sys, runpy; sys.path.append('/runtime'); "
		"runpy.run_path('/work/run.py', run_name='__main__')",
	]
	return command


def run_python(
	source: str, *, limits: Limits | None = None, unsafe_local_exec: bool = False
) -> Execution:
	"""Run one item. Never retry failures or fall back to unsandboxed execution.

	Output goes to size-limited files rather than unbounded PIPE buffers. The temporary
	filesystem has a 16 MiB mount limit, RLIMIT_FSIZE per file and RLIMIT_NOFILE caps.
	RLIMIT_AS bounds Python memory. Seccomp denies child processes and sockets.
	"""
	limits = limits or Limits()
	if sys.platform != "linux":
		raise SandboxUnavailable("the coding runner requires Linux resource limits")
	with tempfile.TemporaryDirectory(prefix="coding-") as temporary:
		directory = Path(temporary)
		(directory / "run.py").write_text(source)
		filter_path = directory / "seccomp.bpf"
		filter_path.write_bytes(_seccomp())
		with (
			filter_path.open("rb") as seccomp,
			(directory / "stdout").open("w+b") as stdout,
			(directory / "stderr").open("w+b") as stderr,
		):
			command = _command(directory, seccomp.fileno(), unsafe_local_exec)
			# prlimit applies before bwrap/Python starts, avoiding preexec_fn in threaded APIs.
			prlimit = shutil.which("prlimit")
			if not prlimit:
				raise SandboxUnavailable("util-linux prlimit is required")
			command = [
				prlimit,
				f"--as={limits.memory_bytes}",
				f"--cpu={limits.cpu_seconds}",
				f"--fsize={limits.file_bytes}",
				"--nofile=32",
				"--core=0",
				"--",
				*command,
			]
			try:
				process = subprocess.Popen(
					command,
					stdin=subprocess.DEVNULL,
					stdout=stdout,
					stderr=stderr,
					pass_fds=(seccomp.fileno(),),
					start_new_session=True,
					env={"PATH": os.defpath},
					cwd=directory,
				)
			except OSError as error:
				raise SandboxUnavailable(str(error)) from error
			status: Literal["ok", "timeout", "error"] = "ok"
			try:
				process.wait(timeout=limits.wall_seconds)
			except subprocess.TimeoutExpired:
				status = "timeout"
				with suppress(ProcessLookupError):
					os.killpg(process.pid, signal.SIGKILL)
				process.wait()
			stdout.seek(0)
			stderr.seek(0)
			output = stdout.read(limits.file_bytes).decode(errors="replace")
			diagnostic = stderr.read(limits.file_bytes).decode(errors="replace")
			if diagnostic.startswith("bwrap:") or diagnostic.startswith("prlimit:"):
				raise SandboxUnavailable(diagnostic.strip())
			if process.returncode == -signal.SIGXCPU:
				status = "timeout"
			return Execution(status, process.returncode, output, diagnostic)
