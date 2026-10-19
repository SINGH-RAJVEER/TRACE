"""Single-sample executable correctness, with explicit failure denominators."""

import argparse
import ast
import json
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from coding.extract import extract_code
from coding.sandbox import Limits, run_python


@dataclass(frozen=True)
class CodingItem:
	"""Tests must invoke assertions; their source is never added to a model prompt."""

	tests: str

	def __post_init__(self) -> None:
		tree = ast.parse(self.tests)
		if not any(isinstance(node, ast.Assert) for node in tree.body):
			raise ValueError("coding items must contain top-level assertions")


@dataclass(frozen=True)
class Score:
	status: Literal["ok", "invalid", "truncated", "error", "timeout"]
	correct: bool
	diagnostic: str = ""


def pass_at_1(
	item: CodingItem,
	response: str,
	*,
	finish_reason: str | None = "stop",
	limits: Limits | None = None,
	unsafe_local_exec: bool = False,
) -> Score:
	"""Timeouts, invalid code and incomplete output score zero, without retries.

	Infrastructure errors raise SandboxUnavailable so they cannot silently become model
	failures. Unit tests share an interpreter with the candidate: this is a quality
	scorer, not a tamper-proof judge against deliberately adversarial submissions.
	"""
	if finish_reason in {"length", "max_tokens"}:
		return Score("truncated", False)
	if finish_reason != "stop":
		return Score("error", False, "generation did not complete normally")
	code = extract_code(response)
	if code is None:
		return Score("invalid", False)
	marker = secrets.token_hex(32)
	# Separate compilation prevents candidate future imports from changing test semantics.
	source = (
		f"exec(compile({code!r}, '<candidate>', 'exec'))\n"
		f"exec(compile({item.tests!r}, '<tests>', 'exec'))\n"
		f"print({marker!r})\n"
	)
	result = run_python(source, limits=limits, unsafe_local_exec=unsafe_local_exec)
	if result.status == "timeout":
		return Score("timeout", False)
	completed = marker in result.stdout.splitlines()
	return Score("ok", result.returncode == 0 and completed, result.stderr)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--tests", type=Path, required=True)
	parser.add_argument("--response", type=Path, required=True)
	parser.add_argument("--finish-reason", default="stop")
	parser.add_argument(
		"--unsafe-local-exec",
		action="store_true",
		help="Explicitly bypass namespace and seccomp isolation",
	)
	arguments = parser.parse_args()
	score = pass_at_1(
		CodingItem(arguments.tests.read_text()),
		arguments.response.read_text(),
		finish_reason=arguments.finish_reason,
		unsafe_local_exec=arguments.unsafe_local_exec,
	)
	print(json.dumps(asdict(score)))


if __name__ == "__main__":
	main()
