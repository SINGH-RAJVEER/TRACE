"""Explicit development-grid commands. Execution requires an operator-supplied runner."""

import importlib
import json
from pathlib import Path
from typing import Annotated

import typer

from distillation.development_grid import derive_grid, execute_grid, plan_grid, write

app = typer.Typer(no_args_is_help=True)


@app.command("plan")
def plan(spec: Annotated[Path, typer.Option()], output: Annotated[Path, typer.Option()]) -> None:
	plan_grid(json.loads(spec.read_text()), output)


@app.command("execute")
def execute(
	output: Annotated[Path, typer.Option()], runner: Annotated[str, typer.Option()]
) -> None:
	"""Call a trusted module:function runner per setting; no implicit server launch."""
	module, name = runner.split(":", 1)
	execute_grid(output, getattr(importlib.import_module(module), name))


@app.command("match")
def match(
	output: Annotated[Path, typer.Option()],
	result: Annotated[Path, typer.Option()],
	development: Annotated[Path | None, typer.Option()] = None,
) -> None:
	derived = derive_grid(output)
	write(result, derived)
	if development is not None:
		write(development, derived["development"])
