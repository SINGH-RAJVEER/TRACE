"""CPU orchestration for cached output experiments; adapters own any teacher calls."""

import importlib
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from distillation.acquisition_manifest import write_new
from distillation.output_sets import (
	Adapters,
	Recipe,
	build_arm,
	select_training_set,
	write_training_set,
)
from distillation.quota_comparison import comparison_report

app = typer.Typer(no_args_is_help=True)


def _json(path: Path) -> Any:
	return json.loads(path.read_text())


@app.command("build")
def build_command(
	manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	recipe: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	adapter: Annotated[str, typer.Option(help="Trusted Python module:factory returning Adapters.")],
	output: Annotated[Path, typer.Option()],
	protected: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	admissions: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	include_pending: Annotated[
		bool, typer.Option(help="Include unreviewed pending responses, never rejected ones.")
	] = False,
) -> None:
	"""Build a hashed arm from cached clean drafts, using explicit validation and tokenization."""
	try:
		module, name = adapter.split(":", 1)
		adapters = getattr(importlib.import_module(module), name)()
		if not isinstance(adapters, Adapters):
			raise ValueError("adapter factory must return output_sets.Adapters")
		result = build_arm(
			manifest,
			Recipe(**_json(recipe)),
			adapters,
			protected=_json(protected) if protected else None,
			admissions=_json(admissions) if admissions else None,
			include_pending=include_pending,
		)
		write_training_set(result, output)
	except (ValueError, KeyError, TypeError, OSError, ImportError, AttributeError) as error:
		raise typer.BadParameter(str(error)) from error


@app.command("select")
def select_command(
	arm: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	by: Annotated[str, typer.Option(help="queries or tokens")],
	budget: Annotated[int, typer.Option(min=1)],
	seed: Annotated[int, typer.Option()],
	output: Annotated[Path, typer.Option()],
) -> None:
	"""Select equal-query or equal-released-token SFT rows, retaining withheld-query costs."""
	if by not in {"queries", "tokens"}:
		raise typer.BadParameter("by must be queries or tokens")
	try:
		write_training_set(
			select_training_set(
				_json(arm), by="queries" if by == "queries" else "tokens", budget=budget, seed=seed
			),
			output,
		)
	except (ValueError, KeyError, OSError) as error:
		raise typer.BadParameter(str(error)) from error


@app.command("comparison-report")
def report_command(
	declaration: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option()],
	gate_report: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
	"""Join both budget views to transfer reports; missing evidence returns no result."""
	try:
		result = comparison_report(_json(declaration), gate_report=gate_report)
		write_new(output, json.dumps(result, indent="\t") + "\n")
		typer.echo(result["verdict"])
	except (ValueError, KeyError, TypeError, OSError) as error:
		raise typer.BadParameter(str(error)) from error
