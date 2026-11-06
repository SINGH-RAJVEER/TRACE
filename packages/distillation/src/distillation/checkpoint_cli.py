"""CLI for chronological checkpoint export, student planning, execution and summary."""

import json
from pathlib import Path
from typing import Annotated

import typer

from distillation.checkpoint_learning import (
	SUPPORTED_TASKS,
	execute_checkpoint_curve,
	plan_checkpoint_curve,
	run_export,
	summarize_checkpoint_curve,
)

app = typer.Typer(no_args_is_help=True)


@app.command("export")
def export(
	campaign: Annotated[Path, typer.Option(exists=True, file_okay=False)],
	output: Annotated[Path, typer.Option(help="New directory; never overwritten.")],
	response_count: Annotated[
		list[int] | None,
		typer.Option("--response-count", help="Declared response-N checkpoint; repeatable."),
	] = None,
) -> None:
	"""Freeze chronological acquisition checkpoints from a completed campaign."""
	try:
		manifest = run_export(campaign, output, tuple(response_count or ()))
	except ValueError as error:
		raise typer.BadParameter(str(error)) from error
	typer.echo(json.dumps(manifest, indent="\t"))


@app.command("plan")
def plan(
	output: Annotated[Path, typer.Option(help="New directory; never overwritten.")],
	export: Annotated[Path, typer.Option(exists=True, file_okay=False)],
	suite: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	target_accuracy: Annotated[
		float,
		typer.Option(help="Preregistered accuracy in (0, 1]; frozen in the plan."),
	],
	task: Annotated[str, typer.Option(help=f"One of: {', '.join(SUPPORTED_TASKS)}.")] = "gsm8k",
	seed: Annotated[list[int] | None, typer.Option("--seed")] = None,
	include_pending: Annotated[
		bool, typer.Option(help="Include unreviewed pending responses, never rejected ones.")
	] = False,
) -> None:
	"""Freeze per-checkpoint training sets, the suite, seeds, target and input hashes."""
	try:
		result = plan_checkpoint_curve(
			output,
			export,
			suite,
			target_accuracy=target_accuracy,
			task=task,
			seeds=tuple(seed) if seed else (42, 43, 44),
			include_pending=include_pending,
		)
	except ValueError as error:
		raise typer.BadParameter(str(error)) from error
	typer.echo(json.dumps(result, indent="\t"))


@app.command("execute")
def execute(output: Annotated[Path, typer.Option(exists=True, file_okay=False)]) -> None:
	"""Train and evaluate missing cells on a GPU; skip verified completed scores."""
	try:
		result = execute_checkpoint_curve(output)
	except ValueError as error:
		raise typer.BadParameter(str(error)) from error
	typer.echo(json.dumps(result, indent="\t"))


@app.command("summary")
def summary(output: Annotated[Path, typer.Option(exists=True, file_okay=False)]) -> None:
	"""Report per-checkpoint accuracy, paired base gain and the first observed target hit."""
	try:
		result = summarize_checkpoint_curve(output)
	except ValueError as error:
		raise typer.BadParameter(str(error)) from error
	typer.echo(json.dumps(result, indent="\t"))
