"""CLI for planning, executing and summarizing frozen development learning curves."""

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from distillation.learning_curve import SIZES, execute_curve, plan_curve, summarize_curve

app = typer.Typer(no_args_is_help=True)


@app.command("plan")
def plan(
	output: Annotated[Path, typer.Option()],
	development: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	a1: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	suite: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	n: Annotated[list[int] | None, typer.Option("--n")] = None,
	seed: Annotated[list[int] | None, typer.Option("--seed")] = None,
	subset_seed: int = 42,
	fraction: float = 0.8,
	gpu_hours_low: float = 4,
	gpu_hours_high: float = 7,
) -> None:
	"""Dry run: freeze nested subsets, provenance, hashes and projected GPU hours."""
	try:
		result = plan_curve(
			output,
			development,
			a1,
			suite,
			sizes=tuple(n) if n else SIZES,
			seeds=tuple(seed) if seed else (42, 43, 44),
			subset_seed=subset_seed,
			fraction=fraction,
			gpu_hours=(gpu_hours_low, gpu_hours_high),
		)
	except ValueError as error:
		raise typer.BadParameter(str(error)) from error
	typer.echo(json.dumps(result, indent="\t"))


@app.command("execute")
def execute(output: Annotated[Path, typer.Option(exists=True, file_okay=False)]) -> None:
	"""Train and evaluate missing cells on a GPU; skip validated completed scores."""
	typer.echo(json.dumps(execute_curve(output), indent="\t"))


@app.command("summary")
def summary(
	output: Annotated[Path, typer.Option(exists=True, file_okay=False)],
	replicates: int = 2000,
	seed: int = 42,
) -> None:
	"""Write paired seed/item bootstrap gains and the prespecified capability threshold."""
	typer.echo(json.dumps(summarize_curve(output, replicates=replicates, seed=seed), indent="\t"))


@app.command("index")
def index(
	output: Annotated[Path, typer.Option(exists=True, file_okay=False)],
	results: Annotated[Path, typer.Option("--index", dir_okay=False)] = Path("readings/index.tsv"),
	reading: Annotated[str, typer.Option(help="Reading that preserves this curve, if any.")] = "",
) -> None:
	"""Replace this curve's student rows in the results index, linked to its training runs."""
	from distillation.results_index import read_index, student_rows, update_index

	try:
		rows = student_rows(output, read_index(results))
		for row in rows:
			row["reading"] = reading
		update_index(results, rows)
	except ValueError as error:
		raise typer.BadParameter(str(error)) from error
	typer.echo(f"{results}: {len(rows)} student rows for {output}")


@app.command("transfer")
def transfer(
	results: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	min_headroom: float = 0.10,
	min_clean_gain: float = 0.05,
	min_effect: float = 0.05,
	min_suppression_fraction: float = 0.25,
	replicates: int = 2000,
	seed: int = 42,
) -> None:
	"""Report transfer metrics from per-item correctness (role -> family -> flags)."""
	from distillation.transfer_metrics import transfer_report, transfer_report_seeds

	options: dict[str, Any] = {
		"min_headroom": min_headroom,
		"min_clean_gain": min_clean_gain,
		"min_effect": min_effect,
		"min_suppression_fraction": min_suppression_fraction,
		"replicates": replicates,
		"seed": seed,
	}
	try:
		data = json.loads(results.read_text())
		report = (
			transfer_report_seeds(data["runs"], **options)
			if set(data) == {"runs"}
			else transfer_report(data, **options)
		)
	except (ValueError, KeyError, TypeError) as error:
		raise typer.BadParameter(str(error)) from error
	typer.echo(json.dumps(report, indent="\t"))
