"""CLI for the headroom and clean-student gates and the evaluations that feed them."""

import json
from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer(no_args_is_help=True)


@app.command("match-quotas")
def match_quotas_command(
	development: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option(help="New quota matching report.")],
) -> None:
	"""Select measured development settings that match every declared utility stratum."""
	from distillation.acquisition_manifest import write_new
	from distillation.quota_matching import match_quotas

	try:
		report = match_quotas(json.loads(development.read_text()))
		write_new(output, json.dumps(report, indent="\t") + "\n")
	except (ValueError, KeyError, OSError) as error:
		raise typer.BadParameter(str(error)) from error


def _finish(
	report: dict[str, Any], output: Path, results: Path, reading: str, student_seeds: int | None
) -> None:
	"""Publish the report, record it in the index, print it, and exit 1 unless it passed."""
	from distillation.baseline_gates import write_gate
	from distillation.results_index import gate_row, update_index

	sha256 = write_gate(report, output)
	update_index(results, [{**gate_row(report, sha256, student_seeds), "reading": reading}])
	typer.echo(json.dumps(report, indent="\t"))
	if not report["passed"]:
		raise typer.Exit(1)


@app.command("headroom")
def headroom_command(
	base_scores: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	teacher_scores: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	suite: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	margin_points: Annotated[
		float, typer.Option(help="Preregistered teacher-base margin in points; no default.")
	],
	output: Annotated[Path, typer.Option(help="New gate report file.")],
	results: Annotated[Path, typer.Option("--index", dir_okay=False)] = Path("readings/index.tsv"),
	reading: Annotated[str, typer.Option(help="Reading that preserves this gate, if any.")] = "",
	replicates: int = 2000,
	seed: int = 42,
) -> None:
	"""Phase 0: pass only if the teacher beats the base by the margin, excluding zero."""
	from distillation.baseline_gates import headroom_gate

	try:
		report = headroom_gate(
			base_scores,
			teacher_scores,
			suite,
			margin_points=margin_points,
			replicates=replicates,
			seed=seed,
		)
		_finish(report, output, results, reading, None)
	except (ValueError, KeyError, OSError) as error:
		raise typer.BadParameter(str(error)) from error


@app.command("clean")
def clean_command(
	curve: Annotated[Path, typer.Option(exists=True, file_okay=False)],
	headroom: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	manifest: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, help="Clean acquisition manifest.")
	],
	output: Annotated[Path, typer.Option(help="New gate report file.")],
	results: Annotated[Path, typer.Option("--index", dir_okay=False)] = Path("readings/index.tsv"),
	reading: Annotated[str, typer.Option(help="Reading that preserves this gate, if any.")] = "",
	replicates: int = 2000,
	seed: int = 42,
) -> None:
	"""Phase 1: pass only if the clean student beats the base at both sizes on every seed."""
	from distillation.baseline_gates import clean_gate

	try:
		report = clean_gate(curve, headroom, manifest, replicates=replicates, seed=seed)
		_finish(report, output, results, reading, len(report["declared_seeds"]))
	except (ValueError, KeyError, OSError) as error:
		raise typer.BadParameter(str(error)) from error


@app.command("evaluate-base")
def evaluate_base_command(
	suite: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option(help="New per-problem score file.")],
	seed: int = 42,
) -> None:
	"""Score the pinned untrained base on a held-out suite; needs a GPU and no teacher."""
	from distillation.baseline_gates import evaluate_base

	try:
		result = evaluate_base(suite, output, seed=seed)
	except (ValueError, OSError) as error:
		raise typer.BadParameter(str(error)) from error
	typer.echo(f"{output}: {sum(result['scores'].values())} of {len(result['scores'])} correct")


@app.command("evaluate-teacher")
def evaluate_teacher_command(
	suite: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option(help="New per-problem score file.")],
	teacher_url: Annotated[str, typer.Option(help="Direct undefended teacher API URL.")],
	teacher_model: str,
	teacher_revision: str,
	concurrency: Annotated[int, typer.Option(min=1)] = 4,
	timeout_seconds: Annotated[float, typer.Option(min=0.001)] = 300,
	token: Annotated[str, typer.Option(envvar="TEACHER_API_TOKEN")] = "",
) -> None:
	"""Score a direct teacher on the same suite; any failed request writes nothing."""
	from distillation.baseline_gates import evaluate_teacher

	try:
		result = evaluate_teacher(
			suite,
			output,
			url=teacher_url,
			model=teacher_model,
			revision=teacher_revision,
			token=token,
			concurrency=concurrency,
			timeout_seconds=timeout_seconds,
		)
	except (ValueError, OSError) as error:
		raise typer.BadParameter(str(error)) from error
	typer.echo(f"{output}: {sum(result['scores'].values())} of {len(result['scores'])} correct")


@app.command("collect")
def collect_command(
	pool: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, help="Attacker prompt pool JSONL.")
	],
	output: Annotated[Path, typer.Option(help="Ledger JSONL; a partial one is resumed.")],
	count: Annotated[int, typer.Option(min=1, help="Prompts to collect.")],
	teacher_url: Annotated[str, typer.Option(help="Direct undefended teacher API URL.")],
	seed: int = 42,
	concurrency: Annotated[int, typer.Option(min=1)] = 4,
	token: Annotated[str, typer.Option(envvar="TEACHER_API_TOKEN")] = "",
	terms_review_id: Annotated[str, typer.Option(envvar="TEACHER_TERMS_REVIEW_ID")] = "",
) -> None:
	"""Collect one task's fresh responses from a direct teacher, once, with frozen settings."""
	from distillation.baseline_gates import collect_clean

	try:
		collected, resumed = collect_clean(
			pool,
			output,
			count=count,
			seed=seed,
			url=teacher_url,
			terms_review_id=terms_review_id,
			token=token,
			concurrency=concurrency,
		)
	except (ValueError, OSError, RuntimeError) as error:
		raise typer.BadParameter(str(error)) from error
	typer.echo(f"{output}: {collected} collected, {resumed} already present")


@app.command("pilot")
def pilot_command(
	output: Annotated[Path, typer.Option(help="New pilot directory.")],
	suite: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	pool: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, help="Attacker prompt pool JSONL.")
	],
	gold: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	n: Annotated[int, typer.Option(min=1)] = 32,
	items: Annotated[int, typer.Option(min=1)] = 16,
	seed: int = 42,
) -> None:
	"""Time one small student cell and record peak GPU memory; not evidence of learning."""
	from distillation.baseline_gates import pilot_student

	try:
		result = pilot_student(output, suite, pool, gold, n=n, items=items, seed=seed)
	except (ValueError, OSError) as error:
		raise typer.BadParameter(str(error)) from error
	typer.echo(json.dumps(result, indent="\t"))
