"""Commands for corpus preparation and harness campaign evaluation."""

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from coding import DEFAULT_POOL_SIZE, DEFAULT_QUALIFICATION_WORKERS
from rich.markup import escape

from distillation.gateway_cli import app as gateway_app
from distillation.learning_curve_cli import app as learning_curve_app
from distillation.output_study_cli import app as output_study_app
from distillation.progress import console

app = typer.Typer(no_args_is_help=True)
tasks_app = typer.Typer(no_args_is_help=True)
app.add_typer(tasks_app, name="tasks")
app.add_typer(gateway_app, name="gateway")
app.add_typer(learning_curve_app, name="learning-curve")
app.add_typer(output_study_app, name="output-study")
traffic_app = typer.Typer(no_args_is_help=True)
app.add_typer(traffic_app, name="traffic")
acquisition_app = typer.Typer(no_args_is_help=True)
app.add_typer(acquisition_app, name="acquisition")


@acquisition_app.command("manifest")
def acquisition_manifest_command(
	ledger: Annotated[
		list[Path],
		typer.Option("--ledger", exists=True, dir_okay=False, help="Repeat for each ledger."),
	],
	response_policy: Annotated[
		str,
		typer.Option(
			help="Identifier such as clean or protected-v1. Use clean only for responses "
			"delivered unmodified; the label is your assertion and is not verified."
		),
	],
	output: Annotated[Path, typer.Option(help="New manifest file; never overwritten.")],
	accepted_only: Annotated[
		bool,
		typer.Option(help="Drop rows not accepted, after validating all rows; counts are kept."),
	] = False,
) -> None:
	"""Write a versioned manifest of the cached teacher responses in one or more ledgers."""
	from distillation.acquisition_manifest import ManifestError, build_manifest, write_manifest

	try:
		manifest = build_manifest(
			ledger, response_policy=response_policy, accepted_only=accepted_only
		)
		write_manifest(manifest, output)
	except ManifestError as error:
		raise typer.BadParameter(str(error)) from error
	totals = manifest["totals"]
	typer.echo(
		f"examples={totals['examples']} unknown_token_rows={totals['unknown_token_rows']} "
		f"digest={manifest['digest']} output={output}"
	)


@acquisition_app.command("select")
def acquisition_select_command(
	manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	by: Annotated[str, typer.Option(help="queries or tokens")],
	budget: Annotated[int, typer.Option(min=1)],
	seed: Annotated[int, typer.Option()],
	split: Annotated[str, typer.Option(help="Only rows of this split are candidates.")] = "train",
	include_unaccepted: Annotated[
		bool, typer.Option(help="Also draw rows whose quality status is not accepted.")
	] = False,
	require_ledgers: Annotated[
		bool, typer.Option(help="Fail when a ledger file is missing instead of skipping its check.")
	] = False,
	output: Annotated[
		Path | None, typer.Option(help="Also write the selection to this new file.")
	] = None,
) -> None:
	"""Select example IDs for an equal-query or equal-token budget, by seed.

	Token budgets are only for the clean policy, whose ledger tokens were delivered as is.
	"""
	from distillation.acquisition_manifest import (
		ManifestError,
		load_manifest,
		select_rows,
		write_new,
	)

	if by not in ("queries", "tokens"):
		raise typer.BadParameter("--by must be queries or tokens")
	try:
		selection = select_rows(
			load_manifest(manifest, require_ledgers=require_ledgers),
			by="queries" if by == "queries" else "tokens",
			budget=budget,
			seed=seed,
			split=split,
			include_unaccepted=include_unaccepted,
		)
		text = json.dumps(selection, indent="\t")
		if output is not None:
			write_new(output, text + "\n")
	except ManifestError as error:
		raise typer.BadParameter(str(error)) from error
	typer.echo(text)


@tasks_app.command("build")
def tasks_build_command(
	task: Annotated[str, typer.Option(help="ag_news, squad, gsm8k, or coding")],
	output_dir: Annotated[Path | None, typer.Option("--output-dir", file_okay=False)] = None,
	seed: int = 42,
	development_limit: Annotated[int, typer.Option(min=0)] = 1000,
	qualification_workers: Annotated[
		int, typer.Option(min=1, help="Coding sandbox workers; use 1 for sequential qualification.")
	] = DEFAULT_QUALIFICATION_WORKERS,
	coding_pool_size: Annotated[
		int,
		typer.Option(
			min=0, help="Eligible coding examples across all roles; 0 qualifies the full source."
		),
	] = DEFAULT_POOL_SIZE,
) -> None:
	"""Split one task's pinned dataset into the paper's pools and freeze its suites.

	Writes the exploratory attacker.jsonl, the sealed attacker-confirmatory.jsonl,
	gold-answers.jsonl, legitimate/*.jsonl, development-suite.json, test-suite.json, and
	manifest.json into --output-dir. Default directories are data/news,
	data/question_answering, and data/math; coding uses its pool size and seed in its name.
	"""
	from distillation.tasks import build_task, task_directory

	directory = output_dir or task_directory(task, coding_pool_size=coding_pool_size, seed=seed)
	manifest = build_task(
		task=task,
		output_dir=directory,
		seed=seed,
		development_limit=development_limit,
		qualification_workers=qualification_workers,
		coding_pool_size=coding_pool_size,
		log=lambda message: console.print(f"[bold]tasks[/] {escape(message)}"),
	)
	summary = {
		key: manifest[key] for key in ("roles", "filtered", "truncated", "max_output_tokens")
	}
	typer.echo(json.dumps({**summary, "output_dir": str(directory)}, indent="\t"))


@traffic_app.command("freeze")
def traffic_freeze(
	output: Annotated[Path, typer.Option()],
	seed: int = 42,
	genuine_count: Annotated[int, typer.Option(min=4)] = 48,
	attack_count: Annotated[int, typer.Option(min=4)] = 144,
	duration_seconds: Annotated[float, typer.Option(min=0.01)] = 60,
	gateway_news: Annotated[
		bool, typer.Option(help="Format authored snippets for gateway news classification.")
	] = False,
) -> None:
	"""Freeze a reproducible small genuine and extraction traffic suite."""
	from distillation.benchmarks.traffic import freeze_traffic

	suite = freeze_traffic(
		output,
		seed=seed,
		genuine_count=genuine_count,
		attack_count=attack_count,
		duration_seconds=duration_seconds,
		gateway_news=gateway_news,
	)
	typer.echo(f"requests={len(suite.requests)} sha256={suite.sha256} output={output}")


@traffic_app.command("run")
def traffic_run(config: Annotated[Path, typer.Option(exists=True)]) -> None:
	"""Replay the configured protected endpoint, without an unprotected run."""
	from distillation.benchmarks.traffic import load_traffic_config
	from distillation.benchmarks.traffic_runner import run_traffic

	settings = load_traffic_config(config)
	if "after" not in settings.targets:
		raise typer.BadParameter("config must define targets.after")
	run = asyncio.run(run_traffic(settings, "after"))
	typer.echo(f"run={run.id} succeeded={sum(s.succeeded for s in run.samples)}/{len(run.samples)}")


@traffic_app.command("report")
def traffic_report(path: Annotated[Path, typer.Option(exists=True)]) -> None:
	"""Report one protected run, separated by ground-truth client type."""
	from distillation.benchmarks.traffic import TrafficRun
	from distillation.benchmarks.traffic_reporting import build_traffic_report

	run = TrafficRun.model_validate_json(path.read_text())
	result = build_traffic_report(path.parent, run.comparison_key())
	typer.echo(json.dumps(result, indent="\t"))


@gateway_app.command("evaluation-report")
def evaluation_report_command(
	events: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
	"""Print the canonical campaign summary, the one in campaign.json, with unknowns kept."""
	from distillation.campaign_summary import read_events, summarize_campaign

	typer.echo(json.dumps(summarize_campaign(read_events(events)), indent="\t"))


@gateway_app.command("comparison-report")
def comparison_report_command(
	view: Annotated[str, typer.Option(help="attacker_budget or legitimate_utility")],
	capacity_evidence: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	benign_schedule: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	index: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	attack: str | None = None,
	arm: Annotated[list[str] | None, typer.Option(help="Repeat for each compared arm.")] = None,
	output_token_budget: Annotated[float | None, typer.Option(min=0)] = None,
	money_budget: Annotated[float | None, typer.Option(min=0)] = None,
	utility_metric: Annotated[
		str | None,
		typer.Option(help="completion_rate, served_only_quality or unconditional_utility"),
	] = None,
	utility_target: float | None = None,
	utility_tolerance: Annotated[float | None, typer.Option(min=0)] = None,
	conditions: Annotated[
		Path | None,
		typer.Option(
			exists=True, dir_okay=False, help="Hand-written conditions instead of --index."
		),
	] = None,
	include_backfill: Annotated[
		bool, typer.Option(help="Also pool backfilled historical rows, which ran other code.")
	] = False,
) -> None:
	"""Compare arms only after the teacher capacity gate passes.

	With --index, conditions and outcomes come from readings/index.tsv for one attack and the
	given arms, with intervals over independent runs. Budgets and utility targets the index
	does not record must be declared with their options; they are never filled in.
	"""
	import hashlib

	from distillation.evaluation_protocol import compare_conditions
	from distillation.results_index import UTILITY_FIELDS, comparison_from_index, read_index

	if (index is None) == (conditions is None):
		raise typer.BadParameter("give exactly one of --index and --conditions")
	if utility_metric is not None and utility_metric not in UTILITY_FIELDS:
		raise typer.BadParameter(f"utility metric must be one of {', '.join(UTILITY_FIELDS)}")
	try:
		outcomes = None
		if index is not None:
			if not attack or not arm:
				raise ValueError("--index needs --attack and at least two --arm options")
			declared = {
				"output_token_budget": output_token_budget,
				"money_budget": money_budget,
				"utility_metric": utility_metric,
				"utility_target": utility_target,
				"utility_tolerance": utility_tolerance,
			}
			rows, outcomes = comparison_from_index(
				read_index(index),
				attack=attack,
				arms=arm,
				view=view,
				declared=declared,
				include_backfill=include_backfill,
			)
		else:
			assert conditions is not None
			rows = json.loads(conditions.read_text())
		result = compare_conditions(
			rows,
			view,
			capacity_evidence=json.loads(capacity_evidence.read_text()),
			schedule_sha256=hashlib.sha256(benign_schedule.read_bytes()).hexdigest(),
		)
	except ValueError as error:
		raise typer.BadParameter(str(error)) from error
	result["source"] = str(index) if index is not None else "hand-written conditions"
	result["outcomes"] = outcomes
	typer.echo(json.dumps(result, indent="\t"))
	if not result["comparable"]:
		raise typer.Exit(1)


@app.command("student-plan")
def student_plan_command(
	family: Annotated[list[str], typer.Option("--family", help="Repeat for each student family.")],
	seed: Annotated[list[int], typer.Option("--seed", help="Repeat for each training seed.")],
	cost_per_run: Annotated[
		float, typer.Option(min=0, help="Estimated cost in a consistent currency per run.")
	],
	output: Annotated[Path, typer.Option()],
) -> None:
	"""Write a learning-curve plan and cost estimate without running training."""
	from distillation.evaluation_protocol import student_plan

	try:
		plan = student_plan(family, seed, cost_per_run)
	except ValueError as error:
		raise typer.BadParameter(str(error)) from error
	output.write_text(json.dumps(plan, indent="\t") + "\n")
	typer.echo(f"runs={len(plan['runs'])} estimated_cost={plan['estimated_cost']} output={output}")


if __name__ == "__main__":
	app()
