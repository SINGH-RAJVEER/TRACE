"""Commands for corpus preparation and harness campaign evaluation."""

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from coding import DEFAULT_POOL_SIZE, DEFAULT_QUALIFICATION_WORKERS
from rich.markup import escape

from distillation.gateway_cli import app as gateway_app
from distillation.progress import console

app = typer.Typer(no_args_is_help=True)
tasks_app = typer.Typer(no_args_is_help=True)
app.add_typer(tasks_app, name="tasks")
app.add_typer(gateway_app, name="gateway")
traffic_app = typer.Typer(no_args_is_help=True)
app.add_typer(traffic_app, name="traffic")


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

	Writes attacker.jsonl, gold-answers.jsonl, legitimate/*.jsonl, development-suite.json,
	test-suite.json, and manifest.json into --output-dir. Default directories are data/news,
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


if __name__ == "__main__":
	app()
