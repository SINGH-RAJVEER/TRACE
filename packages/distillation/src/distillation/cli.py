"""Command-line interface for collection, preparation, and training."""

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer

from distillation.collection import CollectionOptions, collect
from distillation.preparation import prepare_sft_dataset
from distillation.teacher import TeacherAPIGateway
from distillation.training import load_training_config, preflight_training, train

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
benchmark_app = typer.Typer(no_args_is_help=True)
app.add_typer(benchmark_app, name="benchmark", help="Freeze, score, and compare public benchmarks.")


@app.command("collect")
def collect_command(
	input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
	output_path: Annotated[Path, typer.Option("--output", dir_okay=False)],
	teacher_url: Annotated[str, typer.Option("--teacher-url")],
	terms_review_id: Annotated[str, typer.Option("--terms-review-id")],
	token: Annotated[str, typer.Option("--token", envvar="TEACHER_API_TOKEN")] = "",
	temperature: Annotated[float, typer.Option(min=0.0, max=2.0)] = 0.0,
	top_p: Annotated[float | None, typer.Option(min=0.0, max=1.0)] = None,
	top_k: Annotated[int | None, typer.Option(min=1)] = None,
	max_output_tokens: Annotated[int, typer.Option(min=1, max=32768)] = 1024,
	seed: int | None = None,
	concurrency: Annotated[int, typer.Option(min=1)] = 4,
	max_attempts: Annotated[int, typer.Option(min=1)] = 3,
	include_test: Annotated[bool, typer.Option()] = False,
	enable_thinking: Annotated[bool | None, typer.Option("--thinking/--no-thinking")] = None,
) -> None:
	"""Call the teacher API and append immutable provenance records."""

	gateway = TeacherAPIGateway(base_url=teacher_url, token=token)
	collected, skipped = asyncio.run(
		collect(
			input_path=input_path,
			output_path=output_path,
			teacher=gateway,
			options=CollectionOptions(
				terms_review_id=terms_review_id,
				temperature=temperature,
				top_p=top_p,
				top_k=top_k,
				max_output_tokens=max_output_tokens,
				seed=seed,
				concurrency=concurrency,
				max_attempts=max_attempts,
				include_test=include_test,
				enable_thinking=enable_thinking,
			),
		)
	)
	typer.echo(f"collected={collected} skipped={skipped} output={output_path}")


@app.command("prepare")
def prepare_command(
	input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
	output_path: Annotated[Path, typer.Option("--output", dir_okay=False)],
	split: Annotated[str, typer.Option()] = "train",
	reviews_path: Annotated[
		Path | None,
		typer.Option("--reviews", exists=True, dir_okay=False),
	] = None,
	accept_pending: Annotated[bool, typer.Option()] = False,
	prompts_path: Annotated[
		Path | None, typer.Option("--prompts", exists=True, dir_okay=False)
	] = None,
) -> None:
	"""Write accepted, deduplicated TRL prompt-completion records."""

	written, skipped = prepare_sft_dataset(
		input_path=input_path,
		output_path=output_path,
		split=split,
		reviews_path=reviews_path,
		prompts_path=prompts_path,
		accept_pending=accept_pending,
	)
	typer.echo(f"written={written} skipped={skipped} output={output_path}")


@app.command("train")
def train_command(
	config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
	preflight_only: Annotated[bool, typer.Option("--preflight-only")] = False,
) -> None:
	"""Run completion-only LoRA or QLoRA supervised fine-tuning."""

	if os.environ.get("TOKENIZERS_PARALLELISM") is None:
		os.environ["TOKENIZERS_PARALLELISM"] = "false"
	config = load_training_config(config_path)
	if preflight_only:
		typer.echo(json.dumps(preflight_training(config), indent="\t", sort_keys=True))
		return
	output_dir = train(config)
	typer.echo(f"adapter={output_dir}")


@benchmark_app.command("freeze")
def freeze_command(
	output: Annotated[Path, typer.Option("--output")],
	benchmarks: str = "gsm8k,arc_challenge,mmlu,truthfulqa,humaneval,mbpp",
	limit: Annotated[int, typer.Option(min=0)] = 200,
	seed: int = 42,
) -> None:
	"""Freeze publisher test questions. Limit 0 selects every question."""
	from distillation.benchmarks.suites import freeze_suite

	suite = freeze_suite(output, [name.strip() for name in benchmarks.split(",")], limit, seed)
	typer.echo(f"questions={len(suite.items)} sha256={suite.sha256} output={output}")


@benchmark_app.command("run")
def benchmark_run_command(
	config: Annotated[Path, typer.Option("--config", exists=True)],
	stage: Annotated[str, typer.Option("--stage")] = "before",
) -> None:
	"""Evaluate one phase on the identical frozen suite.

	before: the initial teacher and base student. after: the teacher and distilled student.
	"""
	from distillation.benchmarks.models import STAGE_SLOTS, load_config
	from distillation.benchmarks.runner import run_evaluation

	if stage not in STAGE_SLOTS:
		raise typer.BadParameter("stage must be before, after, or all")
	settings = load_config(config)
	slots = STAGE_SLOTS[stage]
	if any(slot not in settings.models for slot in slots):
		raise typer.BadParameter(f"config must define models for {', '.join(slots)}")
	for slot in slots:
		typer.echo(f"Evaluating {slot}")
		run = asyncio.run(
			run_evaluation(
				settings, slot, progress=lambda done, total: typer.echo(f"{done}/{total}")
			)
		)
		typer.echo(f"run={run.id} correct={sum(s.correct for s in run.samples)}/{len(run.samples)}")


@benchmark_app.command("report")
def report_command(
	results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("outputs/benchmarks"),
	comparison: Annotated[str, typer.Option("--comparison")] = "",
	output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
	"""List comparable experiments or export a full comparison as JSON."""
	import json

	from distillation.benchmarks.models import save_json
	from distillation.benchmarks.reporting import build_report, experiments

	value = build_report(results_dir, comparison) if comparison else experiments(results_dir)
	if output:
		save_json(output, value)
	else:
		typer.echo(json.dumps(value, indent="\t"))


@benchmark_app.command("demo")
def demo_command(
	output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("outputs/benchmark-demo"),
) -> None:
	"""Create labeled simulated artifacts for a dashboard preview."""
	from distillation.benchmarks.demo import create_demo

	create_demo(output_dir)
	typer.echo(f"Simulated dashboard fixtures written to {output_dir}")


@benchmark_app.command("dashboard")
def dashboard_command(
	results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("outputs/benchmarks"),
	frontend_dir: Annotated[Path, typer.Option("--frontend-dir")] = Path("apps/dashboard/dist"),
	port: Annotated[int, typer.Option(min=1, max=65535)] = 8002,
) -> None:
	"""Serve the read-only results API and the built dashboard on localhost."""
	import uvicorn

	from distillation.benchmarks.dashboard import create_app

	uvicorn.run(create_app(results_dir, frontend_dir), host="127.0.0.1", port=port)


if __name__ == "__main__":
	app()
