"""Command-line interface for collection, preparation, and training."""

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from distillation.collection import CollectionOptions, CollectionProgress, collect
from distillation.launcher import run_app
from distillation.preparation import prepare_sft_dataset
from distillation.progress import LiveProgress, console
from distillation.teacher import TeacherAPIGateway
from distillation.training import (
	TrainingInterrupted,
	load_training_config,
	preflight_training,
	train,
)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
benchmark_app = typer.Typer(no_args_is_help=True)
app.add_typer(benchmark_app, name="benchmark", help="Freeze, score, and compare public benchmarks.")
traffic_app = typer.Typer(no_args_is_help=True)
benchmark_app.add_typer(traffic_app, name="traffic", help="Measure harness utility and blocking.")
corpus_app = typer.Typer(no_args_is_help=True)
app.add_typer(corpus_app, name="corpus", help="Build the transfer prompt corpus.")
app.add_typer(run_app, name="run", help="Run a whole pipeline scenario on this machine.")


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
	with LiveProgress() as live:

		def show(state: CollectionProgress) -> None:
			live.update(
				"Collecting teacher answers",
				state.done,
				state.total,
				f"new {state.collected:,}  rejected {state.rejected:,}  retries {state.retries:,}",
			)

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
				progress=show,
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


@app.command("review")
def review_command(
	ledger_path: Annotated[Path, typer.Option("--ledger", exists=True, dir_okay=False)],
	output_path: Annotated[Path, typer.Option("--output", dir_okay=False)],
	references_path: Annotated[
		Path | None, typer.Option("--references", exists=True, dir_okay=False)
	] = None,
	overrides_path: Annotated[
		Path | None, typer.Option("--overrides", exists=True, dir_okay=False)
	] = None,
	config_path: Annotated[
		Path | None, typer.Option("--config", exists=True, dir_okay=False)
	] = None,
	code_image: Annotated[str | None, typer.Option("--code-image")] = None,
	code_timeout_seconds: Annotated[int, typer.Option(min=1)] = 10,
) -> None:
	"""Validate teacher responses against the answer key and write review decisions.

	--config rejects records longer than the training max_length. --code-image runs Python
	tests in an isolated Docker container; without it, code is only parsed. Decisions in
	--overrides, such as human reviews, replace the automatic ones.
	"""
	from distillation.review import review_ledger

	with LiveProgress() as live:
		summary = review_ledger(
			ledger_path=ledger_path,
			references_path=references_path,
			output_path=output_path,
			overrides_path=overrides_path,
			config_path=config_path,
			code_image=code_image,
			code_timeout_seconds=code_timeout_seconds,
			progress=live.update,
		)
	typer.echo(json.dumps(summary, indent="\t", sort_keys=True))


@corpus_app.command("build")
def corpus_build_command(
	output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)] = Path(
		"data/prompts"
	),
	authored_path: Annotated[Path, typer.Option("--authored", dir_okay=False)] = Path(
		"data/authored/prompts.jsonl"
	),
	profile: Annotated[str, typer.Option()] = "standard",
	seed: int = 42,
	validation_fraction: Annotated[float, typer.Option(min=0.0, max=0.5)] = 0.04,
	max_prompt_chars: Annotated[int, typer.Option(min=100)] = 3000,
	families: Annotated[str, typer.Option(help="Comma-separated subset of families")] = "",
) -> None:
	"""Download pinned public prompt sets, add authored prompts, and write the corpus.

	Profiles: quick (about 7,200 prompts), standard (about 36,100), full (about 75,900).
	"""
	from distillation.corpus import build_corpus

	manifest = build_corpus(
		output_dir=output_dir,
		authored_path=authored_path,
		profile=profile,
		seed=seed,
		validation_fraction=validation_fraction,
		max_prompt_chars=max_prompt_chars,
		families=[name.strip() for name in families.split(",") if name.strip()] or None,
		log=lambda message: console.print(f"[bold]corpus[/] {escape(message)}"),
	)
	typer.echo(
		json.dumps(
			{
				"total": manifest["total"],
				"splits": manifest["splits"],
				"families": manifest["families"],
				"output_dir": str(output_dir),
			},
			indent="\t",
		)
	)


@app.command("download-model")
def download_model_command(
	model: Annotated[str, typer.Argument(help="Hugging Face repository, such as Qwen/Qwen3.5-9B")],
	revision: Annotated[str, typer.Option()] = "main",
) -> None:
	"""Download model weights into the Hugging Face cache, showing progress.

	vLLM and transformers read the same cache, so they start without downloading again.
	"""
	from huggingface_hub import snapshot_download

	path = snapshot_download(
		model,
		revision=revision,
		# Safetensors weights plus configuration and tokenizer files, never duplicate formats.
		allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.tiktoken"],
	)
	if console.is_terminal:
		# The last download bar leaves the cursor at the end of its line.
		console.line()
	typer.echo(f"model={model} revision={revision} path={path}")


@app.command("train")
def train_command(
	config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
	preflight_only: Annotated[bool, typer.Option("--preflight-only")] = False,
) -> None:
	"""Run completion-only LoRA or QLoRA supervised fine-tuning."""

	if os.environ.get("TOKENIZERS_PARALLELISM") is None:
		os.environ["TOKENIZERS_PARALLELISM"] = "false"
	# Growable allocator segments avoid out-of-memory failures caused by fragmentation.
	# Set before torch initializes CUDA; an explicit setting in either variable wins.
	if not os.environ.get("PYTORCH_ALLOC_CONF") and not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
		os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
	config = load_training_config(config_path)
	if preflight_only:
		typer.echo(json.dumps(preflight_training(config), indent="\t", sort_keys=True))
		return
	try:
		output_dir = train(config)
	except TrainingInterrupted as stopped:
		typer.echo(str(stopped), err=True)
		raise typer.Exit(130) from None
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
	traffic_results_dir: Annotated[Path, typer.Option("--traffic-results-dir")] = Path(
		"outputs/traffic-benchmarks"
	),
) -> None:
	"""Serve the read-only results API and the built dashboard on localhost."""
	import uvicorn

	from distillation.benchmarks.dashboard import create_app

	uvicorn.run(
		create_app(results_dir, frontend_dir, traffic_results_dir=traffic_results_dir),
		host="127.0.0.1",
		port=port,
	)


@traffic_app.command("freeze")
def traffic_freeze_command(
	output: Annotated[Path, typer.Option("--output")],
	seed: int = 42,
	genuine_count: Annotated[int, typer.Option(min=4)] = 48,
	attack_count: Annotated[int, typer.Option(min=4)] = 144,
	duration_seconds: Annotated[float, typer.Option(min=0.01)] = 60,
) -> None:
	"""Freeze authored genuine and extraction traffic without changing training data."""
	from distillation.benchmarks.traffic import freeze_traffic

	suite = freeze_traffic(
		output,
		seed=seed,
		genuine_count=genuine_count,
		attack_count=attack_count,
		duration_seconds=duration_seconds,
	)
	typer.echo(f"requests={len(suite.requests)} sha256={suite.sha256} output={output}")


@traffic_app.command("run")
def traffic_run_command(
	config: Annotated[Path, typer.Option("--config", exists=True)],
	phase: Annotated[str, typer.Option("--phase")] = "before",
) -> None:
	"""Replay against the unprotected teacher before, or the protected teacher after."""
	from distillation.benchmarks.traffic import load_traffic_config
	from distillation.benchmarks.traffic_runner import run_traffic

	if phase not in {"before", "after"}:
		raise typer.BadParameter("phase must be before or after")
	settings = load_traffic_config(config)
	selected = "before" if phase == "before" else "after"
	if selected not in settings.targets:
		raise typer.BadParameter(f"config must define targets.{selected}")
	typer.echo(f"Replaying {selected}; the full schedule must finish before results are complete.")
	run = asyncio.run(run_traffic(settings, selected))
	typer.echo(f"run={run.id} succeeded={sum(s.succeeded for s in run.samples)}/{len(run.samples)}")


@traffic_app.command("report")
def traffic_report_command(
	results_dir: Annotated[Path, typer.Option("--results-dir")] = Path(
		"outputs/traffic-benchmarks"
	),
	comparison: Annotated[str, typer.Option("--comparison")] = "",
	output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
	"""List traffic experiments or export a before/after harness comparison."""
	from distillation.benchmarks.models import save_json
	from distillation.benchmarks.traffic_reporting import build_traffic_report, traffic_experiments

	value = (
		build_traffic_report(results_dir, comparison)
		if comparison
		else traffic_experiments(results_dir)
	)
	if output:
		save_json(output, value)
	else:
		typer.echo(json.dumps(value, indent="\t"))


@traffic_app.command("demo")
def traffic_demo_command(
	output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("outputs/traffic-demo"),
) -> None:
	"""Write explicitly simulated traffic results for previewing the dashboard."""
	from distillation.benchmarks.traffic_demo import create_traffic_demo

	create_traffic_demo(output_dir)
	typer.echo(f"Simulated traffic fixtures written to {output_dir}")


if __name__ == "__main__":
	# scripts/pipeline.py and distill run start this module with python -m.
	app(prog_name="distill")
