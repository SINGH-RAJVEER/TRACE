"""Command-line interface for collection, preparation, and training."""

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer

from distillation.collection import CollectionOptions, collect
from distillation.preparation import prepare_sft_dataset
from distillation.teacher import TeacherAPIGateway
from distillation.training import load_training_config, train

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


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
) -> None:
	"""Run completion-only LoRA or QLoRA supervised fine-tuning."""

	if os.environ.get("TOKENIZERS_PARALLELISM") is None:
		os.environ["TOKENIZERS_PARALLELISM"] = "false"
	output_dir = train(load_training_config(config_path))
	typer.echo(f"adapter={output_dir}")


if __name__ == "__main__":
	app()
