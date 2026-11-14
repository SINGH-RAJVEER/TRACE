"""Watermark key IDs and offline detection over texts, such as a suspect student's outputs."""

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True)


@app.command("key-id")
def key_id_command(
	principal: str,
	namespace: Annotated[str, typer.Option(envvar="HARNESS_WATERMARK_NAMESPACE")] = "trace",
) -> None:
	"""Print the public key ID the gateway sends for a principal."""
	from distillation.watermark import key_id

	typer.echo(key_id(namespace, principal))


@app.command("detect")
def detect_command(
	texts: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, help="JSONL with a text field per line.")
	],
	key_file: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, envvar="TEACHER_WATERMARK_KEY_FILE")
	],
	principal: Annotated[
		list[str], typer.Option(help="Candidate principal; repeat for each. Corrected jointly.")
	],
	tokenizer: Annotated[str, typer.Option(help="Tokenizer of the watermarked teacher.")],
	output: Annotated[Path, typer.Option(dir_okay=False, help="New JSON report.")],
	tokenizer_revision: str = "",
	namespace: Annotated[str, typer.Option(envvar="HARNESS_WATERMARK_NAMESPACE")] = "trace",
	gamma: Annotated[float, typer.Option(envvar="TEACHER_WATERMARK_GAMMA")] = 0.25,
	delta: Annotated[float, typer.Option(envvar="TEACHER_WATERMARK_DELTA")] = 2.0,
	eligible_vocab: Annotated[int, typer.Option(envvar="TEACHER_WATERMARK_ELIGIBLE_VOCAB")] = 32768,
	alpha: float = 1e-3,
	minimum_tokens: int = 200,
) -> None:
	"""Test whether texts carry any candidate principal's watermark. Writes a new report."""
	from distillation.watermark import (
		WatermarkSpec,
		detect,
		hf_tokenizer,
		key_id,
		read_master_secret,
	)

	if output.exists():
		raise typer.BadParameter("the report must be a new file")
	try:
		spec = WatermarkSpec(gamma=gamma, delta=delta, eligible_vocab=eligible_vocab)
		rows = [json.loads(line) for line in texts.read_text().splitlines() if line.strip()]
		if not rows or any(not isinstance(row.get("text"), str) for row in rows):
			raise ValueError("every line needs a text field")
		encode = hf_tokenizer(tokenizer, tokenizer_revision)
		keys = {key_id(namespace, name): name for name in principal}
		report = detect(
			[encode(row["text"]) for row in rows],
			read_master_secret(key_file),
			list(keys),
			spec,
			alpha=alpha,
			minimum_tokens=minimum_tokens,
		)
	except ValueError as error:
		raise typer.BadParameter(str(error)) from error
	for result in report["results"]:
		result["principal"] = keys[result["key_id"]]
	report["tokenizer"] = {"model": tokenizer, "revision": tokenizer_revision or None}
	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text(json.dumps(report, indent=4) + "\n")
	for result in report["results"]:
		typer.echo(f"{result['principal']}: {result['decision']}")


@app.command("detect-kgw")
def detect_kgw_command(
	texts: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	key_file: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, envvar="TEACHER_WATERMARK_KEY_FILE")
	],
	principal: Annotated[list[str], typer.Option(help="Candidate principal; repeat for each.")],
	tokenizer: Annotated[str, typer.Option(help="Exact tokenizer used by the vLLM teacher.")],
	tokenizer_revision: Annotated[str, typer.Option(help="Pinned tokenizer revision.")],
	vocab_size: Annotated[int, typer.Option(envvar="TEACHER_WATERMARK_VOCAB_SIZE")],
	output: Annotated[Path, typer.Option(dir_okay=False)],
	namespace: Annotated[str, typer.Option(envvar="HARNESS_WATERMARK_NAMESPACE")] = "trace",
	gamma: Annotated[float, typer.Option(envvar="TEACHER_WATERMARK_GAMMA")] = 0.25,
	delta: Annotated[float, typer.Option(envvar="TEACHER_WATERMARK_DELTA")] = 2.0,
	context_width: Annotated[int, typer.Option(envvar="TEACHER_WATERMARK_CONTEXT_WIDTH")] = 2,
	epoch: Annotated[int, typer.Option(envvar="TEACHER_WATERMARK_EPOCH")] = 0,
	alpha: float = 1e-3,
	minimum_contexts: int = 200,
) -> None:
	"""Detect kgw-v1 in text pools, deduplicating contexts across all answers."""
	from dataclasses import asdict

	from trace_watermark.detect import detect_texts
	from trace_watermark.serving import KgwSpec, principal_key

	from distillation.watermark import hf_tokenizer, key_id, read_master_secret

	if output.exists():
		raise typer.BadParameter("the report must be a new file")
	if not tokenizer_revision.strip():
		raise typer.BadParameter("a pinned tokenizer revision is required")
	try:
		spec = KgwSpec(vocab_size, gamma, delta, context_width, epoch)
		rows = [json.loads(line) for line in texts.read_text().splitlines() if line.strip()]
		if not rows or any(
			not isinstance(row, dict) or not isinstance(row.get("text"), str) for row in rows
		):
			raise ValueError("every line needs a text field")
		encode = hf_tokenizer(tokenizer, tokenizer_revision)
		keys = {key_id(namespace, name): name for name in principal}
		secret = read_master_secret(key_file)
		results = detect_texts(
			[encode(row["text"]) for row in rows],
			[principal_key(secret, key, spec) for key in keys],
			spec.profile(),
			alpha=alpha,
			min_effective=minimum_contexts,
		)
	except ValueError as error:
		raise typer.BadParameter(str(error)) from error
	report = {
		"scheme": spec.version,
		"spec": asdict(spec),
		"spec_digest": spec.digest,
		"tokenizer": {"model": tokenizer, "revision": tokenizer_revision},
		"alpha": alpha,
		"minimum_contexts": minimum_contexts,
		"results": [
			{**asdict(result), "principal": keys[result.candidate_key_id]} for result in results
		],
	}
	output.parent.mkdir(parents=True, exist_ok=True)
	with output.open("x") as stream:
		stream.write(json.dumps(report, indent=4) + "\n")
	for result in results:
		typer.echo(f"{keys[result.candidate_key_id]}: {result.decision}")
