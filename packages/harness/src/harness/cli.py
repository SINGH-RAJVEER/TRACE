"""Command-line interface for calibrating the harness and replaying request streams."""

import json
import os
import secrets
from pathlib import Path
from typing import Annotated, Literal

import typer

from harness.calibration import (
	BenignRequest,
	HarnessConfig,
	calibrate,
	load_calibration,
	pooled_accounts,
	save_calibration,
)
from harness.replay import replay
from harness.serialization import Task

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
KEY_ENV = "HARNESS_DECEPTION_KEY"
KEY_FILE = Path(os.environ.get("HARNESS_DECEPTION_KEY_FILE", "artifacts/harness/deception.key"))


def deception_key(required: bool = True) -> bytes:
	"""The secret from HARNESS_DECEPTION_KEY, else the server's key file, as hex.

	Without either, a required key is an error and an optional one is a random key for this
	run only. The key never appears in output files.
	"""
	value = os.environ.get(KEY_ENV, "")
	if not value and KEY_FILE.exists():
		value = KEY_FILE.read_text().strip()
	if not value and not required:
		return secrets.token_bytes(32)
	try:
		key = bytes.fromhex(value)
	except ValueError as error:
		raise typer.BadParameter(f"{KEY_ENV} must be hex") from error
	if len(key) < 16:
		raise typer.BadParameter(f"set {KEY_ENV} or {KEY_FILE} to at least 16 bytes in hex")
	return key


def read_requests(path: Path) -> list[BenignRequest]:
	with path.open(encoding="utf-8") as source:
		return [BenignRequest.model_validate_json(line) for line in source if line.strip()]


@app.command("calibrate")
def calibrate_command(
	fitting: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	score_calibration: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	policy_calibration: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
	task: Annotated[
		Task,
		typer.Option(
			help="ag_news, squad, gsm8k, coding, or text: selects the detector serialization."
		),
	] = "text",
	window_size: Annotated[int, typer.Option(min=2)] = 100,
	seed: int = 42,
	device: str | None = None,
) -> None:
	"""Fit references, window p-values, and level thresholds on three benign request files.

	Each file holds BenignRequest JSON lines: account, text, and optional time in seconds.
	The three files must be disjoint; no attack traffic is used.
	"""
	from harness.encoder import BGEEncoder

	calibration, references = calibrate(
		config=HarnessConfig(task=task, window_size=window_size, seed=seed),
		encoder=BGEEncoder(device=device),
		fitting=read_requests(fitting),
		score_calibration=read_requests(score_calibration),
		policy_calibration=read_requests(policy_calibration),
		log=lambda message: typer.echo(f"calibrate: {message}", err=True),
	)
	save_calibration(output_dir, calibration, references)
	typer.echo(
		json.dumps(
			{
				"counts": calibration.counts,
				"thresholds": {
					name: value.model_dump() for name, value in calibration.thresholds.items()
				},
				"baselines": {
					name: value.model_dump() for name, value in calibration.baselines.items()
				},
				"output_dir": str(output_dir),
			},
			indent="\t",
		)
	)


@app.command("replay")
def replay_command(
	calibration_dir: Annotated[Path, typer.Option("--calibration", exists=True, file_okay=False)],
	requests: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option(dir_okay=False)],
	policy: Annotated[str, typer.Option(help="allow, quota, rate, or mmd")] = "mmd",
	accumulator: str = "paper",
	thresholds: Annotated[str, typer.Option(help="calibrated or ville")] = "calibrated",
	l4_action: Annotated[str, typer.Option(help="refuse or deceive")] = "refuse",
	deception_rate: Annotated[float, typer.Option(min=0.0, max=1.0)] = 1.0,
	actions: Annotated[
		str, typer.Option(help="graduated, or direct for the L4-only blocking ablation")
	] = "graduated",
	device: str | None = None,
) -> None:
	"""Replay a request file through one policy and summarize how far each account escalated.

	Requests without times are spaced one second apart in file order. Challenges are counted
	but not solved; the summary describes detection, not what a solving client would receive.
	L4 refuses by default. With --l4-action deceive it answers with keyed noise, keyed by
	HARNESS_DECEPTION_KEY or the server's key file, or by a key for this run only.
	"""
	from harness.encoder import BGEEncoder

	if thresholds not in {"calibrated", "ville"}:
		raise typer.BadParameter("thresholds must be calibrated or ville")
	mode: Literal["calibrated", "ville"] = "ville" if thresholds == "ville" else "calibrated"
	if l4_action not in {"refuse", "deceive"}:
		raise typer.BadParameter("l4-action must be refuse or deceive")
	action: Literal["refuse", "deceive"] = "deceive" if l4_action == "deceive" else "refuse"
	if actions not in {"graduated", "direct"}:
		raise typer.BadParameter("actions must be graduated or direct")
	ladder: Literal["graduated", "direct"] = "direct" if actions == "direct" else "graduated"
	calibration, scorer = load_calibration(calibration_dir)
	summary = replay(
		calibration=calibration,
		scorer=scorer,
		requests=read_requests(requests),
		output=output,
		policy=policy,
		accumulator=accumulator,
		thresholds=mode,
		encoder=BGEEncoder(device=device) if policy == "mmd" else None,
		l4_action=action,
		deception_rate=deception_rate,
		deception_key=deception_key(required=False) if action == "deceive" else None,
		actions=ladder,
	)
	typer.echo(json.dumps(summary, indent="\t"))


@app.command("watermark")
def watermark_command(
	task: Annotated[str, typer.Option(help="ag_news, squad, or gsm8k")],
	probes: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	controls: Annotated[int, typer.Option(min=1)] = 999,
) -> None:
	"""Test whether a suspect model learned the keyed noise answers.

	Each probe line holds request and suspect_response. The p-value ranks agreement under the
	server's key among agreements under unrelated control keys.
	"""
	from harness.deception import Probe, watermark_test

	if task not in {"ag_news", "squad", "gsm8k"}:
		raise typer.BadParameter("task must be ag_news, squad, or gsm8k")
	with probes.open(encoding="utf-8") as source:
		rows = [
			Probe(request=row["request"], suspect_response=row["suspect_response"])
			for row in (json.loads(line) for line in source if line.strip())
		]
	result = watermark_test(task, deception_key(), rows, controls)
	typer.echo(json.dumps(result, indent="\t"))


@app.command("bench")
def bench_command(
	calibration_dir: Annotated[Path, typer.Option("--calibration", exists=True, file_okay=False)],
	requests: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
	task: Annotated[
		str, typer.Option(help="noise format: ag_news, squad, gsm8k, or text")
	] = "text",
	accumulator: str = "paper",
	limit: Annotated[int, typer.Option(min=1)] = 2000,
	encode: Annotated[bool, typer.Option(help="time the BGE encoder too")] = True,
	device: str | None = None,
) -> None:
	"""Time the harness's own serving costs, unit by unit, for the paper's Table X.

	Replays up to --limit requests from a request file as one account. Victim inference is
	excluded; measure it with the traffic benchmark at the same load and hardware.
	"""
	from harness.bench import serving_costs

	calibration, scorer = load_calibration(calibration_dir)
	texts = [request.text for request in read_requests(requests)[:limit]]
	encoder = None
	if encode:
		from harness.encoder import BGEEncoder

		encoder = BGEEncoder(device=device)
	report = serving_costs(
		calibration=calibration,
		scorer=scorer,
		texts=texts,
		encoder=encoder,
		task=task,
		accumulator=accumulator,
		key=deception_key(required=False),
	)
	text = json.dumps(report, indent="\t")
	if output:
		output.parent.mkdir(parents=True, exist_ok=True)
		output.write_text(text + "\n", encoding="utf-8")
	typer.echo(text)


@app.command("pool-accounts")
def pool_accounts_command(
	pool: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option(dir_okay=False)],
	requests_per_account: Annotated[int, typer.Option(min=1)] = 500,
	seed: Annotated[int, typer.Option(help="Seed of the row shuffle before slicing.")] = 42,
) -> None:
	"""Turn a prompt pool into accounts of shuffled requests, for plumbing checks only.

	Rows are shuffled before slicing, so these accounts sample the task uniformly. They are not
	realistic legitimate clients. Real calibration needs accounts built with per-client topic
	concentration and timing.
	"""
	keyed: list[tuple[str, str]] = []
	with pool.open(encoding="utf-8") as source:
		for index, line in enumerate(line for line in source if line.strip()):
			record = json.loads(line)
			messages = record.get("prompt") or record.get("messages") or []
			text = next(m["content"] for m in reversed(messages) if m["role"] == "user")
			keyed.append((str(record.get("example_id", index)), text))
	rows = pooled_accounts(keyed, pool.stem, requests_per_account, seed)
	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
	typer.echo(f"wrote {len(rows):,} requests in {len({r.account for r in rows}):,} accounts")
