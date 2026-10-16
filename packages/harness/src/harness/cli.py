"""Command-line interface for calibrating the harness and replaying request streams."""

import json
import os
from pathlib import Path
from typing import Annotated, Literal

import typer

from harness.calibration import (
	BenignRequest,
	HarnessConfig,
	calibrate,
	load_calibration,
	save_calibration,
)
from harness.replay import replay

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
KEY_ENV = "HARNESS_DECEPTION_KEY"


def deception_key() -> bytes:
	"""The secret from HARNESS_DECEPTION_KEY, as hex; it never appears in output files."""
	value = os.environ.get(KEY_ENV, "")
	try:
		key = bytes.fromhex(value)
	except ValueError as error:
		raise typer.BadParameter(f"{KEY_ENV} must be hex") from error
	if len(key) < 16:
		raise typer.BadParameter(f"set {KEY_ENV} to at least 16 random bytes in hex")
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
		config=HarnessConfig(window_size=window_size, seed=seed),
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
	device: str | None = None,
) -> None:
	"""Replay a request file through one policy and summarize how far each account escalated.

	Requests without times are spaced one second apart in file order. Challenges are counted
	but not solved; the summary describes detection, not what a solving client would receive.
	With --l4-action deceive, the key comes from HARNESS_DECEPTION_KEY.
	"""
	from harness.encoder import BGEEncoder

	if thresholds not in {"calibrated", "ville"}:
		raise typer.BadParameter("thresholds must be calibrated or ville")
	mode: Literal["calibrated", "ville"] = "ville" if thresholds == "ville" else "calibrated"
	if l4_action not in {"refuse", "deceive"}:
		raise typer.BadParameter("l4-action must be refuse or deceive")
	action: Literal["refuse", "deceive"] = "deceive" if l4_action == "deceive" else "refuse"
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
		deception_key=deception_key() if action == "deceive" else None,
	)
	typer.echo(json.dumps(summary, indent="\t"))


@app.command("watermark")
def watermark_command(
	task: Annotated[str, typer.Option(help="ag_news, squad, or gsm8k")],
	probes: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	controls: Annotated[int, typer.Option(min=1)] = 999,
) -> None:
	"""Test whether a suspect model learned the keyed wrong answers.

	Each probe line holds request, victim_response, and suspect_response. The p-value ranks
	agreement under HARNESS_DECEPTION_KEY among agreements under unrelated control keys.
	"""
	from harness.deception import DECEIVERS, Probe, watermark_test

	if task not in DECEIVERS:
		raise typer.BadParameter(f"task must be one of {sorted(DECEIVERS)}")
	with probes.open(encoding="utf-8") as source:
		rows = [Probe(**json.loads(line)) for line in source if line.strip()]
	result = watermark_test(DECEIVERS[task](), deception_key(), rows, controls)
	typer.echo(json.dumps(result, indent="\t"))


@app.command("pool-accounts")
def pool_accounts_command(
	pool: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option(dir_okay=False)],
	requests_per_account: Annotated[int, typer.Option(min=1)] = 500,
) -> None:
	"""Turn a prompt pool into accounts of consecutive requests, for plumbing checks only.

	These accounts sample the task uniformly, so they are not realistic legitimate clients.
	Real calibration needs accounts built with per-client topic concentration and timing.
	"""
	rows: list[BenignRequest] = []
	with pool.open(encoding="utf-8") as source:
		for index, line in enumerate(line for line in source if line.strip()):
			record = json.loads(line)
			messages = record.get("prompt") or record.get("messages") or []
			text = next(m["content"] for m in reversed(messages) if m["role"] == "user")
			rows.append(
				BenignRequest(account=f"{pool.stem}-{index // requests_per_account:05d}", text=text)
			)
	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
	typer.echo(f"wrote {len(rows):,} requests in {len({r.account for r in rows}):,} accounts")
