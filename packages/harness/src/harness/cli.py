"""Command-line interface for calibrating the harness and replaying request streams."""

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import typer

from harness.calibration import (
	BenignRequest,
	HarnessConfig,
	calibrate,
	load_calibration,
	load_shadow,
	pooled_accounts,
	save_calibration,
)
from harness.encoder import Encoder
from harness.mmd import Array
from harness.replay import replay, shadow_check
from harness.serialization import Task, detector_text, gateway_turn
from harness.shadow import ShadowConfig

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


def read_any_requests(path: Path) -> list[BenignRequest]:
	"""BenignRequest lines, or gateway schedule lines converted as `detector-input` does."""
	rows: list[BenignRequest] = []
	with path.open(encoding="utf-8") as source:
		for line in source:
			if not line.strip():
				continue
			value = json.loads(line)
			if "prompt" not in value:
				rows.append(BenignRequest.model_validate(value))
				continue
			prompt = value["prompt"]
			turn = next(m["content"] for m in reversed(prompt["prompt"]) if m["role"] == "user")
			rows.append(
				BenignRequest(
					account=value["account"],
					text=gateway_turn(prompt["task"], turn),
					time=value.get("time"),
					example_id=prompt.get("example_id"),
					client_type=value.get("client_type"),
				)
			)
	return rows


class CachedEncoder:
	"""Precomputed embeddings keyed by the SHA-256 hex of each detector text.

	The npz holds `keys` and `vectors`, as the audit's embedding cache does, so offline checks
	run without the BGE model. A missing text is an error, never a silent substitute.
	"""

	def __init__(self, path: Path) -> None:
		with np.load(path) as data:
			self.index = {str(key): row for row, key in enumerate(data["keys"])}
			self.vectors = np.asarray(data["vectors"], dtype=np.float64)
		self.path = path

	def encode(self, texts: list[str]) -> Array:
		try:
			rows = [self.index[hashlib.sha256(text.encode()).hexdigest()] for text in texts]
		except KeyError as error:
			raise ValueError(f"{self.path} has no embedding for a requested text") from error
		return self.vectors[rows]

	def identity(self) -> dict[str, Any]:
		return {
			"model": "cached",
			"sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
		}


def make_encoder(embeddings: Path | None, device: str | None) -> Encoder:
	if embeddings is not None:
		return CachedEncoder(embeddings)
	from harness.encoder import BGEEncoder

	return BGEEncoder(device=device)


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
	mixture_control_required: bool = False,
	validation_report: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	allow_invalid_workload: bool = False,
	seed: int = 42,
	device: str | None = None,
	shadow: Annotated[
		bool, typer.Option(help="Also fit the shadow scorers; telemetry only, no thresholds.")
	] = False,
	shadow_target: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.05,
	shadow_cells: Annotated[int, typer.Option(min=2)] = 128,
	embeddings: Annotated[
		Path | None,
		typer.Option(
			exists=True, dir_okay=False, help="Cached embeddings npz instead of the BGE model."
		),
	] = None,
) -> None:
	"""Fit references, window p-values, and level thresholds on three benign request files.

	Each file holds BenignRequest JSON lines: account, text, and optional time in seconds.
	The three files must be disjoint; no attack traffic is used. Without --shadow the artifact
	is the same as before the shadow scorers existed.
	"""
	paths = {
		"reference_fit": fitting,
		"score_calibration": score_calibration,
		"policy_development": policy_calibration,
	}
	rows = {role: read_requests(path) for role, path in paths.items()}
	new_workload = any(
		row.workload_version == "gateway-clients-v5" for values in rows.values() for row in values
	)
	validation = None
	if new_workload:
		issues = []
		if any(
			row.workload_version != "gateway-clients-v5"
			for values in rows.values()
			for row in values
		):
			issues.append("mixed or missing workload versions")
		if validation_report is None:
			issues.append("validation report is required")
		else:
			validation = json.loads(validation_report.read_text())
			if validation.get("version") != "role-validation-v1" or not validation.get("passed"):
				issues.append("role validation did not pass")
			for role, path in paths.items():
				if (
					validation.get("detector_sha256", {}).get(role)
					!= hashlib.sha256(path.read_bytes()).hexdigest()
				):
					issues.append(f"{role} detector input hash mismatch")
		if issues and not allow_invalid_workload:
			raise typer.BadParameter("; ".join(issues))
		validation = {
			"passed": not issues,
			"reasons": issues,
			"override": bool(issues and allow_invalid_workload),
			"report_sha256": hashlib.sha256(validation_report.read_bytes()).hexdigest()
			if validation_report
			else None,
		}

	calibration, references = calibrate(
		config=HarnessConfig(
			task=task,
			window_size=window_size,
			seed=seed,
			mixture_control_required=mixture_control_required,
		),
		encoder=make_encoder(embeddings, device),
		fitting=rows["reference_fit"],
		score_calibration=rows["score_calibration"],
		policy_calibration=rows["policy_development"],
		log=lambda message: typer.echo(f"calibrate: {message}", err=True),
		input_validation=validation,
		shadow=ShadowConfig(target=shadow_target, cells=shadow_cells) if shadow else None,
	)
	calibration.input_validation = validation
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


@app.command("shadow-check")
def shadow_check_command(
	calibration_dir: Annotated[Path, typer.Option("--calibration", exists=True, file_okay=False)],
	requests: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
	reorder: Annotated[
		str, typer.Option(help="none, or topic: group each horizon by domain and cell")
	] = "none",
	embeddings: Annotated[
		Path | None,
		typer.Option(
			exists=True, dir_okay=False, help="Cached embeddings npz instead of the BGE model."
		),
	] = None,
	device: str | None = None,
) -> None:
	"""Replay a request file through the shadow scorers only and report both scores.

	Accepts BenignRequest lines or gateway schedule lines. No policy runs. This is an offline
	development check: it does not measure false positives or confirm detection.
	"""
	if reorder not in {"none", "topic"}:
		raise typer.BadParameter("reorder must be none or topic")
	order: Literal["none", "topic"] = "topic" if reorder == "topic" else "none"
	calibration, _ = load_calibration(calibration_dir)
	if load_shadow(calibration_dir, calibration) is None:
		raise typer.BadParameter("the calibration has no shadow scorers; recalibrate with --shadow")
	rows = read_any_requests(requests)
	encoder = make_encoder(embeddings, device)
	vectors = encoder.encode([detector_text(calibration.config.task, row.text) for row in rows])
	report = shadow_check(calibration=calibration, requests=rows, embeddings=vectors, reorder=order)
	text = json.dumps(report, indent="\t")
	if output:
		output.parent.mkdir(parents=True, exist_ok=True)
		output.write_text(text + "\n", encoding="utf-8")
	typer.echo(text)


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


@app.command("adequacy")
def adequacy_command(
	calibration_dir: Annotated[Path, typer.Option("--calibration", exists=True, file_okay=False)],
	output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
	"""Write the selected MMD and quota adequacy verdict; fail when inadequate."""
	calibration, _ = load_calibration(calibration_dir)
	report = calibration.adequacy(accumulators=("paper", "instantaneous"), baselines=("quota",))
	with output.open("x", encoding="utf-8") as target:
		json.dump(report, target, indent="\t")
		target.write("\n")
	if not report["passed"]:
		raise typer.Exit(code=1)


review_app = typer.Typer(no_args_is_help=True)
app.add_typer(review_app, name="review")


def _review_db(path: Path) -> Any:
	import sqlite3

	if not path.is_file():
		raise typer.BadParameter("state database does not exist")
	db = sqlite3.connect(path, isolation_level=None)
	if not db.execute("SELECT 1 FROM sqlite_master WHERE name = 'review_appeals'").fetchone():
		db.close()
		raise typer.BadParameter("review is not enabled in this database")
	return db


@review_app.command("list")
def review_list(state_db: Annotated[Path, typer.Option(envvar="HARNESS_STATE_DB")]) -> None:
	from harness import review

	with _review_db(state_db) as db:
		typer.echo(json.dumps(review.list_appeals(db), indent=4))


@review_app.command("show")
def review_show(
	ref: str, state_db: Annotated[Path, typer.Option(envvar="HARNESS_STATE_DB")]
) -> None:
	from harness import review

	with _review_db(state_db) as db:
		result = review.show(db, ref)
		if result is None:
			raise typer.BadParameter("unknown appeal")
		typer.echo(json.dumps(result, indent=4))


@review_app.command("decide")
def review_decide(
	ref: str,
	action: Annotated[str, typer.Option()],
	operator: Annotated[str, typer.Option()],
	reason: Annotated[str, typer.Option()],
	state_db: Annotated[Path, typer.Option(envvar="HARNESS_STATE_DB")],
	requests: int = 0,
	tokens: int = 0,
	expires_at: float | None = None,
) -> None:
	from harness import review

	with _review_db(state_db) as db:
		try:
			db.execute("BEGIN IMMEDIATE")
			result = review.decide(
				db,
				ref,
				action=action,
				operator=operator,
				reason=reason,
				requests=requests,
				tokens=tokens,
				expires_at=expires_at,
			)
			db.execute("COMMIT")
		except ValueError as error:
			db.execute("ROLLBACK")
			raise typer.BadParameter(str(error)) from error
		typer.echo(result)
