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
	"""Time the harness's own serving costs, unit by unit.

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


@app.command("evidence-calibrate")
def evidence_calibrate_command(
	calibration_dir: Annotated[Path, typer.Option("--calibration", exists=True, file_okay=False)],
	score_calibration: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option(dir_okay=False)],
	embeddings: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	corpus_index: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	coverage_seed: int | None = None,
	coverage_width: int | None = None,
	coverage_ridge: float | None = None,
	device: str | None = None,
) -> None:
	"""Fit separate opt-in evidence nulls using only the declared score-calibration role."""
	from harness.corpus import CorpusIndex
	from harness.evidence import file_hash, fit_evidence
	from harness.logdet import CoverageSpec
	from harness.path_runtime import encoder_contract

	options = (coverage_seed, coverage_width, coverage_ridge)
	if any(value is not None for value in options) and any(value is None for value in options):
		raise typer.BadParameter("coverage requires explicit seed, width and ridge")
	coverage = (
		CoverageSpec(seed=coverage_seed, input_dim=coverage_width, ridge=coverage_ridge)
		if coverage_seed is not None and coverage_width is not None and coverage_ridge is not None
		else None
	)

	calibration, _ = load_calibration(calibration_dir)
	rows = read_any_requests(score_calibration)
	from harness.calibration import requests_sha256

	if requests_sha256(rows) != calibration.sha256.get("score_calibration"):
		raise typer.BadParameter("input must be the calibration's declared score_calibration role")
	load_shadow(calibration_dir, calibration)
	encoder = make_encoder(embeddings, device)
	vectors = encoder.encode([detector_text(calibration.config.task, row.text) for row in rows])
	corpus = CorpusIndex.load(corpus_index) if corpus_index else None
	null = fit_evidence(rows, vectors, calibration, corpus, coverage)
	artifact = {
		"version": "evidence-v1",
		"role": "score_calibration",
		"requests_sha256": file_hash(score_calibration),
		"calibration_sha256": file_hash(calibration_dir / "calibration.json"),
		"encoder": encoder_contract(encoder.identity())
		if coverage is not None
		else encoder.identity(),
		"null": null.model_dump(mode="json"),
	}
	if coverage is not None:
		artifact["fit_accounts"] = sorted({row.account for row in rows})
		artifact["fit_prompts"] = sorted(
			{hashlib.sha256(row.text.encode()).hexdigest() for row in rows}
		)
	output.parent.mkdir(parents=True, exist_ok=True)
	with output.open("x", encoding="utf-8") as sink:
		sink.write(json.dumps(artifact, indent="\t") + "\n")


def _monitor_campaigns(
	path: Path, task: str, encoder: Encoder
) -> tuple[list[str], list[list[tuple[str, str, Any]]]]:
	"""Each account's requests in order as (detector text, domain, embedding) campaigns."""
	from harness.shadow import request_domain

	rows = read_any_requests(path)
	vectors = encoder.encode([detector_text(task, row.text) for row in rows])  # type: ignore[arg-type]
	streams: dict[str, list[tuple[str, str, Any]]] = {}
	for row, vector in zip(rows, vectors, strict=True):
		streams.setdefault(row.account, []).append(
			(detector_text(task, row.text), request_domain(task, row.text), vector)  # type: ignore[arg-type]
		)
	return list(streams), list(streams.values())


@app.command("evidence-monitor-calibrate")
def evidence_monitor_calibrate_command(
	fitting: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	score_calibration: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	policy_calibration: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option(dir_okay=False, help="New monitor calibration.")],
	task: str = "gateway",
	embeddings: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	device: str | None = None,
	look_every: int = 50,
	rarefied_size: int = 40,
	alpha: float = 0.01,
	cells: int = 64,
	seed: int = 0,
	encoder_model: Annotated[
		str | None, typer.Option(help="Model behind --embeddings, checked at API startup.")
	] = None,
	encoder_revision: str | None = None,
) -> None:
	"""Fit the whole-campaign evidence monitor from three disjoint benign roles.

	Feature nulls come from score-calibration accounts and the alert rule from separate
	policy-calibration accounts. Never pass the locked benign test.
	"""
	from harness.campaign_evidence import EvidenceSettings, calibrate

	if embeddings is not None and not (encoder_model and encoder_revision):
		raise typer.BadParameter(
			"cached embeddings need --encoder-model and --encoder-revision of the model behind them"
		)
	encoder = make_encoder(embeddings, device)
	identity = (
		{"model": encoder_model, "revision": encoder_revision, "source": "declared"}
		if embeddings is not None
		else encoder.identity()
	)
	settings = EvidenceSettings(
		look_every=look_every, rarefied_size=rarefied_size, alpha=alpha, cells=cells, seed=seed
	)
	fit_rows = read_any_requests(fitting)
	fit_vectors = encoder.encode([detector_text(task, r.text) for r in fit_rows])  # type: ignore[arg-type]
	score_accounts, score = _monitor_campaigns(score_calibration, task, encoder)
	policy_accounts, policy = _monitor_campaigns(policy_calibration, task, encoder)
	if set(score_accounts) & set(policy_accounts):
		raise typer.BadParameter("score and policy calibration must use different accounts")
	monitor = calibrate(
		fitting=np.asarray(fit_vectors, dtype=np.float64),
		score_campaigns=score,
		policy_campaigns=policy,
		settings=settings,
		encoder=identity,
	)
	output.parent.mkdir(parents=True, exist_ok=True)
	with output.open("x", encoding="utf-8") as sink:
		sink.write(monitor.model_dump_json(indent=4) + "\n")
	for profile, maxima in monitor.maxima.items():
		typer.echo(
			f"{profile}: {len(maxima)} benign campaigns, horizon "
			f"{monitor.horizon_looks[profile]} looks of {look_every}"
		)


@app.command("evidence-monitor-check")
def evidence_monitor_check_command(
	monitor: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	requests: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option(dir_okay=False, help="New JSON report.")],
	task: str = "gateway",
	embeddings: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	device: str | None = None,
) -> None:
	"""Replay accounts through the monitor offline and report each one's first alert."""
	from harness.campaign_evidence import EvidenceCalibration, new_state, observe

	calibration = EvidenceCalibration.model_validate_json(monitor.read_text())
	centroids = calibration.cell_centroids()
	accounts, campaigns = _monitor_campaigns(requests, task, make_encoder(embeddings, device))
	report = []
	for account, campaign in zip(accounts, campaigns, strict=True):
		state = new_state()
		looks = []
		for text, domain, vector in campaign:
			result = observe(
				state,
				text=text,
				domain=domain,
				embedding=vector,
				calibration=calibration,
				centroids=centroids,
			)
			if result is not None:
				looks.append({k: v for k, v in result.items() if k != "features"})
		alert = next((look for look in looks if look["decision"] == "alert"), None)
		report.append(
			{
				"account": account,
				"requests": len(campaign),
				"looks": len(looks),
				"alert_look": alert["look"] if alert else None,
				"alert_requests": alert["requests"] if alert else None,
				"final": looks[-1] if looks else None,
			}
		)
	summary = {
		"version": calibration.settings.version,
		"monitor_sha256": calibration.digest,
		"accounts": len(report),
		"alerted": sum(r["alert_look"] is not None for r in report),
		"results": report,
		"qualified": False,
	}
	output.parent.mkdir(parents=True, exist_ok=True)
	with output.open("x", encoding="utf-8") as sink:
		sink.write(json.dumps(summary, indent=4) + "\n")
	typer.echo(f"{summary['alerted']} of {summary['accounts']} accounts alerted")


@app.command("corpus-index")
def corpus_index_command(
	sources: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
	"""Index an operator's JSON mapping of pinned source labels to prompt strings."""
	from harness.corpus import CorpusIndex

	if output.exists():
		raise typer.BadParameter("corpus index output already exists")
	index = CorpusIndex.from_sources(json.loads(sources.read_text()))
	index.save(output)
	typer.echo(json.dumps({"sha256": index.digest, "output": str(output)}))


@app.command("evidence-check")
def evidence_check_command(
	calibration_dir: Annotated[Path, typer.Option("--calibration", exists=True, file_okay=False)],
	evidence: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	requests: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
	embeddings: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	corpus_index: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	links: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	reorder: str = "none",
	device: str | None = None,
) -> None:
	"""Offline account and optional cluster evidence for benign or A1-A8 schedule files."""
	from harness.corpus import CorpusIndex
	from harness.evidence import check_evidence, file_hash
	from harness.path_runtime import encoder_contract
	from harness.signals import SignalCalibration

	if reorder not in {"none", "topic"}:
		raise typer.BadParameter("reorder must be none or topic")
	calibration, _ = load_calibration(calibration_dir)
	artifact = json.loads(evidence.read_text())
	if artifact["calibration_sha256"] != file_hash(calibration_dir / "calibration.json"):
		raise typer.BadParameter("evidence was fitted against a different calibration")
	load_shadow(calibration_dir, calibration)
	null = SignalCalibration.model_validate(artifact["null"])
	corpus = CorpusIndex.load(corpus_index) if corpus_index else None
	rows = read_any_requests(requests)
	if null.coverage is not None and (
		{row.account for row in rows} & set(artifact.get("fit_accounts", []))
		or {hashlib.sha256(row.text.encode()).hexdigest() for row in rows}
		& set(artifact.get("fit_prompts", []))
		or file_hash(requests) == artifact["requests_sha256"]
	):
		raise typer.BadParameter("coverage comparison requires held-out accounts and prompts")
	encoder = make_encoder(embeddings, device)
	if (
		encoder_contract(encoder.identity()) != encoder_contract(artifact["encoder"])
		if null.coverage is not None
		else encoder.identity() != artifact["encoder"]
	):
		raise typer.BadParameter("evidence encoder differs from the calibrated encoder")
	vectors = encoder.encode([detector_text(calibration.config.task, row.text) for row in rows])
	report = check_evidence(rows, vectors, calibration, null, corpus, reorder)
	if calibration.shadow is not None:
		load_shadow(calibration_dir, calibration)
		report["existing_shadow"] = shadow_check(
			calibration=calibration,
			requests=rows,
			embeddings=vectors,
			reorder="topic" if reorder == "topic" else "none",
		)
	if links:
		from harness.evidence import cluster_evidence

		report["clusters"] = cluster_evidence(
			rows, vectors, calibration, null, json.loads(links.read_text()), corpus
		)
	text = json.dumps(report, indent="\t")
	if output:
		output.parent.mkdir(parents=True, exist_ok=True)
		with output.open("x", encoding="utf-8") as sink:
			sink.write(text + "\n")
	typer.echo(text)


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


linkage_app = typer.Typer(no_args_is_help=True)
app.add_typer(linkage_app, name="linkage")


def _linkage_tx(path: Path, config_path: Path) -> Any:
	import sqlite3

	from harness.linkage import load_linkage_config
	from harness.store import _SQLiteTransaction

	if not path.is_file():
		raise typer.BadParameter("state database does not exist")
	config = load_linkage_config(config_path)
	db = sqlite3.connect(path, isolation_level=None)
	bound = db.execute("SELECT value FROM meta WHERE key = 'linkage_config'").fetchone()
	if bound is None or bound[0] != config.digest:
		db.close()
		raise typer.BadParameter("the database is not bound to this linkage config")
	return db, _SQLiteTransaction(db), config


@linkage_app.command("show")
def linkage_show(
	principal: str,
	state_db: Annotated[Path, typer.Option(envvar="HARNESS_STATE_DB")],
	config: Annotated[Path, typer.Option(envvar="HARNESS_LINKAGE_CONFIG", exists=True)],
) -> None:
	"""The principal's inferred cluster and the signal kinds behind each pair link."""
	from harness.linkage import cluster_id, members, strength

	db, tx, linkage = _linkage_tx(state_db, config)
	with db:
		cluster = members(tx, principal)
		pairs = []
		for index, left in enumerate(cluster):
			for right in cluster[index + 1 :]:
				evidence = tx.linkage(
					"e:" + json.dumps(sorted((left, right)), separators=(",", ":"))
				)
				if evidence:
					pairs.append(
						{
							"pair": [left, right],
							"evidence": evidence,
							"strength": strength(linkage, evidence),
						}
					)
		result = {"cluster": cluster_id(linkage, cluster), "members": cluster, "pairs": pairs}
		typer.echo(json.dumps({**result, "qualified": False}, indent=4))


@linkage_app.command("sever")
def linkage_sever(
	left: str,
	right: str,
	operator: Annotated[str, typer.Option()],
	reason: Annotated[str, typer.Option()],
	state_db: Annotated[Path, typer.Option(envvar="HARNESS_STATE_DB")],
	config: Annotated[Path, typer.Option(envvar="HARNESS_LINKAGE_CONFIG", exists=True)],
) -> None:
	"""Record that two principals are unrelated and rebuild their cluster without that link."""
	import time

	from harness.linkage import LinkageError, sever

	db, tx, linkage = _linkage_tx(state_db, config)
	with db:
		try:
			db.execute("BEGIN IMMEDIATE")
			groups = sever(
				tx, linkage, left, right, operator=operator, reason=reason, now=time.time()
			)
			db.execute("COMMIT")
		except LinkageError as error:
			db.execute("ROLLBACK")
			raise typer.BadParameter(str(error)) from error
		typer.echo(json.dumps({"clusters": groups}, indent=4))


@app.command("separation")
def separation_command(
	benign: Annotated[
		list[Path],
		typer.Option(exists=True, dir_okay=False, help="Benign development role; repeatable."),
	],
	attack: Annotated[
		list[str], typer.Option(help="NAME=PATH of a gateway schedule or ledger; repeatable.")
	],
	output: Annotated[Path, typer.Option(file_okay=False)],
	embeddings: Annotated[
		Path | None,
		typer.Option(
			exists=True, dir_okay=False, help="Cached embeddings npz instead of the BGE model."
		),
	] = None,
	calibration_dir: Annotated[
		Path | None,
		typer.Option(
			"--calibration", exists=True, file_okay=False, help="Pooled MMD baseline references."
		),
	] = None,
	topic_fitting: Annotated[
		Path | None,
		typer.Option(exists=True, dir_okay=False, help="Requests for the topic cells' k-means."),
	] = None,
	order: Annotated[str, typer.Option(help="schedule, topic, or both")] = "both",
	window_size: int = 100,
	source_key: Annotated[
		str, typer.Option(help="group_id, source_id, or content_sha256")
	] = "group_id",
	topic_cells: int = 8,
	permutations: int = 999,
	bootstrap: int = 2000,
	seed: int = 42,
	device: str | None = None,
) -> None:
	"""Development-only separation study: C2ST, learned MMD, pooled MMD and volume features.

	Writes report.json and summary.md. Results are development evidence and never feed
	calibration or enforcement. Locked benign_test inputs are refused.
	"""
	from harness.separation import StudyConfig, run_study, summary_markdown

	attacks: list[tuple[str, Path]] = []
	for value in attack:
		name, separator, path = value.partition("=")
		if not separator or not name or not Path(path).is_file():
			raise typer.BadParameter(f"attack must be NAME=PATH of an existing file: {value}")
		attacks.append((name, Path(path)))
	paths = [*benign, *(p for _, p in attacks), *([topic_fitting] if topic_fitting else [])]
	if any("benign_test" in part for p in paths for part in p.parts):
		raise typer.BadParameter("the separation study must not read benign_test")
	orders: dict[str, list[Literal["schedule", "topic"]]] = {
		"schedule": ["schedule"],
		"topic": ["topic"],
		"both": ["schedule", "topic"],
	}
	if order not in orders:
		raise typer.BadParameter("order must be schedule, topic, or both")
	if output.exists() and any(output.iterdir()):
		raise typer.BadParameter("output directory already has files")
	config = StudyConfig.model_validate(
		{
			"window_size": window_size,
			"source_key": source_key,
			"topic_cells": topic_cells,
			"permutations": permutations,
			"bootstrap": bootstrap,
			"seed": seed,
		}
	)
	scorer = None
	if calibration_dir is not None:
		calibration, fixed = load_calibration(calibration_dir)
		scorer = (fixed, calibration.config.distinct_prompts)
	report = run_study(
		benign=benign,
		attacks=attacks,
		encoder=make_encoder(embeddings, device),
		config=config,
		orders=orders[order],
		scorer=scorer,
		topic_fitting=topic_fitting,
	)
	output.mkdir(parents=True, exist_ok=True)
	with (output / "report.json").open("x", encoding="utf-8") as sink:
		sink.write(json.dumps(report, indent="\t") + "\n")
	with (output / "summary.md").open("x", encoding="utf-8") as sink:
		sink.write(summary_markdown(report))
	typer.echo(f"{report['decision']} ({output / 'summary.md'})")


@app.command("path-calibrate")
def path_calibrate_command(
	calibration_dir: Annotated[Path, typer.Option("--calibration", exists=True, file_okay=False)],
	campaigns: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option(dir_okay=False)],
	request_horizon: Annotated[int, typer.Option(min=2)],
	stratum: Annotated[str, typer.Option()],
	alpha: float = 0.01,
	streams: str = "mmd",
	weights: str | None = None,
	evidence: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	embeddings: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	device: str | None = None,
) -> None:
	"""Fit an unqualified terminal stop from complete benign principal paths, never test data."""
	from harness.evidence import file_hash
	from harness.path_calibration import PathScoring, fit_path_calibration
	from harness.path_runtime import (
		PathRecipe,
		RuntimePathCalibration,
		detector_hash,
		encoder_contract,
		path_look,
	)
	from harness.shadow import request_domain
	from harness.signals import SignalCalibration

	calibration, scorer = load_calibration(calibration_dir)
	if request_horizon < scorer.window_size or request_horizon % scorer.window_size:
		raise typer.BadParameter("request horizon must be a positive multiple of window size")
	names = tuple(streams.split(","))
	if set(names) - {"mmd", "coverage_mean_logdet"}:
		raise typer.BadParameter("streams must be mmd and/or coverage_mean_logdet")
	scoring = PathScoring(
		name="prompt-path-v1",
		streams=names,
		weights=tuple(float(w) for w in weights.split(",")) if weights else None,
	)
	rows = read_requests(campaigns)
	encoder = make_encoder(embeddings, device)
	vectors = encoder.encode([detector_text(calibration.config.task, row.text) for row in rows])
	null = None
	if evidence:
		payload = json.loads(evidence.read_text())
		if payload["calibration_sha256"] != file_hash(calibration_dir / "calibration.json"):
			raise typer.BadParameter("evidence was fitted against a different calibration")
		if encoder_contract(payload["encoder"]) != encoder_contract(encoder.identity()):
			raise typer.BadParameter("evidence uses a different encoder")
		if {row.account for row in rows} & set(payload.get("fit_accounts", [])) or {
			hashlib.sha256(row.text.encode()).hexdigest() for row in rows
		} & set(payload.get("fit_prompts", [])):
			raise typer.BadParameter("path calibration must be disjoint from feature-null fitting")
		null = SignalCalibration.model_validate(payload["null"])
	if "coverage_mean_logdet" in names and (null is None or null.coverage is None):
		raise typer.BadParameter("coverage requires --evidence with fitted coverage nulls")
	recipe = PathRecipe(
		detector_hash=detector_hash(calibration, scorer),
		encoder=encoder_contract(encoder.identity()),
		request_horizon=request_horizon,
		window_size=scorer.window_size,
		evidence=null,
	)
	groups: dict[str, list[int]] = {}
	for i, row in enumerate(rows):
		groups.setdefault(row.account, []).append(i)
	paths = []
	for indices in groups.values():
		if len(indices) > request_horizon or len(indices) < scorer.window_size:
			raise typer.BadParameter("each campaign needs one complete look and at most H prompts")
		if len(indices) % scorer.window_size:
			raise typer.BadParameter("campaign paths must end on a predetermined look")
		path = []
		for end in range(scorer.window_size, len(indices) + 1, scorer.window_size):
			prefix = indices[:end]
			values, reason = path_look(
				recipe,
				names,
				vectors[prefix].tolist(),
				[hashlib.sha256(rows[i].text.encode()).hexdigest() for i in prefix],
				{request_domain(calibration.config.task, rows[i].text) for i in prefix},
				calibration,
				scorer,
				stratum,
			)
			if values is None:
				raise typer.BadParameter(f"benign campaign has an unsupported look: {reason}")
			path.append(values)
		paths.append(path)
	fitted = fit_path_calibration(
		paths,
		scoring=scoring,
		alpha=alpha,
		horizon=request_horizon // scorer.window_size,
		stratum=stratum,
	)
	artifact = RuntimePathCalibration(
		recipe=recipe, calibration=fitted, benign_paths_sha256=file_hash(campaigns)
	)
	output.parent.mkdir(parents=True, exist_ok=True)
	with output.open("x", encoding="utf-8") as sink:
		sink.write(artifact.model_dump_json(indent=4).replace("    ", "\t") + "\n")
	typer.echo(
		json.dumps(
			{
				"calibration_hash": artifact.input_hash,
				"campaigns": fitted.n,
				"abstained": fitted.abstained,
				"abstention_reason": fitted.abstention_reason,
				"qualified": False,
			}
		)
	)


@app.command("campaign-coverage-calibrate")
def campaign_coverage_calibrate_command(
	monitor: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	campaigns: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option(dir_okay=False)],
	embeddings: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	alpha: float = 0.01,
	device: str = "cpu",
) -> None:
	"""Fit independent benign accounts at a complete, declared domain/prefix schedule."""
	from harness.campaign_coverage import CoverageMonitor, fit_coverage_calibration
	from harness.path_runtime import encoder_contract

	recipe = CoverageMonitor.model_validate_json(monitor.read_text())
	encoder = make_encoder(embeddings, device)
	if encoder_contract(encoder.identity()) != recipe.encoder_identity:
		raise typer.BadParameter("encoder identity differs from coverage monitor")
	rows = read_requests(campaigns)
	# These files declare a single domain. Bounds apply to what serving encodes.
	texts = [recipe.serialize(row.text) for row in rows]
	if any(len(text) > recipe.max_text_chars for text in texts):
		raise typer.BadParameter("calibration input exceeds coverage character bound")
	complete = getattr(encoder, "input_complete", None)
	if embeddings is None and (not callable(complete) or any(not complete(t) for t in texts)):
		raise typer.BadParameter("all calibration inputs must have complete embeddings")
	vectors = encoder.encode(texts)
	paths: dict[str, list[Any]] = {}
	for row, vector in zip(rows, vectors, strict=True):
		paths.setdefault(row.account, []).append(vector)
	artifact = fit_coverage_calibration(recipe, paths.values(), alpha)
	with output.open("x") as target:
		target.write(
			json.dumps(artifact.model_dump(mode="json"), indent="\t", allow_nan=False) + "\n"
		)
	typer.echo(json.dumps({"calibration_sha256": artifact.digest, "qualified": False}))


@app.command("link-accounts")
def link_accounts_command(
	requests: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option()],
	max_accounts: int = 256,
	max_requests: int = 512,
	max_features: int = 256,
	max_df: float = 0.5,
	minimum_overlap: float = 0.5,
	time_bin_seconds: float = 10,
	minimum_coactivity: float = 0.5,
) -> None:
	"""Propose offline inferred links from rare prompt overlap AND timing co-activity."""
	from harness.account_links import propose_links

	try:
		# Bound ingestion before collecting a potentially large file in memory.
		rows: list[BenignRequest] = []
		with requests.open() as stream:
			for line in stream:
				if len(line) > 131072 or len(rows) >= max_accounts * max_requests:
					raise ValueError("input exceeds bounds")
				if line.strip():
					rows.append(BenignRequest.model_validate_json(line))
		result = propose_links(
			rows,
			max_accounts=max_accounts,
			max_requests=max_requests,
			max_features=max_features,
			max_df=max_df,
			minimum_overlap=minimum_overlap,
			time_bin_seconds=time_bin_seconds,
			minimum_coactivity=minimum_coactivity,
		)
		with output.open("x") as target:
			target.write(json.dumps(result, indent="\t") + "\n")
	except (ValueError, OSError) as error:
		raise typer.BadParameter(str(error)) from error
