"""Generate inspectable gateway inputs without running a model or changing task pools."""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from distillation.benchmarks.models import save_json
from distillation.io import read_jsonl, write_jsonl
from distillation.schedules import campaign_schedule
from distillation.schemas import GatewayRequest, PromptRecord
from distillation.tasks import LEGITIMATE_ROLES, TASKS, task_directory
from distillation.workloads import (
	DEFAULT_CONCENTRATED_SHARE,
	DEFAULT_LENGTH_TAIL,
	DEFAULT_MAX_REQUESTS_PER_ACCOUNT,
	mixed_clients,
	task_references,
)

app = typer.Typer(no_args_is_help=True)


def _pools(
	tasks_dir: Path, domains: str, role: str, *, external: bool = False
) -> tuple[dict[str, list[PromptRecord]], dict[str, int], dict[str, Any]]:
	names = [name.strip() for name in domains.split(",")]
	if len(set(names)) != len(names) or len(names) < 2 or set(names) - TASKS.keys():
		raise ValueError("domains must list at least two distinct supported tasks")
	pools, caps, provenance = {}, {}, {}
	for domain in sorted(names):
		directory = task_directory(domain, root=tasks_dir)
		manifest_path = directory / "manifest.json"
		manifest = json.loads(manifest_path.read_text())
		if manifest["task"] != domain:
			raise ValueError(f"task manifest does not match domain {domain}")
		caps[domain] = manifest["max_output_tokens"]
		provenance[domain] = {
			"path": str(manifest_path),
			"sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
			"dataset": manifest.get("dataset"),
		}
		path = (
			directory / "attacker.jsonl"
			if role == "attacker"
			else directory / "legitimate" / f"{role}.jsonl"
		)
		if external:
			path = directory / "external" / "attacker.jsonl"
			pinned = directory / "external" / "manifest.json"
			if (
				hashlib.sha256(path.read_bytes()).hexdigest()
				!= json.loads(pinned.read_text())["sha256"][path.name]
			):
				raise ValueError(f"external pool differs from its manifest: {path}")
			provenance[domain]["external"] = {
				"path": str(pinned),
				"sha256": hashlib.sha256(pinned.read_bytes()).hexdigest(),
			}
		pools[domain] = read_jsonl(path, PromptRecord)
	return pools, caps, provenance


def _write(output: Path, requests: list[GatewayRequest], manifest: dict[str, Any]) -> None:
	# Existing campaign inputs are immutable, including partial generation attempts.
	output.mkdir(parents=True, exist_ok=False)
	path = output / "requests.jsonl"
	write_jsonl(path, requests)
	manifest["sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
	save_json(output / "manifest.json", manifest)
	typer.echo(json.dumps(manifest, indent="\t"))


@app.command("benign")
def benign_command(
	output: Annotated[Path, typer.Option()],
	accounts: Annotated[int, typer.Option(min=1)],
	role: str = "benign_test",
	tasks_dir: Path = Path("data"),
	domains: str = "ag_news,squad,gsm8k,coding",
	generalist_share: Annotated[float, typer.Option(min=0, max=1)] = 0.1,
	concentrated_share: Annotated[
		float, typer.Option(min=0, max=0.8, help="Specialist and repetitive client share.")
	] = DEFAULT_CONCENTRATED_SHARE,
	requests_per_account: Annotated[
		int, typer.Option(min=1, help="Minimum length of non-short accounts.")
	] = 100,
	length_tail: Annotated[
		float, typer.Option(min=0, help="Pareto shape of account lengths; 0 keeps them fixed.")
	] = DEFAULT_LENGTH_TAIL,
	max_requests_per_account: Annotated[int, typer.Option(min=1)] = (
		DEFAULT_MAX_REQUESTS_PER_ACCOUNT
	),
	seed: int = 42,
	workload_version: str = "gateway-clients-v3",
) -> None:
	"""Generate mixed benign clients for one separated calibration or test role."""
	try:
		if workload_version != "gateway-clients-v3":
			raise ValueError("use gateway profile-build for source-family roles")
		if role not in LEGITIMATE_ROLES:
			raise ValueError(f"role must be one of {', '.join(LEGITIMATE_ROLES)}")
		pools, caps, provenance = _pools(tasks_dir, domains, role)
		requests, manifest = mixed_clients(
			pools=pools,
			caps=caps,
			role=role,
			accounts=accounts,
			generalist_share=generalist_share,
			requests_per_account=requests_per_account,
			length_tail=length_tail or None,
			max_requests_per_account=max_requests_per_account,
			seed=seed,
			concentrated=concentrated_share,
			references={
				key: item
				for domain in pools
				for key, item in task_references(task_directory(domain, root=tasks_dir)).items()
			},
		)
		manifest["task_manifests"] = provenance
		_write(output, requests, manifest)
	except (ValueError, OSError, KeyError) as error:
		raise typer.BadParameter(str(error)) from None


@app.command("schedule")
def schedule_command(
	output: Annotated[Path, typer.Option()],
	attack: Annotated[
		str,
		typer.Option(
			help="A1 multiplexed, A2 domain-sequential, A3 cover traffic, A4 slow (A1 at the "
			"given interval), A5 Sybil per domain, A6 Sybil random, A7 independent sources."
		),
	],
	budget: Annotated[int, typer.Option(min=1, help="Campaign-wide HTTP attempt limit.")] = 12_000,
	tasks_dir: Path = Path("data"),
	domains: str = "ag_news,squad,gsm8k,coding",
	campaign_id: str = "campaign",
	accounts: Annotated[
		int | None, typer.Option(min=1, help="Total accounts; defaults to 6 for A5, 1 for A1/A2.")
	] = None,
	interval_seconds: Annotated[float, typer.Option(min=0)] = 1,
	deadline_seconds: Annotated[float, typer.Option(min=0)] = 86400,
	cover_share: Annotated[
		float, typer.Option(min=0, max=1, help="A3 share of the budget spent on cover.")
	] = 0.5,
	seed: int = 42,
) -> None:
	"""Generate a fixed campaign inventory with one budget shared by all its accounts.

	A3 and A7 read TASK/external/attacker.jsonl, built by `distill gateway external`; A3 draws
	its cover traffic from the task attacker pools.
	"""
	from typing import cast

	from distillation.schedules import EXTERNAL_ATTACKS, Attack

	try:
		external = attack in EXTERNAL_ATTACKS
		pools, caps, provenance = _pools(tasks_dir, domains, "attacker", external=external)
		cover = _pools(tasks_dir, domains, "attacker")[0] if attack == "A3" else None
		requests, manifest = campaign_schedule(
			pools=pools,
			caps=caps,
			attack=cast(Attack, attack),
			budget=budget,
			seed=seed,
			campaign_id=campaign_id,
			accounts=accounts,
			interval_seconds=interval_seconds,
			deadline_seconds=deadline_seconds,
			cover_pools=cover,
			cover_share=cover_share,
		)
		manifest["task_manifests"] = provenance
		_write(output, requests, manifest)
	except (ValueError, OSError, KeyError) as error:
		raise typer.BadParameter(str(error)) from None


@app.command("external")
def external_command(
	task: Annotated[str, typer.Option(help="ag_news, squad, gsm8k, or coding.")],
	tasks_dir: Path = Path("data"),
) -> None:
	"""Build TASK/external/attacker.jsonl from the domain's independently pinned A7 source."""
	from distillation.external_sources import build_external

	try:
		manifest = build_external(
			task=task, task_dir=task_directory(task, root=tasks_dir), log=typer.echo
		)
		typer.echo(json.dumps({k: manifest[k] for k in ("prompts", "filtered")}, indent="\t"))
	except (ValueError, OSError, KeyError) as error:
		raise typer.BadParameter(str(error)) from None


@app.command("collect")
def collect_command(
	input_dir: Annotated[Path, typer.Option(help="Immutable gateway schedule directory.")],
	output: Annotated[Path, typer.Option()],
	teacher_url: str = "http://127.0.0.1:8003",
	keys_file: Path | None = None,
	token: Annotated[str, typer.Option(envvar="TEACHER_API_TOKEN")] = "",
	terms_review_id: Annotated[str, typer.Option(envvar="TEACHER_TERMS_REVIEW_ID")] = "",
	decision_log: Path | None = None,
	concurrency: Annotated[int, typer.Option(min=1)] = 4,
) -> None:
	"""Execute a schedule with one global HTTP budget and one combined response ledger."""
	import asyncio

	from distillation.collection import CollectionOptions
	from distillation.gateway_campaign import collect_gateway, digest

	try:
		manifest = json.loads((input_dir / "manifest.json").read_text())
		schedule = input_dir / "requests.jsonl"
		if digest(schedule) != manifest["sha256"][schedule.name]:
			raise ValueError("gateway schedule hash does not match its manifest")
		keys = json.loads(keys_file.read_text()) if keys_file else {"default": token}
		result = asyncio.run(
			collect_gateway(
				schedule=schedule,
				output=output,
				url=teacher_url,
				keys=keys,
				options=CollectionOptions(
					terms_review_id=terms_review_id, concurrency=concurrency, enable_thinking=False
				),
				attempt_budget=manifest["attempt_budget"],
				deadline_seconds=manifest["deadline_seconds"],
				decision_log=decision_log,
			)
		)
		from distillation.campaign_timing import campaign_timing

		save_json(output / "campaign-timing.json", campaign_timing(output, schedule=schedule))
		typer.echo(json.dumps(result, indent="\t"))
	except (ValueError, OSError, KeyError) as error:
		raise typer.BadParameter(str(error)) from None


@app.command("detector-input")
def detector_input_command(
	input_dir: Annotated[Path, typer.Option(help="Frozen gateway schedule directory.")],
	output: Annotated[Path, typer.Option(help="Output JSONL; must not already exist.")],
) -> None:
	"""Convert a gateway schedule into the domain-tagged input used by calibration and replay."""
	from harness.calibration import BenignRequest
	from harness.serialization import gateway_turn

	from distillation.gateway_campaign import digest

	try:
		manifest = json.loads((input_dir / "manifest.json").read_text())
		path = input_dir / "requests.jsonl"
		if digest(path) != manifest["sha256"][path.name]:
			raise ValueError("gateway schedule hash does not match its manifest")
		rows = read_jsonl(path, GatewayRequest)
		converted = [
			BenignRequest(
				account=row.account,
				text=gateway_turn(
					row.domain,
					next(m.content for m in reversed(row.prompt.prompt) if m.role == "user"),
				),
				time=row.time,
				example_id=row.prompt.example_id,
				source_group=f"{row.domain}:{row.prompt.source.group_id}",
				client_type=row.client_type,
				workload_version=manifest.get("version")
				if manifest.get("version") == "gateway-clients-v5"
				else None,
			)
			for row in rows
		]
		output.parent.mkdir(parents=True, exist_ok=True)
		with output.open("x") as target:
			for row in converted:
				target.write(
					row.model_dump_json(
						exclude={"workload_version"} if row.workload_version is None else set()
					)
					+ "\n"
				)
		typer.echo(f"requests={len(converted)} output={output}")
	except (ValueError, OSError, KeyError, StopIteration) as error:
		raise typer.BadParameter(str(error)) from None


@app.command("profile-build")
def profile_build_command(
	source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option()],
	workload_version: str = "gateway-clients-v5",
	horizons_per_profile: Annotated[int, typer.Option(min=1)] = 300,
	requests_per_account: Annotated[int, typer.Option(min=1)] = 100,
	seed: int = 42,
) -> None:
	"""Freeze source families and create five profile schedules and detector inputs."""
	from distillation.profiles import PROFILE_WORKLOAD_VERSION, SCORED_PROFILES, profile_clients
	from distillation.role_workflow import ROLES, SourceItem, build_roles
	from distillation.tasks import TASKS

	try:
		if workload_version != PROFILE_WORKLOAD_VERSION:
			raise ValueError(f"profile-build requires {PROFILE_WORKLOAD_VERSION}")
		output.mkdir(parents=True, exist_ok=False)
		build_roles(source, output / "roles", seed)
		for role in ROLES:
			items = read_jsonl(output / "roles" / f"{role}.jsonl", SourceItem)
			pools: dict[str, list[PromptRecord]] = {}
			for item in items:
				pools.setdefault(item.prompt.task, []).append(item.prompt)
			rows, manifest = profile_clients(
				pools=pools,
				caps={domain: TASKS[domain].max_output_tokens for domain in pools},
				role=role,
				accounts=horizons_per_profile * len(SCORED_PROFILES),
				references={item.prompt.example_id: item.reference for item in items},
				families={item.prompt.example_id: item.family for item in items},
				short_share=0,
				requests_per_account=requests_per_account,
				seed=seed,
				length_tail=None,
				max_requests_per_account=requests_per_account,
			)
			_write(output / role, rows, manifest)
			detector_input_command(output / role, output / f"{role}.detector.jsonl")
	except (ValueError, OSError, KeyError) as error:
		raise typer.BadParameter(str(error)) from None


@app.command("profile-validate")
def profile_validate_command(
	input_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
	output: Annotated[Path, typer.Option()],
	min_horizons: Annotated[int, typer.Option(min=1)] = 300,
) -> None:
	"""Check role provenance, leakage, topic mix, profile counts, and detector-input hashes."""
	from distillation.role_workflow import validate_roles

	try:
		report = validate_roles(input_dir, min_horizons=min_horizons)
		if output.exists():
			raise ValueError("validation report already exists")
		save_json(output, report)
		typer.echo(json.dumps(report, indent="\t"))
		if not report["passed"]:
			raise typer.Exit(code=1)
	except (ValueError, OSError, KeyError) as error:
		raise typer.BadParameter(str(error)) from None
