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
			"given interval), A5 Sybil per domain, A6 Sybil random, A7 independent sources, "
			"A8 online observable-feedback coverage."
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
	topic_sorted: Annotated[
		bool, typer.Option(help="Domain-blocked lexical topic reordering.")
	] = False,
	deduplicate: Annotated[bool, typer.Option(help="Remove duplicate request messages.")] = False,
	answer_only: Annotated[
		bool, typer.Option(help="Use answer-only requests for supported templates.")
	] = False,
	per_account_limit: Annotated[
		int | None,
		typer.Option(
			min=1, help="Rotate at this HTTP attempt limit, including proofs and failures."
		),
	] = None,
	round_size: Annotated[
		int, typer.Option(min=1, help="A8 queries selected per feedback round.")
	] = 16,
	paraphrase: Annotated[
		bool, typer.Option(help="Apply the deterministic offline text rewriter.")
	] = False,
	user_prefix: Annotated[
		str | None, typer.Option(help="Prepend this instruction to user messages.")
	] = None,
	assistant_prefill: Annotated[
		str | None, typer.Option(help="Append an assistant prefill when the endpoint supports it.")
	] = None,
	human_bursts: Annotated[
		bool, typer.Option(help="Use seeded sessions and diurnal idle gaps.")
	] = False,
	batch_size: Annotated[
		int,
		typer.Option(
			min=1,
			max=4,
			help="Items per JSON-array request for compatible templates; 1 disables batching.",
		),
	] = 1,
	replay_file: Annotated[
		Path | None,
		typer.Option(
			exists=True, dir_okay=False, help="Benign GatewayRequest JSONL to replay as cover."
		),
	] = None,
	replay_budget: Annotated[
		int, typer.Option(min=0, help="Maximum HTTP slots assigned to whole benign sessions.")
	] = 0,
	seed: int = 42,
) -> None:
	"""Freeze a campaign inventory or A8 candidates for online collection.

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
			topic_sorted=topic_sorted,
			deduplicate=deduplicate,
			answer_only=answer_only,
			per_account_limit=per_account_limit,
			round_size=round_size,
		)
		if paraphrase or user_prefix is not None or assistant_prefill is not None or human_bursts:
			from distillation.attack_transforms import offline_rewrite, transform_inventory

			if human_bursts and (attack == "A8" or per_account_limit):
				raise ValueError("human bursts require a fixed schedule without online rotation")
			requests, settings = transform_inventory(
				requests,
				rewriter=offline_rewrite if paraphrase else None,
				rewriter_id="offline-phrases-v1" if paraphrase else None,
				user_prefix=user_prefix,
				assistant_prefill=assistant_prefill,
				bursts=human_bursts,
				seed=seed,
			)
			manifest.setdefault("variants", {}).update(settings)
		if batch_size > 1:
			from distillation.attack_transforms import batch_inventory

			if attack == "A8" or per_account_limit or replay_file or assistant_prefill:
				raise ValueError(
					"batching requires fixed single-user-message schedules "
					"without replay or rotation"
				)
			requests, members = batch_inventory(requests, batch_size)
			manifest.setdefault("variants", {})["batching"] = {
				"size": batch_size,
				"format": "json-array-v1",
				"members": members,
				"offline_compute": "JSON parsing",
			}
			manifest["planned_queries"] = len(requests)
			manifest["candidate_count"] = len(requests)
		if bool(replay_file) != bool(replay_budget):
			raise ValueError("replay file and positive replay budget must be supplied together")
		if replay_file:
			from distillation.attack_transforms import replay_sessions

			if attack == "A8" or per_account_limit or human_bursts:
				raise ValueError(
					"session replay requires a fixed schedule without rotation or bursts"
				)
			requests, replay_ids = replay_sessions(
				requests, read_jsonl(replay_file, GatewayRequest), replay_budget
			)
			manifest.setdefault("variants", {})["session_replay"] = {
				"sha256": hashlib.sha256(replay_file.read_bytes()).hexdigest(),
				"request_ids": replay_ids,
				"budget": replay_budget,
				"offline_compute": "none",
			}
			manifest["planned_queries"] = len(requests)
			manifest["candidate_count"] = len(requests)
			manifest["domain_allocation"] = {
				d: sum(r.domain == d for r in requests)
				for d in sorted({r.domain for r in requests})
			}
			manifest["accounts"] = sorted({r.account for r in requests})
			manifest["account_domains"] = {
				a: sorted({r.domain for r in requests if r.account == a})
				for a in manifest["accounts"]
			}
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


def _served_policy(url: str) -> str | None:
	"""The policy a harness gateway reports, or None for a teacher that reports none."""
	import httpx

	try:
		response = httpx.get(f"{url.rstrip('/')}/healthz", timeout=10)
		policy = response.json().get("policy") if response.is_success else None
	except httpx.HTTPError, ValueError, AttributeError:
		return None
	return policy if isinstance(policy, str) else None


@app.command("collect")
def collect_command(
	input_dir: Annotated[Path, typer.Option(help="Immutable gateway schedule directory.")],
	output: Annotated[Path, typer.Option()],
	policy: Annotated[
		str,
		typer.Option(
			help="Policy the gateway serves for this arm: mmd, quota, rate, allow, or none "
			"for a direct teacher. mmd arms must pass the detectability gate."
		),
	],
	teacher_url: str = "http://127.0.0.1:8003",
	alternative_url: Annotated[
		str | None, typer.Option(help="Second teacher URL for direct-teacher mixing.")
	] = None,
	alternative_share: Annotated[
		float,
		typer.Option(min=0, max=1, help="Deterministic share assigned to the second teacher."),
	] = 0.0,
	confidence_scorer: Annotated[
		str | None,
		typer.Option(
			help="Offline module:function scoring a ledger record with confidence in [0, 1]."
		),
	] = None,
	confidence_threshold: Annotated[
		float | None,
		typer.Option(
			min=0, max=1, help="Retry or drop responses below this offline confidence score."
		),
	] = None,
	verifier_file: Annotated[
		Path | None,
		typer.Option(
			help="BenchmarkItem JSONL keyed by example ID for offline answer verification."
		),
	] = None,
	selective_retries: Annotated[
		int,
		typer.Option(
			min=0, help="Retry failed or rejected items, spending the shared HTTP budget."
		),
	] = 0,
	keys_file: Path | None = None,
	token: Annotated[str, typer.Option(envvar="TEACHER_API_TOKEN")] = "",
	terms_review_id: Annotated[str, typer.Option(envvar="TEACHER_TERMS_REVIEW_ID")] = "",
	decision_log: Path | None = None,
	concurrency: Annotated[int, typer.Option(min=1)] = 4,
	calibration: Annotated[
		Path | None,
		typer.Option(file_okay=False, help="The frozen calibration the gateway serves."),
	] = None,
	benign_test: Annotated[
		Path | None,
		typer.Option(dir_okay=False, help="Held-out benign detector input, never calibrated on."),
	] = None,
	gate_accumulator: Annotated[
		list[str] | None,
		typer.Option(help="MMD arm of the study to replay; repeat for each. Any one passes."),
	] = None,
	embedding_cache: Annotated[
		Path, typer.Option(file_okay=False, help="Gate embedding cache, outside frozen inputs.")
	] = Path("artifacts/embedding-cache"),
	embedding_archive: Annotated[
		list[Path] | None,
		typer.Option(
			exists=True,
			dir_okay=False,
			help="Read-only .npz of detector-text SHA-256 keys and vectors from the "
			"calibration's encoder; repeat for each.",
		),
	] = None,
	allow_undetected: Annotated[
		bool,
		typer.Option(help="Run an undetected mmd arm anyway, recorded as a quota comparison."),
	] = False,
) -> None:
	"""Execute a schedule or A8 online rounds with one global HTTP budget and response ledger.

	Before an mmd arm, the frozen schedule and the held-out benign accounts are replayed
	offline through the calibration. The campaign is refused unless some declared MMD arm
	escalates the attacker beyond those accounts; passing is not evidence of effectiveness.
	"""
	import asyncio

	from distillation.collection import CollectionOptions
	from distillation.detectability_gate import (
		DEFAULT_ACCUMULATORS,
		UndetectedError,
		detectability,
		gate,
	)
	from distillation.gateway_campaign import collect_gateway, digest

	try:
		if alternative_url and policy != "none":
			raise ValueError("alternative teacher mixing requires --policy none")
		if policy not in {"mmd", "quota", "rate", "allow", "none"}:
			raise ValueError("policy must be mmd, quota, rate, allow, or none")
		manifest = json.loads((input_dir / "manifest.json").read_text())
		schedule = input_dir / "requests.jsonl"
		if digest(schedule) != manifest["sha256"][schedule.name]:
			raise ValueError("gateway schedule hash does not match its manifest")
		served = _served_policy(teacher_url)
		if served is not None and served != policy:
			raise ValueError(f"the gateway serves {served}, not the declared {policy} policy")
		record = None
		if policy == "mmd":
			if calibration is None or benign_test is None:
				raise ValueError("an mmd arm needs --calibration and --benign-test for its gate")
			if embedding_cache.resolve().is_relative_to(output.resolve()):
				raise ValueError("the embedding cache must not live inside the run directory")
			try:
				record = gate(
					detectability(
						calibration_dir=calibration,
						schedule_dir=input_dir,
						benign_test=benign_test,
						accumulators=tuple(gate_accumulator or DEFAULT_ACCUMULATORS),
						embedding_cache=embedding_cache,
						embedding_archives=tuple(embedding_archive or ()),
					),
					allow_undetected=allow_undetected,
				)
			except UndetectedError as error:
				typer.echo(json.dumps(error.report["arms"], indent="\t"), err=True)
				typer.echo(str(error), err=True)
				raise typer.Exit(code=1) from None
			if record["reading"] == "quota comparison":
				typer.echo("detectability gate overridden: reporting a quota comparison", err=True)
		scorer = None
		if confidence_scorer:
			from importlib import import_module

			module, separator, name = confidence_scorer.partition(":")
			if not separator or not module or not name:
				raise ValueError("confidence scorer must use module:function syntax")
			scorer = getattr(import_module(module), name)
			if not callable(scorer):
				raise ValueError("confidence scorer must be callable")
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
				variants=manifest.get("variants"),
				selector_seed=manifest.get("seed", 42),
				detectability=record,
				alternative_url=alternative_url,
				alternative_share=alternative_share,
				verifier_file=verifier_file,
				selective_retries=selective_retries,
				confidence_scorer=scorer,
				confidence_scorer_id=confidence_scorer,
				confidence_threshold=confidence_threshold,
			)
		)
		from distillation.campaign_timing import campaign_timing

		variants = manifest.get("variants", {})
		online_timing = variants.get("online") or variants.get("per_account_limit")
		save_json(
			output / "campaign-timing.json",
			campaign_timing(output, schedule=None if online_timing else schedule),
		)
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


@app.command("teacher-preflight")
def teacher_preflight_command(
	input_dir: Annotated[Path, typer.Option(help="Frozen benign gateway schedule directory.")],
	output: Annotated[Path, typer.Option(help="New JSON evidence file.")],
	teacher_url: Annotated[str, typer.Option(help="Direct undefended teacher API URL.")],
	min_offered_rps: Annotated[
		float, typer.Option(help="Declared minimum offered requests per second.")
	],
	min_completed_rps: Annotated[
		float, typer.Option(help="Declared minimum completed requests per second.")
	],
	max_p95_seconds: Annotated[
		float, typer.Option(help="Maximum schedule-to-response p95 latency.")
	],
	max_error_fraction: Annotated[
		float, typer.Option(help="Maximum fraction of failed responses.")
	],
	max_timeout_fraction: Annotated[
		float, typer.Option(help="Maximum fraction of timed-out responses.")
	],
	max_queued_at_end: Annotated[
		int, typer.Option(help="Maximum queued requests at last arrival.")
	],
	max_queue_growth: Annotated[
		int, typer.Option(help="Maximum queue growth from midpoint to end.")
	],
	concurrency: Annotated[int, typer.Option(min=1)] = 4,
	timeout_seconds: Annotated[float, typer.Option(min=0.001)] = 120,
	token: Annotated[str, typer.Option(envvar="TEACHER_API_TOKEN")] = "",
) -> None:
	"""Measure a direct teacher under a frozen, open-loop benign workload."""
	import asyncio

	from distillation.gateway_campaign import digest
	from distillation.teacher_capacity import CapacitySLO, preflight

	try:
		if output.exists():
			raise ValueError("output exists; preserve it and choose a new evidence path")
		manifest = json.loads((input_dir / "manifest.json").read_text())
		schedule = input_dir / "requests.jsonl"
		sha = digest(schedule)
		if sha != manifest["sha256"][schedule.name]:
			raise ValueError("gateway schedule hash does not match its manifest")
		slo = CapacitySLO(
			min_offered_rps,
			min_completed_rps,
			max_p95_seconds,
			max_error_fraction,
			max_timeout_fraction,
			max_queued_at_end,
			max_queue_growth,
		)
		result = asyncio.run(
			preflight(
				read_jsonl(schedule, GatewayRequest),
				url=teacher_url,
				token=token,
				schedule_sha256=sha,
				slo=slo,
				concurrency=concurrency,
				timeout_seconds=timeout_seconds,
			)
		)
		save_json(output, result)
		typer.echo(json.dumps(result, indent="\t"))
		if not result["passed"]:
			raise typer.Exit(code=1)
	except (ValueError, OSError, KeyError) as error:
		raise typer.BadParameter(str(error)) from None


@app.command("require-teacher-capacity")
def require_teacher_capacity_command(
	evidence: Annotated[Path, typer.Option(help="Teacher preflight JSON evidence.")],
	input_dir: Annotated[Path, typer.Option(help="Frozen benign schedule directory.")],
) -> None:
	"""Fail unless capacity evidence passes for the specified benign schedule."""
	from distillation.gateway_campaign import digest
	from distillation.teacher_capacity import require_capacity_pass

	try:
		manifest = json.loads((input_dir / "manifest.json").read_text())
		schedule = input_dir / "requests.jsonl"
		sha = digest(schedule)
		if sha != manifest["sha256"][schedule.name]:
			raise ValueError("gateway schedule hash does not match its manifest")
		require_capacity_pass(json.loads(evidence.read_text()), schedule_sha256=sha)
		typer.echo("teacher capacity preflight passed")
	except (ValueError, OSError, KeyError) as error:
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


@app.command("session-benign")
def session_benign_command(
	output: Annotated[Path, typer.Option()],
	role: str = "score_calibration",
	tasks_dir: Path = Path("data"),
	domains: str = "ag_news,squad,gsm8k,coding",
	specialist_windows: Annotated[int, typer.Option(min=19)] = 20,
	generalist_horizons: Annotated[int, typer.Option(min=150)] = 150,
	window_size: Annotated[int, typer.Option(min=2)] = 100,
	horizon_size: Annotated[int, typer.Option(min=2)] = 100,
	seed: int = 42,
) -> None:
	"""Generate source-disjoint synthetic sessions for a fresh shadow calibration."""
	from distillation.session_workloads import session_clients

	try:
		pools, caps, provenance = _pools(tasks_dir, domains, role)
		requests, manifest = session_clients(
			pools=pools,
			caps=caps,
			role=role,
			specialist_windows=specialist_windows,
			generalist_horizons=generalist_horizons,
			window_size=window_size,
			horizon_size=horizon_size,
			seed=seed,
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


@app.command("external-corpus-index")
def external_corpus_index_command(
	output: Annotated[Path, typer.Option(dir_okay=False)],
	tasks_dir: Path = Path("data"),
	domains: str = "ag_news,squad,gsm8k,coding",
) -> None:
	"""Register the manifest-verified pinned external A3/A7 prompt pools offline."""
	from harness.corpus import CorpusIndex

	if output.exists():
		raise typer.BadParameter("corpus index output already exists")
	try:
		pools, _, provenance = _pools(tasks_dir, domains, "attacker", external=True)
		sources = {
			f"{domain}:{provenance[domain]['external']['sha256']}": [
				next(m.content for m in reversed(p.prompt) if m.role == "user") for p in pool
			]
			for domain, pool in pools.items()
		}
		index = CorpusIndex.from_sources(sources)
		output.parent.mkdir(parents=True, exist_ok=True)
		index.save(output)
		typer.echo(json.dumps({"sha256": index.digest, "sources": provenance}))
	except (ValueError, OSError, KeyError) as error:
		raise typer.BadParameter(str(error)) from None
