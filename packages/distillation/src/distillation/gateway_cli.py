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
from distillation.tasks import LEGITIMATE_ROLES, TASKS
from distillation.workloads import mixed_clients, task_references

app = typer.Typer(no_args_is_help=True)


def _pools(
	tasks_dir: Path, domains: str, role: str
) -> tuple[dict[str, list[PromptRecord]], dict[str, int], dict[str, Any]]:
	names = [name.strip() for name in domains.split(",")]
	if len(set(names)) != len(names) or len(names) < 2 or set(names) - TASKS.keys():
		raise ValueError("domains must list at least two distinct supported tasks")
	pools, caps, provenance = {}, {}, {}
	for domain in sorted(names):
		directory = tasks_dir / domain
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
	tasks_dir: Path = Path("data/tasks"),
	domains: str = "ag_news,squad,gsm8k,coding",
	generalist_share: Annotated[float, typer.Option(min=0, max=1)] = 0.1,
	requests_per_account: Annotated[int, typer.Option(min=1)] = 100,
	seed: int = 42,
) -> None:
	"""Generate mixed benign clients for one separated calibration or test role."""
	try:
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
			seed=seed,
			references={
				key: item
				for domain in pools
				for key, item in task_references(tasks_dir / domain).items()
			},
		)
		manifest["task_manifests"] = provenance
		_write(output, requests, manifest)
	except (ValueError, OSError, KeyError) as error:
		raise typer.BadParameter(str(error)) from None


@app.command("schedule")
def schedule_command(
	output: Annotated[Path, typer.Option()],
	attack: Annotated[str, typer.Option(help="A1 multiplexed, A2 domain-sequential, A5 Sybil.")],
	budget: Annotated[int, typer.Option(min=1)],
	tasks_dir: Path = Path("data/tasks"),
	domains: str = "ag_news,squad,gsm8k,coding",
	campaign_id: str = "campaign",
	accounts: Annotated[
		int | None, typer.Option(min=1, help="Total accounts; defaults to 6 for A5, 1 for A1/A2.")
	] = None,
	interval_seconds: Annotated[float, typer.Option(min=0)] = 1,
	deadline_seconds: Annotated[float, typer.Option(min=0)] = 86400,
	seed: int = 42,
) -> None:
	"""Generate a fixed campaign inventory with one budget shared by all its accounts."""
	from typing import cast

	from distillation.schedules import Attack

	try:
		pools, caps, provenance = _pools(tasks_dir, domains, "attacker")
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
		)
		manifest["task_manifests"] = provenance
		_write(output, requests, manifest)
	except (ValueError, OSError, KeyError) as error:
		raise typer.BadParameter(str(error)) from None
