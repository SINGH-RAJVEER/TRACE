"""Helpers for scripts/start.sh. Run inside the workspace environment.

corpus-ready TASK_DIR                    exit 0 when a task build is complete and partitioned
caps MANIFEST                            print a benign schedule's per-domain output caps
accounts DIR REQUESTS...                 write DIR/credentials.json (mode 600) and accounts.json
utility SCHEDULE TASKS KEYS OUTPUT URL N [PREFLIGHT OFFERED_RPS MIN_COMPLETION]
                                         measure benign utility, optionally bound to a passing
                                         teacher preflight of the same schedule
summary CAMPAIGN_JSON                    print a one-line campaign summary
export RUN READINGS [NAME]               copy the run's evidence into READINGS/vN, minus secrets,
                                         and replace its rows in READINGS/index.tsv
backfill READINGS                        replace the frozen v3 and v4 rows in READINGS/index.tsv
"""

import asyncio
import fnmatch
import hashlib
import json
import os
import re
import secrets
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

# Never exported: client keys, enforcement state, the deception key, run bookkeeping, large
# service logs, and detector inputs that `distill gateway detector-input` regenerates.
EXCLUDED = (
	"credentials.json",
	"*/credentials.json",
	"*.sqlite3",
	"*.sqlite3-*",
	"deception.key",
	".lock",
	"reading",
	"steps/*",
	"logs/vllm.log",
	"logs/teacher-api.log",
	"benign/*.detector.jsonl",
)


def corpus_ready(directory: Path) -> int:
	try:
		manifest = json.loads((directory / "manifest.json").read_text())
	except OSError, ValueError:
		return 1
	partitions = manifest.get("attacker_partitions")
	if manifest.get("complete") is False or not partitions:
		return 1
	files = [directory / name for name in partitions["files"].values()]
	return 0 if all(path.is_file() for path in files) else 1


def accounts(directory: Path, schedules: list[Path]) -> None:
	names = {"warmup"}
	for schedule in schedules:
		with schedule.open() as stream:
			names.update(json.loads(line)["account"] for line in stream if line.strip())
	directory.mkdir(parents=True, exist_ok=True)
	credentials = directory / "credentials.json"
	if not credentials.exists():
		with open(credentials, "x", opener=lambda p, f: os.open(p, f, 0o600)) as stream:
			json.dump({name: secrets.token_hex(32) for name in sorted(names)}, stream)
	keys = json.loads(credentials.read_text())
	missing = names - keys.keys()
	if missing:
		raise SystemExit(f"{credentials} lacks {len(missing)} scheduled accounts")
	digests = {hashlib.sha256(key.encode()).hexdigest(): name for name, key in keys.items()}
	(directory / "accounts.json").write_text(json.dumps(digests, indent="\t"))


def utility(
	schedule: Path,
	tasks: Path,
	keys_file: Path,
	output: Path,
	url: str,
	concurrency: int,
	preflight: Path | None = None,
	offered_rps: float | None = None,
	min_completion_rate: float | None = None,
) -> None:
	from distillation.io import read_jsonl
	from distillation.schemas import GatewayRequest
	from distillation.tasks import task_directory
	from distillation.teacher_capacity import bind_utility_baseline
	from distillation.utility import measure_utility
	from distillation.workloads import task_references
	from harness.calibration import BenignRequest

	# Matches the default request deadline of measure_utility.
	baseline = None
	if any(value is not None for value in (preflight, offered_rps, min_completion_rate)):
		if preflight is None or offered_rps is None or min_completion_rate is None:
			raise ValueError("utility binding needs preflight, offered rate and minimum completion")
		baseline = bind_utility_baseline(
			preflight,
			schedule_sha256=_sha256(schedule / "requests.jsonl"),
			declared_offered_rps=offered_rps,
			min_completion_rate=min_completion_rate,
			concurrency=concurrency,
			request_deadline=120,
		)
	manifest = json.loads((schedule / "manifest.json").read_text())
	rows = read_jsonl(schedule / "requests.jsonl", GatewayRequest)
	caps = manifest["max_output_tokens"]
	references = {
		key: value
		for domain in caps
		for key, value in task_references(task_directory(domain, root=tasks)).items()
	}
	# The raw user turn, as the v3 utility pass sent it.
	requests = [
		BenignRequest(
			account=row.account,
			text=next(m.content for m in reversed(row.prompt.prompt) if m.role == "user"),
			time=row.time,
			example_id=row.prompt.example_id,
			source_group=f"{row.domain}:{row.prompt.source.group_id}",
			client_type=row.client_type,
		)
		for row in rows
	]
	result = asyncio.run(
		measure_utility(
			requests=requests,
			references=references,
			url=url,
			keys=json.loads(keys_file.read_text()),
			max_output_tokens=caps,
			account_profiles={row.account: row.domain_profile or "unknown" for row in rows},
			output=output,
			baseline=baseline,
			concurrency=concurrency,
		)
	)
	scalars = {k: v for k, v in result.items() if not isinstance(v, (list, dict))}
	print(json.dumps(scalars, indent="\t"))


def summary(path: Path) -> None:
	from distillation.campaign_summary import load_campaign_summary

	campaign = json.loads(path.read_text())
	canonical = load_campaign_summary(path.parent) or {}
	fields = ("received", "blocked", "challenge", "failed", "deadline", "released_tokens")
	parts = [f"{key} {canonical.get(key)}" for key in fields]
	hours = campaign.get("elapsed_seconds", 0) / 3600
	print(", ".join(parts) + f", {hours:.2f} h")


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for chunk in iter(lambda: stream.read(1 << 20), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _reading_name(run: Path, readings: Path, requested: str) -> str:
	"""Reuse this run's reading, else the requested name, else the next free vN."""
	chosen = run / "reading"
	if chosen.exists():
		return chosen.read_text().strip()
	if requested:
		name = requested
	else:
		numbers = [
			int(match.group(1))
			for entry in readings.iterdir()
			if (match := re.fullmatch(r"v(\d+)(?:-.*)?", entry.name))
		]
		name = f"v{max(numbers, default=-1) + 1}"
	chosen.write_text(name + "\n")
	return name


def export(run: Path, readings: Path, requested: str) -> None:
	run_id = run.name
	name = _reading_name(run, readings, requested)
	destination = readings / name
	record = destination / "export.json"
	if destination.exists():
		previous = json.loads(record.read_text()) if record.exists() else {}
		if previous.get("run_id") != run_id:
			raise SystemExit(f"{destination} exists and does not belong to run {run_id}")
		shutil.rmtree(destination)
	files = sorted(
		path
		for path in run.rglob("*")
		if path.is_file()
		and not any(
			fnmatch.fnmatch(path.relative_to(run).as_posix(), pattern) for pattern in EXCLUDED
		)
	)
	checksums = []
	for path in files:
		relative = path.relative_to(run)
		target = destination / relative
		target.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(path, target)
		checksums.append(f"{_sha256(target)}  {relative.as_posix()}")
	(destination / "SHA256SUMS").write_text("\n".join(checksums) + "\n")

	exit_line = (run / "exit").read_text().split(maxsplit=2) if (run / "exit").exists() else []
	code = exit_line[0] if exit_line else "running"
	status = "complete" if code == "0" else f"partial (exit {code})"
	steps = (run / "steps.tsv").read_text().splitlines() if (run / "steps.tsv").exists() else []
	attempts = sorted((run / "provenance").glob("*/code.txt"))
	commit = (
		next(
			(
				line.split()[1]
				for line in attempts[-1].read_text().splitlines()
				if line.startswith("commit")
			),
			"unknown",
		)
		if attempts
		else "unknown"
	)
	config = attempts[-1].with_name("config.env").read_text() if attempts else ""
	from distillation.results_index import run_rows, update_index

	indexed = update_index(readings / "index.tsv", run_rows(run, name, commit))
	summary_table = (run / "summary.tsv").read_text() if (run / "summary.tsv").exists() else ""
	record.write_text(
		json.dumps(
			{
				"run_id": run_id,
				"status": status,
				"exit": " ".join(exit_line),
				"commit": commit,
				"exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
				"files": len(files),
				"excluded": list(EXCLUDED),
			},
			indent="\t",
		)
		+ "\n"
	)
	readme = destination / "README.md"
	rows = [line.split("\t") for line in summary_table.splitlines()]
	table = ""
	if len(rows) > 1:
		table = "| " + " | ".join(rows[0]) + " |\n|" + "---|" * len(rows[0]) + "\n"
		table += "".join("| " + " | ".join(row) + " |\n" for row in rows[1:])
	first = steps[0].split("\t")[0] if steps else "unknown"
	last = steps[-1].split("\t")[0] if steps else "unknown"
	readme.write_text(
		f"# {name}: run {run_id}\n\n"
		f"**Status:** {status}. **Code:** `{commit}`. **Steps:** {first} to {last} UTC.\n\n"
		f"Exported by `scripts/start.sh` from `{run}`. Write the results, "
		"limitations and stop cause here before committing.\n\n"
		"## Settings\n\n```\n" + config + "```\n\n"
		"## Campaign outcomes\n\n" + (table or "No campaign reached an outcome.\n") + "\n"
		"## Contents\n\n"
		"- `provenance/`: code revision, diff, configuration, redacted environment, packages, "
		"hardware and corpus manifests, one folder per attempt.\n"
		"- `run.log`, `steps.tsv`, `logs/`: the run timeline, step timings and per-step output.\n"
		"- `benign/`, `calibration/`, `adequacy.json`, `schedules/`: workloads, calibration and "
		"frozen attack schedules.\n"
		"- `campaigns/ATTACK/ARM/`: decisions, harness log and settings, outcome, evaluation "
		"report, and `campaign/` events, attempts and ledger.\n"
		"- `utility/ARM/`: benign utility under each arm.\n"
		"- `SHA256SUMS`, `export.json`: checksums and export record.\n\n"
		"Not exported: client credentials, harness state databases, the deception key, vLLM "
		"and teacher-API logs, and regenerable detector inputs.\n"
	)
	print(
		f"{destination} ({status}, {len(files)} files); readings/index.tsv has {len(indexed)} rows"
	)


def main() -> None:
	command, *args = sys.argv[1:] or [""]
	if command == "corpus-ready":
		raise SystemExit(corpus_ready(Path(args[0])))
	if command == "caps":
		print(json.dumps(json.loads(Path(args[0]).read_text())["max_output_tokens"]))
	elif command == "accounts":
		accounts(Path(args[0]), [Path(arg) for arg in args[1:]])
	elif command == "utility":
		if len(args) not in (6, 9):
			raise SystemExit(
				"utility needs six arguments, optionally followed by three binding arguments"
			)
		schedule, tasks, keys, output = (Path(arg) for arg in args[:4])
		utility(
			schedule,
			tasks,
			keys,
			output,
			args[4],
			int(args[5]),
			Path(args[6]) if len(args) == 9 else None,
			float(args[7]) if len(args) == 9 else None,
			float(args[8]) if len(args) == 9 else None,
		)
	elif command == "summary":
		summary(Path(args[0]))
	elif command == "backfill":
		from distillation.results_index import backfill_rows, update_index

		rows = update_index(Path(args[0]) / "index.tsv", backfill_rows(Path(args[0])))
		print(f"{args[0]}/index.tsv has {len(rows)} rows")
	elif command == "export":
		export(Path(args[0]), Path(args[1]), args[2] if len(args) > 2 else "")
	else:
		raise SystemExit(__doc__)


# The proof solver's process pool re-imports this module.
if __name__ == "__main__":
	main()
