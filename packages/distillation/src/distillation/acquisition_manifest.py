"""Versioned manifest of cached teacher responses, with equal-budget row selection.

A manifest records what a set of ledgers contains, not how good it is: frozen teacher
settings, hashes of every input file, and one provenance row per cached response. Counts and
tokens come from the ledger as written; an unknown token count stays None and is never
estimated. Pure data and hashing, with no network, model, or GPU.
"""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from distillation.io import canonical_json
from distillation.schemas import LedgerRecord

MANIFEST_VERSION = "acquisition-manifest-v1"
_POLICY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHOWN = 5


class ManifestError(ValueError):
	"""Raised when ledgers or a manifest cannot be trusted."""


def _sha256_json(value: object) -> str:
	return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _where(row: dict[str, Any]) -> str:
	return f"{row['ledger']}:{row['line']}"


def _read_ledger(path: Path) -> tuple[str, list[tuple[int, LedgerRecord]]]:
	"""Hash the file bytes and parse rows from the same buffer, with physical line numbers."""
	data = path.read_bytes()
	records: list[tuple[int, LedgerRecord]] = []
	for number, line in enumerate(data.split(b"\n"), start=1):
		if not line.strip():
			continue
		try:
			records.append((number, LedgerRecord.model_validate_json(line)))
		except ValueError as error:
			raise ManifestError(f"{path}:{number}: invalid ledger record") from error
	return hashlib.sha256(data).hexdigest(), records


def _row(location: str, line: int, record: LedgerRecord, policy: str) -> dict[str, Any]:
	source = record.source
	return {
		"example_id": record.example_id,
		"task": record.task,
		"split": record.split,
		# Without a group_id the whole corpus is one family; the basis field says which.
		"source_family": source.group_id or source.corpus,
		"source_family_basis": "group_id" if source.group_id else "corpus",
		"corpus": source.corpus,
		"source_id": source.source_id,
		"prompt_sha256": _sha256_json([m.model_dump(mode="json") for m in record.prompt]),
		"completion_sha256": _sha256_json([m.model_dump(mode="json") for m in record.completion]),
		"response_policy": policy,
		"output_tokens": record.usage.output_tokens,
		"input_tokens": record.usage.input_tokens,
		"finish_reason": record.finish_reason,
		"quality_status": record.quality.status,
		"ledger": location,
		"line": line,
		# Same formula as learning_curve.candidates, so rows join across both artifacts.
		"record_sha256": hashlib.sha256(record.model_dump_json().encode()).hexdigest(),
	}


def _teacher_settings(record: LedgerRecord) -> dict[str, Any]:
	teacher = record.teacher
	return {
		"provider": teacher.provider,
		"model": teacher.model,
		"endpoint": teacher.endpoint,
		"parameters": dict(teacher.parameters),
	}


def _capped(items: list[str]) -> str:
	more = f" (and {len(items) - _SHOWN} more)" if len(items) > _SHOWN else ""
	return "; ".join(items[:_SHOWN]) + more


def _check_unique(rows: list[dict[str, Any]]) -> None:
	first_by_id: dict[str, dict[str, Any]] = {}
	duplicates: list[str] = []
	for row in rows:
		earlier = first_by_id.setdefault(row["example_id"], row)
		if earlier is not row:
			duplicates.append(f"{row['example_id']} at {_where(earlier)} and {_where(row)}")
	if duplicates:
		raise ManifestError(f"duplicate example IDs: {_capped(duplicates)}")
	first_by_prompt: dict[str, dict[str, Any]] = {}
	collisions: list[str] = []
	for row in rows:
		earlier = first_by_prompt.setdefault(row["prompt_sha256"], row)
		if earlier is not row:
			collisions.append(
				f"prompt {row['prompt_sha256'][:12]} is {earlier['example_id']} at "
				f"{_where(earlier)} and {row['example_id']} at {_where(row)}"
			)
	if collisions:
		raise ManifestError(f"same prompt under different example IDs: {_capped(collisions)}")


def _freeze_settings(rows: list[dict[str, Any]], settings: list[dict[str, Any]]) -> dict[str, Any]:
	first_by_settings: dict[str, int] = {}
	for index, teacher in enumerate(settings):
		first_by_settings.setdefault(canonical_json(teacher), index)
	if len(first_by_settings) > 1:
		shown = [
			f"{canonical_json(settings[index])} first at {_where(rows[index])}"
			for index in first_by_settings.values()
		]
		raise ManifestError(
			"teacher settings are not frozen; build one manifest per setting: " + _capped(shown)
		)
	return settings[0]


def build_manifest(
	ledgers: Sequence[Path], *, response_policy: str, accepted_only: bool = False
) -> dict[str, Any]:
	"""Read LedgerRecord JSONL files into a manifest, refusing anything ambiguous.

	Validation covers every row read. With accepted_only, rows whose quality status is not
	accepted are dropped after validation and the counts are recorded under "filter".
	Ledger paths are stored resolved, so moving a ledger leaves it unverified on load.
	"""
	if not _POLICY.fullmatch(response_policy):
		raise ManifestError(
			f"response policy must be an identifier of letters, digits, '.', '_' or '-': "
			f"{response_policy!r}"
		)
	if not ledgers:
		raise ManifestError("at least one ledger is required")
	locations = [str(path.resolve()) for path in ledgers]
	if len(set(locations)) != len(locations):
		raise ManifestError("the same ledger was given more than once")

	inputs: list[dict[str, Any]] = []
	rows: list[dict[str, Any]] = []
	settings: list[dict[str, Any]] = []
	for path, location in zip(ledgers, locations, strict=True):
		file_sha256, records = _read_ledger(path)
		inputs.append({"path": location, "sha256": file_sha256, "rows": len(records)})
		for line, record in records:
			rows.append(_row(location, line, record, response_policy))
			settings.append(_teacher_settings(record))
	if not rows:
		raise ManifestError("the ledgers contain no records")

	_check_unique(rows)
	teacher = _freeze_settings(rows, settings)

	rows_read = len(rows)
	if accepted_only:
		rows = [row for row in rows if row["quality_status"] == "accepted"]
		if not rows:
			raise ManifestError("no accepted rows remain after filtering")
	manifest: dict[str, Any] = {
		"version": MANIFEST_VERSION,
		"response_policy": response_policy,
		"teacher": teacher,
		"inputs": inputs,
		"filter": {
			"accepted_only": accepted_only,
			"rows_read": rows_read,
			"rows_dropped": rows_read - len(rows),
		},
		"examples": rows,
		"totals": _totals(rows),
	}
	manifest["digest"] = _sha256_json(manifest)
	return manifest


def _totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
	known = [row["output_tokens"] for row in rows if row["output_tokens"] is not None]
	return {
		"examples": len(rows),
		"accepted_examples": sum(row["quality_status"] == "accepted" for row in rows),
		"known_output_tokens": sum(known),
		"unknown_token_rows": len(rows) - len(known),
		"examples_per_task": dict(sorted(Counter(row["task"] for row in rows).items())),
		"examples_per_source_family": dict(
			sorted(Counter(row["source_family"] for row in rows).items())
		),
	}


def write_manifest(manifest: dict[str, Any], output: Path) -> None:
	"""Write a new manifest file; a frozen artifact is never overwritten."""
	if output.exists():
		raise ManifestError(f"{output} already exists; manifests are not overwritten")
	output.parent.mkdir(parents=True, exist_ok=True)
	temporary = output.with_suffix(f"{output.suffix}.tmp")
	temporary.write_text(json.dumps(manifest, indent="\t", ensure_ascii=False) + "\n")
	temporary.replace(output)


def load_manifest(path: Path, *, require_ledgers: bool = False) -> dict[str, Any]:
	"""Load a manifest and re-verify its digest and the hashes of any available ledgers.

	A ledger that is not on disk is skipped unless require_ledgers is set. A ledger that is on
	disk with different bytes is always an error.
	"""
	manifest = json.loads(path.read_text())
	if not isinstance(manifest, dict) or manifest.get("version") != MANIFEST_VERSION:
		raise ManifestError(f"{path}: not an {MANIFEST_VERSION} manifest")
	claimed = manifest.get("digest")
	body = {key: value for key, value in manifest.items() if key != "digest"}
	if claimed != _sha256_json(body):
		raise ManifestError(f"{path}: manifest digest does not match its contents")
	for entry in manifest["inputs"]:
		ledger = Path(entry["path"])
		if not ledger.is_file():
			if require_ledgers:
				raise ManifestError(f"ledger {ledger} is not available")
			continue
		if hashlib.sha256(ledger.read_bytes()).hexdigest() != entry["sha256"]:
			raise ManifestError(f"ledger {ledger} has changed since the manifest was built")
	return manifest


def select_rows(
	manifest: dict[str, Any],
	*,
	by: Literal["queries", "tokens"],
	budget: int,
	seed: int,
	include_unaccepted: bool = False,
) -> dict[str, Any]:
	"""Pick a seeded prefix of the manifest rows for an equal-query or equal-token budget.

	Rows are ordered by sha256(seed:example_id), so every budget at one seed is a prefix of
	the next larger one. Candidates are accepted rows unless include_unaccepted is set; their
	number is returned. A query budget needs exactly that many candidates. A token budget
	stops at the first row that would exceed it, so it is never exceeded, and is refused when
	any candidate has an unknown token count.
	"""
	if by not in ("queries", "tokens"):
		raise ManifestError("select by 'queries' or 'tokens'")
	if budget < 1:
		raise ManifestError("the budget must be at least 1")
	rows = manifest["examples"]
	candidates = [row for row in rows if include_unaccepted or row["quality_status"] == "accepted"]
	ordered = sorted(
		candidates,
		key=lambda row: (
			hashlib.sha256(f"{seed}:{row['example_id']}".encode()).hexdigest(),
			row["example_id"],
		),
	)
	exhausted = False
	if by == "queries":
		if budget > len(ordered):
			raise ManifestError(f"query budget {budget} exceeds the {len(ordered)} candidates")
		chosen = ordered[:budget]
	else:
		unknown = [row["example_id"] for row in ordered if row["output_tokens"] is None]
		if unknown:
			raise ManifestError(
				f"token budget refused: {len(unknown)} of {len(ordered)} candidates have unknown "
				f"output tokens ({', '.join(unknown[:_SHOWN])})"
			)
		chosen = []
		spent = 0
		for row in ordered:
			if spent + row["output_tokens"] > budget:
				break
			chosen.append(row)
			spent += row["output_tokens"]
		exhausted = len(chosen) == len(ordered) and spent < budget
	known = [row["output_tokens"] for row in chosen if row["output_tokens"] is not None]
	return {
		"by": by,
		"budget": budget,
		"seed": seed,
		"include_unaccepted": include_unaccepted,
		"manifest_digest": manifest["digest"],
		"candidates": len(ordered),
		"excluded_unaccepted": len(rows) - len(candidates),
		"count": len(chosen),
		"output_tokens": sum(known) if len(known) == len(chosen) else None,
		"unknown_token_rows": len(chosen) - len(known),
		"exhausted": exhausted,
		"example_ids": [row["example_id"] for row in chosen],
	}
