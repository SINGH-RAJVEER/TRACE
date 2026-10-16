"""Explicit recovery of damaged ledgers; ordinary collection remains strict."""

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from distillation.schemas import LedgerRecord


def _write_exclusive(path: Path, payload: bytes) -> None:
	with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as output:
		output.write(payload)
		output.flush()
		os.fsync(output.fileno())


def repair_ledger(path: Path, *, apply: bool = False) -> dict[str, Any]:
	"""Preserve valid rows and recover only NUL padding outside a complete JSON record.

	Every other invalid row is quarantined unchanged. Duplicate IDs are refused rather than
	choosing between responses. Applying a repair first saves the full original bytes, then
	replaces the ledger atomically. Run only while collection is stopped.
	"""
	original = path.read_bytes()
	kept: list[bytes] = []
	rejected: list[bytes] = []
	issues: list[dict[str, Any]] = []
	seen: set[str] = set()
	recovered = offset = quarantine_offset = 0
	for line_number, raw in enumerate(original.splitlines(keepends=True), 1):
		# Keep clean rows byte-for-byte, including CRLF. Strip only NUL bytes at the
		# boundaries, never from inside the JSON or its generated text.
		body = raw.rstrip(b"\r\n")
		ending = raw[len(body) :]
		candidate = body.strip(b"\x00")
		if not body.strip():
			kept.append(raw)
			offset += len(raw)
			continue
		try:
			record = LedgerRecord.model_validate_json(candidate)
		except ValidationError:
			rejected.append(raw)
			issues.append(
				{
					"line": line_number,
					"offset": offset,
					"bytes": len(raw),
					"quarantine_offset": quarantine_offset,
					"reason": "nul_bytes" if b"\x00" in body else "invalid_ledger_record",
				}
			)
			quarantine_offset += len(raw)
		else:
			if record.example_id in seen:
				raise ValueError(
					f"{path}:{line_number}: duplicate example_id {record.example_id}; "
					"resolve duplicates before repair; original ledger unchanged"
				)
			seen.add(record.example_id)
			recovered += candidate != body
			kept.append(candidate + ending)
		offset += len(raw)
	cleaned = b"".join(kept)
	# A complete last row without a newline would join the next appended JSON object.
	if cleaned and not cleaned.endswith(b"\n"):
		cleaned += b"\n"
	report: dict[str, Any] = {
		"input": str(path),
		"valid_records": len(seen),
		"recovered_nul_padding": recovered,
		"rejected_records": len(rejected),
		"rejected": issues,
		"original_sha256": hashlib.sha256(original).hexdigest(),
		"repaired_sha256": hashlib.sha256(cleaned).hexdigest(),
		"changed": cleaned != original,
		"applied": False,
	}
	if not apply or cleaned == original:
		return report
	if path.read_bytes() != original:
		raise ValueError("ledger changed during inspection; stop collection before repairing")
	backup = path.with_name(f"{path.name}.recovery-{uuid4().hex}.bak")
	quarantine = backup.with_suffix(".rejected.bin")
	report_path = backup.with_suffix(".report.json")
	_write_exclusive(backup, original)
	_write_exclusive(quarantine, b"".join(rejected))
	report |= {"backup": str(backup), "quarantine": str(quarantine), "report": str(report_path)}
	_write_exclusive(report_path, (json.dumps(report, indent="\t") + "\n").encode())
	with tempfile.NamedTemporaryFile(
		dir=path.parent, prefix=f".{path.name}.", delete=False
	) as output:
		temporary = Path(output.name)
		try:
			os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
			output.write(cleaned)
			output.flush()
			os.fsync(output.fileno())
			if path.read_bytes() != original:
				raise ValueError("ledger changed during recovery; original was not replaced")
			# Close first for Windows, which cannot replace an open temporary file.
			output.close()
			temporary.replace(path)
		finally:
			temporary.unlink(missing_ok=True)
	report["applied"] = True
	report_path.write_text(json.dumps(report, indent="\t") + "\n", encoding="utf-8")
	return report
