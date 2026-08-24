"""Strict JSONL input and output helpers."""

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ValidationError


class JSONLValidationError(ValueError):
	"""Raised with a source line when a JSONL record is invalid."""


def read_jsonl[ModelT: BaseModel](path: Path, model: type[ModelT]) -> list[ModelT]:
	records: list[ModelT] = []
	with path.open(encoding="utf-8") as source:
		for line_number, line in enumerate(source, start=1):
			if not line.strip():
				continue
			try:
				records.append(model.model_validate_json(line))
			except (ValueError, ValidationError) as error:
				raise JSONLValidationError(f"{path}:{line_number}: invalid record") from error
	return records


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_suffix(f"{path.suffix}.tmp")
	with temporary.open("w", encoding="utf-8") as destination:
		for record in records:
			destination.write(record.model_dump_json(by_alias=True))
			destination.write("\n")
	temporary.replace(path)


def append_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as destination:
		for record in records:
			destination.write(record.model_dump_json(by_alias=True))
			destination.write("\n")
			destination.flush()


def canonical_json(value: object) -> str:
	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
