"""Typed rows of `readings/index.tsv`, read through the index's own reader.

`results_index.read_index` keeps every cell as text and applies the index's reading rules, such
as unknown validity for unbound utility rows. Here each cell becomes null, a boolean or a
number by column, so an empty cell stays unknown and is never read as zero.
"""

import csv
from pathlib import Path
from typing import Any

from distillation.results_index import COLUMNS, read_index

from dashboard_api.files import LFS_POINTER, sniff

BOOLEAN = frozenset({"attribution_complete", "student_target_reached", "gate_passed"})
INTEGER = frozenset(
	{
		"seed",
		"attempt_budget",
		"attempts",
		"received",
		"blocked",
		"challenge",
		"failed",
		"deadline",
		"pending",
		"released_tokens",
		"released_tokens_unknown",
		"distinct_examples",
		"unattributed_queries",
		"submitted",
		"served",
		"student_n",
		"student_seeds",
		*(f"L{level}_{field}" for level in (1, 2, 3, 4) for field in ("J", "Q")),
	}
)
FLOAT = frozenset(
	{
		"deadline_seconds",
		"first_alert_seconds",
		"first_restriction_seconds",
		"first_refusal_seconds",
		*(f"L{level}_seconds" for level in (1, 2, 3, 4)),
		"completion_rate",
		"served_only_quality",
		"unconditional_utility",
		"latency_p95",
		"student_gain_pp",
		"student_gain_lower",
		"student_gain_upper",
		"min_completion_rate",
	}
)


class CellError(ValueError):
	pass


def cell(column: str, text: str) -> Any:
	"""One stored cell as a typed value; an empty cell is None."""
	if text == "":
		return None
	if column in BOOLEAN:
		if text not in ("True", "False"):
			raise CellError(f"{column} holds {text!r}, not True or False")
		return text == "True"
	try:
		if column in INTEGER:
			number = float(text)
			return int(number) if number.is_integer() else number
		if column in FLOAT:
			return float(text)
	except ValueError as error:
		raise CellError(f"{column} holds {text!r}, not a number") from error
	return text


def load_index(path: Path) -> dict[str, Any]:
	"""All rows with typed cells, or the reason the index cannot be read."""
	label = path.name
	if not path.is_file():
		return {"state": "missing", "path": label, "error": "no results index", "rows": []}
	if sniff(path, len(LFS_POINTER)).startswith(LFS_POINTER):
		return {"state": "lfs-pointer", "path": label, "error": "Git LFS pointer", "rows": []}
	try:
		stored = read_index(path)
		rows = [{column: cell(column, row[column]) for column in COLUMNS} for row in stored]
	except (ValueError, KeyError, csv.Error) as error:
		return {"state": "malformed", "path": label, "error": str(error), "rows": []}
	return {"state": "ok", "path": label, "error": None, "rows": rows}
