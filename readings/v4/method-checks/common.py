"""Shared loaders for the offline MMD audit. Reads committed readings only; never writes there."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from harness.serialization import detector_text, gateway_turn

ROOT = Path(__file__).resolve().parents[3]
READINGS = ROOT / "readings/v3"
CACHE = Path(os.environ.get("MMD_AUDIT_CACHE_DIR", ROOT / "outputs/mmd-audit/cache"))
ROLES = ("fitting", "score_calibration", "policy_calibration", "benign_test")
ATTACKS = ("A1", "A2", "A5", "A6", "A7")


def user_turn(row: dict) -> str:
	return next(m["content"] for m in reversed(row["prompt"]["prompt"]) if m["role"] == "user")


def load(path: Path) -> list[dict]:
	rows = []
	for line in path.open():
		row = json.loads(line)
		row["domain"] = row["prompt"]["task"]
		row["turn"] = user_turn(row)
		row["served"] = gateway_turn(row["domain"], row["turn"])
		row["detector"] = detector_text("gateway", row["served"])
		row["detector_noprefix"] = detector_text(row["domain"], row["turn"])
		rows.append(row)
	return rows


def benign(role: str) -> list[dict]:
	return load(READINGS / "benign" / role / "requests.jsonl")


def schedule(attack: str) -> list[dict]:
	return load(READINGS / "schedules" / attack / "s42-b6000" / "requests.jsonl")


def key(text: str) -> str:
	return hashlib.sha256(text.encode()).hexdigest()


class Embeddings:
	"""Embedding lookup by detector text, from cache/<variant>.npz."""

	def __init__(self, variant: str = "detector") -> None:
		data = np.load(CACHE / f"{variant}.npz")
		self.index = {k: i for i, k in enumerate(data["keys"])}
		self.matrix = data["vectors"]

	def __call__(self, texts: list[str]) -> np.ndarray:
		return self.matrix[[self.index[key(t)] for t in texts]]


class CachedEncoder:
	"""The harness Encoder protocol over the cached BGE vectors, keyed by detector text."""

	def __init__(self, variant: str = "detector") -> None:
		self.lookup = Embeddings(variant)
		self.truncated = 0

	def encode(self, texts: list[str]) -> np.ndarray:
		return self.lookup(texts)

	def identity(self) -> dict:
		return {"model": "BAAI/bge-small-en-v1.5", "cached": True}


def benign_requests(rows: list[dict]):
	"""The calibration's BenignRequest view of gateway rows (gateway_study.benign_requests)."""
	from harness.calibration import BenignRequest

	return [
		BenignRequest(
			account=r["account"],
			text=r["served"],
			time=r["time"],
			example_id=r["prompt"]["example_id"],
			source_group=f"{r['domain']}:{r['prompt']['source']['group_id']}",
			client_type=r.get("client_type"),
		)
		for r in rows
	]
