"""Reference vectors that pin the keyed green-list algorithm to fixed values."""

import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from trace_watermark.core import (
	ALGORITHM_ID,
	ALGORITHM_VERSION,
	WatermarkKey,
	WatermarkProfile,
	context_seed,
	green_mask,
)

MASK_DIGEST_METHOD = "sha256(numpy.packbits(mask, bitorder='big'))"


def load_fixture(path: Path) -> dict[str, Any]:
	fixture = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(fixture, dict):
		raise ValueError("fixture must be a JSON object")
	return fixture


def _case_vectors(key: WatermarkKey, case: Mapping[str, Any]) -> dict[str, Any]:
	profile = WatermarkProfile(
		gamma=case["gamma"],
		delta=case["delta"],
		context_width=case["context_width"],
		vocab_size=case["vocab_size"],
	)
	vectors = []
	for context in case["contexts"]:
		if not all(0 <= token < profile.vocab_size for token in context):
			raise ValueError(f"context {context} falls outside the vocabulary")
		seed = context_seed(key, profile, context)
		mask = green_mask(seed, profile)
		vectors.append(
			{
				"context": list(context),
				"seed": f"{seed:016x}",
				"mask_sha256": hashlib.sha256(np.packbits(mask, bitorder="big")).hexdigest(),
				"mask_first_32": "".join("1" if bit else "0" for bit in mask[:32]),
			}
		)
	return {
		"case_id": case["case_id"],
		"gamma": profile.gamma,
		"delta": profile.delta,
		"context_width": profile.context_width,
		"vocab_size": profile.vocab_size,
		"vectors": vectors,
	}


def compute_vectors(fixture: Mapping[str, Any]) -> dict[str, Any]:
	"""Compute vectors for a fixture labelled nonsecret. The key never reaches the output."""
	if fixture.get("nonsecret") is not True:
		raise ValueError("fixture must declare nonsecret: true; vectors expose derived seeds")
	try:
		key = WatermarkKey(fixture["key_id"], fixture["epoch"], bytes.fromhex(fixture["key_hex"]))
		cases = [_case_vectors(key, case) for case in fixture["cases"]]
	except KeyError as error:
		raise ValueError(f"fixture is missing field {error}") from error
	except (TypeError, struct.error) as error:
		raise ValueError(f"fixture is malformed: {error}") from error
	return {
		"algorithm_id": ALGORITHM_ID,
		"algorithm_version": ALGORITHM_VERSION,
		"mask_digest_method": MASK_DIGEST_METHOD,
		"key_id": key.key_id,
		"epoch": key.epoch,
		"cases": cases,
	}


def find_mismatches(expected: Any, computed: Any, path: str = "$") -> list[str]:
	"""Describe every difference between expected and computed vectors."""
	if isinstance(expected, dict) and isinstance(computed, dict):
		return [
			mismatch
			for name in sorted(expected.keys() | computed.keys())
			for mismatch in find_mismatches(
				expected.get(name), computed.get(name), f"{path}.{name}"
			)
		]
	if isinstance(expected, list) and isinstance(computed, list):
		if len(expected) != len(computed):
			return [f"{path}: expected {len(expected)} items, computed {len(computed)}"]
		return [
			mismatch
			for index, (left, right) in enumerate(zip(expected, computed, strict=True))
			for mismatch in find_mismatches(left, right, f"{path}[{index}]")
		]
	return [] if expected == computed else [f"{path}: expected {expected!r}, computed {computed!r}"]
