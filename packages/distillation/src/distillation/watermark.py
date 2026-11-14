"""Keyed unigram watermark: a fixed green list per key, applied through upstream logit bias.

The teacher adds `delta` to the logits of a secret `gamma` share of the eligible vocabulary,
chosen per key. Because the green list does not depend on context, vLLM can apply it through
`logit_bias` without a custom sampler. vLLM is the only supported teacher inference runtime.
Only token IDs below `eligible_vocab` are biased and counted; low IDs are the most
common subwords in BPE vocabularies, which keeps the per-request bias map small.

Detection counts how many of a pool's distinct eligible tokens fall in a key's green list
and computes a one-sided z-score against the `gamma` share expected without the watermark.
Under a key that did not mark the text, each distinct token is green independently with
probability `gamma`, which is what the binomial z-score assumes; the normal tail is still
only approximate for small pools.
Testing several candidate keys is corrected with Bonferroni. Texts with too few eligible
tokens abstain. A fixed green list is the easiest watermark to learn, which is the point for
attribution: a student trained on enough marked output may inherit the bias. Whether it does
for this teacher and student is unmeasured. Biasing also changes greedy answers, so the
utility cost must be measured before use. Nothing here proves training or intent.

Keys are derived as HMAC-SHA256(master secret, key ID). The master secret stays with the
teacher and the detector; key IDs are public identifiers chosen by the gateway.
"""

import hashlib
import hmac
import json
import math
import os
import secrets
import tempfile
from collections.abc import Callable, Sequence
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

VERSION: Final = "unigram-v1"


class WatermarkSpec(BaseModel):
	model_config = ConfigDict(extra="forbid", frozen=True)
	version: Literal["unigram-v1"] = VERSION
	gamma: float = Field(default=0.25, gt=0, lt=1)
	delta: float = Field(default=2.0, gt=0, le=20)
	eligible_vocab: int = Field(default=32768, ge=256, le=1_048_576)

	@property
	def digest(self) -> str:
		return hashlib.sha256(
			json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
		).hexdigest()


def read_master_secret(path: Path) -> bytes:
	"""A hex secret of at least 16 bytes; never logged or written to manifests."""
	try:
		secret = bytes.fromhex(path.read_text().strip())
	except (OSError, ValueError) as error:
		raise ValueError("the watermark key file must hold a hex secret") from error
	if len(secret) < 16:
		raise ValueError("the watermark secret must be at least 16 bytes")
	return secret


def load_or_create_master_secret(path: Path) -> bytes:
	"""Persist one private random key, publishing it atomically without replacing a key.

	A complete mode-0600 temporary file is hard-linked into place so concurrent starters
	cannot read a partial write or overwrite the winner's key. Existing invalid keys fail
	validation; only a missing key is generated. Detection remains strictly read-only.
	"""
	if path.exists():
		return read_master_secret(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".watermark-", suffix=".watermark.key")
	try:
		with os.fdopen(fd, "w") as stream:
			stream.write(secrets.token_hex(32) + "\n")
			stream.flush()
			os.fsync(stream.fileno())
		# Another starter may have published its complete key first.
		with suppress(FileExistsError):
			os.link(temporary, path)
	finally:
		os.unlink(temporary)
	return read_master_secret(path)


def key_id(namespace: str, principal: str) -> str:
	"""The public key identifier the gateway sends for a principal."""
	return hashlib.sha256(json.dumps([namespace, principal]).encode()).hexdigest()[:32]


@lru_cache(maxsize=256)
def green_ids(secret: bytes, key: str, spec: WatermarkSpec) -> frozenset[int]:
	"""The key's green token IDs, stable across Python versions and processes."""
	derived = hmac.new(secret, key.encode(), hashlib.sha256).digest()
	threshold = int(spec.gamma * 2**64)
	green = set()
	for token in range(spec.eligible_vocab):
		value = hmac.new(derived, token.to_bytes(4, "big"), hashlib.sha256).digest()
		if int.from_bytes(value[:8], "big") < threshold:
			green.add(token)
	return frozenset(green)


def logit_bias(secret: bytes, key: str, spec: WatermarkSpec) -> dict[str, float]:
	return {str(token): spec.delta for token in sorted(green_ids(secret, key, spec))}


def score(
	token_ids: Sequence[Sequence[int]], green: frozenset[int], spec: WatermarkSpec
) -> tuple[int, int]:
	"""Green and total counts over the pool's distinct eligible tokens.

	Each token counts once across the whole pool. Counting it once per text would let common
	tokens recur across texts, so a random key's count would vary far more than the binomial
	the z-score assumes, and unmarked pools would test positive.
	"""
	distinct = {t for text in token_ids for t in text if 0 <= t < spec.eligible_vocab}
	return len(distinct & green), len(distinct)


def detect(
	token_ids: Sequence[Sequence[int]],
	secret: bytes,
	candidates: Sequence[str],
	spec: WatermarkSpec,
	*,
	alpha: float = 1e-3,
	minimum_tokens: int = 200,
) -> dict[str, Any]:
	"""Test each candidate key on the pooled texts. Abstains below `minimum_tokens`."""
	if not candidates or len(set(candidates)) != len(candidates):
		raise ValueError("give distinct candidate key IDs")
	if not 0 < alpha < 1:
		raise ValueError("alpha must lie strictly between zero and one")
	results = []
	for candidate in candidates:
		hits, total = score(token_ids, green_ids(secret, candidate, spec), spec)
		if total < minimum_tokens:
			results.append(
				{"key_id": candidate, "tokens": total, "decision": "insufficient_evidence"}
			)
			continue
		expected = spec.gamma * total
		z = (hits - expected) / math.sqrt(total * spec.gamma * (1 - spec.gamma))
		p = 0.5 * math.erfc(z / math.sqrt(2))
		adjusted = min(1.0, p * len(candidates))
		results.append(
			{
				"key_id": candidate,
				"tokens": total,
				"green": hits,
				"z": z,
				"p_value": p,
				"adjusted_p_value": adjusted,
				"decision": "detected" if adjusted <= alpha else "not_detected",
			}
		)
	return {
		"version": VERSION,
		"spec_sha256": spec.digest,
		"alpha": alpha,
		"correction": "bonferroni",
		"minimum_tokens": minimum_tokens,
		"texts": len(token_ids),
		"results": results,
		"limitations": [
			"Detection shows a statistical bias toward a key's tokens, not training or intent.",
			"Inheritance by distilled students is unmeasured for this teacher and student.",
		],
	}


def hf_tokenizer(model: str, revision: str) -> Callable[[str], list[int]]:
	"""The teacher's tokenizer, which must match the served model's vocabulary."""
	from transformers import AutoTokenizer

	tokenizer = AutoTokenizer.from_pretrained(model, revision=revision or None)
	return lambda text: list(tokenizer(text, add_special_tokens=False)["input_ids"])
