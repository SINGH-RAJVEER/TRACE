"""The keyed green-list definition that generation and detection must share.

Algorithm `kgw-hmac-splitmix` version 1. A context of preceding token IDs and a secret key select
a seed through HMAC-SHA256. The seed then assigns each token ID to the green list through a
splitmix64 finalizer, so a whole-vocabulary mask stays cheap and a single membership test is O(1).

The HMAC step is a standard keyed PRF. The splitmix64 step is not: it is a statistical mixer, and
an adversary who observes many memberships under one context may learn that context's seed.
Serving integration does not remove that limitation or establish adversarial robustness.
"""

import hashlib
import hmac
import math
import operator
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

ALGORITHM_ID = "kgw-hmac-splitmix"
ALGORITHM_VERSION = 1
_DOMAIN = b"trace-watermark-probe"  # Version 1 wire constant; retain the reference vectors.
_GOLDEN = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB
_MASK = (1 << 64) - 1


def exact_integer(value: Any, name: str) -> int:
	"""An exact integer, rejecting bools and floats that would otherwise truncate."""
	if isinstance(value, bool | np.bool_):
		raise ValueError(f"{name} must be an integer, not a boolean")
	try:
		return operator.index(value)
	except TypeError:
		raise ValueError(f"{name} must be an integer") from None


@dataclass(frozen=True)
class WatermarkKey:
	"""Secret key material. The secret never appears in its repr or in a manifest."""

	key_id: str
	epoch: int
	secret: bytes = field(repr=False)

	def __post_init__(self) -> None:
		if not isinstance(self.secret, bytes):
			raise ValueError("watermark secrets must be bytes")
		if len(self.secret) < 16:
			raise ValueError("watermark secrets need at least 16 bytes")
		if not self.key_id or not isinstance(self.key_id, str):
			raise ValueError("key_id must be a non-empty string")
		if not 0 <= exact_integer(self.epoch, "key epoch") < 2**32:
			raise ValueError("key epoch must fit in 32 unsigned bits")


@dataclass(frozen=True)
class WatermarkProfile:
	"""The parameters that generation and detection must agree on."""

	gamma: float
	delta: float
	context_width: int
	vocab_size: int

	def __post_init__(self) -> None:
		if not 0.0 < self.gamma < 1.0:
			raise ValueError("gamma must lie strictly between 0 and 1")
		if not math.isfinite(self.delta) or self.delta < 0.0:
			raise ValueError("delta must be finite and not negative")
		if not 1 <= exact_integer(self.context_width, "context width") < 2**32:
			raise ValueError("context width must be at least 1 and fit in 32 unsigned bits")
		if not 2 <= exact_integer(self.vocab_size, "vocabulary size") < 2**32:
			raise ValueError("vocabulary must hold at least 2 tokens and fit in 32 unsigned bits")
		if self.threshold < 1:
			raise ValueError("gamma is too small to mark any 64-bit value green")

	@property
	def threshold(self) -> int:
		"""A token is green when its 64-bit mix falls below this value."""
		return int(self.gamma * 2**64)

	@property
	def green_probability(self) -> float:
		"""The chance a uniform 64-bit mix is green. Rounding can separate this from gamma."""
		return self.threshold / 2**64


def token_id(token: int, profile: WatermarkProfile) -> int:
	"""Validate one token ID. Generation and detection share this rule."""
	value = exact_integer(token, "token ID")
	if not 0 <= value < profile.vocab_size:
		raise ValueError(f"token {value} lies outside the vocabulary")
	return value


def context_of(history: Sequence[int], width: int) -> tuple[int, ...] | None:
	"""The last `width` token IDs, or None while too few exist to watermark or score a token."""
	if len(history) < width:
		return None
	return tuple(history[-width:])


def context_seed(key: WatermarkKey, profile: WatermarkProfile, context: Sequence[int]) -> int:
	"""Derive the 64-bit seed for one context. Delta is excluded: detection need not know it."""
	if len(context) != profile.context_width:
		raise ValueError("context length must equal the profile's context width")
	message = b"".join(
		(
			_DOMAIN,
			struct.pack(
				">IIIdI",
				ALGORITHM_VERSION,
				key.epoch,
				profile.vocab_size,
				profile.gamma,
				profile.context_width,
			),
			struct.pack(f">{len(context)}I", *(token_id(token, profile) for token in context)),
		)
	)
	digest = hmac.new(key.secret, message, hashlib.sha256).digest()
	return int.from_bytes(digest[:8], "big")


def _mix(seed: int, token: int) -> int:
	value = (seed + (token + 1) * _GOLDEN) & _MASK
	value = ((value ^ (value >> 30)) * _MIX_A) & _MASK
	value = ((value ^ (value >> 27)) * _MIX_B) & _MASK
	return value ^ (value >> 31)


def is_green(seed: int, token: int, profile: WatermarkProfile) -> bool:
	"""Whether one token is green under a context seed."""
	return _mix(seed, token_id(token, profile)) < profile.threshold


def green_mask(seed: int, profile: WatermarkProfile) -> NDArray[np.bool_]:
	"""The green list as a boolean array over the vocabulary. Agrees with `is_green`."""
	tokens = np.arange(1, profile.vocab_size + 1, dtype=np.uint64)
	values = np.uint64(seed) + tokens * np.uint64(_GOLDEN)
	values = (values ^ (values >> np.uint64(30))) * np.uint64(_MIX_A)
	values = (values ^ (values >> np.uint64(27))) * np.uint64(_MIX_B)
	values = values ^ (values >> np.uint64(31))
	return values < np.uint64(profile.threshold)
