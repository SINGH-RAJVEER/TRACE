"""The shared KGW serving contract. No torch, vLLM or app-environment dependencies."""

import hashlib
import hmac
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trace_watermark.core import WatermarkKey, WatermarkProfile, exact_integer
from trace_watermark.keys import read_master_secret

SCHEME = "kgw-v1"
VLLM_VERSION = "0.28.0"
PROCESSOR = "trace_watermark.vllm:KgwLogitsProcessor"
MIDDLEWARE = "trace_watermark.middleware.KgwMiddleware"
CHALLENGE_HEADER = "X-Trace-Watermark-Challenge"
ACK_HEADER = "X-Trace-Watermark-Ack"
_PREFIX = "trace_watermark_"
_FIELDS = {f"{_PREFIX}{name}" for name in ("key", "spec", "nonce", "auth")}


@dataclass(frozen=True)
class KgwSpec:
	"""Explicit model vocabulary plus versioned parameters, identical on both HTTP ends."""

	vocab_size: int
	gamma: float = 0.25
	delta: float = 2.0
	context_width: int = 2
	epoch: int = 0

	def __post_init__(self) -> None:
		self.profile()
		if self.delta <= 0 or self.delta > 20:
			raise ValueError("serving delta must be positive and at most 20")
		if not 0 <= exact_integer(self.epoch, "key epoch") < 2**32:
			raise ValueError("key epoch must fit in 32 unsigned bits")

	def profile(self) -> WatermarkProfile:
		return WatermarkProfile(self.gamma, self.delta, self.context_width, self.vocab_size)

	@property
	def version(self) -> str:
		return SCHEME

	@property
	def wire(self) -> str:
		return json.dumps({"scheme": SCHEME, **asdict(self)}, sort_keys=True, separators=(",", ":"))

	@property
	def digest(self) -> str:
		return hashlib.sha256(self.wire.encode()).hexdigest()

	@classmethod
	def from_env(cls) -> "KgwSpec":
		try:
			return cls(
				vocab_size=int(os.environ["TEACHER_WATERMARK_VOCAB_SIZE"]),
				gamma=float(os.getenv("TEACHER_WATERMARK_GAMMA", "0.25")),
				delta=float(os.getenv("TEACHER_WATERMARK_DELTA", "2.0")),
				context_width=int(os.getenv("TEACHER_WATERMARK_CONTEXT_WIDTH", "2")),
				epoch=int(os.getenv("TEACHER_WATERMARK_EPOCH", "0")),
			)
		except (KeyError, ValueError) as error:
			raise ValueError(
				"invalid KGW settings; declare TEACHER_WATERMARK_VOCAB_SIZE"
			) from error


def engine_secret() -> bytes:
	"""Workers only read a provisioned key; they never generate a replacement."""
	return read_master_secret(
		Path(os.getenv("TEACHER_WATERMARK_KEY_FILE", "artifacts/teacher.watermark.key"))
	)


def principal_key(secret: bytes, key: str, spec: KgwSpec) -> WatermarkKey:
	if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{16,64}", key) is None:
		raise ValueError("invalid watermark key ID")
	derived = hmac.new(secret, b"trace-kgw-key-v1\0" + key.encode(), hashlib.sha256).digest()
	return WatermarkKey(key, spec.epoch, derived)


def _mac(secret: bytes, purpose: str, *values: str) -> str:
	message = json.dumps([SCHEME, purpose, *values], separators=(",", ":")).encode()
	return hmac.new(secret, message, hashlib.sha256).hexdigest()


def request_args(secret: bytes, key: str, spec: KgwSpec, nonce: str) -> dict[str, str]:
	principal_key(secret, key, spec)
	return {
		f"{_PREFIX}key": key,
		f"{_PREFIX}spec": spec.wire,
		f"{_PREFIX}nonce": nonce,
		f"{_PREFIX}auth": _mac(secret, "request", key, spec.wire, nonce),
	}


def validate_request(extra_args: dict[str, Any] | None) -> tuple[KgwSpec, WatermarkKey] | None:
	fields = {key for key in (extra_args or {}) if key.startswith(_PREFIX)}
	if not fields:
		return None
	if fields != _FIELDS or extra_args is None:
		raise ValueError("incomplete KGW request")
	values = [extra_args[f"{_PREFIX}{name}"] for name in ("key", "spec", "nonce", "auth")]
	if not all(isinstance(value, str) for value in values):
		raise ValueError("invalid KGW request fields")
	key, wire, nonce, auth = values
	if re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
		raise ValueError("invalid KGW nonce")
	spec = KgwSpec.from_env()
	secret = engine_secret()
	if wire != spec.wire or not hmac.compare_digest(
		auth, _mac(secret, "request", key, wire, nonce)
	):
		raise ValueError("KGW key or configuration mismatch")
	return spec, principal_key(secret, key, spec)


def acknowledgement(secret: bytes, nonce: str, spec: KgwSpec) -> str:
	"""Authenticated configuration acknowledgement, not a statistical detection claim."""
	return _mac(secret, "configured", nonce, spec.wire)
