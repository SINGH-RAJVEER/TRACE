"""Geometric coverage proxy: how far each request extends the span already observed.

Embeddings are normalized, mapped into 64 dimensions by a frozen seeded Gaussian random
projection and rescaled so that no vector exceeds unit norm. For an account stream the
per-request gain is g_t = log(1 + v_t^T A_t^-1 v_t) with A_t = ridge * I + sum_{i<t} v_i v_i^T.
By the matrix determinant lemma, g_t is the increase in log det(A) caused by request t.

This is a coverage proxy only. It is not mutual information about teacher weights and it does
not predict student learning. The running sum of g_t grows with request count and would
recreate a volume threshold, so consumers must use the per-request gain or a windowed mean,
never the cumulative sum. The windowed mean still depends on request count, so its nulls
are matched by request-count band (length_bin) rather than pooled across horizon lengths.
"""

import hashlib
import math
from functools import lru_cache
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.mmd import Array

COVERAGE_DIM = 64
# Sherman-Morrison subtracts nearly equal terms when the ridge is tiny, and the gains vanish
# when it is huge; within these bounds updates agree with the closed form to about 1e-13.
RIDGE_MIN, RIDGE_MAX = 1e-3, 1e3


def length_bin(count: int) -> str:
	"""Request-count band, doubling in width, within which coverage nulls are matched."""
	if count < 2:
		raise ValueError("coverage needs at least two requests")
	low = 1 << (count.bit_length() - 1)
	return f"{low}-{2 * low - 1}"


@lru_cache(maxsize=8)
def _projection(seed: int, input_dim: int, dim: int) -> Array:
	matrix = np.random.default_rng(seed).standard_normal((dim, input_dim)) / math.sqrt(dim)
	matrix.setflags(write=False)
	return matrix


class CoverageSpec(BaseModel):
	"""Everything needed to reproduce the projection; persisted with the calibration.

	`projection_digest` is the SHA-256 of the projection matrix. It is filled on creation and
	checked on load, so a numpy release that changes the generator stream fails loudly instead
	of silently shifting the null. The spec is frozen so a fitted calibration cannot be
	re-pointed at another projection after its nulls were computed.
	"""

	model_config = ConfigDict(frozen=True)

	version: Literal["coverage-logdet-v1"] = "coverage-logdet-v1"
	seed: int
	input_dim: int = Field(gt=0)
	dim: Literal[64] = 64
	ridge: float = Field(default=1.0, ge=RIDGE_MIN, le=RIDGE_MAX)
	projection_digest: str | None = None

	@model_validator(mode="after")
	def _bind_digest(self) -> Self:
		digest = hashlib.sha256(self.matrix().tobytes()).hexdigest()
		if self.projection_digest not in {None, digest}:
			raise ValueError("coverage projection digest differs from the regenerated matrix")
		object.__setattr__(self, "projection_digest", digest)
		return self

	def matrix(self) -> Array:
		return _projection(self.seed, self.input_dim, self.dim)

	def project(self, embeddings: Array) -> Array:
		"""Normalize rows, project to `dim` and clip each row to norm at most one."""
		x = np.asarray(embeddings, dtype=np.float64)
		if x.ndim != 2 or x.shape[1] != self.input_dim:
			raise ValueError(f"coverage spec expects embeddings of width {self.input_dim}")
		if not np.all(np.isfinite(x)):
			raise ValueError("embeddings must be finite")
		norms = np.linalg.norm(x, axis=1, keepdims=True)
		y = (x / np.where(norms > 0, norms, 1.0)) @ self.matrix().T
		return y / np.maximum(1.0, np.linalg.norm(y, axis=1, keepdims=True))

	def gains(self, embeddings: Array) -> Array:
		"""Per-request gains g_t in stream order, each against the earlier requests only."""
		tracker = LogDetCoverage(self.dim, self.ridge)
		return np.asarray([tracker.update(v) for v in self.project(embeddings)])


class LogDetCoverage:
	"""Streaming g_t with a Sherman-Morrison inverse; O(dim^2) per update, no inversion.

	Float64 throughout. The rank-one correction is an outer product of one vector with
	itself, so the inverse stays exactly symmetric.
	"""

	def __init__(self, dim: int = COVERAGE_DIM, ridge: float = 1.0) -> None:
		if dim <= 0:
			raise ValueError("dim must be positive")
		if not RIDGE_MIN <= ridge <= RIDGE_MAX:
			raise ValueError(f"ridge must be in [{RIDGE_MIN}, {RIDGE_MAX}]")
		self.dim, self.ridge = dim, ridge
		self.reset()

	def reset(self) -> None:
		self._inverse = np.eye(self.dim) / self.ridge

	def update(self, v: Array) -> float:
		"""Return g_t for `v` against the stream so far, then absorb `v` into A."""
		vec = np.asarray(v, dtype=np.float64)
		if vec.shape != (self.dim,) or not np.all(np.isfinite(vec)):
			raise ValueError(f"v must be a finite vector of length {self.dim}")
		if np.linalg.norm(vec) > 1 + 1e-9:
			raise ValueError("v must have norm at most one")
		u = self._inverse @ vec
		quad = max(0.0, float(vec @ u))
		self._inverse -= np.outer(u, u) / (1.0 + quad)
		return math.log1p(quad)
