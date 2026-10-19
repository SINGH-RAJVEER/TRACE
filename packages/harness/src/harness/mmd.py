"""Unbiased squared MMD between query-embedding windows (paper Eq. 1)."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# The kernel is the mean of four RBF kernels with bandwidths c * sigma.
SCALES = (0.5, 1.0, 2.0, 4.0)


def squared_distances(a: Array, b: Array) -> Array:
	distances = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
	return np.maximum(distances, 0.0)


def kernel(distances: Array, sigma: float, scales: tuple[float, ...] = SCALES) -> Array:
	return np.mean([np.exp(-distances / (2 * (c * sigma) ** 2)) for c in scales], axis=0)


def _off_diagonal_mean(matrix: Array) -> float:
	n = matrix.shape[0]
	return float((matrix.sum() - np.trace(matrix)) / (n * (n - 1)))


def mmd2_unbiased(x: Array, z: Array, sigma: float, scales: tuple[float, ...] = SCALES) -> float:
	"""Negative finite-sample estimates are kept; no square root is taken."""
	return (
		_off_diagonal_mean(kernel(squared_distances(x, x), sigma, scales))
		+ _off_diagonal_mean(kernel(squared_distances(z, z), sigma, scales))
		- 2 * float(kernel(squared_distances(x, z), sigma, scales).mean())
	)


def distinct(keys: Sequence[str]) -> list[int]:
	"""Positions of each key's first occurrence, in window order."""
	first: dict[str, int] = {}
	for index, key in enumerate(keys):
		first.setdefault(key, index)
	return list(first.values())


def median_bandwidth(sample: Array) -> float:
	"""The median nonzero pairwise distance in a fixed benign fitting sample."""
	distances = np.sqrt(squared_distances(sample, sample))
	upper = distances[np.triu_indices(len(sample), k=1)]
	nonzero = upper[upper > 0]
	if not len(nonzero):
		raise ValueError("the fitting sample has no distinct embeddings")
	return float(np.median(nonzero))


class WindowScorer:
	"""Averages Eq. 1 over fixed reference windows, caching their self-similarity terms."""

	def __init__(self, references: Array, sigma: float, scales: tuple[float, ...] = SCALES) -> None:
		if references.ndim != 3 or references.shape[1] < 2:
			raise ValueError("references must have shape (windows, window_size, dimension)")
		self.references = references
		self.sigma = sigma
		self.scales = scales
		self.window_size = references.shape[1]
		self._reference_terms = [
			_off_diagonal_mean(kernel(squared_distances(z, z), sigma, scales)) for z in references
		]

	def score(self, window: Array, keys: Sequence[str] | None = None) -> float:
		"""Eq. 1 against each reference window, averaged.

		With `keys`, exact repeats within the window are collapsed to their first occurrence
		before scoring. Repeated prompts are identical points whose kernel value of one would
		otherwise inflate the within-window term. A window of one distinct prompt keeps that
		value, since no off-diagonal pair remains to estimate it.
		"""
		if window.shape[0] != self.window_size:
			raise ValueError(f"windows must hold {self.window_size} queries")
		if keys is not None:
			if len(keys) != window.shape[0]:
				raise ValueError("keys must name every query in the window")
			window = window[distinct(keys)]
		own = (
			_off_diagonal_mean(kernel(squared_distances(window, window), self.sigma, self.scales))
			if window.shape[0] > 1
			else 1.0
		)
		total = 0.0
		for z, reference_term in zip(self.references, self._reference_terms, strict=True):
			cross = float(kernel(squared_distances(window, z), self.sigma, self.scales).mean())
			total += own + reference_term - 2 * cross
		return total / len(self.references)

	def witness(self, window: Array) -> Array:
		"""Per-query MMD witness: mean kernel to the rest of the window minus to the references.

		Large values mark the queries that most separate this window from benign traffic.
		"""
		own = kernel(squared_distances(window, window), self.sigma, self.scales)
		n = window.shape[0]
		inside = (own.sum(1) - np.diag(own)) / (n - 1) if n > 1 else np.ones(1)
		outside = np.mean(
			[
				kernel(squared_distances(window, z), self.sigma, self.scales).mean(1)
				for z in self.references
			],
			axis=0,
		)
		return np.asarray(inside - outside, dtype=np.float64)
