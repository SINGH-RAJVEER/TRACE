"""Window-level two-sample statistics for the development-only separation study.

Everything here is numpy only: a ridge logistic regression, a small MLP, a feature-weighted
Gaussian MMD with train-split kernel selection, rank AUC, held-out permutation p-values and a
bootstrap clustered by source group. Each window is one sample. Nothing here calibrates or
enforces anything.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from harness.mmd import Array, squared_distances


def auc(scores: Array, labels: Array) -> float:
	"""Mann-Whitney AUC that label 1 scores above label 0, with ties counted half."""
	positive = labels == 1
	n1, n0 = int(positive.sum()), int((~positive).sum())
	if not n1 or not n0:
		raise ValueError("AUC needs both labels")
	order = np.argsort(scores, kind="mergesort")
	ranks = np.empty(len(scores))
	ordered = scores[order]
	start = 0
	while start < len(ordered):
		end = start
		while end + 1 < len(ordered) and ordered[end + 1] == ordered[start]:
			end += 1
		ranks[order[start : end + 1]] = (start + end) / 2 + 1
		start = end + 1
	return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def permutation_p(
	statistic: Callable[[Array], float], labels: Array, permutations: int, seed: int
) -> float:
	"""Upper-tail rank of the observed statistic among held-out label permutations."""
	rng = np.random.default_rng(seed)
	observed = statistic(labels)
	exceed = sum(statistic(rng.permutation(labels)) >= observed for _ in range(permutations))
	return (1 + exceed) / (permutations + 1)


def clustered_auc_interval(
	scores: Array,
	labels: Array,
	groups: Sequence[int],
	replicates: int,
	seed: int,
	level: float = 0.95,
) -> list[float] | None:
	"""Percentile interval resampling source groups within each class.

	A window inherits its group's draw. With fewer than two groups in either class, the
	interval is undefined and None is returned rather than a window-level substitute.
	"""
	members: dict[int, dict[int, list[int]]] = {0: {}, 1: {}}
	for index, (label, group) in enumerate(zip(labels, groups, strict=True)):
		members[int(label)].setdefault(group, []).append(index)
	if min(len(members[0]), len(members[1])) < 2:
		return None
	rng = np.random.default_rng(seed)
	pools = {label: list(found.values()) for label, found in members.items()}
	values = []
	for _ in range(replicates):
		chosen: list[int] = []
		for pool in pools.values():
			for pick in rng.integers(len(pool), size=len(pool)):
				chosen.extend(pool[pick])
		values.append(auc(scores[chosen], labels[chosen]))
	tail = (1 - level) / 2
	return [float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))]


def balanced_weights(labels: Array) -> Array:
	counts = np.bincount(labels.astype(int), minlength=2).astype(np.float64)
	return np.asarray(len(labels) / (2 * counts[labels.astype(int)]), dtype=np.float64)


class Standardizer:
	"""Train-split centering and scaling; constant columns keep unit scale."""

	def __init__(self, train: Array) -> None:
		self.mean = train.mean(0)
		scale = train.std(0)
		self.scale = np.where(scale > 1e-12, scale, 1.0)

	def __call__(self, rows: Array) -> Array:
		return np.asarray((rows - self.mean) / self.scale, dtype=np.float64)


def _sigmoid(values: Array) -> Array:
	return np.asarray(1 / (1 + np.exp(-np.clip(values, -40, 40))), dtype=np.float64)


def fit_logistic(
	features: Array, labels: Array, ridge: float = 1.0, iterations: int = 50
) -> Callable[[Array], Array]:
	"""Class-balanced ridge logistic regression by Newton steps; the intercept is unpenalized."""
	scale = Standardizer(features)
	x = np.hstack([scale(features), np.ones((len(features), 1))])
	weights = balanced_weights(labels)
	penalty = np.full(x.shape[1], ridge)
	penalty[-1] = 0.0
	beta = np.zeros(x.shape[1])
	for _ in range(iterations):
		p = _sigmoid(x @ beta)
		gradient = x.T @ (weights * (labels - p)) - penalty * beta
		hessian = (x.T * (weights * p * (1 - p))) @ x + np.diag(penalty) + 1e-9 * np.eye(len(beta))
		step = np.linalg.solve(hessian, gradient)
		beta = beta + step
		if float(np.abs(step).max()) < 1e-8:
			break

	def predict(rows: Array) -> Array:
		return np.asarray(np.hstack([scale(rows), np.ones((len(rows), 1))]) @ beta)

	return predict


def fit_mlp(
	features: Array,
	labels: Array,
	seed: int,
	components: int = 16,
	hidden: int = 16,
	steps: int = 400,
	rate: float = 0.01,
	decay: float = 1e-2,
) -> Callable[[Array], Array]:
	"""One tanh hidden layer on train-split principal components, trained by full-batch Adam."""
	scale = Standardizer(features)
	centered = scale(features)
	rank = max(1, min(components, len(features) - 1, features.shape[1]))
	_, _, vt = np.linalg.svd(centered, full_matrices=False)
	basis = vt[:rank].T
	projected = centered @ basis
	spread = projected.std(0)
	spread = np.where(spread > 1e-12, spread, 1.0)
	x = projected / spread
	rng = np.random.default_rng(seed)
	params = [
		rng.normal(0, 1 / np.sqrt(rank), (rank, hidden)),
		np.zeros(hidden),
		rng.normal(0, 1 / np.sqrt(hidden), hidden),
		np.zeros(1),
	]
	moments = [np.zeros_like(p) for p in params]
	squares = [np.zeros_like(p) for p in params]
	weights = balanced_weights(labels) / len(labels)
	for step in range(1, steps + 1):
		w1, b1, w2, b2 = params
		h = np.tanh(x @ w1 + b1)
		error = weights * (_sigmoid(h @ w2 + b2) - labels)
		dh = np.outer(error, w2) * (1 - h * h)
		grads = [
			x.T @ dh + decay * w1,
			dh.sum(0),
			h.T @ error + decay * w2,
			error.sum(keepdims=True),
		]
		for index, grad in enumerate(grads):
			moments[index] = 0.9 * moments[index] + 0.1 * grad
			squares[index] = 0.999 * squares[index] + 0.001 * grad * grad
			corrected = moments[index] / (1 - 0.9**step)
			params[index] = params[index] - rate * corrected / (
				np.sqrt(squares[index] / (1 - 0.999**step)) + 1e-8
			)
	w1, b1, w2, b2 = params

	def predict(rows: Array) -> Array:
		h = np.tanh(((scale(rows) @ basis) / spread) @ w1 + b1)
		return np.asarray(h @ w2 + b2)

	return predict


def mmd2_from_kernel(matrix: Array, labels: Array) -> float:
	"""Unbiased squared MMD of label-1 against label-0 rows of a joint kernel matrix."""
	x, z = labels == 1, labels == 0
	kxx, kzz, kxz = matrix[np.ix_(x, x)], matrix[np.ix_(z, z)], matrix[np.ix_(x, z)]
	n, m = len(kxx), len(kzz)
	return float(
		(kxx.sum() - np.trace(kxx)) / (n * (n - 1))
		+ (kzz.sum() - np.trace(kzz)) / (m * (m - 1))
		- 2 * kxz.mean()
	)


def _gaussian(a: Array, b: Array, weights: Array, sigma: float) -> Array:
	root = np.sqrt(weights)
	return np.asarray(
		np.exp(-squared_distances(a * root, b * root) / (2 * sigma**2)), dtype=np.float64
	)


def learned_mmd(
	train: Array,
	train_labels: Array,
	test: Array,
	test_labels: Array,
	permutations: int,
	seed: int,
	selection_permutations: int = 200,
) -> dict[str, Any]:
	"""Select a feature-weighted Gaussian kernel on train windows, then test it held out.

	Candidates cross median-heuristic bandwidth multiples with uniform weights or weights from
	train-split standardized mean differences. Selection maximizes the train statistic's
	permutation z-score. Test windows only see the frozen kernel.
	"""
	scale = Standardizer(train)
	tr, te = scale(train), scale(test)
	difference = np.abs(tr[train_labels == 1].mean(0) - tr[train_labels == 0].mean(0))
	weightings = {
		"uniform": np.full(tr.shape[1], 1 / tr.shape[1]),
		"mean_difference": difference / difference.sum()
		if difference.sum() > 0
		else np.full(tr.shape[1], 1 / tr.shape[1]),
	}
	rng = np.random.default_rng(seed)
	best: tuple[float, str, float] | None = None
	for name, weights in weightings.items():
		distances = squared_distances(tr * np.sqrt(weights), tr * np.sqrt(weights))
		upper = np.sqrt(distances[np.triu_indices(len(tr), k=1)])
		median = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
		for multiple in (0.25, 0.5, 1.0, 2.0, 4.0):
			sigma = median * multiple
			matrix = np.exp(-distances / (2 * sigma**2))
			observed = mmd2_from_kernel(matrix, train_labels)
			null = [
				mmd2_from_kernel(matrix, rng.permutation(train_labels))
				for _ in range(selection_permutations)
			]
			z = (observed - float(np.mean(null))) / (float(np.std(null)) + 1e-12)
			if best is None or z > best[0]:
				best = (z, name, sigma)
	assert best is not None
	_, name, sigma = best
	weights = weightings[name]
	joint = _gaussian(te, te, weights, sigma)
	statistic = mmd2_from_kernel(joint, test_labels)
	witness = _gaussian(te, tr[train_labels == 1], weights, sigma).mean(1) - _gaussian(
		te, tr[train_labels == 0], weights, sigma
	).mean(1)
	p = permutation_p(lambda y: mmd2_from_kernel(joint, y), test_labels, permutations, seed + 1)
	return {
		"weighting": name,
		"sigma": sigma,
		"train_z": best[0],
		"test_mmd2": statistic,
		"mmd_p_value": p,
		"witness": witness,
	}
