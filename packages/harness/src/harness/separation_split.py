"""Source-disjoint splits for exploratory window-level studies."""

from collections.abc import Sequence

import numpy as np

from harness.separation_groups import connected_source_groups


def split_windows(
	sources: Sequence[Sequence[str]],
	labels: Sequence[int],
	*,
	seed: int = 42,
	test_fraction: float = 0.5,
	minimum_per_class: int = 1,
) -> tuple[list[int], list[int]]:
	"""Split connected source components while retaining both classes on each side.

	A deterministic search over group assignments avoids breaking components to achieve
	class balance. It prefers the assignment whose worse class is closest to `test_fraction`,
	so a small class is not left almost entirely on one side. Fail closed when the data cannot
	support a usable split with `minimum_per_class` windows of each class per side.
	"""
	if len(sources) != len(labels) or set(labels) != {0, 1}:
		raise ValueError("sources and binary labels must describe the same windows")
	if not 0 < test_fraction < 1:
		raise ValueError("test_fraction must be between zero and one")
	groups = connected_source_groups(sources)
	unique = sorted(set(groups))
	if len(unique) < 2:
		raise ValueError("at least two independent source groups are required")
	totals = {label: sum(1 for value in labels if value == label) for label in (0, 1)}
	rng = np.random.default_rng(seed)
	best: tuple[float, list[int], list[int]] | None = None
	for _ in range(2048):
		permutation = rng.permutation(unique)
		count = max(1, min(len(unique) - 1, round(len(unique) * test_fraction)))
		test_groups = set(permutation[:count])
		train = [i for i, group in enumerate(groups) if group not in test_groups]
		test = [i for i, group in enumerate(groups) if group in test_groups]
		held = {label: sum(1 for i in test if labels[i] == label) for label in (0, 1)}
		if any(
			held[label] < minimum_per_class or totals[label] - held[label] < minimum_per_class
			for label in (0, 1)
		):
			continue
		error = max(abs(held[label] / totals[label] - test_fraction) for label in (0, 1))
		if best is None or error < best[0]:
			best = (error, train, test)
	if best is None:
		raise ValueError("no split with both classes found in source-group search")
	return best[1], best[2]
