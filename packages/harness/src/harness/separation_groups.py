"""Source-connected window groups for development-only separation studies.

Windows sharing any source identifier must be assigned to the same split. Transitive
connections matter: a window containing sources A and B joins all windows containing
A or B, even if those other windows have no sources in common with one another.
"""

from collections.abc import Sequence


def connected_source_groups(windows: Sequence[Sequence[str]]) -> list[int]:
	"""Return deterministic component labels, rejecting windows without provenance."""
	parents = list(range(len(windows)))

	def root(index: int) -> int:
		while parents[index] != index:
			parents[index] = parents[parents[index]]
			index = parents[index]
		return index

	owners: dict[str, int] = {}
	for index, sources in enumerate(windows):
		if not sources or any(not source for source in sources):
			raise ValueError("every window must have nonempty source identifiers")
		for source in sources:
			if source in owners:
				parents[root(index)] = root(owners[source])
			else:
				owners[source] = index
	labels: dict[int, int] = {}
	result: list[int] = []
	for index in range(len(windows)):
		component = root(index)
		if component not in labels:
			labels[component] = len(labels)
		result.append(labels[component])
	return result
