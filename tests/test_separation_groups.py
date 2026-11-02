import pytest
from harness.separation_groups import connected_source_groups


def test_source_connections_are_transitive() -> None:
	groups = connected_source_groups([["a"], ["b", "c"], ["a", "b"], ["d"]])
	assert groups[0] == groups[1] == groups[2]
	assert groups[3] != groups[0]


def test_missing_source_is_rejected() -> None:
	with pytest.raises(ValueError, match="source identifiers"):
		connected_source_groups([["a"], []])


def test_split_keeps_connected_sources_together() -> None:
	from harness.separation_split import split_windows

	sources = [["a"], ["a", "b"], ["b"], ["c"], ["d"], ["e"], ["f"]]
	labels = [0, 0, 1, 0, 1, 0, 1]
	train, test = split_windows(sources, labels)
	assert set(train).isdisjoint(test)
	assert set(train) | set(test) == set(range(len(sources)))
	assert {source for i in train for source in sources[i]}.isdisjoint(
		{source for i in test for source in sources[i]}
	)
	assert {labels[i] for i in train} == {0, 1}
	assert {labels[i] for i in test} == {0, 1}


def test_split_rejects_unusable_source_groups() -> None:
	from harness.separation_split import split_windows

	with pytest.raises(ValueError, match="source-group search"):
		split_windows([["a"], ["a"], ["b"], ["b"]], [0, 0, 1, 1])
