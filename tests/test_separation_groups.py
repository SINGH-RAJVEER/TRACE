import pytest
from harness.separation_groups import connected_source_groups


def test_source_connections_are_transitive() -> None:
	groups = connected_source_groups([["a"], ["b", "c"], ["a", "b"], ["d"]])
	assert groups[0] == groups[1] == groups[2]
	assert groups[3] != groups[0]


def test_missing_source_is_rejected() -> None:
	with pytest.raises(ValueError, match="source identifiers"):
		connected_source_groups([["a"], []])
