import numpy as np
import pytest
from harness.clusters import LinkEvidence, PrincipalHorizon, analyze_clusters
from harness.fusion import FAMILIES, fuse_evidence


def account(name: str, truth: str) -> PrincipalHorizon:
	return PrincipalHorizon(name, [name], np.array([[float(ord(name)), 0.0]]), truth_group=truth)


def test_cluster_edges_are_analysis_only_with_explicit_errors() -> None:
	accounts = [account("a", "one"), account("b", "one"), account("c", "two")]
	result = analyze_clusters(
		accounts,
		edges=[LinkEvidence("a", "b", "declared", 1), LinkEvidence("b", "c", "shared", 0.2)],
	)
	assert result["enforcement"] is False
	assert result["clusters"][0]["members"] == ["a", "b"]
	assert result["edges"][1]["included"] is False
	assert result["benign_link_errors"]["false_merges"] == 0
	# Even a transitive analysis merge never produces policy decisions or bans.
	merged = analyze_clusters(
		accounts, edges=[LinkEvidence("a", "b", "declared", 1), LinkEvidence("b", "c", "shared", 1)]
	)
	assert merged["clusters"][0]["members"] == ["a", "b", "c"]
	assert merged["benign_link_errors"]["false_merges"] == 2
	assert merged["enforcement"] is False
	assert "action" not in merged


def test_org_complementarity_and_false_splits() -> None:
	accounts = [account("a", "one"), account("b", "one")]
	separate = analyze_clusters(accounts)
	assert separate["benign_link_errors"]["false_splits"] == 1
	pooled = analyze_clusters(accounts, organization_mapping={"a": "org", "b": "org"})
	assert pooled["clusters"][0]["complementarity"] == 1.0
	assert pooled["clusters"][0]["novelty_mean"] == 1.0
	assert pooled["edges"][0]["source"] == "declared_organization:org"


def test_fusion_fixed_mixture_and_missing_neutral() -> None:
	result = fuse_evidence({"novelty": 0.01})
	assert result["e_value"] == pytest.approx(1.8)
	assert result["families"]["volume"]["e_value"] == 1.0
	assert result["sequential_validity"] is False
	assert fuse_evidence({})["e_value"] == pytest.approx(1.0)
	with pytest.raises(ValueError):
		fuse_evidence({"novelty": 0.0})
	with pytest.raises(ValueError):
		fuse_evidence({}, weights=dict.fromkeys(FAMILIES, 1.0))


def test_fusion_superuniform_discrete_ranks_even_with_perfect_dependence() -> None:
	# Finite calibration rank is uniform on 1/(m+1), ..., 1 under exchangeability.
	values = [fuse_evidence(dict.fromkeys(FAMILIES, i / 20))["e_value"] for i in range(1, 21)]
	assert float(np.mean(values)) <= 1


def test_family_averaging_avoids_maximum_selection() -> None:
	from harness.fusion import average_calibrated_evidence, fuse_family_evalues

	family = average_calibrated_evidence({"one": 0.01, "two": 1.0})
	assert family["e_value"] == pytest.approx(2.75)
	assert fuse_family_evalues({"session": family["e_value"]})["e_value"] == pytest.approx(1.35)
