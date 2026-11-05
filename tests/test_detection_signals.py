"""Focused tests for isolated, shadow-only horizon telemetry."""

from pathlib import Path

import pytest
from harness.corpus import CorpusIndex
from harness.signals import SignalRow, fit_signals, raw_signals, score_signals


def horizon(offset: float = 0) -> list[SignalRow]:
	return [
		SignalRow("Explain the sorting algorithm", 0, (offset, 0), "coding"),
		SignalRow("Explain the sorting algorithm again", 2, (offset + 1, 0), "coding"),
		SignalRow("Answer only: what is two plus two", 9, (offset + 3, 0), "gsm8k"),
	]


def test_session_and_novelty_values() -> None:
	values = raw_signals(horizon())
	assert values["revisit_followup_rate"] == 0.5
	assert values["rephrasing_rate"] == 0.5
	assert values["answer_only_share"] == pytest.approx(1 / 3)
	assert values["interarrival_cv"] > 0
	assert values["topic_switch_rate"] == 0.5
	assert values["topic_transition_entropy"] > 0
	assert values["novelty_mean_nn"] == 1.5


def test_small_profile_null_abstains() -> None:
	fitted = fit_signals([horizon()] * 18, profiles=["generalist"] * 18)
	report = score_signals(horizon(), profile="generalist", calibration=fitted)
	assert all(v["p_value"] is None for v in report["metrics"].values())
	assert all(v["abstain"] for v in report["metrics"].values())
	fitted = fit_signals([horizon()] * 19, profiles=["generalist"] * 19)
	report = score_signals(horizon(), profile="generalist", calibration=fitted)
	assert report["metrics"]["template_diversity"]["minimum_null"] == 39
	assert report["metrics"]["template_diversity"]["rank_floor"] == 0.1
	assert report["metrics"]["template_diversity"]["abstain"]
	assert report["metrics"]["novelty_mean_nn"]["p_value"] == 1.0
	fitted = fit_signals([horizon()] * 39, profiles=["generalist"] * 39)
	assert all(
		v["p_value"] == 1.0
		for k, v in score_signals(horizon(), profile="generalist", calibration=fitted)[
			"metrics"
		].items()
		if k != "corpus_match_share"
	)
	assert all(
		v["abstain"]
		for v in score_signals(horizon(), profile="specialist", calibration=fitted)[
			"metrics"
		].values()
	)


def test_corpus_external_digest_and_order_invariance(tmp_path: Path) -> None:
	index = CorpusIndex.from_sources({"public/pinned@abc": [horizon()[0].text]})
	path = tmp_path / "index.json"
	index.save(path)
	assert CorpusIndex.load(path).digest == index.digest
	assert index.match(horizon()[0].text)["exact"] is True
	assert raw_signals(horizon(), index)["corpus_match_share"] == pytest.approx(1 / 3)
	reordered = [
		SignalRow(r.text, float(i), r.embedding, r.domain)
		for i, r in enumerate(reversed(horizon()))
	]
	assert (
		raw_signals(reordered, index)["corpus_match_share"]
		== raw_signals(horizon(), index)["corpus_match_share"]
	)
	fitted = fit_signals([horizon()] * 19, profiles=["generalist"] * 19, corpus=index)
	other = CorpusIndex.from_sources({"different": ["another public prompt"]})
	with pytest.raises(ValueError, match="digest"):
		score_signals(horizon(), profile="generalist", calibration=fitted, corpus=other)


def test_near_duplicate_and_missing_embeddings() -> None:
	index = CorpusIndex.from_sources(
		{"source@revision": ["one two three four five six seven eight nine ten"]}
	)
	assert index.match("one two three four five six seven eight nine ten eleven")["matched"]
	assert "novelty_mean_nn" not in raw_signals([SignalRow("a", 0, ())])
	with pytest.raises(ValueError, match="arrival order"):
		raw_signals([SignalRow("a", 2, (0,)), SignalRow("b", 1, (1,))])


def test_absent_statistics_and_topic_labels() -> None:
	fitted = fit_signals([horizon()] * 19, profiles=["g"] * 19)
	report = score_signals([SignalRow("one", 0, ())], profile="g", calibration=fitted)
	for name in (
		"novelty_mean_nn",
		"interarrival_cv",
		"topic_switch_rate",
		"rephrasing_rate",
		"corpus_match_share",
	):
		assert report["metrics"][name]["abstain"]
	rows = [SignalRow("a", 0, (0,), "coding", "sort"), SignalRow("b", 1, (1,), "coding", "graphs")]
	assert raw_signals(rows)["topic_switch_rate"] == 1.0


def test_calibrated_corpus_requires_matching_index() -> None:
	index = CorpusIndex.from_sources({"source@revision": ["one"]})
	fitted = fit_signals([horizon()], profiles=["g"], corpus=index)
	with pytest.raises(ValueError, match="digest"):
		score_signals(horizon(), profile="g", calibration=fitted)


def test_missing_arrival_times_abstain_without_invented_gaps() -> None:
	rows = [SignalRow("one two", None, (0,)), SignalRow("other words", None, (1,))]
	fitted = fit_signals([rows] * 40, profiles=["g"] * 40)
	report = score_signals(rows, profile="g", calibration=fitted)
	assert report["metrics"]["interarrival_cv"]["abstain"]
	assert report["metrics"]["novelty_mean_nn"]["value"] == 1
