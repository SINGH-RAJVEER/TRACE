import hashlib
from typing import Any

import numpy as np
import pytest
from harness.calibration import BenignRequest, HarnessConfig, LevelThresholds, calibrate
from harness.deception import (
	LabelDeceiver,
	NumericTraceDeceiver,
	Probe,
	SpanDeceiver,
	deceive,
	released_text,
	watermark_test,
)
from harness.mmd import Array, WindowScorer
from harness.policy import MMDPolicy

KEY = bytes(range(32))
OTHER = bytes(range(1, 33))
PASSAGE = (
	"The Normans settled in Normandy in 911. Rollo signed a treaty with Charles the Simple, "
	"and by 1066 William of Normandy had conquered England with 7,000 men."
)
SQUAD = f"Answer the question.\n\nPassage: {PASSAGE}\n\nQuestion: Who signed the treaty?"


def test_label_deceiver_is_wrong_consistent_and_keyed() -> None:
	deceiver = LabelDeceiver()
	answers = {deceive(deceiver, KEY, f"article {i}", "Sports") for i in range(40)}
	assert "Sports" not in answers and len(answers) == 3
	assert deceive(deceiver, KEY, "article 1", "Sports") == deceive(
		deceiver, KEY, "article 1", "Sports"
	)
	differing = sum(
		deceive(deceiver, KEY, f"article {i}", "Sports")
		!= deceive(deceiver, OTHER, f"article {i}", "Sports")
		for i in range(40)
	)
	assert differing > 10


def test_span_deceiver_picks_a_same_kind_span_that_avoids_the_answer() -> None:
	deceiver = SpanDeceiver()
	name = deceive(deceiver, KEY, SQUAD, "Rollo")
	assert name and name in PASSAGE and name[0].isupper() and "Rollo" not in name
	number = deceive(deceiver, KEY, SQUAD, "911")
	assert number in {"1066", "7,000"}
	phrase = deceive(deceiver, KEY, SQUAD, "signed a treaty")
	assert phrase and phrase in PASSAGE and "treaty" not in phrase
	assert deceive(deceiver, KEY, "no passage here", "Rollo") is None


def test_numeric_deceiver_rewrites_the_final_answer() -> None:
	solution = "She has 3 * 4 = 12 apples.\nThen 12 + 6 = 18.\n#### 18"
	deceiver = NumericTraceDeceiver()
	text = deceive(deceiver, KEY, "problem", solution)
	assert text is not None
	target = deceiver.extract(text)
	assert target not in {None, "18"} and text.endswith(f"#### {target}")
	assert f"= {target}." in text
	rewritten = deceive(
		NumericTraceDeceiver(rewriter=lambda prompt: "Rewritten steps.\n#### 0"),
		KEY,
		"problem",
		solution,
	)
	assert rewritten == f"Rewritten steps.\n#### {target}"
	assert released_text(False, True, deceiver, KEY, "p", solution) is None
	assert released_text(False, False, None, None, "p", solution) == solution


class Encoder:
	def encode(self, texts: list[str]) -> Array:
		rows = [
			np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).normal(
				size=4
			)
			for t in texts
		]
		return np.array([row / np.linalg.norm(row) for row in rows])

	def identity(self) -> dict[str, Any]:
		return {"model": "fake"}


def test_deceiving_policy_looks_like_l3_and_marks_a_keyed_share() -> None:
	requests = [BenignRequest(account=f"a{i // 30}", text=f"q {i}") for i in range(600)]
	calibration, references = calibrate(
		config=HarnessConfig(window_size=5, reference_windows=2, bandwidth_sample=20),
		encoder=Encoder(),
		fitting=requests[:200],
		score_calibration=requests[200:400],
		policy_calibration=requests[400:],
	)
	calibration.thresholds["paper"] = LevelThresholds(
		thresholds=[-1, -1, -1, -1], achieved=[0] * 4, units=1
	)
	scorer = WindowScorer(references, calibration.sigma)
	with pytest.raises(ValueError, match="secret key"):
		MMDPolicy(calibration, scorer, encoder=Encoder(), l4_action="deceive")
	policy = MMDPolicy(
		calibration,
		scorer,
		encoder=Encoder(),
		l4_action="deceive",
		deception_rate=0.5,
		deception_key=KEY,
	)
	decisions = [policy.admit("attacker", f"r {i}", now=float(i)) for i in range(205)]
	flagged = decisions[4:]
	assert {d.level for d in flagged} == {4} and {d.action for d in flagged} == {"challenge"}
	assert all("deceive" not in d.client_view() for d in flagged)
	assert 60 < sum(d.deceive for d in flagged) < 140
	again = MMDPolicy(
		calibration,
		scorer,
		encoder=Encoder(),
		l4_action="deceive",
		deception_rate=0.5,
		deception_key=KEY,
	)
	repeat = [again.admit("sybil", f"r {i}", now=float(i)) for i in range(205)]
	assert [d.deceive for d in repeat[4:]] == [d.deceive for d in flagged]


def test_watermark_detects_a_student_that_learned_the_keyed_answers() -> None:
	deceiver = LabelDeceiver()
	requests = [f"article {i}" for i in range(200)]
	copied = [
		Probe(r, "World", (deceive(deceiver, KEY, r, "World") or "") if i % 2 else "World")
		for i, r in enumerate(requests)
	]
	clean = [Probe(r, "World", "World" if i % 3 else "Sports") for i, r in enumerate(requests)]
	stolen = watermark_test(deceiver, KEY, copied, controls=199)
	innocent = watermark_test(deceiver, KEY, clean, controls=199)
	assert stolen["agreement"] == pytest.approx(0.5) and stolen["p_value"] <= 0.01
	assert innocent["p_value"] > 0.05
