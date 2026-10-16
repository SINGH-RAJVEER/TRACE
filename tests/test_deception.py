import hashlib
from typing import Any

import numpy as np
import pytest
from harness.calibration import BenignRequest, HarnessConfig, LevelThresholds, calibrate
from harness.deception import AG_NEWS_LABELS, Probe, extract, noise, watermark_test
from harness.mmd import Array, WindowScorer
from harness.policy import MMDPolicy

KEY = bytes(range(32))
OTHER = bytes(range(1, 33))
PASSAGE = (
	"The Normans settled in Normandy in 911. Rollo signed a treaty with Charles the Simple, "
	"and by 1066 William of Normandy had conquered England with 7,000 men."
)
SQUAD = f"Answer the question.\n\nPassage: {PASSAGE}\n\nQuestion: Who signed the treaty?"


def test_noise_is_keyed_consistent_and_in_the_task_format() -> None:
	labels = [noise("ag_news", KEY, f"article {i}", 16) for i in range(60)]
	assert set(labels) == set(AG_NEWS_LABELS)
	assert labels == [noise("ag_news", KEY, f"article {i}", 16) for i in range(60)]
	assert labels != [noise("ag_news", OTHER, f"article {i}", 16) for i in range(60)]
	sentences = [sentence.strip(".") for sentence in PASSAGE.split(". ")]
	for i in range(20):
		span = noise("squad", KEY, SQUAD.replace("treaty", f"treaty {i}"), 64)
		assert 1 <= len(span.split()) <= 4 and any(span in s for s in sentences)


def test_arithmetic_noise_uses_the_problem_and_keeps_its_answer_under_any_budget() -> None:
	problem = "Tom has 3 boxes of 12 apples and gives away 5. How many are left?"
	long = noise("gsm8k", KEY, problem, 256)
	short = noise("gsm8k", KEY, problem, 16)
	assert long.splitlines()[-1].startswith("#### ") and "=" in long
	assert extract("gsm8k", long) == extract("gsm8k", short) is not None
	assert len(short.splitlines()) == 2 and len(long.splitlines()) >= 3
	text = noise("text", KEY, "Explain how tides work near the coast", 8)
	assert text.endswith(".") and len(text.split()) <= 6
	with pytest.raises(ValueError, match="task must be"):
		noise("unknown", KEY, problem, 16)


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


def test_watermark_detects_a_student_that_learned_the_noise() -> None:
	requests = [f"article {i}" for i in range(200)]
	copied = [
		Probe(r, noise("ag_news", KEY, r, 16) if i % 2 else "World") for i, r in enumerate(requests)
	]
	clean = [Probe(r, "World" if i % 3 else "Sports") for i, r in enumerate(requests)]
	stolen = watermark_test("ag_news", KEY, copied, controls=199)
	innocent = watermark_test("ag_news", KEY, clean, controls=199)
	assert stolen["agreement"] > 0.55 and stolen["p_value"] <= 0.01
	assert innocent["p_value"] > 0.05
