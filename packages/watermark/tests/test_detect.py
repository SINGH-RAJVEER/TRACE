import dataclasses
import math
import random

import numpy as np
import pytest

from trace_watermark.core import (
	ALGORITHM_ID,
	ALGORITHM_VERSION,
	WatermarkKey,
	WatermarkProfile,
	context_of,
	context_seed,
	is_green,
)
from trace_watermark.detect import (
	binomial_log_upper_tail,
	binomial_upper_tail,
	binomial_z_score,
	detect,
	detect_candidates,
	holm_adjust,
)

KEY = WatermarkKey("fixture", 3, b"public-test-material-only")
WRONG = WatermarkKey("wrong", 0, b"another-public-test-key")
PROFILE = WatermarkProfile(gamma=0.25, delta=1.0, context_width=2, vocab_size=50000)
PROMPT = (11, 12, 13)
SPECIAL = 49999
ALPHA = 1e-3


def is_green_after(key: WatermarkKey, context: tuple[int, ...], token: int) -> bool:
	return is_green(context_seed(key, PROFILE, context), token, PROFILE)


def green_sequence(
	key: WatermarkKey, length: int, prompt: tuple[int, ...] = PROMPT, seed: int = 0
) -> list[int]:
	"""Every token whose context is available is forced green under `key`."""
	rng = random.Random(seed)
	history = list(prompt)
	for _ in range(length):
		context = context_of(history, PROFILE.context_width)
		token = rng.randrange(PROFILE.vocab_size - 1)
		while context is not None and not is_green_after(key, context, token):
			token = rng.randrange(PROFILE.vocab_size - 1)
		history.append(token)
	return history[len(prompt) :]


def exact_log_tail(k: int, n: int) -> float:
	"""Log P(X >= k) for gamma = 1/4 by integer arithmetic."""
	return math.log(sum(math.comb(n, i) * 3 ** (n - i) for i in range(k, n + 1))) - n * math.log(4)


@pytest.mark.parametrize(
	("k", "n", "expected"),
	[(4, 4, 1 / 256), (0, 4, 1.0), (3, 4, 13 / 256), (1, 4, 175 / 256), (5, 4, 0.0), (0, 0, 1.0)],
)
def test_exact_p_value_small_cases(k: int, n: int, expected: float) -> None:
	assert binomial_upper_tail(k, n, 0.25) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("k", [1100, 1250, 1300, 1500, 2500, 4000, 5000])
def test_log_tail_matches_integer_arithmetic_at_scale(k: int) -> None:
	expected = exact_log_tail(k, 5000)
	assert binomial_log_upper_tail(k, 5000, 0.25) == pytest.approx(expected, rel=1e-9, abs=1e-10)


def test_log_tail_is_monotonic_and_finite_at_scale() -> None:
	values = [binomial_log_upper_tail(k, 5000, 0.25) for k in range(0, 5001, 5)]
	assert all(math.isfinite(value) and value <= 0.0 for value in values)
	assert all(later <= earlier for earlier, later in zip(values, values[1:], strict=False))
	near_and_above_mean = values[1250 // 5 :]
	assert all(b < a for a, b in zip(near_and_above_mean, near_and_above_mean[1:], strict=False))
	assert values[-1] == pytest.approx(5000 * math.log(0.25))


def test_z_score_handles_empty_and_known_values() -> None:
	assert binomial_z_score(0, 0, 0.25) is None
	assert binomial_z_score(25, 100, 0.25) == 0.0
	assert binomial_z_score(4, 4, 0.25) == pytest.approx(3 / math.sqrt(0.75))


def test_repetition_does_not_inflate_significance() -> None:
	# A green 4-cycle, so every n-gram stays green across each junction of the repeat.
	rng = random.Random(1)
	while True:
		phrase = [rng.randrange(1000) for _ in range(4)]
		cycle = phrase + phrase[:2]
		if all(
			is_green_after(KEY, tuple(cycle[i : i + 2]), cycle[i + 2]) for i in range(len(phrase))
		):
			break

	once = detect(phrase * 2, KEY, PROFILE, alpha=ALPHA, min_effective=4)
	many = detect(phrase * 50, KEY, PROFILE, alpha=ALPHA, min_effective=4)
	assert (once.scored_tokens, once.unique_contexts, once.green_count) == (6, 4, 4)
	assert (many.scored_tokens, many.unique_contexts, many.green_count) == (198, 4, 4)
	assert many.raw_p_value == pytest.approx(1 / 256, rel=1e-12)
	assert many.raw_p_value == pytest.approx(once.raw_p_value)
	assert many.decision == "not_detected"


def test_excluded_token_is_unscored_but_stays_in_context() -> None:
	prompt = (1, 2)
	lead = 3
	# Green only if the special token stays in the context: dropping it would change the seed.
	token = next(
		candidate
		for candidate in range(PROFILE.vocab_size - 1)
		if is_green_after(KEY, (lead, SPECIAL), candidate)
		and not is_green_after(KEY, (prompt[-1], lead), candidate)
	)
	result = detect(
		[lead, SPECIAL, token],
		KEY,
		PROFILE,
		prompt,
		frozenset({SPECIAL}),
		alpha=ALPHA,
		min_effective=1,
	)
	assert (result.scored_tokens, result.unique_contexts) == (2, 2)
	assert result.green_count == 1 + is_green_after(KEY, prompt, lead)


def test_text_only_skips_tokens_without_full_context() -> None:
	generated = green_sequence(KEY, 40)
	aware = detect(generated, KEY, PROFILE, PROMPT, alpha=ALPHA, min_effective=1)
	text_only = detect(generated, KEY, PROFILE, alpha=ALPHA, min_effective=1)
	assert aware.scored_tokens == 40
	assert aware.green_count == 40
	assert text_only.scored_tokens == 40 - PROFILE.context_width
	assert text_only.green_count == 40 - PROFILE.context_width


def test_short_prompt_scores_only_tokens_with_full_context() -> None:
	generated = green_sequence(KEY, 10, prompt=(7,))
	result = detect(generated, KEY, PROFILE, (7,), alpha=ALPHA, min_effective=1)
	assert result.scored_tokens == 9


def test_abstains_below_min_effective_and_never_reports_a_negative() -> None:
	generated = green_sequence(KEY, 10)
	result = detect(generated, KEY, PROFILE, PROMPT, alpha=ALPHA, min_effective=11)
	assert result.decision == "insufficient_evidence"
	assert result.raw_p_value is None
	assert result.adjusted_p_value is None
	assert result.tested_hypotheses == 0
	assert result.green_count == 10

	at_floor = detect(generated, KEY, PROFILE, PROMPT, alpha=ALPHA, min_effective=10)
	assert at_floor.decision == "detected"


def test_no_scorable_tokens_abstains_with_no_z_score() -> None:
	result = detect([], KEY, PROFILE, alpha=ALPHA, min_effective=1)
	assert result.decision == "insufficient_evidence"
	assert result.z_score is None
	assert result.unique_contexts == 0


@pytest.mark.parametrize(
	"bad", [{"alpha": 0.0, "min_effective": 1}, {"alpha": 1.0, "min_effective": 1}]
)
def test_rejects_alpha_outside_the_open_interval(bad: dict[str, float]) -> None:
	with pytest.raises(ValueError, match="alpha"):
		detect([1, 2, 3], KEY, PROFILE, alpha=bad["alpha"], min_effective=1)


def test_rejects_a_floor_that_lets_no_evidence_read_as_negative() -> None:
	with pytest.raises(ValueError, match="min_effective"):
		detect([], KEY, PROFILE, alpha=ALPHA, min_effective=0)


def test_thresholds_have_no_defaults() -> None:
	with pytest.raises(TypeError):
		detect([1, 2, 3], KEY, PROFILE, alpha=ALPHA)  # type: ignore[call-arg]
	with pytest.raises(TypeError):
		detect([1, 2, 3], KEY, PROFILE, min_effective=1)  # type: ignore[call-arg]


def test_rejects_scored_token_outside_vocabulary() -> None:
	with pytest.raises(ValueError, match="vocabulary"):
		detect([PROFILE.vocab_size], KEY, PROFILE, PROMPT, alpha=ALPHA, min_effective=1)


def test_accepts_numpy_token_ids() -> None:
	generated = green_sequence(KEY, 16)
	as_numpy = detect(
		np.array(generated, dtype=np.int64),  # type: ignore[arg-type]
		KEY,
		PROFILE,
		np.array(PROMPT, dtype=np.int64),  # type: ignore[arg-type]
		alpha=ALPHA,
		min_effective=1,
	)
	assert as_numpy == detect(generated, KEY, PROFILE, PROMPT, alpha=ALPHA, min_effective=1)


def test_green_sequence_detected_under_right_key_only() -> None:
	generated = green_sequence(KEY, 64)
	right = detect(generated, KEY, PROFILE, PROMPT, alpha=ALPHA, min_effective=16)
	wrong = detect(generated, WRONG, PROFILE, PROMPT, alpha=ALPHA, min_effective=16)

	assert right.decision == "detected"
	assert right.green_count == 64
	assert right.raw_p_value == pytest.approx(0.25**64, rel=1e-9)
	assert right.adjusted_p_value == right.raw_p_value
	assert right.tested_hypotheses == 1
	assert (right.candidate_key_id, right.key_epoch) == ("fixture", 3)
	assert (right.algorithm_id, right.algorithm_version) == (ALGORITHM_ID, ALGORITHM_VERSION)
	assert wrong.decision == "not_detected"
	assert wrong.raw_p_value is not None
	assert wrong.raw_p_value > ALPHA


def test_detection_carries_no_secret_material() -> None:
	result = detect(green_sequence(KEY, 8), KEY, PROFILE, PROMPT, alpha=ALPHA, min_effective=1)
	assert "public-test" not in repr(result)
	assert not any(isinstance(value, bytes) for value in dataclasses.asdict(result).values())
	assert not any("secret" in field.name for field in dataclasses.fields(result))


def test_holm_known_example() -> None:
	assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_clips_at_one() -> None:
	assert holm_adjust([0.9, 0.8]) == pytest.approx([1.0, 1.0])


def test_holm_leaves_abstentions_out_of_the_family() -> None:
	# One abstention must give the same family as the two-hypothesis case: 2 * 0.01, 1 * 0.04.
	adjusted = holm_adjust([0.01, None, 0.04])
	assert adjusted[1] is None
	assert adjusted[0] == pytest.approx(0.02)
	assert adjusted[2] == pytest.approx(0.04)
	assert holm_adjust([None, None]) == [None, None]


def test_candidates_correct_across_keys_and_keep_input_order() -> None:
	others = [WatermarkKey(f"other-{i}", i, f"public-test-key-{i:04d}".encode()) for i in range(2)]
	generated = green_sequence(KEY, 64)
	results = detect_candidates(
		generated,
		[others[0], KEY, others[1]],
		PROFILE,
		alpha=ALPHA,
		min_effective=16,
		prompt_ids=PROMPT,
	)
	assert [result.candidate_key_id for result in results] == ["other-0", "fixture", "other-1"]
	assert [result.decision for result in results] == ["not_detected", "detected", "not_detected"]
	assert all(result.tested_hypotheses == 3 for result in results)
	assert holm_adjust([result.raw_p_value for result in results]) == [
		result.adjusted_p_value for result in results
	]
	assert results[1].raw_p_value is not None
	assert results[1].adjusted_p_value == pytest.approx(3 * results[1].raw_p_value)


def test_decision_uses_adjusted_rather_than_raw_p_value() -> None:
	generated = green_sequence(KEY, 7)
	raw = 0.25**7
	assert raw < 1e-4 < 2 * raw
	alone = detect(generated, KEY, PROFILE, PROMPT, alpha=1e-4, min_effective=1)
	family = detect_candidates(
		generated, [KEY, WRONG], PROFILE, alpha=1e-4, min_effective=1, prompt_ids=PROMPT
	)
	assert alone.decision == "detected"
	assert family[0].raw_p_value == pytest.approx(raw)
	assert family[0].adjusted_p_value == pytest.approx(2 * raw)
	assert family[0].decision == "not_detected"


def test_candidates_below_floor_are_all_untested() -> None:
	results = detect_candidates(
		green_sequence(KEY, 4),
		[KEY, WRONG],
		PROFILE,
		alpha=ALPHA,
		min_effective=5,
		prompt_ids=PROMPT,
	)
	assert [result.decision for result in results] == ["insufficient_evidence"] * 2
	assert [result.tested_hypotheses for result in results] == [0, 0]
	assert [result.adjusted_p_value for result in results] == [None, None]


def test_candidates_require_at_least_one_key() -> None:
	with pytest.raises(ValueError, match="candidate"):
		detect_candidates([1, 2, 3], [], PROFILE, alpha=ALPHA, min_effective=1)


def test_null_false_positive_rate_matches_the_attained_level() -> None:
	alpha = 0.05
	length = 150
	trials = 2000
	rng = random.Random(20261106)
	# The test is discrete, so its true size is the tail at the first count that clears alpha.
	critical = next(k for k in range(length + 1) if binomial_upper_tail(k, length, 0.25) <= alpha)
	attained = binomial_upper_tail(critical, length, 0.25)
	assert 0.02 < attained <= alpha

	detected = 0
	for _ in range(trials):
		key = WatermarkKey("null", 0, rng.randbytes(32))
		generated = [rng.randrange(PROFILE.vocab_size) for _ in range(length)]
		result = detect(generated, key, PROFILE, PROMPT, alpha=alpha, min_effective=length)
		assert result.unique_contexts == length
		detected += result.decision == "detected"

	margin = 4 * math.sqrt(attained * (1 - attained) / trials)
	assert abs(detected / trials - attained) <= margin


def test_tokens_sharing_a_context_count_as_one_event() -> None:
	profile = WatermarkProfile(gamma=2.0**-64, delta=1.0, context_width=1, vocab_size=2)
	result = detect([0, 1], KEY, profile, prompt_ids=[0], alpha=0.05, min_effective=1)
	assert (result.scored_tokens, result.unique_contexts) == (2, 1)
	assert result.raw_p_value in (1.0, 2.0**-64)


def test_selected_event_is_the_first_scorable_token_whatever_the_key() -> None:
	profile = WatermarkProfile(gamma=0.25, delta=1.0, context_width=1, vocab_size=512)
	other = WatermarkKey("other", 0, b"another-public-test-key")
	seed = context_seed(KEY, profile, (7,))
	green = next(t for t in range(512) if is_green(seed, t, profile))
	red = next(t for t in range(512) if not is_green(seed, t, profile))
	# Context 7 is followed by a red token, then a green one. The later green must not count.
	first_red = detect([7, red, 7, green], KEY, profile, alpha=0.05, min_effective=1)
	first_green = detect([7, green, 7, red], KEY, profile, alpha=0.05, min_effective=1)
	assert first_red.unique_contexts == first_green.unique_contexts
	seed_red = context_seed(KEY, profile, (red,))
	seed_green = context_seed(KEY, profile, (green,))
	assert first_red.green_count == is_green(seed_red, 7, profile)
	assert first_green.green_count == 1 + is_green(seed_green, 7, profile)
	selected = detect([7, red, 7, green], other, profile, alpha=0.05, min_effective=1)
	assert selected.unique_contexts == first_red.unique_contexts


def test_excluded_first_token_leaves_the_next_scorable_token_for_its_context() -> None:
	profile = WatermarkProfile(gamma=0.25, delta=1.0, context_width=1, vocab_size=512)
	result = detect(
		[9, 7, 9, 8], KEY, profile, excluded_ids=frozenset({7}), alpha=0.05, min_effective=1
	)
	assert result.unique_contexts == 2
	assert result.scored_tokens == 2


def test_null_holds_for_small_vocabulary_with_repeated_contexts() -> None:
	profile = WatermarkProfile(gamma=0.25, delta=1.0, context_width=1, vocab_size=64)
	rng = random.Random(7)
	alpha = 0.05
	trials = 1500
	hits = 0
	for trial in range(trials):
		key = WatermarkKey("null", 0, f"null-key-material-{trial}".encode())
		sequence = [rng.randrange(64) for _ in range(400)]
		hits += detect(sequence, key, profile, alpha=alpha, min_effective=16).decision == "detected"
	assert hits / trials <= alpha + 4 * math.sqrt(alpha * (1 - alpha) / trials)


def test_duplicate_candidate_keys_keep_holm_valid() -> None:
	profile = WatermarkProfile(gamma=0.25, delta=1.0, context_width=1, vocab_size=512)
	twin = WatermarkKey("twin", KEY.epoch, KEY.secret)
	sequence = list(range(100))
	first, second = detect_candidates(sequence, [KEY, twin], profile, alpha=0.05, min_effective=1)
	assert first.green_count == second.green_count
	assert first.raw_p_value == second.raw_p_value
	assert first.tested_hypotheses == 2
	assert first.raw_p_value is not None
	assert first.adjusted_p_value == second.adjusted_p_value == min(1.0, 2 * first.raw_p_value)


@pytest.mark.parametrize("field", ["generated", "prompt"])
@pytest.mark.parametrize("bad", [-1, 1.9, True, 10**6])
def test_every_supplied_token_is_validated_even_without_scoring(field: str, bad: object) -> None:
	profile = WatermarkProfile(gamma=0.25, delta=1.0, context_width=4, vocab_size=512)
	generated, prompt = ([bad], []) if field == "generated" else ([1], [bad, 1, 2])  # type: ignore[list-item]
	with pytest.raises(ValueError):
		detect(generated, KEY, profile, prompt, alpha=0.05, min_effective=1)  # type: ignore[arg-type]


def test_floor_must_be_an_exact_integer() -> None:
	profile = WatermarkProfile(gamma=0.25, delta=1.0, context_width=1, vocab_size=512)
	with pytest.raises(ValueError):
		detect([1, 2], KEY, profile, alpha=0.05, min_effective=1.5)  # type: ignore[arg-type]


def test_p_value_uses_the_rounded_threshold_rather_than_gamma(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	profile = WatermarkProfile(gamma=1.5 * 2.0**-64, delta=1.0, context_width=1, vocab_size=2)
	assert profile.green_probability == 2.0**-64
	monkeypatch.setattr("trace_watermark.detect.is_green", lambda *_: True)
	result = detect([1], KEY, profile, prompt_ids=[0], alpha=0.05, min_effective=1)
	assert result.green_count == 1
	assert result.raw_p_value == pytest.approx(2.0**-64, rel=1e-12, abs=0.0)
