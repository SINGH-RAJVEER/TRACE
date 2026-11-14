import numpy as np
import pytest

from trace_watermark.bias import KgwBatchProcessor, apply_bias
from trace_watermark.core import WatermarkKey, WatermarkProfile, context_seed, green_mask, is_green

KEY = WatermarkKey("tenant-a", 0, b"public-test-material-only")
OTHER_KEY = WatermarkKey("tenant-b", 0, b"another-public-test-key")
PROFILE = WatermarkProfile(gamma=0.25, delta=1.5, context_width=2, vocab_size=64)


def make_logits(rows: int | None = None) -> np.ndarray:
	shape = PROFILE.vocab_size if rows is None else (rows, PROFILE.vocab_size)
	return np.random.default_rng(7).normal(size=shape)


def mask_for(key: WatermarkKey, context: tuple[int, int]) -> np.ndarray:
	return green_mask(context_seed(key, PROFILE, context), PROFILE)


def test_delta_lands_exactly_on_green_mask_without_mutating_input() -> None:
	logits = make_logits()
	before = logits.copy()
	mask = mask_for(KEY, (3, 4))
	assert mask.any() and not mask.all()
	output = apply_bias(logits, [9, 3, 4], KEY, PROFILE)
	np.testing.assert_array_equal(output[mask], before[mask] + PROFILE.delta)
	np.testing.assert_array_equal(output[~mask], before[~mask])
	np.testing.assert_array_equal(logits, before)
	assert output is not logits


def test_masked_tokens_stay_negative_infinity_and_no_nan_appears() -> None:
	mask = mask_for(KEY, (3, 4))
	masked = np.concatenate([np.flatnonzero(mask)[:2], np.flatnonzero(~mask)[:2]])
	logits = make_logits()
	logits[masked] = -np.inf
	output = apply_bias(logits, [3, 4], KEY, PROFILE)
	assert np.all(np.isneginf(output[masked]))
	assert not np.isnan(output).any()


def test_preserves_float32() -> None:
	logits = make_logits().astype(np.float32)
	assert apply_bias(logits, [3, 4], KEY, PROFILE).dtype == np.float32


@pytest.mark.parametrize(
	"logits",
	[np.zeros(63), np.zeros((1, 64)), np.zeros(64, dtype=np.int64)],
)
def test_bad_logits_are_rejected(logits: np.ndarray) -> None:
	with pytest.raises(ValueError):
		apply_bias(logits, [3, 4], KEY, PROFILE)


def test_short_context_is_skipped_and_counted() -> None:
	logits = make_logits()
	np.testing.assert_array_equal(apply_bias(logits, [3], KEY, PROFILE), logits)

	processor = KgwBatchProcessor(PROFILE)
	processor.add_request(0, "r0", [], KEY)
	batch = logits[np.newaxis]
	np.testing.assert_array_equal(processor.apply(batch, [0]), batch)
	processor.record_token(0, 3)
	np.testing.assert_array_equal(processor.apply(batch, [0]), batch)
	processor.record_token(0, 4)
	np.testing.assert_array_equal(
		processor.apply(batch, [0])[0], apply_bias(logits, [3, 4], KEY, PROFILE)
	)

	status = processor.status(0)
	assert status.rows_skipped_no_context == 2
	assert status.rows_biased == 1
	assert status.tokens_recorded == 2


def test_reordered_rows_with_mixed_keys_match_per_row_bias() -> None:
	processor = KgwBatchProcessor(PROFILE)
	processor.add_request(0, "r0", [1, 2], KEY)
	processor.add_request(1, "r1", [1, 2], OTHER_KEY)
	processor.add_request(2, "r2", [5, 6, 7], KEY)
	processor.record_token(2, 8)
	logits = make_logits(3)
	before = logits.copy()

	first = processor.apply(logits, [0, 1, 2])
	second = processor.apply(logits, [2, 0, 1])

	histories = {0: ([1, 2], KEY), 1: ([1, 2], OTHER_KEY), 2: ([7, 8], KEY)}
	for output, order in ((first, [0, 1, 2]), (second, [2, 0, 1])):
		for row, slot in enumerate(order):
			history, key = histories[slot]
			np.testing.assert_array_equal(
				output[row], apply_bias(logits[row], history, key, PROFILE)
			)
	np.testing.assert_array_equal(logits, before)
	assert not np.array_equal(first[0] - logits[0], first[1] - logits[1])


def test_moved_request_keeps_its_state() -> None:
	processor = KgwBatchProcessor(PROFILE)
	processor.add_request(0, "r0", [1, 2], KEY)
	processor.move_request(0, 5)
	logits = make_logits(1)
	np.testing.assert_array_equal(
		processor.apply(logits, [5])[0], apply_bias(logits[0], [1, 2], KEY, PROFILE)
	)
	assert processor.status(5).request_id == "r0"
	with pytest.raises(ValueError):
		processor.status(0)


def test_recycled_slot_carries_no_history_or_counts() -> None:
	processor = KgwBatchProcessor(PROFILE)
	processor.add_request(0, "old", [1, 2], KEY)
	logits = make_logits(1)
	processor.apply(logits, [0])
	assert processor.remove_request(0).rows_biased == 1

	processor.add_request(0, "new", [], KEY)
	np.testing.assert_array_equal(processor.apply(logits, [0]), logits)
	status = processor.status(0)
	assert status.request_id == "new"
	assert status.rows_biased == 0
	assert status.rows_skipped_no_context == 1


def test_prompt_is_copied() -> None:
	processor = KgwBatchProcessor(PROFILE)
	prompt = [1, 2]
	processor.add_request(0, "r0", prompt, KEY)
	prompt[:] = [8, 9]
	logits = make_logits(1)
	np.testing.assert_array_equal(
		processor.apply(logits, [0])[0], apply_bias(logits[0], [1, 2], KEY, PROFILE)
	)


def test_disabled_row_passes_through_beside_enabled_row() -> None:
	processor = KgwBatchProcessor(PROFILE)
	processor.add_request(0, "off", [1, 2], KEY, enabled=False)
	processor.add_request(1, "on", [1, 2], KEY)
	logits = make_logits(2)
	output = processor.apply(logits, [0, 1])
	np.testing.assert_array_equal(output[0], logits[0])
	assert not np.array_equal(output[1], logits[1])
	assert processor.status(0).rows_disabled == 1
	assert processor.status(0).rows_biased == 0
	assert processor.status(1).rows_biased == 1


def test_slot_errors() -> None:
	processor = KgwBatchProcessor(PROFILE)
	processor.add_request(0, "r0", [1, 2], KEY)
	processor.add_request(1, "r1", [1, 2], KEY)
	calls = [
		lambda: processor.add_request(0, "again", [], KEY),
		lambda: processor.remove_request(9),
		lambda: processor.move_request(9, 8),
		lambda: processor.move_request(0, 1),
		lambda: processor.record_token(9, 1),
		lambda: processor.record_token(0, PROFILE.vocab_size),
		lambda: processor.status(9),
	]
	for call in calls:
		with pytest.raises(ValueError):
			call()  # type: ignore[no-untyped-call]


def test_failed_apply_changes_no_counts() -> None:
	processor = KgwBatchProcessor(PROFILE)
	processor.add_request(0, "r0", [1, 2], KEY)
	logits = make_logits(2)
	for slots in ([0, 9], [0, 0], [0]):
		with pytest.raises(ValueError):
			processor.apply(logits, slots)
	assert processor.status(0).rows_biased == 0


def test_delta_two_raises_empirical_green_rate_above_gamma() -> None:
	profile = WatermarkProfile(gamma=0.25, delta=2.0, context_width=2, vocab_size=1000)
	processor = KgwBatchProcessor(profile)
	processor.add_request(0, "r0", [1, 2], KEY)
	rng = np.random.default_rng(20261106)
	logits = np.zeros((1, profile.vocab_size))
	history = [1, 2]
	green = 0
	draws = 500
	for _ in range(draws):
		scores = processor.apply(logits, [0])[0]
		probabilities = np.exp(scores) / np.exp(scores).sum()
		token = int(rng.choice(profile.vocab_size, p=probabilities))
		green += is_green(context_seed(KEY, profile, history[-2:]), token, profile)
		processor.record_token(0, token)
		history.append(token)
	# Expected rate: 0.25 e^2 / (0.25 e^2 + 0.75), about 0.71.
	assert green / draws > 0.6
	assert processor.status(0).rows_biased == draws


def green_index(profile: WatermarkProfile, history: list[int]) -> int:
	mask = green_mask(context_seed(KEY, profile, tuple(history)), profile)
	return int(np.flatnonzero(mask)[0])


def test_delta_that_overflows_the_dtype_is_an_error_and_masks_never_become_nan() -> None:
	profile = WatermarkProfile(gamma=0.5, delta=65536.0, context_width=1, vocab_size=8)
	logits = np.zeros(8, dtype=np.float16)
	logits[green_index(profile, [1])] = -np.inf
	with pytest.raises(ValueError, match="overflows"):
		apply_bias(logits, [1], KEY, profile)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_masks_survive_on_every_dtype(dtype: type[np.floating]) -> None:
	profile = WatermarkProfile(gamma=0.5, delta=1.0, context_width=1, vocab_size=8)
	logits = np.zeros(8, dtype=dtype)
	masked = green_index(profile, [1])
	logits[masked] = -np.inf
	biased = apply_bias(logits, [1], KEY, profile)
	assert biased[masked] == -np.inf
	assert biased.dtype == dtype
	assert not np.isnan(biased).any()


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_green_logit_overflow_is_an_error_on_every_dtype(dtype: type[np.floating]) -> None:
	largest = float(np.finfo(dtype).max)
	profile = WatermarkProfile(gamma=0.999, delta=largest, context_width=1, vocab_size=8)
	logits = np.full(8, largest, dtype=dtype)
	with pytest.raises(ValueError, match="overflows"):
		apply_bias(logits, [1], KEY, profile)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nan_and_positive_infinity_inputs_are_rejected(bad: float) -> None:
	logits = make_logits()
	logits[0] = bad
	with pytest.raises(ValueError, match="NaN"):
		apply_bias(logits, [3, 4], KEY, PROFILE)
	with pytest.raises(ValueError, match="NaN"):
		apply_bias(logits, [3], KEY, PROFILE)


def test_rejected_batch_changes_no_counters_even_after_earlier_rows_succeed() -> None:
	profile = WatermarkProfile(gamma=0.9, delta=100.0, context_width=1, vocab_size=8)
	processor = KgwBatchProcessor(profile)
	processor.add_request(0, "biased", [1], KEY)
	processor.add_request(1, "skipped", [], KEY)
	processor.add_request(2, "disabled", [1], KEY, enabled=False)
	processor.add_request(3, "overflows", [1], KEY)
	rows = np.zeros((4, 8), dtype=np.float16)
	rows[3] = 65504.0
	with pytest.raises(ValueError, match="overflows"), np.errstate(over="raise"):
		processor.apply(rows, [0, 1, 2, 3])
	for slot in range(4):
		status = processor.status(slot)
		assert (status.rows_biased, status.rows_skipped_no_context, status.rows_disabled) == (
			0,
			0,
			0,
		)
	rows[3] = 0.0
	processor.apply(rows, [0, 1, 2, 3])
	assert processor.status(0).rows_biased == 1
	assert processor.status(1).rows_skipped_no_context == 1
	assert processor.status(2).rows_disabled == 1


def test_repeated_apply_uses_the_same_mask_until_a_token_is_recorded() -> None:
	processor = KgwBatchProcessor(PROFILE)
	processor.add_request(0, "request", [3, 4], KEY)
	rows = make_logits()[np.newaxis]
	first = processor.apply(rows, [0])
	assert np.array_equal(first, processor.apply(rows, [0]))
	status = processor.status(0)
	assert (status.rows_biased, status.tokens_recorded) == (2, 0)
	processor.record_token(0, 5)
	assert not np.array_equal(first, processor.apply(rows, [0]))


@pytest.mark.parametrize("token", [1.9, True])
def test_generation_rejects_inexact_token_ids(token: object) -> None:
	with pytest.raises(ValueError):
		apply_bias(make_logits(), [token, token], KEY, PROFILE)  # type: ignore[list-item]
	processor = KgwBatchProcessor(PROFILE)
	with pytest.raises(ValueError):
		processor.add_request(0, "request", [token], KEY)  # type: ignore[list-item]


@pytest.mark.parametrize("history", [[True], [1.9], [-1], [999, 1, 2]])
def test_direct_calls_validate_short_histories_and_discarded_prefixes(history: list[int]) -> None:
	with pytest.raises(ValueError):
		apply_bias(make_logits(), history, KEY, PROFILE)
