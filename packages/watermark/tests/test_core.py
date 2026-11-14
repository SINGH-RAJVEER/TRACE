import numpy as np
import pytest

from trace_watermark.core import (
	WatermarkKey,
	WatermarkProfile,
	context_of,
	context_seed,
	green_mask,
	is_green,
	token_id,
)

KEY = WatermarkKey("fixture", 0, b"public-test-material-only")
PROFILE = WatermarkProfile(gamma=0.25, delta=1.0, context_width=2, vocab_size=5000)


def test_mask_agrees_with_single_token_membership_and_gamma() -> None:
	seed = context_seed(KEY, PROFILE, (17, 4))
	mask = green_mask(seed, PROFILE)
	assert [bool(value) for value in mask] == [
		is_green(seed, token, PROFILE) for token in range(PROFILE.vocab_size)
	]
	assert np.mean(mask) == pytest.approx(PROFILE.gamma, abs=0.03)


def test_seed_depends_on_key_epoch_and_context_but_not_delta() -> None:
	seed = context_seed(KEY, PROFILE, (1, 2))
	assert seed != context_seed(KEY, PROFILE, (2, 1))
	assert seed != context_seed(WatermarkKey("fixture", 1, KEY.secret), PROFILE, (1, 2))
	assert seed != context_seed(
		WatermarkKey("other", 0, b"another-public-test-key"), PROFILE, (1, 2)
	)
	stronger = WatermarkProfile(gamma=0.25, delta=2.0, context_width=2, vocab_size=5000)
	assert seed == context_seed(KEY, stronger, (1, 2))


def test_secret_stays_out_of_repr() -> None:
	assert "public-test" not in repr(KEY)


def test_short_history_has_no_context() -> None:
	assert context_of([1], 2) is None
	assert context_of([1, 2, 3], 2) == (2, 3)


@pytest.mark.parametrize("delta", [float("nan"), float("inf"), -1.0])
def test_profile_rejects_unusable_delta(delta: float) -> None:
	with pytest.raises(ValueError):
		WatermarkProfile(gamma=0.25, delta=delta, context_width=1, vocab_size=10)


@pytest.mark.parametrize("context", [(-1, 2), (1, 5000)])
def test_context_outside_vocabulary_raises_value_error(context: tuple[int, int]) -> None:
	with pytest.raises(ValueError):
		context_seed(KEY, PROFILE, context)


def test_membership_accepts_numpy_integers_and_rejects_outside_vocabulary() -> None:
	seed = context_seed(KEY, PROFILE, (1, 2))
	assert is_green(seed, np.int64(7), PROFILE) == is_green(seed, 7, PROFILE)  # type: ignore[arg-type]
	with pytest.raises(ValueError):
		is_green(seed, 5000, PROFILE)


@pytest.mark.parametrize(
	"overrides",
	[
		{"vocab_size": 2**32},
		{"vocab_size": 10.0},
		{"vocab_size": True},
		{"context_width": 2**32},
		{"context_width": 1.0},
		{"gamma": 2.0**-70},
	],
)
def test_profiles_that_cannot_be_encoded_are_rejected(overrides: dict[str, object]) -> None:
	values: dict[str, object] = {"gamma": 0.25, "delta": 1.0, "context_width": 1, "vocab_size": 10}
	with pytest.raises(ValueError):
		WatermarkProfile(**{**values, **overrides})  # type: ignore[arg-type]


@pytest.mark.parametrize("epoch", [0.5, True, -1, 2**32])
def test_keys_that_cannot_be_encoded_are_rejected(epoch: object) -> None:
	with pytest.raises(ValueError):
		WatermarkKey("fixture", epoch, b"public-test-material-only")  # type: ignore[arg-type]


def test_largest_encodable_vocabulary_derives_a_seed() -> None:
	largest = WatermarkProfile(gamma=0.25, delta=1.0, context_width=1, vocab_size=2**32 - 1)
	assert context_seed(KEY, largest, (2**32 - 2,)) >= 0


def test_green_probability_reports_the_rounded_threshold() -> None:
	tiny = WatermarkProfile(gamma=2.0**-64, delta=1.0, context_width=1, vocab_size=2)
	assert tiny.threshold == 1
	assert tiny.green_probability == 2.0**-64


@pytest.mark.parametrize("token", [1.9, True, "1"])
def test_token_ids_must_be_exact_integers(token: object) -> None:
	with pytest.raises(ValueError):
		token_id(token, PROFILE)  # type: ignore[arg-type]
