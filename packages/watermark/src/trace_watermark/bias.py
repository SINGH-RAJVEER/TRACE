"""Generation-side soft KGW bias, independent of any serving engine.

The step order for one request is apply, sample, then record_token. Apply can repeat for a retry:
it uses the current history and counts another row, so only record_token advances a request. Nothing
enforces the order, so compare the row counts with tokens_recorded in RequestStatus.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trace_watermark.core import (
	WatermarkKey,
	WatermarkProfile,
	context_of,
	context_seed,
	green_mask,
	token_id,
)

FloatArray = NDArray[np.floating[Any]]


def _check_logits(logits: FloatArray) -> None:
	if not np.issubdtype(logits.dtype, np.floating):
		raise ValueError("logits must have a floating dtype")


def apply_bias(
	logits: FloatArray,
	history: Sequence[int],
	key: WatermarkKey,
	profile: WatermarkProfile,
) -> FloatArray:
	"""Return a copy of one logits row with delta added to green tokens when context exists."""
	if logits.shape != (profile.vocab_size,):
		raise ValueError(f"expected logits shape ({profile.vocab_size},), got {logits.shape}")
	_check_logits(logits)
	if np.isnan(logits).any() or np.isposinf(logits).any():
		raise ValueError("logits cannot hold NaN or positive infinity")
	biased = logits.copy()
	for token in history:
		token_id(token, profile)
	context = context_of(history, profile.context_width)
	if context is None:
		return biased
	green = green_mask(context_seed(key, profile, context), profile)
	# Hard masks stay at negative infinity, and overflow is an error rather than a new infinity.
	update = green & np.isfinite(logits)
	with np.errstate(over="ignore"):
		shifted = logits[update] + np.asarray(profile.delta, dtype=logits.dtype)
	if not np.isfinite(shifted).all():
		raise ValueError("delta overflows this logits dtype")
	biased[update] = shifted
	return biased


@dataclass(frozen=True)
class RequestStatus:
	"""Watermark application counts for one request. Holds no secret."""

	request_id: str
	key_id: str
	key_epoch: int
	enabled: bool
	rows_biased: int
	rows_skipped_no_context: int
	rows_disabled: int
	tokens_recorded: int


@dataclass
class _RequestState:
	request_id: str
	key: WatermarkKey
	enabled: bool
	history: list[int]
	rows_biased: int = 0
	rows_skipped_no_context: int = 0
	rows_disabled: int = 0
	tokens_recorded: int = 0

	def snapshot(self) -> RequestStatus:
		return RequestStatus(
			request_id=self.request_id,
			key_id=self.key.key_id,
			key_epoch=self.key.epoch,
			enabled=self.enabled,
			rows_biased=self.rows_biased,
			rows_skipped_no_context=self.rows_skipped_no_context,
			rows_disabled=self.rows_disabled,
			tokens_recorded=self.tokens_recorded,
		)


class KgwBatchProcessor:
	"""Per-request bias state keyed by slot. All requests share one profile."""

	def __init__(self, profile: WatermarkProfile) -> None:
		self.profile = profile
		self._requests: dict[int, _RequestState] = {}

	def _request(self, slot: int) -> _RequestState:
		try:
			return self._requests[slot]
		except KeyError:
			raise ValueError(f"slot {slot} is empty") from None

	def add_request(
		self,
		slot: int,
		request_id: str,
		prompt_ids: Iterable[int],
		key: WatermarkKey,
		enabled: bool = True,
	) -> None:
		if slot in self._requests:
			raise ValueError(f"slot {slot} is occupied")
		history = [token_id(token, self.profile) for token in prompt_ids]
		del history[: -self.profile.context_width]
		self._requests[slot] = _RequestState(request_id, key, enabled, history)

	def remove_request(self, slot: int) -> RequestStatus:
		"""Free a slot and return the request's final status."""
		status = self._request(slot).snapshot()
		del self._requests[slot]
		return status

	def move_request(self, src: int, dst: int) -> None:
		state = self._request(src)
		if src == dst:
			return
		if dst in self._requests:
			raise ValueError(f"slot {dst} is occupied")
		self._requests[dst] = state
		del self._requests[src]

	def record_token(self, slot: int, token: int) -> None:
		state = self._request(slot)
		state.history.append(token_id(token, self.profile))
		del state.history[: -self.profile.context_width]
		state.tokens_recorded += 1

	def status(self, slot: int) -> RequestStatus:
		return self._request(slot).snapshot()

	def apply(self, logits: FloatArray, row_slots: Sequence[int]) -> FloatArray:
		"""Return a biased copy of a (rows, vocab) array. A rejected call changes no counts."""
		if logits.ndim != 2 or logits.shape[1] != self.profile.vocab_size:
			raise ValueError(
				f"expected logits shape (rows, {self.profile.vocab_size}), got {logits.shape}"
			)
		_check_logits(logits)
		if len(row_slots) != logits.shape[0]:
			raise ValueError("row_slots must name one slot per logits row")
		if len(set(row_slots)) != len(row_slots):
			raise ValueError("row_slots names a slot twice")
		states = [self._request(slot) for slot in row_slots]

		output = logits.copy()
		outcomes: list[str] = []
		for row, state in enumerate(states):
			if not state.enabled:
				outcomes.append("rows_disabled")
				continue
			output[row] = apply_bias(logits[row], state.history, state.key, self.profile)
			has_context = context_of(state.history, self.profile.context_width) is not None
			outcomes.append("rows_biased" if has_context else "rows_skipped_no_context")
		# Counts change only once every row has succeeded, so a caller can retry a rejected batch.
		for state, outcome in zip(states, outcomes, strict=True):
			setattr(state, outcome, getattr(state, outcome) + 1)
		return output
