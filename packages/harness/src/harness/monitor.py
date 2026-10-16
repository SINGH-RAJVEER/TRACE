"""Aggregate monitoring across all accounts (paper Section III-B).

Every admitted request joins one stream. Each complete window of aggregate_window_size
requests is scored against aggregate reference windows, and a score above the calibrated
threshold raises an alert. Alerts never change any account's level: an aggregate shift does
not identify the accounts responsible, which matters most for Sybil campaigns.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from harness.calibration import AggregateCalibration
from harness.mmd import Array, WindowScorer


@dataclass(frozen=True)
class AggregateWindow:
	"""A complete aggregate window waiting to be scored."""

	index: int
	embeddings: Array
	accounts: list[str]
	started_at: float
	ended_at: float


@dataclass(frozen=True)
class AggregateAlert:
	"""The outcome of one aggregate window. Composition is for the experimenter only."""

	index: int
	score: float
	alert: bool
	accounts: int
	top_account_share: float
	started_at: float
	ended_at: float


class AggregateMonitor:
	def __init__(self, scorer: WindowScorer, calibration: AggregateCalibration) -> None:
		if scorer.window_size != calibration.window_size:
			raise ValueError("aggregate references do not match the calibrated window size")
		self.scorer = scorer
		self.calibration = calibration
		self.windows = 0
		self._embeddings: list[Array] = []
		self._accounts: list[str] = []
		self._times: list[float] = []

	def add(self, embedding: Array, account: str, now: float) -> AggregateWindow | None:
		"""Append one request; return the window it completes, if any. Scoring is separate
		so a server can run it off the request path."""
		self._embeddings.append(np.asarray(embedding, dtype=np.float64))
		self._accounts.append(account)
		self._times.append(now)
		if len(self._embeddings) < self.calibration.window_size:
			return None
		window = AggregateWindow(
			index=self.windows,
			embeddings=np.stack(self._embeddings),
			accounts=self._accounts,
			started_at=self._times[0],
			ended_at=self._times[-1],
		)
		self.windows += 1
		self._embeddings, self._accounts, self._times = [], [], []
		return window

	def score(self, window: AggregateWindow) -> AggregateAlert:
		score = self.scorer.score(window.embeddings)
		counts = Counter(window.accounts)
		return AggregateAlert(
			index=window.index,
			score=score,
			alert=score > self.calibration.threshold,
			accounts=len(counts),
			top_account_share=max(counts.values()) / len(window.accounts),
			started_at=window.started_at,
			ended_at=window.ended_at,
		)
