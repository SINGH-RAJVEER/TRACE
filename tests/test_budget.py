from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from harness.budget import BudgetLimits
from harness.store import MemoryStore, Reservation, Settlement, SQLiteStore, StateStore

DAY = 86_400.0
WEEK = 7 * DAY


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StateStore]:
	if request.param == "memory":
		yield MemoryStore()
		return
	durable = SQLiteStore(tmp_path / "state.sqlite3", "scope")
	yield durable
	durable.close()


class Ledger:
	"""One account's claims on a store, numbered as a server would number its admissions."""

	def __init__(
		self, store: StateStore, limits: BudgetLimits, session: str = "s", horizon: float = 0.0
	) -> None:
		self.store, self.limits, self.session = store, limits, session
		self.horizon = horizon
		self.attempts = 0

	def claim(self, tokens: int, now: float, key: str | None, digest: str) -> Reservation:
		self.attempts += 1
		return Reservation(
			account="a",
			horizon_start=self.horizon,
			attempt=self.attempts,
			request_digest=digest,
			reserved_at=now,
			max_output_tokens=tokens,
			idempotency_key=key,
			session=self.session,
		)

	def reserve(
		self, tokens: int, now: float, key: str | None = None, digest: str = "digest"
	) -> str | None:
		"""The reservation's key, or None after recording why it was refused."""
		claim = self.claim(tokens, now, key, digest)
		with self.store.transaction() as tx:
			self.refusal = self.limits.refusal(tx, claim)
			if self.refusal is not None:
				return None
			assert tx.reserve(claim)
		return claim.key

	def settle(self, key: str | None, status: Settlement, tokens: int | None = None) -> None:
		assert key is not None
		with self.store.transaction() as tx:
			assert tx.settle(key, status, 0.0, tokens)

	def refused(self, tokens: int, now: float, key: str | None = None, digest: str = "d") -> str:
		assert self.reserve(tokens, now, key, digest) is None
		assert self.refusal is not None
		return self.refusal

	def usage(self, now: float = 0.0) -> tuple[int, int, int]:
		with self.store.transaction() as tx:
			usage = tx.budget_usage("a", now - self.limits.window_seconds, self.session)
		return usage.requests, usage.output_tokens, usage.in_flight


def test_settling_charges_used_tokens_and_returns_the_rest(store: StateStore) -> None:
	ledger = Ledger(store, BudgetLimits(WEEK, output_tokens=100))
	first = ledger.reserve(100, 0.0)
	assert ledger.usage() == (1, 100, 1)
	assert ledger.refused(1, 0.0) == "budget_tokens"
	ledger.settle(first, "released", 30)
	assert ledger.usage() == (1, 30, 0)
	assert ledger.reserve(70, 0.0) is not None
	assert ledger.refused(1, 0.0) == "budget_tokens"


def test_unknown_usage_charges_everything_and_nothing_released_charges_nothing(
	store: StateStore,
) -> None:
	ledger = Ledger(store, BudgetLimits(WEEK))
	ledger.settle(ledger.reserve(50, 0.0), "released", None)
	ledger.settle(ledger.reserve(50, 0.0), "failed")
	ledger.settle(ledger.reserve(50, 0.0), "cancelled")
	assert ledger.usage() == (1, 50, 0)


def test_noise_draws_nothing_from_the_budget(store: StateStore) -> None:
	ledger = Ledger(store, BudgetLimits(WEEK, requests=1))
	noise = Reservation(
		account="a",
		horizon_start=0.0,
		attempt=99,
		request_digest="digest",
		reserved_at=0.0,
		deceive=True,
		max_output_tokens=10,
		session="s",
	)
	with store.transaction() as tx:
		assert tx.reserve(noise)
		assert tx.settle(noise.key, "released", 0.0, None)
	assert ledger.usage() == (0, 0, 0)


def test_request_and_concurrency_limits_are_separate(store: StateStore) -> None:
	ledger = Ledger(store, BudgetLimits(WEEK, requests=3, concurrency=1))
	first = ledger.reserve(1, 0.0)
	assert ledger.refused(1, 0.0) == "budget_concurrency"
	ledger.settle(first, "released", 1)
	ledger.settle(ledger.reserve(1, 0.0), "released", 1)
	ledger.settle(ledger.reserve(1, 0.0), "released", 1)
	assert ledger.refused(1, 0.0) == "budget_requests"


def test_only_the_rolling_window_frees_a_budget(store: StateStore) -> None:
	ledger = Ledger(store, BudgetLimits(WEEK, requests=2))
	ledger.settle(ledger.reserve(1, 0.0), "released", 1)
	ledger.settle(ledger.reserve(1, DAY), "released", 1)
	assert ledger.refused(1, 2 * DAY) == "budget_requests"
	assert ledger.refused(1, WEEK - 1) == "budget_requests"
	# Each request leaves on its own anniversary, not all at once.
	ledger.settle(ledger.reserve(1, WEEK + 1), "released", 1)
	assert ledger.refused(1, WEEK + 2) == "budget_requests"


def test_a_crash_keeps_the_charge_and_frees_the_in_flight_slot(tmp_path: Path) -> None:
	path = tmp_path / "state.sqlite3"
	limits = BudgetLimits(WEEK, output_tokens=100, concurrency=1)
	first = SQLiteStore(path, "scope")
	before = Ledger(first, limits, "before")
	before.settle(before.reserve(40, 0.0), "released", 10)
	assert before.reserve(60, 0.0) is not None
	first.close()

	restarted = SQLiteStore(path, "scope")
	after = Ledger(restarted, limits, "after", horizon=1.0)
	with restarted.transaction() as tx:
		assert tx.outstanding() == 1
	assert after.usage() == (2, 70, 0)
	assert after.refused(31, 1.0) == "budget_tokens"
	assert after.reserve(30, 1.0) is not None
	restarted.close()


def test_simultaneous_requests_cannot_overspend(tmp_path: Path) -> None:
	path = tmp_path / "state.sqlite3"
	SQLiteStore(path, "scope").close()

	limits = BudgetLimits(WEEK, requests=10)

	def attempt(index: int) -> bool:
		connection = SQLiteStore(path, "scope")
		try:
			return (
				Ledger(connection, limits, f"s{index}", horizon=float(index)).reserve(1, 0.0)
				is not None
			)
		finally:
			connection.close()

	with ThreadPoolExecutor(max_workers=16) as pool:
		accepted = sum(pool.map(attempt, range(64)))
	assert accepted == 10


def test_an_idempotency_key_cannot_collect_a_second_answer(store: StateStore) -> None:
	ledger = Ledger(store, BudgetLimits(WEEK))
	failed = ledger.reserve(1, 0.0, "key", "digest")
	assert ledger.refused(1, 0.0, "key", "digest") == "duplicate_request"
	ledger.settle(failed, "failed")
	ledger.settle(ledger.reserve(1, 0.0, "key", "digest"), "released", 1)
	assert ledger.refused(1, 0.0, "key", "digest") == "duplicate_request"
	assert ledger.refused(1, 0.0, "key", "other") == "key_conflict"
