from pathlib import Path

import pytest
from harness.policy import QuotaPolicy, RatePolicy
from harness.store import (
	MemoryStore,
	Reservation,
	RollingBudget,
	SQLiteStore,
	StateStore,
	reserve_entitlement,
)

DAY = 86_400.0
WEEK = 7 * DAY


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> StateStore:
	if request.param == "memory":
		return MemoryStore()
	return SQLiteStore(tmp_path / "state.sqlite3", "scope")


def _admit(
	store: StateStore, budget: RollingBudget, attempt: int, tokens: int, now: float
) -> tuple[Reservation | None, str | None]:
	row = Reservation("a", now, attempt, "digest", now, reserved_tokens=tokens)
	with store.transaction() as tx:
		return reserve_entitlement(
			tx,
			row,
			now=now,
			horizon_seconds=DAY,
			principal_budget=None,
			organization_budget=None,
			in_flight_limit=None,
			rolling=budget,
		)


def _release(store: StateStore, row: Reservation | None, tokens: int | None, now: float) -> None:
	assert row is not None
	with store.transaction() as tx:
		assert tx.settle(row.key, "released", now, tokens)


def test_a_new_horizon_does_not_refill_a_rolling_budget(store: StateStore) -> None:
	budget = RollingBudget(WEEK, requests=2)
	_release(store, _admit(store, budget, 1, 8, 0.0)[0], 8, 0.0)
	# Each admission is in a later horizon, as a daily renewal would make it.
	_release(store, _admit(store, budget, 1, 8, DAY)[0], 8, DAY)
	assert _admit(store, budget, 1, 8, 2 * DAY) == (None, "rolling_budget_exceeded")
	assert _admit(store, budget, 1, 8, WEEK - 1) == (None, "rolling_budget_exceeded")
	# Each request leaves on its own anniversary, not all at once.
	assert _admit(store, budget, 1, 8, WEEK + 1)[0] is not None
	assert _admit(store, budget, 2, 8, WEEK + 2) == (None, "rolling_budget_exceeded")


def test_rolling_tokens_use_the_ledger_charges(store: StateStore) -> None:
	budget = RollingBudget(WEEK, tokens=100)
	first, _ = _admit(store, budget, 1, 100, 0.0)
	assert _admit(store, budget, 2, 1, 0.0) == (None, "rolling_budget_exceeded")
	_release(store, first, 30, 0.0)
	failed, _ = _admit(store, budget, 3, 20, 0.0)
	assert failed is not None
	with store.transaction() as tx:
		assert tx.settle(failed.key, "failed", 0.0)
		exposure = tx.rolling_exposure("a", -1.0)
	# A failed admission keeps its reservation, as in the horizon budget.
	assert (exposure.requests, exposure.tokens) == (2, 50)
	assert _admit(store, budget, 4, 50, 0.0)[0] is not None
	assert _admit(store, budget, 5, 1, 0.0) == (None, "rolling_budget_exceeded")


def test_noise_draws_nothing_from_a_rolling_budget(store: StateStore) -> None:
	noise = Reservation("a", 0.0, 1, "digest", 0.0, deceive=True, reserved_tokens=10)
	with store.transaction() as tx:
		assert tx.reserve(noise)
		assert tx.settle(noise.key, "released", 0.0, None)
		exposure = tx.rolling_exposure("a", -1.0)
	assert (exposure.requests, exposure.tokens) == (0, 0)


def test_every_restriction_names_its_cause() -> None:
	assert QuotaPolicy(1, 100, 10).admit("a", "q", 0).reason is None
	quota = QuotaPolicy(1, 100, 10)
	quota.admit("a", "q", 0)
	assert quota.admit("a", "q", 1).reason == "quota"
	capped = QuotaPolicy(100, 100, 1)
	capped.admit("a", "q", 0)
	assert capped.admit("a", "q", 1).reason == "attempt_cap"
	rate = RatePolicy(1, 100, 10)
	rate.admit("a", "q", 0)
	assert rate.admit("a", "q", 1).reason == "rate"
