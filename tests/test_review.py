import sqlite3
from pathlib import Path

import pytest
from harness import review
from harness.store import Reservation, RollingBudget, SQLiteStore, reserve_entitlement


def test_credit_lifecycle_and_append_only(tmp_path: Path) -> None:
	store = SQLiteStore(tmp_path / "state.db", "scope", review_enabled=True)
	with store.transaction() as tx:
		ref = review.open_appeal(
			tx.db,
			account="a",
			organization=None,
			code="token_budget_exceeded",
			scope="account",
			reset_at=100,
			horizon=0,
			window_seconds=None,
			now=1,
			digest="digest",
		)
		review.decide(
			tx.db,
			ref,
			action="restore",
			operator="op",
			reason="correction",
			tokens=5,
			expires_at=100,
			now=2,
		)
		with pytest.raises(sqlite3.IntegrityError):
			tx.db.execute("DELETE FROM review_decisions")
		assert len(review.show(tx.db, ref)["decisions"]) == 1  # type: ignore[index]
	for attempt, tokens, expected in [(1, 10, True), (2, 5, True), (3, 1, False)]:
		with store.transaction() as tx:
			row, _ = reserve_entitlement(
				tx,
				Reservation("a", 0, attempt, "x", 3, reserved_tokens=tokens),
				now=3,
				horizon_seconds=100,
				principal_budget=10,
				organization_budget=None,
				in_flight_limit=None,
			)
			assert (row is not None) == expected
	with store.transaction() as tx:
		review.decide(tx.db, ref, action="reject", operator="op", reason="superseded", now=4)
		assert tx.credits("account", "a", "horizon", 4, 0, None).tokens == 0
		assert len(review.show(tx.db, ref)["decisions"]) == 2  # type: ignore[index]


def test_rolling_credit_and_unlink_preserve_old_charges(tmp_path: Path) -> None:
	store = SQLiteStore(tmp_path / "state.db", "scope", review_enabled=True)
	budget = RollingBudget(100, requests=1, tokens=10, scope="organization")

	def admit(attempt: int) -> Reservation | None:
		with store.transaction() as tx:
			row, _ = reserve_entitlement(
				tx,
				Reservation("a", 0, attempt, "x", 2, reserved_tokens=10, organization="org"),
				now=2,
				horizon_seconds=100,
				principal_budget=None,
				organization_budget=None,
				in_flight_limit=None,
				organization_rolling=budget,
			)
			return row

	assert admit(1) is not None
	assert admit(2) is None
	with store.transaction() as tx:
		ref = review.open_appeal(
			tx.db,
			account="a",
			organization="org",
			code="organization_rolling_budget_exceeded",
			scope="organization",
			reset_at=None,
			horizon=None,
			window_seconds=100,
			now=2,
			digest="x",
		)
		review.decide(
			tx.db,
			ref,
			action="restore",
			operator="op",
			reason="correction",
			requests=1,
			tokens=10,
			expires_at=100,
			now=2,
		)
	assert admit(3) is not None
	assert admit(4) is None
	with store.transaction() as tx:
		review.decide(tx.db, ref, action="unlink", operator="op", reason="wrong link", now=2)
	row = admit(5)
	assert row is not None and row.organization is None
	with store.transaction() as tx:
		assert tx.rolling_exposure("org", -1, "organization").requests == 2
		assert tx.rolling_exposure("a", -1).requests == 3


def test_grant_policy_allowance_and_disabled_schema(tmp_path: Path) -> None:
	plain = SQLiteStore(tmp_path / "plain.db", "scope")
	with plain.transaction() as tx:
		assert not tx.db.execute(
			"SELECT 1 FROM sqlite_master WHERE name = 'review_appeals'"
		).fetchone()
	plain.close()
	store = SQLiteStore(tmp_path / "review.db", "scope", review_enabled=True)
	with store.transaction() as tx:
		ref = review.open_appeal(
			tx.db,
			account="a",
			organization=None,
			code="detector",
			scope="account",
			reset_at=100,
			horizon=0,
			window_seconds=None,
			now=1,
			digest="x",
		)
		review.decide(
			tx.db,
			ref,
			action="grant",
			operator="op",
			reason="verified",
			requests=1,
			expires_at=20,
			now=2,
		)
		assert tx.allowance("a", "detector", 3, 10)
		assert not tx.allowance("a", "quota", 3, 10)
		assert not tx.allowance("a", "detector", 20, 10)
		assert tx.reserve(Reservation("a", 0, 1, "x", 3, reserved_tokens=10))
		assert not tx.allowance("a", "detector", 4, 10)
