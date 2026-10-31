from pathlib import Path

import numpy as np
import pytest
from harness.store import MemoryStore, Reservation, SQLiteStore


@pytest.mark.parametrize("durable", [False, True])
def test_outstanding_reservations_survive_rollover_and_late_release_is_rejected(
	tmp_path: Path, durable: bool
) -> None:
	store = SQLiteStore(tmp_path / "state.db", "scope") if durable else MemoryStore()
	row = Reservation(
		"a",
		0,
		1,
		"digest",
		0,
		reserved_tokens=16,
		organization="org",
		organization_horizon=0,
		expires_at=5,
	)
	with store.transaction() as tx:
		assert tx.reserve(row)
	if isinstance(store, SQLiteStore):
		store.close()
		store = SQLiteStore(tmp_path / "state.db", "scope")
	with store.transaction() as tx:
		assert tx.exposure("account", "a", 10, 10).tokens == 16
		assert tx.exposure("organization", "org", 10, 10).tokens == 16
		assert tx.exposure("account", "a", 10, 10).in_flight == 0
		assert not tx.settle(row.key, "released", 10, 1)
		assert tx.settle(row.key, "failed", 10)
		assert not tx.settle(row.key, "released", 10, 1)


def test_sqlite_state_and_aggregate_share_connections_and_reject_wrong_scope(
	tmp_path: Path,
) -> None:
	path = tmp_path / "state.db"
	first = SQLiteStore(path, "scope")
	second = SQLiteStore(path, "scope")
	row = Reservation("a", 0, 1, "digest", 0, reserved_tokens=16)
	with first.transaction() as tx:
		assert tx.reserve(row)
		assert tx.append_aggregate(np.array([1.0, 2.0]), "a", 1) == 1
	with second.transaction() as tx:
		assert not tx.reserve(row)
		assert tx.append_aggregate(np.array([3.0, 4.0]), "b", 2) == 2
		index, records = tx.take_aggregate()
		assert index == 0 and [r[1] for r in records] == ["a", "b"]
		tx.put_aggregate_alert(index, {"alert": True})
		assert tx.settle(row.key, "released", 2, 3)
	first.close()
	second.close()
	third = SQLiteStore(path, "scope")
	with third.transaction() as tx:
		stored = tx.reservation(row.key)
		assert stored is not None and stored.output_tokens == 3
		assert not tx.settle(row.key, "released", 3, 3)
	third.close()
	import sqlite3

	with sqlite3.connect(path) as db:
		assert db.execute("SELECT alert FROM aggregate_alerts WHERE window = 0").fetchone() == (
			'{"alert": true}',
		)
	with pytest.raises(ValueError, match="another policy"):
		SQLiteStore(path, "different")
