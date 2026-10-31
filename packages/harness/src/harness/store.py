"""Enforcement state: account horizons, open challenges, and the release ledger.

Every policy decision reads and writes state inside one store transaction, so an admission,
its attempt count, and any challenge it issues commit together. `MemoryStore` keeps state in
this process, as offline replays and simulations need. `SQLiteStore` keeps it in a file, so a
restart neither renews an account's horizon nor forgets an open challenge, and separate
connections admit against the same counters one transaction at a time.

The release ledger separates admission from delivered output. The server reserves an
admission before it calls the teacher or builds noise, and settles the reservation once:
released (output made available to the client), failed (upstream error), or cancelled (the
client went away before anything was emitted). A reservation is keyed by its admission, so
an admission releases at most once. A reservation left unsettled by a crash stays outstanding
and counts as acquired, because the output may have been emitted before the crash.
"""

import base64
import json
import math
import sqlite3
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from harness.mmd import Array
from harness.pow import Challenge

Settlement = Literal["released", "failed", "cancelled"]


@dataclass
class AccountState:
	horizon_start: float
	attempts: int = 0
	level: int = 0
	statistic: float = 0.0
	windows: int = 0
	buffer: list[Array] = field(default_factory=list)
	buffer_digests: list[str] = field(default_factory=list)
	last_release: float = -math.inf
	recent: deque[float] = field(default_factory=deque)


@dataclass(frozen=True)
class Reservation:
	"""One admission's claim to release output, keyed by account, horizon, and attempt."""

	account: str
	horizon_start: float
	attempt: int
	request_digest: str
	reserved_at: float
	deceive: bool = False
	status: Literal["reserved"] | Settlement = "reserved"
	settled_at: float | None = None
	output_tokens: int | None = None

	@property
	def key(self) -> str:
		return admission_key(self.account, self.horizon_start, self.attempt)


@dataclass(frozen=True)
class Usage:
	"""Release outcomes for one account horizon. `acquired` counts outstanding reservations
	as released, since a crash after emission leaves them unsettled."""

	reserved: int
	released: int
	failed: int
	cancelled: int
	output_tokens: int

	@property
	def acquired(self) -> int:
		return self.released + self.reserved


def admission_key(account: str, horizon_start: float, attempt: int) -> str:
	return json.dumps([account, horizon_start, attempt])


class Transaction(Protocol):
	def account(self, name: str) -> AccountState | None: ...
	def put_account(self, name: str, state: AccountState) -> None: ...
	def challenge(self, nonce: str) -> tuple[Challenge, dict[str, Any]] | None: ...
	def put_challenge(self, challenge: Challenge, held: dict[str, Any]) -> None: ...
	def delete_challenge(self, nonce: str) -> None: ...
	def expire_challenges(self, now: float) -> None: ...
	def reserve(self, reservation: Reservation) -> bool: ...
	def settle(
		self, key: str, status: Settlement, now: float, output_tokens: int | None = None
	) -> bool: ...
	def reservation(self, key: str) -> Reservation | None: ...
	def usage(self, account: str, horizon_start: float) -> Usage: ...


class StateStore(Protocol):
	def transaction(self) -> AbstractContextManager[Transaction]:
		"""A Transaction that commits on exit; SQLite rolls back if the block raises."""
		...


def _usage(rows: list[Reservation]) -> Usage:
	count = {status: 0 for status in ("reserved", "released", "failed", "cancelled")}
	for row in rows:
		count[row.status] += 1
	return Usage(
		reserved=count["reserved"],
		released=count["released"],
		failed=count["failed"],
		cancelled=count["cancelled"],
		output_tokens=sum(row.output_tokens or 0 for row in rows if row.status == "released"),
	)


def _settled(
	row: Reservation, status: Settlement, now: float, output_tokens: int | None
) -> Reservation:
	return Reservation(
		**(asdict(row) | {"status": status, "settled_at": now, "output_tokens": output_tokens})
	)


class _MemoryTransaction:
	def __init__(self, store: MemoryStore) -> None:
		self.store = store

	def account(self, name: str) -> AccountState | None:
		return self.store.accounts.get(name)

	def put_account(self, name: str, state: AccountState) -> None:
		self.store.accounts[name] = state

	def challenge(self, nonce: str) -> tuple[Challenge, dict[str, Any]] | None:
		return self.store.challenges.get(nonce)

	def put_challenge(self, challenge: Challenge, held: dict[str, Any]) -> None:
		self.store.challenges[challenge.nonce] = (challenge, held)

	def delete_challenge(self, nonce: str) -> None:
		self.store.challenges.pop(nonce, None)

	def expire_challenges(self, now: float) -> None:
		for nonce in [n for n, (c, _) in self.store.challenges.items() if c.expires_at < now]:
			del self.store.challenges[nonce]

	def reserve(self, reservation: Reservation) -> bool:
		if reservation.key in self.store.reservations:
			return False
		self.store.reservations[reservation.key] = reservation
		return True

	def settle(
		self, key: str, status: Settlement, now: float, output_tokens: int | None = None
	) -> bool:
		row = self.store.reservations.get(key)
		if row is None or row.status != "reserved":
			return False
		self.store.reservations[key] = _settled(row, status, now, output_tokens)
		return True

	def reservation(self, key: str) -> Reservation | None:
		return self.store.reservations.get(key)

	def usage(self, account: str, horizon_start: float) -> Usage:
		return _usage(
			[
				row
				for row in self.store.reservations.values()
				if (row.account, row.horizon_start) == (account, horizon_start)
			]
		)


class MemoryStore:
	"""State in this process. A restart loses it; use it for replays and simulations."""

	def __init__(self) -> None:
		self.accounts: dict[str, AccountState] = {}
		self.challenges: dict[str, tuple[Challenge, dict[str, Any]]] = {}
		self.reservations: dict[str, Reservation] = {}
		self._lock = threading.RLock()

	@contextmanager
	def transaction(self) -> Iterator[_MemoryTransaction]:
		with self._lock:
			yield _MemoryTransaction(self)


def _array(values: Array) -> str:
	return base64.b64encode(np.asarray(values, dtype=np.float64).tobytes()).decode()


def _encode_state(state: AccountState) -> str:
	return json.dumps(
		{
			"horizon_start": state.horizon_start,
			"attempts": state.attempts,
			"level": state.level,
			"statistic": state.statistic,
			"windows": state.windows,
			# Embeddings round-trip exactly as float64 bytes, so a resumed window scores the
			# same as an uninterrupted one.
			"buffer": [_array(row) for row in state.buffer],
			"buffer_digests": state.buffer_digests,
			"last_release": state.last_release,
			"recent": list(state.recent),
		}
	)


def _decode_state(text: str) -> AccountState:
	data = json.loads(text)
	return AccountState(
		horizon_start=data["horizon_start"],
		attempts=data["attempts"],
		level=data["level"],
		statistic=data["statistic"],
		windows=data["windows"],
		buffer=[np.frombuffer(base64.b64decode(row), dtype=np.float64) for row in data["buffer"]],
		buffer_digests=data["buffer_digests"],
		last_release=data["last_release"],
		recent=deque(data["recent"]),
	)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS accounts (name TEXT PRIMARY KEY, state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS challenges (
	nonce TEXT PRIMARY KEY, expires_at REAL NOT NULL, challenge TEXT NOT NULL, held TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS challenges_expiry ON challenges (expires_at);
CREATE TABLE IF NOT EXISTS reservations (
	key TEXT PRIMARY KEY,
	account TEXT NOT NULL,
	horizon_start REAL NOT NULL,
	attempt INTEGER NOT NULL,
	request_digest TEXT NOT NULL,
	reserved_at REAL NOT NULL,
	deceive INTEGER NOT NULL,
	status TEXT NOT NULL,
	settled_at REAL,
	output_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS reservations_horizon ON reservations (account, horizon_start);
"""

_RESERVATION_COLUMNS = (
	"account, horizon_start, attempt, request_digest, reserved_at, deceive, status, "
	"settled_at, output_tokens"
)


def _reservation(row: tuple[Any, ...]) -> Reservation:
	account, horizon, attempt, digest, reserved_at, deceive, status, settled_at, tokens = row
	return Reservation(
		account=account,
		horizon_start=horizon,
		attempt=attempt,
		request_digest=digest,
		reserved_at=reserved_at,
		deceive=bool(deceive),
		status=status,
		settled_at=settled_at,
		output_tokens=tokens,
	)


class _SQLiteTransaction:
	def __init__(self, connection: sqlite3.Connection) -> None:
		self.db = connection

	def account(self, name: str) -> AccountState | None:
		row = self.db.execute("SELECT state FROM accounts WHERE name = ?", (name,)).fetchone()
		return None if row is None else _decode_state(row[0])

	def put_account(self, name: str, state: AccountState) -> None:
		self.db.execute(
			"INSERT INTO accounts (name, state) VALUES (?, ?) "
			"ON CONFLICT (name) DO UPDATE SET state = excluded.state",
			(name, _encode_state(state)),
		)

	def challenge(self, nonce: str) -> tuple[Challenge, dict[str, Any]] | None:
		row = self.db.execute(
			"SELECT challenge, held FROM challenges WHERE nonce = ?", (nonce,)
		).fetchone()
		if row is None:
			return None
		return Challenge(**json.loads(row[0])), json.loads(row[1])

	def put_challenge(self, challenge: Challenge, held: dict[str, Any]) -> None:
		self.db.execute(
			"INSERT INTO challenges (nonce, expires_at, challenge, held) VALUES (?, ?, ?, ?)",
			(
				challenge.nonce,
				challenge.expires_at,
				json.dumps(asdict(challenge)),
				json.dumps(held),
			),
		)

	def delete_challenge(self, nonce: str) -> None:
		self.db.execute("DELETE FROM challenges WHERE nonce = ?", (nonce,))

	def expire_challenges(self, now: float) -> None:
		self.db.execute("DELETE FROM challenges WHERE expires_at < ?", (now,))

	def reserve(self, reservation: Reservation) -> bool:
		cursor = self.db.execute(
			f"INSERT INTO reservations (key, {_RESERVATION_COLUMNS}) "
			"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (key) DO NOTHING",
			(
				reservation.key,
				reservation.account,
				reservation.horizon_start,
				reservation.attempt,
				reservation.request_digest,
				reservation.reserved_at,
				int(reservation.deceive),
				reservation.status,
				reservation.settled_at,
				reservation.output_tokens,
			),
		)
		return cursor.rowcount == 1

	def settle(
		self, key: str, status: Settlement, now: float, output_tokens: int | None = None
	) -> bool:
		cursor = self.db.execute(
			"UPDATE reservations SET status = ?, settled_at = ?, output_tokens = ? "
			"WHERE key = ? AND status = 'reserved'",
			(status, now, output_tokens, key),
		)
		return cursor.rowcount == 1

	def reservation(self, key: str) -> Reservation | None:
		row = self.db.execute(
			f"SELECT {_RESERVATION_COLUMNS} FROM reservations WHERE key = ?", (key,)
		).fetchone()
		return None if row is None else _reservation(row)

	def usage(self, account: str, horizon_start: float) -> Usage:
		rows = self.db.execute(
			f"SELECT {_RESERVATION_COLUMNS} FROM reservations "
			"WHERE account = ? AND horizon_start = ?",
			(account, horizon_start),
		).fetchall()
		return _usage([_reservation(row) for row in rows])


class SQLiteStore:
	"""State in a SQLite file that survives restarts.

	Each transaction takes SQLite's write lock before reading (BEGIN IMMEDIATE), so admissions
	from separate connections or processes are serialized against the same counters. `scope`
	names the policy and calibration the state belongs to; opening a file written under
	another scope fails, so a new arm cannot inherit an old arm's counters.
	"""

	def __init__(self, path: Path, scope: str, *, timeout_seconds: float = 30.0) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		self.path = path
		self._db = sqlite3.connect(
			path, isolation_level=None, check_same_thread=False, timeout=timeout_seconds
		)
		self._lock = threading.Lock()
		self._db.execute("PRAGMA journal_mode = WAL")
		self._db.execute("PRAGMA synchronous = FULL")
		self._db.executescript(_SCHEMA)
		with self.transaction() as tx:
			row = tx.db.execute("SELECT value FROM meta WHERE key = 'scope'").fetchone()
			if row is None:
				tx.db.execute("INSERT INTO meta (key, value) VALUES ('scope', ?)", (scope,))
		if row is not None and row[0] != scope:
			self._db.close()
			raise ValueError(
				f"{path} holds state for another policy or calibration; "
				"use a new state file for each arm"
			)

	@contextmanager
	def transaction(self) -> Iterator[_SQLiteTransaction]:
		with self._lock:
			self._db.execute("BEGIN IMMEDIATE")
			try:
				yield _SQLiteTransaction(self._db)
			except BaseException:
				self._db.execute("ROLLBACK")
				raise
			self._db.execute("COMMIT")

	def close(self) -> None:
		self._db.close()
