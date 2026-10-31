"""Enforcement state: account horizons, open challenges, the release ledger, idempotency
records, and the aggregate monitor's stream.

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

A reservation also holds the output tokens it may release and, under an in-flight limit, a
lease. Its charge is the actual output tokens once released, and the reserved tokens while
outstanding or after a failure or cancellation, which are never refunded. An outstanding
reservation holds a concurrency slot until its lease ends; a live worker cannot release after
that (`settle` refuses), so a crashed reservation frees its slot without letting two workers
emit under it. It keeps its token charge in the horizon it was admitted in.
"""

import base64
import copy
import json
import math
import sqlite3
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass, field, replace
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
	# The output allowance held while outstanding: the request's max_output_tokens.
	reserved_tokens: int = 0
	# The shared organization scope the reservation also counts against, if any.
	organization: str | None = None
	organization_horizon: float | None = None
	# End of the lease: past it the reservation holds no slot and can no longer be released.
	# None never expires.
	expires_at: float | None = None

	@property
	def key(self) -> str:
		return admission_key(self.account, self.horizon_start, self.attempt)

	@property
	def charged_tokens(self) -> int:
		"""Output tokens charged: the actual count once released, else the reservation."""
		if self.status == "released" and self.output_tokens is not None:
			return self.output_tokens
		return self.reserved_tokens

	def live(self, now: float) -> bool:
		"""Outstanding and inside its lease: it holds a slot and may still release."""
		return self.status == "reserved" and (self.expires_at is None or now <= self.expires_at)


@dataclass(frozen=True)
class Usage:
	"""Release outcomes for one account horizon. `acquired` counts outstanding reservations
	as released, since a crash after emission leaves them unsettled."""

	reserved: int
	released: int
	failed: int
	cancelled: int
	output_tokens: int | None

	@property
	def acquired(self) -> int:
		return self.released + self.reserved


@dataclass(frozen=True)
class Exposure:
	"""What a scope has acquired or may still acquire in one horizon: its reservations, their
	charged output tokens, and the live reservations holding a concurrency slot. Live
	reservations from an earlier horizon are included, so a rollover does not free them."""

	requests: int
	tokens: int
	in_flight: int


@dataclass(frozen=True)
class Outcome:
	"""A stored response, returned again for a retry with the same idempotency key."""

	status_code: int
	body: str
	headers: dict[str, str]


@dataclass(frozen=True)
class IdempotencyRecord:
	"""A client key bound to one request digest. Without an outcome the request is pending."""

	account: str
	key: str
	request_digest: str
	created_at: float
	outcome: Outcome | None = None


def admission_key(account: str, horizon_start: float, attempt: int) -> str:
	return json.dumps([account, horizon_start, attempt])


def _exposure(rows: list[Reservation], now: float) -> Exposure:
	return Exposure(
		requests=len(rows),
		tokens=sum(row.charged_tokens for row in rows),
		in_flight=sum(row.live(now) for row in rows),
	)


Scope = Literal["account", "organization"]


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
	def exposure(self, scope: Scope, name: str, horizon_start: float, now: float) -> Exposure: ...
	def organization(self, name: str) -> float | None: ...
	def put_organization(self, name: str, horizon_start: float) -> None: ...
	def idempotency(self, account: str, key: str) -> IdempotencyRecord | None: ...
	def put_idempotency(self, record: IdempotencyRecord) -> None: ...
	def delete_idempotency(self, account: str, key: str) -> None: ...
	def append_aggregate(self, embedding: Array, account: str, now: float) -> int: ...
	def take_aggregate(self) -> tuple[int, list[tuple[Array, str, float]]]: ...
	def put_aggregate_alert(self, index: int, alert: dict[str, Any]) -> None: ...


class StateStore(Protocol):
	def transaction(self) -> AbstractContextManager[Transaction]:
		"""A Transaction that commits on exit; SQLite rolls back if the block raises."""
		...


def reserve_entitlement(
	tx: Transaction,
	reservation: Reservation,
	*,
	now: float,
	horizon_seconds: float,
	principal_budget: int | None,
	organization_budget: int | None,
	in_flight_limit: int | None,
) -> tuple[Reservation | None, str | None]:
	"""Check both scopes and reserve atomically inside the caller's admission transaction."""
	organization = reservation.organization
	if organization_budget is not None and organization is None:
		return None, "organization_required"
	org_horizon = None
	if organization is not None:
		org_horizon = tx.organization(organization)
		if org_horizon is None or now - org_horizon >= horizon_seconds:
			org_horizon = now
	account_usage = tx.exposure("account", reservation.account, reservation.horizon_start, now)
	org_usage = (
		tx.exposure("organization", organization, org_horizon, now)
		if organization is not None and org_horizon is not None
		else None
	)
	if (
		principal_budget is not None
		and account_usage.tokens + reservation.reserved_tokens > principal_budget
	) or (
		org_usage is not None
		and organization_budget is not None
		and org_usage.tokens + reservation.reserved_tokens > organization_budget
	):
		return None, "token_budget_exceeded"
	if in_flight_limit is not None and account_usage.in_flight >= in_flight_limit:
		return None, "in_flight_limit"
	reservation = replace(reservation, organization_horizon=org_horizon)
	if not tx.reserve(reservation):
		return None, "already_released"
	if organization is not None and org_horizon is not None:
		tx.put_organization(organization, org_horizon)
	return reservation, None


def _usage(rows: list[Reservation]) -> Usage:
	count = {status: 0 for status in ("reserved", "released", "failed", "cancelled")}
	for row in rows:
		count[row.status] += 1
	return Usage(
		reserved=count["reserved"],
		released=count["released"],
		failed=count["failed"],
		cancelled=count["cancelled"],
		output_tokens=None
		if any(row.status == "released" and row.output_tokens is None for row in rows)
		else sum(row.output_tokens or 0 for row in rows if row.status == "released"),
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
		if status == "released" and not row.live(now):
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

	def exposure(self, scope: Scope, name: str, horizon_start: float, now: float) -> Exposure:
		rows = []
		for row in self.store.reservations.values():
			owner, horizon = (
				(row.account, row.horizon_start)
				if scope == "account"
				else (row.organization, row.organization_horizon)
			)
			if owner == name and (horizon == horizon_start or row.status == "reserved"):
				rows.append(row)
		return _exposure(rows, now)

	def organization(self, name: str) -> float | None:
		return self.store.organizations.get(name)

	def put_organization(self, name: str, horizon_start: float) -> None:
		self.store.organizations[name] = horizon_start

	def idempotency(self, account: str, key: str) -> IdempotencyRecord | None:
		return self.store.idempotency.get((account, key))

	def put_idempotency(self, record: IdempotencyRecord) -> None:
		self.store.idempotency[(record.account, record.key)] = record

	def delete_idempotency(self, account: str, key: str) -> None:
		self.store.idempotency.pop((account, key), None)

	def append_aggregate(self, embedding: Array, account: str, now: float) -> int:
		self.store.aggregate.append((np.asarray(embedding, dtype=np.float64), account, now))
		return len(self.store.aggregate)

	def take_aggregate(self) -> tuple[int, list[tuple[Array, str, float]]]:
		rows, self.store.aggregate = self.store.aggregate, []
		self.store.aggregate_windows += 1
		return self.store.aggregate_windows - 1, rows

	def put_aggregate_alert(self, index: int, alert: dict[str, Any]) -> None:
		self.store.aggregate_alerts[index] = alert


class MemoryStore:
	"""State in this process. A restart loses it; use it for replays and simulations."""

	def __init__(self) -> None:
		self.accounts: dict[str, AccountState] = {}
		self.challenges: dict[str, tuple[Challenge, dict[str, Any]]] = {}
		self.reservations: dict[str, Reservation] = {}
		self.organizations: dict[str, float] = {}
		self.idempotency: dict[tuple[str, str], IdempotencyRecord] = {}
		self.aggregate: list[tuple[Array, str, float]] = []
		self.aggregate_windows = 0
		self.aggregate_alerts: dict[int, dict[str, Any]] = {}
		self._lock = threading.RLock()

	@contextmanager
	def transaction(self) -> Iterator[_MemoryTransaction]:
		with self._lock:
			snapshot = copy.deepcopy(
				(
					self.accounts,
					self.challenges,
					self.reservations,
					self.organizations,
					self.idempotency,
					self.aggregate,
					self.aggregate_windows,
					self.aggregate_alerts,
				)
			)
			try:
				yield _MemoryTransaction(self)
			except BaseException:
				(
					self.accounts,
					self.challenges,
					self.reservations,
					self.organizations,
					self.idempotency,
					self.aggregate,
					self.aggregate_windows,
					self.aggregate_alerts,
				) = snapshot
				raise


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
CREATE TABLE IF NOT EXISTS organizations (name TEXT PRIMARY KEY, horizon_start REAL NOT NULL);
CREATE TABLE IF NOT EXISTS idempotency (
	account TEXT NOT NULL,
	key TEXT NOT NULL,
	request_digest TEXT NOT NULL,
	created_at REAL NOT NULL,
	outcome TEXT,
	PRIMARY KEY (account, key)
);
CREATE TABLE IF NOT EXISTS aggregate (
	seq INTEGER PRIMARY KEY AUTOINCREMENT,
	embedding TEXT NOT NULL,
	account TEXT NOT NULL,
	time REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS aggregate_alerts (window INTEGER PRIMARY KEY, alert TEXT NOT NULL);
"""

# Reservation columns added after the first durable ledger; older files gain them on open.
_ADDED_COLUMNS = {
	"reserved_tokens": "INTEGER NOT NULL DEFAULT 0",
	"organization": "TEXT",
	"organization_horizon": "REAL",
	"expires_at": "REAL",
}

_RESERVATION_COLUMNS = (
	"account, horizon_start, attempt, request_digest, reserved_at, deceive, status, "
	"settled_at, output_tokens, reserved_tokens, organization, organization_horizon, expires_at"
)


def _reservation(row: tuple[Any, ...]) -> Reservation:
	(
		account,
		horizon,
		attempt,
		digest,
		reserved_at,
		deceive,
		status,
		settled_at,
		tokens,
		reserved_tokens,
		organization,
		organization_horizon,
		expires_at,
	) = row
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
		reserved_tokens=reserved_tokens,
		organization=organization,
		organization_horizon=organization_horizon,
		expires_at=expires_at,
	)


def _outcome(text: str | None) -> Outcome | None:
	return None if text is None else Outcome(**json.loads(text))


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
			"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (key) DO NOTHING",
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
				reservation.reserved_tokens,
				reservation.organization,
				reservation.organization_horizon,
				reservation.expires_at,
			),
		)
		return cursor.rowcount == 1

	def settle(
		self, key: str, status: Settlement, now: float, output_tokens: int | None = None
	) -> bool:
		# A release must land inside the lease; failures and cancellations may settle late.
		cursor = self.db.execute(
			"UPDATE reservations SET status = ?, settled_at = ?, output_tokens = ? "
			"WHERE key = ? AND status = 'reserved' "
			"AND (? != 'released' OR expires_at IS NULL OR ? <= expires_at)",
			(status, now, output_tokens, key, status, now),
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

	def exposure(self, scope: Scope, name: str, horizon_start: float, now: float) -> Exposure:
		owner, horizon = (
			("account", "horizon_start")
			if scope == "account"
			else ("organization", "organization_horizon")
		)
		rows = self.db.execute(
			f"SELECT {_RESERVATION_COLUMNS} FROM reservations WHERE {owner} = ? AND "
			f"({horizon} = ? OR status = 'reserved')",
			(name, horizon_start),
		).fetchall()
		return _exposure([_reservation(row) for row in rows], now)

	def organization(self, name: str) -> float | None:
		row = self.db.execute(
			"SELECT horizon_start FROM organizations WHERE name = ?", (name,)
		).fetchone()
		return None if row is None else row[0]

	def put_organization(self, name: str, horizon_start: float) -> None:
		self.db.execute(
			"INSERT INTO organizations (name, horizon_start) VALUES (?, ?) "
			"ON CONFLICT (name) DO UPDATE SET horizon_start = excluded.horizon_start",
			(name, horizon_start),
		)

	def idempotency(self, account: str, key: str) -> IdempotencyRecord | None:
		row = self.db.execute(
			"SELECT request_digest, created_at, outcome FROM idempotency "
			"WHERE account = ? AND key = ?",
			(account, key),
		).fetchone()
		if row is None:
			return None
		return IdempotencyRecord(account, key, row[0], row[1], _outcome(row[2]))

	def put_idempotency(self, record: IdempotencyRecord) -> None:
		self.db.execute(
			"INSERT INTO idempotency (account, key, request_digest, created_at, outcome) "
			"VALUES (?, ?, ?, ?, ?) ON CONFLICT (account, key) DO UPDATE SET "
			"request_digest = excluded.request_digest, created_at = excluded.created_at, "
			"outcome = excluded.outcome",
			(
				record.account,
				record.key,
				record.request_digest,
				record.created_at,
				None if record.outcome is None else json.dumps(asdict(record.outcome)),
			),
		)

	def delete_idempotency(self, account: str, key: str) -> None:
		self.db.execute("DELETE FROM idempotency WHERE account = ? AND key = ?", (account, key))

	def append_aggregate(self, embedding: Array, account: str, now: float) -> int:
		self.db.execute(
			"INSERT INTO aggregate (embedding, account, time) VALUES (?, ?, ?)",
			(_array(embedding), account, now),
		)
		return int(self.db.execute("SELECT COUNT(*) FROM aggregate").fetchone()[0])

	def take_aggregate(self) -> tuple[int, list[tuple[Array, str, float]]]:
		rows = self.db.execute(
			"SELECT embedding, account, time FROM aggregate ORDER BY seq"
		).fetchall()
		self.db.execute("DELETE FROM aggregate")
		found = self.db.execute("SELECT value FROM meta WHERE key = 'aggregate_windows'").fetchone()
		index = 0 if found is None else int(found[0])
		self.db.execute(
			"INSERT INTO meta (key, value) VALUES ('aggregate_windows', ?) "
			"ON CONFLICT (key) DO UPDATE SET value = excluded.value",
			(str(index + 1),),
		)
		return index, [
			(np.frombuffer(base64.b64decode(embedding), dtype=np.float64), account, time)
			for embedding, account, time in rows
		]

	def put_aggregate_alert(self, index: int, alert: dict[str, Any]) -> None:
		self.db.execute(
			"INSERT INTO aggregate_alerts (window, alert) VALUES (?, ?)", (index, json.dumps(alert))
		)


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
			present = {row[1] for row in tx.db.execute("PRAGMA table_info(reservations)")}
			for column, kind in _ADDED_COLUMNS.items():
				if column not in present:
					tx.db.execute(f"ALTER TABLE reservations ADD COLUMN {column} {kind}")
			tx.db.execute(
				"CREATE INDEX IF NOT EXISTS reservations_organization "
				"ON reservations (organization, organization_horizon)"
			)
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
