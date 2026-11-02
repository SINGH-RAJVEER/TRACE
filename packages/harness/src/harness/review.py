"""Append-only review records and operator decisions in the enforcement database."""

import json
import sqlite3
import time
import uuid
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_appeals (
	id TEXT PRIMARY KEY, account TEXT NOT NULL, organization TEXT, code TEXT NOT NULL,
	scope TEXT NOT NULL, reset_at REAL, horizon REAL, window_seconds REAL,
	created_at REAL NOT NULL, request_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_decisions (
	id TEXT PRIMARY KEY, appeal_id TEXT NOT NULL REFERENCES review_appeals(id),
	action TEXT NOT NULL, operator TEXT NOT NULL, reason TEXT NOT NULL,
	created_at REAL NOT NULL, parameters TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_credits (
	decision_id TEXT PRIMARY KEY REFERENCES review_decisions(id), scope TEXT NOT NULL,
	name TEXT NOT NULL, kind TEXT NOT NULL, requests INTEGER NOT NULL, tokens INTEGER NOT NULL,
	starts_at REAL NOT NULL, expires_at REAL NOT NULL, horizon REAL, window_seconds REAL
);
CREATE TABLE IF NOT EXISTS review_unlinks (
	decision_id TEXT PRIMARY KEY REFERENCES review_decisions(id),
	account TEXT NOT NULL, organization TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS review_appeals_no_update BEFORE UPDATE ON review_appeals
BEGIN SELECT RAISE(ABORT, 'append only'); END;
CREATE TRIGGER IF NOT EXISTS review_appeals_no_delete BEFORE DELETE ON review_appeals
BEGIN SELECT RAISE(ABORT, 'append only'); END;
CREATE TRIGGER IF NOT EXISTS review_decisions_no_update BEFORE UPDATE ON review_decisions
BEGIN SELECT RAISE(ABORT, 'append only'); END;
CREATE TRIGGER IF NOT EXISTS review_decisions_no_delete BEFORE DELETE ON review_decisions
BEGIN SELECT RAISE(ABORT, 'append only'); END;
CREATE TRIGGER IF NOT EXISTS review_credits_no_update BEFORE UPDATE ON review_credits
BEGIN SELECT RAISE(ABORT, 'append only'); END;
CREATE TRIGGER IF NOT EXISTS review_credits_no_delete BEFORE DELETE ON review_credits
BEGIN SELECT RAISE(ABORT, 'append only'); END;
CREATE TRIGGER IF NOT EXISTS review_unlinks_no_update BEFORE UPDATE ON review_unlinks
BEGIN SELECT RAISE(ABORT, 'append only'); END;
CREATE TRIGGER IF NOT EXISTS review_unlinks_no_delete BEFORE DELETE ON review_unlinks
BEGIN SELECT RAISE(ABORT, 'append only'); END;
"""


def initialize(db: sqlite3.Connection) -> None:
	db.executescript(_SCHEMA)


def open_appeal(
	db: sqlite3.Connection,
	*,
	account: str,
	organization: str | None,
	code: str,
	scope: str,
	reset_at: float | None,
	horizon: float | None,
	window_seconds: float | None,
	now: float,
	digest: str,
) -> str:
	ref = uuid.uuid4().hex
	db.execute(
		"INSERT INTO review_appeals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
		(ref, account, organization, code, scope, reset_at, horizon, window_seconds, now, digest),
	)
	return ref


def list_appeals(db: sqlite3.Connection) -> list[dict[str, Any]]:
	rows = db.execute(
		"SELECT id, account, organization, code, scope, reset_at, created_at "
		"FROM review_appeals ORDER BY created_at, rowid"
	).fetchall()
	return [
		dict(
			zip(
				("id", "account", "organization", "code", "scope", "reset_at", "created_at"),
				row,
				strict=True,
			)
		)
		for row in rows
	]


def show(db: sqlite3.Connection, ref: str) -> dict[str, Any] | None:
	row = db.execute(
		"SELECT id, account, organization, code, scope, reset_at, horizon, window_seconds, "
		"created_at, request_digest FROM review_appeals WHERE id = ?",
		(ref,),
	).fetchone()
	if row is None:
		return None
	result = dict(
		zip(
			(
				"id",
				"account",
				"organization",
				"code",
				"scope",
				"reset_at",
				"horizon",
				"window_seconds",
				"created_at",
				"request_digest",
			),
			row,
			strict=True,
		)
	)
	rows = db.execute(
		"SELECT id, action, operator, reason, created_at, parameters FROM review_decisions "
		"WHERE appeal_id = ? ORDER BY rowid",
		(ref,),
	).fetchall()
	result["decisions"] = [
		dict(
			zip(
				("id", "action", "operator", "reason", "created_at", "parameters"),
				(*r[:5], json.loads(r[5])),
				strict=True,
			)
		)
		for r in rows
	]
	return result


def decide(
	db: sqlite3.Connection,
	ref: str,
	*,
	action: str,
	operator: str,
	reason: str,
	requests: int = 0,
	tokens: int = 0,
	expires_at: float | None = None,
	now: float | None = None,
) -> str:
	now = time.time() if now is None else now
	appeal = show(db, ref)
	if appeal is None:
		raise ValueError("unknown appeal")
	if not operator.strip() or not reason.strip():
		raise ValueError("operator and reason are required")
	if action not in {"grant", "restore", "unlink", "reject"}:
		raise ValueError("unknown action")
	if requests < 0 or tokens < 0:
		raise ValueError("credit amounts cannot be negative")
	if action in {"grant", "restore"}:
		if not (requests or tokens) or expires_at is None or expires_at <= now:
			raise ValueError("positive credit and a future expiry are required")
		if action == "restore" and "budget_exceeded" not in appeal["code"]:
			raise ValueError("restore requires a budget appeal")
	if action == "unlink" and not appeal["organization"]:
		raise ValueError("appeal has no organization")
	decision = uuid.uuid4().hex
	params = {"requests": requests, "tokens": tokens, "expires_at": expires_at}
	db.execute(
		"INSERT INTO review_decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
		(decision, ref, action, operator, reason, now, json.dumps(params)),
	)
	if action in {"grant", "restore"}:
		scope = appeal["scope"]
		name = appeal["account"] if scope == "account" else appeal["organization"]
		kind = (
			"policy"
			if "budget_exceeded" not in appeal["code"]
			else "rolling"
			if appeal["window_seconds"] is not None
			else "horizon"
		)
		db.execute(
			"INSERT INTO review_credits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
			(
				decision,
				scope,
				name,
				kind,
				requests,
				tokens,
				now,
				expires_at,
				appeal["horizon"],
				appeal["window_seconds"],
			),
		)
	elif action == "unlink":
		db.execute(
			"INSERT INTO review_unlinks VALUES (?, ?, ?)",
			(decision, appeal["account"], appeal["organization"]),
		)
	return decision
