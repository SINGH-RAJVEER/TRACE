"""SHA-256 proof-of-work challenges bound to an account, a request, and an expiry (Table I, L3)."""

import hashlib
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from harness.store import Transaction


@dataclass(frozen=True)
class Challenge:
	nonce: str
	account: str
	request_digest: str
	difficulty: int
	expires_at: float

	def prefix(self) -> str:
		"""Everything hashed before the counter; clients append the decimal counter."""
		return f"{self.nonce}:{self.account}:{self.request_digest}:{self.expires_at!r}:"

	def payload(self, counter: int) -> bytes:
		return f"{self.prefix()}{counter}".encode()


def leading_zero_bits(digest: bytes) -> int:
	value = int.from_bytes(digest)
	return len(digest) * 8 - value.bit_length()


def solve_prefix(prefix: str, difficulty: int) -> int:
	"""The smallest counter whose SHA-256 of prefix + counter meets the difficulty."""
	counter = 0
	while leading_zero_bits(hashlib.sha256(f"{prefix}{counter}".encode()).digest()) < difficulty:
		counter += 1
	return counter


def solve(challenge: Challenge) -> tuple[int, int]:
	"""Find a counter meeting the difficulty. Returns it and the number of hashes tried."""
	counter = solve_prefix(challenge.prefix(), challenge.difficulty)
	return counter, counter + 1


class ChallengeBook:
	"""Issues challenges and accepts each solution at most once, before it expires.

	Open challenges live in the policy's state store, together with the held decision that
	the solution releases, so redemption survives a restart and stays single-use across
	connections to the same store.
	"""

	def __init__(self, difficulty: int = 20, ttl_seconds: float = 300) -> None:
		self.difficulty = difficulty
		self.ttl_seconds = ttl_seconds

	def issue(
		self, tx: Transaction, account: str, request_digest: str, now: float, held: dict[str, Any]
	) -> Challenge:
		tx.expire_challenges(now)
		challenge = Challenge(
			nonce=secrets.token_hex(16),
			account=account,
			request_digest=request_digest,
			difficulty=self.difficulty,
			expires_at=now + self.ttl_seconds,
		)
		tx.put_challenge(challenge, held)
		return challenge

	def redeem(
		self,
		tx: Transaction,
		nonce: str,
		account: str,
		request_digest: str,
		counter: int,
		now: float,
	) -> tuple[Challenge, dict[str, Any]] | None:
		"""Accept a valid solution once and return the challenge with its held decision.

		Unknown, reused, expired, or rebound nonces return None.
		"""
		found = tx.challenge(nonce)
		if found is None:
			return None
		challenge, _ = found
		if now > challenge.expires_at:
			tx.delete_challenge(nonce)
			return None
		if (challenge.account, challenge.request_digest) != (account, request_digest):
			return None
		digest = hashlib.sha256(challenge.payload(counter)).digest()
		if leading_zero_bits(digest) < challenge.difficulty:
			return None
		tx.delete_challenge(nonce)
		return found
