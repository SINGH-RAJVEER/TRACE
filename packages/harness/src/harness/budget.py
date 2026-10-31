"""A saved acquisition budget: what each account may receive over a rolling window.

The budget is a limit over the release ledger in harness.store, checked in the transaction that
reserves an admission, so simultaneous requests cannot overspend. A policy horizon renews an
account's attempts and level, but a released answer stays on the ledger until it leaves the
rolling window, so no daily reset refills a budget.

Admissions are charged at their maximum output until they settle. A failed or cancelled
admission released nothing and is not charged, though it still counts as an attempt. One left
unsettled by a crash keeps its full charge and gives back only its in-flight slot. The limits
bound released requests and tokens, not what a student learns from them, and they apply per
account, so more accounts buy more budget.
"""

from dataclasses import dataclass
from typing import Literal

from harness.store import Reservation, Transaction

Refusal = Literal[
	"budget_requests", "budget_tokens", "budget_concurrency", "duplicate_request", "key_conflict"
]


@dataclass(frozen=True)
class BudgetLimits:
	window_seconds: float
	requests: int | None = None
	output_tokens: int | None = None
	concurrency: int | None = None

	@property
	def active(self) -> bool:
		limits = (self.requests, self.output_tokens, self.concurrency)
		return any(limit is not None for limit in limits)

	def refusal(self, tx: Transaction, claim: Reservation) -> Refusal | None:
		"""Why this claim on the ledger cannot be reserved, if any. A key that already released
		output is refused rather than generated again, so a retry cannot collect a new sample."""
		if claim.idempotency_key is not None:
			earlier = tx.idempotent(claim.account, claim.idempotency_key)
			if earlier is not None:
				same = earlier.request_digest == claim.request_digest
				return "duplicate_request" if same else "key_conflict"
		if not self.active:
			return None
		since = claim.reserved_at - self.window_seconds
		usage = tx.budget_usage(claim.account, since, claim.session)
		if self.requests is not None and usage.requests >= self.requests:
			return "budget_requests"
		if (
			self.output_tokens is not None
			and usage.output_tokens + claim.max_output_tokens > self.output_tokens
		):
			return "budget_tokens"
		if self.concurrency is not None and usage.in_flight >= self.concurrency:
			return "budget_concurrency"
		return None
