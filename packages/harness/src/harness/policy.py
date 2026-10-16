"""Per-account policies: the MMD harness and the baselines it is compared with.

Policies take the account, the request text, and the current time as arguments and never read
the clock, so a simulator and an HTTP server can drive the same code. A completed window
includes the current request and is scored before the victim runs, so it can change the action
for that request (paper Section III-D).
"""

import hashlib
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

from harness.calibration import Calibration, HarnessConfig
from harness.deception import keyed_unit
from harness.encoder import Encoder
from harness.mmd import Array, WindowScorer
from harness.pow import Challenge, ChallengeBook
from harness.serialization import detector_text

Action = Literal["serve", "challenge", "refuse"]


@dataclass(frozen=True)
class Decision:
	"""The harness's decision for one attempted request.

	serve releases the victim's output at release_at; challenge releases it at release_at
	only after the challenge is redeemed; refuse releases nothing. Level, p-value, and
	statistic are telemetry for the experimenter and are not shown to the client.
	"""

	account: str
	attempt: int
	level: int
	action: Action
	release_at: float
	challenge: Challenge | None = None
	window_completed: bool = False
	p_value: float | None = None
	statistic: float | None = None
	# Release a keyed wrong answer instead of the victim's. Hidden from the client, which sees
	# the same action, spacing, and challenge as at L3.
	deceive: bool = False
	# When a window raises the level: the window's requests with the largest MMD witness,
	# as (request digest, witness) pairs, most anomalous first.
	witness: tuple[tuple[str, float], ...] = ()

	def client_view(self) -> dict[str, Any]:
		view: dict[str, Any] = {"action": self.action, "release_at": self.release_at}
		if self.challenge:
			view["challenge"] = {
				"nonce": self.challenge.nonce,
				"prefix": self.challenge.prefix(),
				"difficulty": self.challenge.difficulty,
				"expires_at": self.challenge.expires_at,
			}
		return view


class Policy(Protocol):
	def admit(
		self,
		account: str,
		text: str,
		now: float,
		embedding: Array | None = None,
		binding: str | None = None,
	) -> Decision: ...


def request_digest(text: str) -> str:
	return hashlib.sha256(text.encode()).hexdigest()


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


class _HorizonPolicy:
	"""Keeps one state per account and starts a fresh one when a policy horizon ends."""

	def __init__(self, horizon_seconds: float, horizon_attempts: int) -> None:
		self.horizon_seconds = horizon_seconds
		self.horizon_attempts = horizon_attempts
		self.accounts: dict[str, AccountState] = {}

	def _state(self, account: str, now: float) -> AccountState:
		state = self.accounts.get(account)
		if (
			state is None
			or state.attempts >= self.horizon_attempts
			or now - state.horizon_start >= self.horizon_seconds
		):
			state = AccountState(horizon_start=now)
			self.accounts[account] = state
		state.attempts += 1
		return state


class AllowPolicy(_HorizonPolicy):
	"""The undefended service: every request is served immediately."""

	def admit(
		self,
		account: str,
		text: str,
		now: float,
		embedding: Array | None = None,
		binding: str | None = None,
	) -> Decision:
		state = self._state(account, now)
		return Decision(account, state.attempts, 0, "serve", now)


class QuotaPolicy(_HorizonPolicy):
	"""Blocks an account for the rest of the horizon once it exceeds a request quota."""

	def __init__(self, quota: float, horizon_seconds: float, horizon_attempts: int) -> None:
		super().__init__(horizon_seconds, horizon_attempts)
		self.quota = quota

	def admit(
		self,
		account: str,
		text: str,
		now: float,
		embedding: Array | None = None,
		binding: str | None = None,
	) -> Decision:
		state = self._state(account, now)
		if state.level >= 4 or state.attempts > self.quota:
			state.level = 4
			return Decision(account, state.attempts, 4, "refuse", now)
		return Decision(account, state.attempts, 0, "serve", now)


class RatePolicy(_HorizonPolicy):
	"""Blocks an account for the rest of the horizon once it exceeds a per-minute rate."""

	def __init__(self, per_minute: float, horizon_seconds: float, horizon_attempts: int) -> None:
		super().__init__(horizon_seconds, horizon_attempts)
		self.per_minute = per_minute

	def admit(
		self,
		account: str,
		text: str,
		now: float,
		embedding: Array | None = None,
		binding: str | None = None,
	) -> Decision:
		state = self._state(account, now)
		state.recent.append(now)
		while state.recent and state.recent[0] <= now - 60:
			state.recent.popleft()
		if state.level >= 4 or len(state.recent) > self.per_minute:
			state.level = 4
			return Decision(account, state.attempts, 4, "refuse", now)
		return Decision(account, state.attempts, 0, "serve", now)


class MMDPolicy(_HorizonPolicy):
	"""Accumulated MMD evidence per account, mapped to the graduated actions of Table I.

	L0 and L1 serve normally (L1 records an alert). L2 spaces admitted requests by at least
	min_interval seconds. L3 adds a proof-of-work challenge per response. L4 refuses every
	later request in the horizon. The level never decreases within a horizon.

	With l4_action="deceive", L4 instead keeps L3's spacing and challenges and marks a keyed
	share deception_rate of requests for a deceptive answer (see harness.deception). The
	share is keyed by request, not account, so every account asking a request gets the same
	treatment. This is a separately reported arm; the main policy refuses.
	"""

	def __init__(
		self,
		calibration: Calibration,
		scorer: WindowScorer,
		*,
		accumulator: str = "paper",
		thresholds: Literal["calibrated", "ville"] = "calibrated",
		encoder: Encoder | None = None,
		min_interval: float = 1.0,
		difficulty: int = 20,
		challenge_ttl: float = 300,
		l4_action: Literal["refuse", "deceive"] = "refuse",
		deception_rate: float = 1.0,
		deception_key: bytes | None = None,
		witness_size: int = 5,
	) -> None:
		config: HarnessConfig = calibration.config
		if l4_action == "deceive" and not deception_key:
			raise ValueError("deception needs a secret key")
		if not 0 <= deception_rate <= 1:
			raise ValueError("deception_rate must be in [0, 1]")
		super().__init__(config.horizon_seconds, config.horizon_attempts)
		if accumulator not in calibration.thresholds:
			raise ValueError(f"the calibration has no thresholds for {accumulator!r}")
		if thresholds == "ville" and config.accumulators[accumulator].kind != "e_process":
			raise ValueError("Ville thresholds apply only to an e_process accumulator")
		self.calibration = calibration
		self.scorer = scorer
		self.accumulator = accumulator
		self.spec = config.accumulators[accumulator]
		self.threshold_mode: Literal["calibrated", "ville"] = thresholds
		self.encoder = encoder
		self.min_interval = min_interval
		self.challenges = ChallengeBook(difficulty, challenge_ttl)
		self.l4_action: Literal["refuse", "deceive"] = l4_action
		self.deception_rate = deception_rate
		self.deception_key = deception_key or b""
		self.witness_size = witness_size

	def _respond(
		self, account: str, state: AccountState, text: str, now: float, binding: str
	) -> tuple[Action, float, Challenge | None, bool]:
		"""The action, release time, challenge, and deception flag for the current level."""
		if state.level >= 4 and self.l4_action == "refuse":
			return "refuse", now, None, False
		if state.level < 2:
			return "serve", now, None, False
		release_at = max(now, state.last_release + self.min_interval)
		state.last_release = release_at
		if state.level < 3:
			return "serve", release_at, None, False
		challenge = self.challenges.issue(account, request_digest(binding), now)
		deceive = state.level >= 4 and (
			keyed_unit(self.deception_key, "rate", request_digest(text)) < self.deception_rate
		)
		return "challenge", release_at, challenge, deceive

	def admit(
		self,
		account: str,
		text: str,
		now: float,
		embedding: Array | None = None,
		binding: str | None = None,
	) -> Decision:
		"""Decide one request. A challenge binds to `binding`, by default the text; a server
		passes the whole canonical request so a solution cannot be reused with other settings."""
		bound = text if binding is None else binding
		state = self._state(account, now)
		if state.level >= 4:
			action, release_at, challenge, deceive = self._respond(account, state, text, now, bound)
			return Decision(
				account, state.attempts, 4, action, release_at, challenge=challenge, deceive=deceive
			)
		if embedding is None:
			if self.encoder is None:
				raise ValueError("pass an embedding or configure an encoder")
			embedding = self.encoder.encode([detector_text(self.calibration.config.task, text)])[0]
		state.buffer.append(np.asarray(embedding, dtype=np.float64))
		state.buffer_digests.append(request_digest(text))
		p_value: float | None = None
		witness: tuple[tuple[str, float], ...] = ()
		completed = len(state.buffer) == self.scorer.window_size
		if completed:
			window = np.stack(state.buffer)
			score = self.scorer.score(window)
			state.windows += 1
			p_value = self.calibration.p_value(score)
			state.statistic = self.spec.update(state.statistic, p_value)
			reached = self.calibration.level(self.accumulator, state.statistic, self.threshold_mode)
			if reached > state.level:
				values = self.scorer.witness(window)
				top = np.argsort(-values)[: self.witness_size]
				witness = tuple((state.buffer_digests[i], float(values[i])) for i in top)
			state.level = max(state.level, reached)
			state.buffer.clear()
			state.buffer_digests.clear()
		action, release_at, challenge, deceive = self._respond(account, state, text, now, bound)
		return Decision(
			account,
			state.attempts,
			state.level,
			action,
			release_at,
			challenge=challenge,
			window_completed=completed,
			p_value=p_value,
			statistic=state.statistic,
			deceive=deceive,
			witness=witness,
		)

	def redeem(self, decision: Decision, binding: str, counter: int, now: float) -> bool:
		"""Verify a challenge solution for the binding the challenge was issued for.

		The output is released only when this returns True.
		"""
		if decision.challenge is None:
			raise ValueError("this decision has no challenge")
		return self.challenges.redeem(
			decision.challenge.nonce, decision.account, request_digest(binding), counter, now
		)
