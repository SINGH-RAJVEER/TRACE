"""Per-account policies: the MMD harness and the baselines it is compared with.

Policies take the account, the request text, and the current time as arguments and never read
the clock, so a simulator and an HTTP server can drive the same code. A completed window
includes the current request and is scored before the victim runs, so it can change the action
for that request (paper Section III-D).
"""

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

import numpy as np

from harness.calibration import Calibration, HarnessConfig
from harness.deception import keyed_unit
from harness.encoder import Encoder
from harness.mmd import Array, WindowScorer, distinct
from harness.pow import Challenge, ChallengeBook
from harness.serialization import detector_text
from harness.store import AccountState, MemoryStore, StateStore, Transaction

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
	# Start of the account horizon the attempt was counted in. With the account and attempt,
	# it identifies the admission in the release ledger.
	horizon: float = 0.0

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


class _HorizonPolicy:
	"""Keeps one state per account and starts a fresh one when a policy horizon ends.

	State lives in a store, in memory unless one is passed. Each admission reads, decides, and
	writes back in one store transaction, so a durable store keeps attempts, levels, partial
	windows, and open challenges across restarts and serializes concurrent admissions.
	"""

	# Whether the horizon attempt cap applies; the undefended arm serves past it.
	capped = True

	def __init__(
		self, horizon_seconds: float, horizon_attempts: int, store: StateStore | None = None
	) -> None:
		self.horizon_seconds = horizon_seconds
		self.horizon_attempts = horizon_attempts
		self.store: StateStore = store or MemoryStore()

	def state(self, account: str) -> AccountState | None:
		"""The account's stored state, for inspection."""
		with self.store.transaction() as tx:
			return tx.account(account)

	def admit(
		self,
		account: str,
		text: str,
		now: float,
		embedding: Array | None = None,
		binding: str | None = None,
	) -> Decision:
		with self.store.transaction() as tx:
			state = tx.account(account)
			if state is None or now - state.horizon_start >= self.horizon_seconds:
				state = AccountState(horizon_start=now)
			state.attempts += 1
			if self.capped and state.attempts > self.horizon_attempts:
				# The attempt cap denies new admissions; only elapsed time can renew it.
				state.level = 4
				decision = Decision(account, state.attempts, 4, "refuse", now)
			else:
				decision = self._decide(tx, account, state, text, now, embedding, binding)
			tx.put_account(account, state)
		return replace(decision, horizon=state.horizon_start)

	def _decide(
		self,
		tx: Transaction,
		account: str,
		state: AccountState,
		text: str,
		now: float,
		embedding: Array | None,
		binding: str | None,
	) -> Decision:
		raise NotImplementedError


class AllowPolicy(_HorizonPolicy):
	"""The undefended service: every request is served immediately."""

	capped = False

	def _decide(
		self,
		tx: Transaction,
		account: str,
		state: AccountState,
		text: str,
		now: float,
		embedding: Array | None,
		binding: str | None,
	) -> Decision:
		return Decision(account, state.attempts, 0, "serve", now)


class QuotaPolicy(_HorizonPolicy):
	"""Blocks an account for the rest of the horizon once it exceeds a request quota."""

	def __init__(
		self,
		quota: float,
		horizon_seconds: float,
		horizon_attempts: int,
		store: StateStore | None = None,
	) -> None:
		super().__init__(horizon_seconds, horizon_attempts, store)
		self.quota = quota

	def _decide(
		self,
		tx: Transaction,
		account: str,
		state: AccountState,
		text: str,
		now: float,
		embedding: Array | None,
		binding: str | None,
	) -> Decision:
		if state.level >= 4 or state.attempts > self.quota:
			state.level = 4
			return Decision(account, state.attempts, 4, "refuse", now)
		return Decision(account, state.attempts, 0, "serve", now)


class RatePolicy(_HorizonPolicy):
	"""Blocks an account for the rest of the horizon once it exceeds a per-minute rate."""

	def __init__(
		self,
		per_minute: float,
		horizon_seconds: float,
		horizon_attempts: int,
		store: StateStore | None = None,
	) -> None:
		super().__init__(horizon_seconds, horizon_attempts, store)
		self.per_minute = per_minute

	def _decide(
		self,
		tx: Transaction,
		account: str,
		state: AccountState,
		text: str,
		now: float,
		embedding: Array | None,
		binding: str | None,
	) -> Decision:
		if state.level >= 4:
			return Decision(account, state.attempts, 4, "refuse", now)
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

	With actions="direct", the ablation that replaces graduated actions with direct blocking,
	the account serves normally until its statistic exceeds the L4 threshold and is then
	refused. The L4 threshold is fitted to the same target, so no recalibration is needed.
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
		actions: Literal["graduated", "direct"] = "graduated",
		store: StateStore | None = None,
		policy_version: Literal["legacy-v1", "verify-risk-v2"] = "legacy-v1",
	) -> None:
		calibration.require_current_policy()
		config: HarnessConfig = calibration.config
		if config.task == "coding" and l4_action == "deceive":
			raise ValueError("coding deception is not implemented; use L4 refusal")
		if l4_action == "deceive" and policy_version == "legacy-v1" and not deception_key:
			raise ValueError("deception needs a secret key")
		if not 0 <= deception_rate <= 1:
			raise ValueError("deception_rate must be in [0, 1]")
		if policy_version not in ("legacy-v1", "verify-risk-v2"):
			raise ValueError("unknown policy version")
		if policy_version == "verify-risk-v2" and actions != "graduated":
			raise ValueError("verify-risk-v2 requires graduated actions")
		super().__init__(config.horizon_seconds, config.horizon_attempts, store)
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
		self.actions: Literal["graduated", "direct"] = actions
		self.policy_version = policy_version

	def _respond(
		self,
		tx: Transaction,
		account: str,
		state: AccountState,
		text: str,
		now: float,
		binding: str,
	) -> tuple[Action, float, Challenge | None, bool]:
		"""The action, release time, challenge, and deception flag for the current level."""
		if state.level >= 4 and self.l4_action == "refuse" and self.policy_version == "legacy-v1":
			return "refuse", now, None, False
		if state.level < 2:
			# An L0 or L1 admission still counts as the previous release, so L2 spacing
			# applies from the first L2 request.
			state.last_release = now
			return "serve", now, None, False
		release_at = max(now, state.last_release + self.min_interval)
		state.last_release = release_at
		if state.level < 3:
			return "serve", release_at, None, False
		deceive = (
			self.policy_version == "legacy-v1"
			and state.level >= 4
			and (keyed_unit(self.deception_key, "rate", request_digest(text)) < self.deception_rate)
		)
		# The challenge holds what its solution releases, so a restart cannot lose it.
		held = {
			"attempt": state.attempts,
			"level": state.level,
			"release_at": release_at,
			"deceive": deceive,
			"horizon": state.horizon_start,
		}
		challenge = self.challenges.issue(tx, account, request_digest(binding), now, held)
		return "challenge", release_at, challenge, deceive

	def _decide(
		self,
		tx: Transaction,
		account: str,
		state: AccountState,
		text: str,
		now: float,
		embedding: Array | None,
		binding: str | None,
	) -> Decision:
		"""Decide one request. A challenge binds to `binding`, by default the text; a server
		passes the whole canonical request so a solution cannot be reused with other settings."""
		bound = text if binding is None else binding
		if state.level >= 4:
			action, release_at, challenge, deceive = self._respond(
				tx, account, state, text, now, bound
			)
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
			keys = state.buffer_digests if self.calibration.config.distinct_prompts else None
			score = self.scorer.score(window, keys)
			state.windows += 1
			p_value = self.calibration.p_value(score)
			state.statistic = self.spec.update(state.statistic, p_value)
			reached = self.calibration.level(self.accumulator, state.statistic, self.threshold_mode)
			if self.actions == "direct":
				reached = 4 if reached >= 4 else 0
			if reached > state.level:
				# The witness ranks the queries that were scored, so repeats appear once.
				kept = distinct(keys) if keys is not None else list(range(len(window)))
				values = self.scorer.witness(window[kept])
				top = np.argsort(-values)[: self.witness_size]
				witness = tuple((state.buffer_digests[kept[i]], float(values[i])) for i in top)
			state.level = max(state.level, reached)
			state.buffer.clear()
			state.buffer_digests.clear()
		action, release_at, challenge, deceive = self._respond(tx, account, state, text, now, bound)
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

	def redeem(
		self, nonce: str, account: str, binding: str, counter: int, now: float
	) -> Decision | None:
		"""Verify a challenge solution for the account and binding it was issued for.

		Returns the held decision whose output the solution releases, or None. A solution is
		accepted once, also across restarts and connections sharing a durable store.
		"""
		with self.store.transaction() as tx:
			digest = request_digest(binding)
			found = self.challenges.redeem(tx, nonce, account, digest, counter, now)
			if found is None:
				return None
			challenge, held = found
			if self.policy_version == "verify-risk-v2":
				# Checked in the transaction that consumes the proof. A rejected proof stays
				# consumed: the request needs a new admission, subject to the attempt cap.
				state = tx.account(account)
				if (
					state is None
					or state.horizon_start != held["horizon"]
					or now - state.horizon_start >= self.horizon_seconds
					or state.attempts > self.horizon_attempts
					or state.level > held["level"]
				):
					return None
		return Decision(
			account,
			held["attempt"],
			held["level"],
			"challenge",
			held["release_at"],
			challenge=challenge,
			deceive=held["deceive"],
			horizon=held["horizon"],
		)
