"""Unqualified prompt-path feature recipe, shared by fitting and admission.

One campaign is one authenticated principal's first H admitted prompts, with no automatic
renewal. Looks are fixed nonoverlapping MMD windows; coverage uses the whole prefix at
those same looks. Missing streams abstain together. No cumulative log-det stream exists.
"""

import hashlib
import json
import math
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.calibration import Calibration
from harness.mmd import WindowScorer
from harness.path_calibration import PathCalibration, PathState
from harness.shadow import minimum_null, profile, request_domain, upper_p
from harness.signals import SignalCalibration


def encoder_contract(identity: dict[str, Any]) -> dict[str, Any]:
	return {key: value for key, value in identity.items() if key not in {"encoded", "truncated"}}


def detector_hash(calibration: Calibration, scorer: WindowScorer) -> str:
	payload = calibration.model_dump_json().encode() + scorer.references.tobytes()
	payload += repr((scorer.sigma, scorer.scales)).encode()
	return hashlib.sha256(payload).hexdigest()


class PathRecipe(BaseModel):
	model_config = ConfigDict(frozen=True, extra="forbid")

	version: Literal["prompt-path-v1"] = "prompt-path-v1"
	campaign_unit: Literal["principal-first-H"] = "principal-first-H"
	detector_hash: str
	encoder: dict[str, Any]
	request_horizon: int = Field(gt=0)
	window_size: int = Field(ge=2)
	evidence: SignalCalibration | None = None


class RuntimePathCalibration(BaseModel):
	model_config = ConfigDict(frozen=True, extra="forbid")

	version: Literal["runtime-path-v1"] = "runtime-path-v1"
	qualified: Literal[False] = False
	recipe: PathRecipe
	calibration: PathCalibration
	benign_paths_sha256: str | None = None
	input_hash: str | None = None

	@model_validator(mode="after")
	def validate_artifact(self) -> Self:
		r, c = self.recipe, self.calibration
		if r.request_horizon % r.window_size or c.horizon != r.request_horizon // r.window_size:
			raise ValueError("request horizon must contain exactly the declared complete looks")
		if c.scoring.name != r.version or set(c.scoring.streams) - {"mmd", "coverage_mean_logdet"}:
			raise ValueError("unsupported runtime path scoring or stream")
		if "coverage_mean_logdet" in c.scoring.streams and (
			r.evidence is None or r.evidence.coverage is None
		):
			raise ValueError("coverage stream needs a fitted coverage signal calibration")
		payload = {
			"recipe": r.model_dump(mode="json"),
			"calibration": c.model_dump(mode="json"),
			"benign_paths_sha256": self.benign_paths_sha256,
		}
		digest = hashlib.sha256(
			json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
		).hexdigest()
		if self.input_hash not in {None, digest}:
			raise ValueError("runtime path calibration hash differs from its feature recipe")
		object.__setattr__(self, "input_hash", digest)
		return self

	def require_compatible(
		self, calibration: Calibration, scorer: WindowScorer, encoder: dict[str, Any]
	) -> None:
		strata = (
			{"single_domain", "multi_domain"} if calibration.config.task == "gateway" else {"all"}
		)
		if self.calibration.stratum not in strata:
			raise ValueError("path calibration has an unsupported stratum for this task")
		if self.recipe.detector_hash != detector_hash(calibration, scorer):
			raise ValueError("path calibration uses a different detector")
		if self.recipe.encoder != encoder_contract(encoder):
			raise ValueError("path calibration uses a different encoder")
		if self.recipe.window_size != scorer.window_size:
			raise ValueError("path calibration uses a different window schedule")


class CampaignState(BaseModel):
	"""Lives outside renewable account horizons, but commits with each admission."""

	calibration_hash: str
	revision: int = 0
	requests: int = 0
	vectors: list[list[float]] = Field(default_factory=list)
	digests: list[str] = Field(default_factory=list)
	domains: set[str] = Field(default_factory=set)
	running_max: float | None = None
	look_count: int = 0
	stopped: bool = False
	stop_look: int | None = None
	p_value: float | None = None
	unsupported: bool = False
	status: str = "abstain"
	abstention_reason: str | None = "waiting for first look"

	def telemetry(self) -> dict[str, Any]:
		return {
			"calibration_hash": self.calibration_hash,
			"policy_version": "prompt-path-v1",
			"qualified": False,
			"revision": self.revision,
			"requests": self.requests,
			"look_count": self.look_count,
			"running_max": self.running_max,
			"p_value": self.p_value,
			"stopped": self.stopped,
			"stop_look": self.stop_look,
			"status": self.status,
			"abstained": self.status == "abstain",
			"abstention_reason": self.abstention_reason,
		}


def path_look(
	recipe: PathRecipe,
	streams: tuple[str, ...],
	vectors: list[list[float]],
	digests: list[str],
	domains: set[str],
	calibration: Calibration,
	scorer: WindowScorer,
	stratum: str,
) -> tuple[list[float] | None, str | None]:
	"""Return every declared score, or an explicit unsupported-look abstention."""
	matched = profile(calibration.config.task, domains)
	if matched != stratum:
		return None, "unsupported stratum"
	x = np.asarray(vectors, dtype=np.float64)
	values: dict[str, float] = {}
	if "mmd" in streams:
		keys = digests[-recipe.window_size :] if calibration.config.distinct_prompts else None
		values["mmd"] = float(scorer.score(x[-recipe.window_size :], keys))
	if "coverage_mean_logdet" in streams:
		from harness.logdet import length_bin

		null = recipe.evidence
		assert null is not None and null.coverage is not None
		key = f"coverage_mean_logdet[{length_bin(len(vectors))}]"
		band = null.nulls.get(matched, {}).get(key, [])
		if len(band) < minimum_null(null.target):
			return None, f"insufficient coverage null for {matched}:{key}"
		mean = float(np.mean(null.coverage.gains(x)[1:]))
		values["coverage_mean_logdet"] = -math.log(upper_p(band, mean))
	return [values[name] for name in streams], None


def advance_campaign(
	artifact: RuntimePathCalibration,
	prior: CampaignState,
	text: str,
	vector: list[float],
	calibration: Calibration,
	scorer: WindowScorer,
) -> CampaignState:
	"""Pure preparation, outside the write transaction, against one committed revision."""
	if prior.calibration_hash != artifact.input_hash:
		raise ValueError("campaign is bound to another path calibration")
	state = prior.model_copy(deep=True)
	state.revision += 1
	if state.stopped:
		state.status, state.abstention_reason = "stop", None
		return state
	if state.requests >= artifact.recipe.request_horizon:
		state.status, state.abstention_reason = "abstain", "campaign horizon expired; no rearming"
		return state
	state.requests += 1
	state.vectors.append(vector)
	state.digests.append(hashlib.sha256(text.encode()).hexdigest())
	state.domains.add(request_domain(calibration.config.task, text))
	# Stratum is fixed before fitting. Leaving it permanently disables statistical stops.
	if (
		state.requests % artifact.recipe.window_size == 0
		and profile(calibration.config.task, state.domains) != artifact.calibration.stratum
	):
		state.unsupported = True
	if state.unsupported:
		state.status, state.abstention_reason = "abstain", "unsupported stratum in campaign path"
		return state
	if state.requests % artifact.recipe.window_size:
		state.status, state.abstention_reason = "abstain", "between predetermined looks"
		return state
	values, reason = path_look(
		artifact.recipe,
		artifact.calibration.scoring.streams,
		state.vectors,
		state.digests,
		state.domains,
		calibration,
		scorer,
		artifact.calibration.stratum,
	)
	if values is None:
		state.status, state.abstention_reason = "abstain", reason
		return state
	rank = PathState(artifact.calibration)
	rank.running_max, rank.look_count = state.running_max, state.look_count
	rank.stopped, rank.stop_look, rank._p_value = state.stopped, state.stop_look, state.p_value
	result = rank.observe(values)
	state.running_max, state.look_count = rank.running_max, rank.look_count
	state.stopped, state.stop_look, state.p_value = result.stopped, result.stop_look, result.p_value
	state.status = result.decision
	state.abstention_reason = artifact.calibration.abstention_reason if rank.abstained else None
	return state
