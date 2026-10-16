"""Rules that turn a sequence of window p-values into an account's evidence statistic.

Every rule takes the upper-tail rank p of each completed window (paper Eq. 3) and keeps one
float of state. The statistic is compared with thresholds calibrated on benign horizons.
"""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AccumulatorSpec(BaseModel):
	"""One evidence rule.

	- instantaneous: the current window's evidence, -log p, with no memory.
	- decayed_cusum: paper Eq. 4, r = max(0, decay * r - log p - drift).
	- e_process: the running sum of log e-values, e = eta * p ** (eta - 1). Under benign
	traffic with valid conditional p-values, P(ever >= log(1/alpha)) <= alpha (Ville).
	- e_cusum: the same e-values with a floor at zero, log R = max(0, log R) + log e, so a
	long clean history does not bank negative evidence against a later attack.
	"""

	model_config = ConfigDict(extra="forbid")

	kind: Literal["instantaneous", "decayed_cusum", "e_process", "e_cusum"]
	decay: float = Field(default=0.9, ge=0, le=1)
	drift: float = Field(default=1.0, ge=0)
	eta: float = Field(default=0.5, gt=0, lt=1)

	def update(self, state: float, p: float) -> float:
		if not 0 < p <= 1:
			raise ValueError("p must be in (0, 1]")
		if self.kind == "instantaneous":
			return -math.log(p)
		if self.kind == "decayed_cusum":
			return max(0.0, self.decay * state - math.log(p) - self.drift)
		log_e = math.log(self.eta) + (self.eta - 1) * math.log(p)
		if self.kind == "e_process":
			return state + log_e
		return max(0.0, state) + log_e


# The paper's rule and its ablations, the instantaneous baseline, and the e-value variants.
DEFAULT_ACCUMULATORS: dict[str, AccumulatorSpec] = {
	"instantaneous": AccumulatorSpec(kind="instantaneous"),
	"paper": AccumulatorSpec(kind="decayed_cusum", decay=0.9, drift=1.0),
	"paper_no_memory": AccumulatorSpec(kind="decayed_cusum", decay=0.0, drift=1.0),
	"paper_no_decay": AccumulatorSpec(kind="decayed_cusum", decay=1.0, drift=1.0),
	"e_process": AccumulatorSpec(kind="e_process", eta=0.5),
	"e_cusum": AccumulatorSpec(kind="e_cusum", eta=0.5),
}
