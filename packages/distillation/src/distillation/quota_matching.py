"""Select measured development quota settings at matched legitimate task success loss."""

import math
from decimal import Decimal
from typing import Any

from distillation.acquisition_manifest import _sha256_json

BASELINES = ("quota", "quota_tokens")


def finite_number(value: Any, name: str, *, maximum: float | None = None) -> float:
	if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
		raise ValueError(f"{name} must be finite")
	if value < 0 or (maximum is not None and value > maximum):
		raise ValueError(f"{name} out of range")
	return float(value)


def match_quotas(development: dict[str, Any]) -> dict[str, Any]:
	"""Search a measured grid. One selected setting must match every declared stratum.

	Loss is an absolute task-success probability difference relative to allow. The operator
	must measure each grid point on the same development population and frozen workload.
	No curve fitting or interpolation occurs, and test data is rejected.
	"""
	if development.get("split") != "development" or not development.get("population_id"):
		raise ValueError("declare a development population_id")
	if not development.get("schedule_sha256"):
		raise ValueError("declare the development schedule_sha256")
	tolerance = finite_number(development["tolerance"], "tolerance", maximum=1)
	candidate = development["candidate"]
	if not candidate.get("arm") or not candidate.get("loss_by_stratum"):
		raise ValueError("candidate arm and measured loss_by_stratum are required")
	losses = candidate["loss_by_stratum"]
	for value in losses.values():
		finite_number(value, "candidate loss", maximum=1)
	baselines: dict[str, Any] = {}
	for name in BASELINES:
		trials = []
		for setting in development.get("settings", []):
			if setting["policy"] != name:
				continue
			quota = setting["request_quota"]
			if isinstance(quota, bool) or not isinstance(quota, int) or quota < 1:
				raise ValueError("request_quota must be a positive integer")
			if name == "quota_tokens":
				tokens = setting["output_token_budget"]
				if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
					raise ValueError("output_token_budget must be a positive integer")
			observed = setting["loss_by_stratum"]
			strata = {}
			for stratum, loss in losses.items():
				other = observed.get(stratum)
				if other is not None:
					finite_number(other, "baseline loss", maximum=1)
				delta = None if other is None else abs(Decimal(str(other)) - Decimal(str(loss)))
				strata[stratum] = {
					"candidate_loss": loss,
					"baseline_loss": other,
					"absolute_difference": float(delta) if delta is not None else None,
					"status": "matched"
					if delta is not None and delta <= Decimal(str(tolerance))
					else "not matched",
				}
			trials.append(
				{
					"setting": setting,
					"strata": strata,
					"matched": all(r["status"] == "matched" for r in strata.values()),
				}
			)
		matches = [row for row in trials if row["matched"]]
		matches.sort(
			key=lambda r: (
				max(v["absolute_difference"] for v in r["strata"].values()),
				_sha256_json(r["setting"]),
			)
		)
		baselines[name] = {
			"status": "matched" if matches else "not matched",
			"selected": matches[0] if matches else None,
			"trials": trials,
		}
	result = {
		"version": "matched-quota-v1",
		"development": development,
		"baselines": baselines,
		"interpolated": False,
	}
	result["digest"] = _sha256_json(result)
	return result
