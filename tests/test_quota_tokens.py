"""Named request-plus-token quota enforcement through the durable HTTP accounting path."""

from pathlib import Path

import httpx
import pytest
from harness.calibration import Calibration
from harness.mmd import WindowScorer
from harness_api.settings import Settings
from test_harness_api import _app, _request
from test_harness_api import artifacts as artifacts


def test_named_quota_requires_durable_token_budget() -> None:
	with pytest.raises(ValueError, match="quota_tokens requires"):
		Settings(_env_file=None, HARNESS_POLICY="quota_tokens")  # type: ignore[call-arg]


@pytest.mark.asyncio
@pytest.mark.parametrize("request_quota,token_budget,allowed", [(1, 100, 1), (10, 17, 2)])
async def test_named_quota_enforces_both_limits_after_restart(
	artifacts: tuple[Calibration, WindowScorer, WindowScorer],
	tmp_path: Path,
	request_quota: int,
	token_budget: int,
	allowed: int,
) -> None:
	settings = {
		"HARNESS_POLICY": "quota_tokens",
		"HARNESS_REQUEST_QUOTA": request_quota,
		"HARNESS_OUTPUT_TOKEN_BUDGET": token_budget,
		"HARNESS_STATE_DB": tmp_path / "state.db",
	}
	app, teacher, _ = _app(artifacts, tmp_path, [100] * 4, **settings)
	async with httpx.AsyncClient(
		transport=httpx.ASGITransport(app=app), base_url="http://h"
	) as client:
		for i in range(allowed):
			response = await client.post(
				"/v1/teacher/generate", json=_request(f"q{i}"), headers={"X-Benchmark-Actor": "a"}
			)
			assert response.status_code == 200
		assert teacher.calls == allowed
	app2, teacher2, _ = _app(artifacts, tmp_path, [100] * 4, **settings)
	async with httpx.AsyncClient(
		transport=httpx.ASGITransport(app=app2), base_url="http://h"
	) as client:
		response = await client.post(
			"/v1/teacher/generate", json=_request("later"), headers={"X-Benchmark-Actor": "a"}
		)
		assert response.status_code == (403 if request_quota == 1 else 429)
		assert teacher2.calls == 0
		response = await client.post(
			"/v1/teacher/generate",
			json=_request("new principal"),
			headers={"X-Benchmark-Actor": "b"},
		)
		assert response.status_code == 200
