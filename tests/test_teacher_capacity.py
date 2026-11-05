import asyncio
from datetime import UTC, datetime

import httpx
import pytest
from distillation.schemas import (
	GatewayRequest,
	Message,
	PromptRecord,
	PromptSource,
	TeacherGeneration,
)
from distillation.teacher_capacity import CapacitySLO, preflight, require_capacity_pass


def schedule() -> list[GatewayRequest]:
	return [
		GatewayRequest(
			request_id=str(i),
			account="benign",
			time=i * 0.01,
			max_output_tokens=8,
			prompt=PromptRecord(
				example_id=str(i),
				split="test",
				task="squad",
				source=PromptSource(corpus="fixture", source_id=str(i), **{"license": "test"}),
				prompt=[Message(role="user", content="hello")],
			),
		)
		for i in range(8)
	]


@pytest.mark.asyncio
async def test_capacity_healthy_and_overloaded() -> None:
	async def teacher(request: httpx.Request) -> httpx.Response:
		if request.url.host == "slow":
			await asyncio.sleep(0.04)
		return httpx.Response(
			200,
			json=TeacherGeneration(
				id="one",
				provider="fake",
				model="fake",
				generated_at=datetime.now(UTC),
				message=Message(role="assistant", content="ok"),
				endpoint="fake",
			).model_dump(mode="json"),
		)

	slo = CapacitySLO(50, 30, 0.1, 0, 0, 2, 2)
	kwargs = dict(
		rows=schedule(),
		token="",
		schedule_sha256="fixture",
		slo=slo,
		concurrency=1,
		timeout_seconds=1,
		transport=httpx.MockTransport(teacher),
	)
	healthy = await preflight(url="http://fast", **kwargs)
	overloaded = await preflight(url="http://slow", **kwargs)
	assert healthy["passed"]
	assert not overloaded["passed"]
	assert overloaded["queued_at_end"] > healthy["queued_at_end"]
	require_capacity_pass(healthy, schedule_sha256="fixture")
	with pytest.raises(ValueError):
		require_capacity_pass(overloaded, schedule_sha256="fixture")
	with pytest.raises(ValueError):
		require_capacity_pass(healthy, schedule_sha256="other")
