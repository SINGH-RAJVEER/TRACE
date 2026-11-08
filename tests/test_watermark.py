"""Keyed unigram watermark: key derivation, the adapter's logit bias, and detection."""

import json
import random

import httpx
import pytest
from distillation.schemas import Message, TeacherGenerationRequest
from distillation.teacher import OpenAICompatibleTeacher, TeacherWatermarkError
from distillation.watermark import WatermarkSpec, detect, green_ids, key_id, logit_bias

SECRET = bytes(range(16))
SPEC = WatermarkSpec(eligible_vocab=4096)


def test_green_lists_are_keyed_stable_and_near_gamma() -> None:
	first, second = key_id("ns", "alice"), key_id("ns", "bob")
	assert first != second and key_id("ns", "alice") == first
	green = green_ids(SECRET, first, SPEC)
	assert green == green_ids(SECRET, first, SPEC)
	assert abs(len(green) / SPEC.eligible_vocab - SPEC.gamma) < 0.03
	assert green != green_ids(SECRET, second, SPEC)
	assert green != green_ids(bytes(16), first, SPEC)


def test_marked_texts_are_attributed_to_their_key_and_short_texts_abstain() -> None:
	alice, bob = key_id("ns", "alice"), key_id("ns", "bob")
	green = sorted(green_ids(SECRET, alice, SPEC))
	rng = random.Random(7)
	# Each marked text draws 60% of its tokens from alice's green list.
	marked = [
		[rng.choice(green) if rng.random() < 0.6 else rng.randrange(4096) for _ in range(80)]
		for _ in range(20)
	]
	clean = [[rng.randrange(4096) for _ in range(80)] for _ in range(20)]
	report = detect(marked, SECRET, [alice, bob], SPEC)
	decisions = {r["key_id"]: r["decision"] for r in report["results"]}
	assert decisions == {alice: "detected", bob: "not_detected"}
	assert all(
		r["decision"] == "not_detected" for r in detect(clean, SECRET, [alice], SPEC)["results"]
	)
	short = detect(marked[:1], SECRET, [alice], SPEC)["results"][0]
	assert short["decision"] == "insufficient_evidence"


@pytest.mark.asyncio
async def test_adapter_biases_only_keyed_requests_and_refuses_without_a_secret() -> None:
	payloads: list[dict[str, object]] = []

	async def handler(request: httpx.Request) -> httpx.Response:
		payloads.append(json.loads(request.content))
		return httpx.Response(
			200,
			json={"id": "x", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
		)

	key = key_id("ns", "alice")
	request = TeacherGenerationRequest(
		messages=[Message(role="user", content="Question")], watermark_key=key
	)
	teacher = OpenAICompatibleTeacher(
		base_url="https://teacher.example/v1",
		api_key="",
		model="qwen",
		provider="test",
		transport=httpx.MockTransport(handler),
		watermark=SPEC,
		watermark_secret=SECRET,
	)
	plain = OpenAICompatibleTeacher(
		base_url="https://teacher.example/v1",
		api_key="",
		model="qwen",
		provider="test",
		transport=httpx.MockTransport(handler),
	)
	try:
		marked = await teacher.generate(request)
		unmarked = await teacher.generate(request.model_copy(update={"watermark_key": None}))
		with pytest.raises(TeacherWatermarkError):
			await plain.generate(request)
	finally:
		await teacher.close()
		await plain.close()
	assert payloads[0]["logit_bias"] == logit_bias(SECRET, key, SPEC)
	assert "logit_bias" not in payloads[1] and len(payloads) == 2
	assert marked.watermarked is True and unmarked.watermarked is False


def test_unmarked_zipf_pools_rarely_test_positive_against_random_keys() -> None:
	rng = random.Random(3)
	weights = [1 / (rank + 1) for rank in range(4096)]
	# Natural text reuses a few tokens in every text; per-text counting overstated evidence.
	pool = [rng.choices(range(4096), weights=weights, k=120) for _ in range(60)]
	keys = [key_id("ns", f"p{i}") for i in range(200)]
	report = detect(pool, SECRET, keys[:1], SPEC)
	assert report["results"][0]["tokens"] >= 200
	positives = sum(
		detect(pool, SECRET, [key], SPEC, alpha=0.01)["results"][0]["decision"] == "detected"
		for key in keys
	)
	assert positives <= 6
