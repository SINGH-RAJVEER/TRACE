"""Generation through the batch processor followed by detection, on a synthetic language model.

The model is a seeded table of bigram logits. These tests establish that the modules agree with
each other. They say nothing about a real model, tokenizer or task.
"""

import numpy as np
from numpy.typing import NDArray

from trace_watermark.bias import KgwBatchProcessor
from trace_watermark.core import WatermarkKey, WatermarkProfile
from trace_watermark.detect import detect, detect_candidates

VOCAB = 2000
TENANT_A = WatermarkKey("tenant-a", 0, b"public-test-material-tenant-a")
TENANT_B = WatermarkKey("tenant-b", 0, b"public-test-material-tenant-b")
ALPHA = 1e-3
MIN_EFFECTIVE = 32


def profile(delta: float, width: int = 2) -> WatermarkProfile:
	return WatermarkProfile(gamma=0.25, delta=delta, context_width=width, vocab_size=VOCAB)


def bigram_logits(previous: int, sharpness: float) -> NDArray[np.float64]:
	return np.random.default_rng(previous).standard_normal(VOCAB) * sharpness


def sample(logits: NDArray[np.float64], rng: np.random.Generator) -> int:
	return int(np.argmax(logits + rng.gumbel(size=logits.shape)))


def generate(
	key: WatermarkKey | None,
	used: WatermarkProfile,
	*,
	length: int,
	seed: int,
	sharpness: float = 1.0,
) -> tuple[list[int], list[int]]:
	rng = np.random.default_rng(seed)
	prompt = [int(token) for token in rng.integers(0, VOCAB, 8)]
	processor = KgwBatchProcessor(used)
	processor.add_request(
		0, "request", prompt, key if key is not None else TENANT_A, enabled=key is not None
	)
	output: list[int] = []
	for _ in range(length):
		previous = output[-1] if output else prompt[-1]
		row = bigram_logits(previous, sharpness)[np.newaxis]
		token = sample(processor.apply(row, [0])[0], rng)
		processor.record_token(0, token)
		output.append(token)
	return prompt, output


def test_watermarked_output_is_detected_and_attributed_to_the_right_tenant() -> None:
	used = profile(delta=2.0)
	prompt, output = generate(TENANT_A, used, length=200, seed=1)
	results = detect_candidates(
		output,
		[TENANT_A, TENANT_B],
		used,
		alpha=ALPHA,
		min_effective=MIN_EFFECTIVE,
		prompt_ids=prompt,
	)
	by_key = {result.candidate_key_id: result.decision for result in results}
	assert by_key == {"tenant-a": "detected", "tenant-b": "not_detected"}


def test_unwatermarked_output_stays_clean_under_the_null() -> None:
	used = profile(delta=2.0)
	hits = 0
	for seed in range(40):
		prompt, output = generate(None, used, length=120, seed=seed)
		result = detect(
			output, TENANT_A, used, prompt_ids=prompt, alpha=ALPHA, min_effective=MIN_EFFECTIVE
		)
		hits += result.decision == "detected"
	assert hits == 0


def test_detection_without_the_prompt_still_works_and_scores_fewer_tokens() -> None:
	used = profile(delta=2.0)
	prompt, output = generate(TENANT_A, used, length=200, seed=2)
	with_prompt = detect(
		output, TENANT_A, used, prompt_ids=prompt, alpha=ALPHA, min_effective=MIN_EFFECTIVE
	)
	text_only = detect(output, TENANT_A, used, alpha=ALPHA, min_effective=MIN_EFFECTIVE)
	assert with_prompt.scored_tokens == text_only.scored_tokens + used.context_width
	assert text_only.decision == "detected"


def test_low_entropy_output_carries_less_evidence() -> None:
	used = profile(delta=1.0)
	green_rate = {}
	for sharpness in (1.0, 12.0):
		prompt, output = generate(TENANT_A, used, length=300, seed=3, sharpness=sharpness)
		result = detect(output, TENANT_A, used, prompt_ids=prompt, alpha=ALPHA, min_effective=1)
		green_rate[sharpness] = result.green_count / result.unique_contexts
	assert green_rate[1.0] > green_rate[12.0]


def test_interleaved_batch_matches_sequential_generation() -> None:
	used = profile(delta=2.0)
	expected = {
		slot: generate(key, used, length=30, seed=slot)
		for slot, key in ((0, TENANT_A), (1, TENANT_B))
	}
	rngs = {}
	processor = KgwBatchProcessor(used)
	for slot, key in ((0, TENANT_A), (1, TENANT_B)):
		rngs[slot] = np.random.default_rng(slot)
		prompt = [int(token) for token in rngs[slot].integers(0, VOCAB, 8)]
		processor.add_request(slot, f"request-{slot}", prompt, key)
	outputs: dict[int, list[int]] = {0: [], 1: []}
	for step in range(30):
		order = [1, 0] if step % 2 else [0, 1]
		rows = np.stack(
			[
				bigram_logits(outputs[slot][-1] if outputs[slot] else expected[slot][0][-1], 1.0)
				for slot in order
			]
		)
		biased = processor.apply(rows, order)
		for row, slot in zip(biased, order, strict=True):
			token = sample(row, rngs[slot])
			processor.record_token(slot, token)
			outputs[slot].append(token)
	assert outputs[0] == expected[0][1]
	assert outputs[1] == expected[1][1]
