"""Random request lifecycles, checked against requests generated one at a time.

Each logical request follows a script of tokens. However slots get recycled and rows reordered, its
biased logits must equal what apply_bias gives for that request's own history.
"""

import random
from typing import TypedDict

import numpy as np
import pytest

from trace_watermark.bias import KgwBatchProcessor, apply_bias
from trace_watermark.core import WatermarkKey, WatermarkProfile

PROFILE = WatermarkProfile(gamma=0.25, delta=1.5, context_width=2, vocab_size=128)
KEYS = [WatermarkKey(f"tenant-{i}", i, f"public-test-tenant-key-{i}".encode()) for i in range(3)]


class LiveRequest(TypedDict):
	id: str
	key: WatermarkKey
	history: list[int]
	enabled: bool
	biased: int
	skipped: int
	disabled: int


@pytest.mark.parametrize("seed", range(20))
def test_random_lifecycle_matches_each_request_alone(seed: int) -> None:
	rng = random.Random(seed)
	logits_rng = np.random.default_rng(seed)
	processor = KgwBatchProcessor(PROFILE)
	live: dict[int, LiveRequest] = {}
	finished = 0
	serial = 0
	for _ in range(300):
		action = rng.random()
		empty = [slot for slot in range(8) if slot not in live]
		if action < 0.2 and empty:
			slot = rng.choice(empty)
			prompt = [rng.randrange(128) for _ in range(rng.randrange(4))]
			request: LiveRequest = {
				"id": f"request-{serial}",
				"key": rng.choice(KEYS),
				"history": prompt,
				"enabled": rng.random() > 0.2,
				"biased": 0,
				"skipped": 0,
				"disabled": 0,
			}
			serial += 1
			processor.add_request(
				slot, request["id"], prompt, request["key"], enabled=request["enabled"]
			)
			live[slot] = request
		elif action < 0.3 and live:
			slot = rng.choice(list(live))
			request = live.pop(slot)
			status = processor.remove_request(slot)
			assert status.request_id == request["id"]
			assert status.rows_biased == request["biased"]
			assert status.rows_skipped_no_context == request["skipped"]
			assert status.rows_disabled == request["disabled"]
			finished += 1
		elif action < 0.4 and live and empty:
			source = rng.choice(list(live))
			target = rng.choice(empty)
			processor.move_request(source, target)
			live[target] = live.pop(source)
		elif live:
			order = rng.sample(list(live), rng.randint(1, len(live)))
			rows = logits_rng.standard_normal((len(order), 128))
			output = processor.apply(rows, order)
			for row, slot in enumerate(order):
				request = live[slot]
				if not request["enabled"]:
					expected = rows[row]
					request["disabled"] += 1
				else:
					expected = apply_bias(rows[row], request["history"], request["key"], PROFILE)
					has_context = len(request["history"]) >= PROFILE.context_width
					request["biased" if has_context else "skipped"] += 1
				assert np.array_equal(output[row], expected)
				token = rng.randrange(128)
				processor.record_token(slot, token)
				request["history"].append(token)
	assert finished > 0
