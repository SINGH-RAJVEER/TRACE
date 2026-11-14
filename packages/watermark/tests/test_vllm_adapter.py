"""CPU tensor tests of the pinned vLLM batch contract, not a live-engine qualification.

The small engine-interface shim permits testing without installing CUDA vLLM in the app
environment. Run with a CPU torch interpreter as documented in docs/watermark-vllm.md.
"""

import importlib
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trace_watermark.bias import apply_bias
from trace_watermark.detect import detect_candidates, detect_texts
from trace_watermark.serving import KgwSpec, principal_key, request_args

torch = pytest.importorskip("torch")
SECRET = bytes(range(32))  # Public test fixture, never a deployment credential.
ALICE = "a" * 32
BOB = "b" * 32
SPEC = KgwSpec(vocab_size=512)


class Direction(Enum):
	UNIDIRECTIONAL = 1
	SWAP = 2


@dataclass
class Update:
	batch_size: int
	removed: list[int]
	added: list[Any]
	moved: list[Any]


@pytest.fixture
def processor_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
	for key in list(os.environ):
		if key.startswith("TEACHER_WATERMARK_"):
			monkeypatch.delenv(key)
	key_file = tmp_path / "fixture.watermark.key"
	key_file.write_text(SECRET.hex())
	monkeypatch.setenv("TEACHER_WATERMARK_KEY_FILE", str(key_file))
	monkeypatch.setenv("TEACHER_WATERMARK_VOCAB_SIZE", str(SPEC.vocab_size))
	shim = ModuleType("vllm.v1.sample.logits_processor")
	shim.LogitsProcessor = object  # type: ignore[attr-defined]
	shim.BatchUpdate = Update  # type: ignore[attr-defined]
	shim.MoveDirectionality = Direction  # type: ignore[attr-defined]
	monkeypatch.setitem(sys.modules, shim.__name__, shim)
	monkeypatch.delitem(sys.modules, "trace_watermark.vllm", raising=False)
	module = importlib.import_module("trace_watermark.vllm")
	monkeypatch.setattr(module, "version", lambda _: "0.28.0")
	return module


def config(vocab_size: int = 512, speculative: Any = None, async_scheduling: bool = False) -> Any:
	return SimpleNamespace(
		model_config=SimpleNamespace(get_vocab_size=lambda: vocab_size),
		speculative_config=speculative,
		scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
	)


def params(key: str | None = ALICE) -> Any:
	return SimpleNamespace(
		extra_args=request_args(SECRET, key, SPEC, "0" * 32) if key else None,
		logit_bias=None,
	)


def test_engine_bias_matches_reference_and_uses_live_history(processor_module: Any) -> None:
	processor = processor_module.KgwLogitsProcessor(config(), torch.device("cpu"), False)
	prompt = [4, 12]
	output: list[int] = []
	processor.update_state(Update(2, [], [(0, params(), prompt, output)], []))
	logits = torch.randn((2, SPEC.vocab_size), generator=torch.Generator().manual_seed(31))
	logits[0, 17] = -torch.inf
	for _ in range(3):
		result = processor.apply(logits.clone())
		expected = apply_bias(
			logits[0].numpy(), prompt + output, principal_key(SECRET, ALICE, SPEC), SPEC.profile()
		)
		np.testing.assert_array_equal(result[0].numpy(), expected)
		assert torch.equal(result[1], logits[1])
		assert torch.equal(processor.apply(logits.clone()), result)  # Retry is not a sample.
		output.append(int(result[0].argmax()))
		processor.update_state(None)
	assert processor.is_argmax_invariant() is False


def test_reordering_compaction_and_slot_reuse_do_not_leak_keys(processor_module: Any) -> None:
	processor = processor_module.KgwLogitsProcessor(config(), torch.device("cpu"), False)
	processor.update_state(
		Update(3, [], [(0, params(ALICE), [1, 2], []), (2, params(BOB), [7, 8], [])], [])
	)
	original = processor.apply(torch.zeros((3, 512)))
	processor.update_state(Update(3, [], [], [(0, 2, Direction.SWAP)]))
	swapped = processor.apply(torch.zeros((3, 512)))
	assert torch.equal(swapped[0], original[2]) and torch.equal(swapped[2], original[0])
	processor.update_state(Update(2, [0], [], [(2, 0, Direction.UNIDIRECTIONAL)]))
	compacted = processor.apply(torch.zeros((2, 512)))
	assert torch.equal(compacted[0], original[0])
	assert torch.count_nonzero(compacted[1]) == 0
	# Removal and addition in the same slot, and unkeyed replacement, both discard old state.
	processor.update_state(Update(2, [0], [(0, params(BOB), [7, 8], [])], []))
	assert torch.equal(processor.apply(torch.zeros((2, 512)))[0], original[2])
	processor.update_state(Update(1, [1], [(0, params(None), [], [])], []))
	assert torch.count_nonzero(processor.apply(torch.zeros((1, 512)))) == 0


def test_engine_generated_tokens_are_detected_with_the_same_derived_key(
	processor_module: Any,
) -> None:
	processor = processor_module.KgwLogitsProcessor(config(), torch.device("cpu"), False)
	prompt = [11, 23]
	output: list[int] = []
	processor.update_state(Update(1, [], [(0, params(), prompt, output)], []))
	generator = torch.Generator().manual_seed(21)
	for _ in range(256):
		logits = torch.randn((1, 512), generator=generator)
		output.append(int(processor.apply(logits).argmax()))
	results = detect_candidates(
		output,
		[principal_key(SECRET, key, SPEC) for key in (ALICE, BOB)],
		SPEC.profile(),
		prompt_ids=prompt,
		alpha=1e-3,
		min_effective=200,
	)
	assert [result.decision for result in results] == ["detected", "not_detected"]
	once = detect_texts(
		[output],
		[principal_key(SECRET, ALICE, SPEC)],
		SPEC.profile(),
		alpha=1e-3,
		min_effective=200,
	)[0]
	repeated = detect_texts(
		[output] * 20,
		[principal_key(SECRET, ALICE, SPEC)],
		SPEC.profile(),
		alpha=1e-3,
		min_effective=200,
	)[0]
	assert once.unique_contexts == repeated.unique_contexts
	assert once.raw_p_value == repeated.raw_p_value


def test_engine_fails_closed_on_unsupported_configuration(processor_module: Any) -> None:
	with pytest.raises(ValueError, match="vocabulary"):
		processor_module.KgwLogitsProcessor(config(1024), torch.device("cpu"), False)
	with pytest.raises(ValueError, match="speculative"):
		processor_module.KgwLogitsProcessor(config(speculative={}), torch.device("cpu"), False)
	with pytest.raises(ValueError, match="synchronous"):
		processor_module.KgwLogitsProcessor(
			config(async_scheduling=True), torch.device("cpu"), False
		)
	processor = processor_module.KgwLogitsProcessor(config(), torch.device("cpu"), False)
	processor.update_state(Update(1, [], [(0, params(), [1, 2], [])], []))
	with pytest.raises(ValueError, match="shape"):
		processor.apply(torch.zeros((2, 512)))
	for field, value in (
		("trace_watermark_spec", "{}"),
		("trace_watermark_auth", "wrong"),
		("trace_watermark_nonce", 1),
		("trace_watermark_key", "wrong"),
	):
		request = params()
		request.extra_args[field] = value
		with pytest.raises(ValueError):
			processor.validate_params(request)
