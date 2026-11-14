"""Engine-side KGW logits processor for the pinned vLLM 0.28.0 batch interface."""

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import torch

# vLLM is installed only in the separate Python 3.12 inference environment.
from vllm.v1.sample.logits_processor import (  # type: ignore[import-not-found]
	BatchUpdate,
	LogitsProcessor,
	MoveDirectionality,
)

from trace_watermark.core import WatermarkKey, context_of, context_seed, green_mask
from trace_watermark.serving import VLLM_VERSION, KgwSpec, engine_secret, validate_request


@dataclass
class _Request:
	spec: KgwSpec
	key: WatermarkKey
	prompt: list[int]
	# Keep the live list owned by vLLM. Never infer sampling progress from apply() calls.
	output: list[int]


class KgwLogitsProcessor(LogitsProcessor):  # type: ignore[misc]
	"""Bias keyed requests before greedy or stochastic sampling; never alter unkeyed rows."""

	@classmethod
	def validate_params(cls, sampling_params: Any) -> None:
		if validate_request(sampling_params.extra_args) is not None and sampling_params.logit_bias:
			raise ValueError("KGW and static logit_bias cannot be combined")

	def __init__(self, vllm_config: Any, device: torch.device, is_pin_memory: bool) -> None:
		if version("vllm") != VLLM_VERSION:
			raise ValueError(f"KGW requires vLLM {VLLM_VERSION}")
		if vllm_config.speculative_config is not None:
			raise ValueError("KGW does not support speculative decoding")
		if vllm_config.scheduler_config.async_scheduling:
			raise ValueError("KGW requires synchronous scheduling")
		self.spec = KgwSpec.from_env()
		engine_secret()  # Fail startup if the shared key is missing or invalid.
		if vllm_config.model_config.get_vocab_size() != self.spec.vocab_size:
			raise ValueError("KGW vocabulary does not match the engine model")
		self._requests: dict[int, _Request] = {}
		self._batch_size = 0

	def is_argmax_invariant(self) -> bool:
		return False

	def update_state(self, batch_update: BatchUpdate | None) -> None:
		if batch_update is None:
			return
		# The engine can remove and replace a slot in one step. Follow its documented ordering.
		for index in batch_update.removed:
			self._requests.pop(index, None)
		for index, params, prompt, output in batch_update.added:
			self._requests.pop(index, None)
			self.validate_params(params)
			if resolved := validate_request(params.extra_args):
				spec, key = resolved
				if spec != self.spec:
					raise ValueError("KGW configuration changed; restart vLLM")
				if prompt is None:
					raise ValueError("KGW requires prompt token IDs")
				self._requests[index] = _Request(spec, key, prompt, output)
		for source, destination, direction in batch_update.moved:
			left = self._requests.pop(source, None)
			right = self._requests.pop(destination, None)
			if left is not None:
				self._requests[destination] = left
			if direction == MoveDirectionality.SWAP and right is not None:
				self._requests[source] = right
		self._batch_size = batch_update.batch_size

	def apply(self, logits: torch.Tensor) -> torch.Tensor:
		if not self._requests:
			return logits
		if logits.shape != (self._batch_size, self.spec.vocab_size):
			raise ValueError("KGW logits shape does not match the engine batch and vocabulary")
		# Keep logits on their device. Only a boolean green mask crosses CPU -> accelerator.
		for index, request in self._requests.items():
			width = request.spec.context_width
			tail = request.output[-width:]
			if len(tail) < width:
				tail = request.prompt[-(width - len(tail)) :] + tail
			context = context_of(tail, width)
			if context is None:
				continue
			profile = request.spec.profile()
			mask = green_mask(context_seed(request.key, profile, context), profile)
			device_mask = torch.as_tensor(mask, device=logits.device)
			# Negative infinity masks remain masked. vLLM uses float32 sampling logits.
			logits[index].add_(device_mask.to(logits.dtype) * profile.delta)
		return logits
