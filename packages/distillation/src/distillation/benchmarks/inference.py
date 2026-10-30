"""Shared non-thinking evaluation contract for served and local models."""

import asyncio
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from distillation.benchmarks.models import ModelConfig
from distillation.schemas import Message, TeacherGeneration, TeacherGenerationRequest, TokenUsage
from distillation.teacher import OpenAICompatibleTeacher, Teacher, TeacherAPIGateway


class LocalModel:
	def __init__(self, config: ModelConfig) -> None:
		try:
			import torch
			from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
		except ImportError as error:
			raise RuntimeError(
				"local inference needs `uv sync --all-packages --extra train`"
			) from error
		self._config = config
		self._torch = torch
		dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
			config.precision
		]
		quantization = (
			# Matches QLoRA training so an adapter runs on the base it was trained on.
			BitsAndBytesConfig(
				load_in_4bit=True,
				bnb_4bit_quant_type="nf4",
				bnb_4bit_compute_dtype=dtype,
				bnb_4bit_use_double_quant=True,
			)
			if config.load_in_4bit
			else None
		)
		self._model = AutoModelForCausalLM.from_pretrained(
			config.model,
			revision=config.revision,
			torch_dtype=dtype,
			device_map=config.device,
			quantization_config=quantization,
			trust_remote_code=False,
		)
		# Merging folds the LoRA update into the weights once instead of on every forward pass.
		# A 4-bit base cannot absorb it exactly, so QLoRA adapters stay attached.
		merged = bool(config.adapter_path) and not config.load_in_4bit
		if config.adapter_path:
			from peft import PeftModel

			self._model = PeftModel.from_pretrained(self._model, str(config.adapter_path))
			if merged:
				self._model = self._model.merge_and_unload()
		self._model.eval()
		self.batch_size = max(1, int(os.environ.get("DISTILL_EVAL_BATCH_SIZE", "16")))
		self.execution: dict[str, str | int | bool] = {
			"backend": "transformers",
			"batch_size": self.batch_size,
			"padding": "left",
			"adapter_merged": merged,
		}
		self._tokenizer = AutoTokenizer.from_pretrained(
			config.tokenizer_model or config.model,
			revision=config.tokenizer_revision or config.revision,
			trust_remote_code=False,
		)
		if config.chat_template_path:
			self._tokenizer.chat_template = config.chat_template_path.read_text()
		if not self._tokenizer.chat_template:
			raise ValueError("local evaluation requires a shared reviewed chat template")
		self._eos_id = self._tokenizer.eos_token_id
		if config.eos_token:
			if config.eos_token not in self._tokenizer.get_vocab():
				raise ValueError("configured evaluation EOS token is not in the tokenizer")
			self._eos_id = self._tokenizer.convert_tokens_to_ids(config.eos_token)
		if config.adapter_path:
			import hashlib
			import json

			manifest = json.loads((config.adapter_path / "training-manifest.json").read_text())
			if expected_template := manifest.get("chat_template_sha256"):
				actual_template = hashlib.sha256(self._tokenizer.chat_template.encode()).hexdigest()
				if actual_template != expected_template:
					raise ValueError(
						"evaluation chat template differs from the trained adapter template"
					)

	async def generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
		return (await self.generate_batch([request]))[0]

	async def generate_batch(
		self, requests: list[TeacherGenerationRequest]
	) -> list[TeacherGeneration]:
		"""Greedy-decodes several requests together, left-padded so every prompt ends aligned."""
		return await asyncio.to_thread(self._generate_batch, requests)

	def _generate_batch(self, requests: list[TeacherGenerationRequest]) -> list[TeacherGeneration]:
		if len({request.max_output_tokens for request in requests}) != 1:
			raise ValueError("a generation batch needs one output budget")
		self._torch.manual_seed(requests[0].seed or 0)
		prompts = [
			self._tokenizer.apply_chat_template(
				[message.model_dump() for message in request.messages],
				tokenize=False,
				add_generation_prompt=True,
				enable_thinking=False,
			)
			for request in requests
		]
		self._tokenizer.padding_side = "left"
		encoded = self._tokenizer(
			prompts, return_tensors="pt", add_special_tokens=False, padding=True
		).to(self._model.device)
		budget = requests[0].max_output_tokens
		width = encoded.input_ids.shape[1]
		context = getattr(self._model.config, "max_position_embeddings", None)
		if context and width + budget > context:
			raise ValueError("evaluation prompt plus output budget exceeds model context")
		pad = self._tokenizer.pad_token_id
		with self._torch.inference_mode():
			output = self._model.generate(
				**encoded,
				do_sample=False,
				max_new_tokens=budget,
				pad_token_id=self._eos_id if pad is None else pad,
				eos_token_id=self._eos_id,
			)
		eos_ids = self._eos_id if isinstance(self._eos_id, list) else [self._eos_id]
		return [
			self._result(
				output[row, width:].tolist(), int(encoded.attention_mask[row].sum()), eos_ids
			)
			for row in range(len(requests))
		]

	def _result(self, tokens: list[int], input_count: int, eos_ids: list[int]) -> TeacherGeneration:
		# Finished rows keep emitting padding until the batch ends; cut them at their first EOS.
		stop = next((index for index, token in enumerate(tokens) if token in eos_ids), None)
		stopped = stop is not None
		if stop is not None:
			tokens = tokens[: stop + 1]
		return TeacherGeneration(
			id=uuid.uuid4().hex,
			provider="local",
			model=self._config.model,
			generated_at=datetime.now(UTC),
			endpoint="local://transformers",
			message=Message(
				role="assistant", content=self._tokenizer.decode(tokens, skip_special_tokens=True)
			),
			finish_reason="stop" if stopped else "length",
			usage=TokenUsage(input_tokens=input_count, output_tokens=len(tokens)),
		)

	async def close(self) -> None:
		del self._model
		if self._torch.cuda.is_available():
			self._torch.cuda.empty_cache()


def make_model(config: ModelConfig) -> Teacher:
	key = os.environ.get(config.api_key_env, "") if config.api_key_env else ""
	if config.api_key_env and not key:
		raise ValueError(f"missing environment variable {config.api_key_env}")
	if config.backend == "teacher_api":
		return TeacherAPIGateway(
			base_url=config.base_url, token=key, timeout_seconds=config.timeout_seconds
		)
	if config.backend == "openai":
		return OpenAICompatibleTeacher(
			base_url=config.base_url,
			api_key=key,
			model=config.model,
			provider="evaluation",
			timeout_seconds=config.timeout_seconds,
		)
	return LocalModel(config)


def validate_adapter(config: ModelConfig) -> list[Path]:
	if not config.adapter_path:
		return []
	import json

	manifest_path = config.adapter_path / "training-manifest.json"
	manifest: dict[str, Any] = json.loads(manifest_path.read_text())
	trained = manifest["config"]
	if trained["base_model"] != config.model or trained["base_revision"] != config.revision:
		raise ValueError("adapter manifest does not match evaluation base model and revision")
	if trained.get("qlora", False) != config.load_in_4bit:
		raise ValueError("load_in_4bit must match the adapter's qlora training setting")
	from distillation.training import _sha256

	paths = [Path(trained["dataset_path"])]
	if _sha256(paths[0]) != manifest["dataset_sha256"]:
		raise ValueError("training data differs from the adapter manifest")
	if trained.get("validation_dataset_path"):
		validation = Path(trained["validation_dataset_path"])
		if _sha256(validation) != manifest.get("validation_dataset_sha256"):
			raise ValueError("validation data differs from the adapter manifest")
		paths.append(validation)
	return paths
