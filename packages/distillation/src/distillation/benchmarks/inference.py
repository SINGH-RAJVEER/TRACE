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
			BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
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
		if config.adapter_path:
			from peft import PeftModel

			self._model = PeftModel.from_pretrained(self._model, str(config.adapter_path))
		self._model.eval()
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
		return await asyncio.to_thread(self._generate, request)

	def _generate(self, request: TeacherGenerationRequest) -> TeacherGeneration:
		self._torch.manual_seed(request.seed or 0)
		prompt = self._tokenizer.apply_chat_template(
			[message.model_dump() for message in request.messages],
			tokenize=False,
			add_generation_prompt=True,
			enable_thinking=False,
		)
		encoded = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
			self._model.device
		)
		input_count = encoded.input_ids.shape[1]
		context = getattr(self._model.config, "max_position_embeddings", None)
		if context and input_count + request.max_output_tokens > context:
			raise ValueError("evaluation prompt plus output budget exceeds model context")
		with self._torch.inference_mode():
			output = self._model.generate(
				**encoded,
				do_sample=False,
				max_new_tokens=request.max_output_tokens,
				pad_token_id=self._eos_id,
				eos_token_id=self._eos_id,
			)
		tokens = output[0, input_count:]
		eos_ids = self._eos_id
		if not isinstance(eos_ids, list):
			eos_ids = [eos_ids]
		stopped = bool(len(tokens) and int(tokens[-1]) in eos_ids)
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
