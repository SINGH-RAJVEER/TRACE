"""LoRA and QLoRA supervised fine-tuning entry point."""

import hashlib
import importlib.metadata
import json
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, model_validator


class TrainingConfig(BaseModel):
	base_model: str = Field(min_length=1)
	base_revision: str = Field(default="main", min_length=1)
	dataset_path: Path
	validation_dataset_path: Path | None = None
	output_dir: Path
	chat_template_path: str | None = None
	eos_token: str | None = None
	max_length: int = Field(default=2048, ge=128)
	epochs: float = Field(default=1.0, gt=0)
	per_device_batch_size: int = Field(default=1, ge=1)
	gradient_accumulation_steps: int = Field(default=16, ge=1)
	learning_rate: float = Field(default=1e-4, gt=0)
	warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
	logging_steps: int = Field(default=10, ge=1)
	save_steps: int = Field(default=100, ge=1)
	eval_steps: int = Field(default=100, ge=1)
	seed: int = 42
	precision: Literal["bf16", "fp16", "fp32"] = "bf16"
	qlora: bool = True
	gradient_checkpointing: bool = True
	lora_rank: int = Field(default=16, ge=1)
	lora_alpha: int = Field(default=32, ge=1)
	lora_dropout: float = Field(default=0.05, ge=0, lt=1)
	target_modules: str | list[str] = "all-linear"

	@model_validator(mode="after")
	def paths_must_exist(self) -> TrainingConfig:
		if not self.dataset_path.is_file():
			raise ValueError(f"dataset does not exist: {self.dataset_path}")
		if self.validation_dataset_path and not self.validation_dataset_path.is_file():
			raise ValueError(f"validation dataset does not exist: {self.validation_dataset_path}")
		if self.chat_template_path and not Path(self.chat_template_path).is_file():
			raise ValueError(f"chat template does not exist: {self.chat_template_path}")
		return self


class ChatTokenizer(Protocol):
	chat_template: str | None

	def apply_chat_template(
		self,
		conversation: list[dict[str, str]],
		*,
		tokenize: bool,
		add_generation_prompt: bool,
	) -> list[int]: ...


def load_training_config(path: Path) -> TrainingConfig:
	with path.open("rb") as source:
		data = tomllib.load(source)
	return TrainingConfig.model_validate(data)


def train(config: TrainingConfig) -> Path:
	"""Train and save an adapter. Heavy dependencies load only for this command."""

	try:
		import torch
		from datasets import load_dataset
		from peft import LoraConfig, prepare_model_for_kbit_training
		from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
		from trl import SFTConfig, SFTTrainer
	except ImportError as error:
		raise RuntimeError(
			"training dependencies are missing; run `uv sync --all-packages --extra train`"
		) from error

	dtype = {
		"bf16": torch.bfloat16,
		"fp16": torch.float16,
		"fp32": torch.float32,
	}[config.precision]
	quantization_config = None
	if config.qlora:
		quantization_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=dtype,
			bnb_4bit_use_double_quant=True,
		)

	model = AutoModelForCausalLM.from_pretrained(
		config.base_model,
		revision=config.base_revision,
		torch_dtype=dtype,
		quantization_config=quantization_config,
		trust_remote_code=False,
	)
	if config.qlora:
		model = prepare_model_for_kbit_training(
			model,
			use_gradient_checkpointing=config.gradient_checkpointing,
		)

	tokenizer = AutoTokenizer.from_pretrained(
		config.base_model,
		revision=config.base_revision,
		trust_remote_code=False,
	)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	if config.chat_template_path:
		tokenizer.chat_template = Path(config.chat_template_path).read_text(encoding="utf-8")
	if tokenizer.chat_template is None:
		raise ValueError("the student tokenizer has no chat template; configure chat_template_path")

	train_dataset = load_dataset("json", data_files=str(config.dataset_path), split="train")
	eval_dataset = None
	if config.validation_dataset_path:
		eval_dataset = load_dataset(
			"json",
			data_files=str(config.validation_dataset_path),
			split="train",
		)
	_reject_overlong_records(train_dataset, tokenizer, config.max_length, "training")
	if eval_dataset is not None:
		_reject_overlong_records(eval_dataset, tokenizer, config.max_length, "validation")

	peft_config = LoraConfig(
		r=config.lora_rank,
		lora_alpha=config.lora_alpha,
		lora_dropout=config.lora_dropout,
		target_modules=config.target_modules,
		bias="none",
		task_type="CAUSAL_LM",
	)
	training_args = SFTConfig(
		output_dir=str(config.output_dir),
		num_train_epochs=config.epochs,
		per_device_train_batch_size=config.per_device_batch_size,
		per_device_eval_batch_size=config.per_device_batch_size,
		gradient_accumulation_steps=config.gradient_accumulation_steps,
		learning_rate=config.learning_rate,
		warmup_ratio=config.warmup_ratio,
		logging_steps=config.logging_steps,
		save_steps=config.save_steps,
		eval_steps=config.eval_steps,
		eval_strategy="steps" if eval_dataset is not None else "no",
		bf16=config.precision == "bf16",
		fp16=config.precision == "fp16",
		gradient_checkpointing=config.gradient_checkpointing,
		seed=config.seed,
		max_length=config.max_length,
		completion_only_loss=True,
		chat_template_path=config.chat_template_path,
		eos_token=config.eos_token,
		report_to="none",
	)
	trainer = SFTTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		processing_class=tokenizer,
		peft_config=peft_config,
	)
	trainer.train()
	trainer.save_model(str(config.output_dir))
	tokenizer.save_pretrained(str(config.output_dir))
	_write_manifest(config)
	return config.output_dir


def _write_manifest(config: TrainingConfig) -> None:
	config.output_dir.mkdir(parents=True, exist_ok=True)
	manifest = {
		"config": config.model_dump(mode="json"),
		"dataset_sha256": _sha256(config.dataset_path),
		"packages": {
			name: importlib.metadata.version(name)
			for name in ("torch", "transformers", "datasets", "trl", "peft", "accelerate")
		},
	}
	(config.output_dir / "training-manifest.json").write_text(
		json.dumps(manifest, indent="\t", sort_keys=True) + "\n",
		encoding="utf-8",
	)


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as source:
		for chunk in iter(lambda: source.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _reject_overlong_records(
	records: Iterable[Mapping[str, list[dict[str, str]]]],
	tokenizer: ChatTokenizer,
	max_length: int,
	label: str,
) -> None:
	overlong = 0
	longest = 0
	for record in records:
		conversation = record["prompt"] + record["completion"]
		token_ids = tokenizer.apply_chat_template(
			conversation,
			tokenize=True,
			add_generation_prompt=False,
		)
		length = len(token_ids)
		longest = max(longest, length)
		if length > max_length:
			overlong += 1
	if overlong:
		raise ValueError(
			f"{label} dataset has {overlong} records over max_length={max_length}; "
			f"longest={longest}. Filter them or raise max_length before training"
		)
