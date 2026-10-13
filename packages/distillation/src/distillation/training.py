"""LoRA and QLoRA supervised fine-tuning entry point."""

import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from distillation.io import canonical_json, read_jsonl
from distillation.schemas import SFTRecord


class TrainingConfig(BaseModel):
	model_config = ConfigDict(extra="forbid")

	base_model: str = Field(min_length=1)
	base_revision: str = Field(default="main", min_length=1)
	dataset_path: Path
	validation_dataset_path: Path | None = None
	output_dir: Path
	chat_template_path: str | None = None
	eos_token: str | None = None
	chat_template_kwargs: dict[str, bool | str | int | float] = Field(
		default_factory=lambda: dict[str, bool | str | int | float](enable_thinking=False)
	)
	max_length: int = Field(default=2048, ge=128)
	epochs: float = Field(default=1.0, gt=0)
	per_device_batch_size: int = Field(default=1, ge=1)
	gradient_accumulation_steps: int = Field(default=16, ge=1)
	learning_rate: float = Field(default=1e-4, gt=0)
	warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
	logging_steps: int = Field(default=10, ge=1)
	save_steps: int = Field(default=100, ge=1)
	eval_steps: int = Field(default=100, ge=1)
	eval_strategy: Literal["epoch", "steps"] = "epoch"
	select_best_checkpoint: bool = True
	save_total_limit: int = Field(default=2, ge=1)
	seed: int = 42
	precision: Literal["bf16", "fp16", "fp32"] = "bf16"
	qlora: bool = True
	loss_type: Literal["nll", "dft"] = "nll"
	attn_implementation: Literal["sdpa", "eager", "flash_attention_2"] = "sdpa"
	activation_offloading: bool = False
	gradient_checkpointing: bool = True
	selective_gradient_checkpointing: bool = False
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
		if (
			self.validation_dataset_path
			and self.select_best_checkpoint
			and self.eval_strategy == "steps"
			and self.save_steps % self.eval_steps
		):
			raise ValueError(
				"save_steps must be a multiple of eval_steps for best-checkpoint selection"
			)
		if self.attn_implementation == "flash_attention_2" and self.precision == "fp32":
			raise ValueError("FlashAttention 2 requires bf16 or fp16 precision")
		if self.selective_gradient_checkpointing and not self.gradient_checkpointing:
			raise ValueError("selective checkpointing requires gradient_checkpointing=true")
		if self.target_modules == "lm_head" or (
			isinstance(self.target_modules, list)
			and any(module.endswith("lm_head") for module in self.target_modules)
		):
			raise ValueError("the fused loss does not support lm_head adapters")
		return self


class ChatTokenizer(Protocol):
	chat_template: str | None
	eos_token_id: int | None

	def apply_chat_template(
		self,
		conversation: list[dict[str, str]],
		*,
		tokenize: bool,
		add_generation_prompt: bool,
		**kwargs: bool | str | int | float,
	) -> list[int]: ...


def load_training_config(path: Path) -> TrainingConfig:
	with path.open("rb") as source:
		data = tomllib.load(source)
	return TrainingConfig.model_validate(data)


def _load_tokenizer(config: TrainingConfig) -> Any:
	try:
		from transformers import AutoTokenizer
	except ImportError as error:
		raise RuntimeError("preflight needs `uv sync --all-packages --extra train`") from error
	tokenizer = AutoTokenizer.from_pretrained(
		config.base_model,
		revision=config.base_revision,
		trust_remote_code=False,
	)
	if config.chat_template_path:
		tokenizer.chat_template = Path(config.chat_template_path).read_text(encoding="utf-8")
	if tokenizer.chat_template is None:
		raise ValueError("the student tokenizer has no chat template; configure chat_template_path")
	if config.eos_token:
		if config.eos_token not in tokenizer.get_vocab():
			raise ValueError("configured eos_token is not in the student vocabulary")
		tokenizer.eos_token = config.eos_token
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	return tokenizer


def preflight_training(config: TrainingConfig) -> dict[str, Any]:
	"""Check data and tokenizer serialization without loading model weights or training."""
	_validate_dataset_splits(config)
	tokenizer = _load_tokenizer(config)
	datasets = {}
	for label, path in (
		("training", config.dataset_path),
		("validation", config.validation_dataset_path),
	):
		if path is not None:
			rows = [row.model_dump(mode="json") for row in read_jsonl(path, SFTRecord)]
			datasets[label] = {
				"sha256": _sha256(path),
				**_reject_overlong_records(
					rows, tokenizer, config.max_length, label, config.chat_template_kwargs
				),
			}
	return {
		"base_model": config.base_model,
		"base_revision": config.base_revision,
		"chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
		"chat_template_kwargs": config.chat_template_kwargs,
		"eos_token": tokenizer.eos_token,
		"datasets": datasets,
		"validation_enabled": config.validation_dataset_path is not None,
		"loss_type": config.loss_type,
		"attention": config.attn_implementation,
	}


def train(config: TrainingConfig) -> Path:
	"""Train and save an adapter. Heavy dependencies load only for this command."""

	try:
		import torch
		from datasets import load_dataset
		from peft import LoraConfig, prepare_model_for_kbit_training
		from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
		from trl import SFTConfig, SFTTrainer
	except ImportError as error:
		raise RuntimeError(
			"training dependencies are missing; run `uv sync --all-packages --extra train`"
		) from error
	set_seed(config.seed)
	_validate_dataset_splits(config)
	dataset_hashes = {"training": _sha256(config.dataset_path)}
	if config.validation_dataset_path:
		dataset_hashes["validation"] = _sha256(config.validation_dataset_path)
	if config.qlora and not torch.cuda.is_available():
		raise ValueError("QLoRA requires a compatible CUDA GPU; set qlora=false for CPU LoRA")
	if config.activation_offloading and not torch.cuda.is_available():
		raise ValueError("activation offloading requires a CUDA GPU")
	if (
		config.precision == "bf16"
		and torch.cuda.is_available()
		and not torch.cuda.is_bf16_supported()
	):
		raise ValueError("GPU does not support BF16; choose fp16 or fp32")

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

	tokenizer = _load_tokenizer(config)

	train_dataset = load_dataset("json", data_files=str(config.dataset_path), split="train")
	eval_dataset = None
	if config.validation_dataset_path:
		eval_dataset = load_dataset(
			"json",
			data_files=str(config.validation_dataset_path),
			split="train",
		)
	preflight = {
		"training": _reject_overlong_records(
			train_dataset, tokenizer, config.max_length, "training", config.chat_template_kwargs
		)
	}
	train_dataset = train_dataset.add_column(
		"chat_template_kwargs", [config.chat_template_kwargs] * len(train_dataset)
	)
	if eval_dataset is not None:
		preflight["validation"] = _reject_overlong_records(
			eval_dataset, tokenizer, config.max_length, "validation", config.chat_template_kwargs
		)
		eval_dataset = eval_dataset.add_column(
			"chat_template_kwargs", [config.chat_template_kwargs] * len(eval_dataset)
		)

	model = AutoModelForCausalLM.from_pretrained(
		config.base_model,
		revision=config.base_revision,
		dtype=dtype,
		quantization_config=quantization_config,
		attn_implementation=config.attn_implementation,
		trust_remote_code=False,
	)
	if config.qlora:
		model = prepare_model_for_kbit_training(
			model,
			use_gradient_checkpointing=config.gradient_checkpointing,
			gradient_checkpointing_kwargs={"use_reentrant": False},
	)
	model.config.use_cache = False
	if tokenizer.eos_token_id is not None:
		model.config.eos_token_id = tokenizer.eos_token_id
		model.generation_config.eos_token_id = tokenizer.eos_token_id
	if tokenizer.pad_token_id is not None:
		model.config.pad_token_id = tokenizer.pad_token_id
		model.generation_config.pad_token_id = tokenizer.pad_token_id

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
		warmup_steps=config.warmup_ratio,
		logging_steps=config.logging_steps,
		save_steps=config.save_steps,
		eval_steps=config.eval_steps,
		eval_strategy=config.eval_strategy if eval_dataset is not None else "no",
		save_strategy=config.eval_strategy if eval_dataset is not None else "steps",
		load_best_model_at_end=eval_dataset is not None and config.select_best_checkpoint,
		metric_for_best_model="eval_loss",
		greater_is_better=False,
		save_total_limit=config.save_total_limit,
		bf16=config.precision == "bf16",
		fp16=config.precision == "fp16",
		gradient_checkpointing=config.gradient_checkpointing,
		gradient_checkpointing_kwargs={
			"use_reentrant": False,
			**({"selective": True} if config.selective_gradient_checkpointing else {}),
		},
		seed=config.seed,
		data_seed=config.seed,
		max_length=config.max_length,
		completion_only_loss=True,
		packing=False,
		loss_type=config.loss_type,
		activation_offloading=config.activation_offloading,
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
	evaluation = trainer.evaluate() if eval_dataset is not None else None
	trainer.save_model(str(config.output_dir))
	tokenizer.save_pretrained(str(config.output_dir))
	_write_manifest(
		config,
		details={
			"dataset_hashes": dataset_hashes,
			"preflight": preflight,
			"resolved_base_revision": getattr(model.config, "_commit_hash", None),
			"chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
			"eos_token": tokenizer.eos_token,
			"eos_token_id": tokenizer.eos_token_id,
			"best_checkpoint": trainer.state.best_model_checkpoint,
			"best_validation_loss": trainer.state.best_metric,
			"final_evaluation": evaluation,
			"global_step": trainer.state.global_step,
			"training_args": training_args.to_dict(),
			"runtime": {
				"python": sys.version,
				"platform": platform.platform(),
				"cuda": torch.version.cuda,
				"gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
			},
		},
	)
	return config.output_dir


def _write_manifest(config: TrainingConfig, *, details: dict[str, Any]) -> None:
	config.output_dir.mkdir(parents=True, exist_ok=True)
	manifest: dict[str, Any] = {
		"config": config.model_dump(mode="json"),
		"dataset_sha256": details["dataset_hashes"]["training"],
		"validation_dataset_sha256": details["dataset_hashes"].get("validation"),
		"packages": {
			name: importlib.metadata.version(name)
			for name in ("torch", "transformers", "datasets", "trl", "peft", "accelerate")
		},
		**details,
	}
	for name in ("bitsandbytes", "distillation-core"):
		with suppress(importlib.metadata.PackageNotFoundError):
			manifest["packages"][name] = importlib.metadata.version(name)
	for name, command in (
		("code_revision", ["jj", "log", "-r", "@", "--no-graph", "-T", "commit_id"]),
		("nvidia_driver", ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
	):
		try:
			manifest[name] = subprocess.check_output(
				command, text=True, stderr=subprocess.DEVNULL, timeout=10
			).strip()
		except OSError, subprocess.SubprocessError:
			manifest[name] = None
	for label, path in (
		("training", config.dataset_path),
		("validation", config.validation_dataset_path),
	):
		if path and path.with_suffix(".manifest.json").is_file():
			manifest[f"{label}_preparation"] = json.loads(
				path.with_suffix(".manifest.json").read_text()
			)
	if Path("uv.lock").is_file():
		manifest["uv_lock_sha256"] = _sha256(Path("uv.lock"))
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
	chat_template_kwargs: dict[str, bool | str | int | float] | None = None,
) -> dict[str, int | float]:
	overlong = 0
	longest = 0
	count = 0
	completion_tokens = 0
	lengths: list[int] = []
	for record in records:
		count += 1
		conversation = record["prompt"] + record["completion"]
		token_ids = tokenizer.apply_chat_template(
			conversation,
			tokenize=True,
			add_generation_prompt=False,
			return_dict=False,
			**(chat_template_kwargs or {}),
		)
		prompt_ids = tokenizer.apply_chat_template(
			record["prompt"],
			tokenize=True,
			add_generation_prompt=True,
			return_dict=False,
			**(chat_template_kwargs or {}),
		)
		if token_ids[: len(prompt_ids)] != prompt_ids:
			raise ValueError(f"{label} record {count}: chat template changes the prompt prefix")
		completion = token_ids[len(prompt_ids) :]
		if not completion:
			raise ValueError(f"{label} record {count}: no supervised completion tokens")
		if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in completion:
			raise ValueError(f"{label} record {count}: completion has no EOS token")
		completion_tokens += len(completion)
		length = len(token_ids)
		lengths.append(length)
		longest = max(longest, length)
		if length > max_length:
			overlong += 1
	if overlong:
		raise ValueError(
			f"{label} dataset has {overlong} records over max_length={max_length}; "
			f"longest={longest}. Filter them or raise max_length before training"
		)
	if not count:
		raise ValueError(f"{label} dataset is empty")
	lengths.sort()
	total_tokens = sum(lengths)
	return {
		"records": count,
		"longest": longest,
		"shortest": lengths[0],
		"length_p50": statistics.median(lengths),
		"length_p95": lengths[math.ceil(0.95 * count) - 1],
		"total_tokens": total_tokens,
		"completion_tokens": completion_tokens,
		"supervised_fraction": completion_tokens / total_tokens,
	}


def _validate_dataset_splits(config: TrainingConfig) -> None:
	training = read_jsonl(config.dataset_path, SFTRecord)
	if not training:
		raise ValueError("training dataset is empty")
	if config.validation_dataset_path:
		validation = read_jsonl(config.validation_dataset_path, SFTRecord)
		if not validation:
			raise ValueError("validation dataset is empty")
		train_prompts = {canonical_json(row.model_dump(mode="json")["prompt"]) for row in training}
		if any(
			canonical_json(row.model_dump(mode="json")["prompt"]) in train_prompts
			for row in validation
		):
			raise ValueError("training and validation datasets contain overlapping prompts")
