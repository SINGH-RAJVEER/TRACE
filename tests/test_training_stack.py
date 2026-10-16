"""Opt-in local training checks. Tiny checkpoints are built offline in temporary directories."""

import hashlib
import json
import math
import os
import platform
import signal
from pathlib import Path
from typing import Any

import pytest
from distillation.io import write_jsonl
from distillation.schemas import Message, SFTRecord
from distillation.training import TrainingConfig, TrainingInterrupted, train

pytestmark = pytest.mark.skipif(
	os.environ.get("DISTILL_TEST_TRAINING_STACK") != "1",
	reason="set DISTILL_TEST_TRAINING_STACK=1 with the train extra to run native training checks",
)


@pytest.fixture
def tiny_checkpoint(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[Path, Any, Any, Any]:
	import torch
	from safetensors.torch import load_file, save_file
	from tokenizers import Tokenizer
	from tokenizers.models import WordLevel
	from tokenizers.pre_tokenizers import Whitespace
	from transformers import (
		PreTrainedTokenizerFast,
		Qwen3_5Config,
		Qwen3_5ForConditionalGeneration,
	)

	vocabulary = {
		word: i
		for i, word in enumerate(
			[
				"[UNK]",
				"[PAD]",
				"<|im_start|>",
				"<|im_end|>",
				"<think>",
				"</think>",
				"user",
				"assistant",
				"Question",
				"Other",
				"Answer",
			]
		)
	}
	backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
	backend.pre_tokenizer = Whitespace()
	tokenizer = PreTrainedTokenizerFast(
		tokenizer_object=backend,
		unk_token="[UNK]",
		pad_token="[PAD]",
		eos_token="<|im_end|>",
		additional_special_tokens=["<|im_start|>", "<think>", "</think>"],
	)
	# A local Qwen-format fixture, including the empty non-thinking prefix.
	tokenizer.chat_template = (
		"{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' }}"
		"{% if m['role'] == 'assistant' %}{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}"
		"{{ m['content'] + '<|im_end|>\\n' }}{% endfor %}"
		"{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
		"{% if enable_thinking is defined and not enable_thinking %}"
		"{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}{% endif %}"
	)
	base = tmp_path / "base"
	tokenizer.save_pretrained(base)
	# Same layout as the Qwen3.5 checkpoints: a multimodal wrapper around a hybrid text model
	# (linear-attention and full-attention layers), plus an MTP head the loader must ignore.
	hybrid = getattr(request, "param", True)
	model = Qwen3_5ForConditionalGeneration(
		Qwen3_5Config(
			text_config={
				"vocab_size": len(tokenizer),
				"hidden_size": 64,
				"intermediate_size": 128,
				"num_hidden_layers": 2,
				"layer_types": ["linear_attention", "full_attention"]
				if hybrid
				else ["full_attention", "full_attention"],
				"num_attention_heads": 4,
				"num_key_value_heads": 2,
				"head_dim": 16,
				"linear_num_key_heads": 2,
				"linear_num_value_heads": 2,
				"linear_key_head_dim": 16,
				"linear_value_head_dim": 16,
				"max_position_embeddings": 256,
				"rope_parameters": {
					"rope_type": "default",
					"rope_theta": 10000.0,
					"partial_rotary_factor": 0.5,
					"mrope_section": [2, 1, 1],
					"mrope_interleaved": True,
				},
				"eos_token_id": 0,
				"pad_token_id": tokenizer.pad_token_id,
				# The text model reads only its own flag, as in the real 2B config.
				"tie_word_embeddings": True,
			},
			vision_config={
				"depth": 1,
				"hidden_size": 32,
				"intermediate_size": 64,
				"num_heads": 2,
				"out_hidden_size": 64,
			},
			tie_word_embeddings=True,
		)
	)
	model.save_pretrained(base)
	(weights_file,) = base.glob("*.safetensors")
	weights = load_file(weights_file)
	weights["mtp.fc.weight"] = torch.zeros(64, 128)
	save_file(weights, weights_file, metadata={"format": "pt"})
	saved_embeddings = weights["model.language_model.embed_tokens.weight"]
	projection = "linear_attn.in_proj_qkv" if hybrid else "self_attn.q_proj"
	saved_projection = weights[f"model.language_model.layers.0.{projection}.weight"]
	return base, tokenizer, saved_embeddings, saved_projection


@pytest.mark.parametrize("qlora", [False, True])
def test_native_setup_masks_targets_and_preserves_frozen_head(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	qlora: bool,
	tiny_checkpoint: tuple[Path, Any, Any, Any],
) -> None:
	import torch
	from transformers import AutoTokenizer, PreTrainedTokenizerFast
	from trl import SFTTrainer

	if qlora and not torch.cuda.is_available():
		pytest.skip("QLoRA construction requires compatible CUDA hardware")
	base, tokenizer, saved_embeddings, saved_projection = tiny_checkpoint
	training, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"

	def row(question: str) -> SFTRecord:
		return SFTRecord(
			prompt=[Message(role="user", content=question)],
			completion=[Message(role="assistant", content="Answer")],
		)

	write_jsonl(training, [row("Question")])
	write_jsonl(validation, [row("Other Question")])

	class SetupVerified(Exception):
		pass

	def inspect_setup(trainer: Any, *args: Any, **kwargs: Any) -> None:
		record = trainer.train_dataset[0]
		labels = record["labels"]
		assert labels[0] == -100
		target = trainer.processing_class.decode([label for label in labels if label != -100])
		assert "Answer" in target and "<|im_end|>" in target
		assert "<think>" not in target and "</think>" not in target
		assert trainer.args.loss_type == "nll"
		paged = qlora and "microsoft" not in platform.release().lower()
		assert trainer.args.optim == ("paged_adamw_32bit" if paged else "adamw_torch_fused")
		assert not trainer.args.activation_offloading
		assert trainer.args.load_best_model_at_end
		assert trainer.args.eval_strategy == "epoch"
		assert trainer.args.gradient_checkpointing_kwargs == {"use_reentrant": False}
		assert trainer.args.get_warmup_steps(100) == 3
		assert trainer.model.config.eos_token_id == trainer.processing_class.eos_token_id
		assert trainer.model.generation_config.eos_token_id == trainer.processing_class.eos_token_id
		assert trainer.model.get_output_embeddings().__class__.__name__ == "Linear"
		assert any(parameter.requires_grad for parameter in trainer.model.parameters())
		# Only the text model loads, with the checkpoint's weights rather than fresh ones.
		base_model = trainer.model.get_base_model()
		assert base_model.__class__.__name__ == "Qwen3_5ForCausalLM"
		names = [name for name, _ in base_model.named_modules()]
		assert not any("visual" in name or "mtp" in name for name in names)
		adapted = {name.rsplit(".lora_A", 1)[0] for name in names if name.endswith(".lora_A")}
		assert any(name.endswith("linear_attn.in_proj_qkv") for name in adapted)
		assert any(name.endswith("self_attn.q_proj") for name in adapted)
		assert not any("conv1d" in name for name in adapted)
		assert base_model.lm_head.weight is base_model.model.embed_tokens.weight
		if not qlora:
			text = base_model.model
			assert torch.equal(text.embed_tokens.weight.cpu(), saved_embeddings.float())
			projection = text.layers[0].linear_attn.in_proj_qkv.base_layer.weight
			assert torch.equal(projection.cpu(), saved_projection.float())
		raise SetupVerified

	monkeypatch.setattr(SFTTrainer, "train", inspect_setup)
	# Transformers always builds a byte-level BPE tokenizer for Qwen3.5 checkpoints, which
	# cannot read this word-level fixture, so load the fixture as saved.
	monkeypatch.setattr(AutoTokenizer, "from_pretrained", PreTrainedTokenizerFast.from_pretrained)
	config = TrainingConfig(
		base_model=str(base),
		dataset_path=training,
		validation_dataset_path=validation,
		output_dir=tmp_path / "adapter",
		max_length=128,
		qlora=qlora,
		precision="bf16" if qlora else "fp32",
		lora_rank=2,
		lora_alpha=4,
	)
	with pytest.raises(SetupVerified):
		train(config)
	assert not (config.output_dir / "adapter_model.safetensors").exists()


def test_preflight_cli_validates_real_serialization_without_loading_weights(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	tiny_checkpoint: tuple[Path, Any, Any, Any],
) -> None:
	from distillation.cli import app
	from transformers import AutoTokenizer, PreTrainedTokenizerFast
	from typer.testing import CliRunner

	base, tokenizer, _, _ = tiny_checkpoint
	monkeypatch.setattr(AutoTokenizer, "from_pretrained", PreTrainedTokenizerFast.from_pretrained)
	# Removing all weights makes an accidental model load fail, rather than mocking preflight.
	for weights in base.glob("*.safetensors"):
		weights.unlink()
	row = SFTRecord(
		prompt=[Message(role="user", content="Question")],
		completion=[Message(role="assistant", content="Answer")],
	)
	dataset, config = tmp_path / "train.jsonl", tmp_path / "train.toml"
	write_jsonl(dataset, [row])
	output = tmp_path / "adapter"
	config.write_text(
		f"base_model={json.dumps(str(base))}\ndataset_path={json.dumps(str(dataset))}"
		f'\noutput_dir={json.dumps(str(output))}\nmax_length=128\nprecision="fp32"\n'
	)
	runner = CliRunner()
	result = runner.invoke(app, ["train", "--config", str(config), "--preflight-only"])
	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert (
		report["datasets"]["training"]["sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
	)
	assert report["datasets"]["training"]["records"] == 1
	assert report["datasets"]["training"]["completion_tokens"] == 2
	assert report["eos_token"] == "<|im_end|>"
	assert not output.exists()
	write_jsonl(
		dataset,
		[
			row.model_copy(
				update={
					"completion": [Message(role="assistant", content=" ".join(["Answer"] * 130))]
				}
			)
		],
	)
	result = runner.invoke(app, ["train", "--config", str(config), "--preflight-only"])
	assert result.exit_code != 0
	assert result.exception is not None and "over max_length" in str(result.exception)
	assert not output.exists()
	# A template that drops the assistant target must fail before training as well.
	write_jsonl(dataset, [row])
	tokenizer.chat_template = "{{ 'Question' }}"
	tokenizer.save_pretrained(base)
	result = runner.invoke(app, ["train", "--config", str(config), "--preflight-only"])
	assert result.exit_code != 0
	assert result.exception is not None and "no supervised completion" in str(result.exception)
	assert not output.exists()


@pytest.mark.parametrize("tiny_checkpoint", [False], indirect=True, ids=["full-attention-cpu"])
def test_cpu_lora_updates_weights_and_resumes_the_optimizer_exactly(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	tiny_checkpoint: tuple[Path, Any, Any, Any],
) -> None:
	import torch
	from distillation.benchmarks.inference import validate_adapter
	from distillation.benchmarks.models import ModelConfig
	from peft import PeftModel
	from safetensors.torch import load_file
	from transformers import (
		AutoModelForCausalLM,
		AutoTokenizer,
		PreTrainedTokenizerFast,
		TrainerCallback,
	)
	from trl import SFTTrainer
	from trl.trainer import utils

	base, tokenizer, saved_embeddings, _ = tiny_checkpoint
	monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
	monkeypatch.setattr(AutoTokenizer, "from_pretrained", PreTrainedTokenizerFast.from_pretrained)

	class TorchLogProb:
		"""CPU reference for TRL's GPU-only loss kernel; training and autograd stay real."""

		@staticmethod
		def apply(
			hidden: Any,
			weight: Any,
			bias: Any,
			labels: Any,
			temperature: float,
			chunk_size: int,
			softcap: float | None,
			scale: float,
			outputs: tuple[str, ...],
		) -> tuple[Any, ...]:
			logits = torch.nn.functional.linear(hidden, weight, bias) * scale
			if softcap is not None:
				logits = torch.tanh(logits / softcap) * softcap
			log_probs = (logits / temperature).log_softmax(-1)
			selected = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
			entropy = -(log_probs.exp() * log_probs).sum(-1)
			return selected, entropy, None, None, logits.argmax(-1) == labels

	monkeypatch.setattr(utils, "_ChunkedLogProbFunction", TorchLogProb)
	training, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
	write_jsonl(
		training,
		[
			SFTRecord(
				prompt=[Message(role="user", content="Question")],
				completion=[Message(role="assistant", content="Answer")],
			)
		],
	)
	write_jsonl(
		validation,
		[
			SFTRecord(
				prompt=[Message(role="user", content="Other Question")],
				completion=[Message(role="assistant", content="Answer")],
			)
		],
	)
	config = TrainingConfig(
		base_model=str(base),
		dataset_path=training,
		validation_dataset_path=validation,
		output_dir=tmp_path / "uninterrupted",
		max_length=128,
		precision="fp32",
		optimizer="adamw_torch",
		activation_offloading=False,
		lora_rank=2,
		lora_alpha=4,
		lora_dropout=0,
		epochs=4,
		gradient_accumulation_steps=1,
		learning_rate=0.01,
		logging_steps=1,
		select_best_checkpoint=False,
	)
	train(config)
	expected = load_file(config.output_dir / "adapter_model.safetensors")
	assert expected and all(torch.isfinite(value).all() for value in expected.values())
	assert any(value.count_nonzero() for name, value in expected.items() if ".lora_B." in name)
	interrupted = config.model_copy(update={"output_dir": tmp_path / "resumed"})
	original_train = SFTTrainer.train

	class InterruptDuringSecondStep(TrainerCallback):  # type: ignore[misc]
		def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
			if state.global_step == 1:
				signal.raise_signal(signal.SIGTERM)

	def train_until_signal(trainer: Any, *args: Any, **kwargs: Any) -> Any:
		trainer.add_callback(InterruptDuringSecondStep())
		return original_train(trainer, *args, **kwargs)

	previous_handler = signal.getsignal(signal.SIGTERM)
	with monkeypatch.context() as interrupt:
		interrupt.setattr(SFTTrainer, "train", train_until_signal)
		with pytest.raises(TrainingInterrupted, match="stopped at step 2 of 4"):
			train(interrupted)
	assert signal.getsignal(signal.SIGTERM) == previous_handler
	checkpoint = interrupted.output_dir / "checkpoint-2"
	assert json.loads((checkpoint / "trainer_state.json").read_text())["global_step"] == 2
	assert (checkpoint / "optimizer.pt").is_file() and (checkpoint / "scheduler.pt").is_file()
	assert not (interrupted.output_dir / "training-manifest.json").exists()
	assert not (interrupted.output_dir / "adapter_model.safetensors").exists()
	steps: list[int] = []

	class RecordSteps(TrainerCallback):  # type: ignore[misc]
		def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
			steps.append(state.global_step)

	def observe_resume(trainer: Any, *args: Any, **kwargs: Any) -> Any:
		trainer.add_callback(RecordSteps())
		return original_train(trainer, *args, **kwargs)

	with monkeypatch.context() as resume:
		resume.setattr(SFTTrainer, "train", observe_resume)
		train(interrupted)
	assert steps == [3, 4]
	actual = load_file(interrupted.output_dir / "adapter_model.safetensors")
	assert actual.keys() == expected.keys()
	for name in expected:
		torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)
	manifest = json.loads((interrupted.output_dir / "training-manifest.json").read_text())
	assert manifest["global_step"] == 4 and manifest["resumed_from"] == str(checkpoint)
	assert math.isfinite(manifest["final_evaluation"]["eval_loss"])
	assert manifest["dataset_sha256"] == hashlib.sha256(training.read_bytes()).hexdigest()
	evaluation = ModelConfig(
		backend="local",
		model=str(base),
		revision=config.base_revision,
		adapter_path=interrupted.output_dir,
		precision="fp32",
		device="cpu",
	)
	assert validate_adapter(evaluation) == [training, validation]
	# Reload the written adapter through PEFT and run a real forward pass.
	reloaded = PeftModel.from_pretrained(
		AutoModelForCausalLM.from_pretrained(str(base)), str(interrupted.output_dir)
	)
	assert torch.equal(reloaded.get_input_embeddings().weight.cpu(), saved_embeddings.float())
	encoded = tokenizer("Question", return_tensors="pt")
	with torch.inference_mode():
		logits = reloaded(**encoded).logits
	assert logits.shape[-1] == len(tokenizer) and torch.isfinite(logits).all()
	# Reusing a checkpoint with changed data must not overwrite the completed adapter.
	before = (interrupted.output_dir / "adapter_model.safetensors").read_bytes()
	write_jsonl(
		training,
		[
			SFTRecord(
				prompt=[Message(role="user", content="Question")],
				completion=[Message(role="assistant", content="Other Answer")],
			)
		],
	)
	with pytest.raises(ValueError, match="different training config, dataset, or chat template"):
		train(interrupted)
	assert (interrupted.output_dir / "adapter_model.safetensors").read_bytes() == before
