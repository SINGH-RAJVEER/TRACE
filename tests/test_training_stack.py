"""Opt-in native trainer construction. No teacher calls or optimization steps."""

import os
from pathlib import Path
from typing import Any

import pytest
from distillation.io import write_jsonl
from distillation.schemas import Message, SFTRecord
from distillation.training import TrainingConfig, train

pytestmark = pytest.mark.skipif(
	os.environ.get("DISTILL_TEST_TRAINING_STACK") != "1",
	reason="set DISTILL_TEST_TRAINING_STACK=1 with the train extra to check trainer construction",
)


@pytest.mark.parametrize("qlora", [False, True])
def test_native_setup_masks_targets_and_preserves_frozen_head(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	qlora: bool,
) -> None:
	torch = pytest.importorskip("torch")
	if qlora and not torch.cuda.is_available():
		pytest.skip("QLoRA construction requires compatible CUDA hardware")
	from safetensors.torch import load_file, save_file
	from tokenizers import Tokenizer
	from tokenizers.models import WordLevel
	from tokenizers.pre_tokenizers import Whitespace
	from transformers import (
		AutoTokenizer,
		PreTrainedTokenizerFast,
		Qwen3_5Config,
		Qwen3_5ForConditionalGeneration,
	)
	from trl import SFTTrainer

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
	model = Qwen3_5ForConditionalGeneration(
		Qwen3_5Config(
			text_config={
				"vocab_size": len(tokenizer),
				"hidden_size": 64,
				"intermediate_size": 128,
				"num_hidden_layers": 2,
				"layer_types": ["linear_attention", "full_attention"],
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
	saved_projection = weights["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]
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
