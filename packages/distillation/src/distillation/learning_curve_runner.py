"""Optional GPU runner. Import model dependencies only when executing a cell."""

import asyncio
import gc
import importlib
import shutil
from pathlib import Path
from typing import Any

from distillation.benchmarks.models import BenchmarkItem
from distillation.benchmarks.scoring import evaluate_answer


def run_cell(output: Path, recipe: dict[str, Any], n: int | None, seed: int) -> dict[str, Any]:
	import json

	try:
		torch = importlib.import_module("torch")
		tf = importlib.import_module("transformers")
		peft = importlib.import_module("peft")
		datasets = importlib.import_module("datasets")
		trl = importlib.import_module("trl")
	except ImportError as error:
		raise RuntimeError(
			"Execution needs torch, transformers, peft, datasets and trl in the uv environment. "
			"Planning and summary do not need GPU dependencies."
		) from error
	if not torch.cuda.is_available():
		raise RuntimeError("learning-curve execution requires a CUDA GPU")
	tf.set_seed(seed)
	tokenizer = tf.AutoTokenizer.from_pretrained(
		recipe["tokenizer_model"], revision=recipe["tokenizer_revision"]
	)
	tokenizer.pad_token = tokenizer.eos_token
	model = tf.Qwen3_5ForCausalLM.from_pretrained(
		recipe["base_model"],
		revision=recipe["revision"],
		dtype=torch.bfloat16,
		attn_implementation="sdpa",
	).to("cuda")
	trainer = None
	try:
		if n is not None:
			# An incomplete cell restarts from the pinned base, never an arbitrary checkpoint.
			cell_dir = output / "models" / f"n{n}-s{seed}"
			if cell_dir.exists():
				shutil.rmtree(cell_dir)
			rows = json.loads((output / f"subset-{n}.json").read_text())
			rendered = []
			for row in rows:
				sft = row["sft"]
				prompt = tokenizer.apply_chat_template(
					sft["prompt"],
					tokenize=False,
					add_generation_prompt=True,
					enable_thinking=False,
				)
				full = tokenizer.apply_chat_template(
					sft["prompt"] + sft["completion"],
					tokenize=False,
					add_generation_prompt=False,
					enable_thinking=False,
				)
				if not full.startswith(prompt):
					raise ValueError("chat template does not preserve the generation prefix")
				if len(tokenizer(full)["input_ids"]) > recipe["max_length"]:
					raise ValueError("training response would be truncated by the frozen recipe")
				rendered.append({"prompt": prompt, "completion": full[len(prompt) :]})
			data = datasets.Dataset.from_list(rendered)
			adapter = peft.LoraConfig(
				r=recipe["lora_rank"],
				lora_alpha=recipe["lora_alpha"],
				lora_dropout=recipe["lora_dropout"],
				target_modules=recipe["target_modules"],
				task_type="CAUSAL_LM",
			)
			config = trl.SFTConfig(
				output_dir=str(cell_dir),
				learning_rate=recipe["learning_rate"],
				num_train_epochs=recipe["epochs"],
				max_length=recipe["max_length"],
				per_device_train_batch_size=recipe["batch_size"],
				gradient_accumulation_steps=recipe["gradient_accumulation_steps"],
				warmup_ratio=recipe["warmup_ratio"],
				seed=seed,
				data_seed=seed,
				bf16=True,
				gradient_checkpointing=True,
				gradient_checkpointing_kwargs={"use_reentrant": False},
				save_strategy="no",
				report_to="none",
				eos_token="<|im_end|>",
				completion_only_loss=True,
				loss_type="nll",
				optim="adamw_torch",
				packing=False,
			)
			trainer = trl.SFTTrainer(
				model=model,
				args=config,
				train_dataset=data,
				processing_class=tokenizer,
				peft_config=adapter,
			)
			trainer.train()
			trainer.save_model(str(cell_dir))
			model = trainer.model
		model.eval()
		model.config.use_cache = True
		results = {}
		for raw in json.loads((output / "suite.json").read_text()):
			item = BenchmarkItem.model_validate(raw)
			text = tokenizer.apply_chat_template(
				[m.model_dump() for m in item.messages],
				tokenize=False,
				add_generation_prompt=True,
				enable_thinking=False,
			)
			inputs = tokenizer(text, return_tensors="pt").to("cuda")
			with torch.inference_mode():
				generated = model.generate(
					**inputs,
					max_new_tokens=16,
					do_sample=False,
					eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
					pad_token_id=tokenizer.pad_token_id,
				)
			answer_tokens = generated[0, inputs["input_ids"].shape[1] :]
			answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
			ended = int(answer_tokens[-1]) in {
				tokenizer.eos_token_id,
				tokenizer.convert_tokens_to_ids("<|im_end|>"),
			}
			score = asyncio.run(evaluate_answer(item, answer, "stop" if ended else "length"))
			results[item.id] = int(score.correct)
		return results
	finally:
		del trainer
		del model
		gc.collect()
		torch.cuda.empty_cache()
