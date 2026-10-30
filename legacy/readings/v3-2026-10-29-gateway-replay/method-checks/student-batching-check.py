"""Compares batch-1 attached-adapter generation with merged, batched generation on real test items."""
import asyncio, json, os, sys, time
from pathlib import Path
import peft
from distillation.benchmarks.inference import LocalModel
from distillation.benchmarks.models import ModelConfig, load_config, load_suite
from distillation.benchmarks.scoring import evaluate_answer
from distillation.schemas import TeacherGenerationRequest

root = Path("artifacts/gateway-bounded-v1")
cfg = load_config(root / "development/lr0.0001-e3/eval.gsm8k.toml")
before, after = cfg.models["student_before"], cfg.models["student_after"]
caps = {"ag_news": 16, "squad": 64, "gsm8k": 256}
suites = {
	"ag_news": "data/tasks/ag_news-split-v2-s42/test-suite.json",
	"squad": "data/tasks/squad-split-v2-s42/test-suite.json",
	"gsm8k": "data/tasks/gsm8k/test-suite.json",
}
n = int(sys.argv[1])

async def run(model_cfg, batch, merge):
	os.environ["DISTILL_EVAL_BATCH_SIZE"] = str(batch)
	from peft.tuners.lora import LoraModel
	holder = []
	original_load, original_merge = peft.PeftModel.from_pretrained, LoraModel.merge_and_unload
	if not merge:
		def load(*a, **k):
			holder.append(original_load(*a, **k))
			return holder[-1]
		peft.PeftModel.from_pretrained = load
		LoraModel.merge_and_unload = lambda self, *a, **k: holder[-1]
	model = LocalModel(model_cfg)
	peft.PeftModel.from_pretrained, LoraModel.merge_and_unload = original_load, original_merge
	print("model type", type(model._model).__name__, model.execution, flush=True)
	out = {}
	for domain, path in suites.items():
		items = load_suite(Path(path)).items[:n]
		reqs = [TeacherGenerationRequest(messages=i.messages, max_output_tokens=caps[domain], temperature=0, seed=42, enable_thinking=False) for i in items]
		start = time.perf_counter()
		gens = []
		for k in range(0, len(reqs), batch):
			gens += await model.generate_batch(reqs[k:k + batch])
		elapsed = time.perf_counter() - start
		scores = [await evaluate_answer(i, g.message.content, g.finish_reason) for i, g in zip(items, gens)]
		out[domain] = {"seconds": round(elapsed, 1), "texts": [g.message.content for g in gens],
			"correct": sum(bool(s.correct) for s in scores), "f1": [s.f1 for s in scores]}
	await model.close()
	return out

async def main():
	results = {}
	for name, mc, batch, merge in [("base_b1", before, 1, False), ("base_b16", before, 16, False),
			("adapter_attached_b1", after, 1, False), ("adapter_merged_b16", after, 16, True)]:
		results[name] = await run(mc, batch, merge)
		print(name, {d: (r["seconds"], r["correct"]) for d, r in results[name].items()}, flush=True)
	for a, b in [("base_b1", "base_b16"), ("adapter_attached_b1", "adapter_merged_b16")]:
		for d in suites:
			same = sum(x == y for x, y in zip(results[a][d]["texts"], results[b][d]["texts"]))
			print(f"{a} vs {b} {d}: identical {same}/{n}, correct {results[a][d]['correct']} vs {results[b][d]['correct']}, time {results[a][d]['seconds']}s vs {results[b][d]['seconds']}s", flush=True)
	Path("/tmp/evalcheck/results.json").write_text(json.dumps(results, indent=1))

asyncio.run(main())
