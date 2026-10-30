"""Sends stored teacher items 16 at a time and compares them with their one-at-a-time outputs."""
import asyncio, json, sys, time
from pathlib import Path
from distillation.benchmarks.inference import ConcurrentServed, make_model
from distillation.benchmarks.models import load_config, load_suite
from distillation.benchmarks.scoring import evaluate_answer
from distillation.schemas import TeacherGenerationRequest

teacher = Path("artifacts/gateway-bounded-v1/A1/s42-b6000/teacher")
stored = {
	"ag_news": teacher / "benchmarks/ag_news/df2e1b0189634222a64c97d8df8bf931.samples.jsonl",
	"squad": teacher / "benchmarks/squad/4705f0a99a4b46d1848ca84426117fcb.samples.jsonl",
	"gsm8k": teacher / "benchmarks/gsm8k/885a2e1d82634824821b7a5a7f935c6a.samples.jsonl",
}
n = int(sys.argv[1])

async def main():
	for domain, path in stored.items():
		cfg = load_config(teacher / f"eval.{domain}.toml")
		before = {r["item_id"]: r for r in map(json.loads, path.read_text().splitlines())}
		items = [i for i in load_suite(cfg.suite_path).items if i.id in before][:n]
		model = make_model(cfg.models["teacher"])
		assert isinstance(model, ConcurrentServed)
		reqs = [TeacherGenerationRequest(messages=i.messages, max_output_tokens=cfg.generation.max_output_tokens, temperature=0, seed=cfg.generation.seed, enable_thinking=False) for i in items]
		start = time.perf_counter()
		gens = await model.generate_batch(reqs)
		elapsed = time.perf_counter() - start
		await model.close()
		errors = sum(isinstance(g, BaseException) for g in gens)
		same = correct_old = correct_new = 0
		for item, gen in zip(items, gens):
			if isinstance(gen, BaseException):
				continue
			old = before[item.id]
			same += gen.message.content == old["response"]
			correct_old += bool(old["correct"])
			correct_new += bool((await evaluate_answer(item, gen.message.content, gen.finish_reason)).correct)
		sequential = sum(before[i.id]["latency_seconds"] for i in items)
		print(f"{domain}: identical {same}/{len(items)}, errors {errors}, correct {correct_old} -> {correct_new}, "
			f"time {sequential:.0f}s one at a time -> {elapsed:.0f}s at 16", flush=True)

asyncio.run(main())
