"""Three-way GSM8K comparison: stored (yesterday, sequential), today sequential, today at 16 in flight."""
import asyncio, json, os
from pathlib import Path
from distillation.benchmarks.inference import ConcurrentServed, make_model
from distillation.benchmarks.models import load_config, load_suite
from distillation.schemas import TeacherGenerationRequest

teacher = Path("artifacts/gateway-bounded-v1/A1/s42-b6000/teacher")
path = teacher / "benchmarks/gsm8k/885a2e1d82634824821b7a5a7f935c6a.samples.jsonl"
cfg = load_config(teacher / "eval.gsm8k.toml")
stored = {r["item_id"]: r["response"] for r in map(json.loads, path.read_text().splitlines())}
items = [i for i in load_suite(cfg.suite_path).items if i.id in stored][:16]
reqs = [TeacherGenerationRequest(messages=i.messages, max_output_tokens=256, temperature=0, seed=42, enable_thinking=False) for i in items]

async def main():
	os.environ["DISTILL_EVAL_CONCURRENCY"] = "1"
	single = make_model(cfg.models["teacher"])
	seq = [(await single.generate(r)).message.content for r in reqs]
	seq2 = [(await single.generate(r)).message.content for r in reqs[:4]]
	await single.close()
	os.environ["DISTILL_EVAL_CONCURRENCY"] = "16"
	conc = make_model(cfg.models["teacher"])
	par = [g.message.content for g in await conc.generate_batch(reqs)]
	await conc.close()
	old = [stored[i.id] for i in items]
	n = len(items)
	print("today sequential vs stored:", sum(a == b for a, b in zip(seq, old)), "/", n)
	print("today sequential vs today repeat:", sum(a == b for a, b in zip(seq[:4], seq2)), "/ 4")
	print("today concurrent vs today sequential:", sum(a == b for a, b in zip(par, seq)), "/", n)
	print("today concurrent vs stored:", sum(a == b for a, b in zip(par, old)), "/", n)
	d = next(k for k in range(n) if seq[k] != old[k]) if any(seq[k] != old[k] for k in range(n)) else None
	if d is not None:
		p = os.path.commonprefix([seq[d], old[d]])
		print("first diff at char", len(p), repr(old[d][len(p)-40:len(p)+60]), "|", repr(seq[d][len(p)-40:len(p)+60]))

asyncio.run(main())
