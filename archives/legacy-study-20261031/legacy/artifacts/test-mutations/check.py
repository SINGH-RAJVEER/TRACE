import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


root = Path.cwd()
out = root / "artifacts/test-mutations"
cases = [
	("resume-contract", "collection.py",
		'raise ValueError(f"resume mismatch for {record.example_id}; use a new ledger")',
		"pass",
		"tests/test_pipeline.py::test_resume_rejects_changed_contract_without_touching_the_ledger[temperature]"),
	("thought-rejection", "collection.py", "if options.enable_thinking is False and any(",
		"if False and any(",
		"tests/test_pipeline.py::test_http_collection_retries_reviews_and_resumes_without_losing_provenance"),
	("concurrency-bound", "collection.py", "asyncio.Semaphore(options.concurrency)",
		"asyncio.Semaphore(999)",
		"tests/test_pipeline.py::test_collection_bounds_concurrency_and_flushes_before_other_workers_finish"),
	("overlong-target", "training.py", "if overlong:", "if False:",
		"tests/test_training_stack.py::test_preflight_cli_validates_real_serialization_without_loading_weights"),
	("checkpoint-resume", "training.py",
		"trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)",
		"trainer.train()",
		"tests/test_training_stack.py::test_cpu_lora_updates_weights_and_resumes_the_optimizer_exactly[full-attention-cpu]"),
]
results = []
for name, filename, old, new, node in cases:
	with tempfile.TemporaryDirectory(prefix="capstone-test-mutation-") as temporary:
		package = Path(temporary) / "distillation"
		shutil.copytree(root / "packages/distillation/src/distillation", package,
			ignore=shutil.ignore_patterns("__pycache__"))
		path = package / filename
		text = path.read_text()
		assert text.count(old) == 1, name
		path.write_text(text.replace(old, new))
		env = {**os.environ, "PYTHONPATH": temporary, "DISTILL_TEST_TRAINING_STACK": "1",
			"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
			"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
		result = subprocess.run([sys.executable, "-m", "pytest", node, "-q", "--tb=short"],
			cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
			timeout=120)
		(out / f"{name}.log").write_text(result.stdout)
		caught = result.returncode == 1 and bool(re.search(r"\b1 failed\b", result.stdout))
		results.append({"mutation": name, "caught": caught, "exit_code": result.returncode})
		print(name, "caught" if caught else "NOT CAUGHT", flush=True)
(out / "results.json").write_text(json.dumps(results, indent="\t") + "\n")
assert all(result["caught"] for result in results)
