from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import time

import httpx
from teacher_api.settings import Settings


root = Path("artifacts/colab-teacher/q8")
settings = Settings()
assert settings.api_token and settings.terms_review_id
with httpx.Client(timeout=600) as client:
	print("Waiting for the verified Q8 checkpoint to load on Colab", flush=True)
	for _ in range(900):
		try:
			response = client.get("http://127.0.0.1:18011/health", timeout=5)
			if response.status_code == 200:
				break
		except httpx.HTTPError:
			pass
		time.sleep(1)
	else:
		raise RuntimeError("Q8 model server did not become ready")
	models = client.get("http://127.0.0.1:18011/v1/models").json()
	assert models["data"][0]["id"] == "Qwen3.5-9B-Q8_0", models
	slots = client.get("http://127.0.0.1:18011/slots").json()
	assert len(slots) == 4 and all(s["n_ctx"] == 4096 for s in slots), slots
	payload = {
		"model": "Qwen3.5-9B-Q8_0",
		"messages": [{"role": "user", "content": "What is 7 * 8? Reply with only the number."}],
		"max_tokens": 32,
		"temperature": 0,
		"chat_template_kwargs": {"enable_thinking": False},
	}
	response = client.post("http://127.0.0.1:18011/v1/chat/completions", json=payload)
	response.raise_for_status()
	upstream = response.json()
	message = upstream["choices"][0]["message"]
	assert message["content"].strip() == "56", message
	assert not message.get("reasoning_content"), message
	print("Upstream Q8 generation passed", flush=True)
	env = {
		**os.environ,
		"TEACHER_UPSTREAM_BASE_URL": "http://127.0.0.1:18011/v1",
		"TEACHER_UPSTREAM_MODEL": "Qwen3.5-9B-Q8_0",
		"TEACHER_PROVIDER_NAME": "colab-llamacpp-q8_0",
		"TEACHER_TIMEOUT_SECONDS": "600",
		"TEACHER_HOST": "127.0.0.1",
		"TEACHER_PORT": "18010",
		"PYTHONUNBUFFERED": "1",
	}
	with (root / "api.log").open("ab") as log:
		api = subprocess.Popen(
			["uv", "run", "teacher-api"],
			env=env,
			stdin=subprocess.DEVNULL,
			stdout=log,
			stderr=subprocess.STDOUT,
			start_new_session=True,
		)
	(root / "api.pid").write_text(str(api.pid) + "\n")
	for _ in range(60):
		try:
			response = client.get("http://127.0.0.1:18010/healthz", timeout=5)
			if response.status_code == 200 and response.json()["status"] == "ready":
				break
		except httpx.HTTPError:
			pass
		time.sleep(1)
	else:
		raise RuntimeError("Q8 API did not start")
	generation = {
		"messages": payload["messages"],
		"temperature": 0,
		"max_output_tokens": 32,
		"enable_thinking": False,
	}
	assert (
		client.post("http://127.0.0.1:18010/v1/teacher/generate", json=generation).status_code
		== 401
	)
	response = client.post(
		"http://127.0.0.1:18010/v1/teacher/generate",
		headers={"Authorization": "Bearer " + settings.api_token},
		json=generation,
	)
	response.raise_for_status()
	body = response.json()
	assert body["model"] == "Qwen3.5-9B-Q8_0" and body["provider"] == "colab-llamacpp-q8_0", body
	assert body["message"]["content"].strip() == "56", body
	(root / "validation.json").write_text(
		json.dumps(
			{
				"checked_at": datetime.now(UTC).isoformat(),
				"upstream": upstream,
				"teacher_api": body,
				"unauthenticated_status": 401,
				"slots": slots,
			},
			indent="\t",
		)
		+ "\n"
	)
	print("Authenticated Q8 teacher API verified; ready for task collection", flush=True)

print("Start task collection with artifacts/paper-tasks-colab/collect.py", flush=True)
