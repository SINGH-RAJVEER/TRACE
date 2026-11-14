"""Exercise the vLLM launch contract without loading a model or starting a server."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def vllm_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	for key in list(os.environ):
		if key.startswith(("VLLM_", "TEACHER_")):
			monkeypatch.delenv(key)
	stub = tmp_path / "vllm executable"
	stub.write_text('#!/bin/sh\nprintf \'%s\\0\' "$@" > "$TRACE_ARGV_FILE"\n')
	stub.chmod(0o755)
	output = tmp_path / "arguments"
	monkeypatch.setenv("VLLM_BIN", str(stub))
	monkeypatch.setenv("TRACE_ARGV_FILE", str(output))
	return output


@pytest.mark.parametrize("via_just", [False, True])
def test_vllm_launch_defaults_and_argument_boundaries(vllm_capture: Path, via_just: bool) -> None:
	arguments = ["--default-chat-template-kwargs", '{"enable_thinking": false}']
	if via_just:
		just = shutil.which("just")
		if just is None:
			pytest.skip("just executable unavailable")
		command = [just, "--justfile", str(ROOT / "justfile"), "--no-dotenv", "teacher-vllm"]
	else:
		command = ["bash", str(ROOT / "scripts" / "teacher-vllm.sh")]
	result = subprocess.run(command + arguments, capture_output=True, text=True, timeout=30)
	assert result.returncode == 0, result.stderr
	assert vllm_capture.read_bytes().decode().split("\0")[:-1] == [
		"serve",
		"Qwen/Qwen3.5-9B",
		"--revision",
		"c202236235762e1c871ad0ccb60c8ee5ba337b9a",
		"--served-model-name",
		"Qwen/Qwen3.5-9B",
		"--host",
		"127.0.0.1",
		"--port",
		"8001",
		"--max-model-len",
		"8192",
		"--gpu-memory-utilization",
		"0.90",
		"--reasoning-parser",
		"qwen3",
		"--language-model-only",
		*arguments,
	]


def test_vllm_launch_respects_environment(
	vllm_capture: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	for key, value in {
		"TEACHER_UPSTREAM_MODEL": "model with spaces",
		"TEACHER_MODEL_REVISION": "pinned-revision",
		"VLLM_PORT": "8101",
		"VLLM_MAX_MODEL_LEN": "4096",
		"VLLM_GPU_MEMORY_UTILIZATION": "0.75",
		"VLLM_EXTRA_ARGS": "--quantization awq --enforce-eager",
	}.items():
		monkeypatch.setenv(key, value)
	result = subprocess.run(
		["bash", str(ROOT / "scripts" / "teacher-vllm.sh")],
		cwd=tmp_path,
		capture_output=True,
		text=True,
		timeout=30,
	)
	assert result.returncode == 0, result.stderr
	argv = vllm_capture.read_bytes().decode().split("\0")[:-1]
	for flag, value in {
		"--revision": "pinned-revision",
		"--served-model-name": "model with spaces",
		"--port": "8101",
		"--max-model-len": "4096",
		"--gpu-memory-utilization": "0.75",
	}.items():
		assert argv[argv.index(flag) + 1] == value
	assert argv[:2] == ["serve", "model with spaces"]
	assert argv[-3:] == ["--quantization", "awq", "--enforce-eager"]


def test_missing_vllm_fails_without_falling_back(
	vllm_capture: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	monkeypatch.setenv("VLLM_BIN", str(tmp_path / "missing-vllm"))
	result = subprocess.run(
		["bash", str(ROOT / "scripts" / "teacher-vllm.sh")],
		capture_output=True,
		text=True,
		timeout=30,
	)
	assert result.returncode == 1
	assert "no vLLM executable" in result.stderr
	assert not vllm_capture.exists()


def test_all_teacher_launches_use_vllm() -> None:
	assert not (ROOT / "scripts" / "teacher-gguf.sh").exists()
	justfile = (ROOT / "justfile").read_text()
	assert "teacher-gguf" not in justfile
	assert "bash scripts/teacher-vllm.sh" in justfile
	assert "llama-cpp" not in (ROOT / "devenv.nix").read_text()
	for name in ("study.sh", "baseline-gates.sh"):
		script = (ROOT / "scripts" / name).read_text()
		assert "bash scripts/teacher-vllm.sh" in script
		assert '"$VLLM_BIN" serve' not in script


def test_kgw_launch_registers_processor_and_middleware(
	vllm_capture: Path,
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	monkeypatch.setenv("TEACHER_WATERMARK_SCHEME", "kgw-v1")
	monkeypatch.setenv("TEACHER_WATERMARK_VOCAB_SIZE", "512")
	key_file = tmp_path / "new.watermark.key"
	monkeypatch.setenv("TEACHER_WATERMARK_KEY_FILE", str(key_file))
	result = subprocess.run(
		["bash", str(ROOT / "scripts" / "teacher-vllm.sh")],
		capture_output=True,
		text=True,
		timeout=30,
	)
	assert result.returncode == 0, result.stderr
	argv = vllm_capture.read_bytes().decode().split("\0")[:-1]
	assert argv[-5:] == [
		"--logits-processors",
		"trace_watermark.vllm:KgwLogitsProcessor",
		"--middleware",
		"trace_watermark.middleware.KgwMiddleware",
		"--no-async-scheduling",
	]
	assert len(bytes.fromhex(key_file.read_text().strip())) == 32
	assert key_file.read_text().strip() not in result.stdout + result.stderr
