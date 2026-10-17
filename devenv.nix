{ pkgs, lib, ... }:

{
	dotenv.disableHint = true;
	packages = with pkgs; [
		bash
		just
		python314
		# Interpreter for the separate vLLM teacher environment in .venv-vllm.
		python312
		uv
		bun
		# Native compilation for Python extensions and Triton training kernels.
		stdenv.cc
		zlib
	] ++ lib.optionals pkgs.stdenv.isLinux [ pkgs.bubblewrap pkgs.util-linux ];

	env = {
		PYTHONUNBUFFERED = "1";
		UV_PYTHON = "${pkgs.python314}/bin/python";
		UV_PYTHON_DOWNLOADS = "never";
		VLLM_PYTHON = "${pkgs.python312}/bin/python";
	} // lib.optionalAttrs pkgs.stdenv.isLinux {
		# Binary wheels need the C++ runtime; CUDA needs the host NVIDIA driver.
		LD_LIBRARY_PATH = "${lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}:/run/opengl-driver/lib";
		# Triton locates libcuda with ldconfig, which cannot see the NixOS driver path.
		TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
	};

	# uv.lock supplies anyio, httpx, pydantic, typer, FastAPI,
	# pydantic-settings, and uvicorn, plus development tools mypy, packaging,
	# pytest, pytest-asyncio, and Ruff.
	# Optional evaluation: uv sync --all-packages --extra eval
	# Optional training: uv sync --all-packages --extra train
	# Training includes torch, transformers, datasets, accelerate, peft, TRL,
	# and bitsandbytes on Linux. Evaluation includes datasets and huggingface-hub.
	enterShell = ''
		uv sync --all-packages
	'';

	processes.teacher-api.exec = "uv run teacher-api";
}
