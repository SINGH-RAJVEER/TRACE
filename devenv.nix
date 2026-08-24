{ pkgs, ... }:

{
	packages = with pkgs; [
		just
		python314
		uv
	];

	env = {
		PYTHONUNBUFFERED = "1";
		UV_PYTHON = "${pkgs.python314}/bin/python";
	};

	enterShell = ''
		uv sync --all-packages
	'';

	processes.teacher-api.exec = "uv run teacher-api";
}
