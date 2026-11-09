"""Launch validation and shell argument boundaries, with no models or live servers."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from harness.campaign import CampaignConfig, SessionConfig, load_campaign_config
from harness_api import app as app_module
from harness_api.launcher import main, prepare_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
	monkeypatch.chdir(tmp_path)
	for key in list(os.environ):
		if key.startswith("HARNESS_"):
			monkeypatch.delenv(key)
	campaign = tmp_path / "campaign config.json"
	campaign.write_text(CampaignConfig(namespace="fixture").model_dump_json())
	accounts = tmp_path / "accounts.json"
	accounts.write_text("{}")
	calibration = tmp_path / "calibration"
	calibration.mkdir()
	(calibration / "calibration.json").write_text("{}")
	monkeypatch.setenv("HARNESS_ACCOUNTS_FILE", str(accounts))
	monkeypatch.setenv("HARNESS_CALIBRATION_DIR", str(calibration))
	return campaign, tmp_path / "campaign.sqlite3"


def test_config_creation_is_explicit_and_exclusive(tmp_path: Path) -> None:
	path = tmp_path / "nested" / "config.json"
	main(["config", str(path), "--namespace", "fixture"])
	assert load_campaign_config(path).mode == "observe"
	assert '"session"' not in path.read_text()
	original = path.read_bytes()
	with pytest.raises(SystemExit) as error:
		main(["config", str(path), "--namespace", "replacement", "--mode", "throttle"])
	assert error.value.code == 2 and path.read_bytes() == original


def test_validation_has_no_server_or_state_side_effects(
	launch_files: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
	campaign, state = launch_files
	main(["validate", "--config", str(campaign), "--state-db", str(state)])
	output = capsys.readouterr().out
	assert "base_policy=allow" in output
	assert "session=disabled" in output
	assert not state.exists()


def test_validation_prints_enabled_session_version(
	launch_files: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
	campaign, state = launch_files
	campaign.write_text(
		CampaignConfig(namespace="fixture", session=SessionConfig()).model_dump_json()
	)
	main(["validate", "--config", str(campaign), "--state-db", str(state)])
	assert "session=semantic-session-v2" in capsys.readouterr().out
	assert not state.exists()


def test_product_launch_requires_explicit_valid_inputs(
	launch_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
	campaign, state = launch_files
	with pytest.raises(ValueError, match="CAMPAIGN_CONFIG"):
		prepare_settings(None, None, "allow")
	with pytest.raises(ValueError, match="STATE_DB"):
		prepare_settings(campaign, None, "allow")
	campaign.write_text(CampaignConfig(namespace="fixture", mode="throttle").model_dump_json())
	with pytest.raises(ValueError, match="UNQUALIFIED"):
		prepare_settings(campaign, state, "allow")
	monkeypatch.setenv("HARNESS_CAMPAIGN_ALLOW_UNQUALIFIED_ENFORCEMENT", "true")
	settings, _ = prepare_settings(campaign, state, "allow")
	assert settings.campaign_allow_unqualified_enforcement
	monkeypatch.setenv("HARNESS_TEST_CLOCK", "true")
	with pytest.raises(ValueError, match="test actor"):
		prepare_settings(campaign, state, "allow")


def test_legacy_ignores_campaign_settings_from_dotenv(
	launch_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
	campaign, state = launch_files
	Path(".env").write_text(
		f"HARNESS_CAMPAIGN_CONFIG='{campaign}'\n"
		"HARNESS_CAMPAIGN_ALLOW_UNQUALIFIED_ENFORCEMENT=true\n"
		f"HARNESS_STATE_DB='{state}'\n"
	)
	seen: list[Any] = []
	monkeypatch.setattr(app_module, "create_app", lambda **kwargs: seen.append(kwargs["settings"]))
	import uvicorn

	monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
	main(["legacy"])
	assert seen[0].campaign_config is None
	assert not seen[0].campaign_allow_unqualified_enforcement
	assert not state.exists()


@pytest.mark.parametrize(
	("recipe", "arguments", "tail"),
	[
		(
			"reading",
			["run with spaces"],
			["export", "artifacts/runs/run with spaces", "readings", ""],
		),
		(
			"start",
			["--config", "campaign with spaces.json", "--state-db", "state.sqlite3"],
			["serve", "--config", "campaign with spaces.json", "--state-db", "state.sqlite3"],
		),
		(
			"checkpoint-export",
			["source $(literal)", "new output", "--response-count", "100"],
			[
				"checkpoints",
				"export",
				"--campaign",
				"source $(literal)",
				"--output",
				"new output",
				"--response-count",
				"100",
			],
		),
		(
			"checkpoint-plan",
			[
				"export",
				"suite.json",
				"new output",
				"--target-accuracy",
				"0.5",
				"--seed",
				"1",
				"--seed",
				"2",
			],
			[
				"checkpoints",
				"plan",
				"--export",
				"export",
				"--suite",
				"suite.json",
				"--output",
				"new output",
				"--target-accuracy",
				"0.5",
				"--seed",
				"1",
				"--seed",
				"2",
			],
		),
	],
)
def test_just_preserves_argument_boundaries(
	tmp_path: Path, recipe: str, arguments: list[str], tail: list[str]
) -> None:
	just = shutil.which("just")
	if just is None:
		pytest.skip("just executable unavailable")
	binary = tmp_path / "bin"
	binary.mkdir()
	stub = binary / "uv"
	stub.write_text('#!/bin/sh\nprintf \'%s\\0\' "$@" > "$TRACE_ARGV_FILE"\n')
	stub.chmod(0o755)
	output = tmp_path / "arguments"
	result = subprocess.run(
		[just, "--justfile", str(ROOT / "justfile"), "--no-dotenv", recipe, *arguments],
		env=os.environ | {"PATH": f"{binary}:{os.environ['PATH']}", "TRACE_ARGV_FILE": str(output)},
		capture_output=True,
		text=True,
	)
	assert result.returncode == 0, result.stderr
	argv = output.read_bytes().decode().split("\0")[:-1]
	assert argv[-len(tail) :] == tail


def test_study_and_baseline_help_are_offline() -> None:
	for script in ("study.sh", "baseline-gates.sh"):
		result = subprocess.run(
			["bash", str(ROOT / "scripts" / script), "--help"], capture_output=True, text=True
		)
		assert result.returncode == 0 and "Usage:" in result.stdout


def _study_refusal(tmp_path: Path, **settings: str) -> subprocess.CompletedProcess[str]:
	"""Run only the declaration checks: a stub uv ensures nothing real could start after them."""
	stubs = tmp_path / "stubs"
	stubs.mkdir(exist_ok=True)
	stub = stubs / "uv"
	stub.write_text("#!/bin/sh\nexit 99\n")
	stub.chmod(0o755)
	environment = {
		key: value
		for key, value in os.environ.items()
		if not key.startswith(("HARNESS_", "CAMPAIGN_"))
	}
	return subprocess.run(
		["bash", str(ROOT / "scripts" / "study.sh"), "refused"],
		env=environment
		| {
			"PATH": f"{stubs}:{os.environ['PATH']}",
			"RUN_ROOT": str(tmp_path / "runs"),
			"EXPORT": "0",
			**settings,
		},
		capture_output=True,
		text=True,
		timeout=60,
	)


def test_study_campaign_arms_refuse_missing_declarations(tmp_path: Path) -> None:
	config = tmp_path / "campaign.json"
	config.write_text("{}")
	for settings, message in (
		({"ARMS": "campaign-observe"}, "declared campaign config"),
		({"ARMS": "campaign-observe", "CAMPAIGN_OBSERVE_CONFIG": str(config)}, "STRATEGY"),
		(
			{
				"ARMS": "campaign-observe",
				"CAMPAIGN_OBSERVE_CONFIG": str(config),
				"CAMPAIGN_RESTRICTION_STRATEGY": "wait_and_retry",
			},
			"positive CAMPAIGN_RESTRICTION_RETRIES",
		),
		(
			{
				"ARMS": "campaign-throttle",
				"CAMPAIGN_THROTTLE_CONFIG": str(config),
				"CAMPAIGN_RESTRICTION_STRATEGY": "give_up",
			},
			"HARNESS_CAMPAIGN_ALLOW_UNQUALIFIED_ENFORCEMENT",
		),
	):
		result = _study_refusal(tmp_path, **settings)
		assert result.returncode == 2, result.stdout + result.stderr
		assert message in result.stdout + result.stderr
		# Refused before provenance and setup, so no later step ran.
		assert not list((tmp_path / "runs" / "refused" / "provenance").iterdir())


def test_study_unprotected_student_needs_a_declared_target(tmp_path: Path) -> None:
	result = _study_refusal(tmp_path)
	assert result.returncode == 2, result.stdout + result.stderr
	assert "STUDENT_TARGET_ACCURACY" in result.stdout + result.stderr
	result = _study_refusal(tmp_path, ATTACKS="A1")
	assert result.returncode == 2 and "UNPROTECTED_ATTACK" in result.stdout + result.stderr
