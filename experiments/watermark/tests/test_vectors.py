import copy
import hashlib
import hmac
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from watermark_probe.__main__ import main
from watermark_probe.vectors import compute_vectors, find_mismatches, load_fixture

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "reference.json"


@pytest.fixture
def fixture() -> dict[str, Any]:
	return load_fixture(FIXTURE_PATH)


def test_computed_vectors_match_expected(fixture: dict[str, Any]) -> None:
	assert compute_vectors(fixture) == fixture["expected"]
	assert fixture["expected"]["algorithm_id"] == "kgw-hmac-splitmix"
	assert fixture["expected"]["algorithm_version"] == 1


def test_fixture_covers_the_required_profiles(fixture: dict[str, Any]) -> None:
	assert fixture["nonsecret"] is True
	assert {case["context_width"] for case in fixture["cases"]} == {1, 2, 4}
	assert any(case["vocab_size"] == 200000 for case in fixture["cases"])
	assert "key_hex" not in json.dumps(fixture["expected"])
	assert fixture["key_hex"] not in json.dumps(fixture["expected"])


@pytest.mark.parametrize(
	("case_index", "vector_index"),
	[(0, 1), (3, 0)],
	ids=["width1-vocab5000", "width4-vocab200000"],
)
def test_wire_format_by_independent_computation(
	fixture: dict[str, Any], case_index: int, vector_index: int
) -> None:
	case = fixture["cases"][case_index]
	context = case["contexts"][vector_index]
	header = struct.pack(
		">IIIdI", 1, fixture["epoch"], case["vocab_size"], case["gamma"], case["context_width"]
	)
	body = struct.pack(f">{len(context)}I", *context)
	message = b"trace-watermark-probe" + header + body
	digest = hmac.new(bytes.fromhex(fixture["key_hex"]), message, hashlib.sha256).digest()
	expected = fixture["expected"]["cases"][case_index]["vectors"][vector_index]
	assert expected["context"] == context
	assert digest[:8].hex() == expected["seed"]


def test_unlabelled_fixture_is_refused(fixture: dict[str, Any]) -> None:
	del fixture["nonsecret"]
	with pytest.raises(ValueError, match="nonsecret"):
		compute_vectors(fixture)


def test_mismatch_names_the_differing_value(fixture: dict[str, Any]) -> None:
	expected = copy.deepcopy(fixture["expected"])
	expected["cases"][1]["vectors"][2]["seed"] = "0" * 16
	mismatches = find_mismatches(expected, compute_vectors(fixture))
	assert len(mismatches) == 1
	assert mismatches[0].startswith("$.cases[1].vectors[2].seed: expected '0000")


def test_cli_vectors_writes_tab_indented_output(fixture: dict[str, Any], tmp_path: Path) -> None:
	output = tmp_path / "nested" / "vectors.json"
	assert main(["vectors", "--fixture", str(FIXTURE_PATH), "--output", str(output)]) == 0
	text = output.read_text(encoding="utf-8")
	assert '\n\t"algorithm_id"' in text
	assert "\n  " not in text
	assert json.loads(text) == fixture["expected"]


def test_cli_vectors_mismatch_fails_and_writes_nothing(
	fixture: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	fixture["expected"]["cases"][0]["vectors"][0]["mask_first_32"] = "wrong"
	broken = tmp_path / "broken.json"
	broken.write_text(json.dumps(fixture), encoding="utf-8")
	output = tmp_path / "vectors.json"
	assert main(["vectors", "--fixture", str(broken), "--output", str(output)]) == 1
	assert "cases[0].vectors[0].mask_first_32" in capsys.readouterr().err
	assert not output.exists()


def test_cli_vectors_requires_expected_values(fixture: dict[str, Any], tmp_path: Path) -> None:
	del fixture["expected"]
	bare = tmp_path / "bare.json"
	bare.write_text(json.dumps(fixture), encoding="utf-8")
	output = tmp_path / "vectors.json"
	assert main(["vectors", "--fixture", str(bare), "--output", str(output)]) == 1
	assert not output.exists()


@pytest.mark.parametrize(
	"command",
	[
		["replay", "--manifest", "m.jsonl", "--output", "nested/generations.jsonl"],
		["calibrate", "--split", "calibration", "--output", "nested/calibration.json"],
		["evaluate", "--split", "test", "--output", "nested/report.json"],
		["replay"],
	],
	ids=["replay", "calibrate", "evaluate", "replay-bare"],
)
def test_planned_commands_exit_2_and_write_nothing(
	command: list[str],
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	monkeypatch.chdir(tmp_path)
	assert main(command) == 2
	assert "not implemented" in capsys.readouterr().err
	assert list(tmp_path.iterdir()) == []


def test_python_dash_m_runs(tmp_path: Path) -> None:
	output = tmp_path / "vectors.json"
	result = subprocess.run(
		[
			sys.executable,
			"-m",
			"watermark_probe",
			"vectors",
			"--fixture",
			str(FIXTURE_PATH),
			"--output",
			str(output),
		],
		capture_output=True,
		text=True,
	)
	assert result.returncode == 0, result.stderr
	assert output.exists()
