"""Development-only separation study: two-sample tests, balancing and the CLI report."""

import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
from harness.cli import app
from harness.mmd import Array
from harness.separation import DISAPPEARS, SURVIVES, Prompt, StudyConfig, Window, compare, decide
from harness.separation_stats import auc, fit_logistic, fit_mlp, learned_mmd, permutation_p
from harness.serialization import detector_text, gateway_turn
from typer.testing import CliRunner


def held_out_p(shift: float) -> list[float]:
	rng = np.random.default_rng(7)
	x = rng.normal(size=(160, 6))
	y = np.repeat([0, 1], 80)
	x[y == 1, 0] += shift
	train, test = slice(0, None, 2), slice(1, None, 2)
	models = (fit_logistic(x[train], y[train]), fit_mlp(x[train], y[train], seed=3))
	values = [permutation_p(partial(auc, model(x[test])), y[test], 199, 5) for model in models]
	mmd = learned_mmd(x[train], y[train], x[test], y[test], 199, 5)
	return [*values, mmd["mmd_p_value"]]


def test_c2st_identical_distributions_are_not_significant() -> None:
	assert all(p > 0.05 for p in held_out_p(0.0))


def test_shifted_distribution_separates() -> None:
	assert all(p < 0.01 for p in held_out_p(2.0))


def domain_only_fixture() -> tuple[list[Window], list[list[Prompt]], Array]:
	"""Benign specialists and a mixed-domain attack with identical per-domain content."""
	rng = np.random.default_rng(11)
	centers = {"ag_news": np.array([3.0, 0, 0, 0]), "squad": np.array([-3.0, 0, 0, 0])}
	vectors: list[Array] = []

	def prompt(domain: str, source: str) -> Prompt:
		vectors.append(centers[domain] + rng.normal(size=4))
		row = len(vectors) - 1
		return Prompt(f"k{row}", domain, source, 40, float(row), row, 0)

	benign = []
	for account in range(48):
		domain = "ag_news" if account % 2 else "squad"
		rows = tuple(prompt(domain, f"b{account}-{i}") for i in range(8))
		benign.append(Window(0, f"user{account}", rows))
	pools = [
		[prompt(("ag_news", "squad")[i % 2], f"a{pool}-{i}") for i in range(400)] for pool in (0, 1)
	]
	return benign, pools, np.asarray(vectors)


def test_balancing_removes_pure_domain_mix_difference() -> None:
	benign, pools, vectors = domain_only_fixture()
	config = StudyConfig(window_size=8, permutations=199, bootstrap=200, minimum_windows=5)
	result = compare("mix", benign, pools, vectors, lambda _: lambda w: 0.0, config)
	controls = result["controls"]
	assert decide(controls["none"], config)["survives"]
	assert controls["none"]["tests"]["c2st_logistic"]["auc"] > 0.9
	assert not decide(controls["domain"], config)["survives"]
	assert controls["domain"]["windows"]["test"]["1"] >= 5


def write_lines(path: Path, rows: list[dict[str, object]]) -> None:
	path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cli_writes_report_on_synthetic_fixture(tmp_path: Path) -> None:
	rng = np.random.default_rng(13)
	keys, vectors = [], []

	def row(account: str, index: int, domain: str, label: str) -> dict[str, object]:
		turn = f"{label} request {account} {index}"
		keys.append(hashlib.sha256(detector_text("gateway", gateway_turn(domain, turn)).encode()))
		vectors.append(rng.normal(size=4) + (label == "attack"))
		return {
			"account": account,
			"time": float(index),
			"prompt": {
				"task": domain,
				"prompt": [{"role": "user", "content": turn}],
				"source": {"corpus": "fixture", "group_id": f"{account}-{index}"},
			},
		}

	domains = ("ag_news", "squad")
	benign = [row(f"u{a}", i, domains[a % 2], "benign") for a in range(12) for i in range(8)]
	attack = [row("attacker", i, domains[i % 2], "attack") for i in range(96)]
	write_lines(tmp_path / "benign.jsonl", benign)
	write_lines(tmp_path / "attack.jsonl", attack)
	cache = tmp_path / "cache.npz"
	np.savez(cache, keys=np.array([k.hexdigest() for k in keys]), vectors=np.array(vectors))
	output = tmp_path / "study"
	arguments = [
		"separation",
		"--benign",
		str(tmp_path / "benign.jsonl"),
		"--attack",
		f"A1={tmp_path / 'attack.jsonl'}",
		"--embeddings",
		str(cache),
		"--output",
		str(output),
		"--window-size",
		"4",
		"--topic-cells",
		"2",
		"--permutations",
		"19",
		"--bootstrap",
		"100",
	]
	result = CliRunner().invoke(app, arguments)
	assert result.exit_code == 0, result.output
	report = json.loads((output / "report.json").read_text())
	assert report["development_only"] and not report["used_for_calibration_or_enforcement"]
	assert report["decision"] in {SURVIVES, DISAPPEARS}
	assert [c["name"] for c in report["comparisons"]] == ["A1", "A1/topic-sorted"]
	assert set(report["comparisons"][0]["controls"]) == {
		"none",
		"domain",
		"topic",
		"size",
		"client",
		"all",
	}
	assert f"Decision: {report['decision']}" in (output / "summary.md").read_text()
	locked = tmp_path / "benign_test"
	locked.mkdir()
	(locked / "requests.jsonl").write_text((tmp_path / "benign.jsonl").read_text())
	arguments[2] = str(locked / "requests.jsonl")
	arguments[arguments.index("--output") + 1] = str(tmp_path / "other")
	refused = CliRunner().invoke(app, arguments)
	assert refused.exit_code != 0 and "benign_test" in refused.output
