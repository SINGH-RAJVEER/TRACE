"""Dashboard API contracts: real index rows, producer-built artifacts, path safety and redaction.

Artifact fixtures are written by the real producers into tmp_path; the frozen readings are only
read, never written.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from dashboard_api.app import create_app
from dashboard_api.files import Roots, parse_env, read_json
from dashboard_api.index import load_index
from distillation import transfer_metrics
from distillation.baseline_gates import clean_gate, write_gate
from distillation.checkpoint_learning import execute_checkpoint_curve, summarize_checkpoint_curve
from distillation.learning_curve import summarize_curve
from distillation.results_index import COLUMNS
from fastapi.testclient import TestClient
from test_checkpoint_learning import Recorder, build_export, build_suite
from test_checkpoint_learning import plan as checkpoint_plan
from test_learning_curve import _curve
from test_learning_curve import inputs as inputs  # noqa: F401  (pytest fixture)

READINGS = Path(__file__).resolve().parents[1] / "readings"


def client(roots: Roots, index: Path) -> TestClient:
	return TestClient(create_app(roots=roots, index=index, frontend=index.parent / "no-frontend"))


def runs(app: TestClient) -> dict[str, dict[str, Any]]:
	return {run["label"]: run for run in app.get("/api/runs").json()["runs"]}


# The checked-in index, read as it is.


def test_real_index_keeps_refusals_false_values_and_unknowns() -> None:
	loaded = load_index(READINGS / "index.tsv")
	assert loaded["state"] == "ok"
	rows = loaded["rows"]
	refused = [r for r in rows if r["status"] == "gate-refused"]
	assert refused and all(r["run_id"] == "v5-20261102" for r in refused)
	assert all(r["received"] is None and r["attempts"] is None for r in refused)
	a2 = next(r for r in rows if r["run_id"] == "backfill:v3" and r["attack"] == "A2")
	assert a2["attribution_complete"] is False and a2["status"] == "interrupted"
	assert a2["unattributed_queries"] == 1537
	quota = next(r for r in rows if r["run_id"] == "backfill:v3" and r["arm"] == "quota")
	assert quota["blocked"] == 4785 and quota["challenge"] == 0
	utility = [r for r in rows if r["kind"] == "utility"]
	assert utility and {r["validity"] for r in utility} == {"unknown"}
	assert {r["study"] for r in utility if r["run_id"] == "v5-20261102"} == {None}


def test_real_readings_show_unavailable_students_and_gates() -> None:
	app = client(Roots.parse("readings", READINGS.parent), READINGS / "index.tsv")
	found = runs(app)
	v5 = found["v5-20261102"]
	assert v5["studies"] == ["development"] and v5["places"] == [{"root": "readings", "path": "v5"}]
	detail = app.get(f"/api/runs/{v5['id']}").json()
	campaigns = detail["campaigns"]["items"]
	assert len(campaigns) == 18 and detail["campaigns"]["state"] == "available"
	refused = [c for c in campaigns if c["status"] == "gate-refused"]
	assert len(refused) == 8 and all(c["outcome"].startswith("gate refused") for c in refused)
	assert len(detail["gates"]["detectability"]["items"]) == 8
	for name in ("curves", "checkpoints", "transfers", "index"):
		assert detail["students"][name]["state"] == "unavailable"
		assert detail["students"][name]["items"] == []
	assert detail["gates"]["reports"]["state"] == "unavailable"
	assert detail["teacher"]["capacity"]["state"] == "unavailable"
	teacher = detail["provenance"]["sources"][0]["teacher"]
	assert teacher["TEACHER_MODEL"] == "Qwen/Qwen3.5-9B"
	assert teacher["TEACHER_GGUF"] is None
	full = next(a for a in detail["utility"]["items"] if a["arm"] == "full")
	assert full["validity"] == "unknown" and full["study"] is None
	assert {s["group"] for s in full["strata"]} == {"domain", "domain_profile", "client_type"}
	v3 = found["backfill:v3"]
	assert v3["places"] == [{"root": "readings", "path": "v3"}] and v3["commits"] == []


# Index failure states.


def test_index_failures_are_states_not_rows(tmp_path: Path) -> None:
	assert load_index(tmp_path / "absent.tsv")["state"] == "missing"
	pointer = tmp_path / "pointer.tsv"
	pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n")
	assert load_index(pointer)["state"] == "lfs-pointer"
	unknown = tmp_path / "unknown.tsv"
	unknown.write_text("run_id\tsurprise\nx\ty\n")
	assert load_index(unknown)["state"] == "malformed"
	bad = tmp_path / "bad.tsv"
	row = dict.fromkeys(COLUMNS, "")
	row.update(run_id="r", kind="campaign", attribution_complete="maybe")
	bad.write_text("\t".join(COLUMNS) + "\n" + "\t".join(row[c] for c in COLUMNS) + "\n")
	result = load_index(bad)
	assert result["state"] == "malformed" and "attribution_complete" in result["error"]
	app = client(Roots.parse("artifacts", tmp_path), bad)
	assert app.get("/api/runs").json()["index"]["state"] == "malformed"


# Path safety and secrets.


def test_paths_cannot_leave_their_root(tmp_path: Path) -> None:
	root = tmp_path / "artifacts"
	(root / "run").mkdir(parents=True)
	(tmp_path / "outside").mkdir()
	(root / "escape").symlink_to(tmp_path / "outside")
	roots = Roots.parse(f"artifacts={root}", tmp_path)
	assert roots.resolve("artifacts", "run") == (root / "run").resolve()
	for bad in ("../outside", "/etc/passwd", "run/../../outside", "run\\..\\x", "escape", "a\x00b"):
		assert roots.resolve("artifacts", bad) is None, bad
	assert roots.resolve("missing", "run") is None
	with pytest.raises(ValueError):
		Roots.parse("a=x,a=y", tmp_path)
	with pytest.raises(ValueError):
		Roots.parse("bad name=x", tmp_path)
	(root / "run" / "credentials.json").write_text('{"token": "x"}')
	assert read_json(root / "run" / "credentials.json", "c").state == "refused"
	app = client(roots, tmp_path / "index.tsv")
	assert app.get("/api/runs/..%2F..%2Fetc").status_code == 404
	assert app.get("/api/runs/0000000000000000").status_code == 404
	assert app.post("/api/runs").status_code == 405


def test_secrets_are_redacted_and_credentials_never_served(tmp_path: Path) -> None:
	pairs = parse_env(
		"HARNESS_TEACHER_TOKEN=abc\nTEACHER_UPSTREAM_API_KEY=k\nEMPTY_TOKEN=\n"
		"HARNESS_TEACHER_URL=http://user:pw@127.0.0.1:8000\nSEED=42\n# comment\n"
	)
	values = {p["key"]: p["value"] for p in pairs}
	assert values["HARNESS_TEACHER_TOKEN"] == "<redacted>"
	assert values["TEACHER_UPSTREAM_API_KEY"] == "<redacted>"
	assert values["EMPTY_TOKEN"] == ""
	assert values["HARNESS_TEACHER_URL"] == "http://<redacted>@127.0.0.1:8000"
	assert values["SEED"] == "42"

	run = tmp_path / "artifacts" / "runs" / "r1"
	attempt = run / "provenance" / "20260101T000000Z"
	attempt.mkdir(parents=True)
	(attempt / "config.env").write_text(
		"SEED=42\nTEACHER_MODEL=m\nTEACHER_API_TOKEN=very-secret-token\n"
	)
	(attempt / "environment.env").write_text("OPENAI_API_KEY=environment-secret\n")
	(run / "steps.tsv").write_text("")
	(run / "exit").write_text("0 2026-01-01T00:00:00Z\n")
	arm = run / "campaigns" / "A1" / "full"
	arm.mkdir(parents=True)
	(arm / "outcome").write_text(
		"gate refused: attacker not escalated beyond held-out benign accounts\n"
	)
	(arm / "credentials.json").write_text('{"key": "credential-secret"}')
	(arm / "harness.env").write_text("HARNESS_POLICY=mmd\nHARNESS_TEACHER_TOKEN=harness-secret\n")
	app = client(Roots.parse("artifacts", tmp_path), tmp_path / "index.tsv")
	found = runs(app)
	entry = found["artifacts/runs/r1"]
	assert entry["kind"] == "study"
	text = app.get(f"/api/runs/{entry['id']}").text
	for secret in (
		"very-secret-token",
		"environment-secret",
		"credential-secret",
		"harness-secret",
	):
		assert secret not in text
	detail = json.loads(text)
	campaign = detail["campaigns"]["items"][0]
	assert campaign["status"] == "gate-refused" and campaign["source"] == "directory"
	assert campaign["policy"] == "mmd" and campaign["received"] is None
	assert detail["utility"]["state"] == "unavailable"


# Producer-built artifacts.


def test_baseline_gate_run_from_real_producers(
	inputs: tuple[Path, Path, Path],  # noqa: F811
	tmp_path: Path,
) -> None:
	work = tmp_path / "artifacts" / "baseline-gates" / "g1"
	work.parent.mkdir(parents=True)
	curve, headroom, manifest = _curve(work, inputs)
	summarize_curve(curve, replicates=200)
	headroom.rename(work / "gate-headroom.json")
	report = clean_gate(curve, work / "gate-headroom.json", manifest, replicates=1000)
	write_gate(report, work / "gate-clean.json")
	one_seed = _curve(tmp_path / "one", inputs, seeds=(42,))
	write_gate(clean_gate(*one_seed, replicates=1000), work / "gate-clean-one-seed.json")
	tampered = work / "gate-tampered.json"
	tampered.write_text(
		json.dumps({**report, "passed": False, "status": "failed"}).replace("\n", "")
	)
	(work / "gate-truncated.json").write_text('{"version": "clean-baseline-gate-v1", "gate"')
	correct = {
		role: {f"f{i}": [True] * k + [False] * (10 - k) for i in range(6)}
		for role, k in {"teacher": 9, "base": 3, "clean": 7, "defended": 5}.items()
	}
	(work / "transfer.json").write_text(
		json.dumps(transfer_metrics.transfer_report(correct, replicates=200))
	)
	app = client(Roots.parse("artifacts", tmp_path), tmp_path / "index.tsv")
	found = runs(app)
	entry = found["artifacts/baseline-gates/g1"]
	assert entry["kind"] == "baseline-gates"
	assert entry["artifacts"]["learning-curve"] == 1 and entry["artifacts"]["transfer"] == 1
	detail = app.get(f"/api/runs/{entry['id']}").json()

	(head,) = detail["teacher"]["headroom"]["items"]
	assert head["passed"] is True and head["integrity"] == "intact"
	assert head["teacher"] == {"model": "teacher", "revision": "t1"}
	assert head["results"]["headroom"]["gain_pp"] == pytest.approx(50)

	reports = {r["place"]["path"].rsplit("/", 1)[1]: r for r in detail["gates"]["reports"]["items"]}
	assert reports["gate-clean.json"]["passed"] is True
	assert reports["gate-clean.json"]["results"]["clean"][0]["n"] == 300
	single = reports["gate-clean-one-seed.json"]
	assert single["status"] == "development-only" and single["passed"] is False
	assert single["reasons"]
	assert reports["gate-tampered.json"]["integrity"].startswith("digest does not match")
	assert reports["gate-tampered.json"]["passed"] is False
	assert reports["gate-truncated.json"]["file"]["state"] == "malformed"

	(curve_item,) = detail["students"]["curves"]["items"]
	assert curve_item["integrity"] == "intact" and curve_item["task"] == "gsm8k"
	assert curve_item["recipe"]["base_model"] == "Qwen/Qwen3.5-2B-Base"
	assert [r["n"] for r in curve_item["summary"]["results"]] == [300, 1000]
	assert [r["arm"] for r in curve_item["summary"]["gold_results"]] == ["gold", "gold"]

	(transfer,) = detail["students"]["transfers"]["items"]
	single_report = transfer["seeds"]["single"]
	assert single_report["suppression"]["value"] is None
	assert single_report["suppression"]["reason"]
	assert single_report["decision"]["meets_target"] is None
	assert single_report["raw"]["overall"]["defended"] == pytest.approx(0.5)


def test_checkpoint_summary_keeps_censored_checkpoints(tmp_path: Path) -> None:
	export = build_export(tmp_path / "export")
	suite = build_suite(tmp_path / "suite.json")
	root = tmp_path / "artifacts"
	root.mkdir()
	checkpoint_plan(root, export, suite)
	output = root / "plan"
	app = client(Roots.parse("artifacts", tmp_path), tmp_path / "index.tsv")
	planned = runs(app)["artifacts/plan"]
	assert planned["kind"] == "checkpoints"
	detail = app.get(f"/api/runs/{planned['id']}").json()
	(item,) = detail["students"]["checkpoints"]["items"]
	assert item["summary"] is None and item["summary_file"]["state"] == "missing"

	execute_checkpoint_curve(output, Recorder())
	summarize_checkpoint_curve(output)
	detail = app.get(f"/api/runs/{planned['id']}").json()
	(item,) = detail["students"]["checkpoints"]["items"]
	rows = {r["id"]: r for r in item["summary"]["checkpoints"]}
	assert rows["response-100"]["status"] == "censored"
	assert rows["response-100"]["mean_accuracy"] is None
	assert item["summary"]["censored_checkpoints"] == ["response-100"]
	assert item["summary"]["base_meets_target"] is False


def test_utility_from_a_run_directory_keeps_nulls(tmp_path: Path) -> None:
	run = tmp_path / "artifacts" / "runs" / "r2"
	arm = run / "utility" / "quota"
	arm.mkdir(parents=True)
	(run / "steps.tsv").write_text("")
	(run / "exit").write_text("0\n")
	(arm / "harness.env").write_text("HARNESS_POLICY=quota\n")
	latency = {"n": 0, "p50": None, "p95": None, "p99": None}
	(arm / "utility.json").write_text(
		json.dumps(
			{
				"submitted": 4,
				"served": 0,
				"completion_rate": 0.0,
				"served_only_quality": None,
				"unconditional_utility": 0.0,
				"latency_p95": None,
				"latency_p50": None,
				"strata": {
					"domain": {
						"gsm8k": {
							"submitted": 4,
							"served": 0,
							"latency_seconds": latency,
							"served_only_quality": None,
							"unconditional_utility": 0.0,
						}
					}
				},
				"results": [{"account": "a"}],
			}
		)
	)
	app = client(Roots.parse("artifacts", tmp_path), tmp_path / "index.tsv")
	entry = runs(app)["artifacts/runs/r2"]
	detail = app.get(f"/api/runs/{entry['id']}").json()
	(quota,) = detail["utility"]["items"]
	assert quota["source"] == "directory" and quota["policy"] == "quota"
	assert quota["validity"] == "unknown"
	assert quota["pooled"]["served"] == 0 and quota["pooled"]["served_only_quality"] is None
	assert quota["pooled"]["completion_rate"] == 0.0
	(stratum,) = quota["strata"]
	assert stratum == {
		"group": "domain",
		"value": "gsm8k",
		"submitted": 4,
		"served": 0,
		"completion_rate": 0.0,
		"served_only_quality": None,
		"unconditional_utility": 0.0,
		"latency_p95": None,
	}
	assert "results" not in json.dumps(quota["detail"])
