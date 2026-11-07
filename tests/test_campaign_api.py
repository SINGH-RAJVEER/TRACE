"""HTTP integration tests for the experimental campaign detector: opt-in observation and the
explicit, unqualified throttle. CPU fixtures only; the scripted prompts here are not evidence
that the detector is calibrated or effective."""

import asyncio
import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from distillation.benchmarks.traffic_runner import solve_challenge
from harness import campaign as campaign_module
from harness.calibration import BenignRequest, Calibration, HarnessConfig, calibrate
from harness.campaign import CampaignError
from harness.mmd import WindowScorer
from harness_api import app as app_module
from harness_api.app import _policy_fields
from harness_api.settings import Settings
from test_harness_api import Encoder, _app, _events, _request

Artifacts = tuple[Calibration, WindowScorer, WindowScorer]
PATH = "/v1/teacher/generate"


@pytest.fixture(scope="module")
def artifacts() -> Artifacts:
	requests = [BenignRequest(account=f"a{i // 20}", text=f"q {i}") for i in range(400)]
	calibration, references = calibrate(
		config=HarnessConfig(
			window_size=2,
			reference_windows=2,
			bandwidth_sample=20,
			aggregate_window_size=10,
			aggregate_reference_windows=2,
		),
		encoder=Encoder(),
		fitting=requests[:100],
		score_calibration=requests[100:200],
		policy_calibration=requests[200:],
	)
	assert calibration.aggregate is not None and calibration._aggregate_references is not None
	aggregate = WindowScorer(calibration._aggregate_references, calibration.sigma)
	return calibration, WindowScorer(references, calibration.sigma), aggregate


def _config(tmp_path: Path, name: str = "campaign.json", **overrides: Any) -> Path:
	directory = tmp_path / "campaign"
	directory.mkdir(exist_ok=True)
	body = {
		"namespace": "test-deployment",
		"window_size": 8,
		"min_samples": 3,
		"window_seconds": 600.0,
		"cooldown_seconds": 120.0,
		"threshold": 0.8,
		"mode": "throttle",
	} | overrides
	path = directory / name
	path.write_text(json.dumps(body))
	return path


def _campaign_app(
	artifacts: Artifacts,
	tmp_path: Path,
	*,
	mode: str = "throttle",
	levels: list[float] | None = None,
	config: dict[str, Any] | None = None,
	**settings: Any,
) -> tuple[Any, Any, Path]:
	options: dict[str, Any] = {
		"HARNESS_POLICY": "allow",
		"HARNESS_TEST_CLOCK": True,
		"HARNESS_STATE_DB": tmp_path / "state.sqlite3",
		"HARNESS_CAMPAIGN_CONFIG": _config(tmp_path, mode=mode, **(config or {})),
	}
	if mode == "throttle":
		options["HARNESS_CAMPAIGN_ALLOW_UNQUALIFIED_ENFORCEMENT"] = True
	return _app(artifacts, tmp_path, levels or [1e9] * 4, **(options | settings))


def _client(app: Any) -> httpx.AsyncClient:
	return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://harness")


def _templated(index: int) -> str:
	return f"Summarize support ticket number {index} for the weekly report"


async def _post(
	client: httpx.AsyncClient,
	text: str,
	now: float | None = None,
	*,
	actor: str = "u1",
	headers: dict[str, str] | None = None,
) -> httpx.Response:
	sent = {"X-Benchmark-Actor": actor} | (headers or {})
	if now is not None:
		sent["X-Harness-Time"] = str(now)
	return await client.post(PATH, json=_request(text), headers=sent)


async def _until_throttled(
	client: httpx.AsyncClient, *, start: float = 1000.0, limit: int = 12, actor: str = "u1"
) -> list[httpx.Response]:
	responses: list[httpx.Response] = []
	for index in range(limit):
		response = await _post(client, _templated(index), start + index, actor=actor)
		responses.append(response)
		if response.status_code == 429:
			break
	return responses


def _code(response: httpx.Response) -> str:
	return str(response.json()["detail"]["code"])


def test_settings_require_durable_state_and_the_explicit_override(tmp_path: Path) -> None:
	config = _config(tmp_path)
	with pytest.raises(ValueError, match="HARNESS_STATE_DB"):
		Settings(_env_file=None, HARNESS_CAMPAIGN_CONFIG=config)  # type: ignore[call-arg]
	with pytest.raises(ValueError, match="requires HARNESS_CAMPAIGN_CONFIG"):
		Settings(  # type: ignore[call-arg]
			_env_file=None, HARNESS_CAMPAIGN_ALLOW_UNQUALIFIED_ENFORCEMENT=True
		)
	unset = Settings(_env_file=None)  # type: ignore[call-arg]
	assert unset.campaign_config is None and not unset.campaign_allow_unqualified_enforcement
	enabled = Settings(  # type: ignore[call-arg]
		_env_file=None, HARNESS_CAMPAIGN_CONFIG=config, HARNESS_STATE_DB=tmp_path / "s.db"
	)
	assert enabled.campaign_config == config and not enabled.campaign_allow_unqualified_enforcement


def test_throttle_needs_the_override_while_observe_does_not(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	with pytest.raises(ValueError, match="unqualified"):
		_app(
			artifacts,
			tmp_path,
			[1e9] * 4,
			HARNESS_POLICY="allow",
			HARNESS_STATE_DB=tmp_path / "throttle.db",
			HARNESS_CAMPAIGN_CONFIG=_config(tmp_path, mode="throttle"),
		)
	_campaign_app(artifacts, tmp_path, mode="observe")


def test_invalid_config_files_are_refused_at_startup(artifacts: Artifacts, tmp_path: Path) -> None:
	for name, body in {
		"wrapped.json": {"campaign": {"namespace": "x"}},
		"unknown.json": {"namespace": "x", "calibrated": True},
		"qualified.json": {"namespace": "x", "qualified": True},
		"empty.json": {"namespace": " "},
		"samples.json": {"namespace": "x", "window_size": 4, "min_samples": 5},
	}.items():
		path = tmp_path / name
		path.write_text(json.dumps(body))
		with pytest.raises(CampaignError):
			_app(
				artifacts,
				tmp_path,
				[1e9] * 4,
				HARNESS_POLICY="allow",
				HARNESS_STATE_DB=tmp_path / "invalid.db",
				HARNESS_CAMPAIGN_CONFIG=path,
			)
	# The config is read before the state file is opened, so no state was created.
	assert not (tmp_path / "invalid.db").exists()


def _tables(db: Path) -> set[str]:
	with sqlite3.connect(db) as connection:
		return {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}


@pytest.mark.asyncio
async def test_unset_feature_keeps_revision_scope_schema_and_log_contract(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	plain = Settings(_env_file=None)  # type: ignore[call-arg]
	assert "campaign" not in _policy_fields(plain, artifacts[0])
	app, teacher, log = _app(
		artifacts,
		tmp_path,
		[1e9] * 4,
		HARNESS_POLICY="allow",
		HARNESS_TEST_CLOCK=True,
		HARNESS_STATE_DB=tmp_path / "legacy.db",
	)
	async with _client(app) as client:
		for index in range(6):
			assert (await _post(client, _templated(index), 10.0 + index)).status_code == 200
	assert teacher.calls == 6
	assert not any(
		"campaign" in event or "campaign_override" in event
		for event in (json.loads(line) for line in log.read_text().splitlines())
	)
	assert "campaign" not in _tables(tmp_path / "legacy.db")
	with sqlite3.connect(tmp_path / "legacy.db") as connection:
		keys = {row[0] for row in connection.execute("SELECT key FROM meta")}
	assert "campaign_config" not in keys
	# A state file written without the feature cannot be reused under it.
	with pytest.raises(ValueError, match="another policy"):
		_app(
			artifacts,
			tmp_path,
			[1e9] * 4,
			HARNESS_POLICY="allow",
			HARNESS_TEST_CLOCK=True,
			HARNESS_STATE_DB=tmp_path / "legacy.db",
			HARNESS_CAMPAIGN_CONFIG=_config(tmp_path, mode="observe"),
		)


@pytest.mark.asyncio
async def test_enabled_feature_binds_scope_and_revision_to_the_config_digest(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	observe, _, log = _campaign_app(artifacts, tmp_path, mode="observe")
	async with _client(observe) as client:
		assert (await _post(client, _templated(0), 1.0)).status_code == 200
	decision = _events(log, "decision")[0]
	digest = decision["campaign"]["config_sha256"]
	assert len(digest) == 64 and decision["campaign"]["qualified"] is False
	revisions = {
		event["policy_revision"] for event in _events(log, "decision") + _events(log, "release")
	}
	assert len(revisions) == 1
	# Same state file, different campaign config: refused rather than mixed.
	with pytest.raises(ValueError, match="another policy"):
		_campaign_app(artifacts, tmp_path, mode="observe", config={"cooldown_seconds": 60.0})
	enabled = Settings(  # type: ignore[call-arg]
		_env_file=None,
		HARNESS_POLICY="allow",
		HARNESS_STATE_DB=tmp_path / "x.db",
		HARNESS_CAMPAIGN_CONFIG=_config(tmp_path, mode="observe"),
	)
	loaded = campaign_module.load_campaign_config(_config(tmp_path, mode="observe"))
	fields = _policy_fields(enabled, artifacts[0], None, loaded)
	assert fields["campaign"] == {"config_sha256": loaded.digest, "unqualified_enforcement": False}


@pytest.mark.asyncio
async def test_observe_mode_keeps_every_response_and_logs_unqualified_evidence(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	baseline, baseline_teacher, _ = _app(
		artifacts,
		tmp_path,
		[1e9] * 4,
		HARNESS_POLICY="allow",
		HARNESS_TEST_CLOCK=True,
		HARNESS_STATE_DB=tmp_path / "baseline.db",
	)
	observed, teacher, log = _campaign_app(artifacts, tmp_path, mode="observe")
	secret = "zebra-quartz-marker"
	async with _client(baseline) as plain, _client(observed) as client:
		for index in range(12):
			text = f"{_templated(index)} {secret}"
			left = await _post(plain, text, 1000.0 + index)
			right = await _post(client, text, 1000.0 + index)
			assert (left.status_code, left.headers["x-harness-decision"]) == (
				right.status_code,
				right.headers["x-harness-decision"],
			)
			assert set(left.json()) == set(right.json()) and right.status_code == 200
	assert teacher.calls == baseline_teacher.calls == 12
	evidence = [event["campaign"] for event in _events(log, "decision")]
	assert [item["evaluated_count"] for item in evidence] == list(range(1, 13))
	assert all(item["qualified"] is False and item["should_throttle"] is False for item in evidence)
	assert evidence[0]["status"] == "warming" and evidence[0]["score"] is None
	assert evidence[-1]["retained_count"] <= 8
	assert any(item["status"] == "alert" for item in evidence)
	assert not _events(log, "campaign_throttle")
	# The prompt never reaches the campaign evidence or any other log line.
	assert secret not in log.read_text()


@pytest.mark.asyncio
async def test_throttle_refuses_before_the_teacher_and_stays_unqualified(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	app, teacher, log = _campaign_app(artifacts, tmp_path)
	async with _client(app) as client:
		responses = await _until_throttled(client)
		throttled = responses[-1]
		assert throttled.status_code == 429 and _code(throttled) == "campaign_throttled"
		assert [r.status_code for r in responses[:-1]] == [200] * (len(responses) - 1)
		assert throttled.headers["x-harness-decision"] == "throttle"
		assert int(throttled.headers["retry-after"]) >= 1
		body = throttled.json()["campaign"]
		assert body["qualified"] is False and body["mode"] == "throttle"
		assert "score" not in body and "components" not in body
		served = teacher.calls
		assert served == len(responses) - 1
		# Retries during the cooldown are refused without extending its fixed expiry.
		until = body["throttle_until"]
		for index in range(3):
			again = await _post(client, _templated(50 + index), 1100.0 + index)
			assert again.status_code == 429 and again.json()["campaign"]["throttle_until"] == until
		assert teacher.calls == served
	events = _events(log, "campaign_throttle")
	assert {event["stage"] for event in events} == {"admission"}
	assert all(event["qualified"] is False for event in events)
	# A throttled request is neither an admission nor a release.
	assert len(_events(log, "decision")) == served
	assert len(_events(log, "release")) == served
	assert events[0]["campaign"]["status"] == "alert"


@pytest.mark.asyncio
async def test_throttle_isolates_principals_and_organizations(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	keys = {
		"key-alice": ("alice", "org-a"),
		"key-carol": ("carol", "org-a"),
		"key-bob": ("bob", "org-b"),
	}
	accounts = tmp_path / "accounts.json"
	accounts.write_text(
		json.dumps(
			{
				hashlib.sha256(key.encode()).hexdigest(): {"principal": name, "organization": org}
				for key, (name, org) in keys.items()
			}
		)
	)
	app, teacher, _ = _campaign_app(artifacts, tmp_path, HARNESS_ACCOUNTS_FILE=accounts)
	diverse = [
		"Translate the phrase good morning into French",
		"What is the capital of Australia",
		"Write a haiku about autumn rain",
		"Explain how a transistor amplifies current",
		"List three uses of baking soda",
		"Why is the sky blue at noon",
	]

	async def ask(key: str, text: str, now: float) -> httpx.Response:
		return await client.post(
			PATH,
			json=_request(text),
			headers={"Authorization": f"Bearer {key}", "X-Harness-Time": str(now)},
		)

	async with _client(app) as client:
		statuses = []
		for index in range(12):
			statuses.append((await ask("key-alice", _templated(index), 1000.0 + index)).status_code)
			if statuses[-1] == 429:
				break
		assert statuses[-1] == 429
		before = teacher.calls
		# The same organization and an unrelated one are unaffected by alice's cooldown.
		for index, text in enumerate(diverse):
			assert (await ask("key-carol", text, 1020.0 + index)).status_code == 200
			assert (await ask("key-bob", text, 1020.0 + index)).status_code == 200
		assert teacher.calls == before + 2 * len(diverse)
		assert (await ask("key-alice", diverse[0], 1030.0)).status_code == 429


@pytest.mark.asyncio
async def test_idempotent_retries_are_not_counted_and_replay_the_throttle(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	app, teacher, log = _campaign_app(artifacts, tmp_path)
	async with _client(app) as client:
		first = await _post(client, _templated(0), 1000.0, headers={"Idempotency-Key": "k0"})
		replay = await _post(client, _templated(0), 1001.0, headers={"Idempotency-Key": "k0"})
		assert first.status_code == replay.status_code == 200 and first.json() == replay.json()
		assert (await _post(client, _templated(0), 1002.0)).status_code == 200
		conflict = await _post(client, _templated(9), 1003.0, headers={"Idempotency-Key": "k0"})
		assert conflict.status_code == 409 and _code(conflict) == "idempotency_conflict"
		assert teacher.calls == 2
		# Every retry above left the evidence alone: two evaluated admissions so far.
		assert [e["campaign"]["evaluated_count"] for e in _events(log, "decision")] == [1, 2]
		keyed = await _post(client, _templated(1), 1004.0, headers={"Idempotency-Key": "k1"})
		assert keyed.status_code == 429
		active_retry = await _post(client, _templated(1), 1100.0, headers={"Idempotency-Key": "k1"})
		assert active_retry.status_code == 429
		assert active_retry.headers["retry-after"] == "24"
		assert active_retry.json()["campaign"]["retry_after_seconds"] == 24
		assert active_retry.json()["campaign"]["throttle_until"] == 1124.0
		replayed = await _post(client, _templated(1), 1200.0, headers={"Idempotency-Key": "k1"})
		assert replayed.status_code == 200
		assert len(_events(log, "campaign_throttle")) == 1
		assert teacher.calls == 3


@pytest.mark.asyncio
async def test_requests_refused_in_cooldown_cannot_be_replayed_unscored(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	app, teacher, log = _campaign_app(artifacts, tmp_path)
	async with _client(app) as client:
		responses = await _until_throttled(client)
		until = responses[-1].json()["campaign"]["throttle_until"]
		keys = [{"Idempotency-Key": f"banked-{index}"} for index in range(8)]
		for index, headers in enumerate(keys):
			refused = await _post(client, _templated(100 + index), until - 10, headers=headers)
			assert refused.status_code == 429
		served = teacher.calls
		replays = [
			await _post(client, _templated(100 + index), until + 10, headers=headers)
			for index, headers in enumerate(keys)
		]
	# The window restarts at expiry, so replays are scored and earn a new cooldown.
	assert [r.status_code for r in replays] == [200, 200] + [429] * 6
	assert teacher.calls == served + 2
	assert _events(log, "decision")[-1]["campaign"]["retained_count"] == 2


@pytest.mark.asyncio
async def test_exact_repeats_throttle_only_when_they_can_yield_new_samples(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	app, _, _ = _campaign_app(artifacts, tmp_path)
	body = _request("ping")
	sampled = body | {"temperature": 0.7}

	async def statuses(payload: dict[str, Any], actor: str) -> list[int]:
		return [
			(
				await client.post(
					PATH,
					json=payload,
					headers={"X-Benchmark-Actor": actor, "X-Harness-Time": str(1000.0 + index)},
				)
			).status_code
			for index in range(4)
		]

	async with _client(app) as client:
		assert await statuses(body, "u1") == [200] * 4
		assert (await statuses(sampled, "u2"))[-1] == 429


@pytest.mark.asyncio
async def test_questions_past_the_scored_text_bound_are_not_one_template(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	app, _, _ = _campaign_app(artifacts, tmp_path, config={"max_text_chars": 64})
	context = "shared document words " * 10
	async with _client(app) as client:
		for index in range(6):
			response = await _post(client, f"{context} ask {chr(97 + index)}", 1000.0 + index)
			assert response.status_code == 200


@pytest.mark.asyncio
async def test_restart_keeps_the_evidence_and_the_cooldown_until_it_expires(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	first, first_teacher, first_log = _campaign_app(artifacts, tmp_path)
	async with _client(first) as client:
		responses = await _until_throttled(client)
		until = responses[-1].json()["campaign"]["throttle_until"]
	served = first_teacher.calls
	second, second_teacher, second_log = _campaign_app(artifacts, tmp_path)
	async with _client(second) as client:
		# Evidence and cooldown survive the restart; no horizon starts over.
		during = await _post(client, _templated(70), 1000.0 + len(responses) + 5)
		assert during.status_code == 429 and during.json()["campaign"]["throttle_until"] == until
		assert second_teacher.calls == 0
		# After the cooldown, retained evidence still counts, and a clean window restores service.
		later = until + 700.0
		restored = await _post(client, _templated(80), later)
		assert restored.status_code == 200 and second_teacher.calls == 1
	assert _events(second_log, "decision")[0]["campaign"]["evaluated_count"] > served
	assert _events(first_log, "campaign_throttle")


@pytest.mark.asyncio
async def test_cooldown_is_fixed_and_scoring_resumes_after_expiry(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	app, teacher, log = _campaign_app(artifacts, tmp_path)
	async with _client(app) as client:
		responses = await _until_throttled(client)
		until = responses[-1].json()["campaign"]["throttle_until"]
		# Just before expiry the cooldown still holds, and reading it did not move it.
		assert (await _post(client, _templated(60), until - 1.0)).status_code == 429
		# Past expiry and past the evidence window the account is served again.
		assert (await _post(client, _templated(61), until + 1000.0)).status_code == 200
		assert teacher.calls == len(responses)
	assert not any(
		event["throttle_until"] != until
		for event in _events(log, "campaign_throttle")
		if event["time"] < until
	)


@pytest.mark.asyncio
async def test_proof_cannot_redeem_during_an_active_cooldown(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	app, teacher, log = _campaign_app(
		artifacts,
		tmp_path,
		levels=[-1, -1, -1, 1e9],
		HARNESS_POLICY="mmd",
		HARNESS_CHALLENGE_TTL_SECONDS=100000,
	)
	async with _client(app) as client:
		keyed = {"Idempotency-Key": "k2"}
		assert (await _post(client, _templated(1), 1000.0)).status_code == 200
		challenged = await _post(client, _templated(2), 1001.0, headers=keyed)
		assert challenged.status_code == 428
		challenge = challenged.json()["challenge"]
		proof = {
			"Idempotency-Key": "k2",
			"X-Harness-Proof": f"{challenge['nonce']}:"
			f"{solve_challenge(challenge['prefix'], challenge['difficulty'])}",
		}
		for index in range(3, 12):
			throttled = await _post(client, _templated(index), 1000.0 + index)
			if throttled.status_code == 429:
				break
		assert throttled.status_code == 429 and _code(throttled) == "campaign_throttled"
		until = throttled.json()["campaign"]["throttle_until"]
		calls = teacher.calls
		# A valid proof for the earlier challenge cannot bypass the active cooldown, and the
		# refusal spends neither the challenge nor its reservation.
		blocked = await _post(client, _templated(2), until - 5.0, headers=proof)
		assert blocked.status_code == 429 and _code(blocked) == "campaign_throttled"
		assert teacher.calls == calls
		assert {e["stage"] for e in _events(log, "campaign_throttle")} == {"admission", "proof"}
		assert [e["solved"] for e in _events(log, "redeem")] == []
		# After expiry the same proof, with the same key, still redeems exactly once.
		served = await _post(client, _templated(2), until + 1.0, headers=proof)
		assert served.status_code == 200 and teacher.calls == calls + 1
		again = await _post(client, _templated(2), until + 2.0, headers=proof)
		assert again.status_code == 200 and teacher.calls == calls + 1
		assert [e["solved"] for e in _events(log, "redeem")] == [True]


@pytest.mark.asyncio
async def test_one_request_grant_cannot_redeem_multiple_held_proofs(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	app, teacher, _ = _campaign_app(
		artifacts,
		tmp_path,
		levels=[-1, -1, -1, 1e9],
		config={"min_samples": 6},
		HARNESS_POLICY="mmd",
		HARNESS_TEST_CLOCK=False,
		HARNESS_REVIEW_ENABLED=True,
		HARNESS_REVIEW_ADMIN_KEY="admin",
		HARNESS_CHALLENGE_TTL_SECONDS=100000,
	)
	challenges: list[tuple[int, dict[str, Any]]] = []
	async with _client(app) as client:
		for index in range(8):
			response = await _post(
				client, _templated(index), headers={"Idempotency-Key": f"held-{index}"}
			)
			if response.status_code == 428:
				challenges.append((index, response.json()["challenge"]))
			if response.status_code == 429:
				break
		assert response.status_code == 429 and len(challenges) >= 2
		ref = response.json()["detail"]["appeal_ref"]
		grant = await client.post(
			f"/admin/reviews/{ref}/decisions",
			headers={"X-Harness-Admin-Key": "admin"},
			json={
				"action": "grant",
				"operator": "op",
				"reason": "one request",
				"requests": 1,
				"expires_at": 4102444800,
			},
		)
		assert grant.status_code == 200
		calls = teacher.calls
		for index, challenge in challenges:
			response = await _post(
				client,
				_templated(index),
				headers={
					"Idempotency-Key": f"held-{index}",
					"X-Harness-Proof": f"{challenge['nonce']}:"
					f"{solve_challenge(challenge['prefix'], challenge['difficulty'])}",
				},
			)
			assert response.status_code == 429 and _code(response) == "campaign_throttled"
		assert teacher.calls == calls


@pytest.mark.asyncio
async def test_existing_budgets_stay_independent_of_the_cooldown(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	# The spent rolling budget refuses first; those requests are never observed.
	app, teacher, log = _campaign_app(artifacts, tmp_path, HARNESS_ROLLING_REQUEST_BUDGET=2)
	async with _client(app) as client:
		codes = [(await _post(client, _templated(index), 1000.0 + index)) for index in range(5)]
		assert [r.status_code for r in codes] == [200, 200, 429, 429, 429]
		assert {_code(r) for r in codes[2:]} == {"rolling_budget_exceeded"}
		assert teacher.calls == 2
	decisions = _events(log, "decision")
	assert [e["entitlement"] for e in decisions[2:]] == ["rolling_budget_exceeded"] * 3
	assert all("campaign" not in e for e in decisions[2:])
	assert [e["campaign"]["evaluated_count"] for e in decisions[:2]] == [1, 2]
	assert not _events(log, "campaign_throttle")

	# A cooldown charges no budget and does not refill one.
	other = tmp_path / "other"
	other.mkdir()
	app, teacher, log = _campaign_app(artifacts, other, HARNESS_ROLLING_REQUEST_BUDGET=4)
	async with _client(app) as client:
		responses = await _until_throttled(client)
		assert _code(responses[-1]) == "campaign_throttled"
		served = len(responses) - 1
		assert 0 < served < 4 and teacher.calls == served
		until = responses[-1].json()["campaign"]["throttle_until"]
		later = until + 1000.0
		words = ["alpha", "beta", "gamma", "delta", "epsilon"][: 4 - served + 2]
		results = [
			await _post(client, f"Different question {word}", later + index)
			for index, word in enumerate(words)
		]
		# Throttled requests drew nothing from the budget, and the cooldown did not refill it.
		expected = [200] * (4 - served) + [429] * 2
		assert [r.status_code for r in results] == expected
		assert {_code(r) for r in results[4 - served :]} == {"rolling_budget_exceeded"}
		assert teacher.calls == 4


@pytest.mark.asyncio
async def test_failed_observation_is_not_evidence_and_cannot_lift_a_cooldown(
	artifacts: Artifacts, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	real = campaign_module.observe_campaign

	def flaky(*args: Any, **kwargs: Any) -> dict[str, Any]:
		if "REFUSED" in kwargs["text"]:
			raise CampaignError("rejected for the test")
		return real(*args, **kwargs)

	monkeypatch.setattr(app_module, "observe_campaign", flaky)
	app, teacher, log = _campaign_app(artifacts, tmp_path)
	async with _client(app) as client:
		ok = await _post(client, "Explain tides REFUSED", 1000.0)
		assert ok.status_code == 503 and _code(ok) == "campaign_unavailable"
		assert (await _post(client, _templated(0), 1001.0)).status_code == 200
		responses = await _until_throttled(client, start=1002.0)
		assert responses[-1].status_code == 429
		calls = teacher.calls
		# With a cooldown on record, a request whose observation fails is still refused.
		refused = await _post(client, "Explain tides REFUSED", 1050.0)
		assert refused.status_code == 429 and teacher.calls == calls
	errors = _events(log, "campaign_error")
	assert len(errors) == 2 and all("request_digest" in e for e in errors)
	decisions = _events(log, "decision")
	assert decisions[0]["campaign"]["evaluated_count"] == 1


@pytest.mark.asyncio
async def test_review_evidence_and_grant_apply_to_the_throttle(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	app, teacher, _ = _campaign_app(
		artifacts,
		tmp_path,
		config={"cooldown_seconds": 3600.0, "window_seconds": 86400.0},
		HARNESS_TEST_CLOCK=False,
		HARNESS_REVIEW_ENABLED=True,
		HARNESS_REVIEW_ADMIN_KEY="admin",
	)
	async with _client(app) as client:
		throttled = None
		for index in range(12):
			throttled = await _post(client, _templated(index))
			if throttled.status_code == 429:
				break
		assert throttled is not None and throttled.status_code == 429
		detail = throttled.json()["detail"]
		assert (
			detail["code"] == "campaign_throttled" and detail["limit_code"] == "campaign_throttled"
		)
		assert detail["scope"] == "principal" and detail["reset_at"] is not None
		admin = {"X-Harness-Admin-Key": "admin"}
		shown = await client.get(f"/admin/reviews/{detail['appeal_ref']}", headers=admin)
		assert shown.json()["code"] == "campaign_throttled"
		repeated = await _post(client, _templated(89))
		assert repeated.status_code == 429
		assert repeated.json()["detail"]["appeal_ref"] == detail["appeal_ref"]
		decision = {
			"action": "grant",
			"operator": "op",
			"reason": "verified customer",
			"requests": 1,
			"expires_at": 4102444800,
		}
		granted = await client.post(
			f"/admin/reviews/{detail['appeal_ref']}/decisions", json=decision, headers=admin
		)
		assert granted.status_code == 200
		calls = teacher.calls
		assert (await _post(client, _templated(90))).status_code == 200
		assert teacher.calls == calls + 1
		assert (await _post(client, _templated(91))).status_code == 429


@pytest.mark.asyncio
async def test_concurrent_workers_cannot_skip_the_throttle(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	count = 10
	(tmp_path / "base").mkdir()
	baseline, baseline_teacher, _ = _campaign_app(artifacts, tmp_path / "base")
	async with _client(baseline) as client:
		sequential = [
			(await _post(client, _templated(index), 1000.0)).status_code for index in range(count)
		]
	expected = sequential.count(200)
	assert 0 < expected < count and baseline_teacher.calls == expected

	options: dict[str, Any] = {"HARNESS_STATE_DB": tmp_path / "shared.db"}
	one, one_teacher, one_log = _campaign_app(artifacts, tmp_path, **options)
	two, two_teacher, two_log = _campaign_app(artifacts, tmp_path, **options)
	async with _client(one) as a, _client(two) as b:
		results = await asyncio.gather(
			*(_post(a if index % 2 else b, _templated(index), 1000.0) for index in range(count))
		)
	statuses = [r.status_code for r in results]
	assert statuses.count(200) == expected and statuses.count(429) == count - expected
	assert one_teacher.calls + two_teacher.calls == expected
	counts = sorted(
		e["campaign"]["evaluated_count"]
		for log in (one_log, two_log)
		for e in _events(log, "decision")
	)
	# No observation was lost or counted twice across the two workers.
	assert counts == list(range(1, expected + 1))


@pytest.mark.asyncio
async def test_out_of_order_timestamp_preserves_the_observation(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	"""Concurrent encoder work may reorder arrivals; evidence follows commit order."""
	app, teacher, log = _campaign_app(artifacts, tmp_path)
	async with _client(app) as client:
		assert (await _post(client, _templated(0), 100.0)).status_code == 200
		late = await _post(client, _templated(1), 50.0)
		assert late.status_code == 200 and teacher.calls == 2
	decisions = _events(log, "decision")
	assert decisions[0]["campaign"]["evaluated_count"] == 1
	assert decisions[1]["campaign"]["evaluated_count"] == 2
	assert decisions[1]["campaign"]["components"]["ordering_adjusted"] is True
	assert not _events(log, "campaign_error")


def test_session_encoder_identity_is_checked_before_opening_state(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	with pytest.raises(ValueError, match="campaign encoder representation"):
		_campaign_app(
			artifacts,
			tmp_path,
			mode="observe",
			config={"session": {"encoder_identity": {"model": "different"}}},
		)
	assert not (tmp_path / "state.sqlite3").exists()


@pytest.mark.asyncio
async def test_session_encoding_uses_bound_input_and_durable_retry(
	artifacts: Artifacts, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	encoded: list[str] = []
	real_encode = Encoder.encode

	def record(self: Encoder, texts: list[str]) -> Any:
		encoded.extend(texts)
		return real_encode(self, texts)

	monkeypatch.setattr(Encoder, "encode", record)
	app, teacher, log = _campaign_app(
		artifacts,
		tmp_path,
		mode="observe",
		config={"max_text_chars": 32, "session": {"encoder_identity": {"model": "fake"}}},
	)
	text = "Explain the water cycle and its impact on a coastal ecosystem"
	async with _client(app) as client:
		first = await _post(client, text, 1000.0, headers={"Idempotency-Key": "bounded"})
		retry = await _post(client, text, 1001.0, headers={"Idempotency-Key": "bounded"})
	assert first.status_code == retry.status_code == 200
	assert first.json() == retry.json() and teacher.calls == 1
	assert encoded == [text]
	with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
		state = json.loads(connection.execute("SELECT state FROM campaign").fetchone()[0])
	assert state["count"] == 1
	assert state["rows"][0]["vector"] is None
	assert _events(log, "decision")[0]["campaign"]["qualified"] is False


@pytest.mark.asyncio
async def test_session_without_encoder_contract_does_not_reuse_unbound_base_vectors(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	app, teacher, _ = _campaign_app(artifacts, tmp_path, mode="observe", config={"session": {}})
	async with _client(app) as client:
		assert (await _post(client, "Explain evaporation", 1000.0)).status_code == 200
	assert teacher.calls == 1
	with sqlite3.connect(tmp_path / "state.sqlite3") as connection:
		state = json.loads(connection.execute("SELECT state FROM campaign").fetchone()[0])
	assert state["rows"][0]["vector"] is None


@pytest.mark.asyncio
async def test_session_initializes_encoder_for_allow_policy_without_base_monitor(
	artifacts: Artifacts, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	from harness import encoder as encoder_module
	from test_harness_api import Teacher

	constructed: list[dict[str, Any]] = []
	encoded: list[str] = []

	class SessionEncoder(Encoder):
		def input_complete(self, text: str) -> bool:
			return True

		def identity(self) -> dict[str, Any]:
			return {"model": "fake", "encoded": len(encoded), "truncated": 0}

		def encode(self, texts: list[str]) -> Any:
			encoded.extend(texts)
			return super().encode(texts)

	def factory(**kwargs: Any) -> SessionEncoder:
		constructed.append(kwargs)
		return SessionEncoder()

	monkeypatch.setattr(encoder_module, "BGEEncoder", factory)
	settings = Settings(  # type: ignore[call-arg]
		_env_file=None,
		HARNESS_TRUST_ACTOR_HEADER=True,
		HARNESS_POLICY="allow",
		HARNESS_STATE_DB=tmp_path / "state.sqlite3",
		HARNESS_CAMPAIGN_CONFIG=_config(
			tmp_path, mode="observe", session={"encoder_identity": {"model": "fake"}}
		),
		HARNESS_DECISION_LOG=tmp_path / "decisions.jsonl",
		HARNESS_DECEPTION_KEY_FILE=tmp_path / "key",
	)
	teacher = Teacher()
	app = app_module.create_app(
		settings=settings, teacher=teacher, calibration=artifacts[0], scorer=artifacts[1]
	)
	async with _client(app) as client:
		assert (await _post(client, "Explain evaporation")).status_code == 200
	assert len(constructed) == 1
	assert encoded == ["Explain evaporation"]
	assert teacher.calls == 1


@pytest.mark.asyncio
async def test_session_shared_truncated_prefix_does_not_supply_semantic_evidence(
	artifacts: Artifacts, tmp_path: Path
) -> None:
	app, teacher, log = _campaign_app(
		artifacts,
		tmp_path,
		config={"max_text_chars": 32, "session": {"encoder_identity": {"model": "fake"}}},
	)
	prefix = "Return only the answer for this shared document context. " * 3
	async with _client(app) as client:
		for index, suffix in enumerate(["tides", "volcanoes", "astronomy", "botany"]):
			assert (await _post(client, prefix + suffix, 1000.0 + index)).status_code == 200
	assert teacher.calls == 4
	last = _events(log, "decision")[-1]["campaign"]
	assert last["components"]["semantic_repetition"] == 0
	assert last["score"] == 0


@pytest.mark.asyncio
async def test_session_paraphrase_embeddings_reach_online_score(
	artifacts: Artifacts, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	def paraphrases(self: Encoder, texts: list[str]) -> Any:
		return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1))

	monkeypatch.setattr(Encoder, "encode", paraphrases)
	monkeypatch.setattr(Encoder, "input_complete", lambda self, text: True, raising=False)
	app, teacher, log = _campaign_app(
		artifacts,
		tmp_path,
		config={"session": {"encoder_identity": {"model": "fake"}}},
	)
	prompts = [
		"Which liquid falls from clouds? Answer only",
		"Name the precipitation made of water. Just the answer",
		"Identify what wets the ground during a storm. No explanation",
	]
	async with _client(app) as client:
		responses = [
			await _post(client, prompt, 1000.0 + index) for index, prompt in enumerate(prompts)
		]
	assert [response.status_code for response in responses] == [200, 200, 429]
	assert teacher.calls == 2
	evidence = _events(log, "campaign_throttle")[-1]["campaign"]
	assert evidence["qualified"] is False
	assert evidence["components"]["semantic_repetition"] == 1
	assert evidence["components"]["template_repetition"] == 0


def test_bge_completeness_uses_tokenizer_budget_not_bytes() -> None:
	from harness.encoder import BGEEncoder

	encoder = BGEEncoder.__new__(BGEEncoder)
	encoder._lock = threading.Lock()
	encoder.max_tokens = 512
	seen: list[dict[str, Any]] = []

	def tokenize(text: str, **kwargs: Any) -> dict[str, list[int]]:
		seen.append(kwargs)
		return {"input_ids": [0] * (len(text.split()) + 2)}

	encoder.tokenizer = tokenize
	long_complete = "evaporation precipitation " * 30
	assert len(long_complete.encode()) > 512
	assert encoder.input_complete(long_complete)
	assert not encoder.input_complete("rain " * 511)
	assert seen == [{"add_special_tokens": True, "truncation": False}] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [True, False, None])
async def test_session_semantics_require_explicit_complete_representation(
	artifacts: Artifacts,
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	complete: bool | None,
) -> None:
	monkeypatch.setattr(
		Encoder,
		"encode",
		lambda self, texts: np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)),
	)
	if complete is not None:
		monkeypatch.setattr(Encoder, "input_complete", lambda self, text: complete, raising=False)
	app, teacher, log = _campaign_app(
		artifacts,
		tmp_path,
		mode="observe",
		config={"session": {"encoder_identity": {"model": "fake"}}},
	)
	# Deliberately distinct text with a scripted common semantic direction. All inputs
	# exceed the former byte guard, without needing a model or tokenizer download.
	prompts = [
		"evaporation " * 60 + "Return only the final numeric answer, prefixed by ####.",
		"precipitation " * 60 + "Answer with only the topic name.",
		"condensation " * 60 + "Output only the result.",
	]
	async with _client(app) as client:
		for index, prompt in enumerate(prompts):
			assert len(prompt.encode()) > 512
			assert (await _post(client, prompt, 1000.0 + index)).status_code == 200
	assert teacher.calls == 3
	last = _events(log, "decision")[-1]["campaign"]["components"]
	assert last["semantic_repetition"] == (1 if complete is True else 0)
	if complete is True:
		assert last["semantic_contribution"] > 0
		assert last["semantic_missingness"] is None
	elif complete is None:
		assert last["semantic_missingness"] == "unknown_completeness"
	else:
		assert last["semantic_missingness"] == "truncated_current_input"


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [True, False, None])
async def test_coverage_uses_bounded_encoder_completeness(
	artifacts: Artifacts, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: bool | None
) -> None:
	from harness.campaign_coverage import CoverageConfig, CoverageMonitor
	from harness.logdet import CoverageSpec

	if complete is not None:
		monkeypatch.setattr(Encoder, "input_complete", lambda self, text: complete, raising=False)
	recipe = CoverageMonitor(
		spec=CoverageSpec(seed=1, input_dim=4),
		encoder_identity={"model": "fake"},
		domain="text",
		looks=(1, 2),
	)
	app, teacher, log = _campaign_app(
		artifacts,
		tmp_path,
		mode="observe",
		config={"coverage": CoverageConfig(monitor=recipe).model_dump(mode="json")},
	)
	async with _client(app) as client:
		assert (await _post(client, "Explain evaporation", 1000.0)).status_code == 200
	result = _events(log, "decision")[0]["campaign"]["components"]["coverage"]
	assert result["qualified"] is False
	assert teacher.calls == 1
	assert result["abstention_reason"] == (
		"calibration_missing" if complete else "missing_embedding_prefix"
	)
