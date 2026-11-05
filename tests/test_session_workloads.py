"""Fresh shadow references disclose their simulation and independent account units."""

from collections import defaultdict

import pytest
from distillation.session_workloads import SessionModel, session_clients
from workload_fixtures import prompt_record

CAPS = {"coding": 512, "gsm8k": 256}


def pools(count: int):
	return {
		d: [prompt_record(d, "score_calibration", i)[0] for i in range(count)]
		for d in ("coding", "gsm8k")
	}


def test_session_reference_counts_independence_and_reproducibility() -> None:
	kwargs = dict(
		pools=pools(500), caps=CAPS, role="score_calibration", window_size=4, horizon_size=4
	)
	rows, manifest = session_clients(**kwargs)
	assert (rows, manifest) == session_clients(**kwargs)
	assert manifest["specialist_windows_per_domain"] == {"coding": 20, "gsm8k": 20}
	assert manifest["generalist_horizons"] == 150
	assert len(rows) == (40 + 150) * 4
	owners: dict[tuple[str, str | None], str] = {}
	by_account = defaultdict(list)
	for row in rows:
		key = row.prompt.source.corpus, row.prompt.source.group_id
		assert owners.setdefault(key, row.account) == row.account
		by_account[row.account].append(row)
	for account, requests in by_account.items():
		assert len(requests) == 4
		assert all(b.time > a.time for a, b in zip(requests, requests[1:], strict=False))
		if "generalist" in account:
			assert {r.domain for r in requests} == {"coding", "gsm8k"}
	assert "synthetic" in manifest["session_model"]["limitation"]


def test_sessions_include_declared_behaviors_and_irregular_gaps() -> None:
	rows, manifest = session_clients(
		pools=pools(300),
		caps=CAPS,
		role="score_calibration",
		specialist_windows=1,
		generalist_horizons=1,
	)
	assert all(
		manifest["session_model"]["events"][kind] > 0
		for kind in ("fresh", "retry", "rephrase", "follow_up")
	)
	coding = [r for r in rows if "-coding-" in r.account]
	assert len({round(b.time - a.time, 2) for a, b in zip(coding, coding[1:], strict=False)}) > 10


def test_sessions_fail_closed_on_source_capacity() -> None:
	with pytest.raises(ValueError, match="source-disjoint.*capacity"):
		session_clients(pools=pools(2), caps=CAPS, role="score_calibration")


def test_session_model_rejects_invalid_probabilities() -> None:
	with pytest.raises(ValueError, match="retain fresh"):
		SessionModel(retry_share=0.8, follow_up_share=0.2).validate()
