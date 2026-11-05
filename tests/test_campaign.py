"""CPU evidence for the experimental campaign rule and durable state."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from harness.campaign import (
	CampaignConfig,
	CampaignError,
	campaign_cooldown,
	load_campaign_config,
	observe_campaign,
	text_identity,
)
from harness.store import MemoryStore, SQLiteStore, StateStore
from pydantic import ValidationError


def observe(
	store: StateStore,
	config: CampaignConfig,
	ident: str,
	text: str = "Solve problem 1",
	now: float = 1,
	**kwargs: Any,
) -> dict[str, Any]:
	with store.transaction() as tx:
		return observe_campaign(
			tx,
			config,
			principal=kwargs.pop("principal", "alice"),
			organization=kwargs.pop("organization", None),
			request_id=ident,
			text=text,
			now=now,
			domain=kwargs.pop("domain", "text"),
			**kwargs,
		)


def config(**kwargs: Any) -> CampaignConfig:
	return CampaignConfig(
		namespace="test", window_size=4, min_samples=2, cooldown_seconds=10.0, **kwargs
	)


def test_configuration(tmp_path: Path) -> None:
	path = tmp_path / "config.json"
	path.write_text(config().model_dump_json())
	assert load_campaign_config(path).digest == config().digest
	for values in (
		{"namespace": " "},
		{"min_samples": 5, "window_size": 4},
		{"threshold": float("nan")},
		{"window_size": True},
		{"extra": 1},
	):
		with pytest.raises(ValidationError):
			CampaignConfig.model_validate({"namespace": "test", **values})
	path.write_text("{}")
	with pytest.raises(CampaignError):
		load_campaign_config(path)


def test_rule_context_and_benign_negatives() -> None:
	cfg = config(mode="observe")
	store = MemoryStore()
	store.bind_campaign(cfg.digest)
	observe(store, cfg, "1")
	hit = observe(store, cfg, "2", "Solve problem 2", 2)
	assert hit["status"] == "alert" and hit["score"] > 0.8
	assert not hit["should_throttle"] and not hit["qualified"]
	assert hit["components"]["semantic_similarity"] is None
	for principal, texts in (
		(
			"diverse",
			["Bake sourdough bread", "Quantum field geometry", "Tax obligations", "Bird migration"],
		),
		(
			"followups",
			[
				"Why did this fail?",
				"Why did this fail?",
				"Continue that explanation",
				"Also clarify the result",
			],
		),
	):
		for index, text in enumerate(texts):
			result = observe(
				store, cfg, str(index), text, float(index), principal=principal, domain=str(index)
			)
			assert result["status"] != "alert"


@pytest.mark.parametrize("durable", [False, True])
def test_retry_rollback_and_expiry(tmp_path: Path, durable: bool) -> None:
	cfg = config(mode="throttle")
	store: StateStore = SQLiteStore(tmp_path / "state.db", "test") if durable else MemoryStore()
	store.bind_campaign(cfg.digest)
	with pytest.raises(RuntimeError), store.transaction() as tx:
		observe_campaign(
			tx,
			cfg,
			principal="alice",
			organization=None,
			request_id="rolled-back",
			text="q",
			now=0,
			domain="text",
		)
		raise RuntimeError("rollback")
	first = observe(store, cfg, "1")
	assert first["evaluated_count"] == 1
	assert observe(store, cfg, "1") == first
	with pytest.raises(CampaignError):
		observe(store, cfg, "1", "different")
	hit = observe(store, cfg, "2", "Solve problem 2", 2)
	assert hit["should_throttle"] and hit["throttle_until"] == 12
	if durable:
		store = SQLiteStore(tmp_path / "state.db", "test")
		store.bind_campaign(cfg.digest)
	active = observe(store, cfg, "3", "Solve problem 3", 3)
	assert active["throttle_until"] == 12
	# After expiry a retry is fresh evidence, so refused requests cannot be replayed unscored.
	retry = observe(store, cfg, "3", "Solve problem 3", 12)
	assert not retry["should_throttle"] and retry["throttle_until"] is None
	assert retry["evaluated_count"] == 4 and retry["retained_count"] == 1
	late_retry = observe(store, cfg, "3", "Solve problem 3", 11)
	assert late_retry["evaluated_count"] == 4 and late_retry["retained_count"] == 1
	with store.transaction() as tx:
		assert campaign_cooldown(tx, cfg, "alice", 11) is None
	adjusted = observe(store, cfg, "out-of-order", "Solve problem 4", now=11)
	assert adjusted["evaluated_count"] == 5 and adjusted["should_throttle"]
	assert adjusted["components"]["ordering_adjusted"] is True
	blocked = observe(store, cfg, "6", "Unrelated other words", 14)
	assert blocked["status"] == "alert" and "cooldown_active" in blocked["reasons"]


def test_exact_repeats_alert_but_only_resampling_can_throttle() -> None:
	cfg = config(mode="throttle")
	store = MemoryStore()
	store.bind_campaign(cfg.digest)
	observe(store, cfg, "1", "ping")
	observe(store, cfg, "2", "ping", 2)
	repeat = observe(store, cfg, "3", "ping", 3)
	assert repeat["status"] == "alert" and not repeat["should_throttle"]
	assert "exact_repeats_not_enforced" in repeat["reasons"]
	observe(store, cfg, "4", "ping", 4, principal="bob", resampling=True)
	assert observe(store, cfg, "5", "ping", 5, principal="bob", resampling=True)["should_throttle"]


def test_questions_after_the_scored_prefix_are_not_one_template() -> None:
	cfg = config(mode="throttle")
	store = MemoryStore()
	store.bind_campaign(cfg.digest)
	context = "shared document words " * 100
	for index, question in enumerate(("Who signed it?", "When did it expire?", "Name the seller")):
		result = observe(
			store,
			cfg,
			str(index),
			context[:64],
			float(index),
			identity=text_identity(context + question),
		)
	assert result["status"] == "observed" and not result["should_throttle"]


def test_window_bounds_and_vectors() -> None:
	cfg = config(window_seconds=3.0)
	store = MemoryStore()
	store.bind_campaign(cfg.digest)
	for index in range(30):
		result = observe(store, cfg, str(index), now=float(index), embedding=[1.0, 0.0])
		assert result["retained_count"] <= 3
	state = next(iter(store.campaigns.values()))
	assert len(state["rows"]) <= cfg.window_size and len(state["receipts"]) <= cfg.window_size
	assert "Solve problem" not in json.dumps(state)
	assert result["components"]["semantic_similarity"] == 1
	with pytest.raises(CampaignError):
		observe(store, cfg, "bad-width", now=30, embedding=[1.0])
	with pytest.raises(CampaignError):
		observe(store, cfg, "nan", now=30, embedding=[float("nan"), 0.0])
	assert observe(store, cfg, "fresh", now=40)["retained_count"] == 1


def test_identity_and_authoritative_cooldown() -> None:
	cfg = config(mode="throttle")
	store = MemoryStore()
	store.bind_campaign(cfg.digest)
	observe(store, cfg, "a", organization="old")
	result = observe(store, cfg, "b", "Solve problem 2", 2, organization="new")
	assert result["evaluated_count"] == 2 and result["should_throttle"]
	with store.transaction() as tx:
		assert campaign_cooldown(tx, cfg, "alice", 3) == 12
		assert campaign_cooldown(tx, cfg, "alice", 12) is None
		assert campaign_cooldown(tx, cfg, "bob", 3) is None
	for vector in ([True, False], ["1", "0"], [0.0, 0.0], [1.0] * 2049):
		with pytest.raises(CampaignError):
			observe(store, cfg, "invalid", now=3, embedding=vector)


def test_sqlite_lazy_binding_and_concurrent_connections(tmp_path: Path) -> None:
	path = tmp_path / "state.db"
	first = SQLiteStore(path, "test")
	with sqlite3.connect(path) as db:
		assert not db.execute("SELECT name FROM sqlite_master WHERE name='campaign'").fetchall()
		assert not db.execute("SELECT value FROM meta WHERE key='campaign_config'").fetchall()
	cfg = config()
	first.bind_campaign(cfg.digest)
	second = SQLiteStore(path, "test")
	second.bind_campaign(cfg.digest)
	with pytest.raises(ValueError, match="another campaign"):
		second.bind_campaign(config(threshold=0.9).digest)

	def submit(index: int) -> dict[str, Any]:
		return observe(first if index % 2 else second, cfg, str(index), now=1)

	with ThreadPoolExecutor(max_workers=4) as pool:
		results = list(pool.map(submit, range(20)))
	assert sorted(r["evaluated_count"] for r in results) == list(range(1, 21))
	assert observe(second, cfg, "other", principal="bob")["evaluated_count"] == 1
	with sqlite3.connect(path) as db:
		states = [json.loads(row[0]) for row in db.execute("SELECT state FROM campaign")]
	assert len(states) == 2
	assert all(len(s["rows"]) <= 4 and len(s["receipts"]) <= 4 for s in states)
