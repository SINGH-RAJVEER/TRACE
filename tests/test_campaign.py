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
	if config.session is not None and "semantic_complete" not in kwargs:
		kwargs["semantic_complete"] = True if kwargs.get("embedding") is not None else None
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
@pytest.mark.parametrize("session", [False, True])
def test_retry_rollback_and_expiry(tmp_path: Path, durable: bool, session: bool) -> None:
	cfg = session_config(mode="throttle") if session else config(mode="throttle")
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


def session_config(**kwargs: Any) -> CampaignConfig:
	from harness.campaign import SessionConfig

	return config(session=SessionConfig(encoder_identity={"model": "fixture-v1"}), **kwargs)


def test_session_disabled_preserves_v2_digest() -> None:
	# Literal from the pre-extension campaign-v2 configuration at 6dc9b252.
	assert config().digest == "ac6a126e8909c94e52c815c34fb7b962aed7202f2400d8399bea962f9c970ae9"
	assert session_config().digest != config().digest


def test_semantic_paraphrases_and_workflow_controls() -> None:
	cfg = session_config(mode="throttle")
	texts = ["Answer only: calculate seven plus two", "Just the answer: sum nine and four"]
	store = MemoryStore()
	for i, text in enumerate(texts):
		result = observe(store, cfg, str(i), text, float(i), embedding=[1.0, 0.0])
	assert result["should_throttle"] and result["score"] == 1
	assert result["components"]["template_repetition"] == 0
	assert result["components"]["selected_route"] == "semantic"
	assert result["components"]["semantic_contribution"] == 1
	assert result["qualified"] is False
	for principal, prompts in (
		("similarity", ["Calculate seven plus two", "Sum nine and four"]),
		(
			"editing",
			["Rewrite this: answer only seven plus two", "Edit this: answer only nine plus four"],
		),
		("tutoring", ["Explain seven plus two", "Give a hint for nine plus four"]),
		("batch", ["Answer only: label cat", "Answer only: classify ocean"]),
	):
		for i, text in enumerate(prompts):
			# Matched batch has independent semantic topics, other controls share embeddings.
			vector = ([1.0, 0.0] if i == 0 else [0.0, 1.0]) if principal == "batch" else [1.0, 0.0]
			result = observe(
				store, cfg, str(i), text, float(i), principal=principal, embedding=vector
			)
		assert not result["should_throttle"]


def test_session_prefix_does_not_veto_a_window() -> None:
	cfg = session_config(threshold=0.6)
	store = MemoryStore()
	for i, text in enumerate(
		("Answer only: solve 1", "Answer only: solve 2", "Also answer only: solve 3")
	):
		result = observe(store, cfg, str(i), text, float(i), embedding=[1.0, 0.0])
	assert result["status"] == "alert"
	assert result["components"]["contextual_language_share"] == pytest.approx(1 / 3)


def test_session_partial_embeddings_and_long_prefixes() -> None:
	cfg = session_config()
	store = MemoryStore()
	for i, vector in enumerate(([1.0, 0.0], None, [1.0, 0.0])):
		result = observe(
			store,
			cfg,
			str(i),
			f"Answer only: unique wording {chr(97 + i)}",
			float(i),
			embedding=vector,
		)
	assert result["components"]["semantic_pairs_available"] == 1
	assert result["components"]["semantic_pairs_missing"] == 1
	assert result["components"]["semantic_repetition"] == 0.5
	assert result["components"]["semantic_missingness"] == "partial_history"
	assert not result["should_throttle"]
	for i in range(2):
		text = "Answer only: " + "shared document " * 10
		result = observe(
			store,
			cfg,
			str(i),
			text,
			float(i),
			principal="long",
			embedding=[1.0, 0.0],
			identity=text_identity(text + str(i)),
		)
	assert result["components"]["semantic_repetition"] == 0
	assert result["components"]["truncated_count"] == 2


def test_session_repeat_resampling_restart_and_retry(tmp_path: Path) -> None:
	cfg = session_config(mode="throttle")
	path = tmp_path / "session.db"
	store: StateStore = SQLiteStore(path, "test")
	store.bind_campaign(cfg.digest)
	for i in range(3):
		result = observe(store, cfg, str(i), "Answer only: ping", float(i), embedding=[1.0, 0.0])
	assert not result["should_throttle"]
	store = SQLiteStore(path, "test")
	store.bind_campaign(cfg.digest)
	assert observe(store, cfg, "2", "Answer only: ping", 2.0, embedding=[1.0, 0.0]) == result
	for i in range(3):
		result = observe(
			store,
			cfg,
			str(i),
			"Answer only: ping",
			float(i),
			principal="resample",
			embedding=[1.0, 0.0],
			resampling=True,
		)
	assert result["should_throttle"] and result["throttle_until"] == 11
	fresh = observe(
		store,
		cfg,
		"fresh",
		"Answer only: ping",
		11.0,
		principal="resample",
		embedding=[1.0, 0.0],
		resampling=True,
	)
	assert fresh["retained_count"] == 1 and not fresh["should_throttle"]


def test_session_encoder_contract_and_state_binding(tmp_path: Path) -> None:
	from harness.campaign import SessionConfig

	for identity in ({}, {"model": "fixture", "encoded": 2}):
		with pytest.raises(ValidationError, match="stable representation"):
			SessionConfig(encoder_identity=identity)
	cfg = config(session=SessionConfig())
	with pytest.raises(CampaignError, match="bound encoder"):
		observe(MemoryStore(), cfg, "unbound", embedding=[1.0, 0.0])
	store = SQLiteStore(tmp_path / "binding.db", "test")
	store.bind_campaign(config().digest)
	with pytest.raises(ValueError, match="another campaign"):
		store.bind_campaign(cfg.digest)


def test_session_excludes_reported_encoder_token_truncation() -> None:
	from harness.campaign import SessionConfig

	cfg = config(session=SessionConfig(encoder_identity={"model": "fixture", "max_tokens": 32}))
	store = MemoryStore()
	for i, text in enumerate(
		(
			"Answer only: " + "context " * 10 + "first",
			"Just the answer: " + "context " * 10 + "second",
		)
	):
		result = observe(
			store, cfg, str(i), text, float(i), embedding=[1.0, 0.0], semantic_complete=False
		)
	assert result["components"]["semantic_repetition"] == 0
	assert result["components"]["truncated_count"] == 2
	assert result["components"]["semantic_missingness"] == "truncated_current_input"
	assert result["components"]["semantic_contribution"] == 0


@pytest.mark.parametrize("embedding", [None, [1.0, 0.0]])
@pytest.mark.parametrize(
	"template", ["What about chapter {}?", "And for the year {}?", "Why did test {} fail?"]
)
def test_session_preserves_followup_sweep_discount(
	template: str, embedding: list[float] | None
) -> None:
	cfg = CampaignConfig(
		namespace="followups", min_samples=16, mode="throttle", session=session_config().session
	)
	store = MemoryStore()
	for i in range(16):
		result = observe(store, cfg, str(i), template.format(i), float(i), embedding=embedding)
	assert result["score"] == 0
	assert not result["should_throttle"]


@pytest.mark.parametrize(
	"template", ["Explain concept number {} in one paragraph", "Name a word that rhymes with {}"]
)
def test_session_keeps_legacy_lexical_detections(template: str) -> None:
	cfg = session_config(mode="throttle")
	store = MemoryStore()
	for i in range(2):
		result = observe(store, cfg, str(i), template.format(i), float(i))
	assert result["score"] == 1 and result["should_throttle"]
	assert result["components"]["lexical_contribution"] == 1


@pytest.mark.parametrize(
	"instruction",
	[
		"Return only the final numeric answer, prefixed by ####.",
		"Output only the result.",
		"Respond with just the number.",
		"Reply with one word.",
	],
)
def test_session_online_answer_formats_contribute(instruction: str) -> None:
	from harness.session_features import online_answer_only

	assert online_answer_only(instruction)
	cfg = session_config()
	store = MemoryStore()
	for i, question in enumerate(("Calculate seven plus two", "Find the sum of nine and four")):
		result = observe(
			store, cfg, str(i), question + ". " + instruction, float(i), embedding=[1.0, 0.0]
		)
	assert result["components"]["semantic_contribution"] == 1
	assert result["status"] == "alert"


def test_session_actual_task_templates_and_common_words() -> None:
	from distillation.tasks import AG_NEWS_PROMPT, GSM8K_EVAL_SUFFIX, SQUAD_PROMPT
	from harness.session_features import online_answer_only

	for text in (
		AG_NEWS_PROMPT.format(text="A team won a match"),
		SQUAD_PROMPT.format(context="The sky is blue", question="What color is the sky?"),
		"Calculate seven plus two" + GSM8K_EVAL_SUFFIX,
	):
		assert online_answer_only(text)
	cfg = session_config()
	store = MemoryStore()
	for i, question in enumerate(
		("Identify the value that completes this", "Find that missing result")
	):
		result = observe(
			store, cfg, str(i), question + ". Answer only.", float(i), embedding=[1.0, 0.0]
		)
	assert result["score"] == 1
	assert result["components"]["workflow_multiplier"] == 1


def test_session_encoder_free_rephrasing_is_scored_and_bounded() -> None:
	from harness.campaign import SessionConfig, prepare_session_observation

	cfg = config(session=SessionConfig())
	store = MemoryStore()
	for i, suffix in enumerate(("morning", "evening")):
		text = "Answer only: report the result of adding seven and two in the " + suffix
		result = observe(
			store,
			cfg,
			str(i),
			text,
			float(i),
			session_observation=prepare_session_observation(text),
		)
	assert result["components"]["selected_route"] == "rephrasing"
	assert result["components"]["rephrasing_contribution"] == 1
	assert result["components"]["semantic_contribution"] == 0
	assert result["status"] == "alert"
	prepared = prepare_session_observation(" ".join(f"token{i}" for i in range(1000)))
	assert len(prepared.shingles) <= 254
	assert not prepared.lexical_complete
	with pytest.raises(CampaignError, match="differs from scored text"):
		observe(store, cfg, "wrong-preparation", "another text", 3.0, session_observation=prepared)


def test_session_semantic_completeness_is_per_input_and_bound_to_retry() -> None:
	cfg = session_config()
	store = MemoryStore()
	text = "Answer only: " + "some context " * 40
	assert len(text.encode()) > 512
	for i in range(2):
		result = observe(
			store,
			cfg,
			str(i),
			text + str(i),
			float(i),
			embedding=[1.0, 0.0],
			semantic_complete=True,
		)
	assert result["components"]["semantic_contribution"] == 1
	unknown = observe(
		store, cfg, "unknown", "another question", 3.0, embedding=[1.0, 0.0], semantic_complete=None
	)
	assert unknown["components"]["semantic_missingness"] == "unknown_completeness"
	assert unknown["components"]["truncated_count"] == 0
	with pytest.raises(CampaignError, match="different content"):
		observe(
			store,
			cfg,
			"unknown",
			"another question",
			3.0,
			embedding=[1.0, 0.0],
			semantic_complete=False,
		)


def test_session_old_contract_and_duplicate_config_rejected(tmp_path: Path) -> None:
	from harness.campaign import SessionConfig

	with pytest.raises(ValidationError):
		SessionConfig(version="semantic-session-v1")  # type: ignore[arg-type]
	path = tmp_path / "duplicate.json"
	path.write_text(
		'{"namespace":"test","session":{"version":"semantic-session-v2"},"session":null}'
	)
	with pytest.raises(CampaignError, match="duplicate"):
		load_campaign_config(path)
