"""Request-path linkage on the in-memory and SQLite stores, and its operator commands."""

from pathlib import Path

from harness.linkage import (
	ClusterBudget,
	LinkageConfig,
	Signal,
	content_signal,
	members,
	network_signal,
	observe,
	sever,
)
from harness.store import MemoryStore

CONFIG = LinkageConfig(namespace="test", max_value_principals=3, max_cluster_size=3)
QUESTION = "A train leaves the station at noon travelling sixty miles per hour"


def _observe(store: MemoryStore, principal: str, *signals: Signal) -> dict[str, object]:
	with store.transaction() as tx:
		return observe(tx, CONFIG, principal, list(signals), 0.0)


def _cluster(store: MemoryStore, principal: str) -> list[str]:
	with store.transaction() as tx:
		return members(tx, principal)


def test_declared_attributes_link_but_behavior_alone_does_not() -> None:
	store = MemoryStore()
	questions = [content_signal(CONFIG, f"{QUESTION} case {i}") for i in range(3)]
	assert all(q is not None for q in questions)
	shared_network = Signal("network", "203.0.113.0/24")
	for name in ("a", "b"):
		for question in questions:
			assert question is not None
			_observe(store, name, question, shared_network)
	# Three repeated questions and a shared network stay below the default threshold.
	assert _cluster(store, "a") == ["a"]
	result = _observe(store, "c", Signal("declared", "payment=card-1"))
	assert result["size"] == 1
	merged = _observe(store, "a", Signal("declared", "payment=card-1"))
	assert merged["size"] == 2 and _cluster(store, "c") == ["a", "c"]
	assert merged["qualified"] is False


def test_hub_values_and_the_size_cap_stop_growth_and_sever_rebuilds() -> None:
	store = MemoryStore()
	reseller = Signal("declared", "payment=reseller")
	for name in ("a", "b", "c", "d"):
		_observe(store, name, reseller)
	# The fourth principal turns the value into a hub instead of joining.
	assert _cluster(store, "a") == ["a", "b", "c"] and _cluster(store, "d") == ["d"]
	_observe(store, "d", Signal("declared", "device=x"))
	_observe(store, "e", Signal("declared", "device=x"))
	_observe(store, "a", Signal("declared", "device=y"))
	capped = _observe(store, "e", Signal("declared", "device=y"))
	assert capped["size"] == 2 and _cluster(store, "a") == ["a", "b", "c"]
	with store.transaction() as tx:
		groups = sever(tx, CONFIG, "a", "b", operator="ops", reason="separate firms", now=1)
		# b still links to a through c, so one sever keeps the cluster whole.
		assert groups == [["a", "b", "c"]]
		groups = sever(tx, CONFIG, "b", "c", operator="ops", reason="separate firms", now=2)
	assert groups == [["a", "c"], ["b"]]
	assert _cluster(store, "b") == ["b"] and _cluster(store, "c") == ["a", "c"]


def test_loopback_and_short_content_never_link() -> None:
	assert network_signal(CONFIG, "127.0.0.1") is None
	assert network_signal(CONFIG, "203.0.113.7") == Signal("network", "203.0.113.0/24")
	assert content_signal(CONFIG, "hello") is None


def test_cluster_budget_counts_every_member_and_ignores_singletons() -> None:
	from harness.store import Reservation

	store = MemoryStore()
	budget = ClusterBudget(100, requests=2)
	with store.transaction() as tx:
		for index, account in enumerate(("a", "b")):
			tx.reserve(Reservation(account, 0, index + 1, f"d{index}", 10, reserved_tokens=1))
		assert not budget.exceeded(tx, "a", 1, 20)
	_observe(store, "a", Signal("declared", "payment=1"))
	_observe(store, "b", Signal("declared", "payment=1"))
	with store.transaction() as tx:
		assert budget.exceeded(tx, "a", 1, 20) and budget.exceeded(tx, "b", 1, 20)
		assert not budget.exceeded(tx, "a", 1, 200)


def test_operators_show_and_sever_durable_links(tmp_path: Path) -> None:
	import json

	from harness.cli import app
	from harness.store import SQLiteStore
	from typer.testing import CliRunner

	config = tmp_path / "linkage.json"
	config.write_text(CONFIG.model_dump_json())
	state = tmp_path / "state.sqlite3"
	store = SQLiteStore(state, "scope")
	store.bind_linkage(CONFIG.digest)
	for name in ("a", "b"):
		with store.transaction() as tx:
			observe(tx, CONFIG, name, [Signal("declared", "payment=1")], 0.0)
	runner = CliRunner()
	common = ["--state-db", str(state), "--config", str(config)]
	shown = runner.invoke(app, ["linkage", "show", "a", *common])
	assert shown.exit_code == 0, shown.output
	assert json.loads(shown.output)["members"] == ["a", "b"]
	severed = runner.invoke(
		app, ["linkage", "sever", "a", "b", "--operator", "ops", "--reason", "audit", *common]
	)
	assert severed.exit_code == 0, severed.output
	assert json.loads(severed.output)["clusters"] == [["a"], ["b"]]
	other = tmp_path / "other.json"
	other.write_text(LinkageConfig(namespace="other").model_dump_json())
	refused = runner.invoke(
		app, ["linkage", "show", "a", "--state-db", str(state), "--config", str(other)]
	)
	assert refused.exit_code != 0


def test_exported_links_feed_the_offline_cluster_analysis() -> None:
	import numpy as np
	from harness.clusters import LinkEvidence, PrincipalHorizon, analyze_clusters
	from harness.linkage import export_links

	store = MemoryStore()
	for name in ("a", "b", "d"):
		_observe(store, name, Signal("declared", "payment=card-1"))
	_observe(store, "c", Signal("network", "203.0.113.0/24"))
	_observe(store, "a", Signal("network", "203.0.113.0/24"))
	with store.transaction() as tx:
		sever(tx, CONFIG, "a", "b", operator="ops", reason="separate customers", now=0.0)
		links = export_links(tx, CONFIG, {"a", "b", "c"})
	# Severed and out-of-scope pairs are dropped; a weak behavioral pair stays as evidence.
	assert [(e["left"], e["right"]) for e in links["edges"]] == [("a", "c")]
	assert links["edges"][0]["strength"] < links["minimum_strength"] == CONFIG.threshold
	accounts = [PrincipalHorizon(p, ["q"], np.zeros((1, 2))) for p in ("a", "b", "c")]
	report = analyze_clusters(
		accounts,
		edges=[LinkEvidence(**edge) for edge in links["edges"]],
		minimum_strength=links["minimum_strength"],
	)
	assert all(edge["included"] is False for edge in report["edges"])
