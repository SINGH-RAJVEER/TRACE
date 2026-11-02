"""Development-only separation study between attack and benign request windows.

The question is whether attack windows remain distinguishable from benign account windows once
task-domain mix, topic mix, window size and client timing are balanced. Each window of
`window_size` consecutive requests is one sample. Train and test splits are source-disjoint:
`split_windows` keeps windows connected by a shared content source, or by a benign account,
together. An attack schedule is one principal, so its windows are connected only through
content; its prompts are first divided into two pools by source so that each pool can form a
separate component.

Balancing builds new attack windows from a split's own attack prompts. Each copies one benign
window from that split: its per-position domain, per-position (domain, topic cell), its repeat
structure (distinct-prompt count) and its timestamps, as the control requires. Prompts are drawn
in pool order, so a topic-sorted attack pool yields topic-concentrated matched windows.

Source corpus is not a matching factor: an attack drawn from corpora absent from the benign
roles cannot be balanced on it, so the report gives each attack's (domain, corpus) overlap with
the benign roles and marks a low overlap as inadequate rather than building zero windows.

Results are development evidence only. They never feed calibration, thresholds or enforcement.
"""

import hashlib
import json
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from harness.encoder import Encoder
from harness.mmd import Array, WindowScorer, median_bandwidth
from harness.separation_groups import connected_source_groups
from harness.separation_split import split_windows
from harness.separation_stats import (
	auc,
	clustered_auc_interval,
	fit_logistic,
	fit_mlp,
	learned_mmd,
	permutation_p,
)
from harness.serialization import detector_text, gateway_turn
from harness.shadow import assign, kmeans, request_domain

CONTROLS: dict[str, tuple[str, ...]] = {
	"none": (),
	"domain": ("domain",),
	"topic": ("domain", "topic"),
	"size": ("size",),
	"client": ("client",),
	"all": ("domain", "topic", "size", "client"),
}
SEMANTIC_TESTS = ("c2st_logistic", "c2st_mlp", "learned_mmd")
SURVIVES = "separation survives balancing"
DISAPPEARS = "separation disappears after balancing"
Order = Literal["schedule", "topic"]


class StudyConfig(BaseModel):
	window_size: int = Field(default=100, ge=4)
	seed: int = 42
	topic_cells: int = Field(default=8, ge=2)
	permutations: int = Field(default=999, ge=19)
	bootstrap: int = Field(default=2000, ge=100)
	alpha: float = Field(default=0.05, gt=0, lt=1)
	minimum_windows: int = Field(default=20, ge=2)
	source_key: Literal["group_id", "source_id", "content_sha256"] = "group_id"


@dataclass(frozen=True)
class Prompt:
	key: str
	domain: str
	source: str
	length: int
	time: float
	row: int
	topic: int = -1


@dataclass(frozen=True)
class Window:
	label: int
	account: str
	prompts: tuple[Prompt, ...]

	def sources(self) -> list[str]:
		"""Content sources, plus the account for benign windows."""
		found = sorted({prompt.source for prompt in self.prompts})
		return found + ([f"account:{self.account}"] if self.label == 0 else [])


@dataclass(frozen=True)
class Record:
	account: str
	domain: str
	source: str
	length: int
	time: float
	text: str
	corpus: str = ""


def _sha(text: str) -> str:
	return hashlib.sha256(text.encode()).hexdigest()


def read_records(path: Path, source_key: str) -> list[Record]:
	"""Gateway schedule lines, campaign ledger lines, or gateway BenignRequest lines.

	Ledger rows have no account or time, so they become one principal in file order. Rows
	without source metadata use the detector text's hash as their only source.
	"""
	records: list[Record] = []
	with path.open(encoding="utf-8") as lines:
		for index, line in enumerate(lines):
			if not line.strip():
				continue
			value = json.loads(line)
			prompt = value.get("prompt")
			if isinstance(prompt, dict):
				item, account, time = prompt, value["account"], value.get("time")
			elif isinstance(prompt, list):
				item, account, time = value, "ledger", None
			else:
				text = value["text"]
				records.append(
					Record(
						account=value["account"],
						domain=request_domain("gateway", text),
						source="text:" + _sha(detector_text("gateway", text)),
						length=len(json.loads(text)["turn"]),
						time=float(index if value.get("time") is None else value["time"]),
						text=text,
					)
				)
				continue
			domain = item["task"]
			turn = next(m["content"] for m in reversed(item["prompt"]) if m["role"] == "user")
			text = gateway_turn(domain, turn)
			source = item.get("source") or {}
			identifier = source.get(source_key)
			records.append(
				Record(
					account=account,
					domain=domain,
					source=f"{source.get('corpus', '')}:{identifier}"
					if identifier
					else "text:" + _sha(detector_text("gateway", text)),
					length=len(turn),
					time=float(index if time is None else time),
					text=text,
					corpus=str(source.get("corpus", "")),
				)
			)
	if not records:
		raise ValueError(f"{path} has no requests")
	return records


class Embedded:
	"""Embeds each distinct detector text once and hands out row indices."""

	def __init__(self, encoder: Encoder, groups: Sequence[Sequence[Record]]) -> None:
		texts: dict[str, int] = {}
		for records in groups:
			for record in records:
				texts.setdefault(detector_text("gateway", record.text), len(texts))
		self.rows = texts
		self.vectors = encoder.encode(list(texts))

	def prompts(self, records: Sequence[Record]) -> list[Prompt]:
		result = []
		for record in records:
			text = detector_text("gateway", record.text)
			result.append(
				Prompt(
					key=_sha(text),
					domain=record.domain,
					source=record.source,
					length=record.length,
					time=record.time,
					row=self.rows[text],
				)
			)
		return result


def chunk(label: int, account: str, prompts: Sequence[Prompt], size: int) -> list[Window]:
	return [
		Window(label, account, tuple(prompts[start : start + size]))
		for start in range(0, len(prompts) - size + 1, size)
	]


def benign_windows(prompts: Sequence[Prompt], accounts: Sequence[str], size: int) -> list[Window]:
	"""Consecutive full windows within each benign account, in file order."""
	by_account: dict[str, list[Prompt]] = {}
	for prompt, account in zip(prompts, accounts, strict=True):
		by_account.setdefault(account, []).append(prompt)
	return [w for account, rows in by_account.items() for w in chunk(0, account, rows, size)]


def attack_pools(prompts: Sequence[Prompt], seed: int) -> list[list[Prompt]]:
	"""Two source-disjoint pools of roughly equal size, each in schedule order."""
	counts = Counter(prompt.source for prompt in prompts)
	if len(counts) < 2:
		raise ValueError("the attack needs at least two content sources to form two pools")
	rng = np.random.default_rng(seed)
	sources = list(counts)
	pool: dict[str, int] = {}
	total = 0
	for index in rng.permutation(len(sources)):
		source = sources[int(index)]
		pool[source] = 0 if total < len(prompts) / 2 else 1
		total += counts[source]
	return [[p for p in prompts if pool[p.source] == k] for k in (0, 1)]


def topic_sorted(prompts: Sequence[Prompt], vectors: Array) -> list[Prompt]:
	"""The `evidence-check --reorder topic` key; each prompt takes the original slot's time."""
	ordered = sorted(prompts, key=lambda p: (p.domain, str(p.topic), tuple(vectors[p.row])))
	return [replace(p, time=slot.time) for p, slot in zip(ordered, prompts, strict=True)]


def matched(
	benign: Sequence[Window],
	attack: Sequence[Window],
	factors: tuple[str, ...],
	rng: np.random.Generator,
) -> tuple[list[Window], list[Window]]:
	"""Matched benign windows and attack windows that copy their composition.

	Each benign window, in random order, receives one attack window built from unused prompts
	of the same split, taken in pool order. A benign window whose requirements the remaining
	pool cannot meet is dropped. Without the client factor, attack windows keep the time slots
	of the attack's own windows.
	"""
	if not factors:
		return list(benign), list(attack)
	order = [benign[int(i)] for i in rng.permutation(len(benign))]
	if factors == ("client",):
		pairs = list(zip(order, attack, strict=False))
		return [b for b, _ in pairs], [
			replace(
				a,
				prompts=tuple(
					replace(p, time=s.time) for p, s in zip(a.prompts, b.prompts, strict=True)
				),
			)
			for b, a in pairs
		]
	pool = [prompt for window in attack for prompt in window.prompts]

	def requirement(prompt: Prompt) -> tuple[Any, ...]:
		return (
			prompt.domain if "domain" in factors else None,
			prompt.topic if "topic" in factors else None,
		)

	queues: dict[tuple[Any, ...], deque[Prompt]] = {}
	for prompt in pool:
		queues.setdefault(requirement(prompt), deque()).append(prompt)
	kept: list[Window] = []
	built: list[Window] = []
	for window in order:
		slots: dict[str, int] = {}
		positions = []
		for index, prompt in enumerate(window.prompts):
			slot = slots.setdefault(prompt.key, index) if "size" in factors else index
			positions.append(slot)
		needed = Counter(requirement(window.prompts[slot]) for slot in set(positions))
		if any(len(queues.get(key, ())) < count for key, count in needed.items()):
			continue
		drawn = {
			slot: queues[requirement(window.prompts[slot])].popleft()
			for slot in sorted(set(positions))
		}
		slots_from = (
			window.prompts if "client" in factors else attack[len(built) % len(attack)].prompts
		)
		built.append(
			Window(
				1,
				attack[0].account,
				tuple(
					replace(drawn[slot], time=time.time)
					for slot, time in zip(positions, slots_from, strict=True)
				),
			)
		)
		kept.append(window)
	return kept, built


def semantic_features(windows: Sequence[Window], vectors: Array) -> Array:
	"""Mean embedding over positions and within-window dispersion."""
	rows = []
	for window in windows:
		block = vectors[[p.row for p in window.prompts]]
		mean = block.mean(0)
		rows.append(np.append(mean, ((block - mean) ** 2).sum(1).mean()))
	return np.asarray(rows, dtype=np.float64)


VOLUME_FEATURES = (
	"distinct_share",
	"mean_log_length",
	"std_log_length",
	"domain_count",
	"domain_entropy",
	"mean_log_gap",
	"std_log_gap",
	"log_duration",
)


def volume_features(windows: Sequence[Window]) -> Array:
	"""Volume and session features only; no embedding, client label or account identifier."""
	rows = []
	for window in windows:
		prompts = window.prompts
		lengths = np.log1p([p.length for p in prompts])
		shares = np.array(list(Counter(p.domain for p in prompts).values())) / len(prompts)
		times = np.array([p.time for p in prompts])
		gaps = np.log1p(np.maximum(np.diff(times), 0.0))
		rows.append(
			[
				len({p.key for p in prompts}) / len(prompts),
				lengths.mean(),
				lengths.std(),
				len(shares),
				float(-(shares * np.log(shares)).sum()),
				gaps.mean(),
				gaps.std(),
				np.log1p(max(times.max() - times.min(), 0.0)),
			]
		)
	return np.asarray(rows, dtype=np.float64)


RIDGES = (0.1, 1.0, 10.0, 100.0, 1000.0)


def selected_logistic(
	features: Array, labels: Array, windows: Sequence[Window], seed: int
) -> tuple[Callable[[Array], Array], dict[str, Any]]:
	"""Choose the ridge on an inner source-disjoint split of the training windows only.

	When the training windows cannot be split again, the default ridge of 1 is used.
	"""
	try:
		inner_train, inner_test = split_windows(
			[w.sources() for w in windows], [w.label for w in windows], seed=seed + 1
		)
	except ValueError:
		return fit_logistic(features, labels), {"ridge": 1.0, "ridge_selection": "default"}

	def inner(ridge: float) -> float:
		model = fit_logistic(features[inner_train], labels[inner_train], ridge)
		return auc(model(features[inner_test]), labels[inner_test])

	ridge = max(RIDGES, key=inner)
	return fit_logistic(features, labels, ridge), {"ridge": ridge, "ridge_selection": "inner split"}


def _held_out(
	train_scores: Array,
	train_labels: Array,
	scores: Array,
	labels: Array,
	groups: Sequence[int],
	config: StudyConfig,
	orient: bool = False,
) -> dict[str, Any]:
	sign = 1.0
	if orient and auc(train_scores, train_labels) < 0.5:
		sign = -1.0
	scores = sign * scores
	return {
		"auc": auc(scores, labels),
		"ci": clustered_auc_interval(scores, labels, groups, config.bootstrap, config.seed),
		"p_value": permutation_p(
			lambda y: auc(scores, y), labels, config.permutations, config.seed
		),
		"orientation": "lower scores are attack" if sign < 0 else "higher scores are attack",
	}


def run_tests(
	train: Sequence[Window],
	test: Sequence[Window],
	vectors: Array,
	baseline: Callable[[Window], float],
	config: StudyConfig,
) -> dict[str, Any]:
	y_train = np.array([w.label for w in train])
	y_test = np.array([w.label for w in test])
	groups = connected_source_groups([w.sources() for w in test])
	group_counts = {
		str(label): len({g for g, w in zip(groups, test, strict=True) if w.label == label})
		for label in (0, 1)
	}
	semantic_train, semantic_test = (
		semantic_features(train, vectors),
		semantic_features(test, vectors),
	)
	volume_train, volume_test = volume_features(train), volume_features(test)
	logistic, logistic_ridge = selected_logistic(semantic_train, y_train, train, config.seed)
	mlp = fit_mlp(semantic_train, y_train, config.seed)
	volume, volume_ridge = selected_logistic(volume_train, y_train, train, config.seed)
	mmd = learned_mmd(
		semantic_train, y_train, semantic_test, y_test, config.permutations, config.seed
	)
	witness = mmd.pop("witness")
	tests = {
		"c2st_logistic": {
			**_held_out(
				logistic(semantic_train), y_train, logistic(semantic_test), y_test, groups, config
			),
			**logistic_ridge,
		},
		"c2st_mlp": _held_out(
			mlp(semantic_train), y_train, mlp(semantic_test), y_test, groups, config
		),
		"learned_mmd": {
			**_held_out(witness, y_test, witness, y_test, groups, config),
			"auc_source": "train-split witness evaluated on test windows",
			**mmd,
		},
		"pooled_mmd_baseline": _held_out(
			np.array([baseline(w) for w in train]),
			y_train,
			np.array([baseline(w) for w in test]),
			y_test,
			groups,
			config,
			orient=True,
		),
		"volume_session": {
			**_held_out(volume(volume_train), y_train, volume(volume_test), y_test, groups, config),
			**volume_ridge,
			"features": list(VOLUME_FEATURES),
		},
	}
	# The MMD test's own p-value decides; its witness AUC p-value is reported alongside.
	tests["learned_mmd"]["auc_p_value"] = tests["learned_mmd"]["p_value"]
	tests["learned_mmd"]["p_value"] = mmd["mmd_p_value"]
	del tests["learned_mmd"]["orientation"]
	return {"test_source_groups": group_counts, "tests": tests}


def _counts(windows: Sequence[Window]) -> dict[str, int]:
	found = Counter(w.label for w in windows)
	return {"0": found[0], "1": found[1]}


def compare(
	name: str,
	benign: Sequence[Window],
	pools: Sequence[Sequence[Prompt]],
	vectors: Array,
	scorer_factory: Callable[[Sequence[Window]], Callable[[Window], float]],
	config: StudyConfig,
	corpus_overlap: float | None = None,
) -> dict[str, Any]:
	size = config.window_size
	attack_account = f"attack:{name}"
	attack = [w for pool in pools for w in chunk(1, attack_account, pool, size)]
	windows = [*benign, *attack]
	train_index, test_index = split_windows(
		[w.sources() for w in windows],
		[w.label for w in windows],
		seed=config.seed,
		minimum_per_class=2,
	)
	splits = {
		"train": [windows[i] for i in train_index],
		"test": [windows[i] for i in test_index],
	}
	baseline = scorer_factory([w for w in splits["train"] if w.label == 0])
	controls: dict[str, Any] = {}
	for control, factors in CONTROLS.items():
		rng = np.random.default_rng(config.seed)
		built: dict[str, list[Window]] = {}
		for part, members in splits.items():
			kept, attack_windows = matched(
				[w for w in members if w.label == 0],
				[w for w in members if w.label == 1],
				factors,
				rng,
			)
			built[part] = [*kept, *attack_windows]
		train_sources = {s for w in built["train"] for s in w.sources()}
		if any(s in train_sources for w in built["test"] for s in w.sources()):
			raise AssertionError("balanced windows broke source disjointness")
		entry: dict[str, Any] = {
			"factors": list(factors),
			"windows": {part: _counts(found) for part, found in built.items()},
		}
		if min(min(_counts(found).values()) for found in built.values()) < 2:
			entry["skipped"] = "fewer than two windows of a class in a split after matching"
		else:
			entry.update(run_tests(built["train"], built["test"], vectors, baseline, config))
		controls[control] = entry
	reference = controls["none"].get("tests", {})
	for entry in controls.values():
		for test_name, result in entry.get("tests", {}).items():
			base = reference.get(test_name, {}).get("auc")
			result["retained"] = (
				(result["auc"] - 0.5) / (base - 0.5) if base is not None and base > 0.55 else None
			)
	return {
		"name": name,
		"attack_requests": sum(len(pool) for pool in pools),
		"split_windows": {part: _counts(found) for part, found in splits.items()},
		"corpus_overlap": corpus_overlap,
		"controls": controls,
		"decision": decide(controls["all"], config, corpus_overlap),
	}


def decide(
	entry: dict[str, Any], config: StudyConfig, corpus_overlap: float | None = None
) -> dict[str, Any]:
	"""Survival needs a Bonferroni-significant semantic test whose clustered CI excludes 0.5.

	An undefined clustered interval does not veto, but marks the result inadequate, as does a
	matched test split below `minimum_windows` per class. Inadequate disappearance is low
	power, not a stop signal by itself. Less than half of the attack prompts sharing a benign
	(domain, corpus) pair also marks it inadequate: survival may then be corpus provenance.
	"""
	threshold = config.alpha / len(SEMANTIC_TESTS)
	tests = entry.get("tests")
	if tests is None:
		return {
			"line": DISAPPEARS,
			"survives": False,
			"adequate": False,
			"reason": entry.get("skipped"),
			"bonferroni_alpha": threshold,
		}
	surviving = [
		name
		for name in SEMANTIC_TESTS
		if tests[name]["p_value"] < threshold
		and (tests[name]["ci"] is None or tests[name]["ci"][0] > 0.5)
	]
	smallest = min(entry["windows"]["test"].values())
	undefined = [name for name in SEMANTIC_TESTS if tests[name]["ci"] is None]
	reasons = []
	if corpus_overlap is not None and corpus_overlap < 0.5:
		reasons.append(
			f"corpus not balanceable: {corpus_overlap:.0%} of attack prompts share a benign "
			"(domain, corpus)"
		)
	if smallest < config.minimum_windows:
		reasons.append(f"{smallest} matched test windows in the smaller class")
	if undefined:
		reasons.append("clustered CI undefined: fewer than two source groups in a test class")
	return {
		"line": SURVIVES if surviving else DISAPPEARS,
		"survives": bool(surviving),
		"surviving_tests": surviving,
		"adequate": not reasons,
		"reason": "; ".join(reasons) or None,
		"bonferroni_alpha": threshold,
	}


def file_sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def run_study(
	*,
	benign: Sequence[Path],
	attacks: Sequence[tuple[str, Path]],
	encoder: Encoder,
	config: StudyConfig,
	orders: Sequence[Order] = ("schedule", "topic"),
	scorer: tuple[WindowScorer, bool] | None = None,
	topic_fitting: Path | None = None,
) -> dict[str, Any]:
	"""The JSON report. `scorer` is the calibrated pooled MMD and its repeat-collapse flag."""
	benign_records = [r for path in benign for r in read_records(path, config.source_key)]
	attack_records = {name: read_records(path, config.source_key) for name, path in attacks}
	fitting_records = read_records(topic_fitting, config.source_key) if topic_fitting else []
	embedded = Embedded(encoder, [benign_records, *attack_records.values(), fitting_records])
	vectors = embedded.vectors
	fitting = (
		embedded.prompts(fitting_records) if fitting_records else embedded.prompts(benign_records)
	)
	centroids = kmeans(vectors[[p.row for p in fitting]], config.topic_cells, 50, config.seed)

	def with_topics(prompts: list[Prompt]) -> list[Prompt]:
		cells = assign(vectors[[p.row for p in prompts]], centroids)
		return [replace(p, topic=int(c)) for p, c in zip(prompts, cells, strict=True)]

	benign_prompts = with_topics(embedded.prompts(benign_records))
	windows = benign_windows(
		benign_prompts, [r.account for r in benign_records], config.window_size
	)

	def scorer_factory(train_benign: Sequence[Window]) -> Callable[[Window], float]:
		if scorer is not None:
			fixed, distinct = scorer
			if fixed.window_size != config.window_size:
				raise ValueError("window size must equal the calibration's window size")
			return lambda w: fixed.score(
				vectors[[p.row for p in w.prompts]],
				[p.key for p in w.prompts] if distinct else None,
			)
		rng = np.random.default_rng(config.seed)
		picks = np.linspace(0, len(train_benign) - 1, min(20, len(train_benign))).astype(int)
		references = np.stack(
			[vectors[[p.row for p in train_benign[int(i)].prompts]] for i in picks]
		)
		rows = sorted({p.row for w in train_benign for p in w.prompts})
		sample = rng.choice(rows, size=min(2000, len(rows)), replace=False)
		fallback = WindowScorer(references, median_bandwidth(vectors[sample]))
		return lambda w: fallback.score(vectors[[p.row for p in w.prompts]])

	benign_corpora = {(r.domain, r.corpus) for r in benign_records}
	comparisons = []
	for name, records in attack_records.items():
		overlap = sum((r.domain, r.corpus) in benign_corpora for r in records) / len(records)
		pools = attack_pools(with_topics(embedded.prompts(records)), config.seed)
		for order in orders:
			ordered = [topic_sorted(pool, vectors) for pool in pools] if order == "topic" else pools
			label = name if order == "schedule" else f"{name}/topic-sorted"
			result = compare(label, windows, ordered, vectors, scorer_factory, config, overlap)
			comparisons.append({**result, "attack": name, "order": order})
	# An attacker reorders prompts at no query cost, so an attack's verdict is its worst order.
	verdicts = {
		name: all(c["decision"]["survives"] for c in comparisons if c["attack"] == name)
		for name in attack_records
	}
	survives = any(verdicts.values())
	return {
		"version": "separation-v1",
		"development_only": True,
		"used_for_calibration_or_enforcement": False,
		"config": config.model_dump(mode="json"),
		"encoder": encoder.identity(),
		"baseline": "calibrated pooled MMD" if scorer else "pooled MMD on train benign windows",
		"topic_fitting": str(topic_fitting) if topic_fitting else "benign inputs",
		"inputs": {
			"benign": [{"path": str(p), "sha256": file_sha256(p)} for p in benign],
			"attacks": [{"name": n, "path": str(p), "sha256": file_sha256(p)} for n, p in attacks],
		},
		"benign_requests": len(benign_records),
		"benign_windows": len(windows),
		"comparisons": comparisons,
		"attack_decisions": {name: SURVIVES if v else DISAPPEARS for name, v in verdicts.items()},
		"decision": SURVIVES if survives else DISAPPEARS,
	}


def _number(value: float | None, digits: int = 3) -> str:
	return "n/a" if value is None else f"{value:.{digits}f}"


def summary_markdown(report: dict[str, Any]) -> str:
	"""A short human summary: the decision line, then per-test held-out numbers."""
	lines = [
		"# Separation study (development evidence only)",
		"",
		f"Decision: {report['decision']}",
		"",
		"An attack survives only if every tested order survives the `all` control, since an "
		"attacker can reorder prompts without extra queries. The study survives if any attack "
		"does.",
		"",
		*(f"- {name}: {line}" for name, line in report["attack_decisions"].items()),
		"",
		"Driven by: "
		+ (
			", ".join(n for n, v in report["attack_decisions"].items() if v == SURVIVES)
			or "no attack survives"
		)
		+ ".",
		"",
		"Held-out AUC on source-disjoint test windows, 95% CI clustered by source group, and "
		"permutation p (test labels; learned MMD uses its MMD statistic). These results do not "
		"feed calibration or enforcement. Window size "
		f"{report['config']['window_size']}, source key `{report['config']['source_key']}`, "
		f"{report['benign_windows']} benign windows; the attack side is one principal per "
		"schedule regardless of its window count.",
	]
	for comparison in report["comparisons"]:
		decision = comparison["decision"]
		note = "" if decision["adequate"] else f" (inadequate: {decision['reason']})"
		overlap = comparison.get("corpus_overlap")
		lines += [
			"",
			f"## {comparison['name']}",
			"",
			f"Decision: {decision['line']}{note}",
			"",
			f"Attack prompts sharing a benign (domain, corpus): {_number(overlap, 2)}. "
			"Retained is the share of the unbalanced AUC above 0.5 that remains.",
			"",
			"| control | test windows (benign/attack) | test groups | test | AUC | CI | p "
			"| retained |",
			"|---|---|---|---|---|---|---|---|",
		]
		for control, entry in comparison["controls"].items():
			counts = entry["windows"]["test"]
			if "tests" not in entry:
				lines.append(
					f"| {control} | {counts['0']}/{counts['1']} | n/a | skipped | n/a | n/a | n/a "
					"| n/a |"
				)
				continue
			groups = entry["test_source_groups"]
			for test, result in entry["tests"].items():
				ci = result["ci"]
				interval = "undefined" if ci is None else f"{ci[0]:.3f}-{ci[1]:.3f}"
				lines.append(
					f"| {control} | {counts['0']}/{counts['1']} | {groups['0']}/{groups['1']} "
					f"| {test} | {_number(result['auc'])} | {interval} "
					f"| {_number(result['p_value'], 4)} | {_number(result['retained'], 2)} |"
				)
	return "\n".join(lines) + "\n"
