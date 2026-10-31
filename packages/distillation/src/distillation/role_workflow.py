"""Frozen source-family roles and validation for the opt-in profile workload."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from distillation.benchmarks.models import BenchmarkItem, save_json
from distillation.io import read_jsonl, write_jsonl
from distillation.profiles import PROFILE_WORKLOAD_VERSION, SCORED_PROFILES, topic_coverage
from distillation.schemas import GatewayRequest, PromptRecord

ROLES = (
	"representation_fit",
	"reference_fit",
	"score_calibration",
	"policy_development",
	"locked_test",
)


class SourceItem(BaseModel):
	"""An audited source record; family and evidence must come from the original source."""

	model_config = ConfigDict(extra="forbid")
	prompt: PromptRecord
	reference: BenchmarkItem
	family: str = Field(min_length=1)
	family_kind: str = Field(min_length=1)
	provenance: str = Field(min_length=1)


def digest(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_source(item: SourceItem) -> None:
	kinds = {
		"squad": "document_title",
		"coding": "problem_lineage",
		"gsm8k": "problem_family",
		"ag_news": "publisher_topic_period",
	}
	if item.prompt.task not in kinds or item.family_kind != kinds[item.prompt.task]:
		raise ValueError(f"unknown or incompatible family provenance for {item.prompt.example_id}")
	if item.family.strip().lower() in {
		"",
		"unknown",
		"none",
	} or item.provenance.strip().lower() in {"", "unknown", "none"}:
		raise ValueError(f"unknown provenance for {item.prompt.example_id}")
	if not item.reference.category.strip():
		raise ValueError(f"missing topic for {item.prompt.example_id}")


def _shingles(item: SourceItem) -> set[str]:
	text = " ".join(m.content for m in item.prompt.prompt if m.role == "user")
	words = re.findall(r"\w+", text.casefold())
	return {" ".join(words[i : i + 5]) for i in range(max(1, len(words) - 4))}


def screen(items: list[SourceItem], assignments: dict[str, str]) -> list[str]:
	"""Flag cross-role exact and high-Jaccard five-word shingles; not proof of no leakage."""
	shingle_sets = [_shingles(item) for item in items]
	frequencies = Counter(shingle for words in shingle_sets for shingle in words)
	inverted: dict[str, list[int]] = defaultdict(list)
	exact: dict[str, int] = {}
	problems = []
	for index, (item, words) in enumerate(zip(items, shingle_sets, strict=True)):
		text_hash = hashlib.sha256(" ".join(sorted(words)).encode()).hexdigest()
		candidates: set[int] = set()
		if text_hash in exact:
			candidates.add(exact[text_hash])
		else:
			exact[text_hash] = index
		for word in words:
			if frequencies[word] <= 100:
				candidates.update(inverted[word])
		for other_index in candidates:
			other = items[other_index]
			if assignments[item.prompt.example_id] == assignments[other.prompt.example_id]:
				continue
			other_words = shingle_sets[other_index]
			union = len(words | other_words)
			if union and len(words & other_words) / union >= 0.8:
				problems.append(
					f"near duplicate: {other.prompt.example_id} / {item.prompt.example_id}"
				)
		for word in words:
			if frequencies[word] <= 100:
				inverted[word].append(index)
	return problems


def build_roles(source: Path, output: Path, seed: int = 42) -> dict[str, Any]:
	items = read_jsonl(source, SourceItem)
	if not items:
		raise ValueError("source is empty")
	groups: dict[tuple[str, str], list[SourceItem]] = defaultdict(list)
	ids: set[str] = set()
	for item in items:
		_validate_source(item)
		if item.prompt.example_id in ids:
			raise ValueError(f"duplicate example id: {item.prompt.example_id}")
		ids.add(item.prompt.example_id)
		groups[item.prompt.task, item.family].append(item)
	# Assign complete families within each domain/topic stratum, rotating the starting role.
	strata: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
	for key, members in groups.items():
		topics = Counter(item.reference.category for item in members)
		strata[key[0], sorted(topics, key=lambda t: (-topics[t], t))[0]].append(key)
	assignments = {}
	for stratum, keys in sorted(strata.items()):
		keys.sort(key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).digest())
		start = int.from_bytes(
			hashlib.sha256(f"{seed}:{stratum}".encode()).digest()[:4], "big"
		) % len(ROLES)
		for index, key in enumerate(keys):
			for item in groups[key]:
				assignments[item.prompt.example_id] = ROLES[(start + index) % len(ROLES)]
	groups_seen: dict[tuple[str, str], str] = {}
	for item in items:
		group = item.prompt.source.group_id
		if not group:
			raise ValueError(f"missing source group: {item.prompt.example_id}")
		key = (item.prompt.task, group)
		role = assignments[item.prompt.example_id]
		if key in groups_seen and groups_seen[key] != role:
			raise ValueError(f"source group spans family assignments: {key}")
		groups_seen[key] = role
	problems = screen(items, assignments)
	if problems:
		raise ValueError("cross-role near duplicates: " + "; ".join(problems[:5]))
	output.mkdir(parents=True, exist_ok=False)
	for role in ROLES:
		write_jsonl(
			output / f"{role}.jsonl",
			(item for item in items if assignments[item.prompt.example_id] == role),
		)
	manifest = {
		"version": "source-family-roles-v1",
		"seed": seed,
		"source_sha256": digest(source),
		"roles": {role: digest(output / f"{role}.jsonl") for role in ROLES},
		"provenance_note": "Family assertions require independent source audit.",
		"near_duplicate_screen": (
			"five-word shingles, Jaccard >= 0.8; "
			"shared shingles over 100 rows excluded from candidate search"
		),
	}
	save_json(output / "manifest.json", manifest)
	return manifest


def validate_roles(root: Path, *, min_horizons: int = 300) -> dict[str, Any]:
	manifest = json.loads((root / "roles" / "manifest.json").read_text())
	errors: list[str] = []
	all_items: list[SourceItem] = []
	assignments: dict[str, str] = {}
	families: dict[tuple[str, str], str] = {}
	source_groups: dict[tuple[str, str], str] = {}
	inputs: dict[str, str] = {}
	stats = {}
	all_accounts: dict[str, str] = {}
	for role in ROLES:
		path = root / "roles" / f"{role}.jsonl"
		if digest(path) != manifest["roles"][role]:
			errors.append(f"{role}: role hash mismatch")
		items = read_jsonl(path, SourceItem)
		if not items:
			errors.append(f"{role}: empty role")
		for item in items:
			try:
				_validate_source(item)
			except ValueError as error:
				errors.append(str(error))
			key = (item.prompt.task, item.family)
			if key in families and families[key] != role:
				errors.append(f"family overlaps roles: {key}")
			families[key] = role
			group_key = (item.prompt.task, item.prompt.source.group_id or "")
			if not group_key[1]:
				errors.append(f"missing source group: {item.prompt.example_id}")
			if group_key in source_groups and source_groups[group_key] != role:
				errors.append(f"source group overlaps roles: {group_key}")
			source_groups[group_key] = role
			if item.prompt.example_id in assignments:
				errors.append(f"duplicate example id: {item.prompt.example_id}")
			assignments[item.prompt.example_id] = role
		all_items.extend(items)
		schedule = root / role / "requests.jsonl"
		schedule_manifest = json.loads((root / role / "manifest.json").read_text())
		if schedule_manifest.get("version") != PROFILE_WORKLOAD_VERSION or digest(
			schedule
		) != schedule_manifest.get("sha256", {}).get("requests.jsonl"):
			errors.append(f"{role}: invalid schedule or hash")
		rows = read_jsonl(schedule, GatewayRequest)
		by_id = {item.prompt.example_id: item for item in items}
		used: dict[str, set[str]] = defaultdict(set)
		accounts: dict[str, list[GatewayRequest]] = defaultdict(list)
		family_accounts: dict[tuple[str, str], str] = {}
		group_accounts: dict[tuple[str, str], str] = {}
		for row in rows:
			item = by_id.get(row.prompt.example_id)
			if item is None or row.prompt != item.prompt:
				errors.append(f"{role}: request outside frozen role: {row.prompt.example_id}")
				continue
			family_key = (row.domain, item.family)
			if family_key in family_accounts and family_accounts[family_key] != row.account:
				errors.append(f"{role}: family shared between accounts: {family_key}")
			family_accounts[family_key] = row.account
			group_key = (row.domain, item.prompt.source.group_id or "")
			if group_key in group_accounts and group_accounts[group_key] != row.account:
				errors.append(f"{role}: source group shared between accounts: {group_key}")
			group_accounts[group_key] = row.account
			used[row.domain].add(row.prompt.example_id)
			accounts[row.account].append(row)
		for account, account_rows in accounts.items():
			if account in all_accounts:
				errors.append(f"account overlaps roles: {account}")
			all_accounts[account] = role
			if len({row.client_type for row in account_rows}) != 1:
				errors.append(f"{role}: inconsistent profile for {account}")
			if (
				max(row.time for row in account_rows) - min(row.time for row in account_rows)
				>= 86400
			):
				errors.append(f"{role}: account spans multiple daily horizons: {account}")
		counts = Counter(rows[0].client_type for rows in accounts.values() if len(rows) >= 100)
		for profile in SCORED_PROFILES:
			if counts[profile] < min_horizons:
				errors.append(
					f"{role}: {profile} has {counts[profile]} scored accounts; "
					f"requires {min_horizons}"
				)
		coverage = {}
		for domain in sorted({item.prompt.task for item in items}):
			pool = Counter(item.reference.category for item in items if item.prompt.task == domain)
			seen = Counter(by_id[key].reference.category for key in used[domain])
			coverage[domain] = topic_coverage(pool, seen)
			if not used[domain] or not coverage[domain]["passed"]:
				errors.append(f"{role}: skewed or missing {domain} topics")
		detector = root / f"{role}.detector.jsonl"
		from harness.calibration import BenignRequest
		from harness.serialization import gateway_turn

		expected = [
			BenignRequest(
				account=row.account,
				text=gateway_turn(
					row.domain,
					next(m.content for m in reversed(row.prompt.prompt) if m.role == "user"),
				),
				time=row.time,
				example_id=row.prompt.example_id,
				source_group=f"{row.domain}:{row.prompt.source.group_id}",
				client_type=row.client_type,
				workload_version=PROFILE_WORKLOAD_VERSION,
			)
			for row in rows
		]
		if read_jsonl(detector, BenignRequest) != expected:
			errors.append(f"{role}: detector inputs differ from schedule")
		inputs[role] = digest(detector)
		stats[role] = {"scored_accounts": dict(counts), "topic_coverage": coverage}
	errors.extend(screen(all_items, assignments))
	return {
		"version": "role-validation-v1",
		"passed": not errors,
		"reasons": errors,
		"detector_sha256": inputs,
		"roles": stats,
		"minimum_scored_accounts_per_profile": min_horizons,
	}
