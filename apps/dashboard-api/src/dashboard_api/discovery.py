"""Find runs and result artifacts under the configured roots by their content.

A bounded walk sniffs the start of each JSON file for the version its producer writes. Study
run directories are recognized by their runner files. Artifacts nested in a run belong to that
run; others form their own entry. Index rows join a directory by run ID or reading name. Paths
never leave the server: each run is addressed by an opaque ID.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dashboard_api.files import Roots, scan_dir, sniff

MAX_DEPTH = 6
MAX_DIRECTORIES = 20_000
# Bulky or credential-bearing subtrees that hold no artifact the dashboard reads by discovery.
SKIP = frozenset(
	{
		".git",
		"node_modules",
		"embedding-cache",
		"logs",
		"steps",
		"scores",
		"stages",
		"provenance",
		"schedules",
		"calibration",
		"benign",
		"campaigns",
		"utility",
		"collection",
		"method-checks",
	}
)
VERSION = re.compile(rb'^\s*\{\s*"version"\s*:\s*"([^"]{1,80})"')
TRANSFER = re.compile(rb'^\s*\{\s*"evidence"\s*:')

ArtifactKind = Literal[
	"gate", "capacity", "scores", "transfer", "learning-curve", "checkpoints", "benchmark", "study"
]
FILE_KINDS: dict[str, ArtifactKind] = {
	"clean-baseline-gate-v1": "gate",
	"teacher-capacity-v1": "capacity",
	"problem-scores-v1": "scores",
}
CURVE_VERSIONS = ("math-learning-curve-v1", "ag-news-learning-curve-v1")
CHECKPOINT_VERSION = "checkpoint-learning-v1"
# A consolidated student benchmark names the run it scores in its version:
# v7-student-benchmark-v1 belongs to run v7.
BENCHMARK = re.compile(r"(?P<run>[A-Za-z0-9_.:-]+)-student-benchmark-v\d+")


@dataclass(frozen=True)
class Artifact:
	kind: ArtifactKind
	root: str
	path: str
	absolute: Path
	version: str | None = None

	def public(self) -> dict[str, Any]:
		return {"kind": self.kind, "root": self.root, "path": self.path, "version": self.version}


@dataclass
class Run:
	id: str
	key: str
	kind: str
	label: str
	run_id: str | None = None
	readings: set[str] = field(default_factory=set)
	directories: list[Artifact] = field(default_factory=list)
	artifacts: list[Artifact] = field(default_factory=list)
	rows: list[dict[str, Any]] = field(default_factory=list)


def opaque(key: str) -> str:
	return hashlib.sha256(key.encode()).hexdigest()[:16]


def _version(path: Path) -> str | None:
	match = VERSION.match(sniff(path))
	return match[1].decode(errors="replace") if match else None


def _classify(directory: Path, root: str, relative: str) -> list[Artifact]:
	files, dirs = scan_dir(directory)
	names = {entry.name for entry in files}
	subdirs = {entry.name for entry in dirs}
	found: list[Artifact] = []
	if "steps.tsv" in names or ("exit" in names and subdirs & {"campaigns", "utility"}):
		found.append(Artifact("study", root, relative, directory))
	if "plan.json" in names:
		version = _version(directory / "plan.json")
		if version in CURVE_VERSIONS:
			found.append(Artifact("learning-curve", root, relative, directory, version))
		elif version == CHECKPOINT_VERSION:
			found.append(Artifact("checkpoints", root, relative, directory, version))
	for entry in files:
		if not entry.name.endswith(".json") or entry.name in ("plan.json", "summary.json"):
			continue
		path = Path(entry.path)
		child = f"{relative}/{entry.name}" if relative != "." else entry.name
		version = _version(path)
		if version in FILE_KINDS:
			found.append(Artifact(FILE_KINDS[version], root, child, path, version))
		elif version is not None and BENCHMARK.fullmatch(version):
			found.append(Artifact("benchmark", root, child, path, version))
		elif version is None and TRANSFER.match(sniff(path)):
			found.append(Artifact("transfer", root, child, path))
	return found


def walk(roots: Roots) -> list[Artifact]:
	"""Every recognized artifact under the roots, depth- and count-bounded."""
	found: list[Artifact] = []
	visited = 0
	for root in roots.roots:
		if not root.path.is_dir():
			continue
		stack: list[tuple[Path, str, int]] = [(root.path, ".", 0)]
		while stack and visited < MAX_DIRECTORIES:
			directory, relative, depth = stack.pop()
			visited += 1
			found += _classify(directory, root.name, relative)
			if depth >= MAX_DEPTH:
				continue
			_, dirs = scan_dir(directory)
			for entry in reversed(dirs):
				if entry.name in SKIP or entry.name.startswith("."):
					continue
				child = entry.name if relative == "." else f"{relative}/{entry.name}"
				stack.append((Path(entry.path), child, depth + 1))
	return found


def _directory(artifact: Artifact) -> tuple[Path, str]:
	"""The directory that owns an artifact and its root-relative path."""
	if artifact.absolute.is_dir():
		return artifact.absolute, artifact.path
	return artifact.absolute.parent, artifact.path.rsplit("/", 1)[
		0
	] if "/" in artifact.path else "."


def _directory_runs(artifacts: list[Artifact]) -> dict[Path, Run]:
	"""One run per outermost owning directory; nested artifacts join the enclosing run."""
	places: dict[Path, tuple[str, str]] = {}
	kinds: dict[Path, set[str]] = {}
	for artifact in artifacts:
		directory, relative = _directory(artifact)
		places.setdefault(directory, (artifact.root, relative))
		kinds.setdefault(directory, set()).add(artifact.kind)
	runs: dict[Path, Run] = {}
	for directory in sorted(places, key=lambda path: (len(path.parts), str(path))):
		if any(directory.is_relative_to(owner) for owner in runs):
			continue
		root, relative = places[directory]
		found = kinds[directory]
		kind = (
			"study"
			if "study" in found
			else "learning-curve"
			if found == {"learning-curve"}
			else "checkpoints"
			if found == {"checkpoints"}
			else "baseline-gates"
		)
		key = f"dir:{root}/{relative}"
		label = f"{root}/{relative}" if relative != "." else root
		place = Artifact("study", root, relative, directory)
		runs[directory] = Run(opaque(key), key, kind, label, directories=[place])
	for artifact in artifacts:
		directory, _ = _directory(artifact)
		owner = next((o for o in runs if directory.is_relative_to(o)), None)
		if owner is not None and artifact.kind != "study":
			runs[owner].artifacts.append(artifact)
	return runs


def build_runs(
	artifacts: list[Artifact],
	rows: list[dict[str, Any]],
	readings: dict[str, Artifact] | None = None,
) -> list[Run]:
	"""Group artifacts into runs and join index rows by run ID or reading name.

	A student benchmark joins the run its version names, wherever the file lives; with no such
	run it is listed on its own.

	`readings` maps a directory name directly under a root to that directory, so an indexed run
	whose reading has no runner files, such as a backfilled one, still shows its README.
	"""
	benchmarks = [a for a in artifacts if a.kind == "benchmark"]
	runs = _directory_runs([a for a in artifacts if a.kind != "benchmark"])
	by_name: dict[str, list[Path]] = {}
	for directory in runs:
		by_name.setdefault(directory.name, []).append(directory)
	consumed: set[Path] = set()
	indexed: dict[str, Run] = {}
	for row in rows:
		run_id = row["run_id"]
		if run_id is None:
			continue
		run = indexed.get(run_id)
		if run is None:
			key = f"index:{run_id}"
			run = Run(opaque(key), key, "index", run_id, run_id)
			name = run_id.split(":", 1)[1] if ":" in run_id else run_id
			candidates = {name, row.get("reading") or name}
			for candidate in sorted(candidates):
				for directory in by_name.get(candidate, []):
					if directory not in consumed:
						consumed.add(directory)
						run.directories += runs[directory].directories
						run.artifacts += runs[directory].artifacts
				place = (readings or {}).get(candidate)
				if place is not None and all(d.absolute != place.absolute for d in run.directories):
					run.directories.append(place)
			indexed[run_id] = run
		if row.get("reading"):
			run.readings.add(row["reading"])
		run.rows.append(row)
	for run in indexed.values():
		kinds = {row["kind"] for row in run.rows}
		run.kind = (
			"study"
			if kinds & {"campaign", "utility"}
			else "student"
			if kinds == {"student"}
			else "gate"
			if kinds == {"gate"}
			else "index"
		)
	loose = [r for d, r in runs.items() if d not in consumed]
	for artifact in benchmarks:
		name = BENCHMARK.fullmatch(artifact.version or "")["run"]  # type: ignore[index]
		owner = next(
			(r for r in indexed.values() if name in (r.run_id, r.run_id.split(":", 1)[-1])), None
		)
		owner = owner or next((r for r in loose if r.label.rsplit("/", 1)[-1] == name), None)
		if owner is None:
			key = f"dir:{artifact.root}/{artifact.path}"
			owner = Run(opaque(key), key, "benchmark", f"{artifact.root}/{artifact.path}")
			loose.append(owner)
		owner.artifacts.append(artifact)
	return list(indexed.values()) + loose


def reading_directories(roots: Roots) -> dict[str, Artifact]:
	"""Directories directly under each root, by name; the first root wins a shared name."""
	found: dict[str, Artifact] = {}
	for root in roots.roots:
		_, dirs = scan_dir(root.path)
		for entry in dirs:
			if not entry.name.startswith("."):
				found.setdefault(
					entry.name, Artifact("study", root.name, entry.name, Path(entry.path))
				)
	return found
