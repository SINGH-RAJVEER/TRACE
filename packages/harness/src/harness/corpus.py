"""Operator-declared public prompt indices, stored separately from calibrations."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


def normalized(text: str) -> str:
	return " ".join(re.findall(r"\w+", text.casefold()))


def shingles(text: str) -> frozenset[str]:
	words = normalized(text).split()
	return frozenset(" ".join(words[i : i + 3]) for i in range(max(1, len(words) - 2)))


class CorpusIndex(BaseModel):
	"""Content-addressed index; source labels should include pinned revision/provenance."""

	model_config = ConfigDict(extra="forbid")
	version: str = "public-corpus-v1"
	sources: dict[str, list[str]] = Field(default_factory=dict)
	near_threshold: float = Field(default=0.8, gt=0, le=1)
	_exact: dict[bytes, set[str]] = PrivateAttr(default_factory=dict)
	_postings: dict[str, set[int]] = PrivateAttr(default_factory=dict)
	_entries: list[tuple[str, frozenset[str]]] = PrivateAttr(default_factory=list)

	def model_post_init(self, context: object) -> None:
		"""Compile query lookup tables once; register changes by creating a new index."""
		for source, prompts in self.sources.items():
			for prompt in prompts:
				key = hashlib.sha256(normalized(prompt).encode()).digest()
				self._exact.setdefault(key, set()).add(source)
				parts = shingles(prompt)
				index = len(self._entries)
				self._entries.append((source, parts))
				for part in parts:
					self._postings.setdefault(part, set()).add(index)

	@classmethod
	def from_sources(cls, sources: Mapping[str, Sequence[str]]) -> CorpusIndex:
		if any(not label.strip() for label in sources):
			raise ValueError("corpus sources require nonempty provenance labels")
		return cls(sources={k: sorted(set(v)) for k, v in sorted(sources.items())})

	@property
	def digest(self) -> str:
		payload = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
		return hashlib.sha256(payload.encode()).hexdigest()

	def save(self, path: Path) -> None:
		path.write_text(self.model_dump_json(indent=2) + "\n")

	@classmethod
	def load(cls, path: Path) -> CorpusIndex:
		return cls.model_validate_json(path.read_text())

	def match(self, text: str) -> dict[str, object]:
		"""Normalized exact hash or word-trigram Jaccard match, with declared provenance."""
		key = hashlib.sha256(normalized(text).encode()).digest()
		query = shingles(text)
		matches = set(self._exact.get(key, set()))
		exact = bool(matches)
		best = 1.0 if exact else 0.0
		candidates: set[int] = set()
		for part in query:
			candidates.update(self._postings.get(part, set()))
		for index in candidates:
			source, other = self._entries[index]
			similarity = len(query & other) / max(1, len(query | other))
			best = max(best, similarity)
			if similarity >= self.near_threshold:
				matches.add(source)
		return {
			"matched": bool(matches),
			"exact": exact,
			"sources": sorted(matches),
			"similarity": best,
		}
