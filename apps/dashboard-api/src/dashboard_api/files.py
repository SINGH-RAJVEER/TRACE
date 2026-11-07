"""Bounded, read-only access to files under the configured artifact roots.

Every read goes through `Roots.resolve`, which rejects a path that leaves its root, including
through a symlink. Readers return a `Loaded` state rather than raising, so a missing, malformed
or unhydrated file is reported as such and never as an empty result.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

# Git LFS replaces tracked study data with a pointer file until the object is pulled.
LFS_POINTER = b"version https://git-lfs.github.com/spec/"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_TEXT_BYTES = 256 * 1024
# Files that hold credentials or raw account state. They are never opened, whatever the request.
NEVER_READ = re.compile(
	r"(^|/)(credentials\.json|accounts\.json|deception\.key|environment\.env|\.env)$"
	r"|\.sqlite3(-\w+)?$|\.key$"
)
SECRET_KEY = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE", re.IGNORECASE)
URL_USERINFO = re.compile(r"(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s]+@", re.IGNORECASE)
REDACTED = "<redacted>"

State = Literal["ok", "missing", "malformed", "lfs-pointer", "too-large", "refused"]


@dataclass
class Loaded:
	"""A file read: `value` is set only when `state` is ok."""

	state: State
	path: str
	value: Any = None
	error: str | None = None
	sha256: str | None = None

	def public(self) -> dict[str, Any]:
		return {"state": self.state, "path": self.path, "error": self.error, "sha256": self.sha256}


@dataclass(frozen=True)
class Root:
	name: str
	path: Path


@dataclass
class Roots:
	"""Named artifact roots. Paths given to the API are relative to one of them."""

	roots: list[Root] = field(default_factory=list)

	@classmethod
	def parse(cls, spec: str, base: Path) -> Roots:
		"""`name=path` or `path` entries separated by commas; relative paths resolve from base."""
		roots: list[Root] = []
		for entry in (part.strip() for part in spec.split(",")):
			if not entry:
				continue
			name, _, raw = entry.partition("=") if "=" in entry else ("", "", entry)
			path = Path(raw).expanduser()
			path = (path if path.is_absolute() else base / path).resolve()
			name = name.strip() or path.name
			if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
				raise ValueError(f"root name {name!r} must use letters, digits, '.', '_' or '-'")
			if any(root.name == name for root in roots):
				raise ValueError(f"root name {name!r} is configured twice")
			roots.append(Root(name, path))
		return cls(roots)

	def get(self, name: str) -> Root | None:
		return next((root for root in self.roots if root.name == name), None)

	def resolve(self, name: str, relative: str) -> Path | None:
		"""The real path of `relative` under root `name`, or None if it would leave the root."""
		root = self.get(name)
		if root is None:
			return None
		pure = PurePosixPath(relative)
		if pure.is_absolute() or ".." in pure.parts or "\\" in relative or "\x00" in relative:
			return None
		candidate = (root.path / pure).resolve()
		return candidate if candidate.is_relative_to(root.path) else None

	def locate(self, path: Path) -> tuple[str, str] | None:
		"""Root name and relative path of an absolute path inside one of the roots."""
		real = path.resolve()
		for root in self.roots:
			if real.is_relative_to(root.path):
				return root.name, real.relative_to(root.path).as_posix() or "."
		return None


def _refused(path: Path) -> bool:
	"""Symlinks are never followed, so a link in the file or any parent directory cannot reach a
	file outside its root or a credential file under another name. Roots are resolved already."""
	absolute = Path(os.path.abspath(path))
	if path.resolve() != absolute:
		return True
	return bool(NEVER_READ.search(absolute.as_posix()))


def file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for chunk in iter(lambda: stream.read(1 << 20), b""):
			digest.update(chunk)
	return digest.hexdigest()


def read_json(path: Path, label: str, *, max_bytes: int = MAX_JSON_BYTES) -> Loaded:
	"""A JSON file, reported as missing, refused, too large, an LFS pointer or malformed."""
	if _refused(path):
		return Loaded("refused", label, error="a symlink or a file holding credentials or state")
	if not path.is_file():
		return Loaded("missing", label, error="file not found")
	size = path.stat().st_size
	if size > max_bytes:
		return Loaded("too-large", label, error=f"{size} bytes exceeds the {max_bytes} byte limit")
	raw = path.read_bytes()
	if raw.startswith(LFS_POINTER):
		return Loaded(
			"lfs-pointer", label, error="Git LFS pointer; pull the object to read this file"
		)
	sha = hashlib.sha256(raw).hexdigest()
	try:
		value = json.loads(raw)
	except ValueError as error:
		return Loaded("malformed", label, error=f"invalid JSON: {error}", sha256=sha)
	return Loaded("ok", label, value=value, sha256=sha)


def read_text(path: Path, label: str, *, max_bytes: int = MAX_TEXT_BYTES) -> Loaded:
	"""A small text file; longer files are cut at `max_bytes` and say so in `error`."""
	if _refused(path):
		return Loaded("refused", label, error="a symlink or a file holding credentials or state")
	if not path.is_file():
		return Loaded("missing", label, error="file not found")
	with path.open("rb") as stream:
		raw = stream.read(max_bytes + 1)
	if raw.startswith(LFS_POINTER):
		return Loaded("lfs-pointer", label, error="Git LFS pointer; pull the object to read it")
	text = raw[:max_bytes].decode("utf-8", errors="replace")
	error = f"truncated at {max_bytes} bytes" if len(raw) > max_bytes else None
	return Loaded("ok", label, value=text, error=error)


def redact_value(key: str, value: str) -> str:
	if value and SECRET_KEY.search(key):
		return REDACTED
	return URL_USERINFO.sub(lambda match: f"{match['scheme']}{REDACTED}@", value)


def parse_env(text: str) -> list[dict[str, str]]:
	"""KEY=VALUE lines with secrets redacted; comments and other lines are dropped."""
	pairs = []
	for line in text.splitlines():
		if line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		key = key.strip()
		if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
			continue
		pairs.append({"key": key, "value": redact_value(key, value.strip())})
	return pairs


def sniff(path: Path, size: int = 2048) -> bytes:
	"""The first bytes of a file, for content-based recognition without a full parse."""
	if _refused(path):
		return b""
	try:
		with path.open("rb") as stream:
			return stream.read(size)
	except OSError:
		return b""


def scan_dir(path: Path) -> tuple[list[os.DirEntry[str]], list[os.DirEntry[str]]]:
	"""Files and directories directly in `path`, without following symlinked directories."""
	files: list[os.DirEntry[str]] = []
	dirs: list[os.DirEntry[str]] = []
	try:
		with os.scandir(path) as entries:
			for entry in entries:
				if entry.is_dir(follow_symlinks=False):
					dirs.append(entry)
				elif entry.is_file(follow_symlinks=False):
					files.append(entry)
	except OSError:
		pass
	files.sort(key=lambda entry: entry.name)
	dirs.sort(key=lambda entry: entry.name)
	return files, dirs
