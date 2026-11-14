"""Private, persistent master keys shared by the teacher, vLLM workers and detector."""

import os
import secrets
import tempfile
from contextlib import suppress
from pathlib import Path


def read_master_secret(path: Path) -> bytes:
	"""A hex secret of at least 16 bytes; never logged or written to manifests."""
	try:
		secret = bytes.fromhex(path.read_text().strip())
	except (OSError, ValueError) as error:
		raise ValueError("the watermark key file must hold a hex secret") from error
	if len(secret) < 16:
		raise ValueError("the watermark secret must be at least 16 bytes")
	return secret


def load_or_create_master_secret(path: Path) -> bytes:
	"""Atomically publish one private key without replacing a concurrent starter's key."""
	if path.exists():
		return read_master_secret(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".watermark-", suffix=".watermark.key")
	try:
		with os.fdopen(fd, "w") as stream:
			stream.write(secrets.token_hex(32) + "\n")
			stream.flush()
			os.fsync(stream.fileno())
		with suppress(FileExistsError):
			os.link(temporary, path)
	finally:
		os.unlink(temporary)
	return read_master_secret(path)
