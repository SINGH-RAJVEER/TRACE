"""Verify preserved payloads without requiring Git LFS working-tree hydration."""

import csv
import hashlib
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POINTER = re.compile(
	rb"version https://git-lfs.github.com/spec/v1\n"
	rb"oid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n"
)


def lfs_cache() -> Path:
	env = subprocess.check_output(["git", "-C", str(ROOT), "lfs", "env"], text=True)
	return Path(
		next(
			line.split("=", 1)[1] for line in env.splitlines() if line.startswith("LocalMediaDir=")
		)
	)


def payload(path: Path, cache: Path) -> Path:
	if path.stat().st_size < 1024:
		pointer = POINTER.fullmatch(path.read_bytes())
		if pointer:
			oid = pointer[1].decode()
			return cache / oid[:2] / oid[2:4] / oid
	return path


def main() -> None:
	cache = lfs_cache()
	manifest = payload(ROOT / "MANIFEST.tsv", cache)
	count = 0
	with manifest.open(newline="") as stream:
		for row in csv.DictReader(stream, delimiter="\t"):
			relative = Path(row["path"])
			if relative.is_absolute() or ".." in relative.parts:
				raise ValueError(f"Unsafe manifest path: {relative}")
			path = ROOT / relative
			if row["mode"] == "120000":
				data = os.fsencode(os.readlink(path))
				size = len(data)
				digest = hashlib.sha256(data).hexdigest()
			else:
				if path.is_symlink():
					raise ValueError(f"Unexpected symlink: {relative}")
				content = payload(path, cache)
				size = content.stat().st_size
				with content.open("rb") as file:
					digest = hashlib.file_digest(file, "sha256").hexdigest()
			if size != int(row["bytes"]) or digest != row["sha256"]:
				raise ValueError(f"Payload mismatch: {relative}")
			count += 1
	if count != 34829:
		raise ValueError(f"Expected 34829 source entries, found {count}")
	print(f"Verified all {count} preserved entries.")


if __name__ == "__main__":
	main()
