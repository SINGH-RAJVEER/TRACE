"""Command line for the watermark probe."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from watermark_probe.vectors import compute_vectors, find_mismatches, load_fixture


def _vectors(args: argparse.Namespace) -> int:
	try:
		fixture = load_fixture(args.fixture)
		computed = compute_vectors(fixture)
	except (OSError, ValueError) as error:
		print(f"vectors: cannot use fixture {args.fixture}: {error}", file=sys.stderr)
		return 1
	if "expected" not in fixture:
		print(f"vectors: fixture {args.fixture} has no expected values", file=sys.stderr)
		return 1
	if mismatches := find_mismatches(fixture["expected"], computed):
		print(f"vectors: {len(mismatches)} mismatches against expected:", file=sys.stderr)
		for mismatch in mismatches:
			print(f"  {mismatch}", file=sys.stderr)
		print("vectors: no output written", file=sys.stderr)
		return 1
	try:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		args.output.write_text(json.dumps(computed, indent="\t") + "\n", encoding="utf-8")
	except OSError as error:
		print(f"vectors: cannot write {args.output}: {error}", file=sys.stderr)
		return 1
	return 0


def _bench(args: argparse.Namespace) -> int:
	from watermark_probe.benchmark import run_benchmarks

	report = run_benchmarks(quick=args.quick)
	try:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		args.output.write_text(json.dumps(report, indent="\t") + "\n", encoding="utf-8")
	except OSError as error:
		print(f"bench: cannot write {args.output}: {error}", file=sys.stderr)
		return 1
	return 0


def _not_implemented(args: argparse.Namespace) -> int:
	print(
		f"{args.command}: not implemented; it needs a serving stage or data that does not exist",
		file=sys.stderr,
	)
	return 2


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="watermark-probe", description=__doc__)
	commands = parser.add_subparsers(dest="command", required=True)

	vectors = commands.add_parser("vectors", help="check and write the reference vectors")
	vectors.add_argument("--fixture", type=Path, required=True)
	vectors.add_argument("--output", type=Path, required=True)
	vectors.set_defaults(handler=_vectors)

	bench = commands.add_parser("bench", help="measure CPU latency on synthetic inputs")
	bench.add_argument("--output", type=Path, required=True)
	bench.add_argument("--quick", action="store_true", help="prove it runs; timings mean little")
	bench.set_defaults(handler=_bench)

	planned = {
		"replay": (
			"--manifest",
			"--profiles",
			"--temperatures",
			"--lengths",
			"--concurrency",
			"--output",
		),
		"calibrate": ("--split", "--generations", "--familywise-alpha", "--output"),
		"evaluate": ("--split", "--calibration", "--generations", "--output"),
	}
	for name, options in planned.items():
		command = commands.add_parser(name, help="not implemented")
		for option in options:
			command.add_argument(option)
		command.set_defaults(handler=_not_implemented)
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	args = _build_parser().parse_args(argv)
	return int(args.handler(args))


if __name__ == "__main__":
	raise SystemExit(main())
