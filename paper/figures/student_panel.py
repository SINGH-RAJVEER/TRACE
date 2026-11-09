"""Validate recorded v7 evaluations and publish the complete student benchmark panel.

Run `uv run python -m paper.figures.student_panel` from the repository root.
Use --verify to recompute statistics from the published correctness snapshot alone.
No inference, training or teacher calls are performed.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from .students import bootstrap, mcnemar, wilson

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/student-panel.json"
MODELS = (
	("A2/full", "v7-detector-panel", "plan-A2-full", "trace-v7-a2-full-s42"),
	(
		"A2/instantaneous",
		"v7-detector-panel",
		"plan-A2-instantaneous",
		"trace-v7-a2-instantaneous-s42",
	),
	("A5/full", "v7", "plan-A5-full", "trace-v7-a5-full-s42"),
	(
		"A5/instantaneous",
		"v7-detector-panel",
		"plan-A5-instantaneous",
		"trace-v7-a5-instantaneous-s42",
	),
	("unprotected", "v7", "plan-A6-unprotected", "trace-v7-a6-unprotected-s42"),
)
LIMITS = (
	"Recorded inference is reused; this report does not perform a fresh inference run.",
	"One training seed (42), GSM8K development suite only; not a full-test leaderboard result.",
	"Intervals condition on the trained models and resample items, not training or campaign seeds.",
	"Items may share source families; item-level intervals can understate dependence.",
	"Arms differ in training membership, size and campaign history; comparisons are descriptive.",
	"Paired intervals are pointwise. McNemar p-values include Holm adjustment across all pairs.",
	"No matched quota students or general distillation-prevention qualification are provided.",
)


def sha(path):
	return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(cells, ids, resamples=10000):
	"""Recompute every estimate from binary correctness aligned by the same ordered IDs."""
	if not ids or len(ids) != len(set(ids)):
		raise ValueError("need distinct evaluation item IDs")
	marks = {}
	for name, cell in cells.items():
		bits = cell["correctness"]
		if not isinstance(bits, str) or len(bits) != len(ids) or set(bits) - {"0", "1"}:
			raise ValueError(f"invalid or incomplete correctness for {name}")
		marks[name] = [int(v) for v in bits]
	result = {
		name: {
			"correct": sum(values),
			"accuracy": sum(values) / len(ids),
			"wilson95": wilson(sum(values), len(ids)),
		}
		for name, values in marks.items()
	}
	pairs = {}
	for a, b in itertools.combinations(marks, 2):
		diffs = [x - y for x, y in zip(marks[a], marks[b], strict=True)]
		a_only, b_only = diffs.count(1), diffs.count(-1)
		pairs[f"{a} minus {b}"] = {
			"a": a,
			"b": b,
			"difference": sum(diffs) / len(ids),
			"a_only": a_only,
			"b_only": b_only,
			"paired_item_bootstrap95": bootstrap(diffs, draws=resamples, seed=42),
			"mcnemar_exact_p": mcnemar(a_only, b_only),
		}
	previous = 0.0
	for index, row in enumerate(sorted(pairs.values(), key=lambda r: r["mcnemar_exact_p"])):
		previous = max(previous, min(1.0, row["mcnemar_exact_p"] * (len(pairs) - index)))
		row["mcnemar_holm_p"] = previous
	return result, pairs


def collect(source):
	from distillation import checkpoint_learning as cl
	from distillation.baseline_gates import read_items, suite_digest
	from distillation.learning_curve import MATH_RECIPE

	suite = ROOT / "data/math/development-suite.json"
	items = read_items(suite)
	ids = sorted(item.id for item in items)
	canonical_suite = suite_digest(items)
	frozen_suite = json.loads(suite.read_text())["sha256"]
	cells = {}
	for role in ("base", "teacher"):
		path = source / "v7" / f"{role}-dev.json"
		score = json.loads(path.read_text())
		if score["suite_sha256"] != canonical_suite or sorted(score["scores"]) != ids:
			raise ValueError(f"{role} scored another suite")
		if score["evaluation_max_new_tokens"] != 512:
			raise ValueError(f"{role} used another generation cap")
		cl.validate_scores(score["scores"], set(ids))
		cells[role] = {
			"role": role,
			"model": score["model"],
			"revision": score["revision"],
			"source_score_sha256": sha(path),
			"correctness": "".join(str(int(score["scores"][i])) for i in ids),
		}
	if cells["base"]["revision"] != MATH_RECIPE["revision"]:
		raise ValueError("base revision differs from the training recipe")
	for label, folder, directory, model in MODELS:
		out = source / folder / directory
		plan_path = out / "plan.json"
		plan = json.loads(plan_path.read_text())
		if plan["recipe"] != MATH_RECIPE or plan["suite"]["frozen_sha256"] != frozen_suite:
			raise ValueError(f"{label} uses another recipe or suite")
		if plan["seeds"] != [42]:
			raise ValueError(f"{label} is not the expected seed-42 cell")
		for relative, expected in plan["files"].items():
			if sha(out / relative) != expected:
				raise ValueError(f"{label} stage hash mismatch: {relative}")
		for raw, expected in plan["inputs"].items():
			current = Path(raw.replace("/TRACE-main/", "/TRACE/", 1))
			if sha(current) != expected:
				raise ValueError(f"{label} source hash mismatch")
		export = Path(plan["export"]["path"].replace("/TRACE-main/", "/TRACE/", 1))
		for relative, expected in plan["export"]["ledgers"].items():
			if sha(export / relative) != expected:
				raise ValueError(f"{label} export ledger changed")
		cell = next(c for c in plan["cells"] if c["checkpoint"] == "end")
		score = cl._load_cell(
			out,
			sha(plan_path),
			plan["files"]["suite.json"],
			set(ids),
			checkpoint="end",
			n=cell["n"],
			seed=42,
		)
		if score is None:
			raise ValueError(f"{label} evaluation is incomplete")
		model_dir = out / f"stages/end/models/n{cell['n']}-s42"
		config = json.loads((model_dir / "adapter_config.json").read_text())
		if config["base_model_name_or_path"] != MATH_RECIPE["base_model"] or config["r"] != 16:
			raise ValueError(f"{label} adapter identity mismatch")
		receipt = ROOT / "artifacts/hf-upload" / f"{model}-upload-receipt.json"
		hub = json.loads(receipt.read_text())
		cells[label] = {
			"role": "student",
			"model": MATH_RECIPE["base_model"],
			"revision": MATH_RECIPE["revision"],
			"seed": 42,
			"training_examples": cell["n"],
			"checkpoint": "end",
			"plan_sha256": sha(plan_path),
			"source_score_sha256": sha(out / "scores/end-s42.json"),
			"training_manifest_sha256": plan["files"][f"stages/end/subset-{cell['n']}.json"],
			"adapter_weights_sha256": sha(model_dir / "adapter_model.safetensors"),
			"hf_repo": hub["repo"],
			"hf_adapter_commit": hub["commit"],
			"correctness": "".join(str(int(score["scores"][i])) for i in ids),
		}
	return {
		"version": "v7-student-benchmark-v1",
		"task": "gsm8k",
		"split": "development",
		"metric": "numeric_exact_match",
		"suite_sha256": frozen_suite,
		"suite_items_sha256": canonical_suite,
		"items": len(ids),
		"item_ids": ids,
		"generation": {"do_sample": False, "enable_thinking": False, "max_new_tokens": 512},
		"recipe": MATH_RECIPE,
		"evaluation_mode": "validated_recorded_inference",
		"bootstrap": {"replicates": 10000, "seed": 42, "unit": "paired evaluation item"},
		"cells": cells,
		"limitations": list(LIMITS),
	}


def markdown(data):
	lines = [
		"# v7 student benchmark panel",
		"",
		"Validated recorded inference on the same frozen 746-item GSM8K development suite. "
		"No new inference, training or teacher requests were made to assemble this report.",
		"",
		"| Model / condition | Training examples | Correct | Accuracy | Wilson 95% interval |",
		"|---|---:|---:|---:|---:|",
	]
	for label, cell in data["cells"].items():
		low, high = cell["wilson95"]
		lines.append(
			f"| {label} | {cell.get('training_examples', '')} | "
			f"{cell['correct']} / {data['items']} | {cell['accuracy'] * 100:.2f}% | "
			f"[{low * 100:.2f}, {high * 100:.2f}]% |"
		)
	lines += [
		"",
		"## Scope and uncertainty",
		"",
		*[f"- {s}" for s in data["limitations"]],
		"",
		"All 21 pairwise comparisons are retained in student-panel.json, including paired "
		"item-bootstrap intervals, exact McNemar p-values and Holm-adjusted p-values. "
		"Positive differences favor the first named model. These are exploratory comparisons.",
		"",
		"## Reproduction",
		"",
		"Run `uv run python -m paper.figures.student_panel` "
		"with the original local artifacts, or `uv run python -m paper.figures.student_panel "
		"--verify paper/figures/student-panel.json` to verify every statistic from the published "
		"correctness vectors alone. The snapshot contains ordered item IDs and artifact hashes, "
		"without prompts, completions, account files, logs or credentials.",
		"",
	]
	return "\n".join(lines)


def plot(data, destination):
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "v7-student-panel"})
	fig, ax = plt.subplots(figsize=(9, 4.3))
	labels = []
	for index, (name, cell) in enumerate(data["cells"].items()):
		low, high = [100 * v for v in cell["wilson95"]]
		value = cell["accuracy"] * 100
		color = "#52606d" if cell["role"] != "student" else "#225ea8"
		marker = "D" if cell["role"] != "student" else "o"
		ax.plot([low, high], [index, index], color=color, linewidth=2)
		ax.scatter([value], [index], color=color, marker=marker, s=28, zorder=3)
		ax.text(high + 1, index, f"{value:.2f}% ({cell['correct']}/746)", va="center")
		labels.append(
			name + (f" (n={cell['training_examples']})" if "training_examples" in cell else "")
		)
	ax.set_yticks(range(len(labels)), labels)
	ax.set_xlim(0, 100)
	ax.set_ylim(len(labels) - 0.5, -0.75)
	ax.set_xlabel("Numeric exact-match accuracy (%)")
	ax.set_title("GSM8K development: five students and two references", loc="left", pad=16)
	ax.spines[["top", "right", "left"]].set_visible(False)
	ax.tick_params(axis="y", length=0)
	ax.grid(axis="x", alpha=0.15)
	ax.set_axisbelow(True)
	fig.text(
		0.02,
		0.02,
		"746 shared items; seed 42. Wilson 95% intervals. Unequal training sets; development only.",
	)
	fig.tight_layout(rect=(0, 0.05, 1, 1))
	fig.savefig(destination, metadata={"Date": None})
	plt.close(fig)
	lines = destination.read_text().splitlines()
	destination.write_text(
		"\n".join("\t" * (len(line) - len(line.lstrip(" "))) + line.lstrip(" ") for line in lines)
		+ "\n"
	)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--source", type=Path, default=ROOT / "artifacts/students")
	parser.add_argument("--output", type=Path, default=OUT)
	parser.add_argument("--verify", type=Path)
	parser.add_argument(
		"--plot", action="store_true", help="Plot the saved report; needs matplotlib."
	)
	args = parser.parse_args()
	if args.plot:
		plot(json.loads(args.output.read_text()), args.output.with_suffix(".svg"))
		return
	if args.verify:
		data = json.loads(args.verify.read_text())
		metrics, pairs = summarize(data["cells"], data["item_ids"])
		if (
			any(
				any(data["cells"][name][k] != v for k, v in values.items())
				for name, values in metrics.items()
			)
			or pairs != data["pairs"]
		):
			raise ValueError("published benchmark statistics do not match the correctness vectors")
		print("Verified all seven cells and 21 paired comparisons from the published snapshot.")
		return
	data = collect(args.source)
	metrics, data["pairs"] = summarize(data["cells"], data["item_ids"])
	for name, values in metrics.items():
		data["cells"][name].update(values)
	args.output.write_text(json.dumps(data, indent="\t") + "\n")
	args.output.with_suffix(".md").write_text(markdown(data))
	print(f"Wrote {len(data['cells'])} cells and {len(data['pairs'])} comparisons.")


if __name__ == "__main__":
	main()
