"""Draw the figures of "Too late to matter" and write the numbers its text quotes.

Run from the repository root:
uv run --no-project --with matplotlib python paper/too-late-to-matter/figures/plots.py

Inputs are the shared v7 records in paper/figures: v7.json (campaign outcomes, from
readings/v7) and student-panel.json (the validated student benchmark panel). Outputs are
written next to this file: race.pdf, supervision.pdf, paired.pdf, tokens.pdf and
numbers.json, plus a copy of the shared a5-ranks.pdf, so that this directory holds every
file main.tex needs and can be uploaded to Overleaf as it is.
"""

import json
import math
import pathlib
import shutil

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
SHARED = HERE.parents[1] / "figures"
V7 = json.loads((SHARED / "v7.json").read_text())
PANEL = json.loads((SHARED / "student-panel.json").read_text())

# Reference categorical palette, light mode. Slots 1 and 2 encode the detector policy;
# the unprotected run and the reference models use neutral ink.
FULL = "#2a78d6"
INST = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
NEUTRAL = "#8a8984"
GRID = "#e4e3df"
SHADE = "#f1f0ec"

plt.rcParams.update(
	{
		"font.family": "serif",
		"font.size": 8,
		"axes.edgecolor": MUTED,
		"axes.labelcolor": INK,
		"xtick.color": MUTED,
		"ytick.color": MUTED,
		"axes.spines.top": False,
		"axes.spines.right": False,
		"xtick.labelsize": 7,
		"ytick.labelsize": 7,
		"pdf.fonttype": 42,
	}
)

STUDENTS = ["A2/instantaneous", "A5/instantaneous", "A2/full", "A5/full", "unprotected"]
LABEL = {
	"A2/instantaneous": "A2, instantaneous",
	"A5/instantaneous": "A5, instantaneous",
	"A2/full": "A2, full",
	"A5/full": "A5, full",
	"unprotected": "Unprotected",
}
COLOR = {
	"A2/instantaneous": INST,
	"A5/instantaneous": INST,
	"A2/full": FULL,
	"A5/full": FULL,
	"unprotected": MUTED,
}
SHORT = {
	"A2/instantaneous": "A2 inst.",
	"A5/instantaneous": "A5 inst.",
	"A2/full": "A2 full",
	"A5/full": "A5 full",
	"unprotected": "Unprotected",
}
MARKER = {"A2": "s", "A5": "o", "unprotected": "D"}


def style(ax, axis="x"):
	ax.grid(axis=axis, color=GRID, linewidth=0.6)
	ax.set_axisbelow(True)
	ax.tick_params(length=0)


def cell(name):
	return PANEL["cells"][name]


def pair(a, b):
	key = f"{a} minus {b}"
	if key in PANEL["pairs"]:
		p = PANEL["pairs"][key]
		return p["difference"], p["paired_item_bootstrap95"], p["mcnemar_exact_p"], p["mcnemar_holm_p"]
	p = PANEL["pairs"][f"{b} minus {a}"]
	low, high = p["paired_item_bootstrap95"]
	return -p["difference"], [-high, -low], p["mcnemar_exact_p"], p["mcnemar_holm_p"]


def math_released(attack, arm):
	"""GSM8K responses the ledger records as released to the campaign."""
	return V7["campaigns"][attack][arm]["by_domain"].get("gsm8k", {"responses": 0})["responses"]


def numbers():
	base = cell("base")["accuracy"]
	teacher = cell("teacher")["accuracy"]
	unprotected = cell("unprotected")
	gain_u = unprotected["accuracy"] - base
	rows = {}
	for name in STUDENTS:
		c = cell(name)
		n = c["training_examples"]
		row = {
			"training_examples": n,
			"correct": c["correct"],
			"accuracy": c["accuracy"],
			"wilson95": c["wilson95"],
			"gain_over_base": c["accuracy"] - base,
			"share_of_base_teacher_gap": (c["accuracy"] - base) / (teacher - base),
			"share_of_unprotected_supervision": n / unprotected["training_examples"],
			"share_of_unprotected_gain": (c["accuracy"] - base) / gain_u,
			# The recipe's effective batch is 16 and it runs three epochs at every size.
			"optimizer_steps_by_recipe": 3 * math.ceil(n / 16),
		}
		if name != "unprotected":
			d, ci, p, holm = pair(name, "unprotected")
			row["minus_unprotected"] = {"difference": d, "bootstrap95": ci, "mcnemar_p": p, "holm_p": holm}
		rows[name] = row
	d, ci, p, holm = pair("A5/full", "A5/instantaneous")
	tokens_u = math_released_tokens("A6", "quota")
	released = {
		f"{a}/{arm}": {
			"gsm8k_responses": math_released(a, arm),
			"gsm8k_tokens": math_released_tokens(a, arm),
			"share_of_unprotected_gsm8k_tokens": math_released_tokens(a, arm) / tokens_u,
			"released_tokens": V7["campaigns"][a][arm]["released_tokens"],
		}
		for a, arms in V7["campaigns"].items()
		for arm, c in arms.items()
		if "by_domain" in c and a != "A7"
	}
	return {
		"source": ["paper/figures/v7.json", "paper/figures/student-panel.json"],
		"base_accuracy": base,
		"teacher_accuracy": teacher,
		"students": rows,
		"A5_full_minus_A5_instantaneous": {"difference": d, "bootstrap95": ci, "mcnemar_p": p, "holm_p": holm},
		"released": released,
	}


def math_released_tokens(attack, arm):
	return V7["campaigns"][attack][arm]["by_domain"].get("gsm8k", {"tokens": 0})["tokens"]


def race():
	"""Student accuracy and control interventions on one axis of GSM8K answers."""
	fig, (top, bottom) = plt.subplots(
		2,
		1,
		figsize=(7.16, 3.7),
		sharex=True,
		gridspec_kw={"height_ratios": [3.0, 1.55], "hspace": 0.08},
	)
	xmin, xmax = 25, 2600
	for ax in (top, bottom):
		ax.axvspan(xmin, 95, color=SHADE, linewidth=0, zorder=0)
		ax.set_xscale("log")
		ax.set_xlim(xmin, xmax)

	base = 100 * cell("base")["accuracy"]
	teacher = 100 * cell("teacher")["accuracy"]
	top.axhline(teacher, color=NEUTRAL, linewidth=1.0)
	top.axhline(base, color=NEUTRAL, linewidth=1.0, linestyle="--")
	top.text(xmax * 0.97, teacher + 0.8, f"Teacher, {teacher:.1f}%", ha="right", va="bottom", fontsize=7, color=INK)
	top.text(xmax * 0.97, base + 0.8, f"Untrained student, {base:.1f}%", ha="right", va="bottom", fontsize=7, color=INK)
	top.text(
		math.sqrt(xmin * 95),
		61.5,
		"No student\ntrained\nbelow 95",
		ha="center",
		va="center",
		fontsize=6.5,
		color=MUTED,
	)

	# A2 and A5 students at almost the same count are offset by 5% for legibility.
	# Label placement per student: x factor, label y (None keeps the point's), alignment.
	PLACE = {
		"A2/instantaneous": (0.9, None, "right", "center"),
		"A5/instantaneous": (1.0, 77.6, "center", "bottom"),
		"A2/full": (1.0, 66.4, "center", "top"),
		"A5/full": (1.1, None, "left", "center"),
		"unprotected": (0.9, None, "right", "center"),
	}
	dodge = {"A2": 0.95, "A5": 1.05, "unprotected": 1.0}
	for name in STUDENTS:
		c = cell(name)
		attack = name.split("/")[0]
		x = c["training_examples"] * dodge[attack]
		acc = 100 * c["accuracy"]
		low, high = (100 * v for v in c["wilson95"])
		top.plot([x, x], [low, high], color=COLOR[name], linewidth=1.2, solid_capstyle="butt", zorder=2)
		top.scatter(
			[x],
			[acc],
			s=34,
			marker=MARKER[attack],
			color=COLOR[name],
			edgecolor="white",
			linewidth=0.8,
			zorder=3,
		)
		dx, y_text, ha, va = PLACE[name]
		top.text(
			x * dx,
			acc if y_text is None else y_text,
			f"{LABEL[name]}, {c['training_examples']:,}: {acc:.1f}%",
			ha=ha,
			va=va,
			fontsize=6.5,
			color=INK,
		)
	top.set_ylim(45, 90)
	top.set_ylabel("GSM8K accuracy (%)")
	style(top, "y")

	# Each row is one control; x is the GSM8K answers released before it refused math.
	rows = [
		("Instantaneous", INST, [(math_released("A2", "instantaneous"), "A2, A5 refuse", True)]),
		(
			"Full",
			FULL,
			[(99, "alert", False), (math_released("A5", "full"), "A2, A5 refuse", True)],
		),
		(
			"Quota",
			MUTED,
			[
				(math_released("A1", "quota"), "", False),
				(math_released("A3", "quota"), "A1, A3", False),
				(math_released("A5", "quota"), "A5", False),
			],
		),
		("No refusal", NEUTRAL, [(math_released("A6", "quota"), "Unprotected; A1, A6 detectors", True)]),
	]
	for y, (label, color, marks) in enumerate(rows):
		for x, text, trained in marks:
			bottom.scatter(
				[x],
				[y],
				s=30,
				marker="v",
				color=color if trained else "white",
				edgecolor=color,
				linewidth=1.1,
				zorder=3,
			)
			if text:
				bottom.text(x, y - 0.38, text, ha="center", va="bottom", fontsize=6, color=INK)
	bottom.set_yticks(range(len(rows)), [r[0] for r in rows])
	bottom.set_ylim(len(rows) - 0.45, -0.75)
	bottom.set_xlabel("GSM8K answers released to the campaign (log scale)")
	bottom.set_xticks([30, 50, 100, 200, 500, 1000, 2000], ["30", "50", "100", "200", "500", "1,000", "2,000"])
	bottom.minorticks_off()
	style(bottom, "x")
	fig.subplots_adjust(left=0.11, right=0.985, top=0.98, bottom=0.12)
	fig.savefig(HERE / "race.pdf")


def supervision():
	"""Share of the unprotected run's math supervision released, beside share of its gain kept."""
	data = numbers()["students"]
	names = STUDENTS
	fig, (left, right) = plt.subplots(1, 2, figsize=(3.5, 1.95), sharey=True, gridspec_kw={"wspace": 0.16})
	y = list(range(len(names)))
	for ax, key, title in (
		(left, "share_of_unprotected_supervision", "Answers released (%)"),
		(right, "share_of_unprotected_gain", "Gain kept (%)"),
	):
		values = [100 * data[n][key] for n in names]
		colors = [COLOR[n] for n in names]
		ax.barh(y, values, height=0.58, color=colors, edgecolor="white", linewidth=0.8)
		for yi, v in zip(y, values):
			inside = v > 60
			ax.text(
				v - 3 if inside else v + 3,
				yi,
				f"{v:.0f}%",
				va="center",
				ha="right" if inside else "left",
				fontsize=6.5,
				color="white" if inside else INK,
			)
		ax.set_xlim(0, 100)
		ax.set_xticks([0, 50, 100], ["0", "50", "100"])
		ax.set_title(title, fontsize=7.5, color=INK, loc="left")
		style(ax, "x")
	left.set_yticks(y, [f"{SHORT[n]}, {data[n]['training_examples']:,}" for n in names])
	left.invert_yaxis()
	fig.text(0.98, 0.01, "Both as a share of the unprotected student", ha="right", va="bottom", fontsize=6.5, color=MUTED)
	fig.subplots_adjust(left=0.31, right=0.97, top=0.88, bottom=0.17)
	fig.savefig(HERE / "supervision.pdf")


def paired():
	"""Paired accuracy differences with item-bootstrap intervals and McNemar p-values."""
	rows = [(n, "unprotected", f"{SHORT[n]} $-$ unprot.", COLOR[n]) for n in STUDENTS[:-1]]
	rows.append(("A5/full", "A5/instantaneous", "A5 full $-$ A5 inst.", MUTED))
	fig, ax = plt.subplots(figsize=(3.5, 1.95))
	for y, (a, b, label, color) in enumerate(rows):
		d, (low, high), p, holm = pair(a, b)
		y_draw = y + (0.4 if y == len(rows) - 1 else 0)
		ax.plot([100 * low, 100 * high], [y_draw, y_draw], color=color, linewidth=1.4, solid_capstyle="butt")
		ax.scatter([100 * d], [y_draw], s=24, color=color, edgecolor="white", linewidth=0.6, zorder=3)
		ax.text(4.2, y_draw, f"{p:.3f}  {holm:.2f}", va="center", ha="left", fontsize=6.5, color=INK)
	ax.text(4.2, -0.95, "$p$     Holm", va="bottom", ha="left", fontsize=6.5, color=MUTED)
	ax.axhline(len(rows) - 1.3, color=GRID, linewidth=0.6)
	ax.axvline(0, color=INK, linewidth=0.7)
	ax.set_yticks([y + (0.4 if y == len(rows) - 1 else 0) for y in range(len(rows))], [r[2] for r in rows])
	ax.set_ylim(len(rows) - 0.2, -1.1)
	ax.set_xlim(-8, 4)
	ax.set_xticks([-8, -6, -4, -2, 0, 2, 4])
	ax.set_xlabel("Accuracy difference, percentage points")
	ax.spines["left"].set_visible(False)
	style(ax, "x")
	fig.subplots_adjust(left=0.31, right=0.8, top=0.97, bottom=0.2)
	fig.savefig(HERE / "paired.pdf")


DOMAINS = [("gsm8k", "GSM8K"), ("coding", "Coding"), ("squad", "SQuAD"), ("ag_news", "AG News")]
DOMAIN_COLOR = [FULL, INST, "#1baf7a", "#eda100"]
HATCH = ["", "//", "\\\\", ".."]


def tokens():
	"""Released output tokens by domain for every arm that ran. The A6 quota arm is the unprotected run."""
	cells = [
		("A1", "quota", "A1 quota"),
		("A2", "full", "A2 full"),
		("A2", "instantaneous", "A2 instantaneous"),
		("A2", "quota", "A2 quota"),
		("A3", "quota", "A3 quota"),
		("A5", "full", "A5 full"),
		("A5", "instantaneous", "A5 instantaneous"),
		("A5", "quota", "A5 quota"),
		("A6", "quota", "Unprotected"),
		("A7", "quota", "A7 quota"),
	]
	fig, ax = plt.subplots(figsize=(3.5, 2.9))
	y = list(range(len(cells)))[::-1]
	left = [0.0] * len(cells)
	for slot, (key, label) in enumerate(DOMAINS):
		widths = [V7["campaigns"][a][arm]["by_domain"].get(key, {"tokens": 0})["tokens"] / 1000 for a, arm, _ in cells]
		ax.barh(y, widths, left=left, height=0.62, color=DOMAIN_COLOR[slot], hatch=HATCH[slot], edgecolor="white", linewidth=0.8, label=label)
		left = [l + w for l, w in zip(left, widths)]
	for yi, total in zip(y, left):
		ax.text(total + 12, yi, f"{total:,.0f}k", va="center", fontsize=7, color=INK)
	ax.set_yticks(y, [c[2] for c in cells])
	ax.set_xlabel("Released output tokens (thousands)")
	ax.set_xlim(0, 960)
	style(ax)
	ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False, fontsize=7, handlelength=1.4, columnspacing=0.9)
	fig.tight_layout(pad=0.3)
	fig.savefig(HERE / "tokens.pdf")


if __name__ == "__main__":
	(HERE / "numbers.json").write_text(json.dumps(numbers(), indent="\t") + "\n")
	race()
	supervision()
	paired()
	tokens()
	shutil.copyfile(SHARED / "a5-ranks.pdf", HERE / "a5-ranks.pdf")
