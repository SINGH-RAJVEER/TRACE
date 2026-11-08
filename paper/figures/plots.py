"""Draw the paper's figures from figures/v7.json and figures/students.json.

Run from the repository root:
uv run --no-project --with matplotlib python paper/figures/plots.py
"""

import json
import pathlib

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
DATA = json.loads((HERE / "v7.json").read_text())

# Reference categorical palette, light mode, slots 1 to 4 in fixed order.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8984"
DOMAINS = [("gsm8k", "GSM8K"), ("coding", "Coding"), ("squad", "SQuAD"), ("ag_news", "AG News")]
HATCH = ["", "//", "\\\\", ".."]
ARM_LABEL = {"full": "Full", "instantaneous": "Instantaneous", "quota": "Quota"}

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
		"hatch.linewidth": 0.4,
		"pdf.fonttype": 42,
	}
)


def style(ax, axis="x"):
	ax.grid(axis=axis, color=GRID, linewidth=0.6)
	ax.set_axisbelow(True)
	ax.tick_params(length=0)


def released_by_domain():
	cells = [
		("A1", "quota"),
		("A2", "full"),
		("A2", "instantaneous"),
		("A2", "quota"),
		("A3", "quota"),
		("A5", "full"),
		("A5", "instantaneous"),
		("A5", "quota"),
		("A6", "quota"),
		("A7", "quota"),
	]
	labels = []
	for attack, arm in cells:
		name = "unprotected" if attack == "A6" else ARM_LABEL[arm].lower()
		labels.append(f"{attack} {name}")
	fig, ax = plt.subplots(figsize=(3.5, 2.9))
	y = list(range(len(cells)))[::-1]
	left = [0.0] * len(cells)
	for slot, (key, label) in enumerate(DOMAINS):
		widths = [
			DATA["campaigns"][a][arm]["by_domain"].get(key, {"tokens": 0})["tokens"] / 1000
			for a, arm in cells
		]
		ax.barh(
			y,
			widths,
			left=left,
			height=0.62,
			color=SLOT[slot],
			hatch=HATCH[slot],
			edgecolor="white",
			linewidth=0.8,
			label=label,
		)
		left = [l + w for l, w in zip(left, widths)]
	for yi, total in zip(y, left):
		ax.text(total + 12, yi, f"{total:,.0f}k", va="center", fontsize=7, color=INK)
	ax.set_yticks(y, labels)
	ax.set_xlabel("Released output tokens (thousands)")
	ax.set_xlim(0, 960)
	style(ax)
	ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False, fontsize=7, handlelength=1.4, columnspacing=0.9)
	fig.tight_layout(pad=0.3)
	fig.savefig(HERE / "released-tokens.pdf")


def a5_curves():
	fig, ax = plt.subplots(figsize=(3.5, 2.7))
	series = [
		("A6", "quota", "A6, no control fired", NEUTRAL, "-"),
		("A5", "quota", "A5, quota", SLOT[2], "-"),
		("A5", "full", "A5, full policy", SLOT[0], "-"),
		("A5", "instantaneous", "A5, instantaneous", SLOT[1], "--"),
	]
	for attack, arm, label, color, dash in series:
		curve = DATA["campaigns"][attack][arm]["release_curve"]
		xs, ys, total = [0], [0], 0
		for attempt, tokens in curve:
			total += tokens
			xs.append(attempt)
			ys.append(total / 1000)
		if xs[-1] < 6000:
			xs.append(6000)
			ys.append(ys[-1])
		ax.plot(xs, ys, color=color, linestyle=dash, linewidth=1.4, label=f"{label}, {ys[-1]:,.0f}k")
	for attack, arm, color in (("A5", "full", SLOT[0]), ("A5", "instantaneous", SLOT[1])):
		j = DATA["campaigns"][attack][arm]["first_L4_decision"]
		ax.axvline(j, color=color, linewidth=0.7, linestyle=":")
		ax.text(j + 60, 420, f"L4 at {j:,}", rotation=90, fontsize=6.5, color=INK, va="bottom")
	ax.set_xlim(0, 6000)
	ax.set_ylim(0, 850)
	ax.set_xlabel("Query attempt, proof submissions excluded")
	ax.set_ylabel("Released tokens (thousands)")
	ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, frameon=False, fontsize=6.5, handlelength=2.2)
	style(ax, "y")
	fig.tight_layout(pad=0.3)
	fig.savefig(HERE / "a5-release.pdf")


def a5_ranks():
	ranks = DATA["windows"]["A5/full"]
	order = [
		("v7-A5-gsm8k-000", "GSM8K"),
		("v7-A5-squad-000", "SQuAD"),
		("v7-A5-coding-000", "Coding 1"),
		("v7-A5-coding-001", "Coding 2"),
		("v7-A5-ag_news-000", "AG News 1"),
		("v7-A5-ag_news-001", "AG News 2"),
	]
	fig, ax = plt.subplots(figsize=(3.5, 2.0))
	for i, (account, label) in enumerate(order):
		values = ranks[account]
		color = SLOT[0] if "gsm8k" in account else NEUTRAL
		ax.scatter(values, [i] * len(values), s=14, color=color, edgecolor="white", linewidth=0.5, zorder=3)
	ax.axvline(0.3679, color=INK, linewidth=0.7, linestyle="--")
	ax.text(0.3679, -0.75, r"$p=e^{-1}$: windows to the left add risk", fontsize=6.5, color=INK, ha="center", va="bottom")
	ax.set_yticks(range(len(order)), [label for _, label in order])
	ax.set_ylim(len(order) - 0.5, -1.0)
	ax.set_xlim(0, 1)
	ax.set_xlabel("Window rank $p$, full policy")
	style(ax)
	fig.tight_layout(pad=0.3)
	fig.savefig(HERE / "a5-ranks.pdf")


def students():
	data = json.loads((HERE / "students.json").read_text())
	rows = [
		("teacher", "Teacher", NEUTRAL),
		("A6/unprotected end", "A6, no control, 1,473", SLOT[1]),
		("A5/full end", "A5, full policy, 197", SLOT[0]),
		("base", "Untrained base", NEUTRAL),
	]
	fig, ax = plt.subplots(figsize=(3.5, 1.7))
	for y, (key, _, color) in enumerate(rows):
		cell = data["cells"][key]
		low, high = (100 * v for v in cell["wilson95"])
		acc = 100 * cell["accuracy"]
		ax.plot([low, high], [y, y], color=color, linewidth=1.6, solid_capstyle="butt", zorder=2)
		ax.scatter([acc], [y], s=22, color=color, edgecolor="white", linewidth=0.6, zorder=3)
		ax.text(high + 0.8, y, f"{acc:.1f}%", va="center", fontsize=7, color=INK)
	target = 100 * data["target_accuracy"]
	ax.axvline(target, color=INK, linewidth=0.7, linestyle="--")
	ax.text(target, -0.8, f"target {target:.0f}%", fontsize=6.5, color=INK, ha="center", va="bottom")
	ax.set_yticks(range(len(rows)), [label for _, label, _ in rows])
	ax.set_ylim(len(rows) - 0.5, -1.0)
	ax.set_xlim(45, 95)
	ax.set_xlabel("Accuracy on 746 GSM8K items (%)")
	style(ax)
	fig.tight_layout(pad=0.3)
	fig.savefig(HERE / "students.pdf")


if __name__ == "__main__":
	released_by_domain()
	a5_curves()
	a5_ranks()
	students()
