"""Draw the figures of the per-account detection paper and write the numbers its text quotes.

Run from the repository root:
uv run --no-project --with matplotlib python paper/closing-the-tap/figures/plots.py

Inputs are the shared v7 records in paper/figures: v7.json (campaign outcomes, from
readings/v7) and student-panel.json (the validated student benchmark panel). Outputs are
written next to this file: release.pdf, ranks.pdf, tokens.pdf, exposure.pdf,
supervision.pdf, paired.pdf and numbers.json, so that this directory holds every file
main.tex needs and can be uploaded to Overleaf as it is.

The paper reports the sequential (A2) and task-partitioned (A5) campaigns. Its unprotected
reference is the release ledger of the A6 quota arm: A6 sends the same 6,000 prompts split
across six accounts at random, no account reached the quota, and so no control acted and
every response was released. The paper states this provenance in a footnote.
"""

import json
import math
import pathlib

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
SHARED = HERE.parents[1] / "figures"
V7 = json.loads((SHARED / "v7.json").read_text())
PANEL = json.loads((SHARED / "student-panel.json").read_text())

CAMPAIGNS = ["A2", "A5"]
ARMS = ["full", "instantaneous", "quota"]
REFERENCE = ("A6", "quota")

# Reference categorical palette, light mode. Slots 1 and 2 encode the detector policy;
# the quota, the unprotected reference and the reference models use neutral ink.
FULL = "#2a78d6"
INST = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
NEUTRAL = "#8a8984"
GRID = "#e4e3df"
SHADE = "#f1f0ec"
ARM_COLOR = {"full": FULL, "instantaneous": INST, "quota": MUTED}
ARM_DASH = {"full": "-", "instantaneous": "--", "quota": "-"}
ARM_LABEL = {"full": "Full policy", "instantaneous": "Instantaneous policy", "quota": "Quota"}

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
		"hatch.linewidth": 0.4,
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


def campaign(attack, arm):
	return V7["campaigns"][attack][arm]


def reference():
	return campaign(*REFERENCE)


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


def domain(c, key, field):
	return c["by_domain"].get(key, {"responses": 0, "tokens": 0})[field]


def risk_path(ranks, decay=0.9, offset=1.0):
	"""Full-policy risk after each window, by the decayed cumulative sum of Eq. (2)."""
	r, path = 0.0, []
	for p in ranks:
		r = max(0.0, decay * r - math.log(p) - offset)
		path.append(r)
	return path


def detection():
	"""Window ranks and risk for the math windows that escalated each campaign."""
	full_thresholds = V7["thresholds"]["paper"]
	inst_thresholds = V7["thresholds"]["instantaneous"]
	accounts = {"A2": "v7-A2-account", "A5": "v7-A5-gsm8k-000"}
	out = {}
	for attack, account in accounts.items():
		ranks = V7["windows"][f"{attack}/full"][account]
		# A2's account sends 15 AG News windows before its GSM8K block; A5's account sends only GSM8K.
		math_ranks = ranks[-2:]
		other = [
			p
			for name, values in V7["windows"][f"{attack}/full"].items()
			for i, p in enumerate(values)
			if not (name == account and i >= len(values) - 2)
		]
		path = risk_path(ranks)
		out[attack] = {
			"math_window_ranks": math_ranks,
			"full_risk_after_math_windows": path[-2:],
			"instantaneous_score_first_math_window": -math.log(math_ranks[0]),
			"smallest_rank_outside_math_windows": min(other),
			"windows_outside_math": len(other),
		}
	out["full_thresholds"] = full_thresholds
	out["instantaneous_thresholds"] = inst_thresholds
	out["single_window_full_L4_rank"] = math.exp(-(full_thresholds[3] + 1.0))
	out["instantaneous_L4_rank"] = math.exp(-inst_thresholds[3])
	out["rank_floor"] = V7["adequacy"]["empirical_p_value_floor"]
	return out


def released():
	ref = reference()
	ref_math = domain(ref, "gsm8k", "responses")
	ref_math_tokens = domain(ref, "gsm8k", "tokens")
	rows = {}
	for attack in CAMPAIGNS:
		for arm in ARMS:
			c = campaign(attack, arm)
			rows[f"{attack}/{arm}"] = {
				"attempts": c["attempts"],
				"received": c["received"],
				"blocked": c["blocked"],
				"challenge": c["challenge"],
				"first_refusal_decision": c["first_L4_decision"],
				"released_tokens": c["released_tokens"],
				"share_of_reference_tokens": c["released_tokens"] / ref["released_tokens"],
				"gsm8k_responses": domain(c, "gsm8k", "responses"),
				"share_of_reference_gsm8k_responses": domain(c, "gsm8k", "responses") / ref_math,
				"gsm8k_tokens": domain(c, "gsm8k", "tokens"),
				"share_of_reference_gsm8k_tokens": domain(c, "gsm8k", "tokens") / ref_math_tokens,
				"by_domain": c["by_domain"],
			}
	rows["reference"] = {
		"released_tokens": ref["released_tokens"],
		"gsm8k_responses": ref_math,
		"gsm8k_tokens": ref_math_tokens,
		"gsm8k_share_of_tokens": ref_math_tokens / ref["released_tokens"],
		"by_domain": ref["by_domain"],
	}
	return rows


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
	gate = campaign("A2", "full")["gate"]
	return {
		"source": ["paper/figures/v7.json", "paper/figures/student-panel.json"],
		"reference_ledger": f"{REFERENCE[0]}/{REFERENCE[1]}",
		"detection": detection(),
		"released": released(),
		"benign_replay_peaks": {"full": gate["paper"]["benign_peaks"], "instantaneous": gate["instantaneous"]["benign_peaks"]},
		"utility": V7["utility"],
		"base_accuracy": base,
		"teacher_accuracy": teacher,
		"students": rows,
		"A5_full_minus_A5_instantaneous": {"difference": d, "bootstrap95": ci, "mcnemar_p": p, "holm_p": holm},
	}


def release():
	"""Cumulative released tokens over each campaign's attempts, per arm."""
	fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45), sharey=True, gridspec_kw={"wspace": 0.08})
	total = reference()["released_tokens"]
	titles = {
		"A2": "A2: one account, one task block at a time",
		"A5": "A5: six accounts, one task each",
	}
	for ax, attack in zip(axes, CAMPAIGNS):
		for arm in ARMS:
			c = campaign(attack, arm)
			xs, ys, run = [], [], 0
			for attempt, tokens in c["release_curve"]:
				run += tokens
				xs.append(attempt)
				ys.append(run)
			if xs[-1] < 6000:
				xs.append(6000)
				ys.append(ys[-1])
			ax.plot(
				xs,
				ys,
				color=ARM_COLOR[arm],
				linestyle=ARM_DASH[arm],
				linewidth=1.3,
				label=f"{ARM_LABEL[arm]}, {ys[-1] / 1000:,.0f}k",
			)
			j = c["first_L4_decision"]
			ax.axvline(j, color=ARM_COLOR[arm], linewidth=0.6, linestyle=":")
			if attack == "A2":
				# A2's flat tails leave no clear corner for a legend, so its lines are labelled directly.
				factor, va = (0.85, "top") if arm == "instantaneous" else (1.15, "bottom")
				ax.text(5900, ys[-1] * factor, f"{ARM_LABEL[arm]}, {ys[-1] / 1000:,.0f}k", fontsize=6.5, color=INK, ha="right", va=va)
		ax.axhline(total, color=NEUTRAL, linewidth=1.0)
		ax.text(150, total * 1.18, f"No control acted, {total / 1000:,.0f}k", fontsize=6.5, color=INK, va="bottom")
		if attack == "A2":
			for x0, x1, name in ((0, 1500, "AG News"), (1500, 3000, "GSM8K"), (3000, 6000, "SQuAD, coding")):
				ax.text((x0 + x1) / 2, 140, name, fontsize=6.3, color=MUTED, ha="center", va="bottom")
				if x0:
					ax.axvline(x0, color=GRID, linewidth=0.8, zorder=0)
		ax.set_yscale("log")
		ax.set_ylim(100, 2.4e6)
		ax.set_xlim(0, 6000)
		ax.set_xticks([0, 1500, 3000, 4500, 6000], ["0", "1,500", "3,000", "4,500", "6,000"])
		ax.set_yticks([1e2, 1e3, 1e4, 1e5, 1e6], ["100", "1k", "10k", "100k", "1M"])
		ax.minorticks_off()
		ax.set_title(titles[attack], fontsize=7.5, color=INK, loc="left")
		ax.set_xlabel("Campaign attempt, proof submissions excluded")
		style(ax, "y")
	axes[1].legend(loc="lower right", bbox_to_anchor=(1.0, 0.1), frameon=False, fontsize=6.5, handlelength=2.2)
	axes[0].set_ylabel("Released tokens (log scale)")
	fig.subplots_adjust(left=0.07, right=0.975, top=0.9, bottom=0.17)
	fig.savefig(HERE / "release.pdf")


def ranks():
	"""Window ranks of every account in both campaigns under the full policy."""
	a2 = V7["windows"]["A2/full"]["v7-A2-account"]
	a5 = V7["windows"]["A5/full"]
	rows = [
		("A2, AG News block", a2[:-2], NEUTRAL),
		("A2, GSM8K block", a2[-2:], FULL),
		("A5, GSM8K", a5["v7-A5-gsm8k-000"], FULL),
		("A5, SQuAD", a5["v7-A5-squad-000"], NEUTRAL),
		("A5, coding 1", a5["v7-A5-coding-000"], NEUTRAL),
		("A5, coding 2", a5["v7-A5-coding-001"], NEUTRAL),
		("A5, AG News 1", a5["v7-A5-ag_news-000"], NEUTRAL),
		("A5, AG News 2", a5["v7-A5-ag_news-001"], NEUTRAL),
	]
	fig, ax = plt.subplots(figsize=(3.5, 2.35))
	for i, (_, values, color) in enumerate(rows):
		y = i + (0.4 if i >= 2 else 0)
		ax.scatter(values, [y] * len(values), s=14, color=color, edgecolor="white", linewidth=0.5, zorder=3)
	inst = math.exp(-V7["thresholds"]["instantaneous"][3])
	ax.axvline(math.exp(-1), color=INK, linewidth=0.7, linestyle="--")
	ax.axvline(inst, color=INST, linewidth=0.7, linestyle=":")
	ax.text(math.exp(-1) * 0.9, -0.85, "adds risk", fontsize=6.5, color=INK, ha="right", va="bottom")
	ax.text(inst * 0.9, -0.85, "refused at once,\ninstantaneous", fontsize=6.5, color=INK, ha="right", va="bottom")
	ax.set_xscale("log")
	ax.set_xlim(0.004, 1.05)
	ax.set_xticks([0.01, 0.03, 0.1, 0.37, 1], ["0.01", "0.03", "0.1", "0.37", "1"])
	ax.minorticks_off()
	ax.set_yticks([i + (0.4 if i >= 2 else 0) for i in range(len(rows))], [r[0] for r in rows])
	ax.set_ylim(len(rows) - 0.1, -1.05)
	ax.set_xlabel("Window rank $p$ (log scale), full policy")
	style(ax)
	fig.tight_layout(pad=0.3)
	fig.savefig(HERE / "ranks.pdf")


DOMAINS = [("gsm8k", "GSM8K"), ("coding", "Coding"), ("squad", "SQuAD"), ("ag_news", "AG News")]
DOMAIN_COLOR = [FULL, INST, "#1baf7a", "#eda100"]
HATCH = ["", "//", "\\\\", ".."]


def tokens():
	"""Released output tokens by domain for every arm of both campaigns and the reference."""
	cells = [(a, arm, f"{a} {'inst.' if arm == 'instantaneous' else arm}") for a in CAMPAIGNS for arm in ARMS]
	cells.append((*REFERENCE, "No control"))
	fig, ax = plt.subplots(figsize=(3.5, 2.35))
	y = list(range(len(cells)))[::-1]
	left = [0.0] * len(cells)
	for slot, (key, label) in enumerate(DOMAINS):
		widths = [domain(campaign(a, arm), key, "tokens") / 1000 for a, arm, _ in cells]
		ax.barh(
			y,
			widths,
			left=left,
			height=0.62,
			color=DOMAIN_COLOR[slot],
			hatch=HATCH[slot],
			edgecolor="white",
			linewidth=0.8,
			label=label,
		)
		left = [l + w for l, w in zip(left, widths)]
	for yi, total in zip(y, left):
		ax.text(total + 12, yi, f"{total:,.0f}k", va="center", fontsize=7, color=INK)
	ax.set_yticks(y, [c[2] for c in cells])
	ax.set_xlabel("Released output tokens (thousands)")
	ax.set_xlim(0, 960)
	style(ax)
	ax.legend(
		ncol=4,
		loc="lower center",
		bbox_to_anchor=(0.4, 1.0),
		frameon=False,
		fontsize=7,
		handlelength=1.4,
		columnspacing=0.9,
	)
	fig.tight_layout(pad=0.3)
	fig.savefig(HERE / "tokens.pdf")


def exposure():
	"""Student accuracy and control interventions on one axis of GSM8K answers."""
	fig, (top, bottom) = plt.subplots(
		2,
		1,
		figsize=(7.16, 3.5),
		sharex=True,
		gridspec_kw={"height_ratios": [3.0, 1.3], "hspace": 0.08},
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
	top.text(math.sqrt(xmin * 95), 61.5, "No student\ntrained\nbelow 95", ha="center", va="center", fontsize=6.5, color=MUTED)

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
		top.scatter([x], [acc], s=34, marker=MARKER[attack], color=COLOR[name], edgecolor="white", linewidth=0.8, zorder=3)
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
	math_released = lambda attack, arm: domain(campaign(attack, arm), "gsm8k", "responses")
	rows = [
		("Instantaneous", INST, [(math_released("A2", "instantaneous"), "A2, A5 refused", True)]),
		("Full", FULL, [(99, "first escalation", False), (math_released("A5", "full"), "A2, A5 refused", True)]),
		("Quota", MUTED, [(math_released("A5", "quota"), "A5 refused", False)]),
		("No control", NEUTRAL, [(domain(reference(), "gsm8k", "responses"), "Unprotected", True)]),
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
	fig.savefig(HERE / "exposure.pdf")


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


if __name__ == "__main__":
	(HERE / "numbers.json").write_text(json.dumps(numbers(), indent="\t") + "\n")
	release()
	ranks()
	tokens()
	exposure()
	supervision()
	paired()
