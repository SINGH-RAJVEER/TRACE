import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = Path(__file__).parent
rep = json.loads((here / "aggregate.json").read_text())["models"]
fig, ax = plt.subplots(figsize=(8, 4.2))
x = range(len(rep))
ax.errorbar(x, [r["mean_accuracy"] for r in rep], yerr=[[r["mean_accuracy"] - r["family_ci"][0] for r in rep], [r["family_ci"][1] - r["mean_accuracy"] for r in rep]], fmt="o", capsize=4, color="#1f77b4", label="mean of 3 passes, question-family 95% CI")
for i, r in enumerate(rep):
	ax.scatter([i] * 3, r["run_accuracies"], marker="_", s=300, color="#d62728", zorder=3, label="individual passes" if i == 0 else None)
ax.set_xticks(list(x)); ax.set_xticklabels([r["model"] for r in rep], rotation=25, ha="right")
ax.set_ylabel("GSM8K accuracy (746 items)"); ax.set_title("Three fresh inference passes per fixed model")
ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(here / "repeat-accuracy.png", dpi=160)
