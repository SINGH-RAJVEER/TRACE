"""Read-only verification of the repeat batch and separate aggregate report. Never writes into the batch directory."""
import hashlib, json, random, statistics, sys, collections

def boot(g, n, seed):
	rng = random.Random(seed)
	out = []
	for _ in range(n):
		sel = rng.choices(g, k=len(g))
		out.append(sum(sum(s) for s in sel) / sum(len(s) for s in sel))
	return sorted(out)

from pathlib import Path

OUT = Path("/home/rajveer/Projects/TRACE/artifacts/students/v7-repeats-20261109")
REP = Path(__file__).parent
sys.path.insert(0, str(OUT / "runtime"))
from evaluate import bind, result_path, existing, digest, load  # frozen evaluator helpers
from distillation.learning_curve import math_family

LABEL = {"A6/unprotected": "unprotected"}
plan, items = load(OUT)
contract = bind(plan)
summary = json.loads((OUT / "summary.json").read_text())
checks = {}
checks["summary_contract_matches_plan"] = summary["contract"] == contract
checks["summary_state_complete"] = summary["state"] == "complete"
checks["plan_items_746"] = plan["items"] == 746 == len(items)
checks["input_hash_mismatches"] = [p for p, h in plan["inputs"].items() if digest(p) != h]
checks["suite_sha256"] = plan["suite_sha256"]
ids = [i.id for i in items]
orig = json.loads((OUT / "original-reference.json").read_text())
checks["item_ids_match_original_reference"] = ids == orig["item_ids"] or sorted(ids) == sorted(orig["item_ids"])

passes, bad = 0, []
data = {}
for m in plan["models"]:
	runs = []
	for r in range(1, plan["repeats"] + 1):
		recs = []
		for iid in ids:
			p = result_path(OUT, m["slug"], r, iid)
			try:
				v = existing(p, contract, m["slug"], r, iid)  # contract, key, binary, record hash
				assert v is not None
				assert v["answer_sha256"] == hashlib.sha256(v["answer"].encode()).hexdigest()
				assert v["evaluation_seed"] == 41 + r
			except Exception as e:
				bad.append((m["slug"], r, iid, repr(e)))
				continue
			recs.append(v)
		n_files = len(list((OUT / "cells" / m["slug"] / f"run-{r}" / "items").glob("*.json")))
		if len(recs) == len(ids) and n_files == len(ids):
			passes += 1
			sc = json.loads((OUT / "cells" / m["slug"] / f"run-{r}" / "scores.json").read_text())
			assert sc["correct"] == sum(v["correct"] for v in recs) and sc["item_records_sha256"] == bind(recs)
		runs.append(recs)
	data[m["slug"]] = (m, runs)
checks["complete_746_passes"] = passes
checks["bad_records"] = bad[:5]
checks["bad_record_count"] = len(bad)

# independent recomputation of summary statistics
fam = {i.id: math_family(i.question) for i in items}
rep = []
for slug, (m, runs) in data.items():
	label = LABEL.get(m["label"], m["label"])
	accs = [sum(v["correct"] for v in run) / len(ids) for run in runs]
	cor = [[v["correct"] for v in run] for run in runs]
	outs = [[v["answer_sha256"] for v in run] for run in runs]
	means = [statistics.fmean(c[i] for c in cor) for i in range(len(ids))]
	groups = collections.defaultdict(list)
	for iid, x in zip(ids, means):
		groups[fam[iid]].append(x)
	g = list(groups.values())
	draws = boot(g, 2000, 42)
	s = next(e for e in summary["models"] if LABEL.get(e["model"], e["model"]) == label)
	row = dict(
		model=label, slug=slug, run_correct=[sum(c) for c in cor], run_accuracies=accs,
		mean_accuracy=statistics.fmean(accs), repeat_sd=statistics.stdev(accs), minimum=min(accs), maximum=max(accs),
		correctness_agreement=statistics.fmean(len({c[i] for c in cor}) == 1 for i in range(len(ids))),
		exact_output_agreement=statistics.fmean(len({o[i] for o in outs}) == 1 for i in range(len(ids))),
		family_ci=[draws[50], draws[1949]], families=len(g), items=len(ids),
		mean_output_tokens=statistics.fmean(v["output_tokens"] for run in runs for v in run),
		truncated_passes=[sum(v["finish_reason"] == "length" for v in run) for run in runs],
		summary_matches=abs(s["mean_accuracy"] - statistics.fmean(accs)) < 1e-12 and abs(s["question_family_interval"]["lower"] - draws[50]) < 1e-12 and abs(s["question_family_interval"]["upper"] - draws[1949]) < 1e-12,
		correctness_vector_sha256=[bind(c) for c in cor], output_sha256_per_item_sha256=[bind(o) for o in outs],
		mean_correctness_by_item=dict(zip(ids, means)),
	)
	rep.append(row)
checks["summary_matches_recompute"] = all(r["summary_matches"] for r in rep)
checks["ok"] = (passes == 21 and not bad and not checks["input_hash_mismatches"] and checks["summary_matches_recompute"]
	and checks["summary_contract_matches_plan"] and checks["plan_items_746"])

# original single-pass reference for comparison
ref = {}
for k, v in orig.get("cells", {}).items():
	ref[k] = {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))} if isinstance(v, dict) else v
# paired difference among deterministic students on item means (family bootstrap)
def paired(a, b):
	d = {i: a["mean_correctness_by_item"][i] - b["mean_correctness_by_item"][i] for i in ids}
	gr = collections.defaultdict(list)
	for i in ids: gr[fam[i]].append(d[i])
	g = list(gr.values()); dr = boot(g, 5000, 7)
	return statistics.fmean(d.values()), dr[125], dr[4874]
by = {r["model"]: r for r in rep}
pairs = []
for a, b in [("A5/instantaneous","A2/full"),("A5/instantaneous","A5/full"),("unprotected","A5/instantaneous"),("unprotected","A2/full"),("A2/full","A5/full"),("A2/instantaneous","A2/full"),("A2/full","base"),("teacher","unprotected")]:
	d, lo, hi = paired(by[a], by[b]); pairs.append(dict(a=a, b=b, diff=d, lower=lo, upper=hi, excludes_zero=(lo > 0 or hi < 0)))

public = dict(
	version="v7-repeat-aggregate-report-v1", n_items=746, contract=contract, checks=checks,
	models=[{k: v for k, v in r.items() if k not in ("mean_correctness_by_item",)} for r in rep], paired_family_bootstrap=pairs,
	original_single_pass=ref,
	limitations=[
		"n_items is 746; 3 passes x 746 is not 2238 independent problems.",
		"Student passes are greedy with fixed weights and are expected to be identical; they measure inference stability only, not training-seed or campaign variability.",
		"Teacher uses a GGUF llama.cpp server; its differences between passes reflect serving nondeterminism.",
		"Question-family intervals resample normalized question families and condition on the observed runs; families are not certified problem lineages.",
		"One training seed (42) per student; paired intervals are uncorrected for multiple comparisons.",
	])
(REP / "aggregate.json").write_text(json.dumps(public, indent="\t") + "\n")
(REP / "correctness-vectors.json").write_text(json.dumps({r["model"]: r["mean_correctness_by_item"] for r in rep}, indent="\t") + "\n")

L = ["# Fixed-checkpoint repeated inference report", "", f"Verification ok: {checks['ok']}  (21 complete passes: {passes}, bad records: {len(bad)}, input hash mismatches: {len(checks['input_hash_mismatches'])})", "",
	"| model | correct per pass | mean acc | repeat SD | range | correctness agree | exact-output agree | family 95% CI |", "|---|---|---|---|---|---|---|---|"]
for r in rep:
	L.append(f"| {r['model']} | {', '.join(map(str, r['run_correct']))} | {r['mean_accuracy']:.4f} | {r['repeat_sd']:.4f} | {r['minimum']:.4f}-{r['maximum']:.4f} | {r['correctness_agreement']:.4f} | {r['exact_output_agreement']:.4f} | {r['family_ci'][0]:.4f}-{r['family_ci'][1]:.4f} |")
L += ["", "## Paired differences (item-mean, family bootstrap 95%)", "", "| a - b | diff | 95% interval | excludes 0 |", "|---|---|---|---|"]
for p in pairs: L.append(f"| {p['a']} - {p['b']} | {p['diff']:+.4f} | {p['lower']:+.4f} to {p['upper']:+.4f} | {p['excludes_zero']} |")
L += ["", "## Limitations", ""] + [f"- {x}" for x in public["limitations"]]
(REP / "report.md").write_text("\n".join(L) + "\n")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.2))
x = range(len(rep))
ax.errorbar(x, [r["mean_accuracy"] for r in rep], yerr=[[r["mean_accuracy"] - r["family_ci"][0] for r in rep], [r["family_ci"][1] - r["mean_accuracy"] for r in rep]], fmt="o", capsize=4, color="#1f77b4", label="mean of 3 passes, family 95% CI")
for i, r in enumerate(rep):
	ax.scatter([i] * 3, r["run_accuracies"], marker="_", s=300, color="#d62728", zorder=3, label="individual passes" if i == 0 else None)
ax.set_xticks(list(x)); ax.set_xticklabels([r["model"] for r in rep], rotation=25, ha="right"); ax.set_ylabel("GSM8K accuracy (746 items)")
ax.set_title("Three fresh inference passes per fixed model"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(REP / "repeat-accuracy.png", dpi=160)
print(json.dumps(checks, indent=1)); print("\n".join(L))
