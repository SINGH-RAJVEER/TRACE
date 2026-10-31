"""Summarize harness decision logs per account: windows, p-values, statistics, levels."""
import json, sys, collections
from pathlib import Path

for path in sys.argv[1:]:
	rows = [json.loads(l) for l in Path(path).open()]
	events = collections.Counter(r.get("event") for r in rows)
	by = collections.defaultdict(list)
	for r in rows:
		if r.get("event") == "decision":
			by[r["account"]].append(r)
	print(f"== {path}: {len(rows)} rows, events {dict(events)}, accounts {len(by)}")
	for account, ds in sorted(by.items(), key=lambda kv: -len(kv[1])):
		wins = [d for d in ds if d["window_completed"]]
		ps = [d["p_value"] for d in wins]
		levels = collections.Counter(d["level"] for d in ds)
		actions = collections.Counter(d["action"] for d in ds)
		if account.startswith("A") or max(levels) > 0:
			print(f"  {account}: n={len(ds)} attempts {ds[0]['attempt']}..{ds[-1]['attempt']} windows={len(wins)} "
				f"p[min/med/max]={min(ps, default=None)}/{sorted(ps)[len(ps)//2] if ps else None}/{max(ps, default=None)} "
				f"maxstat={max(d['statistic'] or 0 for d in ds):.3f} levels={dict(levels)} actions={dict(actions)}")
	benign = [a for a in by if not a.startswith("A")]
	scored = [a for a in benign if any(d["window_completed"] for d in by[a])]
	print(f"  benign accounts {len(benign)}, scored {len(scored)}, benign decisions {sum(len(by[a]) for a in benign)}")
