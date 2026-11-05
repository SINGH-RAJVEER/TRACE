"""
Embed every distinct detector text in the benign roles and attack schedules with the frozen BGE.
"""

import sys
import time

import numpy as np
from common import ATTACKS, CACHE, ROLES, benign, key, schedule
from harness.encoder import BGEEncoder

variant = sys.argv[1] if len(sys.argv) > 1 else "detector"
field = {"detector": "detector", "noprefix": "detector_noprefix"}[variant]
rows = [r for role in ROLES for r in benign(role)] + [r for a in ATTACKS for r in schedule(a)]
texts = {}
for r in rows:
	texts.setdefault(key(r[field]), r[field])
keys = sorted(texts, key=lambda k: (len(texts[k]), k))
print(f"{variant}: {len(rows)} rows, {len(keys)} distinct texts", flush=True)
encoder = BGEEncoder(device="cpu", batch_size=64)
start, out = time.time(), []
for i in range(0, len(keys), 2048):
	out.append(encoder.encode([texts[k] for k in keys[i : i + 2048]]))
	print(
		f"{i + len(out[-1])}/{len(keys)} {time.time() - start:.0f}s truncated={encoder.truncated}",
		flush=True,
	)
np.savez(CACHE / f"{variant}.npz", keys=np.array(keys), vectors=np.concatenate(out))
print("identity", encoder.identity())
