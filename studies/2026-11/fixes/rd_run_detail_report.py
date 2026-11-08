#!/usr/bin/env python
"""The production answers whose regular re-solve (rotation in front of the
slow wave) is NOT admissible: which wave lowers the entropy, by how much, and
whether it still runs across the Alfven speed (rd_run_detail/, calea 79435).

    python rd_run_detail_report.py rd_run_detail
"""
import glob
import sys
from collections import Counter

import numpy as np

NAMES = ("LF", "LA", "LS", "CD", "RS", "RA", "RF")
rows = []
for f in sorted(glob.glob(sys.argv[1] + "/chunk_*.npy")):
    rows += list(np.load(f, allow_pickle=True))
print("answers %d" % len(rows))
best_bad = []
for r in rows:
    if any(c[2] for c in r["combos"]):
        continue
    # the combination with the smallest entropy fall among the weak solutions
    cand = []
    for d, c in zip(r.get("detail", []), r["combos"]):
        if not c[1]:
            continue
        dS = np.where(d["kind"] == "disc", d["dS"], np.nan)
        dS[3] = np.nan
        w = int(np.nanargmin(dS)) if np.isfinite(dS).any() else -1
        cand.append((float(np.nanmin(dS)) if w >= 0 else 0.0, w, d, c))
    if cand:
        best_bad.append(max(cand, key=lambda t: t[0]) + (r,))
print("regular weak solutions with a falling entropy (best combination each): %d" % len(best_bad))
print("wave with the largest fall:", Counter(NAMES[b[1]] for b in best_bad if b[1] >= 0))
f = np.array([-b[0] for b in best_bad])
print("the fall: quantiles", np.percentile(f, [0, 10, 25, 50, 75, 90, 100]))
for t in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
    print("   falls by more than %.0e: %d" % (t, (f > t).sum()))
inter = Counter(bool(b[2]["inter"][b[1]]) for b in best_bad if b[1] >= 0)
print("that jump still crosses the Alfven speed:", inter)
print("rotation combination of the best:", Counter(str(b[2]["rot"]) for b in best_bad))
print("original answer: source / admissible:", Counter((b[4]["src"], b[4]["adm_old"]) for b in best_bad))
fails = Counter()
for r in rows:
    if any(c[2] for c in r["combos"]) or any(c[1] for c in r["combos"]):
        continue
    for d in r.get("detail", []):
        for l_, w, why in d["failwhy"]:
            fails[(NAMES[w], why)] += 1
        if not d["order_ok"]:
            fails[("order", "")] += 1
print("why the other answers fail the verifier:", fails.most_common(10))
