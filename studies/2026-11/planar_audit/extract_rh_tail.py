#!/usr/bin/env python
"""The RH-certificate tail of every harvested shock (2026-11-04): accepted
exact answers whose slow (or fast) shock does not satisfy its own jump
conditions.  The seven-wave chain closes only eq. (4.17) of a slow shock in
the outer residual; the sub-solver's (f_hug, f_416) are trusted, and the
ladder's loose fallback hands back a grid point that is no root.

    python extract_rh_tail.py OUTDIR RUN_DIR [RUN_DIR ...]

Per run: histogram (log10 bins) of the RH certificate (wavecheck.rh_cert at
the stored speed) of every shock by the solvers' predicates, per family
(slow LS/RS, fast LF/RF) and source (seven-wave, planar5); counts above
1e-6/1e-5/1e-4/1e-3; and OUTDIR/<run>_tail.npz with every ROW whose worst
shock certificate exceeds 1e-5 (states, zones, speeds, source, attempts,
shard, row, per-wave K / rh / lax_min / solver_kind / strength).
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wavecheck as WC           # noqa: E402

W = ("LF", "LS", "RS", "RF")
EDGES = np.arange(-16, 1, 1.0)                     # log10 bins; below 1e-16 -> first
THR = (1e-6, 1e-5, 1e-4, 1e-3)
KEEP = 1e-5


def one(path):
    z = np.load(path)
    if "zones" not in z.files or z["U_L"].shape[0] == 0:
        return None
    g = float(z["gamma"])
    UL, UR = np.asarray(z["U_L"], float), np.asarray(z["U_R"], float)
    Z, S = np.asarray(z["zones"], float), np.asarray(z["speeds"], float)
    n = UL.shape[0]
    src = np.asarray(z["source"]) if "source" in z.files else np.zeros(n, np.int8)
    a = WC.admissibility(UL, UR, Z, S, g)
    hist, cnt = {}, {}
    worst = np.zeros(n)
    for fam, ws in (("slow", ("LS", "RS")), ("fast", ("LF", "RF"))):
        for s, nm in ((0, "seven"), (1, "planar5")):
            x = np.concatenate([a[w + "_rh"][(a[w + "_solver_kind"] == 1) & (src == s)] for w in ws])
            x = x[np.isfinite(x)]
            lx = np.log10(np.maximum(x, 1e-300))
            hist[fam + "_" + nm] = np.histogram(np.clip(lx, -16, 0), bins=np.append(EDGES, 0.5))[0]
            cnt[fam + "_" + nm] = [int(x.size)] + [int((x > t).sum()) for t in THR]
        for w in ws:
            r = np.where(a[w + "_solver_kind"] == 1, a[w + "_rh"], 0.0)
            worst = np.maximum(worst, np.where(np.isfinite(r), r, 0.0))
    i = np.flatnonzero(worst > KEEP)
    out = {"U_L": UL[i], "U_R": UR[i], "zones": Z[i], "speeds": S[i], "source": src[i],
           "attempts": (np.asarray(z["attempts"]) if "attempts" in z.files else np.zeros(n, np.int16))[i],
           "shard": np.full(i.size, int(os.path.basename(path).split("_")[-1].split(".")[0])),
           "row": i, "worst_rh": worst[i]}
    for w in W:
        for k in ("K", "rh", "lax_min", "solver_kind", "strength"):
            out[w + "_" + k] = a[w + "_" + k][i]
    rows = {"rows": [int((src == s).sum()) for s in (0, 1)],
            "rows_gt": {str(t): [int(((worst > t) & (src == s)).sum()) for s in (0, 1)] for t in THR}}
    return out, hist, cnt, rows, g


def main():
    od, runs = sys.argv[1], sys.argv[2:]
    os.makedirs(od, exist_ok=True)
    summ = {}
    with ProcessPoolExecutor(max_workers=int(os.environ.get("NPROC", "60"))) as ex:
        for run in runs:
            name = os.path.basename(run.rstrip("/"))
            shards = sorted(glob.glob(os.path.join(run, "harvest", "solved_*.npz")))
            parts, H, C, R = [], {}, {}, {"rows": [0, 0], "rows_gt": {str(t): [0, 0] for t in THR}}
            g = None
            for r in ex.map(one, shards, chunksize=4):
                if r is None:
                    continue
                parts.append(r[0])
                for k, v in r[1].items():
                    H[k] = H.get(k, 0) + v
                for k, v in r[2].items():
                    C[k] = [a + b for a, b in zip(C.get(k, [0] * len(v)), v)]
                R["rows"] = [a + b for a, b in zip(R["rows"], r[3]["rows"])]
                for t in R["rows_gt"]:
                    R["rows_gt"][t] = [a + b for a, b in zip(R["rows_gt"][t], r[3]["rows_gt"][t])]
                g = r[4]
            cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            cat["gamma"] = np.float64(g)
            np.savez_compressed(os.path.join(od, name + "_tail.npz"), **cat)
            e = {"counts [shocks, >1e-6, >1e-5, >1e-4, >1e-3]": C,
                 "rows [seven, planar5]": R,
                 "hist_log10_edges": EDGES.tolist(),
                 "hist": {k: v.tolist() for k, v in H.items()}}
            summ[name] = e
            print(name, json.dumps({"counts": C, "rows": R}), flush=True)
    json.dump(summ, open(os.path.join(od, "rh_tail_summary.json"), "w"), indent=1)
    print("wrote", od)


if __name__ == "__main__":
    main()
