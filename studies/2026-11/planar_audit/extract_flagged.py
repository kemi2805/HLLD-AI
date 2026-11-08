#!/usr/bin/env python
"""The flagged rows (admiss_all.flags: entropy-falling shock, Lax < -1e-4,
slow/fast overlap) of runs' harvests, with their per-wave diagnostics
(2026-11-04, the paired 64^2 filter runs: which path still produces an
entropy-falling slow shock under RMHD_SLOWSHOCK_ADMISSIBLE=1).

    python extract_flagged.py OUTDIR RUN_DIR [RUN_DIR ...]

Per run OUTDIR/<run>.npz: U_L, U_R, zones, speeds, source, attempts, shard,
row, the flags (f_*), and per magnetosonic wave (LF LS RS RF) K, rh (RH
certificate at the stored speed), lax_min, lax_margins, kind, solver_kind,
strength, drho, dBt, dP; LA/RA strengths; the CD jump.  Plus, over EVERY
row with a slow shock (by the solver's predicate), the RH-certificate
quantiles of those shocks, for scale.
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import admiss_all as AA          # noqa: E402
import wavecheck as WC           # noqa: E402

W = ("LF", "LS", "RS", "RF")
FK = ("entropy", "lax", "cross", "inter_LS", "inter_RS", "ent_LS", "ent_RS", "lax_LS", "lax_RS",
      "lax_F", "Kmin", "laxmin_slow", "cross_amt")
WK = ("K", "rh", "lax_min", "kind", "solver_kind", "strength", "drho", "dBt", "dP")
QS = (0.5, 0.9, 0.99, 0.999, 1.0)


def one(path):
    z = np.load(path)
    if "zones" not in z.files or z["U_L"].shape[0] == 0:
        return None
    g = float(z["gamma"])
    AA.G = g
    UL, UR = np.asarray(z["U_L"], float), np.asarray(z["U_R"], float)
    Z, S = np.asarray(z["zones"], float), np.asarray(z["speeds"], float)
    n = UL.shape[0]
    src = np.asarray(z["source"]) if "source" in z.files else np.zeros(n, np.int8)
    # RH certificate of every slow shock, for scale (seven-wave and planar5 rows)
    a_all = WC.admissibility(UL, UR, Z, S, g)
    rh_slow = {s: np.concatenate([a_all[w + "_rh"][(a_all[w + "_solver_kind"] == 1) & (src == s)]
                                  for w in ("LS", "RS")]) for s in (0, 1)}
    fl = AA.flags(UL, UR, Z, S)
    i = np.flatnonzero(fl["any"])
    out = {"U_L": UL[i], "U_R": UR[i], "zones": Z[i], "speeds": S[i], "source": src[i],
           "attempts": (np.asarray(z["attempts"]) if "attempts" in z.files else np.zeros(n, np.int16))[i],
           "shard": np.full(i.size, int(os.path.basename(path).split("_")[-1].split(".")[0])),
           "row": i}
    for k in FK:
        out["f_" + k] = fl[k][i]
    for w in W:
        for k in WK:
            out[w + "_" + k] = a_all[w + "_" + k][i]
        out[w + "_lax_margins"] = np.asarray(a_all[w + "_lax_margins"])[:, i].T
    out["LA_strength"] = a_all["LA_strength"][i]
    out["RA_strength"] = a_all["RA_strength"][i]
    out["CD_jump"] = np.asarray(a_all["CD_jump"])[:, i].T
    return out, rh_slow, g


def main():
    od, runs = sys.argv[1], sys.argv[2:]
    os.makedirs(od, exist_ok=True)
    summ = {}
    with ProcessPoolExecutor(max_workers=int(os.environ.get("NPROC", "60"))) as ex:
        for run in runs:
            name = os.path.basename(run.rstrip("/"))
            shards = sorted(glob.glob(os.path.join(run, "harvest", "solved_*.npz")))
            parts, rh = [], {0: [], 1: []}
            g = None
            for r in ex.map(one, shards, chunksize=4):
                if r is None:
                    continue
                parts.append(r[0])
                for s in (0, 1):
                    rh[s].append(r[1][s])
                g = r[2]
            cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            cat["gamma"] = np.float64(g)
            np.savez_compressed(os.path.join(od, name + ".npz"), **cat)
            e = {"flagged_rows": int(cat["U_L"].shape[0])}
            for s, nm in ((0, "seven"), (1, "planar5")):
                x = np.concatenate(rh[s]) if rh[s] else np.zeros(0)
                x = x[np.isfinite(x)]
                e[nm + "_slow_shocks"] = int(x.size)
                e[nm + "_slow_rh_q"] = dict(zip(map(str, QS), np.quantile(x, QS).tolist())) if x.size else {}
                e[nm + "_slow_rh_gt_1e-6"] = int((x > 1e-6).sum())
            summ[name] = e
            print(name, json.dumps(e), flush=True)
    json.dump(summ, open(os.path.join(od, "summary.json"), "w"), indent=1)
    print("wrote", od)


if __name__ == "__main__":
    main()
