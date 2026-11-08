#!/usr/bin/env python
"""admiss_all.py's admissibility flags over a run's HARVEST (every solved_*.npz
shard), shard-parallel (planar acceptance study, 2026-11-03).

    python admiss_harvest.py OUT.json RUN_DIR [RUN_DIR ...]

Per run: rows by source (0 seven-wave, 1 planar five-wave) and the flagged
counts (entropy-falling shock, Lax < -1e-4, slow/fast overlap > 1e-6), plus
the entropy-ratio quantiles of the flagged slow shocks and run_meta's face
counts for scale.
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import admiss_all as AA          # noqa: E402

KEYS = ("ent_LS", "ent_RS", "ent_F", "lax_LS", "lax_RS", "lax_F", "inter_LS", "inter_RS",
        "cross_L", "cross_R", "entropy", "lax", "cross", "any")


def one(path):
    z = np.load(path)
    if "zones" not in z.files or z["U_L"].shape[0] == 0:
        return None
    AA.G = float(z["gamma"])
    UL, UR = np.asarray(z["U_L"], float), np.asarray(z["U_R"], float)
    fl = AA.flags(UL, UR, np.asarray(z["zones"], float), np.asarray(z["speeds"], float))
    src = np.asarray(z["source"]) if "source" in z.files else np.zeros(UL.shape[0], np.int8)
    out = {}
    for s in (0, 1):
        m = src == s
        e = {"rows": int(m.sum())}
        for k in KEYS:
            e[k] = int((fl[k] & m).sum())
        e["Kmin_flagged"] = fl["Kmin"][fl["entropy"] & m].tolist()
        out[s] = e
    return out


def main():
    oj, runs = sys.argv[1], sys.argv[2:]
    res = {}
    with ProcessPoolExecutor(max_workers=int(os.environ.get("NPROC", "60"))) as ex:
        for run in runs:
            shards = sorted(glob.glob(os.path.join(run, "harvest", "solved_*.npz")))
            tot = {s: {k: 0 for k in ("rows",) + KEYS} for s in (0, 1)}
            Ks = {0: [], 1: []}
            for r in ex.map(one, shards, chunksize=4):
                if r is None:
                    continue
                for s in (0, 1):
                    for k in ("rows",) + KEYS:
                        tot[s][k] += r[s][k]
                    Ks[s] += r[s]["Kmin_flagged"]
            e = {}
            for s, nm in ((0, "seven"), (1, "planar5")):
                d = dict(tot[s])
                K = np.array(Ks[s])
                d["K_lt_0.99"] = int((K < 0.99).sum())
                d["K_lt_0.9"] = int((K < 0.9).sum())
                e[nm] = d
            meta = os.path.join(run, "run_meta.json")
            if os.path.exists(meta):
                m = json.load(open(meta))
                e["run_meta"] = {k: m.get(k) for k in m if any(s in k for s in (
                    "attempted", "exact", "planar", "seven", "hlld_after"))
                    and not isinstance(m.get(k), (dict, list))}
            res[os.path.basename(run.rstrip("/"))] = e
            print(os.path.basename(run), json.dumps(e), flush=True)
    json.dump(res, open(oj, "w"), indent=1)
    print("wrote", oj)


if __name__ == "__main__":
    main()
