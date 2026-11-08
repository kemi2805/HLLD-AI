#!/usr/bin/env python
"""Concatenate face_table.py's shards into a stratified sample small enough to
analyse on a laptop, with the weights that turn it back into the run's
population (2026-11-04).

    python sample_table.py FACE_DIR OUT.npz [--n-seven 400000] [--n-planar5 200000]

Kept whole: every face that was not answered by the seven-wave or the planar
solver (four-unknown, three-wave, continuation, linear fallback, lost) and
every seven-wave face that needed a retry.  Drawn uniformly (fixed seed): the
first-try seven-wave faces and the planar faces.  `weight` = the inverse of
the draw fraction of the face's stratum, so weighted sums are run totals.
Also writes OUT.json: per outcome and t window, the counts of the whole run.
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("faces")
    ap.add_argument("out")
    ap.add_argument("--n-seven", type=int, default=400_000)
    ap.add_argument("--n-planar5", type=int, default=200_000)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.faces, "faces_*.npz")))
    # pass 1: strata sizes
    strat = []
    for f in files:
        z = np.load(f)
        oc, at = z["outcome"], z["attempts"]
        s = np.where(oc == 1, np.where(at > 1, 1, 0), np.where(oc == 2, 2, 3))   # 0 seven first, 1 seven retry, 2 planar, 3 rest
        strat.append(s)
    allS = np.concatenate(strat)
    N = np.bincount(allS, minlength=4)
    want = np.array([min(a.n_seven, N[0]), N[1], min(a.n_planar5, N[2]), N[3]])
    frac = want / np.maximum(N, 1)
    rng = np.random.default_rng(20261104)
    keep_all = [rng.random(s.size) < frac[s] for s in strat]
    parts = {}
    counts = {}
    for f, s, kp in zip(files, strat, keep_all):
        z = np.load(f)
        for key in z.files:
            if key == "gamma":
                parts["gamma"] = z["gamma"]
                continue
            parts.setdefault(key, []).append(z[key][kp])
        parts.setdefault("weight", []).append((1.0 / frac[s[kp]]).astype(np.float32))
        parts.setdefault("stratum", []).append(s[kp].astype(np.int8))
        # whole-run counts per outcome and t window (physical faces)
        w = np.minimum((z["t"] / 0.05).astype(int), 7)
        for o in np.unique(z["outcome"]):
            for k in range(8):
                m = (z["outcome"] == o) & (w == k) & z["phys"]
                counts.setdefault(str(int(o)), [0] * 8)[k] += int(m.sum())
    out = {k: (np.concatenate(v) if isinstance(v, list) else v) for k, v in parts.items()}
    np.savez_compressed(a.out, **out)
    rep = dict(strata=dict(zip(("seven_first", "seven_retry", "planar5", "rest"), N.tolist())),
               drawn=want.tolist(), rows=int(out["outcome"].size),
               counts_by_outcome_and_t_window_of_0p05=counts)
    json.dump(rep, open(a.out.replace(".npz", ".json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
