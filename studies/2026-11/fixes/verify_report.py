#!/usr/bin/env python
"""Aggregate verify_rmhd.py chunks: how many answers are actual weak solutions
of ideal RMHD, how many are also entropy-admissible, by solver and (on the
ledger) production's answers vs the faces the loss fixes recovered."""
import glob
import sys
import numpy as np
import verify_rmhd as V

D = "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/verify"
pop = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz", allow_pickle=True)
key = lambda A, B: [a.tobytes() + b.tobytes() for a, b in zip(np.ascontiguousarray(A), np.ascontiguousarray(B))]
grp = {k: int(g) for k, g in zip(key(pop["UL"], pop["UR"]), pop["group"])}


def load(which):
    ins, outs = [], []
    for f in sorted(glob.glob("%s/%s/chunk_*_verify.npz" % (D, which))):
        outs.append(dict(np.load(f, allow_pickle=True)))
        ins.append(dict(np.load(f.replace("_verify.npz", ".npz"), allow_pickle=True)))
    cat = lambda L: {k: np.concatenate([d[k] for d in L]) for k in L[0] if np.ndim(L[0][k]) > 0 and k != "failwhy"}
    return cat(ins), cat(outs)


SRC = {1: "seven-wave", 2: "planar (5-wave)", 3: "four-unknown"}
for which in ("ledger", "run"):
    a, r = load(which)
    src = a["source"] if which == "ledger" else np.where(a["source"] == 1, 2, 1)
    print("\n######## %s: %d answers" % (which, len(src)))
    if which == "ledger":
        g = np.array([grp.get(k, -1) for k in key(a["U_L"], a["U_R"])])
        print("matched to the population: controls (production solved) %d, recovered by the fixes %d, unmatched %d" % (
            (g == 0).sum(), (g == 1).sum(), (g == -1).sum()))
        groups = (("production's answers (controls)", g == 0), ("recovered by the fixes", g == 1))
    else:
        groups = (("all sampled", np.ones(len(src), bool)),)
    for gname, gm in groups:
        for s in (1, 2, 3):
            m = gm & (src == s)
            if not m.any():
                continue
            print("  %-32s %-16s n=%6d  weak solution %6d (%.3f%%)   + entropy-admissible %6d (%.3f%%)" % (
                gname, SRC[s], m.sum(), r["weak"][m].sum(), 100 * r["weak"][m].mean(),
                r["admissible"][m].sum(), 100 * r["admissible"][m].mean()))
    # what fails, wave by wave
    k = r["kind"]
    fails = ~r["weak"]
    print("  not weak solutions: %d; out of order %d; failing waves:" % (fails.sum(), (~r["order_ok"]).sum()),
          {nm: int((k[:, w] == "FAIL").sum()) for w, nm in enumerate(V.NAMES)})
    ent = r["weak"] & ~r["admissible"]
    if ent.any():
        dsmin = np.nanmin(np.where((k == "disc") & (np.arange(7)[None] != 3), r["dS"], np.nan)[ent], axis=1)
        print("  weak but entropy falls across a jump: %d -- largest drop p50 %.1e max %.1e" % (
            ent.sum(), np.median(-dsmin), np.max(-dsmin)))
        for w, nm in enumerate(V.NAMES):
            m = ent & (k[:, w] == "disc") & (r["dS"][:, w] < -V.TOL_S)
            if m.any():
                print("     at %s: %d (intermediate, i.e. crossing the Alfven speed: %d)" % (nm, m.sum(), (m & r["inter"][:, w]).sum()))
    d = (k == "disc") & (np.arange(7)[None] != 3)
    f = k == "fan"
    print("  every discontinuity: jump-condition residual p50 %.0e, p99.9 %.0e, max %.0e" % (
        np.median(r["rh"][d]), np.percentile(r["rh"][d], 99.9), r["rh"][d].max()))
    print("  every fan: end-state mismatch p50 %.0e max %.0e; weak-solution identity p50 %.0e max %.0e" % (
        np.median(r["fan"][f]), r["fan"][f].max(), np.median(r["G"][f]), r["G"][f].max()))
    inter = r["inter"] & (k == "disc")
    print("  intermediate shocks (crossing the Alfven speed): %d in %d answers; entropy rising on %d of them" % (
        inter.sum(), inter.any(axis=1).sum(), (inter & (r["dS"] > V.TOL_S)).sum()))
    np.save("%s/%s_fail_idx.npy" % (D, which), np.flatnonzero(fails))
