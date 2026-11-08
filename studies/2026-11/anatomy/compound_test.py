#!/usr/bin/env python
"""Can a compound slow wave answer the faces the four-quadrant 128^2 run lost?
(2026-11-04; user: "could we solve them with the compound wave?")

The element is HLLD scripts/compound_wave.py (experimental): an intermediate
shock that reverses B_t and leaves the flow behind it sonic, with a slow
rarefaction riding attached to its back -- the Brio-Wu structure -- inside the
three-unknown planar solve [ln p_LF, B_t*, ln p_RF].  Its own certificate
counts an answer: contact jumps <= 1e-8, the shock's jump conditions <= 1e-8,
sonic behind to 1e-6, the waves in order (a compound answer is by
construction not a seven-wave solution).

Offered to every face the failure ledger kept as failing (78407: 872) and to
a sample of faces production solves (the controls -- an answer there is a
SECOND solution, the non-uniqueness Balsara 1 showed), with the compound
element where it is consistent (an input with a field reversal: one compound
side, on the left or on the right; without one: both sides), from three
B_t* seeds and, on the first, three pressure scales; a face's later starts
run only if no earlier one was accepted.

    python compound_test.py LEDGER_DIR OUT.npz --shard i --nshards n [--n-ctrl 192]

Numpy slow-shock path only (compound_wave.make refuses RMHD_SLOWSHOCK=njit):
run WITHOUT kernels.env.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HLLD = os.environ.get("HLLD_DIR", "/mnt/rafast/miler/codes/rotor2d/HLLD-harvest")
sys.path.insert(0, HLLD)
sys.path.insert(0, os.path.join(HLLD, "scripts"))
import compound_wave as CW                                      # noqa: E402
from rmhd.batched import planar5_b as P5                        # noqa: E402

MODES = ((True, False), (False, True), (True, True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger")
    ap.add_argument("out")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n-ctrl", type=int, default=192)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    p = np.load(os.path.join(a.ledger, "population.npz"), allow_pickle=True)
    rows = list(csv.DictReader(open(os.path.join(a.ledger, "cards.csv"))))
    fail = np.array([r["group"] == "failing" for r in rows])
    rng = np.random.default_rng(20261104)
    ctrl = rng.choice(np.flatnonzero(~fail), size=min(a.n_ctrl, int((~fail).sum())), replace=False)
    lanes = np.sort(np.concatenate([np.flatnonzero(fail), ctrl]))
    lanes = lanes[a.shard::a.nshards]
    if a.limit:
        lanes = lanes[:a.limit]
    gamma = float(p["gamma"])
    S = CW.make(gamma)
    UL, UR = p["UL"][lanes], p["UR"][lanes]
    left = [UL[:, j].copy() for j in range(7)]
    right = [UR[:, j].copy() for j in range(7)]
    Bn = UL[:, 7].copy()
    L, R, alpha, presid = P5.to_planar(left, right, Bn)
    n = lanes.size
    base = P5.default_seed(L, R)
    bl, br = L[5], R[5]                     # planar frame: bl >= 0 by construction
    rev = br < 0                            # the input reverses the field
    # Only consistent configurations: a compound element reverses B_t on its
    # side and an ordinary slow wave cannot, so an input WITH a reversal needs
    # exactly one compound side (contact B_t of the other side's sign), one
    # WITHOUT needs both (contact B_t reversed against both).
    plans = [  # (mode, lanes, B_t seeds, as functions of (bl, br))
        ((True, False), rev, (lambda l, r: r, lambda l, r: -l, lambda l, r: 0.5 * (r - l))),
        ((False, True), rev, (lambda l, r: l, lambda l, r: -r, lambda l, r: 0.5 * (l - r))),
        ((True, True), ~rev, (lambda l, r: -l, lambda l, r: -r, lambda l, r: -0.5 * (l + r))),
    ]
    ok = np.zeros(n, bool)
    how = np.full((n, 3), -1, np.int8)                  # mode, B_t seed, pressure scale
    unk = np.full((n, 3), np.nan)
    rep_keep = {k: np.full(n, np.nan) for k in ("contact", "L_rh", "R_rh", "L_sonic", "R_sonic",
                                                 "L_intermediate", "R_intermediate",
                                                 "L_Vshock", "R_Vshock", "L_b", "R_b")}
    tries = 0
    for mi, (mode, sel, seeds) in enumerate(plans):
        for si, f in enumerate(seeds):
            for pi_, fp in enumerate((1.0, 0.5, 2.0)):
                if pi_ > 0 and si > 0:               # pressure scales on the first seed only
                    continue
                todo = np.flatnonzero(~ok & sel)
                if todo.size == 0:
                    continue
                sl = [c[todo] for c in L]
                sr = [c[todo] for c in R]
                u0 = [base[0][todo] + np.log(fp), np.array(f(bl[todo], br[todo]), float),
                      base[2][todo] + np.log(fp)]
                u, nrm = S["solve"](sl, sr, Bn[todo], u0, mode)
                good, rep, _ = S["verify"](sl, sr, Bn[todo], u, mode)
                tries += 1
                g = np.flatnonzero(good)
                if g.size:
                    w = todo[g]
                    ok[w] = True
                    how[w] = (mi, si, pi_)
                    for c in range(3):
                        unk[w, c] = u[c][g]
                    for k in rep_keep:
                        if k in rep:
                            rep_keep[k][w] = np.asarray(rep[k], float)[g]
                print("[shard %d] mode %s seed %d x%.1f on %d lanes: +%d (total %d of %d)  %.0f s"
                      % (a.shard, mode, si, fp, todo.size, g.size, ok.sum(), n, time.time() - t0), flush=True)
    np.savez_compressed(a.out, lanes=lanes, failing=fail[lanes], ok=ok, how=how, unk=unk, reversal=rev,
                        presid=np.asarray(presid), tries=tries, **{"rep_" + k: v for k, v in rep_keep.items()})
    print("[shard %d] done: %d of %d answered by a compound wave (%d failing lanes: %d)  %.0f s"
          % (a.shard, ok.sum(), n, fail[lanes].sum(), (ok & fail[lanes]).sum(), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
