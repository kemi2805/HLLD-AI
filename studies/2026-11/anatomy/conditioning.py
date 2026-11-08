#!/usr/bin/env python
"""Why the seven-wave Newton fails where the planar solver answers: both
Jacobians AT the known answer (2026-11-04, anatomy of the four-quadrant 128^2
run).

    python conditioning.py SOLVED_SHARD.npz [...] --n 2000 --out cond.npz

Groups as seed_ceiling.py: first (seven-wave, primary network converged),
retry (seven-wave, a retry converged), planar (the five-wave planar solver).
Per face: the seven-wave residual at the answer (is it a root of the seven-wave
system, can its waves be built there), the six singular values of production's
forward-difference 6x6 Jacobian (`fullcontact_b.fullfuncd6`), the three of the
planar 3x3 (`planar5_b.jacobian`, in the planar frame and flip branch of the
answer), and the slow-Alfven gap of the two states.  In a coplanar problem the
two phi columns of the 6x6 vanish identically, so its rank is at most four and
the Newton drops directions below 1e-6 of the largest (`_svd_step`); the
question is whether a FIFTH direction collapses where the planar solver is
needed, and whether the 3x3 stays well conditioned there.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "planar_audit"))

GROUPS = ("first", "retry", "planar")


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def unk6_true(Z):
    R2, R3, R4, R6, R7 = Z[:, 1], Z[:, 2], Z[:, 3], Z[:, 5], Z[:, 6]
    ang = lambda R: np.arctan2(R[:, 6], R[:, 5])
    return np.stack([np.log(R2[:, 1]), np.log(np.hypot(R4[:, 5], R4[:, 6])), ang(R4),
                     np.log(R6[:, 1]), wrap(ang(R3) - ang(R2)), wrap(ang(R7) - ang(R6))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from rmhd.eos import set_eos
    set_eos("ideal")
    from rmhd.batched import fullcontact_b as FB
    from rmhd.batched import planar5_b as P5
    from rmhd.batched import wave_speeds_b as WB
    from face_table import face_feats
    rows = {k: [] for k in ("U_L", "U_R", "zones", "group")}
    for f in a.shards:
        z = np.load(f)
        g = np.where(z["source"] == 1, 2, np.where(z["attempts"] > 1, 1, 0))
        for k in ("U_L", "U_R", "zones"):
            rows[k].append(z[k])
        rows["group"].append(g)
        gamma = float(z["gamma"])
    S = {k: np.concatenate(v) for k, v in rows.items()}
    rng = np.random.default_rng(20261104)
    pick = np.concatenate([rng.choice(np.flatnonzero(S["group"] == gi),
                                      size=min(a.n, int((S["group"] == gi).sum())), replace=False)
                           for gi in range(3)])
    UL, UR, Z, grp = S["U_L"][pick], S["U_R"][pick], S["zones"][pick], S["group"][pick]
    left = [UL[:, j].copy() for j in range(7)]
    right = [UR[:, j].copy() for j in range(7)]
    Bn = UL[:, 7].copy()
    unk6 = unk6_true(Z)
    fvec, fjac, _, _, _, err = FB.make_solver(gamma)["fullfuncd6"](left, right, [unk6[c].copy() for c in range(6)], Bn)
    err = np.asarray(err, bool)
    r0 = np.where(err, np.inf, np.max(np.abs(np.asarray(fvec)), axis=0))
    sv6 = np.full((pick.size, 6), np.nan)
    sv6[~err] = np.linalg.svd(fjac[~err], compute_uv=False)
    # the planar 3x3 at the same answer, in its own frame and flip branch
    L, R, alpha, presid = P5.to_planar(left, right, Bn)
    unk3, flip = P5.unk6_to_unk3([unk6[c].copy() for c in range(6)], left)
    jac = P5.make_solver(gamma)["jacobian"]
    sv3 = np.full((pick.size, 3), np.nan)
    r3 = np.full(pick.size, np.nan)
    err3 = np.ones(pick.size, bool)
    flip = np.asarray(flip, bool)
    for fl in ((False, False), (True, False), (False, True), (True, True)):
        m = (flip[:, 0] == fl[0]) & (flip[:, 1] == fl[1])
        if not m.any():
            continue
        i = np.flatnonzero(m)
        f3, J3, _, _, e3, _ = jac([c[i] for c in L], [c[i] for c in R], [c[i] for c in unk3], Bn[i], fl)
        e3 = np.asarray(e3, bool)
        err3[i] = e3
        r3[i] = np.max(np.abs(np.asarray(f3)), axis=0)
        ok = ~e3
        sv3[i[ok]] = np.linalg.svd(J3[ok], compute_uv=False)
    F = face_feats(UL, UR, gamma, WB)
    gap = np.minimum(F["gapAS_L"], F["gapAS_R"])
    np.savez_compressed(a.out, group=grp, r0=r0, err6=err, sv6=sv6, sv3=sv3, r3=r3, err3=err3,
                        gapAS=gap, gapFA=np.minimum(F["gapFA_L"], F["gapFA_R"]), fn_max=F["fn_max"],
                        Wt=F["Wt"], beta_min=F["beta_min"], flip=flip)
    print("%-7s %6s %10s %10s %10s %12s %12s %12s %10s %10s" % ("group", "n", "unconstr", "root<=1e-8", "r0 p50",
          "s4/s1 p50", "s5/s1 p50", "s3/s1(3x3)", "gap p50", "planar err"))
    for gi, g in enumerate(GROUPS):
        m = grp == gi
        ok = m & ~err
        q = lambda v: np.nanmedian(v) if np.isfinite(v).any() else np.nan
        print("%-7s %6d %10.3f %10.3f %10.2e %12.2e %12.2e %12.2e %10.2e %10.3f" % (
            g, m.sum(), err[m].mean(), (r0[m] <= 1e-8).mean(), q(r0[ok]),
            q(sv6[ok, 3] / sv6[ok, 0]), q(sv6[ok, 4] / sv6[ok, 0]),
            q(sv3[m & ~err3, 2] / sv3[m & ~err3, 0]), np.median(gap[m]), err3[m].mean()))


if __name__ == "__main__":
    main()
