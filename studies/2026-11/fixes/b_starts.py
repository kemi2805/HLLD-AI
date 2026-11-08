#!/usr/bin/env python
"""The four-unknown planar solver from HLLD's star pressure on the faces the
loose-retry fix still loses (and on controls): every slow-wave steering
(PP, PB, BP, BB) x a few slow-wave start strengths, each answer held to
production's planar4 acceptance (certificate, order, planarity, ray,
physical state).  2026-11-04, user: "try them out ... get to 100%".

    python b_starts.py POP.npz OFF.npz ON.npz OUT.npz [--shard k --nshards n] [--controls N]
"""
import argparse
import os
import sys
import time

import numpy as np

HLLD = os.environ.get("HLLD_DIR", "/mnt/rafast/miler/codes/rotor2d/HLLD")
sys.path.insert(0, HLLD)
from rmhd.eos import set_eos                                     # noqa: E402
set_eos("ideal")
from rmhd.batched import planar4_b as P4B                        # noqa: E402
from src.physics import exact_flux as EF                         # noqa: E402

KINDS = {"PP": (1, 1), "PB": (1, 0), "BP": (0, 1), "BB": (0, 0),
         "SS": (2, 2), "SP": (2, 1), "PS": (1, 2), "SB": (2, 0), "BS": (0, 2)}
DELTAS = (1e-4, 1e-3, -1e-4)


def accept(P, S4, r, L, R, Bn, g):
    """production's _planar4_and_ray acceptance on a solve() result"""
    good = r["converged"].astype(bool).copy()
    star = None
    if good.any():
        A, B_, C_, D = r["zones"]
        zones6 = [A, A, B_, C_, D, D]
        VsLv = [r["speeds"][0], S4["alfven_speed"](A, Bn, "L"), r["speeds"][1]]
        VsRv = [r["speeds"][4], S4["alfven_speed"](D, Bn, "R"), r["speeds"][3]]
        sh = r["shocks"]
        st, reg, e = P["ray"].state_at_xi(
            L, R, zones6, VsLv, VsRv, Bn, g, P["xi"], P["fan_p"], P["fan_n"],
            kinds={"LF": sh[0], "LS": sh[1], "RS": sh[2], "RF": sh[3]},
            slow_by_pressure=(r["kinds"][0] != P4B.KIND_B, r["kinds"][1] != P4B.KIND_B))
        good &= ~e
        rho, Pt, vn, v1, v2, b1, b2 = st
        vv = vn * vn + v1 * v1 + v2 * v2
        W2 = 1.0 / np.maximum(1.0 - vv, 1e-300)
        eta = Bn * vn + b1 * v1 + b2 * v2
        bb = (Bn ** 2 + b1 ** 2 + b2 ** 2) / W2 + eta ** 2
        phys = (rho > 0) & (vv < 1) & (Pt - 0.5 * bb > 0)
        for c in st:
            phys &= np.isfinite(c)
        good &= phys
        star = st
    return good, star


def entropy_min(r, L, R, Bn, g):
    """the smallest entropy ratio K_behind / K_ahead over the shocks of an
    answer (1 where it has none); K = p_gas / rho^gamma"""
    def K(X):
        rho, Pt, vx, vy, vz, By, Bz = X
        v2 = vx * vx + vy * vy + vz * vz
        W2 = 1.0 / np.maximum(1.0 - v2, 1e-300)
        vB = vx * Bn + vy * By + vz * Bz
        return (Pt - 0.5 * ((Bn ** 2 + By ** 2 + Bz ** 2) / W2 + vB ** 2)) / X[0] ** g
    A, B_, C_, D = r["zones"]
    pairs = ((L, A), (A, B_), (D, C_), (R, D))
    m = np.ones(Bn.shape[0])
    for row, (ah, bh) in enumerate(pairs):
        m = np.where(r["shocks"][row], np.minimum(m, K(bh) / K(ah)), m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pop"); ap.add_argument("off"); ap.add_argument("on"); ap.add_argument("out")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--controls", type=int, default=300)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--kinds", default="PP,PB,BP,BB")
    ap.add_argument("--deltas", default="1e-4,1e-3,-1e-4")
    ap.add_argument("--pscale", default="1")
    ap.add_argument("--central", action="store_true")
    ap.add_argument("--pmode", default="star", choices=("star", "side", "both"))
    ap.add_argument("--near", type=float, default=0.02)
    a = ap.parse_args()
    pop, off, on = (np.load(f, allow_pickle=True) for f in (a.pop, a.off, a.on))
    g = float(pop["gamma"])
    P = EF._batched_parts(g)
    S4 = P["planar4"]
    grp = pop["group"]
    lost = np.flatnonzero((grp == 1) & ~on["exact"])
    if os.environ.get("BSTARTS_SKIP"):           # lanes an earlier pass recovered
        lost = np.setdiff1d(lost, np.load(os.environ["BSTARTS_SKIP"]))
    ctrl = np.flatnonzero(grp == 0)
    ctrl = ctrl[np.random.default_rng(1).permutation(ctrl.size)[:a.controls]]
    lanes = np.concatenate([lost, ctrl])
    lanes = lanes[a.shard::a.nshards]
    n = lanes.size
    UL, UR = pop["UL"][lanes], pop["UR"][lanes]
    L = [UL[:, j].copy() for j in range(7)]; R = [UR[:, j].copy() for j in range(7)]
    Bn = UL[:, 7].copy()
    pstar = np.abs(off["pstar"][lanes])                  # HLLD's star pressure
    L0, R0, alpha, presid = P4B.to_planar(L, R, Bn)
    W = S4["waves"]
    out = dict(lane=lanes, ctrl=grp[lanes] == 0, ok=np.zeros(n, bool), how=np.full(n, "", "U12"),
               smin=np.full(n, np.nan), kinds=np.full((2, n), -1, np.int8), shocks=np.zeros((4, n), bool),
               p_ans=np.full(n, np.nan), p_prod=on["pstar"][lanes], tried=0)
    t0 = time.time()
    star_all = [np.full(n, np.nan) for _ in range(7)]
    deltas = [float(x) for x in a.deltas.split(",")]
    for kn, kd, dl, ps in [(k, KINDS[k], d, q) for q in (float(x) for x in a.pscale.split(","))
                           for k in a.kinds.split(",") for d in deltas]:
        if True:
            i = np.flatnonzero(~out["ok"] | out["ctrl"])
            if i.size == 0:
                break
            # the start: both fast waves to p*, the slow waves weak (fans for
            # delta > 0) from the states behind those fast waves
            pl, pr = pstar[i] * ps, pstar[i] * ps
            if a.pmode == "side":
                # a fast wave whose ahead pressure p* nearly equals is weak:
                # start it there (just past it, by ps - 1), not at p*
                for pp, P0 in ((pl, L0[1][i]), (pr, R0[1][i])):
                    near = np.abs(pstar[i] / P0 - 1.0) < a.near
                    pp[near] = P0[near] * (1.0 + (ps - 1.0))
            elif a.pmode == "both":
                # both fast waves at their own ahead pressures
                pl, pr = L0[1][i] * (1.0 + (ps - 1.0)), R0[1][i] * (1.0 + (ps - 1.0))
            lp = np.log(np.maximum(pl, 1e-12))
            lpr = np.log(np.maximum(pr, 1e-12))
            A, _, eA = W["fast_wave"](pl, [c[i] for c in L0], "LF", Bn[i])
            D, _, eD = W["fast_wave"](pr, [c[i] for c in R0], "RF", Bn[i])
            sL = np.log(np.maximum(A[1] * (1 - dl), 1e-300)) if kd[0] == 1 else A[5] * (1 + dl)
            sR = np.log(np.maximum(D[1] * (1 - dl), 1e-300)) if kd[1] == 1 else D[5] * (1 + dl)
            kinds = np.tile(np.array(kd, np.int8)[:, None], (1, i.size))
            r = S4["solve"]([c[i] for c in L], [c[i] for c in R], Bn[i], accuracy=1e-8, max_iter=a.max_iter,
                            planar_tol=EF._PLANAR_TOL, verify_tol=EF._PLANAR5_VERIFY,
                            kinds=kinds, retry=False, unk_init=[lp, sL, sR, lpr],
                            central=a.central)
            good, star = accept(P, S4, r, [c[i] for c in L], [c[i] for c in R], Bn[i], g)
            good &= ~(eA | eD)
            new = good & ~out["ok"][i]
            w = i[new]
            out["ok"][w] = True
            out["how"][w] = "%s %+.0e x%g" % (kn, dl, ps)
            out["p_ans"][w] = star[1][new]
            out["smin"][w] = entropy_min(r, [c[i] for c in L], [c[i] for c in R], Bn[i], g)[new]
            out["kinds"][:, w] = r["kinds"][:, new]
            out["shocks"][:, w] = r["shocks"][:, new]
            for j in range(7):
                star_all[j][w] = star[j][new]
            out["tried"] += 1
            print("  %s delta %+.0e p*x%g: offered %d accepted %d new %d  (%.0f s)" % (
                kn, dl, ps, i.size, good.sum(), new.sum(), time.time() - t0), flush=True)
    out.update({"star_%d" % j: star_all[j] for j in range(7)})
    np.savez_compressed(a.out, **out)
    lo = ~out["ctrl"]
    c = out["ctrl"] & out["ok"]
    dp = np.abs(out["p_ans"][c] - out["p_prod"][c]) / np.abs(out["p_prod"][c])
    print("lost lanes recovered %d / %d; controls answered %d / %d, other root (> 1e-3) on %d" % (
        out["ok"][lo].sum(), lo.sum(), c.sum(), out["ctrl"].sum(), (dp > 1e-3).sum()))


if __name__ == "__main__":
    main()
