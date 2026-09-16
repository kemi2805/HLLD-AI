"""Rescue stubborn rotor interfaces and write them in the harvest schema.

An interface the production solvers cannot answer is, if we can solve it by
any means, the most valuable training row available: it is exactly the tail
the warm-start network is bad at, and it comes with a verified exact answer.

Two rescue stages, cheapest first, every acceptance gated on the FULL
seven-wave residual at 1e-8 so nothing enters the dataset that is not exact:

  stage 1  planar least-squares descent (3 unknowns).  Cheap.  NOTE most of
           its roots are spurious -- measured, only 15% of the reduced-system
           roots lift to the full system -- so the verification is doing real
           work here, not ceremony.
  stage 2  grid over the feasible set, then descent on the SIX-unknown system
           with the rotations FREE.  This is the only search in the project
           that can represent an intermediate-angle solution, which matters
           because 2.3% of measured rotor rotations are neither 0 nor pi.

Output is the harvest schema (U_L, U_R, zones, speeds, attempts, source,
gamma) so the result merges directly with `harvest_to_dataset.py`.  Rescued
rows carry ``source = 2``.

    python scripts/rescue_collect.py results/rotor_64_exact/harvest \
        --out data/rescued.npz --max 4000
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RMHD = os.environ.get("RMHD_ROOT", "/Users/miler/Codes/rmhd_final")
if _RMHD not in sys.path:
    sys.path.append(_RMHD)

from batched.ray_scalar import claim_rmhd_namespace      # noqa: E402
claim_rmhd_namespace()
from eos import set_eos                                   # noqa: E402
set_eos("ideal")
from batched import fullcontact_b as FB                   # noqa: E402
from batched import planar5_b as P5                       # noqa: E402

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
SOURCE_RESCUE = 2


def _load_unsolved(harvest, cap):
    UL, UR = [], []
    for f in sorted(glob.glob(os.path.join(harvest, "unsolved_*.npz"))):
        d = np.load(f)
        UL.append(d["U_L"]); UR.append(d["U_R"])
        if sum(len(a) for a in UL) >= cap:
            break
    if not UL:
        sys.exit("no unsolved shards in " + harvest)
    return (np.concatenate(UL)[:cap].astype(float),
            np.concatenate(UR)[:cap].astype(float))


def _assemble(left, right, unk6, Bn, f6):
    """Turn accepted unknowns into harvest zones (n,8,7) and speeds (n,7)."""
    fv, zones6, VsLv, VsRv, err = f6(left, right, unk6, Bn)
    n = Bn.shape[0]
    nrm = np.where(err, np.inf, np.max(np.abs(fv), axis=0))
    Z = np.empty((n, 8, 7))
    for j in range(7):
        Z[:, 0, j] = left[j]
        Z[:, 7, j] = right[j]
    for k in range(6):
        for j in range(7):
            Z[:, k + 1, j] = zones6[k][j]
    vxc = 0.5 * (zones6[2][2] + zones6[3][2])          # contact speed, R4/R5
    S = np.stack([VsLv[0], VsLv[1], VsLv[2], vxc, VsRv[2], VsRv[1], VsRv[0]], axis=1)
    return Z, S, nrm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("harvest")
    ap.add_argument("--out", default="data/rescued.npz")
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=5.0 / 3.0)
    ap.add_argument("--grid", type=int, default=11)
    ap.add_argument("--starts", type=int, default=16)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--stage2-max", type=int, default=0,
                    help="interfaces given the expensive stage-2 search; measured "
                         "0 rescues in 30 tries at 6x the per-interface cost of "
                         "stage 1, so it is OFF by default")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1,
                    help="split the interfaces across processes; each writes its "
                         "own --out file")
    a = ap.parse_args()

    G = a.gamma
    W = FB.make_solver(G); f6 = W["fullfuncv6"]
    S5 = P5.make_solver(G)

    UL, UR = _load_unsolved(a.harvest, a.max)
    if a.nshards > 1:
        UL, UR = UL[a.shard::a.nshards], UR[a.shard::a.nshards]
    left = [UL[:, j].copy() for j in range(7)]
    right = [UR[:, j].copy() for j in range(7)]
    Bn = UL[:, 7].copy()
    n0 = len(Bn)
    print("loaded %d interfaces the run failed on" % n0, flush=True)

    base = S5["solve"](left, right, Bn, accuracy=1e-10, max_iter=60)
    todo = np.flatnonzero(~base["converged"])
    print("  planar solver rescues %d; %d remain stubborn"
          % (n0 - todo.size, todo.size), flush=True)

    keep_u = {}                                        # index -> unk6
    nfail = {"n": 0, "first": None}                    # never swallow silently
    theta = np.arctan2(UL[:, 6], UL[:, 5])
    Pm = np.maximum(0.5 * (UL[:, 1] + UR[:, 1]), 1e-12)

    # ---- stage 1: planar descent, verified ----------------------------
    Lp, Rp, alpha, _ = P5.to_planar(left, right, Bn)
    t0 = time.time(); rng = np.random.default_rng(5)
    for i in todo:
        Lo = [np.array([c[i]]) for c in Lp]; Ro = [np.array([c[i]]) for c in Rp]
        Bo = np.array([Bn[i]])

        def r3(u):
            f, _, _, bad, _, _ = S5["structure"](
                Lo, Ro, [np.array([u[0]]), np.array([u[1]]), np.array([u[2]])], Bo)
            return np.full(3, 1e3) if bad[0] else np.asarray(f).ravel()

        lp, lb = np.log(Pm[i]), np.sqrt(2 * Pm[i])
        starts = [np.array([lp, lb, lp]), np.array([lp, -lb, lp]),
                  np.array([lp, 0.5 * lb, lp]), np.array([lp, 0.0, lp])]
        bf, bx = np.inf, None
        for u0 in starts:
            try:
                r = least_squares(r3, u0, method="trf", xtol=1e-14, ftol=1e-14,
                                  gtol=1e-14, max_nfev=600)
                v = float(np.max(np.abs(r.fun)))
                if v < bf:
                    bf, bx = v, r.x
            except Exception as exc:
                nfail["n"] += 1
                if nfail["first"] is None:
                    nfail["first"] = repr(exc)
        if bf <= a.tol and bx is not None:
            # lift to the seven-wave unknowns in the PLANAR frame, then verify
            u6 = [np.array([bx[0]]), np.log(np.maximum(np.abs(bx[1]), 1e-30)),
                  np.array([0.0 if bx[1] >= 0 else np.pi]), np.array([bx[2]]),
                  np.zeros(1), np.zeros(1)]
            u6 = [np.atleast_1d(np.asarray(c, dtype=float)) for c in u6]
            full = S5["verify_full"]([np.array([c[i]]) for c in Lp],
                                     [np.array([c[i]]) for c in Rp],
                                     [np.array([bx[0]]), np.array([bx[1]]),
                                      np.array([bx[2]])], Bo)
            if float(full[0]) <= a.tol:
                keep_u[int(i)] = ("planar", bx, float(full[0]))
    print("  stage 1 (planar descent, verified): %d rescued  (%.0f s)"
          % (len(keep_u), time.time() - t0), flush=True)
    if nfail["n"]:
        print("  WARNING: %d descent calls raised; first was %s"
              % (nfail["n"], nfail["first"]), flush=True)
        if nfail["n"] > 0.5 * max(todo.size, 1) * 4:
            sys.exit("stage 1 failed on most starts -- refusing to write a "
                     "dataset from a broken run; fix the error above")

    # ---- stage 2: feasible grid + six-unknown descent, rotations free ---
    rest = [int(i) for i in todo if int(i) not in keep_u][:a.stage2_max]
    ax = np.linspace(-2.0, 2.0, a.grid)
    A, B, C = np.meshgrid(ax, ax, ax, indexing="ij")
    dpL, db, dpR = A.ravel(), B.ravel(), C.ravel()
    m = dpL.size
    t0 = time.time(); n2 = 0
    for i in rest:
        rep = lambda c: np.repeat(c[i], m)
        LL = [rep(c) for c in left]; RR = [rep(c) for c in right]
        BB = np.repeat(Bn[i], m)
        cf, cu = [], []
        for acd in (theta[i], theta[i] + np.pi):
            for pl in (0.0, np.pi):
                for pr in (0.0, np.pi):
                    unk = [np.log(Pm[i]) + dpL, np.log(np.sqrt(2 * Pm[i])) + db,
                           np.full(m, acd), np.log(Pm[i]) + dpR,
                           np.full(m, pl), np.full(m, pr)]
                    fv, _, _, _, e = f6(LL, RR, unk, BB)
                    nrm = np.where(e, np.inf, np.max(np.abs(fv), axis=0))
                    g = np.isfinite(nrm)
                    if g.any():
                        cf.append(nrm[g]); cu.append(np.stack(unk, axis=1)[g])
        if not cf:
            continue
        cfa = np.concatenate(cf); cua = np.concatenate(cu, axis=0)
        Lo = [np.array([c[i]]) for c in left]; Ro = [np.array([c[i]]) for c in right]
        Bo = np.array([Bn[i]])
        last = {"u": None}

        def r6(u):
            fv, _, _, _, e = f6(Lo, Ro, [np.array([x]) for x in u], Bo)
            if e[0]:
                d = np.linalg.norm(u - last["u"]) if last["u"] is not None else 1.0
                return np.full(6, 1e2 * (1.0 + d))
            last["u"] = u.copy()
            return np.asarray(fv).ravel()

        bf, bx = np.inf, None
        for k in np.argsort(cfa)[:a.starts]:
            last["u"] = cua[k].copy()
            try:
                r = least_squares(r6, cua[k], method="trf", xtol=1e-15,
                                  ftol=1e-15, gtol=1e-15, max_nfev=1500)
                v = float(np.max(np.abs(r.fun)))
                if v < bf:
                    bf, bx = v, r.x
            except Exception:
                pass
        if bf <= a.tol and bx is not None:
            keep_u[i] = ("sevenwave", bx, bf); n2 += 1
    print("  stage 2 (feasible grid + 6-unknown descent): %d rescued of %d tried"
          "  (%.0f s)" % (n2, len(rest), time.time() - t0), flush=True)

    if not keep_u:
        print("nothing rescued; no file written")
        return

    idx = sorted(keep_u)
    rows_u6, attempts = [], []
    for i in idx:
        kind, x, _ = keep_u[i]
        if kind == "planar":
            # planar answer lives in the rotated frame; undo the rotation on
            # the contact orientation so the unknowns are in the solver frame
            psi = (0.0 if x[1] >= 0 else np.pi) - alpha[i]
            rows_u6.append([x[0], np.log(max(abs(x[1]), 1e-30)), psi, x[2], 0.0, 0.0])
            attempts.append(3)
        else:
            rows_u6.append(list(x)); attempts.append(4)
    U6 = np.array(rows_u6)
    sub_l = [c[idx] for c in left]; sub_r = [c[idx] for c in right]
    Z, SPD, nrm = _assemble(sub_l, sub_r, [U6[:, j].copy() for j in range(6)],
                            Bn[idx], f6)
    # Admissibility: the jump conditions say nothing about wave ORDER, and a
    # wave that both crosses its neighbour AND carries a jump is a
    # self-crossing fan, not a solution.  A crossing between ZERO-strength
    # waves is harmless -- nothing depends on where a wave with no jump sits,
    # and the planar path leaves both rotations at zero strength by
    # construction, so it crosses on ~90% of rows with no consequence.
    # Measured: this gate removes 0.12% of rescued rows and would remove
    # 2.52% of the production harvest.
    d = np.diff(SPD, axis=1)
    def _rot_jump(A, B):
        return np.maximum(
            np.abs(Z[:, A, 0] - Z[:, B, 0]) / np.maximum(np.abs(Z[:, A, 0]), 1e-30),
            np.sqrt(sum((Z[:, A, k] - Z[:, B, k]) ** 2 for k in (2, 3, 4))))
    crossing = ((d[:, 1] < -1e-6) & (_rot_jump(1, 2) > 1e-8)) | \
               ((d[:, 4] < -1e-6) & (_rot_jump(5, 6) > 1e-8)) | \
               (d[:, [0, 2, 3, 5]] < -1e-6).any(axis=1)
    good = (nrm <= a.tol) & ~crossing
    print("  final verification in the solver frame: %d of %d pass "
          "(%d rejected as inadmissible fans)"
          % (int(good.sum()), len(idx), int((nrm <= a.tol).sum() - good.sum())),
          flush=True)
    if not good.any():
        print("nothing survived the final check; no file written")
        return
    k = np.flatnonzero(good)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(
        a.out,
        U_L=UL[np.array(idx)[k]], U_R=UR[np.array(idx)[k]],
        zones=Z[k], speeds=SPD[k],
        attempts=np.array(attempts, dtype=np.int16)[k],
        source=np.full(k.size, SOURCE_RESCUE, dtype=np.int8),
        gamma=np.float64(G))
    print("wrote %s: %d verified-exact rescued interfaces (%.1f%% of the "
          "stubborn set)" % (a.out, k.size, 100.0 * k.size / max(todo.size, 1)))


if __name__ == "__main__":
    main()
