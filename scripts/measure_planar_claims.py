"""Re-derive every measured number in docs/planar_solver.tex from a harvest.

    python scripts/measure_planar_claims.py results/rotor_64_exact/harvest \
        [--max 4000] [--dataset /Users/miler/Codes/rmhd_final/data/dataset_rotor_train.npz]

The paper section quotes measurements, not estimates, so they have to be
reproducible from data that outlives the session that produced them.  Each
block below prints one claim and the number behind it, in the order the
section makes them.  A harvest directory from any exact run will do; the
published numbers come from the 64^2 run of 2026-09-08.

What this does NOT re-derive, because it needs a full run rather than a
harvest: the replay counts (use `rotor_replay.py replay`), the per-run
coverage (use `harvest_summary.py`), and the cost per step (the run logs).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RMHD = os.environ.get("RMHD_ROOT", "/Users/miler/Codes/rmhd_final")
if _RMHD not in sys.path:
    sys.path.append(_RMHD)

np.seterr(all="ignore")
GAMMA = 5.0 / 3.0


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _load(d, kind, cap=None, seed=0):
    fs = sorted(glob.glob(os.path.join(d, f"{kind}_*.npz")))
    if not fs:
        return None
    z = [np.load(f) for f in fs]
    keys = set(z[0].files)
    out = {k: np.concatenate([q[k] for q in z]) for k in keys if k != "gamma"}
    n = out["U_L"].shape[0]
    if cap is not None and n > cap:
        i = np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))
        out = {k: v[i] for k, v in out.items()}
    return out


def _cross_ratio(UL, UR):
    num = np.abs(UL[:, 5] * UR[:, 6] - UL[:, 6] * UR[:, 5])
    den = np.hypot(UL[:, 5], UL[:, 6]) * np.hypot(UR[:, 5], UR[:, 6])
    return num / np.maximum(den, 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("harvest")
    ap.add_argument("--max", type=int, default=4000,
                    help="cap for the blocks that run a solver")
    ap.add_argument("--dataset", default=None,
                    help="a forward-constructed dataset, for the contrast in "
                         "the coplanarity claim")
    a = ap.parse_args()

    sol = _load(a.harvest, "solved")
    if sol is None:
        sys.exit(f"no solved shards in {a.harvest}")
    UL, UR, Z, SP = sol["U_L"], sol["U_R"], sol["zones"], sol["speeds"]
    n = UL.shape[0]
    print(f"harvest: {a.harvest}\n{n} solved interfaces with their exact "
          f"structure\n")

    # ── 1. the flow is planar ────────────────────────────────────────────
    chi = _cross_ratio(UL, UR)
    print("1. coplanarity  |Bt_L x Bt_R| / (|Bt_L||Bt_R|)")
    print(f"     median {np.median(chi):.2e}   below 1e-6 on "
          f"{100 * (chi < 1e-6).mean():.1f}% of interfaces")
    if a.dataset and os.path.exists(a.dataset):
        d = np.load(a.dataset)
        k = min(200000, d["U_L"].shape[0])
        cd = _cross_ratio(d["U_L"][:k], d["U_R"][:k])
        print(f"     forward-constructed set: median {np.median(cd):.3f}, "
              f"below 1e-4 on {100 * (cd < 1e-4).mean():.2f}%")

    # ── 2. psi is not determined by the slow wave ────────────────────────
    psi = lambda k: np.arctan2(Z[:, k, 6], Z[:, k, 5])
    dpsi_slow = np.abs(_wrap(psi(3) - psi(2)))          # R3 -> R4
    print("\n2. rotation of the tangential field ACROSS the left slow wave")
    print(f"     median {np.median(dpsi_slow):.2e} rad "
          f"({np.degrees(np.median(dpsi_slow)):.3e} deg)")

    # ── 3. the slow wave is weak, so its reachable set is small ──────────
    Bt3 = np.hypot(Z[:, 2, 5], Z[:, 2, 6])
    Bt4 = np.hypot(Z[:, 3, 5], Z[:, 3, 6])
    rel = np.abs(Bt4 - Bt3) / np.maximum(Bt3, 1e-30)
    print("\n3. change in |Bt| across the left slow wave")
    print(f"     median {100 * np.median(rel):.1f}%   below 1% on "
          f"{100 * (rel < 1e-2).mean():.1f}% of interfaces")

    # ── 4. the Alfven and slow waves are nearly coincident ───────────────
    gap = np.abs(np.diff(SP, axis=1))
    print("\n4. separation of the Alfven and slow speeds")
    for k, nm in ((1, "|Vs_LA - Vs_LS|"), (4, "|Vs_RS - Vs_RA|")):
        print(f"     {nm}: median {np.median(gap[:, k]):.2e}   "
              f"p5 {np.percentile(gap[:, k], 5):.2e}")
    print(f"     for contrast |Vs_LF - Vs_LA|: median "
          f"{np.median(gap[:, 0]):.2e}")

    # ── 5. rotations really are zero, or pi ──────────────────────────────
    phiL = np.abs(_wrap(psi(2) - psi(1)))
    phiR = np.abs(_wrap(psi(6) - psi(5)))
    both0 = ((phiL < 1e-6) & (phiR < 1e-6)).mean()
    print("\n5. Alfven rotations in the exact solutions")
    print(f"     both zero (a true five-wave problem): {100 * both0:.1f}%")
    print(f"     at least one genuine rotation:        {100 * (1 - both0):.1f}%")

    # ── 6. the planar solver, and the verification ───────────────────────
    from batched.ray_scalar import claim_rmhd_namespace
    claim_rmhd_namespace()
    from eos import set_eos
    set_eos("ideal")
    from batched import planar5_b as P5, contact_b as CB, fullcontact_b as FB

    m = min(a.max, n)
    i = np.sort(np.random.default_rng(3).choice(n, m, replace=False))
    sL = [UL[i, j].copy() for j in range(7)]
    sR = [UR[i, j].copy() for j in range(7)]
    Bn = UL[i, 7].copy()

    r5 = P5.make_solver(GAMMA)["solve"](sL, sR, Bn, accuracy=1e-10, max_iter=60)
    c5 = r5["converged"]
    print(f"\n6. five-wave planar solver on {m} of them")
    print(f"     converged {100 * c5.mean():.1f}% (trivial seed, no network)")
    print(f"     full seven-wave residual at its answer: median "
          f"{np.median(r5['full_resid'][c5]):.2e}")
    pc = 0.5 * (r5["zones"][1][1] + r5["zones"][2][1])
    err = np.abs(pc - Z[i, 3, 1]) / np.maximum(np.abs(Z[i, 3, 1]), 1e-30)
    print(f"     contact pressure vs the exact answer: median "
          f"{np.median(err[c5]):.2e}")
    sl = np.maximum(np.abs(r5["slack"][0]), np.abs(r5["slack"][1]))
    print(f"     slow-wave slack (an identity in the plane): median "
          f"{np.median(sl[c5]):.2e}")
    print(f"     planarity residue of the inputs: median "
          f"{np.median(r5['planar_resid']):.2e}, above 1e-6 on "
          f"{100 * (r5['planar_resid'] > 1e-6).mean():.1f}%")

    # ── 7. the same test rejects the three-wave solver ───────────────────
    r3 = CB.make_solver(GAMMA)(sL, sR, Bn, 0, accuracy=1e-10)
    c3 = r3["converged"].astype(bool)
    L4, pb = r3["solutionL"], r3["pb"]
    u6 = [np.log(np.maximum(pb, 1e-30)),
          np.log(np.maximum(np.hypot(L4[5], L4[6]), 1e-30)),
          np.arctan2(L4[6], L4[5]), np.log(np.maximum(pb, 1e-30)),
          np.zeros(m), np.zeros(m)]
    fv, _, _, _, e = FB.make_solver(GAMMA)["fullfuncv6"](sL, sR, u6, Bn)
    n3 = np.where(e | ~c3, np.inf, np.max(np.abs(fv), axis=0))
    print(f"\n7. reduced three-wave solver on the same interfaces")
    print(f"     converged {100 * c3.mean():.1f}%")
    print(f"     full seven-wave residual at ITS answer: median "
          f"{np.median(n3[np.isfinite(n3)]):.2e}, below 1e-6 on "
          f"{100 * (n3 < 1e-6).mean():.1f}%")

    # ── 8. the residual is frame-dependent, in BOTH directions ───────────
    # A solution satisfies the seven-wave residual best in the frame it was
    # SOLVED in.  Rows produced by the seven-wave Newton (raw solver frame,
    # orientation set arbitrarily by the sweep direction) lose accuracy when
    # rotated; rows produced by the planar solver lose it when un-rotated.
    # The lesson is not that one frame is better -- it is that a residual
    # check must use the frame the row was made in, or a looser tolerance.
    p5 = c5
    if p5.any():
        j = np.flatnonzero(p5)
        Zi = Z[i][j]
        li = [c[j] for c in sL]
        ri = [c[j] for c in sR]
        Bi = Bn[j]

        def resid(L, R, Zx, B):
            ps = lambda k: np.arctan2(Zx[:, k, 6], Zx[:, k, 5])
            u = [np.log(np.maximum(Zx[:, 1, 1], 1e-30)),
                 np.log(np.maximum(np.hypot(Zx[:, 3, 5], Zx[:, 3, 6]), 1e-30)),
                 ps(3), np.log(np.maximum(Zx[:, 5, 1], 1e-30)),
                 _wrap(ps(2) - ps(1)), _wrap(ps(6) - ps(5))]
            f, _, _, _, er = FB.make_solver(GAMMA)["fullfuncv6"](L, R, u, B)
            return np.where(er, np.inf, np.max(np.abs(f), axis=0))

        raw = resid(li, ri, Zi, Bi)
        Lp, Rp, al, _ = P5.to_planar(li, ri, Bi)
        Zp = Zi.copy()
        ca, sa = np.cos(al), np.sin(al)
        for k in range(8):
            vy, vz = Zi[:, k, 3].copy(), Zi[:, k, 4].copy()
            Zp[:, k, 3] = ca * vy - sa * vz
            Zp[:, k, 4] = sa * vy + ca * vz
            By, Bz = Zi[:, k, 5].copy(), Zi[:, k, 6].copy()
            Zp[:, k, 5] = ca * By - sa * Bz
            Zp[:, k, 6] = sa * By + ca * Bz
        rot = resid(Lp, Rp, Zp, Bi)
        print("\n8. the seven-wave residual is FRAME-DEPENDENT "
              "(same rows, same solution)")
        print(f"   rows from the SEVEN-wave solver (made in the raw frame):")
        print(f"     raw frame:    below 1e-8 on {100 * (raw < 1e-8).mean():5.1f}%"
              f"   unconstructible {100 * (~np.isfinite(raw)).mean():4.1f}%")
        print(f"     planar frame: below 1e-8 on {100 * (rot < 1e-8).mean():5.1f}%"
              f"   unconstructible {100 * (~np.isfinite(rot)).mean():4.1f}%")
        pl = _load(a.harvest, "solved")
        src = pl.get("source")
        if src is not None and (src == 1).any():
            k = np.flatnonzero(src == 1)[:a.max]
            Zk = pl["zones"][k]
            lk = [pl["U_L"][k, q].copy() for q in range(7)]
            rk = [pl["U_R"][k, q].copy() for q in range(7)]
            Bk = pl["U_L"][k, 7].copy()
            raw2 = resid(lk, rk, Zk, Bk)
            Lq, Rq, aq, _ = P5.to_planar(lk, rk, Bk)
            Zq = Zk.copy()
            cq, sq = np.cos(aq), np.sin(aq)
            for q in range(8):
                vy, vz = Zk[:, q, 3].copy(), Zk[:, q, 4].copy()
                Zq[:, q, 3] = cq * vy - sq * vz
                Zq[:, q, 4] = sq * vy + cq * vz
                By, Bz = Zk[:, q, 5].copy(), Zk[:, q, 6].copy()
                Zq[:, q, 5] = cq * By - sq * Bz
                Zq[:, q, 6] = sq * By + cq * Bz
            rot2 = resid(Lq, Rq, Zq, Bk)
            print(f"   rows from the PLANAR solver (made in the planar frame):")
            print(f"     raw frame:    below 1e-8 on {100 * (raw2 < 1e-8).mean():5.1f}%"
                  f"   unconstructible {100 * (~np.isfinite(raw2)).mean():4.1f}%")
            print(f"     planar frame: below 1e-8 on {100 * (rot2 < 1e-8).mean():5.1f}%"
                  f"   unconstructible {100 * (~np.isfinite(rot2)).mean():4.1f}%")
        else:
            print("   (no planar rows in this harvest for the other direction)")


if __name__ == "__main__":
    main()
