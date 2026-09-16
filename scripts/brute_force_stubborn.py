"""Brute force on stubborn rotor interfaces, box sized from measured answers.

Three iterations of this have taught the box, and the box is the whole game:

  v1  +-2 in ln around ln(P_bar) and ln(sqrt(2 P_bar)).  MISSED 35.6% of
      known answers.  Control recovery 82.5%.
  v2  +-1.5 (pressures) and +-9.0 (field), equal spacing.  Control 90.0%.
  v3  (here) the field range made ASYMMETRIC.  Measured over 24,632 control
      answers, the signed offset of ln|Bt*| from ln sqrt(2 P_bar) runs
      -13.11 to +0.47, with 99.85% BELOW the centre and nothing beyond +4.
      The physical bound |Bt*| <= W sqrt(2 P_tot*) caps the upside at +1.62.
      So v2 spent half its 61 field points above the centre, where 0.15% of
      answers live.  Same budget, range [-16, +2]: every point is useful and
      the reachable region extends 7 e-folds deeper.

That direction is not incidental.  Solutions run toward Bt* -> 0, which is
the switch-off limit -- and it is exactly where the log-polar unknowns
degenerate, because the ORIENTATION psi_CD of a vanishing field is
unobservable.  If the stubborn interfaces are small-Bt* solutions, no
enumeration of psi_CD could ever have found them.

Rotations are gridded over the full circle and free in the descent; the
objective is the complete six-component residual, so any root is exact by
construction.

**A zero residual is not a solution.**  The jump conditions do not order the
waves, and measured 2026-09-12 on the four completed runs, 47-57% of the
stubborn "roots" and 11-12% of the control ones were self-crossing fans --
a rotational discontinuity that crosses its slow wave while carrying a jump.
Ungated, that inflated the stubborn rate 2x (7.7-9.0% reported, 3.9-4.1%
admissible) and manufactured a spurious win for one box.  Worse, the search
STOPPED at the first zero residual, so a fan ended the hunt for the real
root.  ``_admissible`` applies the same rule as ``exact_flux.py``
(RMHD_WAVE_ORDER) and ``rescue_collect.py``; a fan is skipped and the descents
continue.  ``*_best_raw`` in the output keeps the ungated number for the record.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
import warnings

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmhd.eos import set_eos                                       # noqa: E402
set_eos("ideal")
from rmhd.batched import fullcontact_b as FB                       # noqa: E402
from rmhd.batched import planar5_b as P5                           # noqa: E402
import torch                                                  # noqa: E402
from src.physics.eos import hybrid_eos                        # noqa: E402
from src.physics.hlld import compute_srmhd_fluxes             # noqa: E402

warnings.filterwarnings("ignore")
np.seterr(all="ignore")


def _bt_centre_from_hlld(UL, UR, gamma):
    """Per-interface field scale from HLLD's HLL average state (user, 2026-09-10).

    HLLD already runs on every interface as the fallback, so this is free.
    MEASURED against 24,632 known answers: ln|Bt*| sits within [-1.54, +1.37]
    of ln|Bt_hll| at p1..p99 -- width 2.90 -- against [-7.53, -0.13], width
    7.40, for the global ln sqrt(2 P_bar).  Same points, 2.5x the resolution.

    Its sibling `p_star` is NOT used: it is tighter than the global centre in
    the median but collapses toward zero on a few percent of interfaces (the
    B_n -> 0 degeneracy), giving a p99 offset of +690 in ln.  The global
    pressure centre is already excellent (width 0.48), so there is nothing to
    gain and an e^690 mis-centring to lose.
    """
    eos = hybrid_eos(K=0.0, gamma=gamma, gamma_th=gamma)
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float64)

    def prim(U):
        rho, Pt, vx, vy, vz, By, Bz = (U[:, k] for k in range(7))
        Bx = U[:, 7]
        W2 = 1.0 / (1.0 - (vx ** 2 + vy ** 2 + vz ** 2))
        eta = Bx * vx + By * vy + Bz * vz
        b2 = (Bx ** 2 + By ** 2 + Bz ** 2) / W2 + eta ** 2
        d = {"rho": t(rho), "p": t(np.maximum(Pt - 0.5 * b2, 1e-12)),
             "vx": t(vx), "vy": t(vy), "vz": t(vz),
             "Bx": t(Bx), "By": t(By), "Bz": t(Bz)}
        d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
        return d

    uL, uR, fL, fR, cmax, cmin = compute_srmhd_fluxes(prim(UL), prim(UR), eos, 0)
    den = np.maximum((cmax + cmin).numpy(), 1e-300)
    comp = [((cmax * uR[k] + cmin * uL[k] + fL[k] - fR[k]).numpy()) / den
            for k in ("By", "Bz")]
    bt = np.hypot(*comp)
    Pm = np.maximum(0.5 * (UL[:, 1] + UR[:, 1]), 1e-12)
    fallback = np.sqrt(2 * Pm)
    ok = np.isfinite(bt) & (bt > 1e-300)
    return np.log(np.where(ok, bt, fallback)), int((~ok).sum())


def _field_axis(blo, bhi, nb, hlld, bcore=0.15, btail=0.6):
    """Uniform for the global box; dense-core + sparse-tails for the HLLD one.

    With the HLLD centre 98% of answers fall inside +-1.6, so a uniform grid
    over the full [-7, +9] safety range would waste most of its points.  Dense
    at ``bcore`` where the answers are, coarse at ``btail`` in the tails that
    only exist to catch the min/max (-5.90 / +8.10 measured).

    ``bcore``/``btail`` default to the spacings this box shipped with, so the
    axis is unchanged unless they are passed explicitly.  They exist because
    the field axis carries 91.6% of the squared distance from the grid to the
    true answer (measured over 1280 known answers: ln p_LF and ln p_RF
    contribute 4.2% each, their true offsets having median +0.004), so
    spending the point budget here rather than on the pressures is what
    shrinks the seed error.
    """
    if not hlld:
        return np.linspace(blo, bhi, nb)
    core = np.arange(-2.5, 2.5 + 1e-9, bcore)
    lo = np.arange(blo, -2.5, btail)
    hi = np.arange(2.5 + btail, bhi + 1e-9, btail)
    return np.unique(np.concatenate([lo, core, hi]))


def _admissible(f6, Lo, Ro, Bo, u, tol=1e-6):
    """True if the structure at ``u`` has its waves in order.

    Rejects a rotational discontinuity that crosses its slow wave AND carries a
    jump (a zero-strength crossing is harmless: the planar family crosses on
    ~90% of its answers with silent rotations), and any inversion among the
    other waves.  Single lane; ``Lo``/``Ro``/``Bo`` are length-1 arrays.
    """
    _, Z, VL, VR, e = f6(Lo, Ro, [np.array([x]) for x in u], Bo)
    if bool(np.asarray(e).ravel()[0]):
        return False
    z = [[float(np.asarray(c).ravel()[0]) for c in R] for R in Z]   # R2..R7
    vl = [float(np.asarray(c).ravel()[0]) for c in VL]              # LF, LA, LS
    vr = [float(np.asarray(c).ravel()[0]) for c in VR]              # RF, RA, RS
    cd = 0.5 * (z[2][2] + z[3][2])
    order = [vl[0], vl[1], vl[2], cd, vr[2], vr[1], vr[0]]
    d = [order[k + 1] - order[k] for k in range(6)]

    def strength(A, B):
        rho = max(abs(z[A][0]), 1e-30)
        return max(abs(z[A][0] - z[B][0]) / rho,
                   math.sqrt(sum((z[A][k] - z[B][k]) ** 2 for k in (2, 3, 4))))
    if d[1] < -tol and strength(0, 1) > 1e-8:
        return False
    if d[4] < -tol and strength(4, 5) > 1e-8:
        return False
    return not any(d[k] < -tol for k in (0, 2, 3, 5))


def run(UL, UR, label, *, f6, S5, nb, blo, bhi, np_, hp, nrot, kstart, tol,
        hlld_box=False, gamma=5.0 / 3.0, bcore=0.15, btail=0.6):
    n = len(UL)
    left = [UL[:, j].copy() for j in range(7)]
    right = [UR[:, j].copy() for j in range(7)]
    Bn = UL[:, 7].copy()
    theta = np.arctan2(UL[:, 6], UL[:, 5])
    Pm = np.maximum(0.5 * (UL[:, 1] + UR[:, 1]), 1e-12)
    axp = np.linspace(-hp, hp, np_)
    if hlld_box:
        bt_ctr, nbad = _bt_centre_from_hlld(UL, UR, gamma)
        axb = _field_axis(blo, bhi, nb, True, bcore, btail)
        print("    field box: HLLD-centred, %d pts (core %.3f, tails %.3f), "
              "%d fell back to sqrt(2P)" % (axb.size, bcore, btail, nbad),
              flush=True)
    else:
        bt_ctr = np.log(np.sqrt(2 * Pm))
        axb = _field_axis(blo, bhi, nb, False)
    rot = np.linspace(-np.pi, np.pi, nrot, endpoint=False)
    A, B, C = np.meshgrid(axp, axb, axp, indexing="ij")
    dpL, db, dpR = A.ravel(), B.ravel(), C.ravel()
    m = dpL.size
    best = np.full(n, np.inf)        # best ADMISSIBLE (or not-yet-converged) residual
    best_raw = np.full(n, np.inf)    # best of anything, for the record
    bu = np.zeros((n, 6))
    n_fan = 0
    n_exc = 0                        # descents that raised -- must never be silent
    n_nogrid = 0                     # interfaces whose grid produced no feasible point
    last_exc = [None]
    t0 = time.time()
    for i in range(n):
        rep = lambda c: np.repeat(c[i], m)
        LL = [rep(c) for c in left]; RR = [rep(c) for c in right]
        BB = np.repeat(Bn[i], m)
        cf, cu = [], []
        for acd in (theta[i], theta[i] + np.pi):
            for pl in rot:
                for pr in rot:
                    unk = [np.log(Pm[i]) + dpL,
                           bt_ctr[i] + db,
                           np.full(m, acd), np.log(Pm[i]) + dpR,
                           np.full(m, pl), np.full(m, pr)]
                    fv, _, _, _, e = f6(LL, RR, unk, BB)
                    nrm = np.where(e, np.inf, np.max(np.abs(fv), axis=0))
                    g = np.isfinite(nrm)
                    if g.any():
                        cf.append(nrm[g]); cu.append(np.stack(unk, axis=1)[g])
        if not cf:
            n_nogrid += 1
            continue
        cfa = np.concatenate(cf); cua = np.concatenate(cu, axis=0)
        Lo = [np.array([c[i]]) for c in left]; Ro = [np.array([c[i]]) for c in right]
        Bo = np.array([Bn[i]]); last = {"u": None}

        def r6(u):
            fv, _, _, _, e = f6(Lo, Ro, [np.array([x]) for x in u], Bo)
            if e[0]:
                d = np.linalg.norm(u - last["u"]) if last["u"] is not None else 1.0
                return np.full(6, 1e2 * (1.0 + d))
            last["u"] = u.copy()
            return np.asarray(fv).ravel()

        for k in np.argsort(cfa)[:kstart]:
            last["u"] = cua[k].copy()
            try:
                r = least_squares(r6, cua[k], method="trf", xtol=1e-15,
                                  ftol=1e-15, gtol=1e-15, max_nfev=1500)
                v = float(np.max(np.abs(r.fun)))
                best_raw[i] = min(best_raw[i], v)
                if v <= 1e-6 and not _admissible(f6, Lo, Ro, Bo, r.x):
                    n_fan += 1        # a self-crossing fan is not a solution
                    continue          # -- and must not end the search
                if v < best[i]:
                    best[i] = v; bu[i] = r.x
            except Exception as exc:
                n_exc += 1
                last_exc[0] = "%s: %s" % (type(exc).__name__, exc)
            if best[i] <= tol:
                break
    fin = np.isfinite(best)
    got = best <= tol
    if fin.any():
        print("  %-32s n=%3d  best ||f||: median %.2e  p10 %.2e  min %.2e"
              % (label, n, np.median(best[fin]), np.percentile(best[fin], 10),
                 best[fin].min()), flush=True)
    else:
        print("  %-32s n=%3d  no admissible residual on any interface" % (label, n),
              flush=True)
    print("  %-32s        EXACT: %5.1f%%   <=1e-6: %5.1f%%   (%.0f s)"
          % ("", 100 * got.mean(), 100 * np.mean(best <= 1e-6), time.time() - t0),
          flush=True)
    print("  %-32s        fans rejected: %d descents hit <=1e-6 with crossed waves; "
          "ungated EXACT would read %.1f%%"
          % ("", n_fan, 100 * np.mean(best_raw <= tol)), flush=True)
    if n_exc or n_nogrid:
        print("  %-32s        WARNING: %d descents raised, %d interfaces had no "
              "feasible grid point%s"
              % ("", n_exc, n_nogrid,
                 ("; last: " + last_exc[0]) if last_exc[0] else ""), flush=True)
    if got.any():
        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
        mx = np.maximum(np.abs(wrap(bu[got, 4])), np.abs(wrap(bu[got, 5])))
        off = bu[got, 1] - bt_ctr[got]
        print("  %-32s        rotations: 0 %.0f%%  pi %.0f%%  INTERMEDIATE %.0f%%"
              % ("", 100 * np.mean(mx < 1e-6), 100 * np.mean(np.abs(mx - np.pi) < 1e-3),
                 100 * np.mean((mx >= 1e-6) & (np.abs(mx - np.pi) >= 1e-3))), flush=True)
        print("  %-32s        ln|Bt*| offset of the solutions: median %+.2f  "
              "min %+.2f  (v2 box stopped at -9)"
              % ("", np.median(off), off.min()), flush=True)
    return best, bu, best_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("harvest")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--nb", type=int, default=61)
    ap.add_argument("--blo", type=float, default=-16.0)
    ap.add_argument("--bhi", type=float, default=2.0)
    ap.add_argument("--npres", type=int, default=11)
    ap.add_argument("--hp", type=float, default=1.5)
    ap.add_argument("--nrot", type=int, default=8)
    ap.add_argument("--kstart", type=int, default=30)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--gamma", type=float, default=5.0 / 3.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1,
                    help="split the interfaces across processes; each writes "
                         "its own --out file")
    ap.add_argument("--out", default=None,
                    help="save per-interface best residuals and unknowns")
    ap.add_argument("--group", default="both", choices=("both","control","stubborn"))
    ap.add_argument("--bcore", type=float, default=0.15,
                    help="HLLD box only: spacing of the dense core over +-2.5")
    ap.add_argument("--btail", type=float, default=0.6,
                    help="HLLD box only: spacing of the sparse tails")
    ap.add_argument("--hlld-box", action="store_true",
                    help="centre the field box on HLLD's HLL average state, "
                         "per interface, with a dense core and sparse tails")
    a = ap.parse_args()

    G = a.gamma
    W = FB.make_solver(G); f6 = W["fullfuncv6"]
    S5 = P5.make_solver(G)
    kw = dict(f6=f6, S5=S5, nb=a.nb, blo=a.blo, bhi=a.bhi, np_=a.npres,
              hp=a.hp, nrot=a.nrot, kstart=a.kstart, tol=a.tol,
              hlld_box=a.hlld_box, gamma=G, bcore=a.bcore, btail=a.btail)

    def load(kind, cap=6000):
        fs = sorted(glob.glob(os.path.join(a.harvest, "%s_*.npz" % kind)))[:3]
        L = np.concatenate([np.load(f)["U_L"] for f in fs]).astype(float)[:cap]
        R = np.concatenate([np.load(f)["U_R"] for f in fs]).astype(float)[:cap]
        return L, R

    if a.hlld_box:
        print("BRUTE FORCE: field box CENTRED PER INTERFACE on HLLD's HLL average\n"
              "  state, offsets [%.1f, %.1f].\n"
              "  Measured over 24,632 known answers: the true ln|Bt*| lies within\n"
              "  [-1.54, +1.37] of that centre (p1..p99) against [-7.53, -0.13] of\n"
              "  the global ln sqrt(2 P_bar); 1.3%% of interfaces need more than\n"
              "  +-3, against 25.6%% for the global centre.\n"
              "  Core spacing %.3f over +-2.5, tails %.3f.\n"
              "  ln p +-%.1f (%d pts).  %d rotations each over the full circle.  "
              "%d descents.\n"
              % (a.blo, a.bhi, a.bcore, a.btail, a.hp, a.npres, a.nrot,
                 a.kstart), flush=True)
    else:
        print("BRUTE FORCE: field box GLOBAL, ln|Bt| in [%.1f, %.1f] around\n"
              "  ln sqrt(2 P_bar) (%d pts, spacing %.2f) -- asymmetric, because\n"
              "  99.85%% of known answers lie BELOW that centre and none above +0.47.\n"
              "  ln p +-%.1f (%d pts).  %d rotations each over the full circle.  "
              "%d descents.\n"
              % (a.blo, a.bhi, a.nb, (a.bhi - a.blo) / (a.nb - 1), a.hp, a.npres,
                 a.nrot, a.kstart), flush=True)
    saved = {}
    if a.group in ("both", "control"):
        SL, SR = load("solved")
        SL, SR = SL[:a.n], SR[:a.n]
        if a.nshards > 1:
            SL, SR = SL[a.shard::a.nshards], SR[a.shard::a.nshards]
        b, u, br = run(SL, SR, "CONTROL (root known)", **kw)
        saved.update(control_best=b, control_unk=u, control_best_raw=br,
                     control_UL=SL, control_UR=SR)
    if a.group in ("both", "stubborn"):
        UL, UR = load("unsolved")
        r = S5["solve"]([UL[:, j].copy() for j in range(7)],
                        [UR[:, j].copy() for j in range(7)], UL[:, 7].copy(),
                        accuracy=1e-10, max_iter=60)
        stub = np.flatnonzero(~r["converged"])[:a.n]
        SU, SV = UL[stub], UR[stub]
        if a.nshards > 1:
            SU, SV = SU[a.shard::a.nshards], SV[a.shard::a.nshards]
        b, u, br = run(SU, SV, "STUBBORN (7-wave AND planar fail)", **kw)
        saved.update(stub_best=b, stub_unk=u, stub_best_raw=br,
                     stub_UL=SU, stub_UR=SV)
    if a.out and saved:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        np.savez_compressed(a.out, **saved)
        print("  wrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
