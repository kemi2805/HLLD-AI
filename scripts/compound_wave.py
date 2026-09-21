"""A compound slow wave for the planar solver -- experimental, not production.

Why
---
The tube test showed the stubborn rotor interfaces reverse B_t inside a jump
that also moves rho and P.  Across any discontinuity (u = v_x - s, a the
normal Alfven speed, j = rho u continuous)

    rho_1 (u_1^2 - a_1^2) B_t1  =  rho_2 (u_2^2 - a_2^2) B_t2

so only a wave whose flow CROSSES the Alfven speed -- an intermediate shock
-- can reverse the field.  compound_planar.py showed that a single
intermediate shock rarely closes the planar system.  The classic Brio-Wu
structure is not one jump but two glued together: an intermediate shock
whose downstream flow moves at exactly the slow characteristic speed, so a
slow rarefaction can ride attached to its back.

The element
-----------
From the state A ahead of it (behind the fast wave), in the planar frame:

1. **Sonic intermediate shock.**  Scan the reversed field b behind the shock
   (sign opposite to A's), solve the slow-family shock to each b with the
   search window opened past the Alfven speed (``slow_shock_b.MARGIN``,
   numpy path), and pick the b where  xi_slow(S(b)) = Vs(b):  the flow
   behind the shock is sonic.  S depends on A alone.
2. **Attached slow rarefaction** from S to the contact's |B_t| (a slow fan
   grows |B_t|, so the contact field must be at least |b|).

The element keeps ONE free parameter, the contact's signed B_t, so the
planar system stays three unknowns [ln p_LF, B_t*, ln p_RF] against the
three contact jumps.

Verification cannot use the seven-wave residual -- a compound answer is by
construction not a seven-wave solution -- so an answer counts when the
contact jumps close (<= 1e-8), the shock satisfies its own jump conditions
(``jump.rh_resid`` <= 1e-8), it is sonic behind (|xi_s(S) - Vs| <= 1e-6), and
the waves are in order.

Validation: Balsara test 1 (the relativistic Brio-Wu problem, gamma = 2)
----------------------------------------------------------------------
Its stored exact answer (Giacomazzo & Rezzolla) reverses the field on the
left with a pi-rotation -- the REGULAR solution.  ``--balsara1`` solves it
both ways and runs the actual PDE as a tube at three resolutions: if the
element is right, the tube converges to the compound answer, not the regular
one, which is the non-uniqueness of coplanar ideal-(R)MHD Riemann problems
made visible.

    python scripts/compound_wave.py --balsara1 --out balsara1_compound.npz

MEASURED 2026-09-21 (calea job 811), and it cuts against the premise: the
element is sound -- it converges from 3 of 4 seeds, contact 5e-14, shock
jump conditions 1e-16, sonic 3e-13, the shock intermediate -- and Balsara 1
has TWO exact solutions: this compound one, and the one production's planar
solver already finds (the stored Giacomazzo-Rezzolla answer to nine digits,
residual 1e-12, no pi-rotation), whose left slow wave is a single
intermediate shock admitted through slow_shock_b.MARGIN = 0.03.  They differ
by 7.7e-4 in the density at the contact.  The PDE picks the single shock:
the tube's distance to it falls 1.6e-3 -> 5.0e-4 -> 7e-5 over 512 -> 2048
cells while its distance to the compound answer stalls near 7e-4, and B_t
jumps +0.66 -> -0.44 and sits flat, with no trailing fan.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
from rmhd.eos import set_eos                                       # noqa: E402
set_eos("ideal")
from rmhd.batched import fullcontact_b as FB                       # noqa: E402
from rmhd.batched import planar5_b as P5                           # noqa: E402
from rmhd.batched import rarefaction_b as RB                       # noqa: E402
from rmhd.batched import slow_shock_b as SSB                       # noqa: E402
from rmhd.batched import slow_shock_njit as SSN                    # noqa: E402
from rmhd.batched.jump import rh_resid                             # noqa: E402

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

WIDE_MARGIN = 1.0
ACC = 1e-10
VERIFY = 1e-8
SONIC_TOL = 1e-6


def sub(state, i):
    return [c[i] for c in state]


def tile(state, k):
    return [np.tile(c, k) for c in state]


@contextlib.contextmanager
def wide_window():
    """Open the slow-shock search past the Alfven speed, then restore it."""
    old = SSB.MARGIN
    SSB.MARGIN = WIDE_MARGIN
    try:
        yield
    finally:
        SSB.MARGIN = old


def make(gamma):
    if SSN.wanted():
        raise SystemExit("RMHD_SLOWSHOCK=njit: the window patch reaches only the "
                         "numpy slow-shock path -- unset it")
    W = FB.make_waves(gamma)
    fast_wave, slow_wave6, xi = W["fast_wave"], W["slow_wave6"], W["xi"]
    alfven_speed = W["alfven_speed"]
    shock = SSB.make_solver(gamma)
    _, fan_normB = RB.make_integrators(gamma, lambda s, sw, B, g: xi(s, sw, B))

    def sonic_shock(A, switch, Bn, ngrid=20, nbisect=30):
        """``(S, Vs, b, ok)``: the intermediate shock from A that reverses B_t
        and leaves the flow behind it sonic.  One batched shock solve per grid
        point (all lanes x points stacked) and per bisection step."""
        n = Bn.size
        sgn = np.where(A[5] >= 0.0, -1.0, 1.0)            # the REVERSED sign
        scale = np.maximum(np.abs(A[5]), 1e-3 * np.sqrt(np.abs(A[1])))
        lg = np.linspace(np.log(1e-4), np.log(4.0), ngrid)

        def g_at(lb, idx):
            b = sgn[idx] * scale[idx] * np.exp(lb)
            with wide_window():
                V, S, e = shock(b, np.zeros_like(b), sub(A, idx), switch,
                                Bn[idx], accuracy=1e-12)
            gv = xi(S, switch, Bn[idx]) - V
            bad = np.asarray(e, bool) | ~np.isfinite(gv)
            return np.where(bad, np.nan, gv), V, S, b

        # the grid, every lane x point in one call
        idx_all = np.tile(np.arange(n), ngrid)
        lb_all = np.repeat(lg, n)
        G, _, _, _ = g_at(lb_all, idx_all)
        G = G.reshape(ngrid, n)
        lo = np.full(n, np.nan)
        hi = np.full(n, np.nan)
        glo = np.full(n, np.nan)
        for k in range(ngrid - 1):                        # smallest |b| first
            ch = (np.isfinite(G[k]) & np.isfinite(G[k + 1])
                  & (np.sign(G[k]) != np.sign(G[k + 1])) & np.isnan(lo))
            lo[ch], hi[ch], glo[ch] = lg[k], lg[k + 1], G[k][ch]
        ok = np.isfinite(lo)
        i = np.flatnonzero(ok)
        for _ in range(nbisect):
            if i.size == 0:
                break
            mid = 0.5 * (lo[i] + hi[i])
            gm, _, _, _ = g_at(mid, i)
            dead = ~np.isfinite(gm)
            ok[i[dead]] = False
            same = np.sign(gm) == np.sign(glo[i])
            lo[i] = np.where(same, mid, lo[i])
            glo[i] = np.where(same, gm, glo[i])
            hi[i] = np.where(same, hi[i], mid)
            i = i[~dead]
        S = [np.array(c, dtype=float, copy=True) for c in A]
        V = np.full(n, np.nan)
        b = np.full(n, np.nan)
        i = np.flatnonzero(ok)
        if i.size:
            gm, Vi, Si, bi = g_at(0.5 * (lo[i] + hi[i]), i)
            for c in range(7):
                S[c][i] = Si[c]
            V[i], b[i] = Vi, bi
            ok[i] &= np.isfinite(gm) & (np.abs(gm) <= SONIC_TOL)
        return S, V, b, ok

    def compound(Bt_t, A, switch, Bn):
        """The whole element: ``(B, Vshock, Vtail, S, b, err)``."""
        S, V, b, ok = sonic_shock(A, switch, Bn)
        ok = ok & (np.sign(Bt_t) == np.sign(b)) & (np.abs(Bt_t) >= np.abs(b))
        B = [np.array(c, dtype=float, copy=True) for c in S]
        Vt = V.copy()
        i = np.flatnonzero(ok & (np.abs(Bt_t) > np.abs(b) * (1 + 1e-12)))
        err = ~ok
        if i.size:
            y, _, e = fan_normB(np.abs(Bt_t[i]), sub(S, i), switch, Bn[i])
            y[5], y[6] = Bt_t[i], np.zeros(i.size)        # the contact's field
            for c in range(7):
                B[c][i] = y[c]
            Vt[i] = xi(y, switch, Bn[i])
            err[i] |= np.asarray(e, bool)
        return B, V, Vt, S, b, err

    def structure(L, R, unk3, Bn, mode):
        """Planar residual with the compound element on the sides in ``mode``."""
        n = Bn.size
        p_LF, Bt, p_RF = np.exp(unk3[0]), unk3[1], np.exp(unk3[2])
        z = np.zeros(n)
        A, VLF, eA = fast_wave(p_LF, L, "LF", Bn)
        D, VRF, eD = fast_wave(p_RF, R, "RF", Bn)
        info = {}
        if mode[0]:
            B, Vsh, Vtl, SL, bL, eB = compound(Bt, A, "LS", Bn)
            info["L"] = (Vsh, Vtl, SL, bL)
        else:
            B, _, _, eB = slow_wave6(Bt, z, A, "LS", Bn)
        if mode[1]:
            C, Vsh, Vtl, SR, bR, eC = compound(Bt, D, "RS", Bn)
            info["R"] = (Vsh, Vtl, SR, bR)
        else:
            C, _, _, eC = slow_wave6(Bt, z, D, "RS", Bn)
        f = np.stack([B[2] - C[2], B[3] - C[3], B[1] - C[1]])
        err = eA | eB | eC | eD | ~np.isfinite(f).all(axis=0)
        f = np.where(err[None], 0.0, f)
        return f, err, dict(A=A, B=B, C=C, D=D, VLF=VLF, VRF=VRF, **info)

    def solve(L, R, Bn, unk3, mode, max_iter=40):
        """Damped Newton on the three unknowns, forward-difference Jacobian."""
        u = [np.array(c, dtype=float, copy=True) for c in unk3]
        n = Bn.size
        f, err, _ = structure(L, R, u, Bn, mode)
        nrm = np.where(err, np.inf, np.max(np.abs(f), axis=0))
        live = ~err & (nrm > ACC)
        for _ in range(max_iter):
            i = np.flatnonzero(live)
            if i.size == 0:
                break
            ui = [c[i] for c in u]
            h = np.stack([1e-6 * np.maximum(1.0, np.abs(c)) for c in ui])
            probes = [np.concatenate([ui[c] + (h[c] if c == j else 0.0)
                                      for j in range(3)]) for c in range(3)]
            f2, e2, _ = structure(tile(sub(L, i), 3), tile(sub(R, i), 3), probes,
                                  np.tile(Bn[i], 3), mode)
            J = (f2.reshape(3, 3, i.size) - f[:, i][:, None, :]) / h[None]
            bad = e2.reshape(3, i.size).any(axis=0) | ~np.isfinite(J).all(axis=(0, 1))
            live[i[bad]] = False
            g = np.flatnonzero(~bad)
            if g.size == 0:
                continue
            ii = i[g]
            Jg = np.transpose(J[:, :, g], (2, 0, 1))
            with np.errstate(all="ignore"):
                U_, s_, Vt_ = np.linalg.svd(Jg)
            keep = s_ > 1e-10 * s_[:, :1]
            s_inv = np.where(keep, 1.0 / np.where(keep, s_, 1.0), 0.0)
            p = np.einsum("nji,nj->ni", Vt_,
                          s_inv * np.einsum("nki,nk->ni", U_, -f[:, ii].T))
            cap = np.minimum(1.0, 0.5 / np.maximum(np.abs(p).max(axis=1), 1e-300))
            p *= cap[:, None]
            accepted = np.zeros(ii.size, bool)
            relax = np.ones(ii.size)
            for _ls in range(10):
                t = np.flatnonzero(~accepted)
                if t.size == 0:
                    break
                cand = [u[c][ii[t]] + relax[t] * p[t, c] for c in range(3)]
                fc, ec, _ = structure(sub(L, ii[t]), sub(R, ii[t]), cand,
                                      Bn[ii[t]], mode)
                nc = np.where(ec, np.inf, np.max(np.abs(fc), axis=0))
                good = nc < nrm[ii[t]]
                for c in range(3):
                    u[c][ii[t[good]]] = cand[c][good]
                f[:, ii[t[good]]] = fc[:, good]
                nrm[ii[t[good]]] = nc[good]
                accepted[t[good]] = True
                relax[t[~good]] *= 0.5
            live[ii[~accepted]] = False
            live &= nrm > ACC
        return u, nrm

    def verify(L, R, Bn, u, mode):
        """The compound answer's own certificate (see the module docstring)."""
        f, err, st = structure(L, R, u, Bn, mode)
        nrm = np.where(err, np.inf, np.max(np.abs(f), axis=0))
        ok = nrm <= VERIFY
        rep = dict(contact=nrm)
        for side, sw, Akey, Vfast in (("L", "LS", "A", "VLF"), ("R", "RS", "D", "VRF")):
            if side not in st:
                continue
            Vsh, Vtl, S, b = st[side]
            A = st[Akey]
            rh = rh_resid(A, S, Vsh, Bn, gamma)
            son = np.abs(xi(S, sw, Bn) - Vsh)
            la = alfven_speed(A, Bn, side)
            inter = (Vsh < la) if side == "L" else (Vsh > la)
            vcd = 0.5 * (st["B"][2] + st["C"][2])
            order = ((st[Vfast] <= Vsh) & (Vsh <= Vtl + 1e-12) & (Vtl <= vcd)
                     if side == "L" else
                     (st[Vfast] >= Vsh) & (Vsh >= Vtl - 1e-12) & (Vtl >= vcd))
            ok &= (rh <= VERIFY) & (son <= SONIC_TOL) & order
            rep.update({side + "_rh": rh, side + "_sonic": son,
                        side + "_intermediate": inter, side + "_order": order,
                        side + "_Vshock": Vsh, side + "_Vtail": Vtl,
                        side + "_b": b})
        return ok, rep, st

    return dict(sonic_shock=sonic_shock, compound=compound, structure=structure,
                solve=solve, verify=verify, xi=xi)


# ── Balsara 1 ────────────────────────────────────────────────────────────

def balsara1(a):
    from rmhd.initial_data import load_initial_data
    from src.physics.eos import hybrid_eos
    from src.physics.tube_seed import run_tubes
    t0 = time.time()
    Bx, gamma, left, right = load_initial_data(8)
    L = [np.array([v], dtype=float) for v in left]
    R = [np.array([v], dtype=float) for v in right]
    Bn = np.array([Bx], dtype=float)
    print("Balsara 1: gamma %.1f  Bn %.2f\n  L %s\n  R %s" % (gamma, Bx, left, right))
    CW = make(gamma)
    S5 = P5.make_solver(gamma)
    gr = [np.array([np.log(0.698933452)]), np.array([-0.428492941]),
          np.array([np.log(0.697622885)])]           # the stored exact answer

    out = {}
    # the regular solution: planar with a pi-rotation on the left
    for fl in ((True, False), (False, False)):
        r = S5["solve"](L, R, Bn, unk3_init=gr, accuracy=ACC, flip=fl,
                        verify_tol=VERIFY)
        key = "regular_flip%d%d" % fl
        out[key] = dict(conv=bool(r["converged"][0]),
                        full=float(r["full_resid"][0]),
                        unk=[float(c[0]) for c in r["unk"]],
                        zones=np.stack([np.stack([c[0] for c in z]) for z in r["zones"]]),
                        speeds=np.array([float(v[0]) for v in r["speeds"]]))
        print("  regular, flip %s: converged %s, seven-wave residual %.1e, "
              "p_LF %.6f  Bt* %+.6f  p_RF %.6f"
              % (fl, out[key]["conv"], out[key]["full"],
                 np.exp(out[key]["unk"][0]), out[key]["unk"][1],
                 np.exp(out[key]["unk"][2])))
    # the compound solution, from the same seed and from a few others
    best = None
    for seed in (gr, [gr[0], -gr[1], gr[2]], [gr[0], gr[1] * 2.0, gr[2]],
                 [gr[0], np.array([-0.8]), gr[2]]):
        u, nrm = CW["solve"](L, R, Bn, seed, (True, False))
        ok, rep, st = CW["verify"](L, R, Bn, u, (True, False))
        print("  compound (left), seed Bt* %+.3f: |f| %.1e  verified %s  "
              "shock rh %.1e sonic %.1e intermediate %s  Vshock %+.5f Vtail %+.5f"
              % (float(seed[1][0]), float(nrm[0]), bool(ok[0]),
                 float(rep.get("L_rh", [np.nan])[0]),
                 float(rep.get("L_sonic", [np.nan])[0]),
                 bool(rep.get("L_intermediate", [False])[0]),
                 float(rep.get("L_Vshock", [np.nan])[0]),
                 float(rep.get("L_Vtail", [np.nan])[0])))
        if ok[0] and best is None:
            best = (u, rep, st)
    if best is not None:
        u, rep, st = best
        out["compound"] = dict(unk=[float(c[0]) for c in u],
                               A=np.array([c[0] for c in st["A"]]),
                               S=np.array([c[0] for c in st["L"][2]]),
                               B=np.array([c[0] for c in st["B"]]),
                               C=np.array([c[0] for c in st["C"]]),
                               D=np.array([c[0] for c in st["D"]]),
                               VLF=float(st["VLF"][0]), VRF=float(st["VRF"][0]),
                               Vshock=float(rep["L_Vshock"][0]),
                               Vtail=float(rep["L_Vtail"][0]))
        print("  compound answer: p_LF %.6f  Bt* %+.6f  p_RF %.6f"
              % (np.exp(u[0][0]), u[1][0], np.exp(u[2][0])))

    # the PDE itself, at three resolutions
    eos = hybrid_eos(K=0.0, gamma=gamma, gamma_th=gamma)
    UL = np.array([left]); UR = np.array([right])
    tubes = {}
    for nc in a.ncells:
        prof, t = run_tubes(UL, UR, Bn, eos, ncells=nc, tend=a.tend,
                            limiter="mc", max_steps=100000)
        xi = (-0.5 + (np.arange(nc) + 0.5) / nc) / t
        tubes[nc] = dict(xi=xi, **{k: prof[k].numpy()[:, 0] for k in
                                   ("rho", "p", "vx", "vy", "By")})
        print("  tube %5d cells done (t = %.3f, %.0fs)" % (nc, t, time.time() - t0))

    # compare plateau values: the state just left of the contact
    def plateau(tb, lo, hi):
        m = (tb["xi"] > lo) & (tb["xi"] < hi)
        return {k: float(np.median(tb[k][m])) for k in ("rho", "p", "vx", "By")} if m.any() else None
    print("\n  left star state (between the left slow structure and the contact):")
    print("  %-26s %9s %9s %9s %9s" % ("", "rho", "p_gas", "vx", "By"))
    rows = []
    for key, lab in (("regular_flip10", "regular (pi-rotation)"),):
        if out.get(key, {}).get("conv"):
            z = out[key]["zones"]                 # A, B, C, D (planar5 zones)
            Bst = z[1]
            rows.append((lab, Bst))
    if "compound" in out:
        rows.append(("compound (this element)", out["compound"]["B"]))
    for lab, Bst in rows:
        # gas pressure from total: p = P_tot - b^2/2
        rho, Pt, vx, vy, vz, By, Bz = Bst
        v2 = vx * vx + vy * vy + vz * vz
        eta = Bx * vx + By * vy + Bz * vz
        b2 = (Bx * Bx + By * By + Bz * Bz) * (1 - v2) + eta * eta
        print("  %-26s %9.5f %9.5f %9.5f %9.5f" % (lab, rho, Pt - 0.5 * b2, vx, By))
    if "compound" in out:
        cp = out["compound"]
        lo = cp["Vtail"] + 0.02
        hi = 0.5 * (cp["B"][2] + cp["C"][2]) - 0.02
        for nc in a.ncells:
            pl = plateau(tubes[nc], lo, hi)
            if pl:
                print("  %-26s %9.5f %9.5f %9.5f %9.5f" % (
                    "tube %d cells" % nc, pl["rho"], pl["p"], pl["vx"], pl["By"]))
    np.savez_compressed(a.out, **{("tube_%d_%s" % (nc, k)): v
                                  for nc, tb in tubes.items() for k, v in tb.items()},
                        **{("%s_%s" % (key, k)): np.asarray(v)
                           for key, d in out.items() for k, v in d.items()})
    print("\n  wrote %s (%.0fs)" % (a.out, time.time() - t0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--balsara1", action="store_true")
    ap.add_argument("--ncells", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--tend", type=float, default=0.4)
    ap.add_argument("--out", default="balsara1_compound.npz")
    a = ap.parse_args()
    if a.balsara1:
        balsara1(a)
    else:
        ap.error("only --balsara1 is implemented so far")


if __name__ == "__main__":
    main()
