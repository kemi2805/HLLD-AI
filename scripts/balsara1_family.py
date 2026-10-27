#!/usr/bin/env python3
"""Balsara 1: the one-parameter family of exact solutions through our answer.

The left slow structure is an intermediate shock from A (the state behind the left
fast fan) to a CHOSEN reversed tangential field b, then a slow wave from that state
X to the contact's field B_t*: a slow shock if |b| > |B_t*|, a detached slow
rarefaction if |b| < |B_t*|.  b = B_t* is our answer (and Giacomazzo & Rezzolla's);
b = -0.310 is the compound wave (the fan attaches: the shock is sonic behind); at
b* = -0.6414 the shock has no density or pressure jump and moves at the Alfven
speed on both sides -- a 180-degree rotational discontinuity -- and the member is
the REGULAR solution (rotation + Lax slow shock).  Beyond b* the shock is no
longer intermediate and the entropy falls.

The rest of the fan is the planar system: three unknowns [ln p_LF, B_t*, ln p_RF]
against the three contact jumps (damped Newton, forward-difference Jacobian).
Every member is certified: the contact jumps, both shocks' jump conditions
(``jump.rh_resid``), the Lax conditions of the slow shock, the wave order, the
entropy across both shocks; ``check`` recomputes the jump conditions with HLLD's
own flux formulas (``src/physics/hlld.py``, no code shared with the exact solver).

    python scripts/balsara1_family.py build  --answers balsara1_compound.npz --out family.npz
    python scripts/balsara1_family.py check  --answers balsara1_compound.npz --b -0.37 -0.47 -0.55
    python scripts/balsara1_family.py regular --answers balsara1_compound.npz

``regular`` needs RMHD_ALFVEN_EXACT=1 (rmhd_final branch alfven-closed-form): the
planar pi-branch then builds the regular solution directly, and it must equal the
family's end.  Must run WITHOUT kernels.env: the intermediate shock needs the numpy
slow-shock path with the window opened (compound_wave.wide_window).

Recreated 2026-10-26 from the 10-17 session scripts (is_family.py, check_family.py,
family_end2.py), which were lost with the /tmp scratchpad.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import balsara1_study as BS                                       # noqa: E402
import compound_wave as CW                                        # noqa: E402
from rmhd.batched import fullcontact_b as FB                      # noqa: E402
from rmhd.batched import slow_shock_b as SSB                      # noqa: E402
from rmhd.batched.jump import rh_resid                            # noqa: E402

np.seterr(all="ignore")
G, Bx = BS.G, float(BS.Bx)
W = FB.make_waves(G)
fast_wave, slow_wave6, xi, alfven_speed = W["fast_wave"], W["slow_wave6"], W["xi"], W["alfven_speed"]
shock = SSB.make_solver(G)
B_STAR_GUESS = (-0.640, -0.645)        # the bracket of the far end (measured 10-17)


def col(st, n):
    return [np.full(n, float(c)) for c in st]


def gas(st):
    rho, Pt, vx, vy, vz, By, Bz = st
    v2 = vx * vx + vy * vy + vz * vz
    eta = Bx * vx + By * vy + Bz * vz
    b2 = (Bx * Bx + By * By + Bz * Bz) * (1 - v2) + eta * eta
    return Pt - 0.5 * b2


def structure(unk, b):
    """Planar residual with the left element 'intermediate shock to b, then slow wave'."""
    n = b.size
    Bn = np.full(n, Bx)
    p_LF, Bt, p_RF = np.exp(unk[0]), unk[1], np.exp(unk[2])
    A, VLF, eA = fast_wave(p_LF, col(BS.LEFT, n), "LF", Bn)
    D, VRF, eD = fast_wave(p_RF, col(BS.RIGHT, n), "RF", Bn)
    with CW.wide_window():
        V1, X, e1 = shock(b, np.zeros(n), A, "LS", Bn, accuracy=1e-12)
    Bst, V2, _, e2 = slow_wave6(Bt, np.zeros(n), X, "LS", Bn)
    C, V3, _, e3 = slow_wave6(Bt, np.zeros(n), D, "RS", Bn)
    f = np.stack([Bst[2] - C[2], Bst[3] - C[3], Bst[1] - C[1]])
    err = (np.asarray(eA, bool) | np.asarray(eD, bool) | np.asarray(e1, bool)
           | np.asarray(e2, bool) | np.asarray(e3, bool) | ~np.isfinite(f).all(axis=0))
    return (np.where(err[None], 0.0, f), err,
            dict(A=A, X=X, B=Bst, C=C, D=D, VLF=VLF, V1=V1, V2=V2, V3=V3, VRF=VRF))


def solve(b, u0, max_iter=40, acc=1e-11):
    n = b.size
    u = [np.array(c, float) for c in u0]
    f, err, _ = structure(u, b)
    nrm = np.where(err, np.inf, np.abs(f).max(axis=0))
    for _ in range(max_iter):
        live = np.isfinite(nrm) & (nrm > acc)
        if not live.any():
            break
        h = [1e-7 * np.maximum(1.0, np.abs(c)) for c in u]
        J = np.zeros((n, 3, 3))
        bad = np.zeros(n, bool)
        for j in range(3):
            up = [c + (h[j] if k == j else 0.0) for k, c in enumerate(u)]
            fj, ej, _ = structure(up, b)
            J[:, :, j] = ((fj - f) / h[j]).T
            bad |= ej
        p = np.zeros((n, 3))
        for i in np.flatnonzero(live & ~bad):
            p[i] = np.linalg.lstsq(J[i], -f[:, i], rcond=1e-12)[0]
        p *= np.minimum(1.0, 0.2 / np.maximum(np.abs(p).max(axis=1), 1e-300))[:, None]
        lam = np.ones(n)
        done = ~live | bad
        for _ in range(12):
            if done.all():
                break
            cand = [u[k] + np.where(done, 0.0, lam * p[:, k]) for k in range(3)]
            fc, ec, _ = structure(cand, b)
            nc = np.where(ec, np.inf, np.abs(fc).max(axis=0))
            good = ~done & (nc < nrm)
            for k in range(3):
                u[k] = np.where(good, cand[k], u[k])
            f[:, good] = fc[:, good]
            nrm = np.where(good, nc, nrm)
            done |= good
            lam = np.where(done, lam, 0.5 * lam)
    return u, nrm


def walk(ans, targets):
    """Continuation from our answer to each b in ``targets`` (one side at a time)."""
    Z = ans["Z"]
    Bt_ours = float(Z[1][5])
    u_ours = [np.log(float(Z[0][1])), Bt_ours, np.log(float(Z[3][1]))]
    sol = {}
    for b in sorted(targets, key=lambda x: abs(x - Bt_ours)):
        near = [bb for bb in sol if (bb - Bt_ours) * (b - Bt_ours) >= 0]
        start = sol[min(near, key=lambda bb: abs(bb - b))][0] if near else u_ours
        u, nrm = solve(np.array([b]), [np.array([s]) for s in start])
        sol[b] = ([float(c[0]) for c in u], float(nrm[0]))
    return sol


def certify(bb, U, nrm, Bt_ours):
    n = bb.size
    Bn = np.full(n, Bx)
    f, err, st = structure(U, bb)
    A, X, Bs, C = st["A"], st["X"], st["B"], st["C"]
    V1, V2 = st["V1"], st["V2"]
    rh1 = rh_resid(A, X, V1, Bn, G)
    shk = np.abs(bb) > abs(Bt_ours) * (1 + 1e-9)
    rh2 = np.where(shk, rh_resid(X, Bs, V2, Bn, G), np.nan)
    lsX, lsB = xi(X, "LS", Bn), xi(Bs, "LS", Bn)
    laA, laX = alfven_speed(A, Bn, "L"), alfven_speed(X, Bn, "L")
    vcd = 0.5 * (Bs[2] + C[2])
    KA, KX, KB = gas(A) / A[0] ** G, gas(X) / X[0] ** G, gas(Bs) / Bs[0] ** G
    rows = []
    for i in range(n):
        kind = "shock" if shk[i] else ("ours" if abs(bb[i] - Bt_ours) < 1e-9 else "fan")
        inter = bool(V1[i] < laA[i] and V1[i] > laX[i])
        if kind == "shock":
            lax = bool(lsX[i] > V2[i] > lsB[i] and V2[i] > V1[i])
            ent = bool(KX[i] > KA[i] and KB[i] > KX[i])
        elif kind == "fan":
            lax, ent = bool(lsX[i] >= V1[i] - 1e-9), bool(KX[i] > KA[i])
        else:
            lax, ent = True, bool(KX[i] > KA[i])
        order = bool(st["VLF"][i] <= V1[i] and V2[i] <= vcd[i])
        ok = (nrm[i] <= 1e-8 and rh1[i] <= 1e-8 and (kind != "shock" or rh2[i] <= 1e-8)
              and inter and lax and ent and order)
        rows.append(dict(b=bb[i], kind=kind, contact=nrm[i], rh_is=rh1[i], rh_2=rh2[i],
                         V_is=V1[i], V_2=V2[i], ls_X=lsX[i], ls_B=lsB[i], a_A=laA[i], a_X=laX[i],
                         rho_X=X[0][i], By_X=X[5][i], rho_B=Bs[0][i], p_B=gas(Bs)[i],
                         vx_B=Bs[2][i], K_A=KA[i], K_X=KX[i], K_B=KB[i], inter=inter,
                         lax=lax, entropy=ent, order=order, ok=ok))
    return rows


def build(a):
    t0 = time.time()
    ans = BS.answers(a.answers)
    Z = ans["Z"]
    Bt_ours = float(Z[1][5])
    bs = [-0.31, -0.33, -0.35, -0.37, -0.39, -0.41, -0.42, Bt_ours, -0.435, -0.44, -0.45,
          -0.46, -0.47, -0.48, -0.485, -0.49, -0.50, -0.52, -0.55, -0.60, -0.61, -0.62,
          -0.63, -0.635, -0.64, -0.65, -0.70]
    sol = walk(ans, bs)
    print("walked %d members in %.0f s" % (len(sol), time.time() - t0), flush=True)
    # the far end: bisect V_IS - a(X) = 0 inside B_STAR_GUESS
    lo, hi = B_STAR_GUESS
    u_lo = sol[lo][0]
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        u, nrm = solve(np.array([mid]), [np.array([s]) for s in u_lo])
        _, _, st = structure(u, np.array([mid]))
        g = float(st["V1"][0] - alfven_speed(st["X"], np.array([Bx]), "L")[0])
        if g > 0:
            lo, u_lo = mid, [float(c[0]) for c in u]
        else:
            hi = mid
    sol[lo] = (u_lo, float(solve(np.array([lo]), [np.array([s]) for s in u_lo])[1][0]))
    bb = np.array(sorted(sol))
    U = [np.array([sol[b][0][k] for b in bb]) for k in range(3)]
    nrm = np.array([sol[b][1] for b in bb])
    rows = certify(bb, U, nrm, Bt_ours)
    p1 = BS.prim(Z[1])
    rel = lambda q, k: 1e3 * (q - p1[k]) / abs(p1[k])
    print("\n   b      kind   contact rh_IS  rh_2   V_IS    V_2     a(A)    a(X)    rho_X   "
          "plateau rho p vx (1e-3 rel. to ours)  ok")
    for r in rows:
        print(" %+.5f %-5s %.0e %.0e %6s %.5f %.5f %.5f %.5f %.4f  %+7.3f %+6.3f %+6.3f  %s%s"
              % (r["b"], r["kind"], r["contact"], r["rh_is"],
                 ("%.0e" % r["rh_2"]) if r["kind"] == "shock" else "-", r["V_is"], r["V_2"],
                 r["a_A"], r["a_X"], r["rho_X"], rel(r["rho_B"], "rho"), rel(r["p_B"], "p"),
                 rel(r["vx_B"], "vx"), "OK" if r["ok"] else "--",
                 "" if r["ok"] else "  (intermediate %s, Lax %s, entropy %s, order %s)"
                 % (r["inter"], r["lax"], r["entropy"], r["order"])))
    end = [r for r in rows if abs(r["b"] - lo) < 1e-12][0]
    print("\nfar end b* = %.6f: shock speed %.6f, Alfven ahead %.6f / behind %.6f, "
          "rho jump %.1e, entropy %.5f -> %.5f; then slow shock %.5f; plateau rho %+.2f e-3"
          % (lo, end["V_is"], end["a_A"], end["a_X"], 0.0, end["K_A"], end["K_X"],
             end["V_2"], rel(end["rho_B"], "rho")))
    print("far-end unknowns: p_LF %.6f  Bt* %+.6f  p_RF %.6f"
          % (np.exp(u_lo[0]), u_lo[1], np.exp(u_lo[2])))
    data = dict(b=bb, unk=np.stack(U), b_star=lo, unk_star=np.array(u_lo))
    for k in rows[0]:
        data.setdefault(k, np.array([r[k] for r in rows]))   # per-member columns
    np.savez_compressed(a.out, **data)
    print("wrote %s (%.0f s)" % (a.out, time.time() - t0))


def check(a):
    import torch
    from src.physics.eos import hybrid_eos
    from src.physics.hlld import compute_srmhd_fluxes
    eos = hybrid_eos(K=0.0, gamma=G, gamma_th=G)
    keys = ("D", "Sx", "Sy", "Sz", "tau", "By", "Bz")

    def tor(q):
        d = {k: torch.tensor([v], dtype=torch.float64) for k, v in q.items()}
        d["Bx"] = torch.tensor([Bx], dtype=torch.float64)
        d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
        return d

    def jump(Pl, Pr, s):
        uL, uR, fL, fR, _, _ = compute_srmhd_fluxes(tor(Pl), tor(Pr), eos, idir=0)
        sc = max(max(abs(float(fL[k])), abs(float(fR[k]))) for k in keys)
        return max(abs(float(fR[k] - fL[k] - s * (uR[k] - uL[k]))) for k in keys) / sc

    ans = BS.answers(a.answers)
    Bt_ours = float(ans["Z"][1][5])
    sol = walk(ans, list(a.b))
    for b in a.b:
        u, nrm = sol[b]
        f, err, st = structure([np.array([c]) for c in u], np.array([b]))
        P = {k: BS.prim([float(c[0]) for c in st[k]]) for k in ("A", "X", "B", "C", "D")}
        V1, V2, V3 = (float(st[k][0]) for k in ("V1", "V2", "V3"))
        vcd = 0.5 * (P["B"]["vx"] + P["C"]["vx"])
        K = {k: P[k]["p"] / P[k]["rho"] ** G for k in P}
        print("b = %+.3f  contact |f| %.1e" % (b, nrm))
        print("   intermediate shock A->X at %.5f: jump conditions %.1e   entropy %.5f -> %.5f"
              % (V1, jump(P["A"], P["X"], V1), K["A"], K["X"]))
        if abs(b) > abs(Bt_ours):
            print("   slow shock X->B at %.5f:         jump conditions %.1e   entropy %.5f -> %.5f"
                  % (V2, jump(P["X"], P["B"], V2), K["X"], K["B"]))
        else:
            print("   slow rarefaction X->B (isentropic: %.5f, %.5f)" % (K["X"], K["B"]))
        print("   contact B->C at %.5f:            jump conditions %.1e" % (vcd, jump(P["B"], P["C"], vcd)))
        print("   right slow shock C->D at %.5f:   jump conditions %.1e   entropy %.5f -> %.5f"
              % (V3, jump(P["D"], P["C"], V3), K["D"], K["C"]))


def regular(a):
    from rmhd.batched import planar5_b as P5
    if not getattr(FB, "ALFVEN_EXACT", False):
        raise SystemExit("needs RMHD_ALFVEN_EXACT=1 (rmhd_final branch alfven-closed-form)")
    ans = BS.answers(a.answers)
    Z = ans["Z"]
    gr = [np.array([np.log(float(Z[0][1]))]), np.array([float(Z[1][5])]),
          np.array([np.log(float(Z[3][1]))])]
    S5 = P5.make_solver(G)
    L = [np.array([float(c)]) for c in BS.LEFT]
    R = [np.array([float(c)]) for c in BS.RIGHT]
    r = S5["solve"](L, R, np.array([Bx]), unk3_init=gr, accuracy=1e-10, flip=(True, False),
                    verify_tol=1e-8)
    print("planar pi-branch (left): converged %s, seven-wave residual %.1e, p_LF %.6f Bt* %+.6f p_RF %.6f"
          % (bool(r["converged"][0]), float(r["full_resid"][0]), float(np.exp(r["unk"][0][0])),
             float(r["unk"][1][0]), float(np.exp(r["unk"][2][0]))))
    if a.family:
        z = np.load(a.family)
        us = z["unk_star"]
        print("family's far end (b* = %.6f):                               p_LF %.6f Bt* %+.6f p_RF %.6f"
              % (float(z["b_star"]), np.exp(us[0]), us[1], np.exp(us[2])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("build", "check", "regular"))
    ap.add_argument("--answers", required=True, help="compound_wave.py --balsara1 output")
    ap.add_argument("--out", default="balsara1_family.npz")
    ap.add_argument("--b", type=float, nargs="+", default=[-0.37, -0.47, -0.55])
    ap.add_argument("--family", help="build's output, for `regular` to compare with")
    a = ap.parse_args()
    {"build": build, "check": check, "regular": regular}[a.mode](a)


if __name__ == "__main__":
    main()
