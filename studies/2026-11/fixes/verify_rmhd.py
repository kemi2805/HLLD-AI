#!/usr/bin/env python
"""Is a Riemann answer an actual solution of ideal RMHD?  (user, 2026-11-05)

Input: answers in the harvest schema -- U_L, U_R (N, 8: rho, P_tot, vx, vy,
vz, By, Bz, Bn), zones (N, 8, 7) = [left, R2..R7, right], speeds (N, 7) in
spatial order (LF, LA, LS, CD, RS, RA, RF) -- whatever solver produced them.

Every wave between two consecutive zones is judged with rmhd_ideal (the
conservation laws written from the equations, independent of the solver):

  discontinuity  all seven jump conditions at its speed: rh_residual <= TOL_RH
  rarefaction    re-integrated from its own state ahead (the rarefaction ODE,
                 in total pressure and in signed B_t, whichever lands closer)
                 to the state behind: mismatch <= TOL_FAN; the characteristic
                 speed rises across it in the direction of travel; and the
                 sampled fan satisfies the self-similar weak-solution identity
                 F(U) - xi U + int U dxi = const (rmhd_ideal fluxes)
  contact        jump conditions at its speed (= vx)
  rotation       a discontinuity with rho and p continuous (an Alfven wave)
  order          every wave's xi-range at or after the previous wave's

and, for every discontinuity but the contact, the entropy p / rho^gamma must
not fall (second law) -- reported separately from the weak-solution test:
a jump can satisfy the conservation laws and still be inadmissible.

    python verify_rmhd.py IN.npz OUT.npz         (IN: U_L, U_R, zones, speeds, gamma)
"""
import sys
import numpy as np
import rmhd_ideal as RI
import sol_profiles as SP

TOL_RH, TOL_FAN, TOL_G, TOL_S, TOL_ORDER = 1e-6, 1e-6, 1e-6, 1e-9, 1e-9
# TOL_RH: the solvers converge the contact and the waves' coupling to a scaled
# residual of 1e-8 (which reads ~1e-7 in rmhd_ideal's scaling); a genuine
# discontinuity sits at round-off (median 2e-13), a fan at >= 1e-3.
NAMES = ("LF", "LA", "LS", "CD", "RS", "RA", "RF")
FAN_SW = {0: "LF", 2: "LS", 4: "RS", 6: "RF"}
K = 33                                   # samples per fan for the identity (odd: Simpson)
FAILWHY = []                             # (lane, wave, which fan test) -- filled by verify()


def _st(Z, j, i=None):
    """zone j of every lane (or of lanes i) as a 7-list of arrays"""
    return [Z[:, j, c] if i is None else Z[i, j, c] for c in range(7)]


def _rel(a, b, Bn):
    """max over components of |a - b| / that component's physical scale:
    rho and P_tot by themselves, velocities by c, field components by |B|
    (an out-of-plane component that should be 0 and is stored as 1e-11 is
    round-off, not a mismatch)."""
    a, b = np.stack(a), np.stack(b)
    Bm = np.sqrt(Bn ** 2 + np.maximum(a[5] ** 2 + a[6] ** 2, b[5] ** 2 + b[6] ** 2))
    sc = np.stack([np.abs(b[0]), np.abs(b[1]), np.ones_like(Bm), np.ones_like(Bm), np.ones_like(Bm), Bm, Bm])
    return np.max(np.abs(a - b) / sc, axis=0)


def _identity(Ai, Bi, sw, Bn, use_s, K, g):
    """max over components of the self-similar weak-solution identity across
    a fan, F(U) - xi U + int U dxi at the tail minus at the head, scaled; the
    fan sampled at K points (odd) along the re-integration."""
    n = Bn.size
    Ak = [np.repeat(c, K) for c in Ai]
    Bk = np.repeat(Bn, K)
    us = np.repeat(use_s, K)
    y = [np.zeros(n * K) for _ in range(7)]
    for flag, fan, comp in ((False, SP.FAN_P, 1), (True, SP.S4["fan_signed"], 5)):
        m = us == flag
        if m.any():
            tg = np.linspace(Ai[comp], Bi[comp], K).T.ravel()[m]
            yy, _, _ = fan(tg, [c[m] for c in Ak], sw, Bk[m])
            for c in range(7):
                y[c][m] = yy[c]
    xi = SP.XI(y, sw, Bk).reshape(-1, K)
    U, F = RI.cons_flux(y, Bk, g)
    U, F = U.reshape(7, -1, K), F.reshape(7, -1, K)
    x0, x1, x2 = xi[:, 0:-2:2], xi[:, 1:-1:2], xi[:, 2::2]
    h0, h1 = x1 - x0, x2 - x1
    with np.errstate(divide="ignore", invalid="ignore"):
        w0 = np.where(h0 * h1 != 0, (h0 + h1) / 6 * (2 - h1 / h0), 0.0)
        w1 = np.where(h0 * h1 != 0, (h0 + h1) / 6 * (h0 + h1) ** 2 / (h0 * h1), 0.0)
        w2 = np.where(h0 * h1 != 0, (h0 + h1) / 6 * (2 - h0 / h1), 0.0)
    panels = w0[None] * U[:, :, 0:-2:2] + w1[None] * U[:, :, 1:-1:2] + w2[None] * U[:, :, 2::2]
    intU = np.concatenate([np.zeros((7, n, 1)), np.cumsum(panels, axis=2)], axis=2)
    ev = slice(0, None, 2)
    Gk = F[:, :, ev] - xi[None, :, ev] * U[:, :, ev] + intU
    sc = np.abs(F).max(axis=2) + np.abs(xi).max(axis=1)[None] * np.abs(U).max(axis=2)
    sc = sc + 1e-3 * sc.max(axis=0)[None]
    return np.max(np.abs(Gk - Gk[:, :, :1]).max(axis=2) / sc, axis=0)


def verify(UL, UR, Z, S, g):
    N = Z.shape[0]
    Bn = UL[:, 7]
    kind = np.full((N, 7), "", dtype="U8")      # zero / disc / fan / FAIL
    rh = np.full((N, 7), np.nan)
    fanm = np.full((N, 7), np.nan)
    G = np.full((N, 7), np.nan)
    dS = np.full((N, 7), np.nan)
    lo = np.full((N, 7), np.nan)
    hi = np.full((N, 7), np.nan)
    inter = np.zeros((N, 7), bool)
    for w in range(7):
        left_going = w < 3
        a, b = (w, w + 1) if left_going else (w + 1, w)          # ahead, behind
        A, B = _st(Z, a), _st(Z, b)
        zero = _rel(A, B, Bn) < 1e-13
        kind[zero, w] = "zero"
        V = S[:, w] if w != 3 else 0.5 * (Z[:, 3, 2] + Z[:, 4, 2])
        rh[:, w] = RI.rh_residual(A, B, V, Bn, g)
        dS[:, w] = RI.entropy(B, Bn, g) / RI.entropy(A, Bn, g) - 1.0
        disc = ~zero & (rh[:, w] <= TOL_RH)
        kind[disc, w] = "disc"
        lo[disc, w] = hi[disc, w] = V[disc]
        if w in FAN_SW:
            sw = FAN_SW[w]
            i = np.flatnonzero(~zero & ~disc)
            if i.size:
                Ai, Bi = _st(Z, a, i), _st(Z, b, i)
                yp, _, ep = SP.FAN_P(Bi[1], Ai, sw, Bn[i])
                ys, _, es = SP.S4["fan_signed"](Bi[5], Ai, sw, Bn[i])
                mp = np.where(ep, np.inf, _rel(yp, Bi, Bn[i]))
                ms = np.where(es, np.inf, _rel(ys, Bi, Bn[i]))
                use_s = ms < mp
                fanm[i, w] = np.minimum(mp, ms)
                la, lb = SP.XI(Ai, sw, Bn[i]), SP.XI(Bi, sw, Bn[i])
                expands = (lb >= la) if left_going else (lb <= la)
                # the identity along the fan in the better parametrisation, 33
                # samples, refined (257) where Simpson's error at 33 is above
                # tolerance: it falls at fourth order on a genuine fan
                G[i, w] = _identity(Ai, Bi, sw, Bn[i], use_s, K, g)
                ref = np.flatnonzero(G[i, w] > TOL_G)
                if ref.size:
                    G[i[ref], w] = _identity([c[ref] for c in Ai], [c[ref] for c in Bi], sw,
                                             Bn[i[ref]], use_s[ref], 257, g)
                okf = (fanm[i, w] <= TOL_FAN) & expands & (G[i, w] <= TOL_G)
                why = np.where(fanm[i, w] > TOL_FAN, "end", np.where(~expands, "order", np.where(G[i, w] > TOL_G, "identity", "")))
                for l_, y_ in zip(i[~okf], why[~okf]):
                    FAILWHY.append((int(l_), w, str(y_)))
                kind[i[okf], w] = "fan"
                kind[i[~okf], w] = "FAIL"
                lo[i, w], hi[i, w] = np.minimum(la, lb), np.maximum(la, lb)
                dS[i, w] = RI.entropy(Bi, Bn[i], g) / RI.entropy(Ai, Bn[i], g) - 1.0
        else:
            bad = ~zero & ~disc
            kind[bad, w] = "FAIL"
        if w in (1, 5):                          # an Alfven wave keeps rho and p
            rho_a, p_a = RI.prim(A, Bn, g)[:2]
            rho_b, p_b = RI.prim(B, Bn, g)[:2]
            notrot = disc & ((np.abs(rho_b / rho_a - 1) > 1e-8) | (np.abs(p_b / p_a - 1) > 1e-8))
            kind[notrot, w] = "FAIL"
            # a rotation production itself treats as none (exact_flux._self_crossing:
            # density and velocity change <= 1e-8) may sit anywhere; it is a
            # rotation of the field by ~1e-10 rad, not a wave to order
            dv = np.sqrt(sum((A[c] - B[c]) ** 2 for c in (2, 3, 4)))
            tiny = disc & ~notrot & (np.maximum(np.abs(rho_b / rho_a - 1), dv) <= 1e-8)
            kind[tiny, w] = "zero"
        if w in (0, 2, 4, 6):                    # a shock crossing the Alfven speed
            la_a, ra_a = SP.WB.xi_all(*A, Bn, g)[1:3]
            la_b, ra_b = SP.WB.xi_all(*B, Bn, g)[1:3]
            al_a, al_b = (la_a, la_b) if left_going else (ra_a, ra_b)
            inter[:, w] = disc & ((V - al_a) * (V - al_b) < 0)
    # order: each non-zero wave at or after the previous non-zero one
    order_ok = np.ones(N, bool)
    last = np.full(N, -np.inf)
    for w in range(7):
        has = kind[:, w] != "zero"
        order_ok &= ~has | (lo[:, w] >= last - TOL_ORDER)
        last = np.where(has, hi[:, w], last)
    weak = order_ok & ~(kind == "FAIL").any(axis=1)
    ent = np.ones(N, bool)
    for w in (0, 1, 2, 4, 5, 6):
        ent &= ~((kind[:, w] == "disc") & (dS[:, w] < -TOL_S))
    return dict(kind=kind, rh=rh, fan=fanm, G=G, dS=dS, inter=inter, order_ok=order_ok,
                weak=weak, admissible=weak & ent)


def report(r, label):
    k, n = r["kind"], r["kind"].shape[0]
    print("== %s: %d answers" % (label, n))
    print("  actual weak solutions of ideal RMHD (every wave passes, waves in order): %d (%.2f%%)" % (
        r["weak"].sum(), 100 * r["weak"].mean()))
    print("  ... and entropy-admissible (no discontinuity lowers the entropy):      %d (%.2f%%)" % (
        r["admissible"].sum(), 100 * r["admissible"].mean()))
    for w, nm in enumerate(NAMES):
        c = {t: int((k[:, w] == t).sum()) for t in ("zero", "disc", "fan", "FAIL")}
        d = k[:, w] == "disc"
        extra = ""
        if d.any() and w != 3:
            extra = "  disc: rh p50 %.0e max %.0e, entropy falls on %d, intermediate %d" % (
                np.median(r["rh"][d, w]), r["rh"][d, w].max(), int((r["dS"][d, w] < -TOL_S).sum()),
                int(r["inter"][d, w].sum()))
        f = k[:, w] == "fan"
        if f.any():
            extra += "  fan: end mismatch p50 %.0e max %.0e, identity max %.0e" % (
                np.median(r["fan"][f, w]), r["fan"][f, w].max(), r["G"][f, w].max())
        print("   %s %s%s" % (nm, c, extra))
    print("  out of order: %d" % (~r["order_ok"]).sum())


if __name__ == "__main__":
    d = np.load(sys.argv[1], allow_pickle=True)
    r = verify(d["U_L"], d["U_R"], d["zones"], d["speeds"], float(d["gamma"]))
    np.savez_compressed(sys.argv[2], failwhy=np.array(FAILWHY, dtype=object), **r)
    report(r, sys.argv[1])
    if FAILWHY:
        from collections import Counter
        print("  fan failures by test:", dict(Counter((NAMES[V_[1]], V_[2]) for V_ in FAILWHY)))
