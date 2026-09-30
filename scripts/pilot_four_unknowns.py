#!/usr/bin/env python
"""PILOT: a planar solve in four unknowns, one strength per wave.

The production planar solver has three unknowns -- the total pressure behind
each fast wave and ONE tangential field at the contact, which both slow waves
are told to reach.  On interfaces where the slow waves carry pressure and
hardly any field (docs/what_we_solve.md, 5a) that target is the coordinate
that does not move.  Here each slow wave has its own strength, steered by
whichever of (B_t, P_tot) moves along its family at the state it starts
from, and the contact closes four conditions: [[vx]], [[vt]], [[Bt]], [[P]].
It starts from slow waves of zero strength.

    python scripts/pilot_four_unknowns.py results/failure_ledger_64 1394,719
    python scripts/pilot_four_unknowns.py results/failure_ledger_64 <lanes> --quiet

A pilot, not a solver: one lane at a time, no batching, a forward-difference
Jacobian, and "converged" means the four contact conditions close to 1e-9 on
a state built from the wave sub-solvers, every one of which satisfies its
own jump conditions (a Bt-steered wave with slack counts as not
constructed).  It has NOT been through production's acceptance; the full
residual it prints rebuilds the waves through the Bt-method and therefore
fails right answers on exactly the interfaces this pilot is for.
"""
import sys, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmhd.eos import set_eos; set_eos("ideal")
from rmhd.batched import planar5_b as P5, shock_b as SB, slow_shock_b as SSB
from rmhd.batched import wave_speeds_b as WB, fullcontact_b as FB, rarefaction_b as RB
from rmhd.batched.jump import rh_resid
G = 5 / 3
xi = FB.make_xi(G); xi_fn = lambda s, sw, b, g: xi(s, sw, b)
fan_p, fan_n = RB.make_integrators(G, xi_fn)
rhs_p, rhs_n = RB.make_rhs(G, xi_fn)
W = FB.make_waves(G)
pshock = SB.make_shockfunc(G); pbehind = SB.make_postshock(G)


def slow_by_P(Pt, X, sw, Bn):
    """Slow wave to a total pressure: shock (evolutionary root) or fan."""
    if abs(Pt[0] - X[1][0]) <= 1e-14 * X[1][0]:
        v = xi(X, sw, Bn)[0]
        return [c.copy() for c in X], False, (v, v, "none")
    if Pt[0] < X[1][0]:
        y, V, e = fan_p(Pt, X, sw, Bn)
        return y, bool(e[0]), (V[0], xi(y, sw, Bn)[0], "fan")
    ea = WB.xi_all(*X, Bn, G)[0][0]; la, ra = SSB.alfven_bounds(X, Bn, G)
    lo, hi = (la[0], ea[1]) if sw == "LS" else (ea[2], ra[0])
    V = np.linspace(lo, hi, 4001)
    n = V.size
    f, _ = pshock(V, np.full(n, Pt[0]), [np.full(n, c[0]) for c in X], np.full(n, Bn[0]))
    fin = np.isfinite(f)
    sc = np.flatnonzero(fin[:-1] & fin[1:] & (np.sign(f[:-1]) * np.sign(f[1:]) < 0))
    # the weakest first: nearest the slow eigenvalue
    sc = sc[::-1] if sw == "LS" else sc
    for k in sc:
        Vm = SB.bisect(lambda v: pshock(v, Pt, X, Bn), V[k:k+1], f[k:k+1], V[k+1:k+2], f[k+1:k+2], tol=1e-14, max_iter=60)
        bh, e = pbehind(Vm, Pt, X, Bn)
        if not e[0]:
            return bh, False, (Vm[0], Vm[0], "shock")
    return [c.copy() for c in X], True, None


def slow_by_Bt(Bt, X, sw, Bn):
    B, V, sl, e = W["slow_wave6"](Bt, np.zeros(1), X, sw, Bn)
    shock = abs(Bt[0]) <= abs(X[5][0])
    tail = V[0] if shock else xi(B, sw, Bn)[0]
    # The Bt-method IMPOSES the target and carries what the wave could not do
    # in its slack (a fan cannot turn the field; eq. 4.17 for a shock).  The
    # production planar solver drops the slack and lets the full residual
    # catch it; here a wave with slack is a wave that was not constructed.
    e = bool(e[0]) or not abs(sl[0]) <= 1e-9 * max(1.0, abs(Bt[0]))
    return B, e, (V[0], tail, "shock" if shock else "fan")


def kind_of(X, sw, Bn):
    """'B' where the tangential field moves along the slow family, else 'P'."""
    x = np.hypot(X[5], X[6]); k = rhs_n(x, X, sw, Bn, G)
    dBt = abs(1.0 / np.sqrt(2 * X[1][0])); dP = abs(k[1][0] / X[1][0])
    return "B" if dBt >= dP else "P"


def build(L, R, Bn, u, kinds):
    A, VA, eA = W["fast_wave"](np.exp(u[0:1]), L, "LF", Bn)
    D, VD, eD = W["fast_wave"](np.exp(u[3:4]), R, "RF", Bn)
    if eA[0] or eD[0]:
        return None, "fast"
    out, spd = [], []
    for X, sw, s, kd in ((A, "LS", u[1:2], kinds[0]), (D, "RS", u[2:3], kinds[1])):
        Y, e, v = (slow_by_Bt(s, X, sw, Bn) if kd == "B" else slow_by_P(np.exp(s), X, sw, Bn))
        if e:
            return None, "slow " + sw
        out.append(Y)
        spd.append(v)
    B, C = out
    build.speeds = (VA[0], xi(A, "LF", Bn)[0], spd[0], 0.5 * (B[2][0] + C[2][0]),
                    spd[1], xi(D, "RF", Bn)[0], VD[0])
    sc = np.sqrt(2 * 0.5 * (L[1][0] + R[1][0]))
    f = np.array([B[2][0] - C[2][0], B[3][0] - C[3][0], (B[5][0] - C[5][0]) / sc,
                  (B[1][0] - C[1][0]) / (0.5 * (L[1][0] + R[1][0]))])
    return f, (A, B, C, D)


def solve(L, R, Bn, verbose=True, max_iter=30):
    p0 = np.log(0.5 * (L[1][0] + R[1][0]))
    A, _, eA = W["fast_wave"](np.exp(np.array([p0])), L, "LF", Bn)
    D, _, eD = W["fast_wave"](np.exp(np.array([p0])), R, "RF", Bn)
    if eA[0] or eD[0]:
        return "a fast wave fails at the start", None
    kinds = (kind_of(A, "LS", Bn), kind_of(D, "RS", Bn))
    # start with slow waves of zero strength
    u = np.array([p0, A[5][0] if kinds[0] == "B" else np.log(A[1][0]),
                  D[5][0] if kinds[1] == "B" else np.log(D[1][0]), p0])
    if verbose:
        print("   slow waves steered by: left %s, right %s" % kinds)
    f, z = build(L, R, Bn, u, kinds)
    if f is None:
        return "cannot construct at the start (%s)" % z, None
    for it in range(max_iter):
        nrm = np.abs(f).max()
        if verbose:
            print("   it %2d  |f| %.3e   u = %s" % (it, nrm, np.array2string(u, precision=8)))
        if nrm <= 1e-9:
            build(L, R, Bn, u, kinds)              # the speeds of THIS state
            return "converged", (u, z, kinds, build.speeds)
        J = np.zeros((4, 4)); okJ = True
        for c in range(4):
            h = 1e-6 * max(1.0, abs(u[c])) if True else 1e-6
            up = u.copy(); up[c] += h
            fp, _ = build(L, R, Bn, up, kinds)
            if fp is None:
                up = u.copy(); up[c] -= h; h = -h
                fp, _ = build(L, R, Bn, up, kinds)
            if fp is None:
                okJ = False; break
            J[:, c] = (fp - f) / h
        if not okJ:
            return "a Jacobian probe cannot be constructed", None
        try:
            p = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            return "singular Jacobian", None
        if verbose:
            print("          condition %.1e  step %s" % (np.linalg.cond(J), np.array2string(p, precision=3)))
        lam, done = 1.0, False
        for _ in range(12):
            fc, zc = build(L, R, Bn, u + lam * p, kinds)
            if fc is not None and np.abs(fc).max() < nrm:
                u, f, z, done = u + lam * p, fc, zc, True
                break
            lam *= 0.5
        if not done:
            return "the line search found no smaller residual (|f| %.2e)" % nrm, None
    return "out of iterations", None


if __name__ == "__main__":
    pop = np.load(sys.argv[1] + "/population.npz")
    P = P5.make_solver(G)
    quiet = "--quiet" in sys.argv
    for t in sys.argv[2].split(","):
        j = int(t)
        L = [pop["UL"][j:j+1, c].copy() for c in range(7)]
        R = [pop["UR"][j:j+1, c].copy() for c in range(7)]
        Bn = pop["UL"][j:j+1, 7].copy()
        Lp, Rp, a, pr = P5.to_planar(L, R, Bn)
        if not quiet:
            print("interface", j)
        how, sol = solve(Lp, Rp, Bn, verbose=not quiet)
        extra = ""
        if sol is not None:
            u, (A, B, C, D), kinds, sp = sol
            # the fan, left to right: every wave's leading and trailing edge
            LFh, LFt, (LSh, LSt, kL), cd, (RSh, RSt, kR), RFt, RFh = sp
            edges = [min(LFh, LFt), max(LFh, LFt), min(LSh, LSt),
                     max(LSh, LSt), cd, min(RSh, RSt), max(RSh, RSt),
                     min(RFh, RFt), max(RFh, RFt)]
            ordered = bool(np.all(np.diff(edges) >= -1e-9))
            # production's own check, through the Bt-method waves
            full = P["verify_full"](Lp, Rp, [u[0:1], B[5].copy(), u[3:4]], Bn)[0]
            extra = ("  p* %.8f  Bt* %.6f  slow waves %s/%s (steered by %s%s)  "
                     "fan ordered %s  full residual through the Bt-method %.1e"
                     % (B[1][0], B[5][0], kL, kR, kinds[0], kinds[1],
                        "yes" if ordered else "NO", full))
        print("interface %d: %s%s" % (j, how, extra), flush=True)
