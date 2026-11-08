"""Exact solutions of newly solved faces as profiles in xi = x/t (lab frame),
built from the four-unknown solver's answer: plateaus, shocks as jumps, fans
sampled along their own integral curves (the state at xi is the fan state
whose characteristic speed is xi).  Used by figs_solutions.py."""
import os
import sys
import numpy as np
sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/rmhd_fix"))
np.seterr(all="ignore")
from rmhd.batched import planar4_b as P4, planar5_b as P5, rarefaction_b as RB, wave_speeds_b as WB

POP = os.environ.get("SP_POP", "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz")
_pop = np.load(POP, allow_pickle=True)
GAMMA = float(_pop["gamma"])
S4 = P4.make_solver(GAMMA)
W = S4["waves"]
XI = W["xi"]
FAN_P, FAN_N = RB.make_integrators(GAMMA, lambda s, sw, B, g: XI(s, sw, B))


def problem(lane):
    UL, UR = _pop["UL"][lane], _pop["UR"][lane]
    L = [np.array([UL[j]]) for j in range(7)]
    R = [np.array([UR[j]]) for j in range(7)]
    return L, R, np.array([UL[7]])


def solve(lane, kinds, pl, pr, delta=1e-4, unk=None):
    """planar4 from fast pressures (pl, pr) with weak slow waves of the given
    steering (0 field, 1 pressure, 2 signed field), central differences --
    production's star retry -- or from ``unk`` (planar frame)."""
    L, R, Bn = problem(lane)
    L0, R0, alpha, _ = S4["to_planar"](L, R, Bn)
    if unk is None:
        A, _, _ = W["fast_wave"](np.array([pl]), L0, "LF", Bn)
        D, _, _ = W["fast_wave"](np.array([pr]), R0, "RF", Bn)
        st = lambda k, X: np.log(X[1] * (1 - delta)) if k == 1 else X[5] * (1 + delta)
        unk = [np.log(np.array([pl])), st(kinds[0], A), st(kinds[1], D), np.log(np.array([pr]))]
    kd = np.array(kinds, np.int8).reshape(2, 1)
    r = S4["solve"](L, R, Bn, kinds=kd, retry=False, unk_init=unk, central=True)
    assert r["converged"][0], "lane %d did not converge (reason %d)" % (lane, r["reason"][0])
    f, z, V, err, sh = S4["structure"](L0, R0, r["unk"], Bn, kd)
    return dict(lane=lane, L0=L0, R0=R0, alpha=alpha, Bn=Bn, zones=z, V=[v[0] for v in V],
                shocks=sh[:, 0], kinds=kinds, unk=[c[0] for c in r["unk"]], r=r)


def _rep(X, k):
    return [np.repeat(c, k) for c in X]


def fan_states(ahead, behind, sw, kind, Bn, k=80):
    """States along a fan from ``ahead`` to ``behind`` (planar frame)."""
    if sw in ("LF", "RF") or kind == 1:
        tg = np.linspace(ahead[1][0], behind[1][0], k)
        y, _, e = FAN_P(tg, _rep(ahead, k), sw, np.repeat(Bn, k))
    elif kind == 2:
        tg = np.linspace(ahead[5][0], behind[5][0], k)
        y, _, e = S4["fan_signed"](tg, _rep(ahead, k), sw, np.repeat(Bn, k))
    else:
        tg = np.linspace(abs(ahead[5][0]), abs(behind[5][0]), k)
        y, _, e = FAN_N(tg, _rep(ahead, k), sw, np.repeat(Bn, k))
    return y, XI(y, sw, np.repeat(Bn, k))


def lab(state, alpha):
    return P5.from_planar(state, alpha)


def profile(sol, xi_lo=-1.0, xi_hi=1.0):
    """(xi, states[7]) of the whole fan, lab frame, in xi order; plus the
    list of waves (name, kind of wave, xi range)."""
    L0, R0, Bn, al = sol["L0"], sol["R0"], sol["Bn"], sol["alpha"]
    A, B, C, D = sol["zones"]
    V, sh = sol["V"], sol["shocks"]
    kL, kR = sol["kinds"]
    pieces, waves = [], []

    def plateau(X, a, b):
        pieces.append((np.array([a, b]), [np.repeat(c, 2) for c in X]))

    def wave(name, ahead, behind, v, is_shock, kind, left_going):
        if is_shock:
            waves.append((name, "shock", v, v))
            return v, v
        y, xi = fan_states(ahead, behind, name, kind, Bn)
        o = np.argsort(xi)
        pieces.append((xi[o], [c[o] for c in y]))
        waves.append((name, "fan", xi.min(), xi.max()))
        return xi.min(), xi.max()

    a0, a1 = wave("LF", L0, A, V[0], sh[0], None, True)
    plateau(L0, xi_lo, a0)
    b0, b1 = wave("LS", A, B, V[1], sh[1], kL, True)
    plateau(A, a1, b0)
    plateau(B, b1, V[2])
    waves.append(("CD", "contact", V[2], V[2]))
    c0, c1 = wave("RS", D, C, V[3], sh[2], kR, False)
    plateau(C, V[2], c0)
    d0, d1 = wave("RF", R0, D, V[4], sh[3], None, False)
    plateau(D, c1, d0)
    plateau(R0, d1, xi_hi)
    pieces.sort(key=lambda p: p[0][0])
    xi = np.concatenate([p[0] for p in pieces])
    st = [np.concatenate([p[1][j] for p in pieces]) for j in range(7)]
    return xi, lab(st, np.repeat(al, xi.size)), waves


def tube(lane, nc=4096):
    for d in ("/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/tubes",
              "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/tubes_rem"):
        f = "%s/tube_%d_%d.npz" % (d, lane, nc)
        if os.path.exists(f):
            z = np.load(f)
            t = float(z["t"])
            return ((np.arange(nc) + 0.5) / nc - 0.5) / t, z
    return None, None
