#!/usr/bin/env python
"""What the wave sub-solvers do on the failure ledger's interfaces.

The ledger (`failure_ledger.py`) says WHICH interfaces nothing solves; this
asks why, one level down: at the planar solver's default start, which of the
four waves it has to construct cannot be constructed, and -- for the fast
shocks -- how many roots the Hugoniot residual really has there.

    python scripts/ledger_subwaves.py results/failure_ledger_64
    python scripts/ledger_subwaves.py results/failure_ledger_64 --pairs
    python scripts/ledger_subwaves.py results/failure_ledger_64 --slow-family

Light: one residual evaluation per lane (`--pairs`: a 30,000-node scan of the
failing fast waves only).  RMHD_FAST_EDGE_SCAN is honoured, so the same call
with the switch on shows what it repairs.
"""
import argparse
import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rmhd.eos import set_eos                                    # noqa: E402
set_eos("ideal")
from rmhd.batched import fullcontact_b as FB                    # noqa: E402
from rmhd.batched import planar5_b as P5                        # noqa: E402
from rmhd.batched import rarefaction_b as RB                    # noqa: E402
from rmhd.batched import shock_b as SB                          # noqa: E402
from rmhd.batched import wave_speeds_b as WB                    # noqa: E402

GAMMA = 5.0 / 3.0


def load(d):
    pop = np.load(os.path.join(d, "population.npz"))
    n = pop["group"].size
    ok = {m: np.zeros(n, bool) for m in "abcd"}
    comp = np.zeros(n, bool)
    for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        z = np.load(f)
        for m in "abcd":
            if m + "_ok" in z.files:
                ok[m][z["idx"]] = z[m + "_ok"]
        if "e_compound" in z.files:
            comp[z["idx"]] = z["e_compound"]
    fail = pop["group"] == 1
    R = np.zeros(n, int)
    R[fail] = 4
    R[fail & ok["d"]] = 3
    R[fail & ok["c"]] = 2
    R[fail & (ok["a"] | ok["b"])] = 1
    return pop, R, comp


def planar(pop, i):
    L = [pop["UL"][i, j].copy() for j in range(7)]
    R = [pop["UR"][i, j].copy() for j in range(7)]
    Bn = pop["UL"][i, 7].copy()
    Lp, Rp = P5.to_planar(L, R, Bn)[:2]
    return Lp, Rp, Bn, P5.default_seed(Lp, Rp)


def q(x, p=(5, 50, 95)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return "-"
    return "[" + " ".join("%.1e" % v for v in np.percentile(x, p)) + "]"


def subwaves(pop, groups):
    W = FB.make_waves(GAMMA)
    print("at the planar solver's default start (the mean total pressure, the "
          "mean tangential field):\n")
    print("  %-22s %5s | %-17s | %-17s | %s" % (
        "group", "n", "fast wave fails", "slow wave fails", "any"))
    out = {}
    for name, m in groups:
        i = np.flatnonzero(m)
        Lp, Rp, Bn, unk = planar(pop, i)
        z0 = np.zeros(i.size)
        A, _, eA = W["fast_wave"](np.exp(unk[0]), Lp, "LF", Bn)
        B, _, _, eB = W["slow_wave6"](unk[1], z0, A, "LS", Bn)
        D, _, eD = W["fast_wave"](np.exp(unk[2]), Rp, "RF", Bn)
        C, _, _, eC = W["slow_wave6"](unk[1], z0, D, "RS", Bn)
        eB, eC = eB & ~eA, eC & ~eD
        bad = eA | eB | eD | eC
        print("  %-22s %5d | L %4d   R %4d   | L %4d   R %4d   | %4d (%.0f%%)"
              % (name, i.size, eA.sum(), eD.sum(), eB.sum(), eC.sum(),
                 bad.sum(), 100.0 * bad.mean()))
        out[name] = (i, eA, eD)
        for side, st, e, pt in (("L", Lp, eA, np.exp(unk[0])),
                                ("R", Rp, eD, np.exp(unk[2]))):
            if e.any():
                sh = pt >= st[1]
                print("  %22s       fast %s: %d of the failures are shocks; "
                      "strength dp/p %s; |B_n|/|B_t| %s (where it works: %s)"
                      % ("", side, (e & sh).sum(), q(np.abs(pt / st[1] - 1)[e]),
                         q((np.abs(Bn) / np.abs(st[5]))[e]),
                         q((np.abs(Bn) / np.abs(st[5]))[~e])))
    return out


def pairs(pop, i, eA, eD, n=30000):
    """Every root of the fast-shock residual on the failing fast waves."""
    sf = SB.make_shockfunc(GAMMA)
    bf = SB.make_postshock(GAMMA)
    Lp, Rp, Bn, unk = planar(pop, i)
    nroot, nevo, sep, pos, ratio = [], [], [], [], []
    for side, e in (("LF", eA), ("RF", eD)):
        right = side == "RF"
        ah = [c[e] for c in (Rp if right else Lp)]
        pb = np.exp(unk[2 if right else 0])[e]
        bn = Bn[e]
        k = 3 if right else 0
        sgn = 1.0 if right else -1.0
        xa = WB.xi_all(*ah, bn, GAMMA)[0][:, k]
        xmin, xmax = SB.search_bounds(side, ah, bn, GAMMA)
        alf, cone = (xmin, xmax) if right else (xmax, xmin)
        gap, far = np.abs(alf - xa), np.abs(cone - xa)
        ratio += list(gap / ((xmax - xmin) / 500.0))
        for l in range(pb.size):
            t = np.concatenate([-np.linspace(1, 0, 50) * gap[l],
                                1e-4 * gap[l] * (far[l] / (1e-4 * gap[l]))
                                ** np.linspace(0, 1, n)])
            V = xa[l] + sgn * t
            one = [np.full(V.size, c[l]) for c in ah]
            f, _ = sf(V, np.full(V.size, pb[l]), one, np.full(V.size, bn[l]))
            fin = np.isfinite(f)
            sc = np.flatnonzero(fin[:-1] & fin[1:]
                                & (np.sign(f[:-1]) * np.sign(f[1:]) < 0))
            a0 = [c[l:l + 1] for c in ah]
            where, evo = [], 0
            for j in sc:
                Vm = SB.bisect(lambda v: sf(v, pb[l:l + 1], a0, bn[l:l + 1]),
                               V[j:j + 1], f[j:j + 1], V[j + 1:j + 2],
                               f[j + 1:j + 2], tol=1e-13, max_iter=60)
                bh, eb = bf(Vm, pb[l:l + 1], a0, bn[l:l + 1])
                xb = WB.xi_all(*bh, bn[l:l + 1], GAMMA)[0][0, k]
                la, ra = SB.alfven_speeds(bh, bn[l:l + 1], GAMMA)
                if right:
                    ok = (Vm[0] >= xa[l] - 1e-6) and (Vm[0] <= xb + 1e-6) \
                        and (Vm[0] >= ra[0] - 1e-6)
                else:
                    ok = (Vm[0] <= xa[l] + 1e-6) and (Vm[0] >= xb - 1e-6) \
                        and (Vm[0] <= la[0] + 1e-6)
                evo += bool(ok and not eb[0])
                where.append(sgn * (Vm[0] - xa[l]) / gap[l])
            nroot.append(len(where))
            nevo.append(evo)
            if len(where) == 2:
                sep.append(abs(where[1] - where[0]))
                pos.append(max(abs(where[0]), abs(where[1])))
    print("\nthe failing fast waves, scanned with %d nodes each (%d waves):"
          % (n, len(nroot)))
    print("  fast-Alfven gap / one step of the standard scan   %s (max %.2f)"
          % (q(ratio), max(ratio)))
    print("  roots per wave                     %s"
          % dict(zip(*np.unique(nroot, return_counts=True))))
    print("  of them evolutionary fast shocks   %s   (two-sided Lax on the "
          "fast family and super-Alfvenic behind)"
          % dict(zip(*np.unique(nevo, return_counts=True))))
    print("  distance between the two roots, in gaps   %s (min %.3f)"
          % (q(sep), min(sep)))
    print("  the outer root from the fast eigenvalue, in gaps   %s (max %.0f)"
          % (q(pos), max(pos)))


def slow_family(pop, groups):
    """Which quantity MOVES along the slow family, at the input states.

    The tangent of the slow wave curve (the fan's right-hand side), written
    as relative changes and scaled so that its largest component is 1.  A
    component near zero is a coordinate that cannot steer the wave.
    """
    xi = FB.make_xi(GAMMA)
    rhs_n = RB.make_rhs(GAMMA, lambda s, sw, b, g: xi(s, sw, b))[1]
    names = ("d ln rho", "d ln p_gas", "d ln P_tot", "d|Bt| / sqrt(2P)",
             "|dv|")

    def comps(st, sw, Bn):
        rho, P, vx, vy, vz, By, Bz = st
        x = np.hypot(By, Bz)
        k = rhs_n(x, st, sw, Bn, GAMMA)
        v2 = vx * vx + vy * vy + vz * vz
        B2 = Bn * Bn + By * By + Bz * Bz
        vB = Bn * vx + By * vy + Bz * vz
        b2 = B2 * (1.0 - v2) + vB * vB
        dv2 = 2.0 * (vx * k[2] + vy * k[3] + vz * k[4])
        dvB = Bn * k[2] + By * k[3] + k[5] * vy + Bz * k[4] + k[6] * vz
        db2 = (2.0 * (By * k[5] + Bz * k[6]) * (1.0 - v2) - B2 * dv2
               + 2.0 * vB * dvB)
        dBt = (By * k[5] + Bz * k[6]) / np.maximum(x, 1e-300)
        c = np.abs(np.stack([
            k[0] / rho, (k[1] - 0.5 * db2) / (P - 0.5 * b2), k[1] / P,
            dBt / np.sqrt(2.0 * P),
            np.sqrt(k[2] ** 2 + k[3] ** 2 + k[4] ** 2)]))
        return c / np.nanmax(c, axis=0), k[1] * x / P

    print("\nthe tangent of the slow wave curve at the input states, largest "
          "component = 1:")
    for name, m in groups:
        i = np.flatnonzero(m)
        Lp, Rp, Bn, _ = planar(pop, i)
        cl, sl = comps(Lp, "LS", Bn)
        cr, sr = comps(Rp, "RS", Bn)
        c = np.concatenate([cl, cr], axis=1)
        sl = np.concatenate([sl, sr])
        print("\n  %s (%d slow waves)   d ln P_tot / d ln|Bt|: median %.2e, "
              "positive on %.0f%%" % (name, c.shape[1], np.nanmedian(sl),
                                      100.0 * np.nanmean(sl > 0)))
        print("      %-18s %10s %10s %26s" % (
            "", "5% quant.", "median", "below 1e-2 of the largest"))
        for row, nm in zip(c, names):
            row = row[np.isfinite(row)]
            print("      %-18s %10.1e %10.1e %25.1f%%" % (
                nm, np.percentile(row, 5), np.median(row),
                100.0 * (row < 1e-2).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ledger")
    ap.add_argument("--pairs", action="store_true")
    ap.add_argument("--slow-family", action="store_true")
    a = ap.parse_args()
    pop, R, comp = load(a.ledger)
    print("RMHD_FAST_EDGE_SCAN = %s\n"
          % os.environ.get("RMHD_FAST_EDGE_SCAN", "0"))
    groups = [("controls (solved)", pop["group"] == 0), ("R1", R == 1),
              ("R2", R == 2), ("R3", R == 3),
              ("R4, tube: compound", (R == 4) & comp),
              ("R4, tube: elementary", (R == 4) & ~comp)]
    out = subwaves(pop, groups)
    if a.slow_family:
        slow_family(pop, groups)
    if a.pairs:
        pairs(pop, *out["R4, tube: elementary"])


if __name__ == "__main__":
    main()
