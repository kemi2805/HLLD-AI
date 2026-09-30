#!/usr/bin/env python
"""One interface, one planar Newton, every iteration spelled out.

The failure ledger's cards say THAT a start did not converge; this shows how:
per iteration the unknowns, the three contact residuals, the conditioning of
the Jacobian, the step, and what each trial of the line search ran into --
which wave could not be constructed, or that the residual did not fall.

    python scripts/newton_trace.py results/failure_ledger_64 --lanes 719,1394

It repeats `planar5_b.solve`'s iteration with the solver's own `structure`
and `jacobian` (same step cap, same ten halvings), so what it prints is what
production did.  RMHD_FAST_EDGE_SCAN is honoured.
"""
import argparse
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

GAMMA = 5.0 / 3.0
NAMES = ("fast L", "slow L", "fast R", "slow R")


def waves(W, L, R, Bn, unk):
    """The four waves one by one: which fail, and the contact mismatch."""
    z0 = np.zeros(1)
    A, VLF, eA = W["fast_wave"](np.exp(unk[0]), L, "LF", Bn)
    B, VLS, sL, eB = W["slow_wave6"](unk[1], z0, A, "LS", Bn)
    D, VRF, eD = W["fast_wave"](np.exp(unk[2]), R, "RF", Bn)
    C, VRS, sR, eC = W["slow_wave6"](unk[1], z0, D, "RS", Bn)
    err = [bool(eA[0]), bool(eB[0] and not eA[0]), bool(eD[0]),
           bool(eC[0] and not eD[0])]
    f = np.array([B[2][0] - C[2][0], B[3][0] - C[3][0], B[1][0] - C[1][0]])
    kind = ["shock" if np.exp(unk[0][0]) >= L[1][0] else "fan",
            "shock" if abs(unk[1][0]) <= abs(A[5][0]) else "fan",
            "shock" if np.exp(unk[2][0]) >= R[1][0] else "fan",
            "shock" if abs(unk[1][0]) <= abs(D[5][0]) else "fan"]
    return f, err, kind, (A, B, C, D), (VLF[0], VLS[0], VRS[0], VRF[0])


def trace(pop, j, max_iter, out):
    L = [pop["UL"][j:j + 1, c].copy() for c in range(7)]
    R = [pop["UR"][j:j + 1, c].copy() for c in range(7)]
    Bn = pop["UL"][j:j + 1, 7].copy()
    Lp, Rp, alpha, presid = P5.to_planar(L, R, Bn)
    P = P5.make_solver(GAMMA)
    W = FB.make_waves(GAMMA)
    unk = P5.default_seed(Lp, Rp)
    w = out.write
    w("=" * 96 + "\n")
    w("interface %d   B_n = %+.6f   planar frame (rotated by %.4f rad)\n"
      % (j, Bn[0], float(np.asarray(alpha).reshape(-1)[0])))
    w("            rho      P_tot         vx         vt         Bt\n")
    for t, s in (("L", Lp), ("R", Rp)):
        w("    %s %10.6f %10.6f %+10.6f %+10.6f %+10.6f\n"
          % (t, s[0][0], s[1][0], s[2][0], s[3][0], s[5][0]))
    w("  unknowns: p_LF, Bt_CD, p_RF -- the total pressure behind each fast "
      "wave, the tangential field at the contact\n")
    w("  residuals: the jumps of vx, vt, P_tot across the contact\n\n")
    for it in range(max_iter + 1):
        f, err, kind, zones, V = waves(W, Lp, Rp, Bn, unk)
        nrm = np.abs(f).max()
        w("  it %2d  p_LF %.8f  Bt_CD %+.8f  p_RF %.8f\n"
          % (it, np.exp(unk[0][0]), unk[1][0], np.exp(unk[2][0])))
        w("         waves: %s\n" % ",  ".join(
            "%s %s%s" % (n_, k_, " FAILS" if e_ else "")
            for n_, k_, e_ in zip(NAMES, kind, err)))
        if any(err):
            w("         => the state cannot be constructed here: stop\n")
            return "a wave cannot be constructed at the start" if it == 0 \
                else "a wave cannot be constructed"
        w("         speeds  LF %+.6f  LS %+.6f  RS %+.6f  RF %+.6f\n" % V)
        w("         [[vx]] %+.3e  [[vt]] %+.3e  [[P]] %+.3e   |f| = %.3e\n"
          % (f[0], f[1], f[2], nrm))
        if nrm <= 1e-8:
            w("         => converged\n")
            return "converged"
        if it == max_iter:
            break
        f_, J, z_, V_, e_, sk_ = P["jacobian"](Lp, Rp, unk, Bn)
        if e_[0]:
            w("         => the Jacobian's probes cannot be constructed: stop\n")
            return "a Jacobian probe cannot be constructed"
        U_, s_, Vt_ = np.linalg.svd(J[0])
        keep = s_ > 1e-10 * s_[0]
        y = np.where(keep, 1.0 / np.where(keep, s_, 1.0), 0.0) * (U_.T @ -f_[:, 0])
        p = Vt_.T @ y
        cap = min(1.0, 0.5 / max(np.abs(p).max(), 1e-300))
        w("         Jacobian: singular values %.2e %.2e %.2e  (condition %.1e)\n"
          % (s_[0], s_[1], s_[2], s_[0] / max(s_[2], 1e-300)))
        w("         Newton step: d ln p_LF %+.3e  d Bt %+.3e  d ln p_RF %+.3e"
          "%s\n" % (p[0], p[1], p[2],
                    "" if cap == 1.0 else "   (capped to %.3g of it)" % cap))
        p = p * cap
        relax, taken = 1.0, False
        for ls in range(10):
            cand = [unk[c] + relax * p[c] for c in range(3)]
            fc, ec, kc, _, _ = waves(W, Lp, Rp, Bn, cand)
            if any(ec):
                why = "%s cannot be constructed" % ", ".join(
                    n_ + " " + k_ for n_, k_, e2 in zip(NAMES, kc, ec) if e2)
            elif np.abs(fc).max() < nrm:
                w("         line search: fraction %.4g accepted, |f| %.3e -> "
                  "%.3e\n" % (relax, nrm, np.abs(fc).max()))
                unk, taken = cand, True
                break
            else:
                why = "|f| = %.3e, no smaller" % np.abs(fc).max()
            w("         line search: fraction %.4g refused -- %s\n"
              % (relax, why))
            relax *= 0.5
        if not taken:
            w("         => ten halvings, none accepted: stop at |f| = %.3e\n"
              % nrm)
            return "the line search found no smaller residual"
    w("         => out of iterations at |f| = %.3e\n" % nrm)
    return "out of iterations"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ledger")
    ap.add_argument("--lanes", required=True,
                    help="comma-separated interface numbers (the cards')")
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--quiet", action="store_true",
                    help="only the one-line outcome per interface")
    a = ap.parse_args()
    pop = np.load(os.path.join(a.ledger, "population.npz"))
    sink = open(os.devnull, "w") if a.quiet else sys.stdout
    for t in a.lanes.split(","):
        j = int(t)                     # the cards' number: the population's
        how = trace(pop, j, a.max_iter, sink)
        print("interface %s: %s" % (t, how))


if __name__ == "__main__":
    main()
