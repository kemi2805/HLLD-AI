"""Is the exact flux mirror-consistent?

Orszag-Tang is symmetric under the pi rotation (x, y) -> (1-x, 1-y), which
negates the in-plane vectors.  A face with states (L, R) has a mirror face
that sees (R*, L*), * negating vx, vy, Bx, By; its flux must be
F' = -R(F): D, tau, Sz, Bz components negated, Sx, Sy, Bx, By kept.  Any
difference is the solver's own asymmetry.

Face states: neighbouring cell averages of a snapshot along x (first-order
faces -- this tests the solver, not the reconstruction).

    python scripts/mirror_probe.py SNAP GAMMA TAU_WEAK

with production's environment (kernels.env, the ML checkpoints, the pool) --
calea job 40966 ran it on results/ot_64_exact_open/snap_00{1,2}.npz, 4/3,
1e-2 (docs/what_we_solve.md, section 7, Orszag-Tang).  The pi rotation is
Orszag-Tang's symmetry; the four-quadrant problem's is the diagonal x <-> y,
which this does not test.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.physics.eos import hybrid_eos                     # noqa: E402
from src.physics import exact_flux as EF                   # noqa: E402
from src.physics.hlld import LAST_DIAG, hlld_flux          # noqa: E402


def main():
    snap, gam, tau = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    eos = hybrid_eos(K=0.0, gamma=gam, gamma_th=gam)
    z = np.load(snap)
    print("snapshot %s t=%.4f, tau_weak %g" % (snap, float(z["t"]), tau))
    t = lambda a: torch.tensor(np.ascontiguousarray(a).reshape(-1), dtype=torch.float64)

    def state(a, sign):
        d = {"rho": t(a["rho"]), "p": t(a["p"]),
             "vx": sign * t(a["vx"]), "vy": sign * t(a["vy"]),
             "vz": t(np.zeros_like(a["rho"])),
             "Bx": sign * t(a["Bx"]), "By": sign * t(a["By"]),
             "Bz": t(np.zeros_like(a["rho"]))}
        d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
        return d

    f = {k: z[k] for k in ("rho", "p", "vx", "vy", "Bx", "By")}
    Lc = dict(f)
    Rc = {k: np.roll(v, -1, axis=0) for k, v in f.items()}   # the x-neighbour
    # the normal field is single-valued at a face (constrained transport): both
    # sides get the mean of the two cells
    Lc["Bx"] = Rc["Bx"] = 0.5 * (Lc["Bx"] + Rc["Bx"])
    # whichever array axis is x, the pairs are valid problems along the
    # normal: the result is a property of the solver
    sL, sR = state(Lc, 1.0), state(Rc, 1.0)
    mL, mR = state(Rc, -1.0), state(Lc, -1.0)                 # (R*, L*)

    kw = dict(idir=0, tau_weak=tau, tau_bt=1e-9, n_retries=2, max_iter=40)
    F, _, p1 = EF.exact_flux_batched(sL, sR, eos, **kw)
    ex1 = LAST_DIAG["exact_mask"].numpy().copy()
    at1 = np.zeros(ex1.size, bool)
    at1[np.asarray(LAST_DIAG["sel"])] = True
    G, _, p2 = EF.exact_flux_batched(mL, mR, eos, **kw)
    ex2 = LAST_DIAG["exact_mask"].numpy().copy()
    at2 = np.zeros(ex2.size, bool)
    at2[np.asarray(LAST_DIAG["sel"])] = True

    print("flux keys:", list(F))
    # F' = -R(F), R negating the in-plane vector components: the Sx, Sy, Bx, By
    # fluxes keep their sign, the D, tau, Sz, Bz fluxes flip
    keep = {"Sx", "Sy", "Bx", "By"}
    sgn = {k: (1.0 if k in keep else -1.0) for k in F}
    num = sum((G[k] - sgn[k] * F[k]).abs() for k in F).numpy().reshape(-1)
    den = sum(F[k].abs() for k in F).numpy().reshape(-1) + 1e-300
    rel = num / den
    both, one = ex1 & ex2, ex1 ^ ex2
    print("faces %d: exact both %d, exactly one of the pair %d"
          % (ex1.size, both.sum(), one.sum()))
    print("  attempted: %d and %d (the pair disagrees on %d)" % (at1.sum(), at2.sum(), (at1 ^ at2).sum()))
    groups = (("exact both", both), ("exactly one", one),
              ("HLLD both", at1 & at2 & ~ex1 & ~ex2), ("not attempted", ~at1 & ~at2))
    for name, m in groups:
        if m.any():
            r = rel[m]
            print("  %-12s mirror defect p50 %.1e  p90 %.1e  max %.1e  bitwise %.1f%%"
                  % (name, np.median(r), np.percentile(r, 90), r.max(), 100 * np.mean(r == 0)))

    # control: HLLD on the same pairs (the HLLD run kept the symmetry to 1e-12, so
    # its defect must be rounding -- this validates the sign map)
    Fh = hlld_flux(sL, sR, eos, idir=0)[0]
    Gh = hlld_flux(mL, mR, eos, idir=0)[0]
    rh = (sum((Gh[k] - sgn[k] * Fh[k]).abs() for k in Fh).numpy().reshape(-1)
          / (sum(Fh[k].abs() for k in Fh).numpy().reshape(-1) + 1e-300))
    print("  HLLD control  mirror defect p50 %.1e  p90 %.1e  max %.1e  bitwise %.1f%%"
          % (np.median(rh), np.percentile(rh, 90), rh.max(), 100 * np.mean(rh == 0)))


if __name__ == "__main__":   # the pool spawns workers that re-import this
    main()
