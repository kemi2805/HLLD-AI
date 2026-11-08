#!/usr/bin/env python
"""Smoke test of RMHD_HARVEST_RESCUES: one exact-flux call over a small
population with a harvester attached; prints the solved rows by source and
the new columns.

    python harvest_smoke.py POP.npz OUTDIR"""
import os
import sys

import numpy as np
import torch

HLLD = os.environ["HLLD_DIR"]
sys.path.insert(0, HLLD)
from rmhd.eos import set_eos                                     # noqa: E402
set_eos("ideal")
from src.physics import exact_flux as EF                         # noqa: E402
from src.physics.eos import hybrid_eos                           # noqa: E402
from src.physics.harvest import Harvester                        # noqa: E402
from src.physics.hlld import LAST_DIAG                           # noqa: E402

p = np.load(sys.argv[1], allow_pickle=True)
UL, UR = p["UL"], p["UR"]
gam = float(p["gamma"])
eos = hybrid_eos(K=0.0, gamma=gam, gamma_th=gam)


def lab(U):
    rho, Pt, vx, vy, vz, By, Bz, Bx = (U[:, j] for j in range(8))
    W2 = 1.0 / (1.0 - (vx * vx + vy * vy + vz * vz))
    eta = Bx * vx + By * vy + Bz * vz
    b2 = (Bx * Bx + By * By + Bz * Bz) / W2 + eta * eta
    t = lambda v: torch.tensor(np.ascontiguousarray(v), dtype=torch.float64)
    d = {"rho": t(rho), "vx": t(vx), "vy": t(vy), "vz": t(vz),
         "p": t(Pt - 0.5 * b2), "Bx": t(Bx), "By": t(By), "Bz": t(Bz)}
    d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
    return d


H = Harvester(sys.argv[2], gam, only_retried=False, max_solved=10**7, max_unsolved=10**7,
              flush_every=10**7)
EF.exact_flux_batched(lab(UL), lab(UR), eos, idir=0, tau_weak=1e-2, tau_bt=1e-9,
                      n_retries=2, max_iter=40, harvester=H)
d = dict(LAST_DIAG)
print("late rung: offered %s exact %s; four-unknown exact %s" % (
    d.get("n_coplanar7_late_attempted"), d.get("n_coplanar7_late_exact"),
    int(np.asarray(d.get("planar4_mask", [])).sum())))
print(H.close()["faces"])
import glob
for f in sorted(glob.glob(sys.argv[2] + "/solved_*.npz")):
    z = np.load(f)
    print(f.split("/")[-1], {k: z[k].shape for k in z.files})
    src = z["source"]
    for s in np.unique(src):
        m = src == s
        extra = ""
        if "steer" in z.files:
            extra = "  steer %s  rot %s  stage %s" % (
                np.unique(z["steer"][m], axis=0).tolist(), np.unique(z["rot"][m], axis=0).tolist(),
                np.unique(z["stage"][m]).tolist())
        print("  source %d: %d rows%s" % (s, m.sum(), extra))
