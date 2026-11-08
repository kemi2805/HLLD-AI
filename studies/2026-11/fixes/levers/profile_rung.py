#!/usr/bin/env python
"""Profile one exact-flux call on a handful of hard faces (the late chain), as
a sweep with few faces past the five-wave solver: where does a planar solve's
~2 s go?   python profile_rung.py FACES.npz RUN SRC N [--prof out.prof]"""
import cProfile
import os
import pstats
import sys
import time

import numpy as np
import torch

HLLD = os.environ["HLLD_DIR"]
sys.path.insert(0, HLLD)
from rmhd.eos import set_eos                                     # noqa: E402
set_eos("ideal")
from src.physics import exact_flux as EF                         # noqa: E402
from src.physics import exact_pool as XP                         # noqa: E402
from src.physics.eos import hybrid_eos                           # noqa: E402

f, run, src, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
z = np.load(f)
UL, UR = z["%s_src%d_UL" % (run, src)], z["%s_src%d_UR" % (run, src)]
gam = float(z["%s_src%d_gamma" % (run, src)])
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


T = {}
_ml, _pp = XP.map_lanes, EF._run_pipeline


def map_lanes(name, *a, **kw):
    t0 = time.perf_counter()
    try:
        return _ml(name, *a, **kw)
    finally:
        T[name] = T.get(name, 0.0) + time.perf_counter() - t0


def run_pipeline(*a, **kw):
    t0 = time.perf_counter()
    try:
        return _pp(*a, **kw)
    finally:
        T["_run_pipeline"] = T.get("_run_pipeline", 0.0) + time.perf_counter() - t0


XP.map_lanes, EF._run_pipeline = map_lanes, run_pipeline
call = lambda j: EF.exact_flux_batched(lab(UL[j]), lab(UR[j]), eos, idir=0, tau_weak=1e-2,
                                       tau_bt=1e-9, n_retries=2, max_iter=40)
call(np.arange(n, 2 * n))                                       # warm-up: kernels, networks
T.clear()
pr = cProfile.Profile()
t0 = time.perf_counter()
pr.enable()
call(np.arange(n))
pr.disable()
wall = time.perf_counter() - t0
from src.physics.hlld import LAST_DIAG as D                     # noqa: E402
print("faces %d, wall %.2f s; stages: %s" % (n, wall, {k: round(v, 2) for k, v in T.items()}))
from rmhd.batched import planar_njit as _PNJ                    # noqa: E402
if _PNJ.CHECK:
    print("numpy vs njit (calls, lanes, lanes that differ):", _PNJ.CHECK)
print("late rung offered %s answered %s; four-unknown attempted %s exact %s" % (
    D.get("n_coplanar7_late_attempted"), D.get("n_coplanar7_late_exact"),
    D.get("n_planar4_attempted"), D.get("n_planar4_exact")))
if len(sys.argv) > 6:
    pr.dump_stats(sys.argv[6])
st = pstats.Stats(pr)
st.sort_stats("tottime").print_stats(25)
