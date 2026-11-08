#!/usr/bin/env python
"""The 1D tubes of chosen ledger lanes (2026-11-04, lost-face examples):
run_tubes (HLLD, MC) at one resolution, one lane per process, profiles saved.

    python tubes_examples.py LEDGER_DIR LANE NCELLS OUT.npz
"""
import os
import sys
import time

import numpy as np

HLLD = os.environ.get("HLLD_DIR", "/mnt/rafast/miler/codes/rotor2d/HLLD-harvest")
sys.path.insert(0, HLLD)
from src.physics.eos import hybrid_eos                       # noqa: E402
from src.physics.tube_seed import run_tubes                   # noqa: E402

led, lane, nc, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
p = np.load(os.path.join(led, "population.npz"), allow_pickle=True)
gamma = float(p["gamma"])
eos = hybrid_eos(K=0.0, gamma=gamma, gamma_th=gamma)
UL, UR = p["UL"][[lane]], p["UR"][[lane]]
t0 = time.time()
pr, tt = run_tubes(UL[:, :7], UR[:, :7], UL[:, 7], eos, ncells=nc, tend=0.18, max_steps=40 * nc)
np.savez_compressed(out, lane=lane, ncells=nc, t=tt, UL=UL[0], UR=UR[0], gamma=gamma,
                    **{k: v.numpy()[:, 0] for k, v in pr.items()})
print("lane %d, %d cells: t %.4f in %.0f s" % (lane, nc, tt, time.time() - t0), flush=True)
