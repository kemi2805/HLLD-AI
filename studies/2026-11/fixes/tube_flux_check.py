#!/usr/bin/env python
"""The recovered faces against their own 1D tubes: the flux at x/t = 0 of the
tube (1024 / 2048 / 4096 cells) vs the replay's OFF flux (HLLD) and ON flux
(the new exact answer).  Max relative difference over the flux components.

    HLLD_DIR=... python tube_flux_check.py OFF.npz ON.npz LANE [LANE ...]
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.environ.get("HLLD_DIR", "/Users/miler/Codes/HLLD"))
from src.physics.eos import hybrid_eos
from src.physics.hlld import compute_srmhd_fluxes
T = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/tubes/tube_%d_%d.npz"
off, on = np.load(sys.argv[1]), np.load(sys.argv[2])
KEYS = ("D", "Sx", "Sy", "Sz", "tau", "By", "Bz")
for lane in [int(a) for a in sys.argv[3:]]:
    row = np.flatnonzero(off["lane"] == lane)[0]
    Foff = np.array([off["F_" + k][row] for k in KEYS])
    Fon = np.array([on["F_" + k][row] for k in KEYS])
    msg = "lane %4d exact off/on %d/%d  |F_on - F_off| %.1e  vs tube:" % (
        lane, off["exact"][row], on["exact"][row], np.max(np.abs(Fon - Foff)) / np.max(np.abs(Foff)))
    for nc in (1024, 2048, 4096):
        z = np.load(T % (lane, nc)); g = float(z["gamma"]); t = float(z["t"])
        eos = hybrid_eos(K=0.0, gamma=g, gamma_th=g)
        xi = ((np.arange(nc) + 0.5) / nc - 0.5) / t
        # x/t = 0 sits on the face between the two central cells: average their fluxes
        Fs = []
        for i0 in (nc // 2 - 1, nc // 2):
            st = {k: torch.as_tensor(z[k][i0:i0 + 1]) for k in ("rho", "p", "vx", "vy", "vz", "Bx", "By", "Bz")}
            st["eps"] = eos.eps__press_rho(st["p"], st["rho"])
            _, _, F0, _, _, _ = compute_srmhd_fluxes(st, st, eos, idir=0)
            Fs.append(np.array([float(F0[k][0]) for k in KEYS]))
        Ft = 0.5 * (Fs[0] + Fs[1])
        sc = np.max(np.abs(Ft))
        msg += "  %d: HLLD %.1e exact %.1e" % (nc, np.max(np.abs(Foff - Ft)) / sc, np.max(np.abs(Fon - Ft)) / sc)
    print(msg)
