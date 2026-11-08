"""Fluxes saved by probe_s_answer.py (exact, HLLD) against the lane's tubes."""
import sys
import numpy as np
import torch
sys.path.insert(0, "/Users/miler/Codes/HLLD")
from src.physics.eos import hybrid_eos
from src.physics.hlld import compute_srmhd_fluxes
KEYS = ("D", "Sx", "Sy", "Sz", "tau", "By", "Bz")
D = sys.argv[1]
for lane in [int(a) for a in sys.argv[2:]]:
    Fex, Fh = np.load("sflux_%d.npy" % lane)
    msg = "lane %d:" % lane
    for nc in (2048, 4096):
        try:
            t = np.load("%s/tube_%d_%d.npz" % (D, lane, nc))
        except FileNotFoundError:
            continue
        g = float(t["gamma"]); eos = hybrid_eos(K=0.0, gamma=g, gamma_th=g)
        Fs = []
        for i0 in (nc // 2 - 1, nc // 2):
            q = {k: torch.as_tensor(t[k][i0:i0 + 1]) for k in ("rho", "p", "vx", "vy", "vz", "Bx", "By", "Bz")}
            q["eps"] = eos.eps__press_rho(q["p"], q["rho"])
            _, _, F0, _, _, _ = compute_srmhd_fluxes(q, q, eos, idir=0)
            Fs.append(np.array([float(F0[k][0]) for k in KEYS]))
        Ft = 0.5 * (Fs[0] + Fs[1]); sc = np.abs(Ft).max()
        msg += "  %d: HLLD %.1e exact %.1e" % (nc, np.abs(Fh - Ft).max() / sc, np.abs(Fex - Ft).max() / sc)
    print(msg)
