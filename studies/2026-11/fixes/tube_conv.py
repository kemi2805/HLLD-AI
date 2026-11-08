#!/usr/bin/env python
"""Do the faces' own 1D tubes keep converging to the exact answer as the grid is
refined (2048 .. 16384 cells), where that answer carries a field-reversing slow
jump whose entropy falls (or rises)?  Flux at x/t = 0 of the tube (mean of the
two central cells' physical fluxes) against the replay's HLLD and exact fluxes;
max relative difference over the components.

    HLLD_DIR=... python tube_conv.py HLLD.npz EXACT.npz GATED.npz LANE [LANE ...]
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.environ.get("HLLD_DIR", "/Users/miler/Codes/HLLD"))
from src.physics.eos import hybrid_eos
from src.physics.hlld import compute_srmhd_fluxes
F = "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes"
T = {2048: F + "/tubes_rem/tube_%d_2048.npz", 4096: F + "/tubes_rem/tube_%d_4096.npz",
     8192: F + "/tubes_hi/tube_%d_8192.npz", 16384: F + "/tubes_hi/tube_%d_16384.npz"}
KEYS = ("D", "Sx", "Sy", "Sz", "tau", "By", "Bz")


def tube_flux(z, nc):
    g = float(z["gamma"])
    eos = hybrid_eos(K=0.0, gamma=g, gamma_th=g)
    Fs = []
    for i0 in (nc // 2 - 1, nc // 2):
        st = {k: torch.as_tensor(z[k][i0:i0 + 1]) for k in ("rho", "p", "vx", "vy", "vz", "Bx", "By", "Bz")}
        st["eps"] = eos.eps__press_rho(st["p"], st["rho"])
        _, _, F0, _, _, _ = compute_srmhd_fluxes(st, st, eos, idir=0)
        Fs.append(np.array([float(F0[k][0]) for k in KEYS]))
    return 0.5 * (Fs[0] + Fs[1])


def main():
  off, on, gated = (np.load(a) for a in sys.argv[1:4])
  print("%5s %6s | %s" % ("lane", "gate", "  ".join("%5d: HLLD   exact" % n for n in T)))
  for lane in [int(a) for a in sys.argv[4:]]:
    row = np.flatnonzero(off["lane"] == lane)[0]
    Fh = np.array([off["F_" + k][row] for k in KEYS])
    Fe = np.array([on["F_" + k][row] for k in KEYS])
    verdict = "keeps" if gated["exact"][row] else "REFUSES"
    msg = "%5d %7s |" % (lane, verdict)
    for nc, pat in T.items():
        if not os.path.exists(pat % lane):
            msg += "  %5d:    --      --" % nc
            continue
        Ft = tube_flux(np.load(pat % lane), nc)
        sc = np.max(np.abs(Ft))
        msg += "  %5d: %.1e %.1e" % (nc, np.max(np.abs(Fh - Ft)) / sc, np.max(np.abs(Fe - Ft)) / sc)
    print(msg)


if __name__ == "__main__":
    main()
