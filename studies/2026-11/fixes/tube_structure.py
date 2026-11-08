"""Print the structure of a lost face's 1D tube near x/t = 0: the profile in xi
with the local slow / Alfven / fast speeds, so shocks (jumps), fans (lambda =
xi) and sonic attachments can be read off.

    python tube_structure.py LANE [NCELLS] [XI_LO XI_HI]
"""
import sys
import numpy as np
sys.path.insert(0, "/Users/miler/Codes/rmhd_final")
from rmhd.batched import wave_speeds_b as WB

import os
T = os.environ.get("TUBES", "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/tubes") + "/tube_%d_%d.npz"
lane = int(sys.argv[1]); nc = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
lo, hi = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (-0.06, 0.06)
z = np.load(T % (lane, nc))
t, g = float(z["t"]), float(z["gamma"])
x = (np.arange(nc) + 0.5) / nc - 0.5
xi = x / t
Bn = z["Bx"]
rho, p, vx, vy, vz, By, Bz = (z[k] for k in ("rho", "p", "vx", "vy", "vz", "By", "Bz"))
# P_tot for the solver's state layout (rho, P_tot, vx, vy, vz, By, Bz)
v2 = vx**2 + vy**2 + vz**2
W2 = 1 / (1 - v2)
vB = vx * Bn + vy * By + vz * Bz
b2 = (Bn**2 + By**2 + Bz**2) / W2 + vB**2
Pt = p + 0.5 * b2
eig, la, ra, ok = WB.xi_all(rho, Pt, vx, vy, vz, By, Bz, Bn, g)
print("lane", lane, "ncells", nc, "t", t, "UL", np.round(z["UL"], 5), "UR", np.round(z["UR"], 5))
print("eig columns: ", eig.shape)
sel = np.flatnonzero((xi >= lo) & (xi <= hi))
step = max(1, len(sel) // 120)
print("%9s %9s %9s %9s %9s %9s | %9s %9s %9s %9s %9s %9s" % ("xi", "rho", "p", "vx", "vy", "By",
      "LF", "LA", "LS", "RS", "RA", "RF"))
for i in sel[::step]:
    e = eig[i]
    print("%9.5f %9.6f %9.6f %9.6f %9.6f %9.6f | %9.5f %9.5f %9.5f %9.5f %9.5f %9.5f" % (
        xi[i], rho[i], p[i], vx[i], vy[i], By[i], e[0], la[i], e[1], e[2], ra[i], e[3]))
