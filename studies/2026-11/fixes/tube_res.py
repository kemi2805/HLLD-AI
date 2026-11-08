"""One lane's tube at every resolution on a common xi window: By, rho, p, vx."""
import sys
import numpy as np
sys.path.insert(0, "/Users/miler/Codes/rmhd_final")
from rmhd.batched import wave_speeds_b as WB
T = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/tubes/tube_%d_%d.npz"
lane = int(sys.argv[1]); lo, hi = float(sys.argv[2]), float(sys.argv[3])
var = sys.argv[4] if len(sys.argv) > 4 else "By"
z0 = np.load(T % (lane, 4096)); g = float(z0["gamma"])
for U, nm in ((z0["UL"], "UL"), (z0["UR"], "UR")):
    e, la, ra, ok = WB.xi_all(*[np.array([U[j]]) for j in range(7)], np.array([U[7]]), g)
    print(nm, "LF %.5f LA %.5f LS %.5f | vx %.5f | RS %.5f RA %.5f RF %.5f" % (e[0,0], la[0], e[0,1], U[2], e[0,2], ra[0], e[0,3]))
cols = {}
for nc in (1024, 2048, 4096):
    z = np.load(T % (lane, nc)); t = float(z["t"])
    xi = ((np.arange(nc) + 0.5) / nc - 0.5) / t
    cols[nc] = (xi, z[var])
xs = np.linspace(lo, hi, 41)
print("%9s %10s %10s %10s" % ("xi", "1024", "2048", "4096"))
for x in xs:
    print("%9.5f" % x, " ".join("%10.6f" % np.interp(x, *cols[nc]) for nc in (1024, 2048, 4096)))
