"""rmhd_ideal.cons_flux against HLLD's compute_srmhd_fluxes on random states
(an independent implementation must agree with the production one)."""
import sys
import numpy as np
import torch
sys.path.insert(0, "/Users/miler/Codes/HLLD")
from src.physics.eos import hybrid_eos
from src.physics.hlld import compute_srmhd_fluxes
import rmhd_ideal as RI
rng = np.random.default_rng(3)
N, g = 2000, 5.0 / 3.0
rho = 10 ** rng.uniform(-1, 1, N); p = 10 ** rng.uniform(-2, 1.5, N)
v = rng.uniform(-0.6, 0.6, (3, N)); Bx = rng.uniform(-2, 2, N); By, Bz = rng.uniform(-2, 2, (2, N))
v2 = (v ** 2).sum(0); vB = Bx * v[0] + By * v[1] + Bz * v[2]
b2 = (Bx ** 2 + By ** 2 + Bz ** 2) * (1 - v2) + vB ** 2
st = [rho, p + 0.5 * b2, v[0], v[1], v[2], By, Bz]
U, F = RI.cons_flux(st, Bx, g)
eos = hybrid_eos(K=0.0, gamma=g, gamma_th=g)
s = {k: torch.tensor(a) for k, a in zip(("rho", "p", "vx", "vy", "vz", "Bx", "By", "Bz"), (rho, p, v[0], v[1], v[2], Bx, By, Bz))}
s["eps"] = eos.eps__press_rho(s["p"], s["rho"])
uL, uR, fL, fR, *_ = compute_srmhd_fluxes(s, s, eos, idir=0)
keys = [k for k in fL]
print("HLLD flux keys:", keys, " cons keys:", list(uL.keys()))
mapF = {"D": 0, "Sx": 1, "Sy": 2, "Sz": 3, "tau": 4, "By": 5, "Bz": 6}
for k, j in mapF.items():
    if k in fL:
        a = fL[k].numpy(); d = np.max(np.abs(a - F[j]) / (np.abs(a) + 1e-12))
        print("  F[%s] max rel diff %.1e" % (k, d))
    if k in uL:
        a = uL[k].numpy(); d = np.max(np.abs(a - U[j]) / (np.abs(a) + 1e-12))
        print("  U[%s] max rel diff %.1e" % (k, d))
