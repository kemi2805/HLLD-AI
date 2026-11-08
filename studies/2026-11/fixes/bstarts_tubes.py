#!/usr/bin/env python
"""Recovered lanes of a b_starts run that have tubes: the exact flux (its xi = 0
state) and HLLD's against the tube at x/t = 0, with the answer's smallest
entropy ratio over its shocks.

    python bstarts_tubes.py BSTARTS_DIR TUBE_DIR [TUBE_DIR ...]
"""
import glob
import os
import sys
import numpy as np
import torch
sys.path.insert(0, "/Users/miler/Codes/HLLD")
from src.physics.eos import hybrid_eos
from src.physics.hlld import compute_srmhd_fluxes
KEYS = ("D", "Sx", "Sy", "Sz", "tau", "By", "Bz")
pop = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz", allow_pickle=True)
off = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/replay_final/replay_off.npz")
cols = {}
for f in sorted(glob.glob(sys.argv[1] + "/shard_*.npz")):
    z = np.load(f, allow_pickle=True)
    for k in z.files:
        if k in ("tried",):
            continue
        v = z[k]
        cols.setdefault(k, []).append(v.T if k in ("kinds", "shocks") else v)
z = {k: np.concatenate(v) for k, v in cols.items()}
g = 5.0 / 3.0
eos = hybrid_eos(K=0.0, gamma=g, gamma_th=g)


def flux(st, Bn):
    rho, Pt, vx, vy, vz, By, Bz = (float(c) for c in st)
    W2 = 1 / (1 - vx * vx - vy * vy - vz * vz); vB = vx * Bn + vy * By + vz * Bz
    s = {"rho": rho, "p": Pt - 0.5 * ((Bn * Bn + By * By + Bz * Bz) / W2 + vB * vB),
         "vx": vx, "vy": vy, "vz": vz, "Bx": Bn, "By": By, "Bz": Bz}
    s = {k: torch.tensor([v], dtype=torch.float64) for k, v in s.items()}
    s["eps"] = eos.eps__press_rho(s["p"], s["rho"])
    _, _, F, _, _, _ = compute_srmhd_fluxes(s, s, eos, idir=0)
    return np.array([float(F[k][0]) for k in KEYS])


def tube_flux(path, nc):
    t = np.load(path)
    Fs = []
    for i0 in (nc // 2 - 1, nc // 2):
        q = {k: torch.as_tensor(t[k][i0:i0 + 1]) for k in ("rho", "p", "vx", "vy", "vz", "Bx", "By", "Bz")}
        q["eps"] = eos.eps__press_rho(q["p"], q["rho"])
        _, _, F0, _, _, _ = compute_srmhd_fluxes(q, q, eos, idir=0)
        Fs.append(np.array([float(F0[k][0]) for k in KEYS]))
    return 0.5 * (Fs[0] + Fs[1])


for i in np.flatnonzero(z["ok"] & ~z["ctrl"]):
    lane = int(z["lane"][i])
    paths = [p for d in sys.argv[2:] for p in sorted(glob.glob("%s/tube_%d_*.npz" % (d, lane)))]
    if not paths:
        continue
    Fex = flux([z["star_%d" % j][i] for j in range(7)], pop["UL"][lane][7])
    Fh = np.array([off["F_" + k][lane] for k in KEYS])
    msg = "lane %4d %-10s smin-1 %+.1e:" % (lane, z["how"][i], z["smin"][i] - 1)
    for pth in paths:
        nc = int(os.path.basename(pth).split("_")[-1].split(".")[0])
        Ft = tube_flux(pth, nc); sc = np.abs(Ft).max()
        msg += "  %d: HLLD %.1e exact %.1e" % (nc, np.abs(Fh - Ft).max() / sc, np.abs(Fex - Ft).max() / sc)
    print(msg)
