#!/usr/bin/env python
"""Capture every exact answer the production path (and the loss fixes, per
the env) gives on a failure-ledger population, in the harvest schema, for
verify_rmhd.py: seven-wave and planar answers through a Harvester
(only_retried off), four-unknown answers by wrapping
exact_flux._planar4_and_ray (run with RMHD_POOL=0 so every call is seen).

    python capture_answers.py POP.npz OUTDIR [--shard k --nshards n]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HLLD = os.environ.get("HLLD_DIR", "/mnt/rafast/miler/codes/rotor2d/HLLD")
sys.path.insert(0, HLLD)
from rmhd.eos import set_eos                                     # noqa: E402
set_eos("ideal")
from src.physics import exact_flux as EF                         # noqa: E402
from src.physics.eos import hybrid_eos                           # noqa: E402
from src.physics.harvest import Harvester                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pop"); ap.add_argument("out")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    p = np.load(a.pop, allow_pickle=True)
    idx = np.arange(len(p["UL"]))[a.shard::a.nshards]
    g = float(p["gamma"])
    eos = hybrid_eos(K=0.0, gamma=g, gamma_th=g)
    hdir = os.path.join(a.out, "harvest_%03d" % a.shard)
    os.makedirs(hdir, exist_ok=True)
    H = Harvester(hdir, g, stride=1, max_solved=10 ** 7, max_unsolved=10 ** 7,
                  only_retried=False, flush_every=10 ** 9, rank=a.shard)
    rows = []
    orig = EF._planar4_and_ray

    def wrapped(P, gamma, sL4, sR4, B4, **kw):
        out = orig(P, gamma, sL4, sR4, B4, **kw)
        good = np.asarray(out["good"], bool)
        if good.any():
            i = np.flatnonzero(good)
            u8 = lambda s: np.stack([np.asarray(c)[i] for c in s] + [np.asarray(B4)[i]], axis=1)
            Z = np.zeros((i.size, 8, 7))
            Z[:, 0] = u8(sL4)[:, :7]
            Z[:, 7] = u8(sR4)[:, :7]
            for z in range(6):
                for j in range(7):
                    Z[:, z + 1, j] = np.asarray(out["zones6"][z][j])[i]
            VL, VR = out["VsLv"], out["VsRv"]
            S = np.stack([VL[0][i], VL[1][i], VL[2][i],
                          0.5 * (Z[:, 3, 2] + Z[:, 4, 2]),
                          VR[2][i], VR[1][i], VR[0][i]], axis=1)
            rows.append(dict(U_L=u8(sL4), U_R=u8(sR4), zones=Z, speeds=S,
                             kinds=np.asarray(out["kinds"])[:, i].T))
        return out

    EF._planar4_and_ray = wrapped

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

    t0 = time.time()
    for c0 in range(0, idx.size, 256):
        j = idx[c0:c0 + 256]
        EF.exact_flux_batched(lab(p["UL"][j]), lab(p["UR"][j]), eos, idir=0, tau_weak=1e-2,
                              tau_bt=1e-9, n_retries=2, max_iter=40, harvester=H)
    H.flush()
    if rows:
        np.savez_compressed(os.path.join(a.out, "planar4_%03d.npz" % a.shard), gamma=g,
                            **{k: np.concatenate([r[k] for r in rows]) for k in rows[0]})
    print("shard %d: %d lanes, %d four-unknown answers, %.0f s" % (
        a.shard, idx.size, sum(len(r["U_L"]) for r in rows), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
