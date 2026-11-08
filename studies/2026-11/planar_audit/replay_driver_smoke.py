#!/usr/bin/env python
"""Replay the four-quadrant 64^2 run's inadmissible faces through the
production path under the CURRENT environment (planar acceptance study,
2026-11-03).  The job sets the configuration:

    RMHD_ADMISSIBLE=report|1            HLLD-admiss: count / refuse at acceptance
    RMHD_SLOWSHOCK_ADMISSIBLE=1          rmhd_final-admiss: admissible slow-shock roots only
    RMHD_SLOWSHOCK=numpy                 (the filter is numpy-only; its baseline too)

    HLLD_DIR=<HLLD-admiss> python r2d_admiss_replay.py FLAGGED.npz ALL.npz OUT [--n-control N] [--max N]

Faces: every unique problem admiss_all.py flagged in it1/all.npz (entropy,
Lax or slow/fast order), plus N random unflagged problems of the same file
as controls.  States are rebuilt as r2d_resolve.py does (x the normal, gas
pressure from the total) and solved in chunks of 1024 with the run's
arguments (tau_weak 1e-2, two retries, 40 iterations).  Saves OUT/faces.npz
(per face: flux, exact, rung, group) and OUT/harvest (solved rows, for an
independent admissibility check).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("HLLD_DIR", os.getcwd()))
CH = 1024


def keys16(UL, UR):
    a = np.ascontiguousarray(np.concatenate([UL, UR], axis=1), dtype=np.float64)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flagged")
    ap.add_argument("all")
    ap.add_argument("out")
    ap.add_argument("--n-control", type=int, default=5000)
    ap.add_argument("--max", type=int, default=None)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    from src.physics import exact_flux as EF
    from src.physics.eos import hybrid_eos
    from src.physics.harvest import Harvester
    from src.physics.hlld import LAST_DIAG
    import rmhd
    print("== rmhd", rmhd.__file__, " HLLD", EF.__file__, flush=True)
    print("== env", {k: os.environ.get(k) for k in ("RMHD_ADMISSIBLE", "RMHD_SLOWSHOCK_ADMISSIBLE",
                                                   "RMHD_SLOWSHOCK", "RMHD_SHOCK", "RMHD_FAN", "RMHD_ALFVEN",
                                                   "RMHD_ML_CKPT", "RMHD_ML_CKPTS", "RMHD_POOL")}, flush=True)
    fz = np.load(a.flagged)
    UL, UR = np.asarray(fz["all_U_L"], float), np.asarray(fz["all_U_R"], float)
    kf = keys16(UL, UR)
    _, u = np.unique(kf, return_index=True)
    u = np.sort(u)
    ULf, URf = UL[u], UR[u]
    za = np.load(a.all)
    ULa, URa = np.asarray(za["U_L"], float), np.asarray(za["U_R"], float)
    ka = keys16(ULa, URa)
    _, ua = np.unique(ka, return_index=True)
    ua = ua[~np.isin(ka[ua], kf)]
    rng = np.random.default_rng(1103)
    pick = np.sort(rng.choice(ua, min(a.n_control, ua.size), replace=False))
    UL = np.concatenate([ULf, ULa[pick]])
    UR = np.concatenate([URf, URa[pick]])
    group = np.concatenate([np.zeros(len(ULf), np.int8), np.ones(len(pick), np.int8)])
    src_row = np.concatenate([np.asarray(fz["all_idx"])[u], pick])
    if a.max is not None and a.max < len(UL):
        s = np.sort(rng.choice(len(UL), a.max, replace=False))
        UL, UR, group, src_row = UL[s], UR[s], group[s], src_row[s]
    print("== %d flagged problems + %d controls = %d faces" % ((group == 0).sum(), (group == 1).sum(), len(UL)),
          flush=True)
    gamma = float(za["gamma"])
    eos = hybrid_eos(K=0.0, gamma=gamma, gamma_th=gamma)

    def lab(U):
        rho, Pt, vx, vy, vz, By, Bz, Bx = (U[:, j] for j in range(8))
        W2 = 1.0 / (1.0 - (vx * vx + vy * vy + vz * vz))
        eta = Bx * vx + By * vy + Bz * vz
        b2 = (Bx * Bx + By * By + Bz * Bz) / W2 + eta * eta
        t = lambda v: torch.tensor(np.ascontiguousarray(v), dtype=torch.float64)
        d = {"rho": t(rho), "vx": t(vx), "vy": t(vy), "vz": t(vz), "p": t(Pt - 0.5 * b2),
             "Bx": t(Bx), "By": t(By), "Bz": t(Bz)}
        d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
        return d

    os.makedirs(a.out, exist_ok=True)
    H = Harvester(os.path.join(a.out, "harvest"), gamma, stride=1, max_solved=10 ** 8,
                  max_unsolved=10 ** 8, only_retried=False, flush_every=20000, rank=0)
    n = len(UL)
    exact = np.zeros(n, bool)
    rung = np.zeros(n, np.int8)            # 0 not exact, 1 seven, 2 planar5, 3 planar4, 4 other exact
    linfb = np.zeros(n, bool)
    inadm7 = np.zeros(n, bool)
    Fs, keys = None, None
    counts = {}
    t0 = time.time()
    for c0 in range(0, n, CH):
        c1 = min(n, c0 + CH)
        F, _, _ = EF.exact_flux_batched(lab(UL[c0:c1]), lab(UR[c0:c1]), eos, idir=0, tau_weak=1e-2,
                                        tau_bt=1e-9, n_retries=2, max_iter=40, harvester=H)
        H.flush()
        if keys is None:
            keys = sorted(F.keys())
            Fs = np.zeros((n, len(keys)))
        Fs[c0:c1] = np.stack([F[k].detach().cpu().numpy().reshape(-1) for k in keys], axis=1)
        d = dict(LAST_DIAG)
        sel = np.asarray(d["sel"])
        ex = d["exact_mask"].numpy()[sel]
        p5 = np.asarray(d["planar5_mask"], bool) & ex
        p4 = np.asarray(d.get("planar4_mask", np.zeros(sel.size, bool)), bool) & ex
        r = np.where(ex, np.where(p5, 2, np.where(p4, 3, 1)), 0).astype(np.int8)
        exact[c0 + sel] = ex
        rung[c0 + sel] = r
        linfb[c0 + sel] = np.asarray(d.get("linear_fallback_mask", np.zeros(sel.size, bool)), bool)
        if "inadmissible7_mask" in d:
            inadm7[c0 + sel] = np.asarray(d["inadmissible7_mask"], bool)
        for k in ("n_attempted", "n_exact", "n_planar5_exact", "n_planar4_exact", "n_inadmissible7",
                  "n_wave_order_rejected", "n_hlld_after_attempt", "n_linear_fallback"):
            counts[k] = counts.get(k, 0) + int(d.get(k, 0))
        for st in ("inadmissible7", "inadmissible5", "inadmissible4"):
            for kk, vv in (d.get(st) or {}).items():
                counts[st + "_" + kk] = counts.get(st + "_" + kk, 0) + int(vv)
        print("   %5d / %d faces  %.0f s  %s" % (c1, n, time.time() - t0,
              {k: counts[k] for k in ("n_attempted", "n_exact", "n_planar5_exact", "n_planar4_exact")}), flush=True)
    H.close()
    np.savez_compressed(os.path.join(a.out, "faces.npz"), U_L=UL, U_R=UR, group=group, src_row=src_row,
                        F=Fs, F_keys=np.array(keys), exact=exact, rung=rung, linear_fb=linfb,
                        inadm7=inadm7, wall=time.time() - t0)
    print("== done %.0f s  %s" % (time.time() - t0, counts), flush=True)
    for g, name in ((0, "flagged"), (1, "control")):
        m = group == g
        print("   %-8s faces %5d exact %5d  seven %5d planar5 %5d planar4 %5d  not exact %5d" % (
            name, m.sum(), (exact & m).sum(), ((rung == 1) & m).sum(), ((rung == 2) & m).sum(),
            ((rung == 3) & m).sum(), (~exact & m).sum()), flush=True)


if __name__ == "__main__":
    main()
