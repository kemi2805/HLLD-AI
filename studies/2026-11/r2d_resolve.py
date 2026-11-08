#!/usr/bin/env python
"""Solve four-quadrant faces through the production path with a chosen
warm-start network and harvest every face it answers as training data
(2026-11-02, retrain loop).

    HLLD_DIR=<HLLD> RMHD_ML_CKPT=<net> python r2d_resolve.py lost HARVEST OUT
    HLLD_DIR=<HLLD> RMHD_ML_CKPT=<net> python r2d_resolve.py perturb DATA.npz OUT --n N [--sigma 0.05]

solved   the faces the run SOLVED (every solved_*.npz row): the check that
         this rebuild of the states is right -- they must solve again.
lost     every face the run LOST -- an unsolved row without a planar-solved
         twin in its shard or the next, as `failure_ledger.py collect-harvest`
         finds them -- replayed through `exact_flux_batched` (gate 1e-2, two
         retries, every rung at its default): the ledger retried with the
         network RMHD_ML_CKPT names.
perturb  N new faces made from the dataset's own (DATA.npz, trainer schema):
         a base face -- half of them complex (exact rotations > 0.5 rad), the
         rest uniform -- with its density, gas pressure and in-plane field
         scaled by exp(sigma * normal) (the normal field by one factor for both
         sides, so it stays continuous) and the in-plane components of its
         four-velocity W v moved by sigma * normal; B_z = v_z = 0 kept, as on
         every face of the run.  Four-quadrant faces near the real ones, not a
         sampling model of them; solved the same way.

The lab-frame states are rebuilt with x as the normal (the harvest keeps the
solver frame (n, t1, t2) and not the sweep direction), the gas pressure from
the total (failure_ledger.py's `lab`).  A Harvester (only_retried=False)
writes every answered face into OUT/harvest; counts by rung are printed.
"""
import argparse
import bisect
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("HLLD_DIR", os.getcwd()))
CH = 1024


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def lost_faces(harvest):
    """(U_L, U_R) of the faces the run lost (failure_ledger.collect_harvest)."""
    files = sorted(glob.glob(os.path.join(harvest, "unsolved_*.npz")))
    solved_files = sorted(glob.glob(os.path.join(harvest, "solved_*.npz")))
    key = lambda A, B: [x.tobytes() + y.tobytes() for x, y in zip(A, B)]

    def planar_keys(fs):
        if not os.path.exists(fs):
            return set()
        z = np.load(fs)
        p = z["source"] == 1
        return set(key(z["U_L"][p], z["U_R"][p]))

    UL, UR, n_unsolved = [], [], 0
    for fu in files:
        u = np.load(fu)
        n_unsolved += u["U_L"].shape[0]
        fs = fu.replace("unsolved_", "solved_")
        j = bisect.bisect_right(solved_files, fs)
        rescued = planar_keys(fs) | planar_keys(solved_files[j] if j < len(solved_files) else "")
        m = np.array([k not in rescued for k in key(u["U_L"], u["U_R"])], bool)
        UL.append(u["U_L"][m])
        UR.append(u["U_R"][m])
    UL, UR = np.concatenate(UL), np.concatenate(UR)
    print("%s: %d unsolved rows, %d lost (no planar twin)" % (harvest, n_unsolved, len(UL)), flush=True)
    return UL, UR, float(np.load(files[0])["gamma"])


def solved_faces(harvest):
    """(U_L, U_R) of every face the run solved."""
    fs = sorted(glob.glob(os.path.join(harvest, "solved_*.npz")))
    zs = [np.load(f) for f in fs]
    UL = np.concatenate([z["U_L"] for z in zs])
    UR = np.concatenate([z["U_R"] for z in zs])
    print("%s: %d solved rows" % (harvest, len(UL)), flush=True)
    return UL, UR, float(zs[0]["gamma"])


def perturbed_faces(data, n, sigma, rng):
    z = np.load(data)
    UL0, UR0, Z = z["U_L"], z["U_R"], z["zones"]
    ang = lambda R: np.arctan2(R[:, 6], R[:, 5])
    rot = np.abs(wrap(ang(Z[:, 2]) - ang(Z[:, 1]))) + np.abs(wrap(ang(Z[:, 6]) - ang(Z[:, 5])))
    cx = np.flatnonzero(rot > 0.5)
    base = np.concatenate([rng.choice(cx, n // 2), rng.choice(len(UL0), n - n // 2)])
    gamma = float(z["gamma"])
    fBn = np.exp(sigma * rng.standard_normal(n))          # one factor, both sides
    out = []
    for U0 in (UL0[base], UR0[base]):
        rho, Pt, vx, vy, vz, By, Bz, Bx = (U0[:, j].copy() for j in range(8))
        W = 1.0 / np.sqrt(1.0 - (vx * vx + vy * vy + vz * vz))
        eta = Bx * vx + By * vy + Bz * vz
        b2 = (Bx * Bx + By * By + Bz * Bz) / W ** 2 + eta * eta
        p = Pt - 0.5 * b2
        g = lambda: np.exp(sigma * rng.standard_normal(n))
        rho, p, By, Bx = rho * g(), p * g(), By * g(), Bx * fBn
        ux, uy = W * vx + sigma * rng.standard_normal(n), W * vy + sigma * rng.standard_normal(n)
        Wn = np.sqrt(1.0 + ux * ux + uy * uy)
        vx, vy = ux / Wn, uy / Wn
        vz, Bz = 0.0 * vz, 0.0 * Bz
        eta = Bx * vx + By * vy
        b2 = (Bx * Bx + By * By) / Wn ** 2 + eta * eta
        out.append(np.stack([rho, p + 0.5 * b2, vx, vy, vz, By, Bz, Bx], axis=1))
    print("%s: %d perturbed faces (sigma %.3f) from %d complex and %d uniform bases"
          % (data, n, sigma, n // 2, n - n // 2), flush=True)
    return out[0], out[1], gamma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("lost", "solved", "perturb"))
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max", type=int, default=None,
                    help="solve only this many faces, drawn at random (a smoke test)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    from src.physics import exact_flux as EF
    from src.physics.eos import hybrid_eos
    from src.physics.harvest import Harvester
    from src.physics.hlld import LAST_DIAG

    rng = np.random.default_rng(a.seed)
    if a.mode == "lost":
        UL, UR, gamma = lost_faces(a.src)
    elif a.mode == "solved":
        UL, UR, gamma = solved_faces(a.src)
    else:
        UL, UR, gamma = perturbed_faces(a.src, a.n, a.sigma, rng)
    if a.max is not None and a.max < len(UL):
        pick = np.sort(rng.choice(len(UL), a.max, replace=False))
        UL, UR = UL[pick], UR[pick]
        print("   --max: %d of them, at random" % a.max, flush=True)
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
    tot = dict(faces=0, gated=0, attempted=0, exact=0, seven=0, planar5=0, planar4=0)
    t0 = time.time()
    for c0 in range(0, len(UL), CH):
        sL, sR = lab(UL[c0:c0 + CH]), lab(UR[c0:c0 + CH])
        EF.exact_flux_batched(sL, sR, eos, idir=0, tau_weak=1e-2, tau_bt=1e-9, n_retries=2,
                              max_iter=40, harvester=H)
        # one shard per call: the harvester stacks a shard's coverage maps,
        # which only works for calls of one size (a 2D run's sweeps are), and
        # the last chunk is shorter
        H.flush()
        d = dict(LAST_DIAG)
        sel = np.asarray(d["sel"])
        ex = d["exact_mask"].numpy()[sel]
        p5 = np.asarray(d["planar5_mask"], bool) & ex
        p4 = (np.asarray(d["planar4_mask"], bool) & ex) if "planar4_mask" in d else np.zeros_like(ex)
        tot["faces"] += len(UL[c0:c0 + CH])
        tot["gated"] += len(UL[c0:c0 + CH]) - sel.size
        tot["attempted"] += sel.size
        tot["exact"] += int(ex.sum())
        tot["planar5"] += int(p5.sum())
        tot["planar4"] += int(p4.sum())
        tot["seven"] += int((ex & ~p5 & ~p4).sum())
        if (c0 // CH) % 20 == 0:
            print("   %7d / %d faces  %.0f s  exact %d of %d attempted" % (
                c0 + CH, len(UL), time.time() - t0, tot["exact"], tot["attempted"]), flush=True)
    info = H.close()
    att = max(tot["attempted"], 1)
    print("== %s: %d faces, %d under the gate, %d attempted: exact %d (%.2f%%) -- seven-wave (incl. its "
          "rescues) %d, planar five-wave %d, four-unknown %d; harvested %s solved / %s unsolved; %.0f s"
          % (a.mode, tot["faces"], tot["gated"], tot["attempted"], tot["exact"], 100.0 * tot["exact"] / att,
             tot["seven"], tot["planar5"], tot["planar4"], info.get("harvest_solved"),
             info.get("harvest_unsolved"), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
