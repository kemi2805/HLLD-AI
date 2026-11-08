#!/usr/bin/env python3
"""Production's seed on the planar dataset's validation and test rows, in
planar_data.py's target units: the baseline the planar network has to beat.

The late rung starts from the sweep's seven-wave seed. That seed is the
primary checkpoint's candidate 0, clamped (exact_flux._ml_seeds; production:
ml_guess_rotor_ft.pt). coplanar7_b.planar_seed converts it into the planar
unknowns: both slow waves are steered to the contact field
|Bt_CD| cos(psi_CD + alpha). The pressure-steered stage steers both slow waves
to the total pressure behind the left one, read off planar4's structure at
that seed. The same numbers are mapped here through planar_data.rows_to_arrays,
so they land in the targets' frame and units:

  ln p_LF, ln p_RF   seed6[0], seed6[3]
  By_CD              the seed's contact field (lab frame), rotated and scaled
  ln P_CD            the structure's zone-B total pressure (the late rung's lpB)
  rot                the rotations the seed suggests (|phi| > pi/2)

The features are rebuilt from the stored states on the way. They must equal
the dataset's own; the largest difference is printed as a check that the rows
line up.

    python3 planar_baseline.py DATA.npz OUT.npz [--from 0.7] [--ckpt data/ml_guess_rotor_ft.pt]
                               [--workers 32] [--chunk 10000]

Nearly all of the time is planar4's structure at the seed (the fast fans'
integration): ~3 ms a row on one core, hence the worker processes (spawned,
one thread each, every chunk independent).

Needs HLLD_DIR, and rmhd on the path (the calea worktrees).
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HLLD = os.environ["HLLD_DIR"]
sys.path.insert(0, HLLD)
from rmhd.eos import set_eos                                     # noqa: E402
set_eos("ideal")
from rmhd.batched import coplanar7_b as CP7                      # noqa: E402
from rmhd.batched import planar4_b as P4                         # noqa: E402
from rmhd.batched.planar5_b import to_planar                     # noqa: E402
from src.physics import exact_flux as EF                         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planar_data as PD                                         # noqa: E402

_SOLVERS = {}


def _solvers(gamma):
    if gamma not in _SOLVERS:
        _SOLVERS[gamma] = (CP7.make_solver(gamma), P4.make_solver(gamma))
    return _SOLVERS[gamma]


def baseline(model, scaler, UL, UR, src, gamma):
    """The production seed of n rows as (features, targets, rotations, ok)."""
    n = len(UL)
    L = [np.ascontiguousarray(UL[:, j]) for j in range(7)]
    Rr = [np.ascontiguousarray(UR[:, j]) for j in range(7)]
    Bn = np.ascontiguousarray(UL[:, 7])
    s6 = [np.asarray(c, dtype=float) for c in EF._ml_seeds(model, scaler, L, Rr, Bn, 0)["seed6"]]
    C7, S4 = _solvers(gamma)
    u0, rot0 = C7["planar_seed"](L, Rr, Bn, s6)
    # the pressure-steered stage's seed, as coplanar7_b.solve builds it
    L0, R0 = to_planar(L, Rr, Bn)[:2]
    combos = np.array(CP7._COMBOS, dtype=bool)
    sugg = rot0[0].astype(int) + 2 * rot0[1].astype(int)
    with np.errstate(all="ignore"):
        st0 = S4["structure"](L0, R0, u0, Bn, np.full((2, n), P4.KIND_S, np.int8),
                              rot=combos[sugg].T)
        e0 = st0[3]
        lp_mean = np.log(np.maximum(0.5 * (L0[1] + R0[1]), 1e-300))
        lpB = np.where(e0, lp_mean, np.log(np.maximum(st0[1][1][1], 1e-300)))
        Z = np.zeros((n, 8, 7))
        Z[:, 0, :] = UL[:, :7]
        Z[:, 7, :] = UR[:, :7]
        Z[:, 1, 1] = np.exp(s6[0])
        Z[:, 6, 1] = np.exp(s6[3])
        bt = np.exp(s6[1])
        Z[:, 3, 5] = bt * np.cos(s6[2])
        Z[:, 3, 6] = bt * np.sin(s6[2])
        Z[:, 3, 1] = np.exp(lpB)
        X, Y, _, _, _, ok = PD.rows_to_arrays(UL, UR, Z, src, None, None, gamma)
    return X, Y, np.stack(rot0, axis=1), ok, e0


_W = {}


def _init(ckpt):
    import torch
    torch.set_num_threads(1)
    _W["model"] = EF._load_model_cached(EF._rmhd_paths.resolve(ckpt))


def _work(args):
    UL, UR, src, g = args
    model, scaler = _W["model"]
    return baseline(model, scaler, UL, UR, src, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--from", dest="frm", type=float, default=0.7,
                    help="rows with t_frac at least this (validation and test)")
    ap.add_argument("--ckpt", default="data/ml_guess_rotor_ft.pt")
    ap.add_argument("--chunk", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    d = np.load(a.data, allow_pickle=True)
    idx = np.flatnonzero(d["t_frac"] >= a.frm)
    UL, UR, X, src = d["UL"][idx], d["UR"][idx], d["X"][idx], d["source"][idx]
    gam = X[:, 13]
    n = len(idx)
    Yb = np.full((n, 4), np.nan)
    Rb = np.zeros((n, 2), bool)
    okb = np.zeros(n, bool)
    errb = np.zeros(n, bool)
    dX = np.zeros(n)
    chunks = []
    for g in np.unique(gam):
        m = np.flatnonzero(gam == g)
        chunks += [(m[b:b + a.chunk], float(g)) for b in range(0, m.size, a.chunk)]
    args = [(UL[j], UR[j], src[j], g) for j, g in chunks]
    if a.workers > 1:
        pool = mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.ckpt,))
        results = pool.imap(_work, args)
    else:
        _init(a.ckpt)
        pool, results = None, map(_work, args)
    for c, ((j, g), (Xb, Y, rot, ok, e0)) in enumerate(zip(chunks, results)):
        Yb[j], Rb[j], okb[j], errb[j] = Y, rot, ok, e0
        with np.errstate(invalid="ignore"):
            dX[j] = np.nanmax(np.abs(Xb - X[j]), axis=1)
        if (c + 1) % 10 == 0 or c + 1 == len(chunks):
            print("chunk %d/%d (%.0f s)" % (c + 1, len(chunks), time.time() - t0), flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    print("rows %d, finite baseline %d (%.2f%%), structure failed at the seed %d; "
          "features rebuilt vs stored: max |dX| %.3g" % (
              n, okb.sum(), 100.0 * okb.mean(), errb.sum(), np.nanmax(dX)), flush=True)
    np.savez_compressed(a.out, idx=idx, Yb=Yb, rot=Rb, ok=okb, struct_err=errb, ckpt=a.ckpt,
                        frm=a.frm)
    print("saved %s (%.0f s)" % (a.out, time.time() - t0))


if __name__ == "__main__":
    main()
