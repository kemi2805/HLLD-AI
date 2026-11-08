#!/usr/bin/env python
"""Paired one-attempt test of warm-start networks on held-out four-quadrant
faces (2026-11-02, retrain loop).

    HLLD_DIR=<HLLD checkout> python r2d_eval_seeds.py EVAL.npz OUT.npz N name=ckpt [name=ckpt ...]

From EVAL.npz (r2d_split.py) N faces drawn at random -- the natural mix -- and
every complex face (exact rotations > 0.5 rad, up to 2000), each solved by
the seven-wave Newton, one attempt, max_iter 40, accuracy 1e-8 (production's
budget), from every named network's seed, clamped as production clamps it,
and from the solver's neutral guess (mean total pressure, mean tangential
field, no rotation).  Prints, per arm, converged on the natural draw and on
the complex faces, Newton iterations, and for every network the union with
the neutral guess (both starts, one attempt each).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("HLLD_DIR", os.getcwd()))


def main():
    ev_path, out_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    nets = [s.split("=", 1) for s in sys.argv[4:]]
    from rmhd.eos import set_eos
    set_eos("ideal")
    from rmhd import ml_guess as mg
    from rmhd.batched import api as API
    from rmhd.batched import ml_b as MB
    from src.physics import exact_flux as EF

    z = np.load(ev_path)
    gamma = float(z["gamma"])
    rng = np.random.default_rng(20261102)
    N = z["U_L"].shape[0]
    nat = np.sort(rng.choice(N, size=min(n, N), replace=False))
    cx_all = np.flatnonzero(z["rot"] > 0.5)
    cx = np.sort(rng.choice(cx_all, size=min(2000, cx_all.size), replace=False))
    idx = np.union1d(nat, cx)
    in_nat, in_cx = np.isin(idx, nat), np.isin(idx, cx)
    UL, UR = z["U_L"][idx], z["U_R"][idx]
    left = [UL[:, j].copy() for j in range(7)]
    right = [UR[:, j].copy() for j in range(7)]
    Bn = UL[:, 7].copy()
    print("%d held-out faces: %d natural draw, %d complex (of %d held out, %d complex)"
          % (idx.size, nat.size, cx.size, N, cx_all.size), flush=True)
    solve = API.make_solver(gamma)
    out = dict(idx=idx, in_nat=in_nat, in_cx=in_cx)
    arms = [(name, path) for name, path in nets] + [("neutral", None)]
    for name, path in arms:
        t0 = time.time()
        if path is None:
            seed = None
        else:
            model, scaler = mg.load(EF._rmhd_paths.resolve(path))
            u, _ = MB.predict_unk6(model, scaler, np.stack(left, axis=1), np.stack(right, axis=1), Bn)
            u, _ = MB.clamp_seed_physical(u, left, right, Bn)
            seed = [np.array(u[c], float) for c in range(6)]
        res, _ = solve(left, right, Bn, seed6=seed, accuracy=1e-8, max_iter=40, n_retries=0, tau_bt=1e-9)
        conv = np.asarray(res["converged"], bool)
        it = np.asarray(res["n_iter"], float)
        out[name + " conv"] = conv
        out[name + " iters"] = it
        print("   %-18s natural %5.1f%%   complex %5.1f%%   iterations (converged) %.2f   %.0f s"
              % (name, 100 * conv[in_nat].mean(), 100 * conv[in_cx].mean(),
                 it[conv].mean() if conv.any() else np.nan, time.time() - t0), flush=True)
    neu = out["neutral conv"]
    print("   with the neutral guess as a second start (one attempt each):")
    for name, _ in nets:
        u = out[name + " conv"] | neu
        print("   %-18s natural %5.1f%%   complex %5.1f%%" % (name + " + neutral", 100 * u[in_nat].mean(),
                                                               100 * u[in_cx].mean()), flush=True)
    np.savez_compressed(out_path, **out)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
