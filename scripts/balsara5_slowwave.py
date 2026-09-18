"""Balsara (2001) test 5: the 1D problem where the slow waves carry the physics.

Runs the 1D shock tube with the HLLD flux and with the exact flux, computes
the exact self-similar solution on the same cells and on a fine grid, and
saves everything one plot needs -- the profiles, the seven wave speeds, the
L1 errors and the fraction of interfaces the exact solver actually owned.

Why this problem: all seven waves are present with comparable strength, the
tangential fields are NOT coplanar (chi = 0.986, against 7.6e-11 on the
rotor), so the seven-wave solver is well posed and the warm-start network is
in its home distribution -- measured 2026-09-14, both production checkpoints
converge from the pure network seed with no jitter.  And it carries a large
rotational discontinuity (phi_L = 1.59 rad) beside a resolved slow wave
(RA 0.410 vs RS 0.364): structure the planar family cannot represent and
HLLD, which drops the slow waves, does not resolve.

    python scripts/balsara5_slowwave.py [--ncells 400] [--out results/balsara5]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.driver import run                      # noqa: E402
from src.physics.hlld import LAST_DIAG                  # noqa: E402

VARS = ("rho", "p", "vx", "vy", "vz", "By", "Bz")


def exact_solution(cfg, x_cells, x_fine, t):
    """Solve the Riemann problem ONCE from the network seed; read it at rays."""
    from rmhd.eos import set_eos
    set_eos("ideal")
    from rmhd import ml_guess as mg, paths as rmhd_paths
    from rmhd.batched import api as API
    from rmhd.batched import ml_b as MB
    from rmhd.batched import profile as PR

    rc = cfg["run"]
    g = float(cfg["eos"]["gamma"])
    x0 = float(rc["x_interface"])

    l7, Bn = PR.seven_from_prim(rc["primL"])
    r7, _ = PR.seven_from_prim(rc["primR"])

    ck = os.environ.get("RMHD_ML_CKPT", "data/ml_guess_gamma53_v5.pt")
    model, scaler = mg.load(rmhd_paths.resolve(ck))
    L1 = [np.array([c]) for c in l7]
    R1 = [np.array([c]) for c in r7]
    B1 = np.array([Bn])
    s0, _ = MB.predict_unk6(model, scaler, np.stack(L1, 1), np.stack(R1, 1), B1)
    seed = [np.array([s0[c][0]]) for c in range(6)]
    t0 = time.perf_counter()
    r, diag = API.make_solver(g)(L1, R1, B1, seed6=seed, accuracy=1e-9, max_iter=60)
    t_solve = time.perf_counter() - t0
    if not bool(r["converged"][0]):
        raise RuntimeError("the network seed did not converge: %r" % diag)
    unk = [float(c[0]) for c in r["unk"]]
    VsL = [float(v[0]) for v in r["VsLv"]]      # LF, LA, LS
    VsR = [float(v[0]) for v in r["VsRv"]]      # RF, RA, RS
    zones0 = [[float(z[k][0]) for k in range(7)] for z in r["zones"]]   # R2..R7
    cd = 0.5 * (zones0[2][2] + zones0[3][2])
    speeds = np.array([VsL[0], VsL[1], VsL[2], cd, VsR[2], VsR[1], VsR[0]])

    sampler = PR.make_sampler(g)

    def read(x):
        prim, region, err = PR.profile_at(l7, r7, Bn, zones0, VsL, VsR, g,
                                          (x - x0) / t, sampler)
        return dict(prim, region=region, err=err)

    return dict(cells=read(x_cells), fine=read(x_fine), speeds=speeds,
                unk=np.array(unk), seed=np.array([s0[c][0] for c in range(6)]),
                t_solve=t_solve, x0=x0, Bn=Bn, gamma=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/Giacomazzo/Balsara5.yaml")
    ap.add_argument("--ncells", type=int, default=400)
    ap.add_argument("--nfine", type=int, default=4000)
    ap.add_argument("--solvers", default="hlld,exact")
    ap.add_argument("--out", default="results/balsara5")
    ap.add_argument("--tau-weak", type=float, default=None,
                    help="weak-jump gate; 0 attempts EVERY interface (a fully "
                         "exact run). Default leaves the driver's 1e-6.")
    ap.add_argument("--progress-every", type=int, default=20,
                    help="steps between progress lines; the exact arm can take "
                         "hours and its per-step cost grows as the fans widen, "
                         "so a silent run is indistinguishable from a hung one")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg0 = yaml.safe_load(open(a.config))
    t_end = float(cfg0["run"]["t_end"])
    save = dict(ncells=a.ncells, t_end=t_end, config=a.config)

    for solver in a.solvers.split(","):
        cfg = yaml.safe_load(open(a.config))
        cfg["grid"]["ncells"] = a.ncells
        cfg["run"]["solver"] = solver
        cfg["output"]["dir"] = os.path.join(a.out, "snap_" + solver)
        cfg["output"]["every_n_steps"] = a.progress_every
        if a.tau_weak is not None:
            cfg["run"]["tau_weak"] = a.tau_weak
        t0 = time.perf_counter()
        _t, grid, prims, _cons = run(cfg)
        wall = time.perf_counter() - t0
        d = dict(LAST_DIAG)
        for k in ("n_interfaces", "n_weak_gate", "n_upwind_skip", "n_attempted",
                  "n_exact", "n_hlld_fallback", "n_verified",
                  "n_wave_order_rejected", "n_planar5_exact"):
            if k in d:
                save["%s_%s" % (solver, k)] = int(d[k])
        ng = getattr(grid, "ng", 2)
        save["x"] = grid.x[ng:-ng].numpy()
        for v in VARS:
            save["%s_%s" % (solver, v)] = prims[v][ng:-ng].numpy()
        save["%s_wall" % solver] = wall
        save["%s_frac_exact" % solver] = float(d.get("frac_exact", 0.0))
        print("  %-6s %7.1f s   exact fraction %5.1f%%   steps %s"
              % (solver, wall, 100 * d.get("frac_exact", 0.0), d.get("n_steps", "?")),
              flush=True)

    x = save["x"]
    xf = np.linspace(x.min(), x.max(), a.nfine)
    ex = exact_solution(cfg0, x, xf, t_end)
    for v in VARS:
        save["ref_" + v] = ex["cells"][v]
        save["fine_" + v] = ex["fine"][v]
    save["x_fine"] = xf
    save["ref_region"] = ex["cells"]["region"]
    save["fine_region"] = ex["fine"]["region"]
    save["speeds"] = ex["speeds"]            # LF LA LS CD RS RA RF
    save["unk"] = ex["unk"]; save["seed"] = ex["seed"]
    save["t_solve"] = ex["t_solve"]; save["x0"] = ex["x0"]; save["Bn"] = ex["Bn"]
    print("  exact solve from the network seed: %.2f s;  seed error per unknown: %s"
          % (ex["t_solve"], np.round(np.abs(ex["unk"] - ex["seed"]), 3)), flush=True)
    print("  wave speeds [LF LA LS CD RS RA RF] = %s" % np.round(ex["speeds"], 4), flush=True)

    print("\n  %-8s" % "L1 error" + "".join("%14s" % s for s in a.solvers.split(",")), flush=True)
    for v in VARS:
        row = "  %-8s" % v
        for s in a.solvers.split(","):
            e = float(np.abs(save["%s_%s" % (s, v)] - save["ref_" + v]).mean())
            save["L1_%s_%s" % (s, v)] = e
            row += "%14.4e" % e
        print(row, flush=True)
    np.savez_compressed(os.path.join(a.out, "balsara5_%d.npz" % a.ncells), **save)
    print("  wrote %s" % os.path.join(a.out, "balsara5_%d.npz" % a.ncells), flush=True)


if __name__ == "__main__":
    main()
