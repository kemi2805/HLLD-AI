#!/usr/bin/env python3
"""The 1D flux studies of 2026-10-15: when does the exact flux beat the
approximate ones in one dimension, and why so rarely.

The September tubes (docs/tests.tex, Balsara 5 and the constructed slow-shock
tube) ran FIRST order -- the 1D driver's default reconstruction is "pcm" and
neither config sets one -- with the exact flux on only 27-45% of the faces.
These experiments redo them with today's solver chain and add what was
missing:

  tubes     Balsara 5 and the two constructed slow-shock tubes (sst: the
            committed config, B_n = 1.5; sst0: the earlier variant, B_n = 1,
            |B_t| = 2, whose Alfven-to-slow band is 2.5 cells at 400 cells),
            HLLE / HLLC / HLLD / exact, first order (pcm) and second order
            (mc), 100 to 1600 cells, each against the exact self-similar
            solution of its Riemann problem.
  standing  one slow shock in its own rest frame, sitting on a cell face
            (slowA: the slow shock of sst0, slowB: that of sst), and the same
            shock moving slowly (s = 0.01, 0.05).  An exact Godunov flux keeps
            a stationary discontinuity exact -- the face sees the shock's two
            states and returns F(U_L) = F(U_R) -- where every approximate
            flux misses that face flux by O(1) (7.5 absolute for HLLE, HLLC
            and HLLD on slowA).  The moving shock tests the post-shock
            oscillations that exact Godunov schemes are known for.

Each task is ONE run (a flux, a reconstruction, a resolution), so a job can
run many side by side on one node, one core each:

    python scripts/tube_study.py list tubes --out results/tube_study > tasks.txt
    xargs -P 56 -L 1 -a tasks.txt python scripts/tube_study.py
    python scripts/tube_study.py analyse results/tube_study

The exact arm uses the production settings of the 2D runs: the weak-jump
gate at 1e-2 with the linearised flux below it, two retries, every rung on
(the modules' defaults), the checkpoints of the environment (the calea
launcher exports the production three).  Coverage -- what happened to every
face -- comes from the harvester's per-face tally and is saved with the run.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import glob
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

VARS = ("rho", "p", "vx", "vy", "vz", "By", "Bz")
SOLVERS = ("hlle", "hllc", "hlld", "exact")
LIMITERS = ("pcm", "mc")
ALL_N = (100, 200, 400, 800, 1600)
G53 = 5.0 / 3.0

TUBE_CONFIGS = {"b5": "configs/Giacomazzo/Balsara5.yaml",
                "sst": "configs/slow_shock_tube.yaml"}
# make_slow_shock_tube.py arguments of the two constructed tubes
TUBE_BUILD = {"sst0": dict(),                                   # the script's defaults
              "sst": dict(Bn=1.5, Bt=0.75, p=0.05, factor=1.5)}  # the committed config
# the standing shocks: the LEFT slow shock of these constructed tubes
SHOCK_FROM = {"slowA": "sst0", "slowB": "sst"}


# ── cases ────────────────────────────────────────────────────────────────────

def _build_args(**kw):
    d = dict(rho=1.0, p=0.1, vx=0.1, vy=0.0, vz=0.0, Bn=1.0, Bt=2.0, fast=1.15,
             rot=0.5, factor=2.0, contact=2.0)
    d.update(kw)
    return argparse.Namespace(**d)


def _prim8(st, Bn):
    """Solver seven-state (TOTAL pressure) -> the driver's primitive dict."""
    import make_slow_shock_tube as M
    rho, P, vx, vy, vz, By, Bz = (float(c) for c in st)
    return dict(rho=rho, p=float(M.gas_p(st, Bn)), vx=vx, vy=vy, vz=vz,
                Bx=float(Bn), By=By, Bz=Bz)


def boost_x(q, beta):
    """The primitive state ``q`` seen from a frame moving at ``beta`` along x.

    rho and p are rest-frame scalars; u^mu and b^mu are four-vectors; the lab
    field is B^i = u^0 b^i - u^i b^0.  B_x is invariant under a boost along x.
    """
    vx, vy, vz = q["vx"], q["vy"], q["vz"]
    Bx, By, Bz = q["Bx"], q["By"], q["Bz"]
    W = 1.0 / math.sqrt(1.0 - (vx * vx + vy * vy + vz * vz))
    u0, ux, uy, uz = W, W * vx, W * vy, W * vz
    b0 = W * (vx * Bx + vy * By + vz * Bz)
    bx, by, bz = Bx / W + b0 * vx, By / W + b0 * vy, Bz / W + b0 * vz
    g = 1.0 / math.sqrt(1.0 - beta * beta)
    u0p, uxp = g * (u0 - beta * ux), g * (ux - beta * u0)
    b0p, bxp = g * (b0 - beta * bx), g * (bx - beta * b0)
    # u0p * bxp - uxp * b0p is Bx again, to rounding: keep it exact, so both
    # sides of a face carry the same normal field
    return dict(rho=q["rho"], p=q["p"], vx=uxp / u0p, vy=uy / u0p, vz=uz / u0p,
                Bx=Bx, By=u0p * by - uy * b0p, Bz=u0p * bz - uz * b0p)


def _base_cfg(L, R, gamma, t_end, x0=0.5):
    return dict(grid=dict(xmin=0.0, xmax=1.0, ncells=400, ng=2),
                eos=dict(K=0.0, gamma=gamma, gamma_th=gamma),
                run=dict(problem="generic", t_end=t_end, cfl=0.4, atmo_rho=1e-10,
                         solver="hlld", device="cpu", primL=L, primR=R,
                         x_interface=x0),
                output=dict(dir="output", every_n_steps=1000))


def jump_residual(L, R, s, gamma):
    """max_k |F_R - F_L - s (U_R - U_L)|, relative to the largest flux component."""
    import torch
    from src.physics.eos import hybrid_eos
    from src.physics.hlld import compute_srmhd_fluxes
    eos = hybrid_eos(K=0.0, gamma=gamma, gamma_th=gamma)
    t = lambda q: {k: torch.tensor([float(v)], dtype=torch.float64) for k, v in q.items()}
    pL, pR = t(L), t(R)
    for P in (pL, pR):
        P["eps"] = eos.eps__press_rho(P["p"], P["rho"])
    uL, uR, fL, fR, _, _ = compute_srmhd_fluxes(pL, pR, eos, idir=0)
    keys = ("D", "Sx", "Sy", "Sz", "tau", "By", "Bz")
    scale = max(max(abs(float(fL[k])), abs(float(fR[k]))) for k in keys)
    return max(abs(float(fR[k] - fL[k] - s * (uR[k] - uL[k]))) for k in keys) / scale


def make_case(case):
    """``(cfg, info)`` for a case name: b5, sst, sst0, or slowA:<s> / slowB:<s>."""
    import make_slow_shock_tube as M
    if case in TUBE_CONFIGS:
        # the committed configs, read as committed (gamma 1.6666666, as in September)
        return yaml.safe_load(open(ROOT / TUBE_CONFIGS[case])), dict(kind="tube")
    if case == "sst0":
        d = M.build(_build_args(**TUBE_BUILD["sst0"]))
        L, R = _prim8(d["L"], d["Bn"]), _prim8(d["R"], d["Bn"])
        # the gamma make_slow_shock_tube.py writes into its YAML, as the
        # September run of this variant had it
        return _base_cfg(L, R, 1.6666666, 0.4), dict(kind="tube")
    if case.split(":")[0] in SHOCK_FROM:
        name, s = case.split(":")
        s = float(s)
        d = M.build(_build_args(**TUBE_BUILD[SHOCK_FROM[name]]))
        zs, sp, Bn = d["zones"], d["speeds"], d["Bn"]
        up, dn = _prim8(zs[1], Bn), _prim8(zs[2], Bn)      # either side of the left slow shock
        VS = float(sp[2])                                  # its lab-frame speed
        beta = (VS - s) / (1.0 - VS * s)                   # the frame in which it moves at s
        L, R = boost_x(up, beta), boost_x(dn, beta)
        res = jump_residual(L, R, s, G53)
        if res > 1e-10:
            raise RuntimeError("%s: jump conditions violated, %.1e" % (case, res))
        # exactly 5/3: the states were built at 5/3, and a stationary shock
        # stays stationary only if the run's EOS is the one it was built with
        cfg = _base_cfg(L, R, G53, 2.0)
        return cfg, dict(kind="shock", s=s, x0=0.5, beta=beta, rh=res,
                         lab_speed=VS, up=L, dn=R)
    raise ValueError("unknown case %r" % case)


# ── the truth ────────────────────────────────────────────────────────────────

def tube_reference(cfg, x, t, x_fine=None):
    """The exact self-similar solution of a tube at time ``t``: one Riemann
    solve, read along the rays.  Tries the tube checkpoint's bare seed first
    (it converges on these tubes, docs/tests.tex), then an escalating
    multistart, as shocktube_compare.exact_profile does."""
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
    rng = np.random.default_rng(0)
    res = None
    for ck in ("data/ml_guess_gamma53_v5.pt", "data/ml_guess_gamma53.pt"):
        model, scaler = mg.load(rmhd_paths.resolve(ck))
        for K, mi, sigma in ((1, 60, 0.0), (16, 60, 0.5), (48, 100, 1.0)):
            L1 = [np.full(K, float(c)) for c in l7]
            R1 = [np.full(K, float(c)) for c in r7]
            B1 = np.full(K, float(Bn))
            s0, _ = MB.predict_unk6(model, scaler, np.stack(L1, 1), np.stack(R1, 1), B1)
            seeds = []
            for c in range(6):
                col = np.full(K, s0[c][0])
                col[1:] += sigma * rng.standard_normal(K - 1)
                seeds.append(col)
            r, _ = API.make_solver(g)(L1, R1, B1, seed6=seeds, accuracy=1e-9, max_iter=mi)
            if r["converged"].any():
                res = r
                break
        if res is not None:
            break
    if res is None:
        raise RuntimeError("the reference Riemann solve did not converge")
    j = int(np.flatnonzero(res["converged"])[0])
    zones = [[float(z[k][j]) for k in range(7)] for z in res["zones"]]
    VsL = [float(v[j]) for v in res["VsLv"]]
    VsR = [float(v[j]) for v in res["VsRv"]]
    cd = 0.5 * (zones[2][2] + zones[3][2])
    speeds = np.array([VsL[0], VsL[1], VsL[2], cd, VsR[2], VsR[1], VsR[0]])
    sampler = PR.make_sampler(g)
    out = {}
    for tag, xx in (("ref", x), ("fine", x_fine)):
        if xx is None:
            continue
        prim, region, err = PR.profile_at(l7, r7, float(Bn), zones, VsL, VsR, g,
                                          (np.asarray(xx) - x0) / t, sampler)
        for v in VARS:
            out["%s_%s" % (tag, v)] = np.asarray(prim[v], dtype=float)
        out["%s_region" % tag] = np.asarray(region)
        out["%s_err" % tag] = np.asarray(err)
    out["speeds"] = speeds
    return out


def shock_reference(info, x, dx, t):
    """The exact solution of a single shock: the step at x0 + s t, cell-averaged."""
    xs = info["x0"] + info["s"] * t
    frac = np.clip((x + 0.5 * dx - xs) / dx, 0.0, 1.0)     # downstream share of each cell
    out = {}
    for v in VARS:
        out["ref_" + v] = (1.0 - frac) * info["up"][v] + frac * info["dn"][v]
    out["x_shock"] = xs
    return out


# ── one run ──────────────────────────────────────────────────────────────────

def tag_of(case, solver, limiter, n, suffix=""):
    return "%s_%s_%s_%d%s" % (case.replace(":", "_s"), solver, limiter, n, suffix)


def run_one(a):
    import torch
    torch.set_num_threads(1)
    from src.physics.driver import run
    from src.physics.hlld import LAST_DIAG

    out = Path(a.out)
    tag = tag_of(a.case, a.solver, a.limiter, a.ncells, a.suffix)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    cfg, info = make_case(a.case)
    cfg = copy.deepcopy(cfg)
    if a.tend is not None:
        cfg["run"]["t_end"] = a.tend
    t_end = float(cfg["run"]["t_end"])
    cfg["grid"]["ncells"] = a.ncells
    rc = cfg["run"]
    rc.update(solver=a.solver, limiter=a.limiter, tau_weak=a.tau_weak,
              n_retries=a.retries, exact_max_iter=40)
    n_steps = int(math.ceil(t_end / (float(rc["cfl"]) / a.ncells) - 1e-9))
    snap_every = a.snap_every or max(1, n_steps // 20)
    cfg["output"] = dict(dir=str(out / "snaps" / tag), every_n_steps=snap_every)
    if a.solver == "exact":
        cfg["harvest"] = dict(enabled=True, dir=str(out / "harvest" / tag),
                              only_retried=True, max_solved=200_000,
                              max_unsolved=200_000, flush_every=20_000)
    save = dict(case=a.case, solver=a.solver, limiter=a.limiter, ncells=a.ncells,
                t_end=t_end, cfl=float(rc["cfl"]), tau_weak=a.tau_weak,
                retries=a.retries, n_steps=n_steps, kind=info["kind"],
                gamma=float(cfg["eos"]["gamma"]),
                ckpt=os.environ.get("RMHD_ML_CKPT", "(module default)"),
                ckpts=os.environ.get("RMHD_ML_CKPTS", ""), suffix=a.suffix,
                env=" ".join(a.env))
    t0 = time.perf_counter()
    _t, grid, prims, _cons = run(cfg)
    wall = time.perf_counter() - t0
    ng = getattr(grid, "ng", 2)
    x = grid.x[ng:-ng].numpy()
    save["x"] = x
    save["wall"] = wall
    for v in VARS:
        save[v] = prims[v][ng:-ng].numpy()
    d = dict(LAST_DIAG)
    if a.solver == "exact":
        faces = d.get("faces") or {}
        for k, v in faces.items():
            save["cov_" + k] = int(v)
        if "jump_hist" in d:
            save["jump_hist"] = np.asarray(d["jump_hist"], dtype=np.int64)
            save["jump_rows"] = np.array(d.get("jump_rows", []))
    print("== %s  wall %.1f s  steps %d" % (tag, wall, n_steps), flush=True)
    # the truth, after the run, so a failed reference never costs the run
    try:
        if info["kind"] == "tube":
            xf = np.linspace(0.0, 1.0, 4001)[:-1] + 0.5 / 4000 if a.ncells == 400 else None
            ref = tube_reference(cfg, x, t_end, xf)
            if xf is not None:
                save["x_fine"] = xf
        else:
            ref = shock_reference(info, x, 1.0 / a.ncells, t_end)
            for k in ("s", "x0", "beta", "rh", "lab_speed"):
                save[k] = info[k]
            for v in VARS:
                save["up_" + v] = info["up"][v]
                save["dn_" + v] = info["dn"][v]
        save.update(ref)
        for v in VARS:
            save["L1_" + v] = float(np.abs(save[v] - save["ref_" + v]).mean())
    except Exception:                                       # noqa: BLE001
        save["ref_failed"] = traceback.format_exc()
        print(save["ref_failed"], flush=True)
    np.savez_compressed(out / "runs" / (tag + ".npz"), **save)
    summ = {k: (v if isinstance(v, (int, float, str)) else None) for k, v in save.items()
            if k.startswith(("L1_", "cov_")) or k in ("case", "solver", "limiter", "ncells",
                                                      "wall", "n_steps", "t_end")}
    (out / "runs" / (tag + ".json")).write_text(json.dumps(summ, indent=1))
    print("   L1 " + "  ".join("%s %.3e" % (v, save.get("L1_" + v, float("nan"))) for v in VARS),
          flush=True)


# ── task lists ───────────────────────────────────────────────────────────────

def tasks(experiment, out):
    """One `run ...` argument line per task, the most expensive first."""
    T = []
    if experiment in ("tubes", "all"):
        # The exact Balsara 5 stops at 400 cells: its left slow wave sits under
        # a cell from its Alfven wave, the hardest faces of the three tubes, and
        # with the continuation rung 800 cells would take days.
        for case, Ns, Nx, w in (("b5", ALL_N, (100, 200, 400), 6.0),
                                ("sst", ALL_N, ALL_N, 1.0),
                                ("sst0", ALL_N, ALL_N, 1.0)):
            for n in Ns:
                for lim in LIMITERS:
                    for sol in SOLVERS:
                        if sol == "exact" and n not in Nx:
                            continue
                        cost = w * n * n * (300.0 if sol == "exact" else 1.0)
                        T.append((cost, "run --case %s --solver %s --limiter %s --ncells %d --out %s"
                                  % (case, sol, lim, n, out)))
                # the exact arm as the 2D production runs it (rotor networks, no
                # continuation): how much the coverage moves the answer
                if n == 400:
                    for lim in LIMITERS:
                        T.append((w * n * n * 300.0,
                                  "run --case %s --solver exact --limiter %s --ncells %d --suffix _prod "
                                  "--env RMHD_ML_CKPT=data/ml_guess_rotor_ft.pt --env RMHD_CONTINUATION=0 "
                                  "--out %s" % (case, lim, n, out)))
    if experiment in ("standing", "all"):
        for name in SHOCK_FROM:
            for s in ("0", "0.01", "0.05"):
                for lim in LIMITERS:
                    for sol in SOLVERS:
                        cost = 200.0 * 200.0 * (30.0 if sol == "exact" else 1.0)
                        T.append((cost, "run --case %s:%s --solver %s --limiter %s --ncells 200 "
                                  "--snap-every 10 --out %s" % (name, s, sol, lim, out)))
    if experiment == "pilot":
        for line in ("--case sst --solver exact --limiter mc --ncells 100 --tend 0.1",
                     "--case b5 --solver exact --limiter pcm --ncells 100 --tend 0.1",
                     "--case sst0 --solver exact --limiter mc --ncells 100 --tend 0.1",
                     "--case slowA:0 --solver exact --limiter mc --ncells 200 --tend 0.1 --snap-every 10",
                     "--case slowB:0.05 --solver exact --limiter pcm --ncells 200 --tend 0.1 --snap-every 10",
                     "--case slowA:0 --solver hlld --limiter mc --ncells 200 --tend 0.4 --snap-every 10",
                     "--case b5 --solver hlld --limiter pcm --ncells 400",
                     "--case sst --solver hlld --limiter pcm --ncells 400",
                     "--case sst0 --solver hlle --limiter mc --ncells 400"):
            T.append((0.0, "run %s --out %s" % (line, out)))
    if experiment == "levers":
        V = (("_v5", "--env RMHD_ML_CKPT=data/ml_guess_gamma53_v5.pt"),
             ("_cont", "--env RMHD_CONTINUATION=1"),
             ("_v5cont", "--env RMHD_ML_CKPT=data/ml_guess_gamma53_v5.pt --env RMHD_CONTINUATION=1"))
        for line in ("--case sst --solver exact --limiter mc --ncells 100 --tend 0.1",
                     "--case sst0 --solver exact --limiter mc --ncells 100 --tend 0.1",
                     "--case slowB:0.05 --solver exact --limiter pcm --ncells 200 --tend 0.1 --snap-every 10",
                     "--case b5 --solver exact --limiter pcm --ncells 100 --tend 0.1"):
            for suf, env in V:
                T.append((0.0, "run %s --suffix %s %s --out %s" % (line, suf, env, out)))
    T.sort(key=lambda c: -c[0])
    return [line for _, line in T]


# ── analysis ─────────────────────────────────────────────────────────────────

def load_runs(out):
    R = []
    for f in sorted(glob.glob(os.path.join(out, "runs", "*.npz"))):
        d = dict(np.load(f, allow_pickle=True))
        d = {k: (v.item() if isinstance(v, np.ndarray) and v.shape == () else v)
             for k, v in d.items()}
        R.append(d)
    return R


def coverage_line(d):
    f = {k[4:]: d[k] for k in d if k.startswith("cov_")}
    if not f:
        return ""
    fa = max(f.get("faces", 0), 1)
    at = max(f.get("attempted", 0), 1)
    return ("faces %d: weak %.1f%% upwind %.1f%% attempted %.1f%% -> exact %.1f%% "
            "(seven %d planar5 %d planar4 %d degenerate %d) linear-fb %d HLLD %d"
            % (f.get("faces", 0), 100 * f.get("weak", 0) / fa, 100 * f.get("upwind", 0) / fa,
               100 * f.get("attempted", 0) / fa, 100 * f.get("exact", 0) / at,
               f.get("seven", 0), f.get("planar5", 0), f.get("planar4", 0),
               f.get("degenerate", 0), f.get("linear_fallback", 0),
               f.get("hlld_after_attempt", 0)))


def analyse(a):
    R = load_runs(a.out)
    print("%d runs in %s" % (len(R), a.out))
    by = {}
    for d in R:
        # a variant (--suffix, e.g. the exact arm with production settings)
        # is its own "solver": exact_prod
        by[(d["case"], d["solver"] + str(d.get("suffix") or ""), d["limiter"],
            int(d["ncells"]))] = d
    cases = sorted({k[0] for k in by})
    for case in cases:
        Ns = sorted({k[3] for k in by if k[0] == case})
        print("\n== %s" % case)
        for lim in LIMITERS:
            for n in Ns:
                row = "  %-3s %5d " % (lim, n)
                for sol in SOLVERS:
                    d = by.get((case, sol, lim, n))
                    row += " %-5s %s" % (sol, ("%.3e" % d["L1_rho"]) if d is not None and "L1_rho" in d
                                         else "   -     ")
                dh, de = by.get((case, "hlld", lim, n)), by.get((case, "exact", lim, n))
                if dh is not None and de is not None and "L1_rho" in dh and "L1_rho" in de:
                    row += "   exact/HLLD rho %.3f p %.3f vx %.3f By %.3f   wall x%.0f" % (
                        de["L1_rho"] / dh["L1_rho"], de["L1_p"] / dh["L1_p"],
                        de["L1_vx"] / dh["L1_vx"], de["L1_By"] / max(dh["L1_By"], 1e-300),
                        de["wall"] / max(dh["wall"], 1e-9))
                print(row)
                if de is not None:
                    print("            " + coverage_line(de))
                dp = by.get((case, "exact_prod", lim, n))
                if dp is not None and "L1_rho" in dp and dh is not None and "L1_rho" in dh:
                    print("            production settings: exact/HLLD rho %.3f p %.3f   %s"
                          % (dp["L1_rho"] / dh["L1_rho"], dp["L1_p"] / dh["L1_p"], coverage_line(dp)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--case", required=True)
    r.add_argument("--solver", required=True, choices=SOLVERS)
    r.add_argument("--limiter", required=True, choices=("pcm", "minmod", "mc"))
    r.add_argument("--ncells", type=int, required=True)
    r.add_argument("--tend", type=float, default=None)
    r.add_argument("--tau-weak", type=float, default=1e-2,
                   help="weak-jump gate of the exact arm (production 1e-2)")
    r.add_argument("--retries", type=int, default=2)
    r.add_argument("--snap-every", type=int, default=0,
                   help="steps between snapshots (0: ~20 per run)")
    r.add_argument("--out", required=True)
    r.add_argument("--suffix", default="",
                   help="appended to the run's tag: a variant of the same run")
    r.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="set before the solver modules are imported (they read "
                        "their switches at import), e.g. RMHD_CONTINUATION=1")
    li = sub.add_parser("list")
    li.add_argument("experiment", choices=("tubes", "standing", "all", "pilot", "levers"))
    li.add_argument("--out", required=True)
    an = sub.add_parser("analyse")
    an.add_argument("out")
    sh = sub.add_parser("show")
    sh.add_argument("case")
    a = ap.parse_args()

    if a.cmd == "list":
        print("\n".join(tasks(a.experiment, a.out)))
    elif a.cmd == "analyse":
        analyse(a)
    elif a.cmd == "show":
        cfg, info = make_case(a.case)
        print(json.dumps(dict(L=cfg["run"]["primL"], R=cfg["run"]["primR"],
                              gamma=cfg["eos"]["gamma"], t_end=cfg["run"]["t_end"],
                              **{k: v for k, v in info.items() if k not in ("up", "dn")}),
                         indent=1, default=float))
    else:
        for kv in a.env:
            k, v = kv.split("=", 1)
            os.environ[k] = v
        tag = tag_of(a.case, a.solver, a.limiter, a.ncells, a.suffix)
        logf = Path(a.out) / "logs" / (tag + ".log")
        logf.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print("start %s  %s" % (tag, time.strftime("%H:%M:%S")), flush=True)
        with open(logf, "w", buffering=1) as fh, \
                contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
            try:
                run_one(a)
                ok = True
            except Exception:                               # noqa: BLE001
                traceback.print_exc()
                ok = False
        print("%s %s  %.0f s" % ("done  " if ok else "FAILED", tag, time.time() - t0), flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
