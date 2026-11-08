#!/usr/bin/env python
"""Paired replay of held-out late-rung sweeps, with and without the planar
network (RMHD_COPLANAR7_NET, set or not by the caller).

Population: the harvest runs' late-rung faces in the TEST block (t/t_end >=
--from, default 0.8; the network saw none of them in training or
validation). These are the late rung's answers (source 2) and the
four-unknown solver's (source 3): every face the late rung was offered that
someone answered. Faces nobody answered left no row and are not in it. A
shard's rows are split into sweeps with its coverage counts: per sweep, the
next `coplanar7` source-2 rows and the next `planar4` source-3 rows, in order;
the counts match in every shard of the three runs. So each call holds one
sweep's late-rung faces, as the run had them together.

Each sweep goes through the run's call (exact_flux_batched: tau_weak 1e-2,
tau_bt 1e-9, 2 retries, 40 iterations), with the chain the caller's
environment sets and no pool. Per sweep: the wall, each rung's wall, and the
late rung's planar4 solves and (lane, branch) pairs (coplanar7_b.STATS). Per
face: exact, the rung that answered (1 seven-wave/other, 2 five-wave, 3 late
rung, 4 four-unknown, 5 three-wave, 6 continuation, 0 none), the late rung's
stage, and the flux.

    python replay_net.py OUT.npz RUN [RUN ...] [--from 0.8] [--max-sweeps N]
                         [--select RUN] [--part K/N]

--select / --part replay one run's sweeps, or the K-th of N contiguous blocks
of them, so that one arm can run as parallel parts; the run index stays the
index in the full RUN list, so the parts concatenate.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HLLD = os.environ["HLLD_DIR"]
sys.path.insert(0, HLLD)
from rmhd.eos import set_eos                                     # noqa: E402
set_eos("ideal")
from rmhd.batched import coplanar7_b as CP7                      # noqa: E402
from src.physics import exact_flux as EF                         # noqa: E402
from src.physics import exact_pool as XP                         # noqa: E402
from src.physics.eos import hybrid_eos                           # noqa: E402
from src.physics.harvest import Harvester                        # noqa: E402
from src.physics.hlld import LAST_DIAG                           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planar_data as PD                                         # noqa: E402

WALL = {}
_map_lanes, _pipeline = XP.map_lanes, EF._run_pipeline


def map_lanes(name, *a, **k):
    t0 = time.perf_counter()
    try:
        return _map_lanes(name, *a, **k)
    finally:
        WALL[name] = WALL.get(name, 0.0) + time.perf_counter() - t0


def run_pipeline(*a, **k):
    t0 = time.perf_counter()
    try:
        return _pipeline(*a, **k)
    finally:
        WALL["_run_pipeline"] = WALL.get("_run_pipeline", 0.0) + time.perf_counter() - t0


XP.map_lanes, EF._run_pipeline = map_lanes, run_pipeline
RUNGS = ("_run_pipeline", "_planar_and_ray", "_planar_ladder", "_coplanar7_and_ray", "_planar4_and_ray")


def sweeps_of(run, t_from):
    """The run's test-block sweeps: their late-rung faces (module doc)."""
    t_of, t_end = PD.run_times(run)
    H = os.path.join(PD.R, run, "harvest")
    out = []
    for f in sorted(glob.glob(os.path.join(H, "coverage_r*_*.npz"))):
        tag = os.path.basename(f)[len("coverage_"):-len(".npz")]
        z = np.load(f)
        sw = z["sweep"]
        t = np.array([t_of.get(int(s) // PD.PER + 1, np.nan) for s in sw])
        if not (np.nan_to_num(t) / t_end >= t_from).any():
            continue
        cnt = lambda key: np.array([int(np.unpackbits(z[key][j])[:int(z["n"][j])].sum())
                                    for j in range(len(sw))])
        n7, n4 = cnt("coplanar7"), cnt("planar4")
        s = np.load(os.path.join(H, "solved_" + tag + ".npz"))
        src = s["source"]
        UL, UR = s["U_L"], s["U_R"]
        gamma = float(np.asarray(s["gamma"]).reshape(-1)[0])
        i2, i3 = np.flatnonzero(src == 2), np.flatnonzero(src == 3)
        if i2.size != n7.sum() or i3.size != n4.sum():
            raise RuntimeError("%s %s: %d/%d source-2 rows, %d/%d source-3 rows" % (
                run, tag, i2.size, n7.sum(), i3.size, n4.sum()))
        o2 = np.concatenate([[0], np.cumsum(n7)])
        o3 = np.concatenate([[0], np.cumsum(n4)])
        for j in range(len(sw)):
            if not t[j] / t_end >= t_from:
                continue
            rows = np.concatenate([i2[o2[j]:o2[j + 1]], i3[o3[j]:o3[j + 1]]])
            if rows.size:
                out.append(dict(run=run, sweep=int(sw[j]), t=float(t[j]), gamma=gamma,
                                UL=UL[rows], UR=UR[rows], src=src[rows]))
    return out


_EOS = {}


def lab(U, eos):
    rho, Pt, vx, vy, vz, By, Bz, Bx = (U[:, j] for j in range(8))
    W2 = 1.0 / (1.0 - (vx * vx + vy * vy + vz * vz))
    eta = Bx * vx + By * vy + Bz * vz
    b2 = (Bx * Bx + By * By + Bz * Bz) / W2 + eta * eta
    t = lambda v: torch.tensor(np.ascontiguousarray(v), dtype=torch.float64)
    d = {"rho": t(rho), "vx": t(vx), "vy": t(vy), "vz": t(vz),
         "p": t(Pt - 0.5 * b2), "Bx": t(Bx), "By": t(By), "Bz": t(Bz)}
    d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--from", dest="frm", type=float, default=0.8)
    ap.add_argument("--max-sweeps", type=int, default=0)
    ap.add_argument("--select", default="", help="replay only this run's sweeps")
    ap.add_argument("--part", default="", help="K/N: the K-th of N contiguous blocks of the sweeps")
    ap.add_argument("--harvest", default="", help="harvest every answer into this directory "
                    "(one gamma: use with --select; RMHD_HARVEST_RESCUES=1 for the rescue columns)")
    a = ap.parse_args()
    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("RMHD_")}
    print("env:", env, flush=True)
    print("network: %r, fallback %s" % (CP7.NET, CP7.NET_FALLBACK), flush=True)
    sweeps = []
    for run in a.runs:
        if a.select and run != a.select:
            continue
        s = sweeps_of(run, a.frm)
        print("%-34s %4d sweeps, %6d faces (late rung %d, four-unknown %d)" % (
            run, len(s), sum(len(x["src"]) for x in s), sum(int((x["src"] == 2).sum()) for x in s),
            sum(int((x["src"] == 3).sum()) for x in s)), flush=True)
        sweeps += s
    if a.part:
        kp, np_ = (int(x) for x in a.part.split("/"))
        edges = np.linspace(0, len(sweeps), np_ + 1).round().astype(int)
        sweeps = sweeps[edges[kp]:edges[kp + 1]]
        print("part %d/%d: sweeps %d-%d" % (kp, np_, edges[kp], edges[kp + 1] - 1), flush=True)
    if a.max_sweeps:
        sweeps = sweeps[:a.max_sweeps]
    H = None
    if a.harvest:
        gs = sorted({sw["gamma"] for sw in sweeps})
        if len(gs) != 1:
            raise SystemExit("--harvest needs one gamma (use --select), got %s" % gs)
        H = Harvester(a.harvest, gs[0], only_retried=False, max_solved=10 ** 9,
                      max_unsolved=10 ** 9, flush_every=200000)
    face = {k: [] for k in ("run", "sweep", "src", "exact", "rung", "stage", "flux")}
    per = {k: [] for k in ("run", "sweep", "faces", "wall", "solves", "branch_lanes") + RUNGS}
    t_all = time.time()
    for k, sw in enumerate(sweeps):
        g = sw["gamma"]
        if g not in _EOS:
            _EOS[g] = hybrid_eos(K=0.0, gamma=g, gamma_th=g)
        eos = _EOS[g]
        WALL.clear()
        st0 = dict(CP7.STATS)
        t0 = time.perf_counter()
        F, U, _ = EF.exact_flux_batched(lab(sw["UL"], eos), lab(sw["UR"], eos), eos, idir=0,
                                        tau_weak=1e-2, tau_bt=1e-9, n_retries=2, max_iter=40,
                                        harvester=H)
        wall = time.perf_counter() - t0
        if H is not None:
            # the harvester's per-call coverage map is made for a run, whose
            # sweeps all have the grid's length; a replay's calls differ in
            # length, so the maps would not stack at the flush (11-08: two
            # parts of 84611 died there and lost their results) -- and mean
            # nothing here.  The answer rows are kept.
            H._coverage.clear()
        n = len(sw["src"])
        d = LAST_DIAG
        ex = d["exact_mask"]
        exact = np.asarray(ex.numpy() if hasattr(ex, "numpy") else ex, bool)
        sel = np.asarray(d["sel"], int)
        rung = np.zeros(n, np.int8)
        for key, code in (("planar5_mask", 2), ("coplanar7_late_mask", 3), ("planar4_mask", 4),
                          ("three_wave_mask", 5), ("continuation_mask", 6)):
            m = np.asarray(d.get(key, np.zeros(sel.size, bool)), bool)
            if m.size != sel.size:
                raise RuntimeError("%s has %d entries for %d attempted lanes" % (key, m.size, sel.size))
            rung[sel[m]] = code
        rung[(rung == 0) & exact] = 1
        rung[~exact] = 0
        stage = np.zeros(n, np.int8)
        st = np.asarray(d.get("coplanar7_late_stage", np.zeros(sel.size, np.int8)))
        stage[sel] = st
        keys = sorted(F)
        flux = np.stack([np.asarray(F[q].reshape(-1)) for q in keys], axis=1)
        for key, v in (("run", np.full(n, a.runs.index(sw["run"]), np.int8)),
                       ("sweep", np.full(n, sw["sweep"], np.int32)), ("src", sw["src"]),
                       ("exact", exact), ("rung", rung), ("stage", stage), ("flux", flux)):
            face[key].append(v)
        for key, v in (("run", a.runs.index(sw["run"])), ("sweep", sw["sweep"]), ("faces", n),
                       ("wall", wall), ("solves", CP7.STATS["solves"] - st0["solves"]),
                       ("branch_lanes", CP7.STATS["branch_lanes"] - st0["branch_lanes"])):
            per[key].append(v)
        for r in RUNGS:
            per[r].append(WALL.get(r, 0.0))
        if (k + 1) % 50 == 0 or k + 1 == len(sweeps):
            fr = np.concatenate(face["rung"])
            print("sweep %d/%d: %d faces, exact %d, late rung %d, four-unknown %d, lost %d; "
                  "late-rung solves %d; %.0f s" % (
                      k + 1, len(sweeps), fr.size, int((fr > 0).sum()), int((fr == 3).sum()),
                      int((fr == 4).sum()), int((fr == 0).sum()), int(np.sum(per["solves"])),
                      time.time() - t_all), flush=True)
    # the results first: a failing harvest must not take them along
    np.savez_compressed(a.out, runs=np.array(a.runs), flux_keys=np.array(keys),
                        env=json.dumps(env), net=CP7.NET, fallback=CP7.NET_FALLBACK,
                        regular=getattr(CP7, "REGULAR", False), cp7_stats=json.dumps(CP7.STATS),
                        **{"face_" + k: np.concatenate(v) for k, v in face.items()},
                        **{"sweep_" + k: np.asarray(v) for k, v in per.items()})
    print("saved %s (%.0f s)" % (a.out, time.time() - t_all), flush=True)
    if H is not None:
        s = H.close()
        print("harvest: %d solved, %d unsolved rows -> %s" % (
            s["harvest_solved"], s["harvest_unsolved"], s["harvest_dir"]), flush=True)


if __name__ == "__main__":
    main()
