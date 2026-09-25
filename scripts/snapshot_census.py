"""Every attempted interface of the rotor, at known positions: solved today,
and tube-tested.  (science plan F, step 1b)

The harvest's unsolved rows carry no grid position, so a tube label cannot
be put back on the grid.  This builds the population from the SNAPSHOTS
instead, which store the cell-centred primitives at t = 0, 0.05, ..., 0.4:

1. **Faces as the code makes them.**  Each snapshot is ghost-padded
   (outflow, ng = 2, as the rotor runs) and reconstructed with the code's own
   ``reconstruct_prims`` (PLM, MC limiter, W v reconstructed), in both
   directions.  The normal field is single-valued at a face, as constrained
   transport makes it: here the average of the two adjacent cells, which is
   what the cell-centred snapshot allows.
2. **The attempted set, as production decides it.**  Bad states,
   ``relative_jump < 1e-2`` (the rotor runs' ``--tau-weak``) and HLLD's
   upwind skip are applied with ``exact_flux``'s own functions; what remains
   is what the exact solver would have been handed.
3. **Two labels per interface.**  Solvable today: the production path
   (``coplanar_limit.production_retest``: ML seed, ensemble retries, order
   gate) or a verified planar answer.  Compound: the tube test
   (``tube_features.classify_tubes``) finds a field reversal fused to a
   density or magnitude jump (``fkind == 2``).

Every interface keeps its time, direction and face indices, so both labels
can be drawn on the grid and the classifier (``compound_classifier.py``)
can be cross-validated by time.  Faces: x-faces (fi, j) with fi in [0, nx]
between cells fi-1 and fi; y-faces (i, fj) likewise.

    python scripts/snapshot_census.py results/rotor_64_exact --shard 0 \
        --nshards 64 --out results/snapshot_census/shard_000.npz
    python scripts/snapshot_census.py --summarise results/snapshot_census
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import warnings

import numpy as np
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "scripts"))
import coplanar_limit as CLM                                       # noqa: E402
import tube_features as TF                                         # noqa: E402
from rmhd.batched import planar5_b as P5                           # noqa: E402
from src.physics import exact_flux as EF                           # noqa: E402
from src.physics.eos import hybrid_eos                             # noqa: E402
from src.physics.hlld import compute_srmhd_fluxes                  # noqa: E402
from src.physics.reconstruction import reconstruct_prims           # noqa: E402

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
torch.set_num_threads(1)

GAMMA = 5.0 / 3.0
TAU_WEAK = 1e-2
_BKEY = ("Bx", "By")


def load_snaps(rundir, times=None):
    out = []
    for f in sorted(glob.glob(os.path.join(rundir, "snap_*.npz"))):
        if f.endswith("snap_fin.npz"):
            continue
        z = np.load(f)
        t = float(z["t"])
        if times is None or any(abs(t - x) < 0.01 for x in times):
            out.append({k: z[k] for k in z.files})
    out.sort(key=lambda s: float(s["t"]))
    return out


def run_scheme(rundir, override=None):
    """(limiter, ng) of the run whose snapshots these are.

    The census rebuilds the faces itself, so it has to rebuild them with the
    SAME reconstruction the run used -- analysing an mp5 run with PLM faces
    changes the attempted set, the gates and every label, silently.  Read from
    `run_meta.json`, else from the snapshot's own stamp, else PLM with a
    warning (runs made before either existed).
    """
    from src.physics.reconstruction import ghosts_needed
    if override:
        return override, ghosts_needed(override)
    meta = os.path.join(rundir, "run_meta.json")
    if os.path.exists(meta):
        lim = json.load(open(meta)).get("limiter", "mc")
        return lim, ghosts_needed(lim)
    snaps = sorted(glob.glob(os.path.join(rundir, "snap_*.npz")))
    if snaps:
        z = np.load(snaps[0])
        if "limiter" in z.files:
            lim = str(z["limiter"])
            return lim, ghosts_needed(lim)
    print("  ! %s carries no limiter stamp; assuming mc (ng 2)" % rundir)
    return "mc", 2


def faces(snap, eos, limiter="mc", ng=2):
    """Attempted interfaces of one snapshot, both directions.

    Returns a dict of arrays: UL, UR (N, 8; solver frame, B_n in column 7),
    idir, fa, fb (face indices), and the counts the gates removed."""
    nx, ny = snap["rho"].shape
    t = lambda a: torch.as_tensor(np.pad(np.asarray(a, float), ng, mode="edge"),
                                  dtype=torch.float64)
    zero = np.zeros((nx, ny))
    prims = dict(rho=t(snap["rho"]), p=t(snap["p"]), vx=t(snap["vx"]),
                 vy=t(snap["vy"]), vz=t(zero), Bx=t(snap["Bx"]), By=t(snap["By"]),
                 Bz=t(zero))
    prims["eps"] = eos.eps__press_rho(prims["p"], prims["rho"])
    out = {k: [] for k in ("UL", "UR", "idir", "fa", "fb")}
    gates = dict(bad=0, weak=0, upwind=0, total=0)
    for axis in (0, 1):
        L, R = reconstruct_prims(prims, axis=axis, eos=eos, limiter=limiter,
                                 vel_var="Wv")
        B = prims[_BKEY[axis]]
        ext = torch.cat([B.narrow(axis, 0, 1), B, B.narrow(axis, B.shape[axis] - 1, 1)],
                        dim=axis)
        n_f = ext.shape[axis] - 1
        Bnf = 0.5 * (ext.narrow(axis, 0, n_f) + ext.narrow(axis, 1, n_f))
        L[_BKEY[axis]] = Bnf
        R[_BKEY[axis]] = Bnf
        phys = ((slice(ng, ng + nx + 1), slice(ng, ng + ny)) if axis == 0 else
                (slice(ng, ng + nx), slice(ng, ng + ny + 1)))
        shp = L["rho"][phys].shape
        sL = {k: v[phys].reshape(-1) for k, v in L.items()}
        sR = {k: v[phys].reshape(-1) for k, v in R.items()}
        v2L = sL["vx"] ** 2 + sL["vy"] ** 2 + sL["vz"] ** 2
        v2R = sR["vx"] ** 2 + sR["vy"] ** 2 + sR["vz"] ** 2
        bad = (v2L >= 1) | (v2R >= 1) | (sL["rho"] <= 0) | (sR["rho"] <= 0)
        weak = (~bad) & (EF.relative_jump(sL, sR) < TAU_WEAK)
        *_, cmax, cmin = compute_srmhd_fluxes(sL, sR, eos, axis)
        upwind = (~bad) & (~weak) & ((cmin <= 0.0) | (cmax <= 0.0))
        att = ((~bad) & (~weak) & (~upwind)).numpy()
        gates["bad"] += int(bad.sum()); gates["weak"] += int(weak.sum())
        gates["upwind"] += int(upwind.sum()); gates["total"] += int(bad.numel())
        (L7, BnL), (R7, BnR) = EF.to_solver_frame(sL, eos, axis), EF.to_solver_frame(sR, eos, axis)
        UL = np.column_stack([c.numpy() for c in L7] + [BnL.numpy()])
        UR = np.column_stack([c.numpy() for c in R7] + [BnR.numpy()])
        fa, fb = np.meshgrid(np.arange(shp[0]), np.arange(shp[1]), indexing="ij")
        out["UL"].append(UL[att]); out["UR"].append(UR[att])
        out["idir"].append(np.full(int(att.sum()), axis, np.int8))
        out["fa"].append(fa.reshape(-1)[att]); out["fb"].append(fb.reshape(-1)[att])
    return {k: np.concatenate(v) for k, v in out.items()}, gates


def population(rundir, times, limiter="mc", ng=2):
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    parts, g_all = [], []
    for s in load_snaps(rundir, times):
        f, g = faces(s, eos, limiter, ng)
        f["t"] = np.full(f["idir"].size, float(s["t"]))
        parts.append(f)
        g_all.append((float(s["t"]), g))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, g_all


def run(a):
    t0 = time.time()
    tag = "[%s shard %d/%d]" % (socket.gethostname(), a.shard, a.nshards)
    log = lambda s: print("%s %6.0fs  %s" % (tag, time.time() - t0, s), flush=True)
    times = [float(x) for x in a.times.split(",")] if a.times else None
    limiter, ng = run_scheme(a.run, a.limiter)
    log("reconstruction: limiter %s, ng %d" % (limiter, ng))
    pop, gates = population(a.run, times, limiter, ng)
    if a.shard == 0:
        for t, g in gates:
            log("t = %.3f: %d faces; gated out bad %d, weak %d, upwind %d"
                % (t, g["total"], g["bad"], g["weak"], g["upwind"]))
    sel = np.arange(pop["idir"].size)[a.shard::a.nshards]
    if a.limit:
        sel = sel[:a.limit]
    pop = {k: v[sel] for k, v in pop.items()}
    n = sel.size
    log("%d attempted interfaces in this shard" % n)
    if n == 0:
        raise SystemExit("empty shard")
    UL, UR = pop["UL"], pop["UR"]
    Bn = UL[:, 7].copy()

    models = CLM.load_models(a.ckpt, a.extra_ckpts)
    prod_ok, prod_cls, prod_att = CLM.production_retest(UL, UR, models, a.max_iter)
    S5 = P5.make_solver(GAMMA)
    L0, R0, _, _ = P5.to_planar([UL[:, j].copy() for j in range(7)],
                                [UR[:, j].copy() for j in range(7)], Bn)
    ref = S5["solve"](L0, R0, Bn, accuracy=1e-10, max_iter=a.max_iter)
    solvable = prod_ok | ref["converged"]
    log("solvable today: %d/%d (seven-wave %d, planar %d)"
        % (solvable.sum(), n, prod_ok.sum(), ref["converged"].sum()))

    tf, _ = TF.classify_tubes(UL[:, :7], UR[:, :7], Bn, a.ncells, a.tend, log=log)
    compound = (tf["fkind"] == 2).any(axis=1)
    btzero = ((tf["fbtmin"] < 0.1) & (tf["fkind"] >= 0)).any(axis=1)
    log("compound: %d/%d; among unsolvable %d/%d"
        % (compound.sum(), n, (compound & ~solvable).sum(), (~solvable).sum()))

    np.savez_compressed(
        a.out, t=pop["t"], idir=pop["idir"], fa=pop["fa"], fb=pop["fb"], UL=UL, UR=UR,
        Bn=Bn, prod_ok=prod_ok, ref_ok=ref["converged"], prod_cls=prod_cls,
        prod_att=prod_att, solvable=solvable, compound=compound, btzero=btzero,
        **{k: tf[k] for k in ("nfeat", "fkind", "fdpsi", "fdmag", "fbtmin", "fbtA",
                              "fbtB", "fspeed", "ffam")},
        gates=np.array([[t, g["total"], g["bad"], g["weak"], g["upwind"]]
                        for t, g in gates]),
        ncells=a.ncells, tend=a.tend, wall=time.time() - t0,
        host=socket.gethostname(),
        hlld_rev=subprocess.run(["git", "-C", _HERE, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip(),
        script_md5=hashlib.md5(open(__file__, "rb").read()).hexdigest(),
        argv=" ".join(sys.argv))
    log("wrote %s" % a.out)


def summarise(d):
    Z = [np.load(f) for f in sorted(glob.glob(os.path.join(d, "shard_*.npz")))]
    if not Z:
        raise SystemExit("no shards in %s" % d)
    c = lambda k: np.concatenate([z[k] for z in Z])
    t, idir, sv, cp = c("t"), c("idir"), c("solvable"), c("compound")
    print("snapshot census: %d shards, %d attempted interfaces, %d times"
          % (len(Z), t.size, np.unique(t).size))
    print("\n  %-6s %-3s %7s %10s %10s %16s %16s" % (
        "t", "dir", "n", "unsolvable", "compound", "P(cmp | unsolv.)",
        "P(unsolv. | cmp)"))
    for tt in np.unique(t):
        for d, lab in ((0, "x"), (1, "y")):
            m = (t == tt) & (idir == d)
            if not m.any():
                continue
            u = ~sv[m]
            k = cp[m]
            print("  %-6.3f %-3s %7d %9.1f%% %9.1f%% %15.1f%% %15.1f%%" % (
                tt, lab, m.sum(), 100 * u.mean(), 100 * k.mean(),
                100 * k[u].mean() if u.any() else np.nan,
                100 * u[k].mean() if k.any() else np.nan))
    u, k = ~sv, cp
    print("\n  all: unsolvable %.1f%%, compound %.1f%%; P(compound | unsolvable) "
          "%.1f%%, P(unsolvable | compound) %.1f%%" % (
              100 * u.mean(), 100 * k.mean(), 100 * k[u].mean(), 100 * u[k].mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", nargs="?", help="rotor run directory with snap_*.npz")
    ap.add_argument("--times", help="comma-separated snapshot times (default all)")
    ap.add_argument("--limiter", help="override the run's own reconstruction "
                                      "(default: read it from the run)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="at most this many lanes "
                    "per shard (a check run)")
    ap.add_argument("--ncells", type=int, default=256)
    ap.add_argument("--tend", type=float, default=0.25)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--ckpt", default=os.environ.get("RMHD_ML_CKPT",
                                                     CLM.DEFAULT_CKPT))
    ap.add_argument("--extra-ckpts", default=os.environ.get("RMHD_ML_CKPTS",
                                                            CLM.DEFAULT_EXTRA))
    ap.add_argument("--out")
    ap.add_argument("--summarise", metavar="DIR")
    a = ap.parse_args()
    if a.summarise:
        summarise(a.summarise)
        return
    if not (a.run and a.out):
        ap.error("need <run> --out (or --summarise DIR)")
    run(a)


if __name__ == "__main__":
    main()
