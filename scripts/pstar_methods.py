"""Secant or Newton-Raphson for the HLLD star pressure?

`hlld_flux` finds the star-region total pressure p* by a root-find on
Mignone's pressure-balance residual.  Production uses a clamped SECANT
(`safe_secant_bisection` -- the name is historical, it does no bisection).
This measures the alternative, Newton-Raphson (`newton_pstar`), with the
derivative taken two ways: exactly by autograd ("newton"), and by central
difference ("newton_fd").

The prior, stated before measuring: secant converges with order ~1.62 using
ONE residual evaluation per step; Newton converges with order 2 but needs
f'(p) as well.  Per unit of work Newton only wins if the derivative is much
cheaper than a residual evaluation.  A central difference costs two
evaluations, so "newton_fd" should lose; autograd costs one backward pass,
typically 1-3 forward passes, so "newton" is the real contest.

And the risk that is not about cost: the HLLD residual is multi-rooted (seven
sign changes at the ST1 discontinuity), so two root-finders started from the
same bracket can land on DIFFERENT roots.  A different root is a different
flux, and the wave-ordering check then decides whether it survives or the
interface falls back to HLLE.  So the comparison reports, per method:

* converged fraction and HLLE-fallback fraction;
* iterations, residual evaluations and backward passes;
* wall time for the whole flux call (best of several);
* against the secant: how far p* moved, how many interfaces landed on a
  different root (relative change > 1e-6 where both converged and neither
  fell back), and the largest flux change.

The interface states are the production ones: each rotor snapshot is
reconstructed exactly as the sweep does it (`reconstruct_prims`, the run's
own limiter, W v reconstruction), with the normal field made single-valued at
the face.

    python scripts/pstar_methods.py results/rotor_128_hlld_mc --times 0.1,0.25,0.4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.physics import hlld as H                                  # noqa: E402
from src.physics.eos import hybrid_eos                             # noqa: E402
from src.physics.reconstruction import ghosts_needed, reconstruct_prims  # noqa: E402

METHODS = ("secant", "newton", "newton_safe", "newton_fd", "newton_fd1")
GAMMA = 5.0 / 3.0


def load(rundir, times):
    snaps = []
    for f in sorted(glob.glob(os.path.join(rundir, "snap_*.npz"))):
        if f.endswith("snap_fin.npz"):
            continue
        z = np.load(f)
        t = float(z["t"])
        if times and not any(abs(t - w) < 5e-3 for w in times):
            continue
        snaps.append({k: z[k] for k in z.files})
    meta = os.path.join(rundir, "run_meta.json")
    lim = json.load(open(meta)).get("limiter", "mc") if os.path.exists(meta) else "mc"
    return snaps, lim


def faces(snap, eos, limiter):
    """Both directions' interface states of one snapshot, flattened."""
    ng = ghosts_needed(limiter)
    pad = lambda a: torch.as_tensor(np.pad(np.asarray(a, float), ng, mode="edge"),
                                    dtype=torch.float64)
    z = np.zeros_like(snap["rho"])
    prims = dict(rho=pad(snap["rho"]), p=pad(snap["p"]), vx=pad(snap["vx"]),
                 vy=pad(snap["vy"]), vz=pad(z), Bx=pad(snap["Bx"]),
                 By=pad(snap["By"]), Bz=pad(z))
    prims["eps"] = eos.eps__press_rho(prims["p"], prims["rho"])
    out = []
    for axis, bkey in ((0, "Bx"), (1, "By")):
        L, R = reconstruct_prims(prims, axis=axis, eos=eos, limiter=limiter,
                                 vel_var="Wv")
        # the normal field is single-valued at a face; the run takes it from
        # the staggered field, here from the average of the two cells
        B = prims[bkey]
        pb = torch.cat([B.narrow(axis, 0, 1), B, B.narrow(axis, -1, 1)], axis)
        Bn = 0.5 * (pb.narrow(axis, 0, pb.shape[axis] - 1)
                    + pb.narrow(axis, 1, pb.shape[axis] - 1))
        L[bkey] = Bn.clone(); R[bkey] = Bn.clone()
        flat = lambda d: {k: v.reshape(-1).contiguous() for k, v in d.items()}
        out.append((axis, flat(L), flat(R)))
    return out


_CALLS = [0]
_orig_call = H.HLLDComputation.__call__


def _counted(self, p):
    _CALLS[0] += 1
    return _orig_call(self, p)


def run(sL, sR, eos, axis, method, repeats):
    """Best-of-`repeats` wall time; the evaluation count from the last call.

    Every batched residual evaluation goes through HLLDComputation.__call__
    (the bracket widening, the secant, and Newton's forward passes), so
    counting there is uniform across methods.  Timing runs uncounted."""
    H._PSTAR_METHOD = method
    best = np.inf
    H.HLLDComputation.__call__ = _orig_call
    for _ in range(repeats):
        t0 = time.perf_counter()
        F, U, ps = H.hlld_flux(sL, sR, eos, idir=axis)
        best = min(best, time.perf_counter() - t0)
    H.HLLDComputation.__call__ = _counted
    _CALLS[0] = 0
    F, U, ps = H.hlld_flux(sL, sR, eos, idir=axis)
    H.HLLDComputation.__call__ = _orig_call
    d = dict(H.LAST_DIAG)
    d["batched_evals"] = _CALLS[0]
    return F, ps, d, best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--times", default="0.1,0.25,0.4")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--methods", help="comma-separated subset (secant first)")
    ap.add_argument("--fd-h", type=float, help="relative finite-difference "
                    "step for newton_fd / newton_fd1")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))
    global METHODS
    if a.methods:
        METHODS = tuple(a.methods.split(","))
    if a.fd_h:
        H._PSTAR_FD_H = a.fd_h
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    times = [float(v) for v in a.times.split(",")] if a.times else []
    snaps, lim = load(a.run, times)
    print("%s: %d snapshots, limiter %s, methods %s\n"
          % (a.run, len(snaps), lim, ", ".join(METHODS)))

    agg = {m: dict(n=0, conv=0, fb=0, wob=0, iters=[], ev=0, bw=0, halv=0,
                   wall=0.0) for m in METHODS}
    cmp = {m: dict(both=0, moved=0, dp=[], dflux=0.0) for m in METHODS[1:]}
    for s in snaps:
        for axis, sL, sR in faces(s, eos, lim):
            res = {m: run(sL, sR, eos, axis, m, a.repeats) for m in METHODS}
            F0, p0, d0, _ = res["secant"]
            for m, (F, ps, d, wall) in res.items():
                g = agg[m]
                g["n"] += ps.numel()
                g["conv"] += ps.numel() - d["n_not_converged"]
                g["fb"] += d["n_hlle_fallback"]
                g["wob"] += d["n_wave_order_bad"]
                g["ev"] += d["batched_evals"]
                if d.get("mean_iters") == d.get("mean_iters"):   # not nan
                    g["iters"].append((d["mean_iters"], ps.numel()))
                c = d.get("pstar_cost")
                if c:
                    g["bw"] += c["backward"]; g["halv"] += c["halvings"]
                g["wall"] += wall
                if m == "secant":
                    continue
                # where both produced a real HLLD answer (no fallback), did
                # they find the same root?
                good = (p0 > 0) & (ps > 0)
                rel = ((ps - p0).abs() / p0.abs().clamp(min=1e-300))[good]
                cm = cmp[m]
                cm["both"] += int(good.sum())
                cm["moved"] += int((rel > 1e-6).sum())
                cm["dp"].append(rel.numpy())
                gsc = max(float(F0[k].abs().max()) for k in F0) or 1.0
                for k in F0:
                    sc = max(float(F0[k].abs().max()), 1e-6 * gsc)
                    cm["dflux"] = max(cm["dflux"],
                                      float((F[k] - F0[k]).abs().max()) / sc)

    print("%-12s %8s %9s %9s %11s %7s %8s %8s %9s"
          % ("method", "faces", "converged", "HLLE f/b", "wrong order",
             "iters", "evals", "backward", "wall (s)"))
    for m in METHODS:
        g = agg[m]
        it = (sum(v * w for v, w in g["iters"]) / max(sum(w for _, w in g["iters"]), 1))
        print("%-12s %8d %8.3f%% %8.3f%% %10.3f%% %7.2f %8d %8s %9.3f"
              % (m, g["n"], 100 * g["conv"] / g["n"], 100 * g["fb"] / g["n"],
                 100 * g["wob"] / g["n"], it, g["ev"], g["bw"] or "-",
                 g["wall"]))
    print("(evals: batched residual evaluations summed over the calls, "
          "bracket widening included; backward: autograd passes)")
    print("\nagainst the secant (interfaces where both returned an HLLD p*):")
    for m, cm in cmp.items():
        dp = np.concatenate(cm["dp"]) if cm["dp"] else np.zeros(0)
        q = np.quantile(dp, [0.5, 0.99, 1.0]) if dp.size else [np.nan] * 3
        print("  %-10s p* rel. change median %.1e  p99 %.1e  max %.1e | "
              "different root (>1e-6): %d of %d | max flux change %.1e"
              % (m, q[0], q[1], q[2], cm["moved"], cm["both"], cm["dflux"]))
    H._PSTAR_METHOD = "secant"


if __name__ == "__main__":
    main()
