"""Solve the interfaces a run could not, by whatever means works.

    python scripts/solve_interfaces.py <harvest-dir|shard.npz> --out data/solved.npz \
        [--ncells 1024] [--tend 0.40] [--max N] [--chunk 400] [--limiter mc]

The production solver leaves ~56% of the interfaces it attempts unsolved, and
today's measurements say why: the seven-wave Newton has no root to find inside
the physically bounded box for most of them (an exhaustive scan over the eight
planar angle choices crossed with a 7^3 magnitude grid -- 2744 points -- leaves
the best residual at 4.7e-2 and converges 8.5%).  Better seeds do not help
something that is not there.

So the interface is answered a different way: evolve it as a high-resolution
1D shock tube, which solves the PDE and therefore produces the real wave
structure whatever its degeneracies, and read the eight zones and seven wave
speeds off the profile.  Where that read lets the Newton converge, the EXACT
root is kept instead -- the tube is then only a starting guess and the answer
is a root of the system to 1e-8.  Where it does not, the tube answer is kept
and flagged, because an approximate answer to the real problem is worth more
as training data than an exact answer to a problem the run never presents.

Output is the harvest/trainer schema -- U_L, U_R (n,8), zones (n,8,7), speeds
(n,7), gamma -- plus:

    exact     bool, True where a converged seven-wave root was found
    resid     the residual achieved (1e-8 or below on exact rows)
    ncells    the tube resolution used
    method    0 tube only, 1 Newton from the tube read, 2 Newton from an
              enumerated angle combination

Rows with ``exact = False`` carry a tube answer whose accuracy is set by the
resolution; measure it with scripts/../tube_resolution before trusting it, and
keep the flag so those rows can be ablated.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RMHD = os.environ.get("RMHD_ROOT", "/Users/miler/Codes/rmhd_final")
if _RMHD not in sys.path:
    sys.path.append(_RMHD)

np.seterr(all="ignore")

from src.physics.eos import hybrid_eos                       # noqa: E402
from src.physics.tube_seed import run_tubes, read_zones      # noqa: E402


def _load(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "unsolved_*.npz")))
        else:
            files += sorted(glob.glob(p))
    if not files:
        sys.exit("no input shards found")
    UL = np.concatenate([np.load(f)["U_L"] for f in files]).astype(float)
    UR = np.concatenate([np.load(f)["U_R"] for f in files]).astype(float)
    return UL, UR, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ncells", type=int, default=1024)
    ap.add_argument("--tend", type=float, default=0.40,
                    help="a wave travels at most t from the centre and the "
                         "domain half-width is 0.5, so 0.45 is the ceiling; "
                         "later is better than finer at equal cost, because "
                         "the zones separate linearly in t while the contact "
                         "smears only as its square root")
    ap.add_argument("--limiter", default="mc", choices=["mc", "minmod", "pcm"])
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=5.0 / 3.0)
    ap.add_argument("--accuracy", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    eos = hybrid_eos(K=0.0, gamma=a.gamma, gamma_th=a.gamma)

    from batched.ray_scalar import claim_rmhd_namespace
    claim_rmhd_namespace()
    from eos import set_eos
    set_eos("ideal")
    from batched import fullcontact_b as FB
    W = FB.make_solver(a.gamma)
    full6 = W["fullcontact6"]

    UL, UR, files = _load(a.inputs)
    if a.max is not None and UL.shape[0] > a.max:
        rng = np.random.default_rng(a.seed)
        i = np.sort(rng.choice(UL.shape[0], a.max, replace=False))
        UL, UR = UL[i], UR[i]
    n = UL.shape[0]
    print(f"{n} interfaces from {len(files)} shards; tube {a.ncells} cells to "
          f"t = {a.tend}, limiter {a.limiter}", flush=True)

    ANG = [(dp, pl, pr) for dp in (0.0, np.pi) for pl in (0.0, np.pi)
           for pr in (0.0, np.pi)]
    out = dict(U_L=UL, U_R=UR,
               zones=np.full((n, 8, 7), np.nan), speeds=np.full((n, 7), np.nan),
               exact=np.zeros(n, dtype=bool), resid=np.full(n, np.inf),
               method=np.zeros(n, dtype=np.int8))
    t_start = time.time()

    for a0 in range(0, n, a.chunk):
        b0 = min(n, a0 + a.chunk)
        sl = slice(a0, b0)
        m = b0 - a0
        L = UL[sl]; R = UR[sl]
        Bn = L[:, 7].copy()
        left = [L[:, j].copy() for j in range(7)]
        right = [R[:, j].copy() for j in range(7)]

        prof, t_end = run_tubes(L[:, :7], R[:, :7], Bn, eos, ncells=a.ncells,
                                tend=a.tend, limiter=a.limiter, max_steps=100000)
        unk, ok, zt, st = read_zones(prof, Bn, a.gamma, full=True,
                                     dx=1.0 / a.ncells, t=t_end)
        out["zones"][sl] = zt
        out["speeds"][sl] = st

        # candidates: the tube's own read, then the eight planar angle choices
        # on the tube's magnitudes.  Each is solved with the angles FROZEN --
        # they are not free in a planar flow, and freezing removes the three
        # Jacobian columns that are finite-difference noise there.
        psi0 = np.arctan2(L[:, 6], L[:, 5])
        frozen = np.zeros((m, 6), dtype=bool)
        frozen[:, 2] = frozen[:, 4] = frozen[:, 5] = True
        best = np.full(m, np.inf)
        for k, cand in enumerate([None] + ANG):
            if cand is None:
                u = [np.where(ok, unk[c], 0.0) for c in range(6)]
                if not ok.any():
                    continue
            else:
                dp, pl, pr = cand
                u = [unk[0], unk[1], psi0 + dp, unk[3],
                     np.full(m, pl), np.full(m, pr)]
                u = [np.where(ok, c, 0.0) for c in u]
            r = full6(left, right, u, Bn, accuracy=a.accuracy,
                      max_iter=a.max_iter, zero_cols=frozen)
            c = r["converged"].astype(bool) & ok
            nrm = np.where(c, r["nrm"], np.inf)
            take = np.flatnonzero(nrm < best)
            if take.size:
                best[take] = nrm[take]
                g = a0 + take
                out["exact"][g] = True
                out["resid"][g] = nrm[take]
                out["method"][g] = 1 if cand is None else 2
                for z in range(6):
                    for j in range(7):
                        out["zones"][g, z + 1, j] = r["zones"][z][j][take]
                out["zones"][g, 0, :] = L[take, :7]
                out["zones"][g, 7, :] = R[take, :7]
                vxc = 0.5 * (out["zones"][g, 3, 2] + out["zones"][g, 4, 2])
                out["speeds"][g] = np.stack(
                    [r["VsLv"][0][take], r["VsLv"][1][take], r["VsLv"][2][take],
                     vxc, r["VsRv"][2][take], r["VsRv"][1][take],
                     r["VsRv"][0][take]], axis=1)
        done = b0
        el = time.time() - t_start
        print(f"  {done:>7}/{n}  exact {int(out['exact'][:done].sum()):>7} "
              f"({100 * out['exact'][:done].mean():5.1f}%)   "
              f"{el:6.0f}s  {1000 * el / done:5.0f} ms/interface", flush=True)

    keep = np.isfinite(out["zones"]).all(axis=(1, 2))
    print(f"\nusable rows {int(keep.sum())}/{n}   exact {int(out['exact'].sum())} "
          f"({100 * out['exact'].mean():.1f}%)   tube-only "
          f"{int((keep & ~out['exact']).sum())}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, gamma=a.gamma, ncells=a.ncells, tend=a.tend,
                        **{k: v[keep] for k, v in out.items()})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
