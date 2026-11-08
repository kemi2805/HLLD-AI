#!/usr/bin/env python3
"""Training arrays for the planar warm-start network, from the planar dataset
(build_dataset.py: results/planar_ml/<run>/) and, later, the harvest runs.

Every solved row is an exact answer of a coplanar face in the seven-wave
layout (zones left, R2..R7, right).  It is mapped to the planar canonical
frame the planar solvers use:

  rotate  about the normal so the LEFT tangential field lies along +y
          (planar5_b.to_planar); if the left field is zero, the right one;
  reflect y -> -y when the left field is zero and the right one points down;
  scale   rho, P *= 1/rho_L, B *= 1/sqrt(rho_L)  (velocities unchanged).

Features (14): log P_L, vx_L, vy_L, By_L, asinh(By_L/1e-3),
               log rho_R, log P_R, vx_R, vy_R, By_R, asinh(By_R/1e-3),
               Bn, asinh(Bn/1e-3)                         (canonical units),
               gamma -- the dataset mixes 5/3 (4Q, rotor) and 4/3 (OT), and
               the answer depends on it; the seven-wave networks leave it
               out and are one per EOS
Targets:
  reg (4)   ln p_LF, ln p_RF (total pressure behind each fast wave),
            By_CD, ln P_CD (the field and total pressure at the contact,
            R4) -- the planar4 seed in either steering is
            [ln p_LF, By_CD, By_CD, ln p_RF] (signed field) or
            [ln p_LF, ln P_CD, ln P_CD, ln p_RF] (pressure);
  rot (2)   a rotation in front of each slow wave: the late rung's own flag
            (source 2), else the Alfven zone differs from the fast one;
  steer (2) the slow wave steered by the pressure (planar4_b.KIND_P), only
            where a planar4-family solver answered (source 2, 3; `steer_mask`).
Each row also gets its mirror image (x -> -x: swap the sides, negate v_x and
the tangential field), flagged `mirror`, with its base row's id, so a split
on `base` keeps the pair together.  Rows carry `t` (the face's time in its
run) and `t_frac` (t / t_end) for a split by time.

    python3 planar_data.py OUT.npz DATASET_DIR [DATASET_DIR ...]
"""
import glob
import json
import os
import sys

import numpy as np

R = "/mnt/rafast/miler/codes/rotor2d/HLLD/results"
PER = 6                      # sweeps per step (RK3, x and y) in the 2D runs
TINY = 1e-12


def run_times(run):
    """step -> t from the run's diag.csv, and t_end."""
    d = np.genfromtxt(os.path.join(R, run, "diag.csv"), delimiter=",", names=True,
                      usecols=(0, 1))
    return dict(zip(d["step"].astype(int), d["t"])), float(d["t"][-1])


def shard_time(run, shard, t_of):
    """Mean time of a harvest shard of `run` (its coverage sweeps)."""
    f = os.path.join(R, run, "harvest", "coverage_r%03d_%05d.npz" % (shard // 100000, shard % 100000))
    sw = np.load(f)["sweep"]
    steps = sw // PER + 1
    return float(np.mean([t_of.get(int(s), np.nan) for s in steps]))


def planar(U, Z):
    """Rotate U (n, 8) and the zones Z (n, 8, 7) about x so the left
    tangential field lies along +y (the right one if the left is zero), then
    reflect y if the left field is zero and the right one points down.
    Returns rotated copies and the out-of-plane residue."""
    UL, UR = U
    L, Rr = UL.copy(), UR.copy()
    ByL, BzL = UL[:, 5], UL[:, 6]
    ByR, BzR = UR[:, 5], UR[:, 6]
    left_zero = np.hypot(ByL, BzL) <= TINY * np.maximum(1.0, np.abs(UL[:, 7]))
    ang = np.where(left_zero, np.arctan2(BzR, ByR), np.arctan2(BzL, ByL))
    ca, sa = np.cos(-ang), np.sin(-ang)

    def rot(X):                                  # X (..., 7+) with vy, vz, By, Bz at 3,4,5,6
        Y = X.copy()
        c = ca.reshape((-1,) + (1,) * (X.ndim - 2))
        s = sa.reshape((-1,) + (1,) * (X.ndim - 2))
        Y[..., 3] = c * X[..., 3] - s * X[..., 4]
        Y[..., 4] = s * X[..., 3] + c * X[..., 4]
        Y[..., 5] = c * X[..., 5] - s * X[..., 6]
        Y[..., 6] = s * X[..., 5] + c * X[..., 6]
        return Y

    L, Rr, Zr = rot(L), rot(Rr), rot(Z)
    sB = np.maximum(np.maximum(np.hypot(ByL, BzL), np.hypot(ByR, BzR)), 1e-30)
    sv = np.maximum(np.maximum(np.abs(L[:, 3]), np.abs(Rr[:, 3])), 1e-30)
    resid = np.maximum(np.maximum(np.abs(L[:, 6]), np.abs(Rr[:, 6])) / sB,
                       np.maximum(np.abs(L[:, 4]), np.abs(Rr[:, 4])) / sv)
    flip = left_zero & (Rr[:, 5] < 0.0)
    for X in (L, Rr):
        X[flip, 3] *= -1.0
        X[flip, 5] *= -1.0
    Zr[flip, :, 3] *= -1.0
    Zr[flip, :, 5] *= -1.0
    return L, Rr, Zr, resid


def mirror(UL, UR, Z):
    """x -> -x: swap the sides and the zone order, negate v_x and the
    tangential field (B is a pseudovector), keep B_n."""
    def M(X):
        Y = X.copy()
        Y[..., 2] *= -1.0
        Y[..., 5] *= -1.0
        Y[..., 6] *= -1.0
        return Y
    nUL, nUR = M(UR), M(UL)
    nUL[:, 7], nUR[:, 7] = UL[:, 7], UL[:, 7]
    return nUL, nUR, M(Z[:, ::-1, :])


def rows_to_arrays(UL, UR, Z, source, rot_col, steer_col, gamma):
    """Features and targets of n rows (module doc)."""
    L, Rr, Zr, resid = planar((UL, UR), Z)
    lam = 1.0 / L[:, 0]
    sq = np.sqrt(lam)
    Bn = UL[:, 7] * sq
    ByL, ByR = L[:, 5] * sq, Rr[:, 5] * sq
    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.stack([np.log(L[:, 1] * lam), L[:, 2], L[:, 3], ByL, np.arcsinh(ByL / 1e-3),
                      np.log(Rr[:, 0] * lam), np.log(Rr[:, 1] * lam), Rr[:, 2], Rr[:, 3], ByR,
                      np.arcsinh(ByR / 1e-3), Bn, np.arcsinh(Bn / 1e-3),
                      np.full(len(Bn), gamma)], axis=1)
        Y = np.stack([np.log(Zr[:, 1, 1] * lam), np.log(Zr[:, 6, 1] * lam),
                      Zr[:, 3, 5] * sq, np.log(Zr[:, 3, 1] * lam)], axis=1)
    # a rotation: the Alfven zone (R3 / R6) differs from the fast one (R2 / R7)
    def differs(a, b):
        d = np.abs(Zr[:, a, 3] - Zr[:, b, 3]) + np.abs(Zr[:, a, 5] - Zr[:, b, 5]) * sq
        s = 1e-8 * (1.0 + np.abs(Zr[:, b, 3]) + np.abs(Zr[:, b, 5]) * sq)
        return d > s
    rot = np.stack([differs(2, 1), differs(5, 6)], axis=1)
    if rot_col is not None:
        late = source == 2
        rot[late] = rot_col[late] > 0
    steer = np.zeros((len(source), 2), bool)
    mask = np.isin(source, (2, 3))
    if steer_col is not None:
        steer = steer_col == 1                       # planar4_b.KIND_P
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1) & (resid <= 1e-6)
    return X, Y, rot, steer, mask, ok


def load_dataset(d):
    """One dataset directory: its rows with time, problem and source."""
    man = json.load(open(os.path.join(d, "manifest.json")))
    run = man["run"]
    t_of, t_end = run_times(run)
    cache = {}
    out = []
    for c, info in sorted(man["chunks"].items(), key=lambda kv: int(kv[0])):
        f = os.path.join(d, "solved_r000_%05d.npz" % int(c))
        if not os.path.exists(f):
            continue
        z = np.load(f)
        s0, s1 = info["shards"]
        ts = []
        for s in range(s0, s1 + 1):
            if s not in cache:
                try:
                    cache[s] = shard_time(run, s, t_of)
                except FileNotFoundError:
                    cache[s] = np.nan
            ts.append(cache[s])
        out.append(dict(z=z, t=float(np.nanmean(ts)), chunk=int(c)))
    return run, t_end, float(man["gamma"]), out


def main():
    dst, dirs = sys.argv[1], sys.argv[2:]
    cols = {k: [] for k in ("X", "Y", "rot", "steer", "steer_mask", "source", "stage", "run",
                            "t", "t_frac", "mirror", "base", "chunk")}
    runs = []
    nbase = 0
    for ri, d in enumerate(dirs):
        run, t_end, gamma, chunks = load_dataset(d)
        runs.append(run)
        n_run = 0
        for ch in chunks:
            z = ch["z"]
            UL, UR, Z, src = z["U_L"], z["U_R"], z["zones"], z["source"]
            rcol = z["rot"] if "rot" in z.files else None
            scol = z["steer"] if "steer" in z.files else None
            stage = z["stage"] if "stage" in z.files else np.zeros(len(src), np.int8)
            base = nbase + np.arange(len(src))
            nbase += len(src)
            for mir in (False, True):
                if mir:
                    uL, uR, zz = mirror(UL, UR, Z)
                    rc = None if rcol is None else rcol[:, ::-1]
                    sc = None if scol is None else scol[:, ::-1]
                else:
                    uL, uR, zz, rc, sc = UL, UR, Z, rcol, scol
                X, Y, rot, steer, mask, ok = rows_to_arrays(uL, uR, zz, src, rc, sc, gamma)
                k = int(ok.sum())
                for key, v in (("X", X[ok]), ("Y", Y[ok]), ("rot", rot[ok]), ("steer", steer[ok]),
                               ("steer_mask", mask[ok]), ("source", src[ok]), ("stage", stage[ok]),
                               ("run", np.full(k, ri, np.int8)), ("t", np.full(k, ch["t"])),
                               ("t_frac", np.full(k, ch["t"] / t_end)),
                               ("mirror", np.full(k, mir)), ("base", base[ok]),
                               ("chunk", np.full(k, ch["chunk"]))):
                    cols[key].append(v)
                n_run += k
        print("%-28s %d rows (with mirrors) from %d chunks" % (run, n_run, len(chunks)), flush=True)
    arr = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez_compressed(dst, runs=np.array(runs), **arr)
    src = arr["source"][~arr["mirror"]]
    print("total %d rows; base rows by source %s; rotations L/R %.1f%% / %.1f%%; "
          "pressure steering where defined L/R %.1f%% / %.1f%%" % (
              len(arr["source"]), np.bincount(src, minlength=4).tolist(),
              100 * arr["rot"][:, 0].mean(), 100 * arr["rot"][:, 1].mean(),
              100 * arr["steer"][arr["steer_mask"], 0].mean() if arr["steer_mask"].any() else 0,
              100 * arr["steer"][arr["steer_mask"], 1].mean() if arr["steer_mask"].any() else 0))


if __name__ == "__main__":
    main()
