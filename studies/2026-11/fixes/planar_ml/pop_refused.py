#!/usr/bin/env python3
"""Population for the planar dataset: every face a 64^2/128^2 run's
seven-wave Newton did not answer admissibly, from its harvest --

  kind 0  the unsolved rows (the Newton's refusals, in face order per shard);
  kind 1  the solved seven-wave rows (source 0) with a jump across which the
          entropy falls by more than 1e-9 (entropy_min7 on the stored zones):
          the answers the admissibility gate refuses, which a run without the
          gate kept and a run with it recorded before refusing them.

    python3 pop_refused.py RUN OUT.npz [--stride K]   (every K-th unsolved row)

Rows keep their shard, a stand-in for time (shards are written in order)."""
import argparse, glob, os
import numpy as np
R = "/mnt/rafast/miler/codes/rotor2d/HLLD/results"
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("out")
ap.add_argument("--stride", type=int, default=1)
a = ap.parse_args()


def K(X, Bn, gamma):
    rho, Pt, vx, vy, vz, By, Bz = (X[:, j] for j in range(7))
    W2 = 1.0 / np.maximum(1.0 - (vx * vx + vy * vy + vz * vz), 1e-300)
    vB = vx * Bn + vy * By + vz * Bz
    return (Pt - 0.5 * ((Bn ** 2 + By ** 2 + Bz ** 2) / W2 + vB ** 2)) / rho ** gamma


def entropy_min7(Z, Bn, gamma):
    """zones (k, 8, 7) = left, R2..R7, right: the smallest K_behind/K_ahead
    over the six waves (the contact R4|R5 excluded)."""
    m = np.ones(Z.shape[0])
    with np.errstate(all="ignore"):
        for ah, bh in ((0, 1), (1, 2), (2, 3), (5, 4), (6, 5), (7, 6)):
            q = K(Z[:, bh], Bn, gamma) / K(Z[:, ah], Bn, gamma)
            m = np.minimum(m, np.where(np.isfinite(q), q, 1.0))
    return m


H = os.path.join(R, a.run, "harvest")
UL, UR, SH, KD = [], [], [], []
gamma = None
n_u = n_c = n_s0 = 0
for fu in sorted(glob.glob(H + "/unsolved_r*.npz")):
    tag = os.path.basename(fu)[len("unsolved_"):-4]          # r000_00012
    shard = int(tag.split("_")[1]) + 100000 * int(tag[1:4])
    z = np.load(fu)
    gamma = float(z["gamma"])
    sel = np.arange(len(z["U_L"]))[::a.stride]
    UL.append(z["U_L"][sel]); UR.append(z["U_R"][sel])
    SH.append(np.full(sel.size, shard)); KD.append(np.zeros(sel.size, np.int8))
    n_u += sel.size
    fs = os.path.join(H, "solved_%s.npz" % tag)
    if os.path.exists(fs):
        s = np.load(fs)
        src = s["source"] if "source" in s.files else np.zeros(len(s["U_L"]), np.int8)
        i0 = np.flatnonzero(src == 0)
        n_s0 += i0.size
        if i0.size:
            em = entropy_min7(s["zones"][i0], s["U_L"][i0, 7], gamma)
            c = i0[em < 1.0 - 1e-9]
            UL.append(s["U_L"][c]); UR.append(s["U_R"][c])
            SH.append(np.full(c.size, shard)); KD.append(np.ones(c.size, np.int8))
            n_c += c.size
UL, UR, SH, KD = (np.concatenate(x) for x in (UL, UR, SH, KD))
np.savez_compressed(a.out, UL=UL, UR=UR, shard=SH, kind=KD, gamma=gamma, run=a.run,
                    group=np.zeros(len(SH), np.int8))
print("%s: %d unsolved rows (stride %d) + %d entropy-falling seven-wave answers of %d (%.2f%%) "
      "-> %d faces in %d shards" % (a.run, n_u, a.stride, n_c, n_s0, 100.0 * n_c / max(n_s0, 1),
                                    len(SH), len(np.unique(SH))))
