#!/usr/bin/env python
"""Split the four-quadrant run's solved faces for the retrain loop
(2026-11-02, iteration 1).

    python r2d_split.py ALL.npz OUTDIR [--dup 10] [--rot 0.5] [--seed 1]

ALL.npz: scripts/harvest_to_dataset.py --keep-tags --hold-out-t 99 of
results/r2d_64_exact_ssretry/harvest (every row, with source, attempts and the
recovered time t; t = 99 holds nothing out).  Writes to OUTDIR:

  eval.npz   every row within 0.0125 of t = 0.175 or 0.325 (the windows the
             rotor study held out), with its tags and rotation -- never
             trained on;
  train.npz  the other rows, the complex ones (exact Alfven rotations
             |phi_L| + |phi_R| > --rot rad) repeated --dup times in all:
             2% of the faces, and the only ones where the seed decides
             convergence (paired test 2026-11-02);
  proxy.npz  3000 held-out faces for the trainer's checkpoint selection
             (train_variant.py --proxy-evalset): every held-out complex face
             up to 1000, the rest drawn at random.
"""
import argparse
import os

import numpy as np

WINDOWS = (0.175, 0.325)
DT = 0.0125
KEYS = ("U_L", "U_R", "zones", "speeds")


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def rotation(Z):
    """|phi_L| + |phi_R| of the exact solution (zones (N, 8, 7), R1..R8)."""
    ang = lambda R: np.arctan2(R[:, 6], R[:, 5])
    return np.abs(wrap(ang(Z[:, 2]) - ang(Z[:, 1]))) + np.abs(wrap(ang(Z[:, 6]) - ang(Z[:, 5])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("all")
    ap.add_argument("outdir")
    ap.add_argument("--dup", type=int, default=10)
    ap.add_argument("--rot", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    z = np.load(a.all)
    gamma = float(z["gamma"])
    t = np.asarray(z["t"], float)
    rot = rotation(np.asarray(z["zones"], float))
    near = np.zeros(t.size, bool)
    for w in WINDOWS:
        near |= np.abs(t - w) < DT
    cx = rot > a.rot
    rng = np.random.default_rng(a.seed)

    ev = np.flatnonzero(near)
    np.savez_compressed(os.path.join(a.outdir, "eval.npz"), gamma=gamma, rot=rot[ev], t=t[ev],
                        source=z["source"][ev], attempts=z["attempts"][ev],
                        **{k: z[k][ev] for k in KEYS})
    tr = np.flatnonzero(~near)
    rep = np.concatenate([tr] + [np.flatnonzero(~near & cx)] * (a.dup - 1))
    rng.shuffle(rep)
    np.savez_compressed(os.path.join(a.outdir, "train.npz"), gamma=gamma, **{k: z[k][rep] for k in KEYS})
    evc = np.flatnonzero(near & cx)
    pick_c = rng.choice(evc, size=min(1000, evc.size), replace=False)
    rest = np.setdiff1d(ev, pick_c)
    pick_r = rng.choice(rest, size=3000 - pick_c.size, replace=False)
    pick = np.sort(np.concatenate([pick_c, pick_r]))
    np.savez_compressed(os.path.join(a.outdir, "proxy.npz"), U_L=z["U_L"][pick], U_R=z["U_R"][pick])
    print("rows %d (t %.3f..%.3f); complex (rotation > %.2f) %d = %.2f%%"
          % (t.size, t.min(), t.max(), a.rot, cx.sum(), 100 * cx.mean()))
    print("eval (held out) %d rows, %d complex; train %d rows -> %d with the complex x%d; proxy %d (%d complex)"
          % (ev.size, evc.size, tr.size, rep.size, a.dup, pick.size, pick_c.size))


if __name__ == "__main__":
    main()
