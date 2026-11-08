#!/usr/bin/env python
"""The proper split of the four-quadrant data (2026-11-03; user: "fix the
validation split and retrain properly").

    python r2d_split2.py RETRAIN_DIR OUTDIR [--buffer 0.00625] [--dup 10]
                         [--rot 0.5] [--seed 1] [--chunk 100000]

r2d_split.py (iteration 1) trained on duplicated problems, validated on a
random slice of the same file, drew the proxy from the test windows and left
no gap between training and test times.  Here, from it1/all.npz (every solved
face of the 10-29 64^2 run, with its time) and it2/perturb.npz:

  1. one row per problem: the (U_L, U_R) pair bit for bit (the diagonal
     mirror twins are identical rows; each problem occurs at one time);
  2. only problems whose stored solution IS a seven-wave solution: the
     residual at its six unknowns <= 1e-6 and the wave-order gate's rule
     passed (r2d_audit.py, 11-03: 39 planar rescues and field reversals
     fail it);
  3. time blocks: TEST t = 0.175 and 0.325 +- 0.0125 (the windows every
     earlier test used), VAL t = 0.25 +- 0.0125, --buffer dropped on both
     sides of every block, TRAIN the rest;
  4. a perturbed face is kept only if its nearest run problem (in the
     production network's normalised features) is a TRAIN problem -- their
     bases were drawn from iteration 1's training file, which held the VAL
     block and the buffers;
  5. the complex TRAIN problems (exact rotations > --rot) repeated --dup
     times, in training only.

Writes OUTDIR/train_run.npz (5 on the run's problems), train_run_pert.npz
(the same plus the kept perturbed faces), val.npz, test.npz (unique problems
with t, rot, source), proxy_val.npz (3000 VAL problems, every complex one up
to 1000, for train_variant --proxy-evalset) and split.json; prints the
leakage check: every VAL / TEST problem's nearest TRAIN problem against the
spacing inside TRAIN.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

from rmhd import ml_guess as mg
from rmhd.batched import fullcontact_b as FB
from rmhd.batched.ml_b import canonical_features

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2d_audit import gate_crossed, keys16, residual, rotation   # noqa: E402

TEST, VAL, HALF = (0.175, 0.325), (0.25,), 0.0125
KEYS = ("U_L", "U_R", "zones", "speeds")
PROD = "/mnt/rafast/miler/codes/rotor2d/rmhd_final/data/ml_guess_rotor_ft.pt"


def load(path):
    z = np.load(path)
    d = {k: np.asarray(z[k]) for k in z.files}
    d["U_L"], d["U_R"] = d["U_L"].astype(float), d["U_R"].astype(float)
    return d


def seven_wave_ok(d, W, chunk):
    """Residual at the stored unknowns <= 1e-6 and the gate's order rule."""
    res, err = residual(d["U_L"], d["U_R"], d["zones"], W, chunk)
    return (res <= 1e-6) & ~err & ~gate_crossed(d["zones"], d["speeds"], 1e-6), res


def features(d, idx, xm, xs):
    F = canonical_features(d["U_L"][idx, :7], d["U_R"][idx, :7], d["U_L"][idx, 7])[0]
    return (np.asarray(F, float) - xm) / xs


def save(path, d, idx, gamma, extra=()):
    np.savez_compressed(path, gamma=gamma, **{k: d[k][idx] for k in KEYS + tuple(extra)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("out")
    ap.add_argument("--buffer", type=float, default=0.00625)
    ap.add_argument("--dup", type=int, default=10)
    ap.add_argument("--rot", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=100_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rep = {}

    run = load(os.path.join(a.dir, "it1/all.npz"))
    gamma = float(run["gamma"])
    assert abs(gamma - 5.0 / 3.0) < 1e-12, gamma
    _, first = np.unique(keys16(run["U_L"], run["U_R"]), return_index=True)
    first = np.sort(first)
    run = {k: (v[first] if np.ndim(v) and v.shape[:1] == run["U_L"].shape[:1] else v) for k, v in run.items()}
    n = first.size
    W = FB.make_waves(gamma)
    ok, res = seven_wave_ok(run, W, a.chunk)
    rep["run_rows"], rep["run_problems"] = int(np.load(os.path.join(a.dir, "it1/all.npz"))["t"].size), int(n)
    rep["run_dropped_not_seven_wave"] = int((~ok).sum())

    t = run["t"].astype(float)
    near = lambda cs, h: np.any([np.abs(t - c) < h for c in cs], axis=0)
    te = near(TEST, HALF)
    va = near(VAL, HALF) & ~te
    buf = near(TEST + VAL, HALF + a.buffer) & ~te & ~va
    tr = ~te & ~va & ~buf
    rot = rotation(run["zones"])
    cx = rot > a.rot
    sets = {"train": np.flatnonzero(tr & ok), "val": np.flatnonzero(va & ok), "test": np.flatnonzero(te & ok)}
    rep["buffer"] = a.buffer
    rep["dropped_in_buffers"] = int((buf & ok).sum())
    for s, i in sets.items():
        rep[s] = dict(problems=int(i.size), complex=int(cx[i].sum()),
                      seven_wave=int((run["source"][i] == 0).sum()), planar5=int((run["source"][i] == 1).sum()),
                      t=[float(t[i].min()), float(t[i].max())])

    # held-out sets, one row per problem
    run["rot"] = rot
    save(os.path.join(a.out, "val.npz"), run, sets["val"], gamma, extra=("t", "rot", "source"))
    save(os.path.join(a.out, "test.npz"), run, sets["test"], gamma, extra=("t", "rot", "source"))
    vc = sets["val"][cx[sets["val"]]]
    pc = rng.choice(vc, size=min(1000, vc.size), replace=False)
    pr = rng.choice(np.setdiff1d(sets["val"], pc), size=3000 - pc.size, replace=False)
    pick = np.sort(np.concatenate([pc, pr]))
    np.savez_compressed(os.path.join(a.out, "proxy_val.npz"), U_L=run["U_L"][pick], U_R=run["U_R"][pick])
    rep["proxy_val"] = dict(problems=int(pick.size), complex=int(pc.size))

    # training: the run's TRAIN problems, the complex ones x dup
    trn = sets["train"]
    rows = np.concatenate([trn] + [trn[cx[trn]]] * (a.dup - 1))
    rng.shuffle(rows)
    save(os.path.join(a.out, "train_run.npz"), run, rows, gamma)
    rep["train_run_rows"] = int(rows.size)

    # perturbed faces: kept where the nearest run problem is a TRAIN problem
    _, scaler = mg.load(PROD)
    xm = np.asarray(scaler.x_mean.cpu(), float)
    xs = np.asarray(scaler.x_std.cpu(), float)
    good = np.flatnonzero(ok)
    label = np.where(tr, 0, np.where(va, 1, np.where(te, 2, 3)))[good]   # 3 = buffer
    tree_all = cKDTree(features(run, good, xm, xs))
    pert = load(os.path.join(a.dir, "it2/perturb.npz"))
    _, pf = np.unique(keys16(pert["U_L"], pert["U_R"]), return_index=True)
    pert = {k: (v[np.sort(pf)] if np.ndim(v) and v.shape[:1] == pert["U_L"].shape[:1] else v)
            for k, v in pert.items()}
    pok, _ = seven_wave_ok(pert, W, a.chunk)
    pn = np.arange(pert["U_L"].shape[0])
    _, nn = tree_all.query(features(pert, pn, xm, xs), k=1, workers=-1)
    keep = pok & (label[nn] == 0)
    rep["perturbed"] = dict(problems=int(pn.size), dropped_not_seven_wave=int((~pok).sum()),
                            nearest_val=int((pok & (label[nn] == 1)).sum()),
                            nearest_test=int((pok & (label[nn] == 2)).sum()),
                            nearest_buffer=int((pok & (label[nn] == 3)).sum()), kept=int(keep.sum()),
                            kept_complex=int((rotation(pert["zones"])[keep] > a.rot).sum()))
    both = {k: np.concatenate([run[k][rows], pert[k][keep]]) for k in KEYS}
    order = rng.permutation(both["U_L"].shape[0])
    np.savez_compressed(os.path.join(a.out, "train_run_pert.npz"), gamma=gamma,
                        **{k: v[order] for k, v in both.items()})
    rep["train_run_pert_rows"] = int(order.size)

    # leakage: nearest TRAIN problem of every VAL / TEST problem, against TRAIN's own spacing
    Ftr = features(run, trn, xm, xs)
    tree = cKDTree(Ftr)
    smp = rng.choice(trn.size, size=min(20000, trn.size), replace=False)
    sp = tree.query(Ftr[smp], k=2, workers=-1)[0][:, 1]
    rep["train_spacing_median"] = float(np.median(sp))
    for s in ("val", "test"):
        d = tree.query(features(run, sets[s], xm, xs), k=1, workers=-1)[0]
        tt = t[sets[s]]
        cen = VAL if s == "val" else TEST
        edge = HALF - np.min([np.abs(tt - c) for c in cen], axis=0)       # distance to the block's edge
        rep[s + "_nearest_train"] = dict(
            median=float(np.median(d)),
            below_train_spacing=int((d < rep["train_spacing_median"]).sum()),
            below_1e_3=int((d < 1e-3).sum()), exact=int((d == 0).sum()),
            outer_third_below_1e_3=int((d[edge < HALF / 3] < 1e-3).sum()),
            outer_third=int((edge < HALF / 3).sum()))
    json.dump(rep, open(os.path.join(a.out, "split.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
