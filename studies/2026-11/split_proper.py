#!/usr/bin/env python
"""The proper split of any 2D run's solved faces (2026-11-03; user: "Please
also train the networks with the fixes we did ... for the rotor and orszag
tang").  r2d_split2.py (the four-quadrant split, job 77245) generalised:

    phase 1:  python split_proper.py ALL.npz OUTDIR --tend T
    phase 2:  python split_proper.py ALL.npz OUTDIR --tend T --perturb PERTURB.npz

ALL.npz = harvest_to_dataset.py --keep-tags --hold-out-t 99 of the run's
harvest (every solved face with its time t).  The time blocks are the
four-quadrant's, scaled to the run's end time (at t_end 0.4: TEST t = 0.175 /
0.325 +- 0.0125, VAL t = 0.25 +- 0.0125, 0.00625 buffers dropped around every
block), and the problems are deduplicated on (U_L, U_R) rounded to --bits
significant bits (33: ~1e-10), not bit for bit: the Orszag-Tang's symmetric
twins agree to ~1e-13 without being identical (two of its harvest shards:
4636 rows, 4156 bitwise-unique, 2090 to 1e-12).  As in r2d_split2.py, only
problems whose stored solution IS a seven-wave solution are kept (the residual
at its six unknowns <= 1e-6 and the wave-order gate's rule), and the complex
TRAIN problems (exact rotations > --rot) are repeated --dup times, in training
only.

Phase 1 writes val.npz and test.npz (one row per problem, with t, rot,
source), proxy_val.npz (3000 VAL problems, every complex one up to 1000, for
train_variant --proxy-evalset), train_run.npz, index.npz (what phase 2 needs)
and split.json, with the leakage check: every VAL / TEST problem's nearest
TRAIN problem against the spacing inside TRAIN.  Between the phases the
perturbed faces are drawn from train_run.npz and solved (r2d_resolve.py
perturb).  Phase 2 keeps a perturbed face only if its nearest run problem is a
TRAIN problem, writes train_run_pert.npz (train_run.npz plus the kept faces)
and adds the "perturbed" entry to split.json.
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

# fractions of t_end: the four-quadrant's blocks at t_end = 0.4
TEST_F, VAL_F, HALF_F, BUF_F = (0.4375, 0.8125), (0.625,), 0.03125, 0.015625
KEYS = ("U_L", "U_R", "zones", "speeds")
PROD = os.environ.get("ML_PROD", "/mnt/rafast/miler/codes/rotor2d/rmhd_final/data/ml_guess_rotor_ft.pt")


def load(path):
    z = np.load(path)
    d = {k: np.asarray(z[k]) for k in z.files}
    d["U_L"], d["U_R"] = d["U_L"].astype(float), d["U_R"].astype(float)
    return d


def take(d, idx):
    n = d["U_L"].shape[0]
    return {k: (v[idx] if np.ndim(v) and v.shape[:1] == (n,) else v) for k, v in d.items()}


def keys_round(UL, UR, bits):
    """(U_L, U_R) with every value rounded to `bits` significant bits."""
    a = np.ascontiguousarray(np.concatenate([UL, UR], axis=1), dtype=np.float64)
    m, e = np.frexp(a)
    q = np.ascontiguousarray(np.ldexp(np.round(np.ldexp(m, bits)), e - bits) + 0.0)   # + 0.0: no -0
    return q.view(np.dtype((np.void, 8 * q.shape[1]))).ravel()


def unique_first(d, bits):
    """Indices of the first row of every problem, in row order."""
    _, first = np.unique(keys_round(d["U_L"], d["U_R"], bits), return_index=True)
    return np.sort(first)


def seven_wave_ok(d, W, chunk):
    """Residual at the stored unknowns <= 1e-6 and the gate's order rule."""
    res, err = residual(d["U_L"], d["U_R"], d["zones"], W, chunk)
    return (res <= 1e-6) & ~err & ~gate_crossed(d["zones"], d["speeds"], 1e-6), res


def features(d, idx, xm, xs):
    F = canonical_features(d["U_L"][idx, :7], d["U_R"][idx, :7], d["U_L"][idx, 7])[0]
    return (np.asarray(F, float) - xm) / xs


def save(path, d, idx, gamma, extra=()):
    np.savez_compressed(path, gamma=gamma, **{k: d[k][idx] for k in KEYS + tuple(extra)})


def blocks(t, tend):
    """Labels per problem: 0 TRAIN, 1 VAL, 2 TEST, 3 buffer."""
    test_c, val_c = [c * tend for c in TEST_F], [c * tend for c in VAL_F]
    half, buf = HALF_F * tend, BUF_F * tend
    near = lambda cs, h: np.any([np.abs(t - c) < h for c in cs], axis=0)
    te = near(test_c, half)
    va = near(val_c, half) & ~te
    bf = near(test_c + val_c, half + buf) & ~te & ~va
    return np.where(te, 2, np.where(va, 1, np.where(bf, 3, 0))), test_c, val_c, half, buf


def scaler_stats():
    _, scaler = mg.load(PROD)
    return np.asarray(scaler.x_mean.cpu(), float), np.asarray(scaler.x_std.cpu(), float)


def phase1(a, run, first, gamma, W, rep):
    rng = np.random.default_rng(a.seed)
    ok, res = seven_wave_ok(run, W, a.chunk)
    rep["run_dropped_not_seven_wave"] = int((~ok).sum())
    rep["run_residual_above_1e-8"] = int((res > 1e-8).sum())
    t = run["t"].astype(float)
    label, test_c, val_c, half, buf = blocks(t, a.tend)
    rep["blocks"] = dict(t_end=a.tend, test=test_c, val=val_c, half_width=half, buffer=buf)
    rot = rotation(run["zones"])
    cx = rot > a.rot
    sets = {s: np.flatnonzero((label == k) & ok) for s, k in (("train", 0), ("val", 1), ("test", 2))}
    rep["dropped_in_buffers"] = int(((label == 3) & ok).sum())
    for s, i in sets.items():
        rep[s] = dict(problems=int(i.size), complex=int(cx[i].sum()),
                      seven_wave=int((run["source"][i] == 0).sum()), planar5=int((run["source"][i] == 1).sum()),
                      t=[float(t[i].min()), float(t[i].max())])
    np.savez_compressed(os.path.join(a.out, "index.npz"), first=first, ok=ok, label=label.astype(np.int8),
                        res=res, bits=a.bits, tend=a.tend)

    # held-out sets, one row per problem
    run["rot"] = rot
    save(os.path.join(a.out, "val.npz"), run, sets["val"], gamma, extra=("t", "rot", "source"))
    save(os.path.join(a.out, "test.npz"), run, sets["test"], gamma, extra=("t", "rot", "source"))
    vc = sets["val"][cx[sets["val"]]]
    pc = rng.choice(vc, size=min(1000, vc.size), replace=False)
    pr = rng.choice(np.setdiff1d(sets["val"], pc), size=min(3000 - pc.size, sets["val"].size - pc.size),
                    replace=False)
    pick = np.sort(np.concatenate([pc, pr]))
    np.savez_compressed(os.path.join(a.out, "proxy_val.npz"), U_L=run["U_L"][pick], U_R=run["U_R"][pick])
    rep["proxy_val"] = dict(problems=int(pick.size), complex=int(pc.size))

    # training: the run's TRAIN problems, the complex ones x dup
    trn = sets["train"]
    rows = np.concatenate([trn] + [trn[cx[trn]]] * (a.dup - 1))
    rng.shuffle(rows)
    save(os.path.join(a.out, "train_run.npz"), run, rows, gamma)
    rep["train_run_rows"] = int(rows.size)

    # leakage: nearest TRAIN problem of every VAL / TEST problem, against TRAIN's own spacing
    xm, xs = scaler_stats()
    Ftr = features(run, trn, xm, xs)
    tree = cKDTree(Ftr)
    smp = rng.choice(trn.size, size=min(20000, trn.size), replace=False)
    sp = tree.query(Ftr[smp], k=2, workers=-1)[0][:, 1]
    rep["train_spacing_median"] = float(np.median(sp))
    for s, cen in (("val", val_c), ("test", test_c)):
        d = tree.query(features(run, sets[s], xm, xs), k=1, workers=-1)[0]
        tt = t[sets[s]]
        edge = half - np.min([np.abs(tt - c) for c in cen], axis=0)       # distance to the block's edge
        rep[s + "_nearest_train"] = dict(
            median=float(np.median(d)),
            below_train_spacing=int((d < rep["train_spacing_median"]).sum()),
            below_1e_3=int((d < 1e-3).sum()), exact=int((d == 0).sum()),
            outer_third_below_1e_3=int((d[edge < half / 3] < 1e-3).sum()),
            outer_third=int((edge < half / 3).sum()))


def phase2(a, run, gamma, W, rep):
    """The perturbed faces: kept where the nearest run problem is a TRAIN problem."""
    rng = np.random.default_rng(a.seed + 1)
    ix = np.load(os.path.join(a.out, "index.npz"))
    if int(ix["bits"]) != a.bits or float(ix["tend"]) != a.tend:
        sys.exit("index.npz was written with other --bits / --tend")
    ok, label = ix["ok"], ix["label"]
    good = np.flatnonzero(ok)
    xm, xs = scaler_stats()
    tree_all = cKDTree(features(run, good, xm, xs))
    pert = load(a.perturb)
    if abs(float(pert["gamma"]) - gamma) > 1e-12:
        sys.exit("the perturbed faces were solved at gamma %r, the run at %r" % (float(pert["gamma"]), gamma))
    pert = take(pert, unique_first(pert, a.bits))
    pok, _ = seven_wave_ok(pert, W, a.chunk)
    pn = np.arange(pert["U_L"].shape[0])
    _, nn = tree_all.query(features(pert, pn, xm, xs), k=1, workers=-1)
    lab = label[good][nn]
    keep = pok & (lab == 0)
    rep["perturbed"] = dict(file=a.perturb, problems=int(pn.size), dropped_not_seven_wave=int((~pok).sum()),
                            nearest_val=int((pok & (lab == 1)).sum()), nearest_test=int((pok & (lab == 2)).sum()),
                            nearest_buffer=int((pok & (lab == 3)).sum()), kept=int(keep.sum()),
                            kept_complex=int((rotation(pert["zones"])[keep] > a.rot).sum()))
    tr = np.load(os.path.join(a.out, "train_run.npz"))
    both = {k: np.concatenate([np.asarray(tr[k], float), np.asarray(pert[k], float)[keep]]) for k in KEYS}
    order = rng.permutation(both["U_L"].shape[0])
    np.savez_compressed(os.path.join(a.out, "train_run_pert.npz"), gamma=gamma,
                        **{k: v[order] for k, v in both.items()})
    rep["train_run_pert_rows"] = int(order.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("all")
    ap.add_argument("out")
    ap.add_argument("--tend", type=float, required=True)
    ap.add_argument("--perturb", default=None, help="phase 2: the solved perturbed faces (trainer schema)")
    ap.add_argument("--bits", type=int, default=33)
    ap.add_argument("--dup", type=int, default=10)
    ap.add_argument("--rot", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--max", type=int, default=None, help="the first N rows only (a smoke test)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jpath = os.path.join(a.out, "split.json")

    run = load(a.all)
    if "t" not in run:
        sys.exit("%s has no t: build it with harvest_to_dataset --keep-tags --hold-out-t 99" % a.all)
    if a.max is not None:
        run = take(run, np.arange(min(a.max, run["U_L"].shape[0])))
    gamma = float(run["gamma"])
    W = FB.make_waves(gamma)
    nrows = run["U_L"].shape[0]
    first = unique_first(run, a.bits)
    if a.perturb is None:
        rep = dict(all=a.all, gamma=gamma, run_rows=int(nrows),
                   run_bitwise_unique=int(np.unique(keys16(run["U_L"], run["U_R"])).size),
                   bits=a.bits, run_problems=int(first.size))
        phase1(a, take(run, first), first, gamma, W, rep)
    else:
        rep = json.load(open(jpath))
        if not np.array_equal(np.load(os.path.join(a.out, "index.npz"))["first"], first):
            sys.exit("index.npz does not match %s" % a.all)
        phase2(a, take(run, first), gamma, W, rep)
    json.dump(rep, open(jpath, "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
