#!/usr/bin/env python
"""How good must a seed be?  The seven-wave Newton started from the known
answer, perturbed, on the faces of a finished rotor run.

A run with the five-wave rescue harvests exact answers for three kinds of
face (`src/physics/harvest.py`, solved shards):

    first    source 0, attempts 1   the primary network's seed converged
    retry    source 0, attempts > 1 a retry network's seed converged
    planar   source 1               every network seed failed; the planar
                                    solver answered (from no seed at all)

Before a network is retrained on them, this measures what retraining could
buy.  It starts the seven-wave Newton -- one attempt, production's budget and
accuracy -- from

    * each production network's own seed (reproduces the run's verdict),
    * the exact unknowns moved by a random step of fixed size (``--radius``,
      in the norm the seed error has always been quoted in),
    * the primary network's seed with its error shrunk (``--shrink``): what
      a network k times more accurate, in the same direction, would give,

and counts, per group, how often it converges and how often to the known
answer.  If the Newton does not converge from a small step on the "planar"
faces, no network can make it converge there, and retraining cannot move
those faces off the planar solver.

    python scripts/seed_ceiling.py results/rotor_64_exact_defaults/harvest \\
        --n 500 --out seed_ceiling.npz

The unknowns are ``ml_guess.unk6_true``'s: [ln p_LF, ln|Bt_CD|, psi_CD,
ln p_RF, phi_L, phi_R]; angles enter the error wrapped to (-pi, pi].
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

GROUPS = ("first", "retry", "planar")
ANGLES = (2, 4, 5)


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def unk6_true(Z):
    """Vectorised `rmhd.ml_guess.unk6_true` over zones (N, 8, 7)."""
    R2, R3, R4, R6, R7 = Z[:, 1], Z[:, 2], Z[:, 3], Z[:, 5], Z[:, 6]
    ang = lambda R: np.arctan2(R[:, 6], R[:, 5])
    return np.stack([np.log(R2[:, 1]),
                     np.log(np.hypot(R4[:, 5], R4[:, 6])),
                     ang(R4),
                     np.log(R6[:, 1]),
                     wrap(ang(R3) - ang(R2)),
                     wrap(ang(R7) - ang(R6))])


def seed_error(u, ref):
    d = np.asarray(u, float) - ref
    for k in ANGLES:
        d[k] = wrap(d[k])
    return np.sqrt((d ** 2).sum(axis=0))


def sample(harvest, n, rng):
    """`n` faces per group, drawn uniformly over the whole run."""
    files = sorted(glob.glob(os.path.join(harvest, "solved_*.npz")))
    if not files:
        raise SystemExit("no solved_*.npz under %s" % harvest)
    meta = []
    for f in files:
        z = np.load(f)
        src = z["source"] if "source" in z.files else np.zeros(z["attempts"].size, np.int8)
        meta.append((f, src.astype(int), z["attempts"].astype(int)))
    pick = {g: [] for g in GROUPS}
    for fi, (f, src, att) in enumerate(meta):
        grp = np.where(src == 1, 2, np.where(att > 1, 1, 0))
        for gi, g in enumerate(GROUPS):
            for r in np.flatnonzero(grp == gi):
                pick[g].append((fi, r))
    chosen = {}
    for g in GROUPS:
        allg = pick[g]
        k = min(n, len(allg))
        idx = rng.choice(len(allg), size=k, replace=False) if k else []
        chosen[g] = sorted(allg[i] for i in idx)
        print("  %-7s %8d in the run, %d drawn" % (g, len(allg), k), flush=True)
    rows = {k: [] for k in ("U_L", "U_R", "zones", "attempts", "group")}
    for fi, (f, src, att) in enumerate(meta):
        want = [(g, r) for g in GROUPS for (f2, r) in chosen[g] if f2 == fi]
        if not want:
            continue
        z = np.load(f)
        r_ = np.array([r for _, r in want])
        rows["U_L"].append(z["U_L"][r_])
        rows["U_R"].append(z["U_R"][r_])
        rows["zones"].append(z["zones"][r_])
        rows["attempts"].append(att[r_])
        rows["group"].append(np.array([GROUPS.index(g) for g, _ in want]))
    return {k: np.concatenate(v) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("harvest")
    ap.add_argument("--n", type=int, default=500, help="faces per group")
    ap.add_argument("--radius", type=float, nargs="+", default=[0.0, 0.03, 0.1, 0.3])
    ap.add_argument("--draws", type=int, default=2, help="random steps per radius > 0")
    ap.add_argument("--shrink", type=float, nargs="+", default=[0.5, 0.25, 0.1],
                    help="the primary seed's error times these")
    ap.add_argument("--ckpts", default="data/ml_guess_rotor_ft.pt,"
                    "data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt",
                    help="production's networks, primary first")
    ap.add_argument("--gamma", type=float, default=5.0 / 3.0)
    ap.add_argument("--accuracy", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from rmhd.eos import set_eos
    set_eos("ideal")
    from rmhd import ml_guess as mg
    from rmhd.batched import api as API
    from rmhd.batched import ml_b as MB
    from src.physics import exact_flux as EF

    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    S = sample(a.harvest, a.n, rng)
    N = S["group"].size
    left = [S["U_L"][:, j].copy() for j in range(7)]
    right = [S["U_R"][:, j].copy() for j in range(7)]
    Bn = S["U_L"][:, 7].copy()
    exact = unk6_true(S["zones"])
    grp = S["group"]
    print("%d faces loaded in %.0f s" % (N, time.time() - t0), flush=True)

    # the production networks' seeds, clamped as production clamps them
    nets = {}
    for path in a.ckpts.split(","):
        model, scaler = mg.load(EF._rmhd_paths.resolve(path.strip()))
        u, _ = MB.predict_unk6(model, scaler, np.stack(left, axis=1),
                               np.stack(right, axis=1), Bn)
        u, _ = MB.clamp_seed_physical(u, left, right, Bn)
        nets[os.path.basename(path.strip())] = np.array(u, float)
    primary = next(iter(nets))

    seeds = {}
    for name, u in nets.items():
        seeds["net " + name] = u
    for r in a.radius:
        for d in range(1 if r == 0.0 else a.draws):
            step = rng.standard_normal((6, N))
            step *= r / np.sqrt((step ** 2).sum(axis=0))
            seeds["exact + %.2f step #%d" % (r, d)] = exact + step
    err_dir = nets[primary] - exact
    for k in ANGLES:
        err_dir[k] = wrap(err_dir[k])
    for s in a.shrink:
        seeds["%s error x %.2f" % (primary, s)] = exact + s * err_dir

    solve = API.make_solver(a.gamma)
    out = dict(group=grp, attempts=S["attempts"], exact=exact)

    # the seven-wave residual AT the known answer: a lane whose waves cannot
    # even be constructed there (err) is one the Newton never steps on
    from rmhd.batched import fullcontact_b as FB
    fvec, _, _, _, err = FB.make_solver(a.gamma)["fullfuncv6"](
        left, right, [exact[c].copy() for c in range(6)], Bn)
    err = np.asarray(err, bool)
    r0 = np.where(err, np.inf, np.max(np.abs(np.asarray(fvec)), axis=0))
    out.update(resid_at_exact=r0, unconstructible_at_exact=err)
    print("\n== the seven-wave residual at the known answer")
    for gi, g in enumerate(GROUPS):
        m = grp == gi
        fin = r0[m][np.isfinite(r0[m])]
        print("   %-7s unconstructible %5.1f%%   <= 1e-8 %5.1f%%   median of the rest %.1e"
              % (g, 100 * err[m].mean(), 100 * (r0[m] <= a.accuracy).mean(),
                 np.median(fin) if fin.size else np.nan), flush=True)
    for name, u in nets.items():
        out["seed_err " + name] = seed_error(u, exact)
    print("\n== the networks' seed error on each group (median / p90)")
    for name in nets:
        e = out["seed_err " + name]
        print("   %-34s" % name + "".join(
            "  %s %.3f / %.3f" % (g, np.median(e[grp == gi]), np.percentile(e[grp == gi], 90))
            for gi, g in enumerate(GROUPS)), flush=True)

    print("\n== the seven-wave Newton, one attempt, max_iter %d, accuracy %g"
          % (a.max_iter, a.accuracy))
    print("   %-36s" % "start" + "".join("  %-26s" % g for g in GROUPS))
    for name, u in seeds.items():
        t1 = time.time()
        res, _ = solve(left, right, Bn, seed6=[u[c].copy() for c in range(6)],
                       accuracy=a.accuracy, max_iter=a.max_iter, n_retries=0,
                       tau_bt=1e-9)
        conv = np.asarray(res["converged"], bool)
        got = np.array(res["unk"], float)
        dev = np.abs(got - exact)
        for k in ANGLES:
            dev[k] = np.abs(wrap(got[k] - exact[k]))
        # the known root: pressures to 1e-5 in ln p, the field to 1e-3 (the
        # seven-wave form pins the state only to ~1e-4 at a 1e-8 residual)
        same = conv & (np.maximum(dev[0], dev[3]) < 1e-5) & (dev[1] < 1e-3)
        out["conv " + name] = conv
        out["same " + name] = same
        out["iters " + name] = np.asarray(res["n_iter"])
        out["cls"] = np.asarray(res["cls"])
        cells = []
        for gi in range(len(GROUPS)):
            m = grp == gi
            cells.append("%5.1f%% conv, %5.1f%% same" % (100 * conv[m].mean(),
                                                       100 * same[m].mean()))
        print("   %-36s" % name + "".join("  %-26s" % c for c in cells)
              + "   (%.0f s)" % (time.time() - t1), flush=True)
    np.savez_compressed(a.out, **out)
    print("\nwrote %s, %.0f s" % (a.out, time.time() - t0))


if __name__ == "__main__":
    main()
