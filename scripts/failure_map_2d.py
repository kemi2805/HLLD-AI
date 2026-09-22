"""Where on the rotor does the exact solver fail?  (science plan F, step 1a)

Every exact-flux call writes one coverage row to the harvest: a complete
bitmap of which interfaces it ATTEMPTED (they passed the bad-state, weak-jump
and upwind gates) and which came out EXACT (``src/physics/harvest.py``,
``record_coverage``).  A failure is ``attempted & ~exact`` -- the interface
kept HLLD's flux as a fallback.  This puts those failures back on the grid.

Mapping.  ``sweep.py`` flattens the face arrays in C order, so for a grid of
nx x ny cells with ng = 2 ghosts (nxt = nx + 2 ng):

    x-sweep (idir 0): faces (nxt+1, nyt)  -> [fi, j], face fi between cells
                      fi-1 and fi; physical fi in [ng, ng+nx], j in [ng, ng+ny)
    y-sweep (idir 1): faces (nxt, nyt+1)  -> [i, fj], symmetric

and a run does one SSP-RK3 step per time step, three stages, each an x- then
a y-sweep, so sweep s is step s // 6, stage (s % 6) // 2, direction s % 2.
The stage times are t_n, t_n + dt, t_n + dt/2 with t_n from ``diag.csv``.

Counts are accumulated in a window around each snapshot time and drawn over
that snapshot's density, so a failure can be read against the flow it sat
in.  Two diagnostics come with the map:

* **Caveat, the sign bug.**  These harvests predate the Alfven sign fix
  (rmhd_final d09ffe2), which changed answers only where B_n < 0.  Each face's
  normal field sign is read off the nearest snapshot (the two adjacent
  cells' average) and failures on B_n < 0 faces are drawn hollow and counted
  separately: some of them are the bug, not the physics.
* **Enrichment near field reversals.**  The guess to test is that failures
  sit where the tangential field reverses.  A face "is near a reversal" if
  the tangential field (By across an x-face, Bx across a y-face) changes
  sign between the two cells beside it or beside a neighbouring face.
  Enrichment = (share of failures near a reversal) / (share of attempted
  faces near one); 1 means no preference.

    python scripts/failure_map_2d.py results/rotor_64_exact --out figs/failure_map
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

NG = 2
HALF_WINDOW = 0.025


def _unpack(bits, n):
    # = scripts/harvest_summary.py:_unpack
    return np.unpackbits(bits, axis=-1)[..., :n].astype(bool)


def load_diag(rundir):
    d = np.genfromtxt(os.path.join(rundir, "diag.csv"), delimiter=",", names=True)
    return np.atleast_1d(d["t"]), np.atleast_1d(d["dt"])


def sweep_time(s, t_after, dt):
    """Simulation time at which sweep ``s`` evaluated its fluxes."""
    step = s // 6
    stage = (s % 6) // 2
    tn = t_after[step] - dt[step]
    return tn + np.array([0.0, 1.0, 0.5])[stage] * dt[step]


def load_snaps(rundir):
    snaps = []
    for f in sorted(glob.glob(os.path.join(rundir, "snap_*.npz"))):
        if f.endswith("snap_fin.npz"):
            continue
        z = np.load(f)
        snaps.append({k: z[k] for k in z.files})
    snaps.sort(key=lambda s: float(s["t"]))
    return snaps


def face_fields(snap):
    """Per physical face: normal-field sign and whether the tangential field
    reverses between the two adjacent cells.  x-faces (nx+1, ny), y-faces
    (nx, ny+1); boundary faces use the one adjacent cell."""
    Bx, By = snap["Bx"], snap["By"]

    def normal_avg(B, axis):
        pad = np.concatenate([np.take(B, [0], axis), B, np.take(B, [-1], axis)], axis)
        a = np.take(pad, range(0, pad.shape[axis] - 1), axis)
        b = np.take(pad, range(1, pad.shape[axis]), axis)
        return 0.5 * (a + b), a, b

    # a reversal counts only where the field is real on BOTH sides: in the
    # ambient medium B_y is round-off around zero and its sign flips at random
    # (the first version drew "reversals" over the whole domain from that)
    floor = 0.05 * float(np.max(np.hypot(Bx, By)))
    bnx, _, _ = normal_avg(Bx, 0)                  # x-faces: normal Bx
    _, ta, tb = normal_avg(By, 0)                  # tangential By either side
    rev_x = (np.sign(ta) * np.sign(tb) < 0) & (np.minimum(abs(ta), abs(tb)) > floor)
    bny, _, _ = normal_avg(By, 1)                  # y-faces: normal By
    _, ta, tb = normal_avg(Bx, 1)
    rev_y = (np.sign(ta) * np.sign(tb) < 0) & (np.minimum(abs(ta), abs(tb)) > floor)
    # "near": on the face, or on the face one cell along the sweep either side
    near = lambda r, ax: r | np.roll(r, 1, ax) | np.roll(r, -1, ax)
    return dict(neg=(bnx < 0, bny < 0), near=(near(rev_x, 0), near(rev_y, 1)))


def accumulate(rundir, snaps, nx, ny):
    """Attempted and failed counts per face and direction, per snapshot window."""
    t_after, dt = load_diag(rundir)
    nxt, nyt = nx + 2 * NG, ny + 2 * NG
    shapes = {0: ((nxt + 1, nyt), (slice(NG, NG + nx + 1), slice(NG, NG + ny))),
              1: ((nxt, nyt + 1), (slice(NG, NG + nx), slice(NG, NG + ny + 1)))}
    T = np.array([float(s["t"]) for s in snaps])
    K = T.size
    att = {d: np.zeros((K,) + ((nx + 1, ny) if d == 0 else (nx, ny + 1)), np.int32)
           for d in (0, 1)}
    fail = {d: np.zeros_like(att[d]) for d in (0, 1)}
    per_sweep = []
    files = sorted(glob.glob(os.path.join(rundir, "harvest", "coverage_*.npz")))
    if not files:
        raise SystemExit("no coverage files under %s/harvest" % rundir)
    for f in files:
        z = np.load(f)
        for r in range(z["sweep"].shape[0]):
            s, d, n = int(z["sweep"][r]), int(z["idir"][r]), int(z["n"][r])
            shp, phys = shapes[d]
            if n != shp[0] * shp[1]:
                raise SystemExit("sweep %d: n = %d does not fit a %s grid" % (s, n, shp))
            a = _unpack(z["attempted"][r], n)
            e = _unpack(z["exact"][r], n)
            per_sweep.append((s, d, int(a.sum()), int(e.sum())))
            if s // 6 >= t_after.size:
                continue
            ts = sweep_time(s, t_after, dt)
            k = np.flatnonzero(np.abs(T - ts) <= HALF_WINDOW)
            if k.size == 0:
                continue
            A = a.reshape(shp)[phys]
            Fl = (a & ~e).reshape(shp)[phys]
            att[d][k[0]] += A
            fail[d][k[0]] += Fl
    return T, att, fail, per_sweep


def summarise(T, att, fail, snaps):
    print("%-7s %-3s %9s %8s %7s %22s %11s" % (
        "t", "dir", "attempted", "failed", "rate", "B_n < 0: fails / att.",
        "enrichment"))
    rows = []
    for k, t in enumerate(T):
        ff = face_fields(snaps[k])
        for d, lab in ((0, "x"), (1, "y")):
            A, Fl = att[d][k], fail[d][k]
            nA, nF = int(A.sum()), int(Fl.sum())
            if nA == 0:
                continue
            neg, near = ff["neg"][d], ff["near"][d]
            share_f = Fl[near].sum() / max(nF, 1)
            share_a = A[near].sum() / max(nA, 1)
            enr = share_f / share_a if share_a > 0 else np.nan
            onneg = Fl[neg].sum() / max(nF, 1)
            attneg = A[neg].sum() / max(nA, 1)
            rows.append((t, lab, nA, nF, enr, onneg, attneg))
            print("%-7.3f %-3s %9d %8d %6.1f%% %10.1f%% / %6.1f%% %11.2f" % (
                t, lab, nA, nF, 100 * nF / nA, 100 * onneg, 100 * attneg, enr))
    return rows


def plot(T, att, fail, snaps, out, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    for k, t in enumerate(T):
        sn = snaps[k]
        x, y = sn["x"], sn["y"]
        dx, dy = x[1] - x[0], y[1] - y[0]
        ff = face_fields(sn)
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 6.0))
        for d, lab in ((0, "x-faces"), (1, "y-faces")):
            a = ax[d]
            a.pcolormesh(x, y, np.log10(sn["rho"]).T, cmap="Greys", shading="auto")
            # where the tangential field reverses: By = 0 across x-faces,
            # Bx = 0 across y-faces
            rv = np.nonzero(ff["near"][d])
            if d == 0:
                rx, ry = x[0] - 0.5 * dx + rv[0] * dx, y[rv[1]]
            else:
                rx, ry = x[rv[0]], y[0] - 0.5 * dy + rv[1] * dy
            a.scatter(rx, ry, s=26, marker="s", facecolors="none",
                      edgecolors="#2563eb", linewidths=0.5, alpha=0.6)
            A, Fl = att[d][k], fail[d][k]
            ii, jj = np.nonzero(Fl)
            if d == 0:
                fx, fy = x[0] - 0.5 * dx + ii * dx, y[jj]
            else:
                fx, fy = x[ii], y[0] - 0.5 * dy + jj * dy
            rate = Fl[ii, jj] / np.maximum(A[ii, jj], 1)
            neg = ff["neg"][d][ii, jj]
            a.scatter(fx[~neg], fy[~neg], s=10, c=rate[~neg], cmap="autumn_r",
                      vmin=0, vmax=1, edgecolors="none")
            a.scatter(fx[neg], fy[neg], s=12, facecolors="none",
                      edgecolors="#b91c1c", linewidths=0.6)
            a.set_aspect("equal")
            a.set_title("%s, t = %.2f: %d failed of %d attempted" % (
                lab, t, int(Fl.sum()), int(A.sum())), fontsize=10)
            a.set_xlabel("x"); a.set_ylabel("y")
        fig.suptitle("%s -- exact-solver failures (filled: rate; hollow: on "
                     "B_n < 0, possibly the pre-fix sign bug); blue squares: "
                     "faces near a real tangential-field reversal; over log density" % tag, fontsize=10)
        fig.tight_layout()
        f = os.path.join(out, "%s_t%.2f.png" % (tag, t))
        fig.savefig(f, dpi=120)
        plt.close(fig)
        print("wrote", f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rundir")
    ap.add_argument("--out", default="figs/failure_map")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    snaps = load_snaps(a.rundir)
    nx, ny = snaps[0]["rho"].shape
    T, att, fail, per_sweep = accumulate(a.rundir, snaps, nx, ny)
    tot_a = sum(p[2] for p in per_sweep)
    tot_e = sum(p[3] for p in per_sweep)
    print("%s: %dx%d, %d sweeps, attempted %d, exact %d (%.1f%%), failed %d"
          % (a.rundir, nx, ny, len(per_sweep), tot_a, tot_e,
             100 * tot_e / max(tot_a, 1), tot_a - tot_e))
    print("windows of +-%.3f around the %d snapshot times\n" % (HALF_WINDOW, T.size))
    summarise(T, att, fail, snaps)
    if not a.no_plots:
        plot(T, att, fail, snaps, a.out, os.path.basename(os.path.normpath(a.rundir)))


if __name__ == "__main__":
    main()
