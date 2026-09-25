"""The rotor's output, and which solver answered which cell.

Two figures per snapshot time:

* **the state** -- density, gas pressure, |B| and div(B), the last one as the
  check that constrained transport is doing its job;
* **the solver map** -- every physical face coloured by what actually
  produced its flux.  A run of this code is a HYBRID: most faces never reach
  the exact solver at all (the jump is too weak to matter, or the fan is
  entirely upwind), and of those that do, some are answered by the seven-wave
  Newton, some by the five-wave planar rescue, some by the reduced three-wave
  one, and the rest fall back to HLLD.  That mixture is usually quoted as a
  single percentage; drawn on the grid it says something a percentage cannot,
  namely WHERE each solver earns its place.

The map is built from the coverage bitmaps in the harvest
(`src/physics/harvest.py::record_coverage`), which are written per sweep on
the ghost-inclusive face grid and mapped back to (x, y) exactly as
`failure_map_2d.py` does -- sweep s is step s // 6, RK stage (s % 6) // 2,
direction s % 2, and the counts are accumulated in a window around each
snapshot time.

Runs harvested before the provenance bitmaps existed carry only `attempted`
and `exact`; the map then degrades to three categories and the caption says
so.  An ordinary HLLD run has no harvest at all, so only the state figure is
drawn.

    python scripts/plot_rotor_output.py results/rotor_64_exact --out figs/rotor
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import failure_map_2d as FM                                    # noqa: E402

# drawn in this order, first match wins: a face answered by the seven-wave
# solver is never also "fell back", and a face that was never attempted is
# labelled by the gate that stopped it
CATEGORIES = [
    ("seven", "exact: seven-wave", "#1d4ed8"),
    ("planar5", "exact: five-wave planar", "#0891b2"),
    ("three_wave", "three-wave rescue", "#7c3aed"),
    ("degenerate", "degenerate class", "#a16207"),
    ("failed", "attempted, fell back to HLLD", "#dc2626"),
    ("routed", "routed past the solver", "#db2777"),
    ("weak", "not attempted: weak jump", "#d1d5db"),
    ("upwind", "not attempted: upwind fan", "#9ca3af"),
    ("bad", "not attempted: bad state", "#111827"),
    ("exact", "exact (older harvest: solver not recorded)", "#1d4ed8"),
]


def load_meta(rundir):
    """The run's stamp, or what can be recovered without one.

    Runs made before `run_meta.json` existed carry the flux in their
    directory name and nothing about the limiter; say so rather than
    printing a question mark that looks like a bug.
    """
    f = os.path.join(rundir, "run_meta.json")
    if os.path.exists(f):
        return json.load(open(f))
    name = os.path.basename(os.path.normpath(rundir))
    flux = next((k for k in ("exact", "hlld", "hlle", "hllc")
                 if k in name.split("_")), "?")
    return dict(solver=flux, limiter="unstamped (mc in every run "
                                     "before 2026-09-25)", unstamped=True)


def coverage_maps(rundir, snaps, nx, ny, ng):
    """Per snapshot time and direction, the count of sweeps in which each
    physical face fell into each category."""
    files = sorted(glob.glob(os.path.join(rundir, "harvest", "coverage_*.npz")))
    if not files:
        return None, []
    t_after, dt = FM.load_diag(rundir)
    T = np.array([float(s["t"]) for s in snaps])
    nxt, nyt = nx + 2 * ng, ny + 2 * ng
    shapes = {0: ((nxt + 1, nyt), (slice(ng, ng + nx + 1), slice(ng, ng + ny))),
              1: ((nxt, nyt + 1), (slice(ng, ng + nx), slice(ng, ng + ny + 1)))}
    have = [k for k in np.load(files[0]).files
            if k in {c[0] for c in CATEGORIES}]
    keys = [k for k in ("seven", "planar5", "three_wave", "degenerate",
                        "routed", "weak", "upwind", "bad") if k in have]
    legacy = "seven" not in have
    out = {d: {k: np.zeros((T.size,) + ((nx + 1, ny) if d == 0 else (nx, ny + 1)),
                           np.int32)
               for k in keys + ["attempted", "exact", "failed"]}
           for d in (0, 1)}
    for f in files:
        z = np.load(f)
        for r in range(z["sweep"].shape[0]):
            s, d, n = int(z["sweep"][r]), int(z["idir"][r]), int(z["n"][r])
            shp, phys = shapes[d]
            if n != shp[0] * shp[1]:
                raise SystemExit("sweep %d: n = %d does not fit %s (ng = %d)"
                                 % (s, n, shp, ng))
            if s // 6 >= t_after.size:
                continue
            k = np.flatnonzero(np.abs(T - FM.sweep_time(s, t_after, dt))
                               <= FM.HALF_WINDOW)
            if k.size == 0:
                continue
            att = FM._unpack(z["attempted"][r], n)
            ex = FM._unpack(z["exact"][r], n)
            out[d]["attempted"][k[0]] += att.reshape(shp)[phys]
            out[d]["exact"][k[0]] += ex.reshape(shp)[phys]
            out[d]["failed"][k[0]] += (att & ~ex).reshape(shp)[phys]
            for key in keys:
                out[d][key][k[0]] += FM._unpack(z[key][r], n).reshape(shp)[phys]
    return out, (["exact", "failed"] if legacy
                 else [k for k in ("seven", "planar5", "three_wave",
                                   "degenerate", "failed", "routed", "weak",
                                   "upwind", "bad") if k in out[0]])


def face_xy(x, y, d, ii, jj):
    dx, dy = x[1] - x[0], y[1] - y[0]
    if d == 0:
        return x[0] - 0.5 * dx + ii * dx, y[jj]
    return x[ii], y[0] - 0.5 * dy + jj * dy


def plot_state(snap, meta, out, tag):
    import matplotlib.pyplot as plt
    x, y = snap["x"], snap["y"]
    magB = np.hypot(snap["Bx"], snap["By"])
    fields = [("rho", snap["rho"], "density", "viridis"),
              ("p", snap["p"], "gas pressure", "magma"),
              ("magB", magB, "|B|", "cividis"),
              ("divB", np.abs(snap["divB"]), "|div B|", "Greys")]
    fig, ax = plt.subplots(2, 2, figsize=(11.0, 9.5))
    for a, (k, q, lab, cm) in zip(ax.ravel(), fields):
        if k == "divB":
            q = np.log10(np.maximum(q, 1e-18))
            lab = "log10 |div B|"
        m = a.pcolormesh(x, y, q.T, cmap=cm, shading="auto")
        fig.colorbar(m, ax=a, fraction=0.046)
        a.set_aspect("equal")
        a.set_title(lab, fontsize=10)
        a.set_xlabel("x"); a.set_ylabel("y")
    fig.suptitle("%s   t = %.3f   %d^2, %s flux, %s reconstruction"
                 % (tag, float(snap["t"]), snap["rho"].shape[0],
                    meta.get("solver", "?"), meta.get("limiter", "?")),
                 fontsize=11)
    fig.tight_layout()
    f = os.path.join(out, "%s_state_t%.2f.png" % (tag, float(snap["t"])))
    fig.savefig(f, dpi=120)
    plt.close(fig)
    return f


def plot_solver_map(snap, cov, order, k, meta, out, tag):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colour = {c[0]: c[2] for c in CATEGORIES}
    label = {c[0]: c[1] for c in CATEGORIES}
    x, y = snap["x"], snap["y"]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.4))
    shares = {}
    for d, name in ((0, "x-faces"), (1, "y-faces")):
        a = ax[d]
        a.pcolormesh(x, y, np.log10(snap["rho"]).T, cmap="Greys",
                     shading="auto", alpha=0.55)
        # first match wins, so a face is drawn as the strongest thing that
        # happened to it rather than once per category
        claimed = np.zeros_like(cov[d]["attempted"][k], dtype=bool)
        for key in order:
            m = (cov[d][key][k] > 0) & ~claimed
            claimed |= m
            ii, jj = np.nonzero(m)
            shares[key] = shares.get(key, 0) + int(m.sum())
            if ii.size:
                fx, fy = face_xy(x, y, d, ii, jj)
                a.scatter(fx, fy, s=5, c=colour[key], edgecolors="none")
        a.set_aspect("equal")
        a.set_title("%s" % name, fontsize=10)
        a.set_xlabel("x"); a.set_ylabel("y")
    nx, ny = snap["rho"].shape
    tot = (nx + 1) * ny + nx * (ny + 1)          # every physical face
    drawn = sum(shares.values())
    if drawn < tot:
        shares["untouched"] = tot - drawn
    handles = [Line2D([], [], marker="o", ls="", color=colour[key],
                      label="%s  %.1f%%" % (label[key], 100 * shares[key] / tot))
               for key in order if shares.get(key)]
    if shares.get("untouched"):
        handles.append(Line2D([], [], marker="o", ls="", color="#e5e7eb",
                              label="not attempted  %.1f%%"
                                    % (100 * shares["untouched"] / tot)))
    ax[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                 fontsize=8, frameon=False)
    fig.suptitle("%s   t = %.3f   which solver answered each face   "
                 "(%s flux, %s reconstruction)"
                 % (tag, float(snap["t"]), meta.get("solver", "?"),
                    meta.get("limiter", "?")), fontsize=11)
    fig.tight_layout()
    f = os.path.join(out, "%s_solver_t%.2f.png" % (tag, float(snap["t"])))
    fig.savefig(f, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f, shares


def plot_shares(hist, order, out, tag):
    """The map's quantitative twin: the share of faces per category, in time."""
    import matplotlib.pyplot as plt
    colour = dict({c[0]: c[2] for c in CATEGORIES}, untouched="#e5e7eb")
    label = dict({c[0]: c[1] for c in CATEGORIES}, untouched="not attempted")
    T = sorted(hist)
    order = list(order) + ["untouched"]
    fig, a = plt.subplots(figsize=(8.5, 5.0))
    bottom = np.zeros(len(T))
    for key in order:
        v = np.array([100.0 * hist[t].get(key, 0) / max(sum(hist[t].values()), 1)
                      for t in T])
        if v.max() <= 0:
            continue
        a.bar(range(len(T)), v, bottom=bottom, color=colour[key],
              label=label[key], width=0.75)
        bottom += v
    a.set_xticks(range(len(T)))
    a.set_xticklabels(["%.2f" % t for t in T])
    a.set_xlabel("t"); a.set_ylabel("share of physical faces (%)")
    a.set_title("%s: what answered the interfaces, over the run" % tag,
                fontsize=10)
    a.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    f = os.path.join(out, "%s_solver_shares.png" % tag)
    fig.savefig(f, dpi=130)
    plt.close(fig)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rundir")
    ap.add_argument("--out", default="figs/rotor_output")
    ap.add_argument("--times", help="comma-separated snapshot times")
    ap.add_argument("--no-state", action="store_true")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    snaps = FM.load_snaps(a.rundir)
    if a.times:
        want = [float(v) for v in a.times.split(",")]
        # the run saves on the first step past each target, so 0.2 is stored
        # as 0.201 -- match loosely or --times silently drops snapshots
        snaps = [s for s in snaps
                 if any(abs(float(s["t"]) - w) < 5e-3 for w in want)]
    if not snaps:
        raise SystemExit("no snapshots in %s" % a.rundir)
    os.makedirs(a.out, exist_ok=True)
    meta = load_meta(a.rundir)
    tag = os.path.basename(os.path.normpath(a.rundir))
    nx, ny = snaps[0]["rho"].shape
    ng = FM.run_ghosts(a.rundir)
    if not a.no_state:
        for s in snaps:
            print("wrote", plot_state(s, meta, a.out, tag))
    cov, order = coverage_maps(a.rundir, snaps, nx, ny, ng)
    if cov is None:
        print("no harvest under %s: state figures only (an HLLD run records "
              "no per-face provenance)" % a.rundir)
        return
    if "seven" not in order:
        print("this harvest predates the per-solver bitmaps: the map shows "
              "exact / fell back / not attempted only")
    hist = {}
    for k, s in enumerate(snaps):
        f, shares = plot_solver_map(s, cov, order, k, meta, a.out, tag)
        hist[float(s["t"])] = shares
        print("wrote", f)
    print("wrote", plot_shares(hist, order, a.out, tag))


if __name__ == "__main__":
    main()
