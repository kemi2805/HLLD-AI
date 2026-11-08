"""The four-quadrant 64^2 exact runs side by side (2026-11-02, retrain loop;
user: "rerun 4Q with 64^2 and measure the failed faces and walltime").

    python r2d_compare64.py OUTDIR name=RUNDIR [name=RUNDIR ...] --ref REF.npz [--base NAME]

RUNDIR (fetch_runs64.sh) holds run_meta.json, diag.csv, snap_*.npz,
harvest/coverage_*.npz and slurm.out.  Per run: the faces lost (attempted,
not exact, given HLLD -- run_meta's hlld_after_attempt; the maps count the
physical faces only), which rung answered the rest, the wall clock, and
L1(rho) at t = 0.4 against the 512^2 HLLD reference averaged over its 8x8
blocks (make_2d.py's measure).  Figures, for Luciano's standing request
(WHERE a run fails):
  lost_maps.png   lost faces per face, by time window, every run, plus the
                  difference to --base (staggered grid: x-faces between the
                  cells along x, y-faces along y), density contours of the
                  snapshot nearest the window's middle
  error_maps.png  |rho - reference| at t = 0.4 per run, and rho - rho(base)
  wall.png        seconds per step and cumulative hours against t
A run still in progress is read up to its last finished step (stated).

Any N (2026-11-03, the 128^2 / 512^2 ft3 runs): the titles follow the grid,
the reference is block-averaged by 512/N, and --coarsen K (default N/128,
at least 1) sums the lost-face maps over KxK cells for display -- a 512^2
face is smaller than a pixel.  The difference row needs runs on one grid.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm

WINDOWS = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4))
LINE = {0: "#7f7f7f", 1: "#0072B2", 2: "#D55E00", 3: "#009E73"}      # fixed order (Okabe-Ito)


def unpack(bits, n):
    return np.unpackbits(bits, axis=-1)[..., :n].astype(bool)


def sweep_time(s, t_after, dt):
    """failure_map_2d.sweep_time: SSP-RK3, three stages x two directions."""
    step = s // 6
    return t_after[step] - dt[step] + (0.0, 1.0, 0.5)[(s % 6) // 2] * dt[step]


def load_run(d):
    meta = json.load(open(os.path.join(d, "run_meta.json")))
    diag = np.genfromtxt(os.path.join(d, "diag.csv"), delimiter=",", names=True)
    t_after, dt = np.atleast_1d(diag["t"]), np.atleast_1d(diag["dt"])
    snaps = [dict(np.load(f, allow_pickle=True)) for f in sorted(glob.glob(os.path.join(d, "snap_*.npz")))]
    nx, ny = snaps[0]["rho"].shape
    ng = int(meta.get("ng", 2))
    nxt, nyt = nx + 2 * ng, ny + 2 * ng
    shapes = {0: ((nxt + 1, nyt), (slice(ng, ng + nx + 1), slice(ng, ng + ny))),
              1: ((nxt, nyt + 1), (slice(ng, ng + nx), slice(ng, ng + ny + 1)))}
    lost = np.zeros((len(WINDOWS), 2 * nx + 1, 2 * ny + 1), np.int32)
    keys = ("attempted", "exact", "lost", "seven", "planar5", "planar4")
    phys = {k: 0 for k in keys}
    nsweep = 0
    for f in sorted(glob.glob(os.path.join(d, "harvest", "coverage_*.npz"))):
        z = np.load(f)
        for r in range(z["sweep"].shape[0]):
            s, idir, n = int(z["sweep"][r]), int(z["idir"][r]), int(z["n"][r])
            if s // 6 >= t_after.size:            # a step the diagnostics have not closed
                continue
            shp, sl = shapes[idir]
            m = {k: unpack(z[k][r], n).reshape(shp)[sl] for k in ("attempted", "exact", "seven", "planar5",
                                                              "planar4", "linear_fallback")}
            m["lost"] = m["attempted"] & ~m["exact"] & ~m["linear_fallback"]
            for k in keys:
                phys[k] += int(m[k].sum())
            t = sweep_time(s, t_after, dt)
            w = min(int(t / 0.1), len(WINDOWS) - 1)
            if idir == 0:
                lost[w, 0::2, 1::2] += m["lost"]
            else:
                lost[w, 1::2, 0::2] += m["lost"]
            nsweep += 1
    # per-step wall clock from the log: "step N t=... (Ns)", cumulative
    wall = []
    for line in open(os.path.join(d, "slurm.out"), errors="replace"):
        g = re.match(r"\s*step\s+(\d+) t=([0-9.]+).*\((\d+)s\)\s*$", line)
        if g:
            wall.append((int(g.group(1)), float(g.group(2)), float(g.group(3))))
    wall = np.array(wall) if wall else np.zeros((0, 3))
    done = "finished" in meta and "wall_s" in meta
    return dict(meta=meta, t=t_after, snaps=snaps, lost=lost, phys=phys, nsweep=nsweep,
                wall=wall, steps=t_after.size, done=bool(done), nx=nx, ny=ny)


def l1(rho, ref):
    k = ref.shape[0] // rho.shape[0]
    r = ref.reshape(rho.shape[0], k, rho.shape[1], k).mean(axis=(1, 3))
    return np.abs(rho - r).mean() / r.mean(), r


def coarse(L, k):
    """A (2nx+1, 2ny+1) staggered map summed over kxk cells (2k x 2k entries)."""
    if k == 1:
        return L
    a = L[:-1, :-1]
    return a.reshape(a.shape[0] // (2 * k), 2 * k, a.shape[1] // (2 * k), 2 * k).sum(axis=(1, 3))


def nearest_snap(run, t):
    return min(run["snaps"], key=lambda s: abs(float(s["t"]) - t))


def extent(s):
    x, y = s["x"], s["y"]
    dx, dy = x[1] - x[0], y[1] - y[0]
    return [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2]


def contours(ax, s, color="0.35"):
    lr = np.log10(np.maximum(s["rho"], 1e-6))
    # levels between the initial states' values (0.1, 1, ...): a level AT a flat
    # quadrant's density draws its round-off as noise
    ax.contour(s["x"], s["y"], lr.T, levels=np.arange(-1.375, 0.6, 0.25), colors=color,
               linewidths=0.35, alpha=0.6)


def table(runs, names, ref):
    rows = []
    for nm in names:
        r = runs[nm]
        m = r["meta"]
        f = m.get("harvest", {}).get("faces", {}) if r["done"] else {}
        p = r["phys"]                      # a run in progress: its coverage so far
        att = f.get("attempted", p["attempted"])
        ex = f.get("exact", p["exact"])
        lost = f.get("hlld_after_attempt", p["lost"])
        fin = [s for s in r["snaps"] if abs(float(s["t"]) - 0.4) < 1e-9]
        L = l1(fin[0]["rho"], ref)[0] if fin else None
        w = r["wall"]
        rows.append(dict(name=nm, done=r["done"], steps=r["steps"],
                         attempted=att, exact=ex, lost=lost,
                         seven=f.get("seven", p["seven"]), planar5=f.get("planar5", p["planar5"]),
                         planar4=f.get("planar4", p["planar4"]), phys=p,
                         wall_s=m["wall_s"] if r["done"] else (float(w[-1, 2]) if len(w) else None),
                         s_per_step=m.get("s_per_step") if r["done"] else None, L1_rho=L))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+", help="name=RUNDIR")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--base", default=None, help="the run the others are differenced against")
    ap.add_argument("--label", nargs="*", default=[], help="name=label for the figures")
    ap.add_argument("--coarsen", type=int, default=None,
                    help="sum the lost maps over KxK cells (default N/128, at least 1)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = [s.split("=", 1)[0] for s in a.runs]
    runs = {s.split("=", 1)[0]: load_run(s.split("=", 1)[1]) for s in a.runs}
    label = dict(s.split("=", 1) for s in a.label)
    lab = lambda nm: label.get(nm, nm)
    base = a.base or names[0]
    ref = np.load(a.ref, allow_pickle=True)["rho"]

    rows = table(runs, names, ref)
    print("%-8s %5s %10s %10s %7s %8s %8s %7s %9s %8s %8s %8s" % (
        "run", "steps", "attempted", "exact", "lost", "solved%", "seven", "planar5", "planar4",
        "wall_h", "s/step", "L1(rho)"))
    for r in rows:
        pc = 100.0 * r["exact"] / r["attempted"] if r["attempted"] else float("nan")
        print("%-8s %5d %10s %10s %7s %8.3f %8s %7s %9s %8.2f %8s %8s%s" % (
            r["name"], r["steps"], r["attempted"], r["exact"], r["lost"], pc, r["seven"], r["planar5"],
            r["planar4"], (r["wall_s"] or 0) / 3600.0,
            "%.1f" % r["s_per_step"] if r["s_per_step"] else "-",
            "%.4f" % r["L1_rho"] if r["L1_rho"] is not None else "-",
            "" if r["done"] else "   (running: %d steps so far)" % r["steps"]))
        p = r["phys"]
        print("         physical faces: attempted %d, exact %d, lost %d" % (p["attempted"], p["exact"], p["lost"]))
    json.dump(rows, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)

    # --- where the faces are lost, by time window
    cg = {nm: a.coarsen or max(1, runs[nm]["nx"] // 128) for nm in names}
    lostc = {nm: np.stack([coarse(L, cg[nm]) for L in runs[nm]["lost"]]) for nm in names}
    others = [nm for nm in names if nm != base]
    diff = bool(others) and lostc[others[-1]].shape == lostc[base].shape
    n = len(names) + (1 if diff else 0)
    fig, ax = plt.subplots(n, len(WINDOWS), figsize=(3.6 * len(WINDOWS), 3.55 * n),
                           constrained_layout=True, squeeze=False)
    vmax = max(int(L.max()) for L in lostc.values()) or 1
    norm = LogNorm(vmin=1, vmax=vmax)
    s0 = runs[names[0]]["snaps"][0]
    ext = extent(s0)
    for i, nm in enumerate(names):
        r = runs[nm]
        for j, (t0, t1) in enumerate(WINDOWS):
            L = np.ma.masked_equal(lostc[nm][j], 0)
            im = ax[i, j].imshow(L.T, origin="lower", extent=ext, cmap="Oranges", norm=norm,
                                 interpolation="nearest")
            contours(ax[i, j], nearest_snap(r, 0.5 * (t0 + t1)))
            ax[i, j].set_title("%s, t %.1f-%.1f: %d lost" % (lab(nm), t0, t1, int(r["lost"][j].sum())),
                               fontsize=9.5)
    cb = fig.colorbar(im, ax=ax[:len(names), :].ravel().tolist(), shrink=0.6, pad=0.01)
    k = max(cg.values())
    cb.set_label("RK stages in which the face was lost (HLLD)"
                 + ("" if k == 1 else ", summed over %dx%d cells" % (k, k)))
    if diff:
        nm = others[-1]
        dm = max(1, int(np.abs(lostc[nm] - lostc[base]).max()))
        for j, (t0, t1) in enumerate(WINDOWS):
            D = (lostc[nm][j] - lostc[base][j]).astype(float)
            imd = ax[-1, j].imshow(np.ma.masked_equal(D, 0).T, origin="lower", extent=ext, cmap="RdBu_r",
                                   norm=TwoSlopeNorm(0, -dm, dm), interpolation="nearest")
            contours(ax[-1, j], nearest_snap(runs[nm], 0.5 * (t0 + t1)))
            ax[-1, j].set_title("%s - %s, t %.1f-%.1f: %+d" % (lab(nm), lab(base), t0, t1, int(D.sum())),
                                fontsize=9.5)
        cbd = fig.colorbar(imd, ax=ax[-1, :].tolist(), shrink=0.8, pad=0.01)
        cbd.set_label("lost: more (red) / fewer (blue)")
    for x in ax.ravel():
        x.set_xticks([])
        x.set_yticks([])
    grids = "/".join(sorted({"%d²" % runs[nm]["nx"] for nm in names}, key=lambda g: int(g[:-1])))
    fig.suptitle("Four-quadrant problem, %s, exact flux: faces lost to HLLD (contours: log10 ρ)" % grids,
                 fontsize=12)
    fig.savefig(os.path.join(a.out, "lost_maps.png"), dpi=150)
    plt.close(fig)

    # --- where the solution error lies at t = 0.4
    fin = {nm: [s for s in runs[nm]["snaps"] if abs(float(s["t"]) - 0.4) < 1e-9] for nm in names}
    have = [nm for nm in names if fin[nm]]
    if have:
        dpanel = (base in have and len(have) > 1 and fin[base][0]["rho"].shape
                  == fin[[x for x in have if x != base][-1]][0]["rho"].shape)
        cols = len(have) + (1 if dpanel else 0)
        fig, ax = plt.subplots(1, cols, figsize=(4.0 * cols, 4.0), constrained_layout=True, squeeze=False)
        errs = {nm: np.abs(fin[nm][0]["rho"] - l1(fin[nm][0]["rho"], ref)[1]) for nm in have}
        em = max(e.max() for e in errs.values())
        for j, nm in enumerate(have):
            im = ax[0, j].imshow(errs[nm].T, origin="lower", extent=ext, cmap="Purples", vmin=0, vmax=em,
                                 interpolation="nearest")
            contours(ax[0, j], fin[nm][0])
            ax[0, j].set_title("|ρ − ref|, %s\nL1 = %.4f" % (lab(nm), l1(fin[nm][0]["rho"], ref)[0]), fontsize=10)
        kb = ref.shape[0] // runs[have[0]]["nx"]
        fig.colorbar(im, ax=ax[0, :len(have)].tolist(), shrink=0.8, pad=0.01).set_label(
            "|ρ − ρ(512² HLLD)|" if kb == 1 else "|ρ − ρ(512², %d×%d mean)|" % (kb, kb))
        if dpanel:
            nm = [x for x in have if x != base][-1]
            D = fin[nm][0]["rho"] - fin[base][0]["rho"]
            dm = np.abs(D).max() or 1.0
            imd = ax[0, -1].imshow(D.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-dm, vmax=dm,
                                   interpolation="nearest")
            contours(ax[0, -1], fin[nm][0])
            ax[0, -1].set_title("ρ(%s) − ρ(%s)\nmax |Δρ| = %.2e" % (lab(nm), lab(base), dm), fontsize=10)
            fig.colorbar(imd, ax=ax[0, -1], shrink=0.8, pad=0.01).set_label("Δρ")
        for x in ax.ravel():
            x.set_xticks([])
            x.set_yticks([])
        fig.savefig(os.path.join(a.out, "error_maps.png"), dpi=150)
        plt.close(fig)

    # --- the wall clock
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for i, nm in enumerate(names):
        w = runs[nm]["wall"]
        if len(w) < 2:
            continue
        per = np.diff(w[:, 2])
        k = 5
        sm = np.convolve(per, np.ones(k) / k, mode="valid")
        c = LINE[i % len(LINE)]
        ls = "--" if i == 0 and len(names) > 2 else "-"
        ax[0].plot(w[1 + k // 2:len(per) + 1 - k // 2, 1], sm, color=c, lw=2, ls=ls, label=lab(nm))
        ax[1].plot(w[:, 1], w[:, 2] / 3600.0, color=c, lw=2, ls=ls, label=lab(nm))
        ax[1].annotate("%.2f h" % (w[-1, 2] / 3600.0), (w[-1, 1], w[-1, 2] / 3600.0), xytext=(4, 0),
                       textcoords="offset points", fontsize=9, color="0.2", va="center")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("seconds per step (5-step mean)")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("wall clock (hours)")
    for x in ax:
        x.grid(alpha=0.25, lw=0.6)
        x.spines[["top", "right"]].set_visible(False)
    ax[1].legend(frameon=False, loc="upper left")
    fig.savefig(os.path.join(a.out, "wall.png"), dpi=150)
    plt.close(fig)
    print("wrote", ", ".join(os.path.join(a.out, f) for f in ("summary.json", "lost_maps.png", "error_maps.png",
                                                            "wall.png")))


if __name__ == "__main__":
    main()
