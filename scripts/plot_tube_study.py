#!/usr/bin/env python3
"""Figures and tables of the 1D flux studies written by scripts/tube_study.py.

    python scripts/plot_tube_study.py results/tube_study --out figs/tube_study

Needs matplotlib, which calea's venv does not have: fetch results/tube_study
(runs/ and snaps/slow*) to the laptop and plot there.

tube_convergence  L1(rho) against the exact self-similar solution vs cells,
                  per tube, four fluxes, first order dashed / second solid;
                  below, each flux's error over HLLD's (geometric mean of
                  the ratio over the seven variables).
standing_profiles the density across the standing / slowly moving slow shock
                  at the final time, in cells from the exact shock position.
standing_history  per snapshot: the shock's width (cells with 0.1 < phi < 0.9
                  within 50 cells of it) and the largest deviation 5-60
                  cells downstream (the post-shock oscillations).
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

SOLVERS = ("hlle", "hllc", "hlld", "exact")
NAME = {"hlle": "HLLE", "hllc": "HLLC", "hlld": "HLLD", "exact": "exact"}
COL = {"hlle": "#8C8C8C", "hllc": "#7B3294", "hlld": "#D9822B", "exact": "#2E6FAC"}
MARK = {"hlle": "v", "hllc": "^", "hlld": "o", "exact": "s"}
TUBE = {"b5": "Balsara 5", "sst": "slow-shock tube",
        "sst0": "slow-shock tube, narrow band"}
VARS = ("rho", "p", "vx", "vy", "vz", "By", "Bz")


def load(out):
    R = {}
    for f in sorted(glob.glob(os.path.join(out, "runs", "*.npz"))):
        d = dict(np.load(f, allow_pickle=True))
        d = {k: (v.item() if isinstance(v, np.ndarray) and v.shape == () else v)
             for k, v in d.items()}
        R[(d["case"], d["solver"] + str(d.get("suffix") or ""), d["limiter"],
           int(d["ncells"]))] = d
    return R


def tube_convergence(R, out):
    tubes = [c for c in ("b5", "sst", "sst0") if any(k[0] == c for k in R)]
    if not tubes:
        return
    fig, ax = plt.subplots(2, len(tubes), figsize=(5.2 * len(tubes), 7.6),
                           squeeze=False, gridspec_kw=dict(height_ratios=(1.6, 1)))
    for j, case in enumerate(tubes):
        a, b = ax[0, j], ax[1, j]
        for lim, ls, mfc in (("pcm", "--", "white"), ("mc", "-", None)):
            for sol in SOLVERS:
                Ns = sorted(k[3] for k in R if k[:3] == (case, sol, lim) and "L1_rho" in R[k])
                if not Ns:
                    continue
                e = [R[(case, sol, lim, n)]["L1_rho"] for n in Ns]
                a.loglog(Ns, e, ls, marker=MARK[sol], color=COL[sol], mfc=mfc or COL[sol],
                         lw=1.8, ms=6, label="%s, %s" % (NAME[sol], "1st" if lim == "pcm" else "2nd"))
                if sol == "hlld":
                    continue
                # against HLLD: the geometric mean of L1/L1(HLLD) over the
                # seven variables -- density alone swings by a few per cent
                # with where an unresolved structure falls on the grid
                rx, ry = [], []
                for n in Ns:
                    h = R.get((case, "hlld", lim, n))
                    if h is None or "L1_rho" not in h:
                        continue
                    d = R[(case, sol, lim, n)]
                    rx.append(n)
                    ry.append(float(np.exp(np.mean([np.log(d["L1_" + v] / h["L1_" + v])
                                                    for v in VARS]))))
                if rx:
                    b.semilogx(rx, ry, ls, marker=MARK[sol], color=COL[sol],
                               mfc=mfc or COL[sol], lw=1.8, ms=6)
        a.set_title(TUBE.get(case, case), fontsize=13)
        a.set_ylabel("L1(ρ) vs exact solution" if j == 0 else "")
        a.grid(alpha=0.3, which="both")
        b.axhline(1.0, color="k", lw=0.8)
        b.set_xlabel("cells")
        b.set_ylabel("L1 / L1(HLLD), mean\nover the 7 variables" if j == 0 else "")
        b.grid(alpha=0.3, which="both")
    ax[0, 0].legend(fontsize=8.5, ncol=2, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out + "/tube_convergence." + ext, dpi=150)
    plt.close(fig)


def shock_metrics(snap, d):
    """(t, width, downstream noise, upstream noise) of a snapshot.

    phi = (rho - rho_up) / (rho_dn - rho_up), 0 -> 1 across the exact shock.
    width: cells with 0.1 < phi < 0.9 within 50 cells of the exact shock
    position (the transition itself, not the noise behind it); noise: the
    largest |phi - 1| 5 to 60 cells downstream (|phi| upstream), away from
    the outflow boundaries."""
    t = float(snap["t"])
    x = snap["x"]
    dx = x[1] - x[0]
    xs = d["x0"] + d["s"] * t
    phi = (snap["prim_rho"] - d["up_rho"]) / (d["dn_rho"] - d["up_rho"])
    r = (x - xs) / dx
    near = np.abs(r) <= 50
    width = int(np.count_nonzero(near & (phi > 0.1) & (phi < 0.9)))
    dn = (r > 5) & (r < 60) & (x < x[-1] - 10 * dx)
    up = (r < -5) & (r > -60) & (x > x[0] + 10 * dx)
    dev_dn = float(np.max(np.abs(phi[dn] - 1.0))) if dn.any() else np.nan
    dev_up = float(np.max(np.abs(phi[up]))) if up.any() else np.nan
    return t, width, dev_dn, dev_up


def snaps_of(out, tag):
    return sorted(glob.glob(os.path.join(out, "snaps", tag, "snap_*.npz")))


def standing(R, out, src):
    shocks = sorted({k[0] for k in R if R[k].get("kind") == "shock"})
    if not shocks:
        return
    names = sorted({c.split(":")[0] for c in shocks})
    speeds = sorted({float(c.split(":")[1]) for c in shocks})
    for name in names:
        fig, ax = plt.subplots(len(speeds), 2, figsize=(11, 3.2 * len(speeds)), squeeze=False,
                               sharex=True)
        for i, s in enumerate(speeds):
            case = "%s:%s" % (name, ("%g" % s))
            for j, lim in enumerate(("pcm", "mc")):
                a = ax[i, j]
                for sol in SOLVERS:
                    d = R.get((case, sol, lim, 200))
                    if d is None:
                        continue
                    dx = d["x"][1] - d["x"][0]
                    xs = d["x0"] + d["s"] * d["t_end"]
                    r = (d["x"] - xs) / dx
                    phi = (d["rho"] - d["up_rho"]) / (d["dn_rho"] - d["up_rho"])
                    m = np.abs(r) < 16
                    a.plot(r[m], phi[m], marker=MARK[sol], color=COL[sol], lw=1.2, ms=5,
                           mfc="white" if sol != "exact" else COL[sol], label=NAME[sol])
                rr = np.linspace(-16, 16, 641)
                a.plot(rr, (rr > 0).astype(float), color="k", lw=1.0, zorder=0, label="exact solution")
                a.set_title("%s, s = %g, %s" % (name, s, "1st order" if lim == "pcm" else "2nd order (MC)"),
                            fontsize=11)
                a.grid(alpha=0.3)
                if j == 0:
                    a.set_ylabel("(ρ − ρ_up) / (ρ_dn − ρ_up)")
        for a in ax[-1]:
            a.set_xlabel("cells from the exact shock position")
        ax[0, 0].legend(fontsize=8.5)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out + "/standing_profiles_%s.%s" % (name, ext), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(2, len(speeds), figsize=(4.6 * len(speeds), 6.4), squeeze=False)
        for i, s in enumerate(speeds):
            case = "%s:%s" % (name, ("%g" % s))
            for lim, ls in (("pcm", "--"), ("mc", "-")):
                for sol in SOLVERS:
                    d = R.get((case, sol, lim, 200))
                    if d is None:
                        continue
                    tag = "%s_%s_%s_200" % (case.replace(":", "_s"), sol, lim)
                    M = np.array([shock_metrics(np.load(f), d) for f in snaps_of(src, tag)])
                    if M.size == 0:
                        continue
                    ax[0, i].semilogy(M[:, 0], np.maximum(M[:, 1], 0.5), ls, color=COL[sol], lw=1.6,
                                  label="%s, %s" % (NAME[sol], "1st" if lim == "pcm" else "2nd"))
                    ax[1, i].semilogy(M[:, 0], np.maximum(M[:, 2], 1e-16), ls, color=COL[sol], lw=1.6)
            ax[0, i].set_title("%s, s = %g" % (name, s), fontsize=11)
            ax[0, i].set_ylabel("shock width, cells (0.1 < φ < 0.9)\n0 drawn at 0.5" if i == 0 else "")
            ax[1, i].set_ylabel("max |φ − 1|, 5–60 cells downstream" if i == 0 else "")
            ax[1, i].set_xlabel("t")
            for a in ax[:, i]:
                a.grid(alpha=0.3)
        ax[0, 0].legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out + "/standing_history_%s.%s" % (name, ext), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.src, "figs")
    os.makedirs(out, exist_ok=True)
    R = load(a.src)
    print("%d runs" % len(R))
    tube_convergence(R, out)
    standing(R, out, a.src)
    print("wrote figures to %s" % out)


if __name__ == "__main__":
    main()
