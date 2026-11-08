#!/usr/bin/env python
"""Figures of the anatomy study of the four-quadrant 128^2 exact run with ft3
(2026-11-04).  Reads data/sample.npz (face_table.py + sample_table.py on
calea), data/cond.npz (conditioning.py), the run's snapshots and coverage
(../runs64/ft3_128) and, when present, data/seed_ceiling.npz and the ledger.

    python anatomy_figs.py            -> figs/*.png, figs/numbers.json
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from feats import NAMES, load, matrix                       # noqa: E402
from r2d_compare64 import unpack                            # noqa: E402

RUN = os.path.join(os.path.dirname(HERE), "runs64", "ft3_128")
OUT = os.path.join(HERE, "figs")
# one identity colour per class, fixed order (Okabe-Ito, as the other figures)
COL = ["#999999", "#CC79A7", "#0072B2", "#009E73", "#D55E00"]
RAMP = ["Greys", "RdPu", "Blues", "Greens", "Oranges"]
WIN = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4))
INK, MUTED = "#222222", "#666666"
NUM = {}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")


def snaps():
    out = []
    for f in sorted(glob.glob(os.path.join(RUN, "snap_0*.npz"))):
        z = np.load(f, allow_pickle=True)
        out.append({k: z[k] for k in ("t", "x", "y", "rho")})
    return out


def contour(ax, s, scale=1.0, color="0.45", lw=0.35):
    lr = np.log10(np.maximum(s["rho"], 1e-6))
    ax.contour(s["x"] / scale, s["y"] / scale, lr.T, levels=np.arange(-1.375, 0.6, 0.25),
               colors=color, linewidths=lw, alpha=0.7)


def atlas(d, S):
    """Where each outcome happens: four time windows and the self-similar frame."""
    g, w, t, x, y, ph = d["group"], d["weight"], d["t"], d["x"], d["y"], d["phys"]
    rows = (1, 2, 3, 4)
    fig, ax = plt.subplots(len(rows), 5, figsize=(17.5, 3.55 * len(rows) + 0.4), constrained_layout=True)
    for i, gi in enumerate(rows):
        hs = []
        for j, (t0, t1) in enumerate(WIN):
            m = (g == gi) & (t >= t0) & (t < t1) & ph
            H, _, _ = np.histogram2d(x[m], y[m], bins=128, range=[[-0.5, 0.5], [-0.5, 0.5]], weights=w[m])
            hs.append(H)
        vmax = max(h.max() for h in hs) or 1
        for j, (t0, t1) in enumerate(WIN):
            a = ax[i, j]
            im = a.imshow(np.ma.masked_less(hs[j].T, 0.5), origin="lower", extent=[-0.5, 0.5, -0.5, 0.5],
                          cmap=RAMP[gi], norm=LogNorm(1, vmax), interpolation="nearest")
            s = min(S, key=lambda s_: abs(float(s_["t"]) - 0.5 * (t0 + t1)))
            contour(a, s)
            a.set_title("%s, t %.1f–%.1f: %s faces" % (NAMES[gi], t0, t1, format(int(round(hs[j].sum())), ",")),
                        fontsize=9, color=INK)
            a.set_xticks([]); a.set_yticks([])
        # the self-similar frame: every face after t = 0.1 at (x/t, y/t)
        m = (g == gi) & (t > 0.1) & ph
        H, _, _ = np.histogram2d(x[m] / t[m], y[m] / t[m], bins=220, range=[[-1.3, 1.3], [-1.3, 1.3]], weights=w[m])
        a = ax[i, 4]
        a.imshow(np.ma.masked_less(H.T, 0.5), origin="lower", extent=[-1.3, 1.3, -1.3, 1.3], cmap=RAMP[gi],
                 norm=LogNorm(1, H.max() or 1), interpolation="nearest")
        contour(a, S[-1], scale=float(S[-1]["t"]))
        a.set_title("%s, all t > 0.1, at (x/t, y/t)" % NAMES[gi], fontsize=9, color=INK)
        a.set_xlabel("x / t", fontsize=8.5, color=MUTED); a.set_ylabel("y / t", fontsize=8.5, color=MUTED)
        style(a)
        fig.colorbar(im, ax=ax[i, :4].tolist(), shrink=0.85, pad=0.005).set_label(
            "faces per cell per window\n(RK stages, weighted)", fontsize=8, color=MUTED)
    fig.suptitle("Four-quadrant problem, 128², ft3: where the faces the seven-wave solver does not answer at "
                 "first try lie (contours: log10 ρ)", fontsize=12, color=INK)
    fig.savefig(os.path.join(OUT, "atlas.png"), dpi=130)
    plt.close(fig)


def timeline(d):
    """Per step, from the coverage bitmaps (exact counts), the retry split from the face table."""
    cov = sorted(glob.glob(os.path.join(RUN, "harvest", "coverage_*.npz")))
    steps = {}
    for f in cov:
        z = np.load(f)
        for r in range(z["sweep"].shape[0]):
            n = int(z["n"][r])
            st = int(z["sweep"][r]) // 6
            m = {k: unpack(z[k][r], n) for k in ("attempted", "exact", "seven", "planar5", "planar4",
                                                 "linear_fallback", "degenerate")}
            c = steps.setdefault(st, np.zeros(5))
            c += [m["seven"].sum(), m["planar5"].sum(), m["planar4"].sum(),
                  (m["attempted"] & ~m["exact"] & ~m["linear_fallback"]).sum(), m["attempted"].sum()]
    k = np.array(sorted(steps))
    C = np.array([steps[s] for s in k])
    diag = np.genfromtxt(os.path.join(RUN, "diag.csv"), delimiter=",", names=True)
    tt = np.atleast_1d(diag["t"])[k]
    # retries per step from the face table (kept whole there)
    g, w = d["group"], d["weight"]
    rstep = np.bincount(d["step"][g == 1], weights=w[g == 1], minlength=k.max() + 1)[k]
    seven1 = C[:, 0] - rstep
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    def sm(v, k=7):
        # a running mean that does not fall to zero at the ends
        p = np.pad(np.asarray(v, float), (k // 2, k // 2), mode="edge")
        return np.convolve(p, np.ones(k) / k, mode="valid")
    parts = [(seven1, 0), (rstep, 1), (C[:, 1], 2), (C[:, 2], 3), (C[:, 3], 4)]
    base = np.zeros_like(tt)
    for v, gi in parts:
        ax[0].fill_between(tt, base, base + sm(v), color=COL[gi], lw=0, label=NAMES[gi])
        base = base + sm(v)
    ax[0].set_xlabel("t", color=MUTED); ax[0].set_ylabel("attempted faces per step (6 sweeps)", color=MUTED)
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    ax[0].set_title("What answers the attempted faces, step by step", fontsize=10, color=INK)
    ends = []
    for v, gi in parts[1:]:
        ax[1].plot(tt, 100 * sm(v, 15) / sm(C[:, 4], 15), color=COL[gi], lw=2)
        ends.append((100 * sm(v, 15)[-1] / sm(C[:, 4], 15)[-1], gi))
    # direct labels, nudged apart in log space where two lines end together
    ends.sort()
    last = -np.inf
    for yv, gi in ends:
        ly = max(np.log10(yv), last + 0.12)
        last = ly
        ax[1].annotate(NAMES[gi], (tt[-1], yv), xytext=(tt[-1] + 0.006, 10 ** ly), fontsize=8.5,
                       color=INK, va="center", arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.6))
    ax[1].set_yscale("log")
    ax[1].set_xlabel("t", color=MUTED); ax[1].set_ylabel("% of the attempted faces", color=MUTED)
    ax[1].set_xlim(0, 0.52)
    ax[1].set_title("Shares of the rescues and losses", fontsize=10, color=INK)
    for a in ax:
        style(a); a.grid(alpha=0.25, lw=0.6)
    fig.savefig(os.path.join(OUT, "timeline.png"), dpi=140)
    plt.close(fig)
    NUM["per_step_attempted_first_last"] = [float(C[0, 4]), float(C[-1, 4])]


def wq(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]); return x[o][np.searchsorted(c, q * c[-1])]


def physics(d):
    g, w, al = d["group"], d["weight"], d["aligned"]
    gap = np.log10(np.maximum(d["gapAS"], 1e-9))
    fn = d["f_fn_max"].astype(float)
    Wt = d["f_Wt"].astype(float)
    beta = np.log10(np.maximum(d["f_beta_min"].astype(float), 1e-6))
    jump = np.log10(np.maximum(d["f_jump"].astype(float), 1e-9))
    fig = plt.figure(figsize=(15.5, 9.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    # (a) the planar share over (W_t, slow-Alfven gap), seven + planar faces
    a = fig.add_subplot(gs[0, 0])
    m = al & (g <= 2)
    xb = np.linspace(1, 7.5, 40); yb = np.linspace(-7, -1, 40)
    H1, _, _ = np.histogram2d(Wt[m], gap[m], bins=[xb, yb], weights=w[m] * (g[m] == 2))
    H0, _, _ = np.histogram2d(Wt[m], gap[m], bins=[xb, yb], weights=w[m])
    R = np.where(H0 > 30, H1 / np.maximum(H0, 1), np.nan)
    im = a.pcolormesh(xb, yb, R.T, cmap="Blues", vmin=0, vmax=1, shading="flat")
    fig.colorbar(im, ax=a, shrink=0.85).set_label("share answered by the planar solver", fontsize=8.5, color=MUTED)
    a.set_xlabel("tangential Lorentz factor of the faster side", color=MUTED)
    a.set_ylabel("log10 slow–Alfvén speed gap", color=MUTED)
    a.set_title("(a) The planar rescue: degenerate slow–Alfvén\nspeeds, most of all in the 0.99c shear", fontsize=10, color=INK)
    style(a)
    # (b) the lost share among faces the seven-wave refused, over (|Bn|/|B|, jump)
    a = fig.add_subplot(gs[0, 1])
    m = al & (g >= 2)
    xb = np.linspace(0.3, 1.0, 36); yb = np.linspace(-2.6, -0.2, 36)
    H1, _, _ = np.histogram2d(fn[m], jump[m], bins=[xb, yb], weights=w[m] * (g[m] == 4))
    H0, _, _ = np.histogram2d(fn[m], jump[m], bins=[xb, yb], weights=w[m])
    R = np.where(H0 > 20, H1 / np.maximum(H0, 1), np.nan)
    im = a.pcolormesh(xb, yb, R.T, cmap="Oranges", vmin=0, vmax=1, shading="flat")
    fig.colorbar(im, ax=a, shrink=0.85).set_label("share lost to HLLD", fontsize=8.5, color=MUTED)
    a.set_xlabel("normal-field fraction |Bn|/|B| (larger side)", color=MUTED)
    a.set_ylabel("log10 jump across the face", color=MUTED)
    a.set_title("(b) Lost, among the faces the seven-wave refused:\nnear-normal field and larger jumps", fontsize=10, color=INK)
    style(a)
    # (c) the four-unknown share among the rescued, over (|Bn|/|B|, beta)
    a = fig.add_subplot(gs[0, 2])
    m = al & ((g == 2) | (g == 3))
    xb = np.linspace(0.3, 1.0, 36); yb = np.linspace(-1.5, 2.5, 36)
    H1, _, _ = np.histogram2d(fn[m], beta[m], bins=[xb, yb], weights=w[m] * (g[m] == 3))
    H0, _, _ = np.histogram2d(fn[m], beta[m], bins=[xb, yb], weights=w[m])
    R = np.where(H0 > 20, H1 / np.maximum(H0, 1), np.nan)
    im = a.pcolormesh(xb, yb, R.T, cmap="Greens", vmin=0, vmax=1, shading="flat")
    fig.colorbar(im, ax=a, shrink=0.85).set_label("share needing the four-unknown solver", fontsize=8.5, color=MUTED)
    a.set_xlabel("normal-field fraction |Bn|/|B| (larger side)", color=MUTED)
    a.set_ylabel("log10 plasma beta (smaller side)", color=MUTED)
    a.set_title("(c) The four-unknown rescue: near-normal,\nstrong field (beta < 1)", fontsize=10, color=INK)
    style(a)
    # (d-f) the distributions per class
    for k, (vals, lab, rng_) in enumerate(((gap, "log10 slow–Alfvén speed gap", (-8, -0.5)),
                                           (fn, "normal-field fraction |Bn|/|B| (larger side)", (0, 1)),
                                           (np.minimum(Wt, 8), "tangential Lorentz factor (faster side)", (1, 8)))):
        a = fig.add_subplot(gs[1, k])
        bins = np.linspace(*rng_, 60)
        for gi in range(5):
            m = al & (g == gi) & np.isfinite(vals)
            h, e = np.histogram(vals[m], bins=bins, weights=w[m])
            a.step(e[:-1], h / max(h.sum(), 1), where="post", color=COL[gi], lw=1.8, label=NAMES[gi])
        a.set_xlabel(lab, color=MUTED); a.set_ylabel("fraction of the class", color=MUTED)
        a.set_yscale("log"); a.set_ylim(1e-4, 1)
        style(a)
        if k == 0:
            a.legend(frameon=False, fontsize=8)
    fig.suptitle("What separates the outcomes: the two input states alone", fontsize=12, color=INK)
    fig.savefig(os.path.join(OUT, "physics.png"), dpi=140)
    plt.close(fig)


def regimes(d):
    """The outcome mix over the two axes that organise it: the slow-Alfven gap
    and the normal-field fraction, with the two regime boundaries."""
    g, w, al = d["group"], d["weight"], d["aligned"]
    gap = np.log10(np.maximum(d["gapAS"], 1e-9))
    fn = d["f_fn_max"].astype(float)
    xb = np.linspace(-8, -0.5, 46); yb = np.linspace(0, 1, 41)
    H0, _, _ = np.histogram2d(gap[al], fn[al], bins=[xb, yb], weights=w[al])
    fig, ax = plt.subplots(1, 4, figsize=(18.5, 4.6), constrained_layout=True)
    a = ax[0]
    im = a.pcolormesh(xb, yb, np.ma.masked_less(H0.T, 1), cmap="Greys", norm=LogNorm(1, H0.max()), shading="flat")
    fig.colorbar(im, ax=a, shrink=0.85).set_label("attempted faces", fontsize=8.5, color=MUTED)
    a.set_title("All attempted faces", fontsize=10, color=INK)
    for k, (gi, ttl) in enumerate(((2, "planar rescue"), (3, "four-unknown rescue"), (4, "lost to HLLD"))):
        a = ax[k + 1]
        H1, _, _ = np.histogram2d(gap[al & (g == gi)], fn[al & (g == gi)], bins=[xb, yb], weights=w[al & (g == gi)])
        R = np.where(H0 > 50, H1 / np.maximum(H0, 1), np.nan)
        im = a.pcolormesh(xb, yb, R.T, cmap=RAMP[gi], vmin=0, vmax=np.nanpercentile(R, 99.5), shading="flat")
        fig.colorbar(im, ax=a, shrink=0.85).set_label("share of the attempted faces", fontsize=8.5, color=MUTED)
        a.set_title("Share answered by the %s" % ttl if gi != 4 else "Share lost to HLLD", fontsize=10, color=INK)
    for a in ax:
        a.axvline(-3, color=INK, lw=0.9, ls="--")
        a.axhline(0.9, color=INK, lw=0.9, ls="--")
        a.set_xlabel("log10 slow–Alfvén speed gap of the two states", color=MUTED)
        a.set_ylabel("normal-field fraction |Bn|/|B| (larger side)", color=MUTED)
        style(a)
    ax[0].text(-7.8, 0.94, "B: near-normal field", fontsize=8.5, color=INK)
    ax[0].text(-7.8, 0.45, "A: degenerate,\noblique field", fontsize=8.5, color=INK)
    ax[0].text(-2.8, 0.45, "neither:\nfirst try", fontsize=8.5, color=INK)
    fig.suptitle("Two regimes hold every rescue and every loss: the slow and Alfvén speeds nearly equal (A), "
                 "or the field nearly normal to the face (B)", fontsize=11.5, color=INK)
    fig.savefig(os.path.join(OUT, "regimes.png"), dpi=140)
    plt.close(fig)
    near = fn > 0.9
    deg = (d["gapAS"] < 1e-3) & ~near
    tab = {}
    for lab, sel in (("A degenerate, oblique", deg), ("B near-normal", near), ("neither", ~deg & ~near)):
        m = al & sel
        tab[lab] = dict(faces=float(w[m].sum()), **{NAMES[i]: float(np.average(g[m] == i, weights=w[m])) for i in range(5)})
    NUM["regimes"] = tab


def mechanism(d):
    c = np.load(os.path.join(HERE, "data", "cond.npz"))
    grp, gap, err, sv6, sv3 = c["group"], c["gapAS"], c["err6"], c["sv6"], c["sv3"]
    fig, axs = plt.subplots(2, 2, figsize=(15.5, 10.6), constrained_layout=True)
    lab = ("seven-wave, first try", "seven-wave, after retry", "planar rescue")
    # (a) the answer's slow wave against the Alfven speed ahead of it
    a = axs[0, 0]
    g, w = d["group"], d["weight"]
    LSs, RSs = d["f_LS_str"].astype(float), d["f_RS_str"].astype(float)
    mL = np.where(LSs > 1e-3, np.abs(d["f_LS_alfven_margin"].astype(float)), np.inf)
    mR = np.where(RSs > 1e-3, np.abs(d["f_RS_alfven_margin"].astype(float)), np.inf)
    marg = np.minimum(mL, mR)
    bins = np.logspace(-9, 0, 64)
    for gi in (0, 1, 2):
        m = (g == gi) & np.isfinite(marg)
        h, e = np.histogram(np.maximum(marg[m], 1.1e-9), bins=bins, weights=w[m])
        a.step(e[:-1], h / w[g == gi].sum(), where="post", color=(COL[0], COL[1], COL[2])[gi], lw=2, label=lab[gi])
        NUM.setdefault("slow_at_alfven_1e-4", {})[lab[gi]] = float(w[(g == gi) & (marg < 1e-4)].sum() / w[g == gi].sum())
    a.axvline(1e-4, color=MUTED, lw=1, ls="--")
    a.set_xscale("log")
    a.set_xlabel("closest slow wave (strength > 1e-3) to the Alfvén speed ahead of it, |s − λ_A|", color=MUTED)
    a.set_ylabel("fraction of the group", color=MUTED)
    a.set_title("(a) In the planar answers a slow wave rides the Alfvén speed\n"
                "(within 1e-4: %.0f%% of planar, %.0f%% of first-try answers)"
                % (100 * NUM["slow_at_alfven_1e-4"][lab[2]], 100 * NUM["slow_at_alfven_1e-4"][lab[0]]),
                fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(a)
    ax = [axs[0, 1], axs[1, 0], axs[1, 1]]
    a = ax[0]
    cc = (COL[0], COL[1], COL[2])
    for gi in (0, 1, 2):
        m = (grp == gi) & ~err
        a.scatter(np.maximum(gap[m], 1e-8), sv6[m, 4] / sv6[m, 0], s=5, color=cc[gi], alpha=0.5, lw=0, label=lab[gi])
        mf = (grp == gi) & err
        if mf.any():
            a.scatter(np.maximum(gap[mf], 1e-8), np.full(mf.sum(), 2e-9), s=9, marker="x", color=cc[gi], lw=0.8,
                      label="%s: Jacobian cannot be built (%d)" % (lab[gi], mf.sum()))
    a.axhline(1e-6, color=MUTED, lw=1, ls="--")
    a.annotate("Newton drops directions below 1e-6 of the largest", (1.2e-8, 1.4e-6), fontsize=8, color=MUTED)
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1e-9, 1)
    a.set_xlabel("slow–Alfvén speed gap of the two states", color=MUTED)
    a.set_ylabel("σ5 / σ1 of the seven-wave Jacobian at the answer", color=MUTED)
    a.set_title("(b) At the answer, the seven-wave Jacobian loses a fifth\ndirection, or cannot be built, where the planar solver is needed",
                fontsize=10, color=INK)
    a.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.5, loc="upper left")
    style(a)
    # (b) seed ceiling, when there
    a = ax[1]
    sc = os.path.join(HERE, "data", "seed_ceiling.json")
    if os.path.exists(sc):
        S = json.load(open(sc))
        r = np.array(S["radius"])
        xr = np.where(r == 0, 1e-4, r)            # the known answer itself, drawn at 1e-4
        for gi, gname in enumerate(("first", "retry", "planar")):
            v = np.array(S["conv"][gname])
            a.plot(xr, 100 * v, marker="o", ms=6, color=cc[gi], lw=2, label=lab[gi], mec="white", mew=1)
            a.scatter([3.5e-5], [100 * S["net"][gname]], marker="D", s=46, color=cc[gi], zorder=3,
                      edgecolor="white", linewidth=1)
        ticks = [3.5e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
        a.set_xscale("log"); a.set_xlim(2.2e-5, 0.16); a.set_ylim(-3, 103)
        a.set_xticks(ticks)
        a.set_xticklabels(["ft3's\nseed", "0", "3e-4", "1e-3", "3e-3", "0.01", "0.03", "0.1"], fontsize=8)
        a.minorticks_off()
        a.set_xlabel("start: the known answer moved by a random step of this size", color=MUTED)
        a.set_ylabel("% converged (one attempt, production budget)", color=MUTED)
        a.legend(frameon=False, fontsize=8, loc="center right")
    else:
        a.text(0.5, 0.5, "seed ceiling: job 78408 running", ha="center", transform=a.transAxes, color=MUTED)
    a.set_title("(c) Not the seed: started a hair from the answer, the seven-wave\nNewton still misses most planar answers",
                fontsize=10, color=INK)
    style(a)
    # (c) classifier importances
    a = ax[2]
    imp = json.load(open(os.path.join(HERE, "data", "importance.json")))
    tasks = list(imp)
    feats = []
    for tname in tasks:
        for f_, v in sorted(imp[tname]["importance"].items(), key=lambda kv: -kv[1])[:4]:
            if f_ not in feats:
                feats.append(f_)
    yb = np.arange(len(feats))
    hgt = 0.8 / len(tasks)
    for k, tname in enumerate(tasks):
        vals = [imp[tname]["importance"].get(f_, 0.0) for f_ in feats]
        a.barh(yb + (k - (len(tasks) - 1) / 2) * hgt, vals, height=hgt * 0.92,
               color=(COL[2], COL[4], COL[3], COL[1])[k], label="%s (AUC %.2f)" % (tname, imp[tname]["auc"]))
    a.set_yticks(yb); a.set_yticklabels(feats, fontsize=8.5)
    a.invert_yaxis()
    a.set_xlabel("drop in AUC when the feature is shuffled (t ≥ 0.3, held out)", color=MUTED)
    a.set_title("(d) Which input-state features predict the rung", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=7.5, loc="lower right")
    style(a)
    fig.savefig(os.path.join(OUT, "mechanism.png"), dpi=140)
    plt.close(fig)


def importance(d):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    X, names = matrix(d)
    g, w, al, t = d["group"], d["weight"], d["aligned"], d["t"]
    out = {}
    for name, pos, neg in (("planar vs seven-wave", g == 2, g <= 1),
                           ("lost vs rescued", g == 4, (g == 2) | (g == 3)),
                           ("four-unknown vs planar", g == 3, g == 2),
                           ("retry vs first try", g == 1, g == 0)):
        m = al & (pos | neg)
        y = pos[m].astype(int); Xm, wm, tm = X[m], w[m], t[m]
        tr, te = tm < 0.3, tm >= 0.3
        clf = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xm[tr], y[tr], sample_weight=wm[tr])
        auc = roc_auc_score(y[te], clf.predict_proba(Xm[te])[:, 1], sample_weight=wm[te])
        sub = np.random.default_rng(0).choice(np.flatnonzero(te), size=min(30000, int(te.sum())), replace=False)
        pi = permutation_importance(clf, Xm[sub], y[sub], sample_weight=wm[sub], scoring="roc_auc",
                                    n_repeats=4, random_state=0)
        out[name] = dict(auc=float(auc), importance={nm: float(v) for nm, v in zip(names, pi.importances_mean)})
    json.dump(out, open(os.path.join(HERE, "data", "importance.json"), "w"), indent=1)
    NUM["classifier"] = {k: v["auc"] for k, v in out.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load(os.path.join(HERE, "data", "sample.npz"))
    S = snaps()
    w, g = d["weight"], d["group"]
    NUM["faces_by_group"] = {NAMES[i]: float(w[g == i].sum()) for i in range(5)}
    atlas(d, S)
    timeline(d)
    physics(d)
    regimes(d)
    if not os.path.exists(os.path.join(HERE, "data", "importance.json")):
        importance(d)
    mechanism(d)
    json.dump(NUM, open(os.path.join(OUT, "numbers.json"), "w"), indent=1)
    print(json.dumps(NUM, indent=1))


if __name__ == "__main__":
    main()
