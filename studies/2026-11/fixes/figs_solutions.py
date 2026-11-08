#!/usr/bin/env python
"""Figures for the user (2026-11-05): what the newly solved faces' exact
solutions look like, and how the new slow wave carries B_t through zero.

    figs/new_solutions.png   four faces: B_t(x/t) of the exact solution vs the
                             face's own 1D tube (HLLD + MC, 4096 cells), full
                             fan and a zoom on the wave that matters
    figs/zero_crossing.png   the mechanism: B_t, |B_t|, the slow speed and the
                             entropy along the slow wave's integral curve; why
                             pressure fails at beta ~ 300; the old vs new rule
    figs/typeA_hole.png      the type-A defect: the planar residual along the
                             contact field, loose fallback vs retry
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sol_profiles as SP
from rmhd.batched import planar5_b as P5, slow_shock_b as SSB

OUT = "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/figs"
INK, MUTED, GRID = "#111827", "#6b7280", "#e5e7eb"
EXACT, TUBE, FAN, OLD = "#111827", "#2563eb", "#f59e0b", "#dc2626"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})
off = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/replay_final2/replay_off.npz")
ps = lambda l: float(abs(off["pstar"][l]))

CASES = [
    (62, "Type A (0.99c shear)", "two ordinary slow shocks -- the slow-shock solver had a hole at this root",
     dict(kinds=(0, 0), unk=[np.array([0.3029687]), np.array([0.3188758]), np.array([0.3188758]),
                             np.array([0.3029711])]), (0.0015, 0.0185)),
    (215, "Type B weak (field nearly normal, beta ~ 20)",
     "a slow RAREFACTION on each side carries B_t through zero (pressure-steered)",
     dict(kinds=(1, 1), pl=ps(215), pr=ps(215)), "LS"),
    (900, "Type B weak (beta ~ 300, field normal to 0.7%)",
     "left: a slow fan 1e-6 wide reverses B_t; right: a reversing jump at the Alfven speed",
     dict(kinds=(2, 2), pl=ps(900), pr=ps(900)), "LS"),
    (858, "Type B strong (beta ~ 0.3, fast wave 6e-4 from the Alfven speed)",
     "a regular fan -- found once the weak fast wave starts at its own pressure",
     dict(kinds=(1, 1), pl=ps(858), pr=float(SP.problem(858)[1][1][0]) * 1.001), (0.44, 0.66)),
]


def new_solutions():
    fig, axes = plt.subplots(len(CASES), 2, figsize=(13.5, 3.2 * len(CASES)),
                             gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    for row, (lane, title, note, c, zoom) in enumerate(CASES):
        sol = SP.solve(lane, c["kinds"], c.get("pl"), c.get("pr"), unk=c.get("unk"))
        xi, st, waves = SP.profile(sol)
        txi, tz = SP.tube(lane)
        ax, az = axes[row]
        for name, kind, a, b in waves:
            if kind == "fan":
                for x in (ax, az):
                    x.axvspan(a, b, color=FAN, alpha=0.18, lw=0)
        ax.plot(txi, tz["By"], color=TUBE, lw=1.0, label="1D tube, 4096 cells")
        ax.plot(xi, st[5], color=EXACT, lw=1.8, label="exact solution")
        crosses = st[5].min() < 0 < st[5].max()
        if crosses:
            ax.axhline(0, color=MUTED, lw=0.7, ls="--")
        ext = [w for w in waves if w[0] in ("LF", "RF")]
        lo = min(w[2] for w in ext) - 0.06
        hi = max(w[3] for w in ext) + 0.06
        ax.set_xlim(max(lo, -1), min(hi, 1))
        ax.set_title("Lane %d -- %s\n%s" % (lane, title, note), loc="left", fontsize=9.8, color=INK)
        ax.set_ylabel("B_t (lab)")
        if row == len(CASES) - 1:
            ax.set_xlabel("x / t")
        # the zoom: a range, or the named slow wave and a margin of tube cells
        if isinstance(zoom, str):
            w = [w for w in waves if w[0] == zoom][0]
            cell = 1.0 / (4096 * float(tz["t"]))
            zlo, zhi = w[2] - 4 * cell, w[3] + 4 * cell
        else:
            zlo, zhi = zoom
        m = (txi >= zlo) & (txi <= zhi)
        az.plot(txi[m], tz["By"][m], "o", ms=3.2, color=TUBE, label="tube cells")
        az.plot(xi, st[5], color=EXACT, lw=1.8, label="exact")
        if crosses:
            az.axhline(0, color=MUTED, lw=0.7, ls="--")
        az.set_xlim(zlo, zhi)
        inz = (xi >= zlo) & (xi <= zhi)
        yv = np.concatenate([st[5][inz], tz["By"][m]])
        pad = 0.08 * (yv.max() - yv.min() + 1e-12)
        az.set_ylim(yv.min() - pad, yv.max() + pad)
        if isinstance(zoom, str):
            w = [w for w in waves if w[0] == zoom][0]
            az.set_title("zoom: the %s slow %s, %.1e wide in x/t" % (
                "left" if zoom == "LS" else "right", w[1], w[3] - w[2]), loc="left", fontsize=9.5, color=MUTED)
        else:
            az.set_title("zoom: the slow / Alfven / contact cluster", loc="left", fontsize=9.5, color=MUTED)
        az.ticklabel_format(axis="x", useOffset=True, style="plain")
        if row == 0:
            ax.legend(loc="upper right", fontsize=8.5)
            az.legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Faces production lost that the fixes now solve: the exact solution (black) against the face's own "
                 "high-resolution 1D tube (blue); shaded = rarefaction fans", fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(os.path.join(OUT, "new_solutions.png"), dpi=140)
    plt.close(fig)


def zero_crossing():
    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    # lane 215, left slow wave: walk the integral curve past the solution's end
    sol = SP.solve(215, (1, 1), ps(215), ps(215))
    A = sol["zones"][0]
    Bn = sol["Bn"]
    al = sol["alpha"]
    k = 200
    Pt = np.linspace(A[1][0], A[1][0] - 1.2e-3, k)
    y, _, e = SP.FAN_P(Pt, SP._rep(A, k), "LS", np.repeat(Bn, k))
    lam = SP.XI(y, "LS", np.repeat(Bn, k))
    yl = SP.lab(y, np.repeat(al, k))
    Bt, dP = yl[5], A[1][0] - Pt
    B_end = SP.lab(sol["zones"][1], al)[5][0]

    def K(X):
        rho, P, vx, vy, vz, By, Bz = X
        W2 = 1 / (1 - vx * vx - vy * vy - vz * vz); vB = vx * Bn + vy * By + vz * Bz
        return (P - 0.5 * ((Bn ** 2 + By ** 2 + Bz ** 2) / W2 + vB ** 2)) / X[0] ** SP.GAMMA

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(dP * 1e3, Bt, color=EXACT, lw=2, label="B_t (signed)")
    ax.plot(dP * 1e3, np.abs(Bt), color=OLD, lw=1.4, ls="--", label="|B_t|")
    ax.axhline(0, color=MUTED, lw=0.7, ls="--")
    ax.axvline((A[1][0] - sol["zones"][1][1][0]) * 1e3, color=FAN, lw=1.2)
    ax.text((A[1][0] - sol["zones"][1][1][0]) * 1e3, 0.12, " end of this face's fan", color=FAN, fontsize=8.5)
    ax.set_xlabel("pressure drop along the fan, x 1e-3")
    ax.set_ylabel("B_t behind (lab)")
    ax.set_title("a) lane 215, left slow fan: B_t passes\nsmoothly through zero; |B_t| has a kink there", loc="left", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(Bt, lam, color=EXACT, lw=2)
    ax.axvline(0, color=MUTED, lw=0.7, ls="--")
    ax.set_xlabel("B_t (lab)")
    ax.set_ylabel("slow characteristic speed")
    ax.set_title("b) the slow speed rises steadily through B_t = 0:\nthe characteristics keep spreading, so it stays a fan", loc="left", fontsize=10)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(Bt, K(y) / K(A) - 1, color=EXACT, lw=0.8)
    ax.set_ylim(-2e-13, 2e-13)
    ax.axvline(0, color=MUTED, lw=0.7, ls="--")
    ax.set_xlabel("B_t (lab)")
    ax.set_ylabel("entropy change K/K_ahead - 1")
    ax.set_title("c) and it is a true rarefaction:\nisentropic to round-off (1e-14) through the crossing", loc="left", fontsize=10)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # lane 900: pressure is useless as the coordinate at beta ~ 300
    sol9 = SP.solve(900, (2, 2), ps(900), ps(900))
    A9 = sol9["zones"][0]; Bn9 = sol9["Bn"]; al9 = sol9["alpha"]
    tg = np.linspace(A9[5][0], -A9[5][0] - 0.5, 400)
    y9, _, e9 = SP.S4["fan_signed"](tg, SP._rep(A9, 400), "LS", np.repeat(Bn9, 400))
    rel = (A9[1][0] - y9[1]) / A9[1][0]
    Bt9 = SP.lab(y9, np.repeat(al9, 400))[5]
    ax = fig.add_subplot(gs[1, 0])
    m = rel > 0
    ax.semilogx(rel[m], Bt9[m], color=EXACT, lw=2)
    ax.axhline(0, color=MUTED, lw=0.7, ls="--")
    ax.set_xlabel("relative pressure drop along the fan")
    ax.set_ylabel("B_t behind (lab)")
    ax.set_title("d) lane 900 (beta ~ 300): a relative pressure change of 1e-7\nmoves B_t by 0.1 -- so the new wave steps in B_t, not in P", loc="left", fontsize=10)

    # the old and new rule for the slow wave to a contact field b
    ax = fig.add_subplot(gs[1, 1:])
    ax.set_xlim(-1.75, 1.6); ax.set_ylim(-0.2, 2.25); ax.axis("off")
    a = 0.5          # the field ahead (positive), schematic
    def band(y0, segs, label):
        for x0, x1, col, txt in segs:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, 0.42, color=col, alpha=0.85, lw=0))
            ax.text(0.5 * (x0 + x1), y0 + 0.21, txt, ha="center", va="center", fontsize=8.8, color="white")
        ax.text(-1.7, y0 + 0.62, label, fontsize=10, color=INK, va="center")
    band(1.45, [(-1.7, -a, OLD, "fan in |B_t|, sign kept: WRONG"),
                (-a, a, "#6b7280", "|b| <= |B_t|: shock"),
                (a, 1.5, "#2563eb", "fan in |B_t|")],
         "before: |b| against |B_t| decides shock or fan -- a fan can never reverse the field")
    band(0.35, [(-1.7, a, "#16a34a", "fan side (pressure falls): a fan stepped in signed B_t, through 0"),
                (a, 1.5, "#7c3aed", "shock to the signed b")],
         "now (signed-field slow wave): the side is decided by the fan's own direction, dB_t/dP")
    for y0 in (1.45, 0.35):
        ax.plot([a, a], [y0 - 0.05, y0 + 0.47], color=INK, lw=1.2)
        ax.plot([0, 0], [y0 - 0.05, y0 + 0.47], color=INK, lw=0.8, ls=":")
    ax.text(a, 0.05, "B_t ahead", ha="center", fontsize=8.5, color=INK)
    ax.text(0, 0.05, "0", ha="center", fontsize=8.5, color=INK)
    ax.text(1.5, 0.05, "target b at the contact ->", ha="right", fontsize=8.5, color=MUTED)
    ax.set_title("e) the slow wave to a target field b, schematic (field ahead positive; the fan direction is this face's)",
                 loc="left", fontsize=10)
    fig.suptitle("How the new slow wave carries B_t through zero", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "zero_crossing.png"), dpi=140)
    plt.close(fig)


def typeA_hole():
    lane = 62
    L, R, Bn = SP.problem(lane)
    S5 = P5.make_solver(SP.GAMMA)
    L0, R0 = S5["to_planar"](L, R, Bn)[:2]
    u0, u2 = 0.3029687, 0.3029711
    bts = 0.3188758 + np.linspace(-0.012, 0.012, 241)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for on, col, lab in ((False, OLD, "before: loose fallback"), (True, EXACT, "with the retry")):
        SSB.LOOSE_RETRY = on
        F = []
        for b in bts:
            f, *_ = S5["structure"](L0, R0, [np.array([u0]), np.array([b]), np.array([u2])], Bn)
            F.append(f[:, 0])
        F = np.array(F)
        ax[0].plot(bts, F[:, 1], color=col, lw=1.6 if on else 2.4, label=lab, zorder=3 if on else 2)
        ax[1].plot(bts, F[:, 2], color=col, lw=1.6 if on else 2.4, label=lab, zorder=3 if on else 2)
    SSB.LOOSE_RETRY = True
    for a, nm in zip(ax, ("[[v_t]] at the contact", "[[P_tot]] at the contact")):
        a.axhline(0, color=MUTED, lw=0.7, ls="--"); a.axvline(0.3188758, color=FAN, lw=1)
        a.set_xlabel("contact field B_t (planar frame)"); a.set_title(nm, loc="left", fontsize=10)
    ax[0].legend(fontsize=8.5)
    ax[1].set_ylim(-0.25, 0.05)
    fig.suptitle("Type A (lane 62): around the root (orange) the slow-shock solver returned a grid point instead of a "
                 "root,\nso the residual had a hole exactly where it should cross zero; the retry closes it (black)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(OUT, "typeA_hole.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["new", "zero", "hole"]
    if "new" in which: new_solutions()
    if "zero" in which: zero_crossing()
    if "hole" in which: typeA_hole()
    print("wrote", OUT)
