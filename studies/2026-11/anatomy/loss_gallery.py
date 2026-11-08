#!/usr/bin/env python
"""What the two kinds of lost face look like (2026-11-04; user: "We have these
2 loss types. I want to see examples for them. How they look like").

One figure per loss type, one column per example face (failure ledger 78407
lanes, all R4: no method finds a root).  Per example:
  1. its two states (solver frame: n normal, t the tangential direction of
     the plane);
  2. its Riemann fan in the (x, t) plane, coloured by the tangential field
     B_t of the self-similar tube solution (B_t(x, t) = B_t(x/t)), with the
     fast, Alfven and slow speeds of the two states as rays;
  3. density and gas pressure through the fan;
  4. B_t where the slow and Alfven waves sit, at 4096 / 2048 / 1024 cells.
The tube is the face's own 1D Riemann problem evolved with HLLD + MC
(tubes_examples.py on calea, t = 0.18).

    python loss_gallery.py A LANE ... --labels ... --out figs/loss_type_A.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
INK, MUTED = "#222222", "#666666"
SPEED = {"fast": "#7f7f7f", "Alfvén": "#CC79A7", "slow": "#0072B2"}
NC = (4096, 2048, 1024)
TITLE = {
    "A": "Loss type A — the 0.99c shear lines: the slow wave runs at the Alfvén speed",
    "B": "Loss type B — the field nearly normal to the face",
}


def tube(lane, nc):
    z = np.load(os.path.join(HERE, "data", "tubes", "tube_%d_%d.npz" % (lane, nc)))
    return {k: z[k] for k in z.files}


def physics(U, gamma):
    rho, P, vn, vy, vz, By, Bz, Bn = U
    v2 = vn * vn + vy * vy + vz * vz
    W = 1 / np.sqrt(1 - v2)
    vB = Bn * vn + By * vy + Bz * vz
    b2 = (Bn * Bn + By * By + Bz * Bz) / W ** 2 + vB ** 2
    p = P - 0.5 * b2
    h = 1 + gamma / (gamma - 1) * p / rho
    use_z = abs(Bz) + abs(vz) > abs(By) + abs(vy)
    vt, bt = (vz, Bz) if use_z else (vy, By)
    return dict(rho=rho, p=p, vn=vn, vt=vt, Wt=1 / np.sqrt(1 - vt * vt), Bn=Bn, Bt=bt,
                fn=abs(Bn) / np.sqrt(Bn * Bn + By * By + Bz * Bz), beta=2 * p / b2, sigma=b2 / (rho * h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("A", "B"))
    ap.add_argument("lanes", type=int, nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compact", action="store_true", help="slides: the fan and the B_t zoom only, larger type")
    a = ap.parse_args()
    from rmhd.batched import wave_speeds_b as WB
    p = np.load(os.path.join(HERE, "data", "ledger", "population.npz"), allow_pickle=True)
    gamma = float(p["gamma"])
    n = len(a.lanes)
    if a.compact:
        plt.rcParams.update({"font.size": 15})
        fig = plt.figure(figsize=(6.4 * n, 9.6), constrained_layout=True)
        gs = fig.add_gridspec(2, n, height_ratios=[1.25, 1.0])
    else:
        fig = plt.figure(figsize=(4.6 * n, 14.2), constrained_layout=True)
        gs = fig.add_gridspec(4, n, height_ratios=[0.85, 1.25, 1.0, 1.0])
    ROW = {"fan": 0, "zoom": 1} if a.compact else {"fan": 1, "zoom": 3}
    TK, LB, LG = (13, 14, 12) if a.compact else (8, 8.5, 7.5)
    for c, ln in enumerate(a.lanes):
        UL, UR = p["UL"][ln], p["UR"][ln]
        L, R = physics(UL, gamma), physics(UR, gamma)
        eL = WB.xi_all(*[np.array([UL[j]]) for j in range(7)], np.array([UL[7]]), gamma)
        eR = WB.xi_all(*[np.array([UR[j]]) for j in range(7)], np.array([UR[7]]), gamma)
        sp = {"fast": (float(np.asarray(eL[0])[0, 0]), float(np.asarray(eR[0])[0, 3])),
              "Alfvén": (float(np.asarray(eL[1])[0]), float(np.asarray(eR[2])[0])),
              "slow": (float(np.asarray(eL[0])[0, 1]), float(np.asarray(eR[0])[0, 2]))}
        have = [nc for nc in NC if os.path.exists(os.path.join(HERE, "data", "tubes", "tube_%d_%d.npz" % (ln, nc)))]
        T = {nc: tube(ln, nc) for nc in have}
        use_z = abs(UL[6]) + abs(UR[6]) > abs(UL[5]) + abs(UR[5])
        bk = "Bz" if use_z else "By"
        sgn = 1.0
        xi = {nc: (((np.arange(nc) + 0.5) / nc - 0.5) / float(T[nc]["t"])) for nc in have}
        top = have[0]
        # 1. the two states
        ax = fig.add_subplot(gs[0, c]) if not a.compact else None
        if ax is not None:
            ax.axis("off")
        rows = [("", "left", "right"),
                ("density ρ", "%.4g" % L["rho"], "%.4g" % R["rho"]),
                ("gas pressure p", "%.4g" % L["p"], "%.4g" % R["p"]),
                ("normal velocity", "%+.3f" % L["vn"], "%+.3f" % R["vn"]),
                ("tangential velocity", "%+.4f" % L["vt"], "%+.4f" % R["vt"]),
                ("  its Lorentz factor", "%.2f" % L["Wt"], "%.2f" % R["Wt"]),
                ("normal field B_n", "%.4f" % L["Bn"], "%.4f" % R["Bn"]),
                ("tangential field B_t", "%+.4f" % L["Bt"], "%+.4f" % R["Bt"]),
                ("|B_n| / |B|", "%.4f" % L["fn"], "%.4f" % R["fn"]),
                ("plasma β", "%.3g" % L["beta"], "%.3g" % R["beta"]),
                ("slow − Alfvén speed", "%.1e" % abs(sp["slow"][0] - sp["Alfvén"][0]),
                 "%.1e" % abs(sp["slow"][1] - sp["Alfvén"][1]))]
        for k, (lab, l_, r_) in enumerate(rows if ax is not None else []):
            y = 0.97 - k * 0.087
            ax.text(0.0, y, lab, fontsize=9, color=MUTED if k else INK, transform=ax.transAxes, va="top")
            ax.text(0.62, y, l_, fontsize=9, color=INK, transform=ax.transAxes, va="top", ha="right",
                    family="monospace")
            ax.text(0.97, y, r_, fontsize=9, color=INK, transform=ax.transAxes, va="top", ha="right",
                    family="monospace")
        if ax is not None:
            ax.set_title(a.labels[c] if a.labels else "lane %d" % ln, fontsize=10.5, color=INK, loc="left")
        # 2. the fan in (x, t), coloured by B_t(x/t)
        ax = fig.add_subplot(gs[ROW["fan"], c])
        X = max(abs(sp["fast"][0]), abs(sp["fast"][1])) * 1.12
        xs = np.linspace(-X, X, 700); ts = np.linspace(1e-3, 1.0, 400)
        XX, TT = np.meshgrid(xs, ts)
        bt = np.interp(XX / TT, xi[top], T[top][bk])
        b0 = T[top][bk]
        flips = b0.min() < 0 < b0.max()
        if flips:                       # the field reverses: diverging, white at B_t = 0
            lim = max(abs(b0).max(), 1e-12)
            im = ax.pcolormesh(xs, ts, bt, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -lim, lim), shading="auto",
                               rasterized=True)
            clab = "B_t (red +, blue −, white 0)"
        else:                           # no reversal: its own range, sequential
            im = ax.pcolormesh(xs, ts, bt, cmap="viridis", vmin=b0.min(), vmax=b0.max(), shading="auto",
                               rasterized=True)
            clab = "B_t (one sign throughout)"
        for nm, (s1, s2) in sp.items():
            for s_ in (s1, s2):
                ax.plot([0, s_], [0, 1], color=SPEED[nm], lw=1.0, ls=":")
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_xlim(-X, X); ax.set_ylim(0, 1)
        ax.set_xlabel("x (the face at x = 0)", fontsize=LB, color=MUTED)
        ax.set_ylabel("t", fontsize=LB, color=MUTED)
        ax.tick_params(labelsize=TK, colors=MUTED)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
        cb.set_label(clab, fontsize=LB - 1, color=MUTED)
        cb.ax.tick_params(labelsize=TK)
        ax.set_title(("%s\n" % a.labels[c] if a.compact and a.labels else "") + "the Riemann fan, coloured by B_t",
                     fontsize=16 if a.compact else 9.5, color=INK, loc="left" if a.compact else "center")
        # 3. density and pressure through the fan
        ax = fig.add_subplot(gs[2, c]) if not a.compact else None
        if ax is None:
            pass
        else:
            for key, col, lab in (("rho", INK, "ρ / ρ_left"), ("p", "#E69F00", "p / p_left")):
                for k, nc in enumerate(have):
                    ax.plot(xi[nc], T[nc][key] / T[nc][key][0], color=col, lw=(1.6, 0.9, 0.7)[k],
                            alpha=(1, 0.5, 0.3)[k], label=lab if k == 0 else None)
            for nm, (s1, s2) in sp.items():
                for s_ in (s1, s2):
                    ax.axvline(s_, color=SPEED[nm], lw=0.9, ls=":")
            ax.axvline(0, color=INK, lw=0.6)
            ax.set_xlim(-X, X)
            ax.set_title("density and pressure through the fan", fontsize=9.5, color=INK)
            ax.legend(frameon=False, fontsize=8)
            ax.tick_params(labelsize=TK, colors=MUTED); ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel("x / t", fontsize=LB, color=MUTED)
        # 4. B_t where the slow and Alfven waves sit (the side whose B_t changes most there)
        ax = fig.add_subplot(gs[ROW["zoom"], c])
        b = T[top][bk]; x0 = xi[top]
        best, centre = -1.0, 0.0
        for side in (0, 1):
            cc = 0.5 * (sp["slow"][side] + sp["Alfvén"][side])
            w = (np.abs(x0 - cc) < 0.03)
            ch = np.ptp(b[w]) if w.any() else 0.0
            if ch > best:
                best, centre = ch, cc
        hw = 0.05 if abs(centre) < 0.1 else 0.09
        for k, nc in enumerate(have):
            ax.plot(xi[nc], T[nc][bk], color="#D55E00", lw=(1.7, 1.0, 0.8)[k], alpha=(1, 0.55, 0.3)[k],
                    marker="." if k == 0 else None, ms=2.5, label="%d cells" % nc)
        for nm in ("Alfvén", "slow"):
            for s_ in sp[nm]:
                ax.axvline(s_, color=SPEED[nm], lw=1.1, ls=":")
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_xlim(centre - hw, centre + hw)
        win = (x0 > centre - hw) & (x0 < centre + hw)
        lo, hi = b[win].min(), b[win].max()
        if lo < 0 < hi:
            ax.axhline(0, color="#bbbbbb", lw=0.7)
        pad = 0.08 * (hi - lo + 1e-12)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title("B_t where the slow and Alfvén waves sit", fontsize=16 if a.compact else 9.5, color=INK)
        ax.legend(frameon=False, fontsize=LG)
        ax.tick_params(labelsize=TK, colors=MUTED); ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("x / t   (dotted: slow blue, Alfvén pink, often on top of each other)",
                      fontsize=LB, color=MUTED)
    if not a.compact:
        fig.suptitle(TITLE[a.kind], fontsize=13, color=INK)
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
