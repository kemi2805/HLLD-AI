#!/usr/bin/env python3
"""Balsara 1: which exact answer does a finite-volume run approach?

    measure        every run's distance to both exact answers: the mean |q - answer|
                   over xi 0.026-0.050 (the left slow wave and the compound fan) for
                   q = B_y, rho, p, v_y, and the star plateau's median (xi
                   Vtail + 0.02 .. the contact - 0.02) against each answer, per
                   quantity.  Writes measure.npz.
    figure         the high-resolution figure: HLLE + MC and HLLE + WENO5-Z
                   resolution sequences near the left slow wave, B_y at 16384 for
                   both fluxes, the local distance to each answer against cells, the
                   plateau density against cells
    family-figure  the solution family (balsara1_family.py build) with the MC runs
                   on it: profiles of three exact members against the runs, the
                   plateau density against b, and the 32768-cell MC run's plateau
                   (the gas shocked most recently matches the member of the run's
                   current b)

    python scripts/balsara1_selection.py measure --data DIR --out DIR/measure.npz
    python scripts/balsara1_selection.py figure  --data DIR --measure DIR/measure.npz --out fig.png
    python scripts/balsara1_selection.py family-figure --data DIR --family DIR/balsara1_family.npz --out fig.png

DIR holds the September runs in DIR/sept (b1_*.npz and balsara1_compound.npz, from
calea results/compound_selection_2026-09) and the high-resolution runs in DIR/hires
(results/compound_selection_2026-10).  The fluxes are not recorded in the files: the
September ones are named by file below, the hires ones by their file names.

Recreated 2026-10-26 from the 10-17 session scripts (measure.py, plot_hires.py,
plot_family.py), lost with the /tmp scratchpad.  Needs matplotlib for the figures.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import balsara1_study as BS                                       # noqa: E402

T = 0.4
WINDOW = (0.026, 0.050)
SEPT = (("b1_mc.npz", "hlld"), ("b1_weno5z.npz", "hlld"), ("b1_mp5.npz", "hlld"),
        ("b1_mp7.npz", "hlld"), ("b1_hlld_mc.npz", "hlld"), ("b1_hlld_mp5.npz", "hlld"),
        ("b1_hlle_mc.npz", "hlle"), ("b1_hlle_mp5.npz", "hlle"))


def load_all(data):
    specs = ["%s:%s" % (flux, os.path.join(data, "sept", f)) for f, flux in SEPT
             if os.path.exists(os.path.join(data, "sept", f))]
    specs += ["%s:%s" % (os.path.basename(f).split("_")[1], f)
              for f in sorted(glob.glob(os.path.join(data, "hires", "b1_*_*_*.npz")))]
    return BS.load_runs(specs)


class Answers:
    """Both exact answers on any rays; the compound fan tabulated once (it is smooth)."""

    def __init__(self, f, n_fan=2001):
        self.ans = a = BS.answers(f)
        self.p0, self.p1 = BS.prim(a["Z"][0]), BS.prim(a["Z"][1])
        self.pA, self.pS, self.pB = BS.prim(a["A"]), BS.prim(a["S"]), BS.prim(a["B"])
        self.pC = BS.prim(np.load(f)["compound_C"])
        self.XF = np.linspace(a["Vsh"], a["Vtl"], n_fan)
        fan, self.fan_checks = BS.compound_fan(a, self.XF)
        self.FAN = {k: np.asarray(v, dtype=float) for k, v in fan.items()}

    def ours(self, xi, k):
        return np.where(xi < self.ans["sp"][1], self.p0[k], self.p1[k])

    def comp(self, xi, k):
        a = self.ans
        out = np.where(xi < a["Vsh"], self.pA[k], self.pB[k]).astype(float)
        m = (xi >= a["Vsh"]) & (xi < a["Vtl"])
        out[m] = np.interp(xi[m], self.XF, self.FAN[k])
        return out


def measure(a):
    runs = load_all(a.data)
    A = Answers(os.path.join(a.data, "sept", "balsara1_compound.npz"))
    print("compound fan table: %s" % A.fan_checks)
    lo = A.ans["Vtl"] + 0.02
    hi = 0.5 * (A.pB["vx"] + A.pC["vx"]) - 0.02
    keys = sorted(runs, key=lambda k: (k[0], k[1], k[2]))
    loc = np.zeros((len(keys), 4, 2))
    pl = np.zeros((len(keys), 4, 2))
    for n, key in enumerate(keys):
        r = runs[key]
        xi = r["xi"]
        m = (xi >= WINDOW[0]) & (xi <= WINDOW[1])
        for j, q in enumerate(("By", "rho", "p", "vy")):
            loc[n, j] = (np.mean(np.abs(r[q][m] - A.ours(xi[m], q))),
                         np.mean(np.abs(r[q][m] - A.comp(xi[m], q))))
        mp = (xi > lo) & (xi < hi)
        for j, q in enumerate(("rho", "p", "vx", "By")):
            med = float(np.median(r[q][mp]))
            pl[n, j] = ((med - A.p1[q]) / abs(A.p1[q]), (med - A.pB[q]) / abs(A.pB[q]))
        print("%-5s %-7s %6d | local By ours %.3e comp %.3e | plateau rho rel. ours %+.2e comp %+.2e"
              % (key[0], key[1], key[2], loc[n, 0, 0], loc[n, 0, 1], pl[n, 0, 0], pl[n, 0, 1]))
    xs = np.linspace(*WINDOW, 20001)
    own = float(np.mean(np.abs(A.ours(xs, "By") - A.comp(xs, "By"))))
    print("the two answers' own local distance in B_y: %.4f" % own)
    np.savez_compressed(a.out, flux=np.array([k[0] for k in keys]), lim=np.array([k[1] for k in keys]),
                        n=np.array([k[2] for k in keys]), local=loc, plateau=pl,
                        local_vars=np.array(["By", "rho", "p", "vy"]),
                        plateau_vars=np.array(["rho", "p", "vx", "By"]), answers_distance=own,
                        plateau_window=np.array([lo, hi]))
    print("wrote", a.out)


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


INK, MUTED, OURS, COMP = "#222222", "#666666", "#111111", "#7B3294"
ORANGES = ["#FDBE85", "#FD8D3C", "#E6550D", "#A63603"]
BLUES = ["#9ECAE1", "#4292C6", "#2171B5", "#08306B"]
MC, WENO = "#E6550D", "#2171B5"
X0, X1 = 0.5112, 0.5172


def _answers_on(ax, A, k, lw=1.8):
    a = A.ans
    ax.axvspan(0.5 + T * a["Vsh"], 0.5 + T * a["Vtl"], color="#EFE3F5", lw=0, zorder=0)
    xs = 0.5 + T * a["sp"][1]
    ax.plot([X0, xs, xs, X1], [A.p0[k], A.p0[k], A.p1[k], A.p1[k]], "-", color=OURS, lw=lw,
            zorder=5, label="exact: intermediate shock (ours, = GR06)")
    cx = [X0, 0.5 + T * a["Vsh"], 0.5 + T * a["Vsh"]] + list(0.5 + T * A.XF[::10]) + [X1]
    cy = [A.pA[k], A.pA[k], A.pS[k]] + list(A.FAN[k][::10]) + [A.pB[k]]
    ax.plot(cx, cy, "--", color=COMP, lw=lw, zorder=5, label="exact: compound wave")


def _run_line(ax, runs, key, k, color, ls="-", lw=1.1, label=None):
    r = runs[key]
    m = (r["x"] > X0) & (r["x"] < X1)
    ax.plot(r["x"][m], r[k][m], linestyle=ls, color=color, lw=lw, label=label)


def figure(a):
    plt = _style()
    runs = load_all(a.data)
    A = Answers(os.path.join(a.data, "sept", "balsara1_compound.npz"), n_fan=301)
    M = np.load(a.measure)
    own = float(M["answers_distance"])
    fig, axs = plt.subplots(2, 3, figsize=(19, 11))
    for ax, lim, Ns, ramp, name in ((axs[0, 0], "mc", (4096, 8192, 16384, 32768), ORANGES, "HLLE + MC (PLM)"),
                                    (axs[0, 1], "weno5z", (2048, 4096, 8192, 16384), BLUES, "HLLE + WENO5-Z")):
        _answers_on(ax, A, "rho")
        for N, c in zip(Ns, ramp):
            if ("hlle", lim, N) in runs:
                _run_line(ax, runs, ("hlle", lim, N), "rho", c, label="%s, %d cells" % (name, N))
        ax.set_ylim(0.66, 0.79)
        ax.set_title("%s: density behind the left slow wave" % name, fontsize=12, color=INK)
        ax.legend(fontsize=8, loc="upper right")
    ax = axs[0, 2]
    _answers_on(ax, A, "By")
    for fl, ls in (("hlld", "-"), ("hlle", (0, (4, 2)))):
        for lim, col, nm in (("mc", MC, "MC"), ("weno5z", WENO, "WENO5-Z")):
            if (fl, lim, 16384) in runs:
                _run_line(ax, runs, (fl, lim, 16384), "By", col, ls=ls, lw=1.3,
                          label="%s + %s, 16384" % (fl.upper(), nm))
    ax.set_ylim(-0.52, -0.22)
    ax.set_title("B_y at 16384 cells: HLLD (solid) and HLLE (dashed)", fontsize=12, color=INK)
    ax.legend(fontsize=8, loc="upper right")
    for ax in axs[0]:
        ax.set_xlim(X0, X1)
        ax.set_xlabel("x  (t = 0.4); shaded: the compound fan", color=MUTED)
        ax.grid(alpha=0.25)
    flux, lim, n = M["flux"], M["lim"], M["n"]
    for ax, l, name in ((axs[1, 0], "mc", "MC (PLM)"), (axs[1, 1], "weno5z", "WENO5-Z")):
        for fl, ls, mk in (("hlld", "-", "o"), ("hlle", "--", "s")):
            m = (flux == fl) & (lim == l)
            o = np.argsort(n[m])
            ax.loglog(n[m][o], M["local"][m][o, 0, 0], ls, marker=mk, color=OURS, lw=1.6, ms=6,
                      label="%s + %s -> ours" % (fl.upper(), name))
            ax.loglog(n[m][o], M["local"][m][o, 0, 1], ls, marker=mk, color=COMP, lw=1.6, ms=6,
                      mfc="white", label="%s + %s -> compound" % (fl.upper(), name))
        ax.axhline(own, color=MUTED, lw=1, ls=":")
        ax.text(240, own * 0.91, "the two answers apart: %.3f" % own, fontsize=8.5, color=MUTED, va="top")
        ax.set_xlim(220, 45000); ax.set_ylim(8e-3, 0.4)
        ax.set_xlabel("cells", color=MUTED)
        ax.set_ylabel("mean |B_y - answer|, xi 0.026-0.050", color=MUTED)
        ax.set_title("%s: distance to each exact answer near the left slow wave" % name, fontsize=12, color=INK)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8, loc="upper right")
    ax = axs[1, 2]
    for fl, ls, mk in (("hlld", "-", "o"), ("hlle", "--", "s")):
        for l, col, nm in (("mc", MC, "MC"), ("weno5z", WENO, "WENO5-Z")):
            m = (flux == fl) & (lim == l) & (n >= 1024)
            o = np.argsort(n[m])
            ax.semilogx(n[m][o], 1e3 * M["plateau"][m][o, 0, 0], ls, marker=mk, color=col, lw=1.6,
                        ms=6, label="%s + %s" % (fl.upper(), nm))
    ax.axhline(0.0, color=OURS, lw=1.6, label="ours")
    ax.axhline(1e3 * (A.pB["rho"] - A.p1["rho"]) / A.p1["rho"], color=COMP, lw=1.6, ls="--", label="compound")
    ax.set_xlabel("cells", color=MUTED)
    ax.set_ylabel("plateau density relative to ours  (x 1e-3)", color=MUTED)
    w = M["plateau_window"]
    ax.set_title("Star plateau density (median, xi %.3f-%.3f)" % (w[0], w[1]), fontsize=12, color=INK)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.suptitle("Balsara 1 at high resolution: the left slow wave against both exact answers (t = 0.4)",
                 fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


def family_figure(a):
    plt = _style()
    import balsara1_family as BF
    runs = load_all(a.data)
    A = Answers(os.path.join(a.data, "sept", "balsara1_compound.npz"), n_fan=301)
    Fz = np.load(a.family)
    fb, frho, fok = Fz["b"], Fz["rho_B"], Fz["ok"].astype(bool)
    Bt_ours = float(A.ans["Z"][1][5])
    foff = 1e3 * (frho - A.p1["rho"]) / A.p1["rho"]
    bstar = float(Fz["b_star"])
    offstar = float(foff[np.argmin(np.abs(fb - bstar))])
    # members for the profiles, continued from the stored unknowns
    mem = {}
    for b in (-0.44, -0.46, -0.47):
        i = int(np.argmin(np.abs(fb - b)))
        u, nrm = BF.solve(np.array([b]), [np.array([c]) for c in Fz["unk"][:, i]])
        _, _, st = BF.structure(u, np.array([b]))
        mem[b] = dict(X=BS.prim([float(c[0]) for c in st["X"]]), B=BS.prim([float(c[0]) for c in st["B"]]),
                      V1=float(st["V1"][0]), V2=float(st["V2"][0]))
    shock_side = fok & (fb <= Bt_ours + 1e-12)
    o = np.argsort(foff[shock_side])
    b_of_off = lambda off: np.interp(off, foff[shock_side][o], fb[shock_side][o])
    pts = []
    print("run                 b behind its shock   recent plateau -> b   early plateau -> b")
    for key in (("hlle", "mc", 8192), ("hlle", "mc", 16384), ("hlle", "mc", 32768),
                ("hlld", "mc", 8192), ("hlld", "mc", 16384)):
        if key not in runs:
            continue
        r = runs[key]
        xi = r["xi"]
        brun = float(r["By"][(xi > 0.0315) & (xi < 0.0400)].min())
        rec = 1e3 * (np.median(r["rho"][(xi > 0.045) & (xi < 0.060)]) - A.p1["rho"]) / A.p1["rho"]
        early = 1e3 * (np.median(r["rho"][(xi > 0.200) & (xi < 0.235)]) - A.p1["rho"]) / A.p1["rho"]
        pts.append((key, brun, rec))
        print("%-4s %-3s %6d       %+.4f          %+.3f e-3 -> %+.4f   %+.3f e-3 -> %+.4f"
              % (key + (brun, rec, b_of_off(rec), early, b_of_off(early))))
    fig, axs = plt.subplots(2, 2, figsize=(15.5, 11))
    for ax, k, lab, ylim in ((axs[0, 0], "rho", "density", (0.66, 0.79)), (axs[0, 1], "By", "B_y", (-0.52, -0.22))):
        _answers_on(ax, A, k, lw=2)
        m = mem[-0.47]
        x1, x2 = 0.5 + T * m["V1"], 0.5 + T * m["V2"]
        ax.plot([X0, x1, x1, x2, x2, X1], [A.p0[k], A.p0[k], m["X"][k], m["X"][k], m["B"][k], m["B"][k]],
                "-.", color="#C2410C", lw=2, zorder=6, label="exact, b = -0.47: intermediate shock + slow shock")
        for key, c, name in ((("hlle", "mc", 32768), "#E6550D", "HLLE + MC, 32768"),
                             (("hlld", "mc", 16384), "#A63603", "HLLD + MC, 16384"),
                             (("hlle", "weno5z", 16384), "#2171B5", "HLLE + WENO5-Z, 16384")):
            if key in runs:
                _run_line(ax, runs, key, k, c, lw=1.0, label=name)
        ax.set_xlim(X0, X1); ax.set_ylim(*ylim)
        ax.set_xlabel("x  (t = 0.4); shaded: the compound fan", color=MUTED)
        ax.set_title("%s near the left slow wave" % lab, fontsize=12, color=INK)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    ax = axs[1, 0]
    fan_side = fok & (fb >= Bt_ours - 1e-12)
    ax.axvspan(bstar, fb[fok].max(), color="#F4F4F4", lw=0, zorder=0)
    ax.plot(fb[fan_side], foff[fan_side], "-o", color=COMP, ms=4, lw=1.8,
            label="intermediate shock + detached slow rarefaction")
    o2 = np.argsort(fb[shock_side])
    ax.plot(fb[shock_side][o2], foff[shock_side][o2], "-o", color="#C2410C", ms=4, lw=1.8,
            label="intermediate shock + slow shock")
    ax.plot([Bt_ours], [0.0], "o", color=OURS, ms=10, zorder=7, label="b = %.3f: intermediate shock alone (ours, GR06)" % Bt_ours)
    ax.plot([fb[fok].max()], [foff[fok][np.argmax(fb[fok])]], "D", color=COMP, ms=9, zorder=7,
            label="b = %.3f: compound wave (shock sonic, fan attached)" % fb[fok].max())
    ax.plot([bstar], [offstar], "*", color="#1B7837", ms=16, zorder=7,
            label="b = %.3f: 180-degree rotational discontinuity + slow shock (regular)" % bstar)
    shades = {8192: "#FD8D3C", 16384: "#E6550D", 32768: "#A63603"}
    for key, brun, rec in pts:
        ax.plot([brun], [rec], "s" if key[0] == "hlle" else "^", color=shades[key[2]], mec="#5A2A0A",
                mew=0.8, ms=8, zorder=8)
    ax.plot([], [], "s", color="#E6550D", mec="#5A2A0A", label="HLLE + MC runs (8192 -> 32768, darker = finer)")
    ax.plot([], [], "^", color="#E6550D", mec="#5A2A0A", label="HLLD + MC runs (8192, 16384)")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlim(-0.66, -0.29); ax.set_ylim(-2.0, 12.5)
    ax.set_xlabel("b = B_y behind the left field reversal (the family's parameter)", color=MUTED)
    ax.set_ylabel("star-plateau density relative to ours  (x 1e-3)", color=MUTED)
    ax.set_title("Star-plateau density against b: exact solutions (lines) and MC runs (markers)", fontsize=12, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.6, loc="upper right")
    ax = axs[1, 1]
    if ("hlle", "mc", 32768) in runs:
        r = runs[("hlle", "mc", 32768)]
        mm = (r["xi"] > 0.03) & (r["xi"] < 0.25)
        ax.plot(r["xi"][mm], r["rho"][mm], "-", color="#E6550D", lw=0.9, label="HLLE + MC, 32768")
    for b, ls in ((-0.44, ":"), (-0.46, "--"), (-0.47, "-.")):
        ax.axhline(mem[b]["B"]["rho"], color="#C2410C", ls=ls, lw=1.2, label="exact member b = %.2f" % b)
    ax.axhline(A.p1["rho"], color=OURS, lw=1.6, label="ours (b = %.3f)" % Bt_ours)
    ax.axhline(A.pB["rho"], color=COMP, lw=1.6, ls="--", label="compound")
    ax.set_ylim(0.7075, 0.7110)
    ax.set_xlabel("xi = (x - 0.5) / t   (gas near the contact passed the left wave early, near xi 0.04 late)",
                  color=MUTED)
    ax.set_ylabel("density", color=MUTED)
    ax.set_title("Star plateau of HLLE + MC at 32768 cells, with exact plateau levels", fontsize=12, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    fig.suptitle("Balsara 1 (t = 0.4): exact solutions with a left field reversal, and finite-volume runs",
                 fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(a.out, dpi=125)
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("measure", "figure", "family-figure"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--measure")
    ap.add_argument("--family")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"measure": measure, "figure": figure, "family-figure": family_figure}[a.mode](a)


if __name__ == "__main__":
    main()
