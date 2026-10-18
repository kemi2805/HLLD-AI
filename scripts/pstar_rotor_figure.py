"""The rotor with hlld vs hlld_ai: log10 rho of both at t_end, their
difference, and the pi-rotation symmetry error over time.

    python scripts/pstar_rotor_figure.py RUNS_DIR --n 256 --out figs/pstar_rotor_256.png

RUNS_DIR holds calea_pstar_rotor.sh's run directories ({hlld,ai,ws}_<n>,
each with snap_fin.npz and diag.csv).  Colours: one-hue sequential ramp for
log10 rho, blue <-> red through a gray midpoint for the difference, the
first three categorical slots for the runs (dataviz reference palette;
validated all-pairs), resolution as line style.  Titles say what is shown,
nothing more.
"""
import argparse
import csv
import os

import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
       "#0d366b"]
DIV = ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f0a0a0", "#e34948",
       "#9b1c1c"]
RUN_COLOR = {"hlld": "#2a78d6", "ai": "#eb6834", "ws": "#1baf7a"}
RUN_LABEL = {"hlld": "hlld", "ai": "hlld_ai one-shot", "ws": "hlld_ai warm start"}


def read_sym(path):
    t, s = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            v = row.get("sym_err", "nan")
            if v not in ("", "nan"):
                t.append(float(row["t"]))
                s.append(float(v))
    return np.array(t), np.array(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-label", default="v3_s1")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "text.color": INK, "axes.labelcolor": INK2,
                         "axes.edgecolor": "#c3c2b7", "xtick.color": INK2,
                         "ytick.color": INK2, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    seq = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    div = LinearSegmentedColormap.from_list("div_blue_red", DIV)

    run = lambda m, n: os.path.join(a.runs, f"{m}_{n}")
    zh = np.load(os.path.join(run("hlld", a.n), "snap_fin.npz"))
    za = np.load(os.path.join(run("ai", a.n), "snap_fin.npz"))
    t_h, t_a = float(zh["t"]), float(za["t"])
    x, y = zh["x"], zh["y"]
    ext = [x[0] - 0.5 * (x[1] - x[0]), x[-1] + 0.5 * (x[1] - x[0]),
           y[0] - 0.5 * (y[1] - y[0]), y[-1] + 0.5 * (y[1] - y[0])]
    lh, la = np.log10(zh["rho"]), np.log10(za["rho"])
    lo, hi = min(lh.min(), la.min()), max(lh.max(), la.max())
    d = za["rho"] - zh["rho"]
    lim = float(np.abs(d).max()) or 1.0

    fig = plt.figure(figsize=(15.5, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.32)
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax, f, title in ((axs[0], lh, f"log10 rho - hlld, {a.n}^2, t = {t_h:.3f}"),
                         (axs[1], la, f"log10 rho - hlld_ai one-shot ({a.ckpt_label}), "
                                      f"{a.n}^2, t = {t_a:.3f}")):
        im = ax.imshow(f.T, origin="lower", extent=ext, cmap=seq, vmin=lo, vmax=hi,
                       interpolation="nearest")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8, colors=INK2)
    im = axs[2].imshow(d.T, origin="lower", extent=ext, cmap=div,
                       norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                       interpolation="nearest")
    axs[2].set_title(f"rho(hlld_ai) - rho(hlld), {a.n}^2, t = {t_a:.3f}",
                     fontsize=9, color=INK)
    axs[2].set_xlabel("x"); axs[2].set_ylabel("y")
    cb = fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=8, colors=INK2)

    ax = axs[3]
    for n, ls in ((a.n, "-"), (64, "--")):
        for m in ("hlld", "ai", "ws"):
            f = os.path.join(run(m, n), "diag.csv")
            if not os.path.exists(f):
                continue
            t, s = read_sym(f)
            if t.size == 0:
                continue
            ax.semilogy(t, s, ls, color=RUN_COLOR[m], lw=1.6,
                        label=f"{RUN_LABEL[m]}, {n}^2")
            ax.annotate(f"{RUN_LABEL[m]} {n}^2", (t[-1], s[-1]),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=INK2, va="center")
    ax.set_title("pi-rotation symmetry error (diag.csv)", fontsize=9, color=INK)
    ax.set_xlabel("t"); ax.set_ylabel("symmetry error")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_xlim(0, max(t_h, t_a) * 1.28)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"{a.out}: max |rho diff| {lim:.3e}, mean |rho diff| {np.abs(d).mean():.3e}")


if __name__ == "__main__":
    main()
