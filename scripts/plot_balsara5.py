"""Figure: Balsara test 5 -- exact solution, HLLD flux, exact flux.

Reads the npz written by scripts/balsara5_slowwave.py and draws the six
primitive profiles at t_end with the seven waves marked.  The point of the
figure is the region between each Alfven wave and its slow wave: HLLD has no
slow waves, so it cannot resolve that structure however fine the grid, while
the exact flux places it where the self-similar solution has it.

    python scripts/plot_balsara5.py results/balsara5/balsara5_400.npz
    python scripts/plot_balsara5.py results/slow_shock/balsara5_400.npz \
        --name "Slow-shock tube" --out docs/figs/slow_shock_tube
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("npz", nargs="?", default="results/balsara5/balsara5_400.npz")
ap.add_argument("--name", default="Balsara test 5", help="problem name for the title")
ap.add_argument("--out", default="docs/figs/balsara5_slowwave", help="output stem")
args = ap.parse_args()
f = args.npz
d = np.load(f, allow_pickle=True)
x, xf = d["x"], d["x_fine"]
t, x0 = float(d["t_end"]), float(d["x0"])
sp = d["speeds"]                       # LF LA LS CD RS RA RF
names = ["LF", "LA", "LS", "CD", "RS", "RA", "RF"]
xw = x0 + sp * t
n = int(d["ncells"])

panels = [("rho", r"$\rho$"), ("p", r"$p_{\rm gas}$"), ("vx", r"$v_x$"),
          ("vy", r"$v_y$"), ("By", r"$B_y$"), ("Bz", r"$B_z$")]
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
for ax, (v, lab) in zip(axes.ravel(), panels):
    ax.plot(xf, d["fine_" + v], color="k", lw=1.3, zorder=3,
            label="exact solution (self-similar)")
    ax.plot(x, d["hlld_" + v], "o", ms=3.2, mfc="none", mec="#D85A30", mew=0.9,
            zorder=4, label="HLLD flux, %d cells" % n)
    ax.plot(x, d["exact_" + v], "s", ms=2.6, color="#1D9E75", zorder=5,
            label="exact flux, %d cells" % n)
    # the slow-wave regions HLLD cannot represent
    ax.axvspan(min(xw[1], xw[2]), max(xw[1], xw[2]), color="#F5C4B3", alpha=0.45, lw=0)
    ax.axvspan(min(xw[4], xw[5]), max(xw[4], xw[5]), color="#F5C4B3", alpha=0.45, lw=0)
    for k, xx in enumerate(xw):
        ax.axvline(xx, color="0.55", lw=0.6, ls="--", zorder=1)
    ax.set_ylabel(lab, fontsize=12)
    ax.grid(alpha=0.25)
    ax.set_xlim(x.min(), x.max())
# wave labels once, along the top of the first row.  Neighbouring waves can
# sit within a cell of each other (an Alfven wave and its slow wave usually
# do), so a label whose neighbour is closer than 4.5% of the axis goes on a
# second row instead of on top of it.
# Each label takes the lowest row whose previous label is far enough away,
# so three near-coincident waves get three rows rather than one pile.
span = x.max() - x.min()
row = np.zeros(7, dtype=int)
last = {}                                   # row -> x of the last label placed there
for k in np.argsort(xw):
    r = 0
    while r in last and xw[k] - last[r] < 0.045 * span:
        r += 1
    row[k] = r
    last[r] = xw[k]
for k, xx in enumerate(xw):
    for a in axes[0]:
        a.text(xx, 1.015 + 0.075 * row[k], names[k], transform=a.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=8.5, color="0.3")
for a in axes[1]:
    a.set_xlabel(r"$x$", fontsize=12)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9.5,
           frameon=False, bbox_to_anchor=(0.5, 0.005))
fig.suptitle(r"%s at $t=%.1f$: shaded = between each Alfvén wave and its slow wave, "
             r"the structure HLLD has no wave for" % (args.name, t), fontsize=11.5, y=0.995)
fig.tight_layout(rect=(0, 0.045, 1, 0.955))
for ext in ("pdf", "png"):
    fig.savefig("%s.%s" % (args.out, ext), dpi=170 if ext == "png" else None)
print("wrote %s.{pdf,png}" % args.out)
print("waves x(t): " + "  ".join("%s=%.3f" % (nm, xx) for nm, xx in zip(names, xw)))
for v, _ in panels:
    print("  L1 %-3s  HLLD %.3e   exact %.3e   ratio %.2f"
          % (v, d["L1_hlld_" + v], d["L1_exact_" + v], d["L1_hlld_" + v] / max(d["L1_exact_" + v], 1e-300)))
