"""The four-quadrant problem's flow structures, named on the density at t = 0.4,
with the lost faces of t = 0.3-0.4 by type (2026-11-04; user: "I dont know
what rings around the roll up and the jet head, the shear line ends mean")."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from feats import load                                   # noqa: E402

plt.rcParams.update({"font.size": 16})
s = np.load(os.path.join(os.path.dirname(HERE), "runs64", "ft3_128", "snap_008.npz"), allow_pickle=True)
d = load(os.path.join(HERE, "data", "sample.npz"))
m = (d["group"] == 4) & d["aligned"] & d["phys"] & (d["t"] > 0.1)
# the flow grows in time but keeps its shape (self-similar): every lost face
# after t = 0.1 drawn where it would sit at t = 0.4
X, Y = d["x"] * 0.4 / d["t"], d["y"] * 0.4 / d["t"]
fn, beta, gap = d["f_fn_max"].astype(float), d["f_beta_min"].astype(float), d["gapAS"]
A = (gap < 1e-3) & (fn <= 0.9)
Bw = (fn > 0.9) & (beta >= 1)
Bs = (fn > 0.9) & (beta < 1)
fig, ax = plt.subplots(figsize=(10.5, 10), constrained_layout=True)
ax.imshow(np.log10(s["rho"]).T, origin="lower", extent=[-0.5, 0.5, -0.5, 0.5], cmap="Greys", alpha=0.8,
          interpolation="nearest")
for sel, col, lab in ((A, "#D55E00", "A: slow ≈ Alfvén speed"), (Bw, "#0072B2", "B, weak field"),
                      (Bs, "#009E73", "B, strong field")):
    k = m & sel
    ax.scatter(X[k], Y[k], s=7, color=col, lw=0, alpha=0.45, label=lab)
box = dict(boxstyle="round,pad=0.35", fc="white", ec="#444444", lw=0.8, alpha=0.95)
arrow = dict(arrowstyle="-|>", color="#222222", lw=1.4)
# the four initial quadrants
ax.text(-0.47, 0.44, "moving right at 0.99c", fontsize=15, va="top", bbox=box)
ax.text(0.47, -0.44, "moving up at 0.99c", fontsize=15, ha="right", va="bottom", bbox=box)
ax.text(-0.47, -0.44, "dense gas at rest", fontsize=15, va="bottom", bbox=box)
ax.text(0.47, 0.06, "low pressure,\nat rest", fontsize=15, ha="right", va="bottom", bbox=box)
# the structures
ax.annotate("shear layer", xy=(-0.40, 0.005), xytext=(-0.46, 0.16), fontsize=15, bbox=box, arrowprops=arrow)
ax.annotate("shear layer", xy=(0.012, -0.30), xytext=(0.10, -0.17), fontsize=15, bbox=box, arrowprops=arrow)
ax.annotate("dense tongue", xy=(0.15, 0.18), xytext=(-0.10, 0.30), fontsize=15, bbox=box, arrowprops=arrow)
ax.annotate("tip of the tongue (\"jet head\")", xy=(0.40, 0.40), xytext=(-0.08, 0.465), fontsize=15, bbox=box,
            arrowprops=arrow, va="top")
ax.annotate("low-density pocket where\nthe two shear layers meet\n(\"roll-up\")", xy=(-0.1, -0.1), xytext=(-0.46, -0.30),
            fontsize=15, bbox=box, arrowprops=arrow)
ax.set_xticks([]); ax.set_yticks([])
ax.legend(loc="lower right", fontsize=14, markerscale=3, frameon=True, framealpha=0.95,
          title="lost faces, all t > 0.1,\ndrawn where they sit at t = 0.4", title_fontsize=13, bbox_to_anchor=(1.0, 0.10))
ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)
fig.savefig(os.path.join(HERE, "meeting_2026-11-04", "c_structures.png"), dpi=140)
print("ok")
