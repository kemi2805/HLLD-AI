"""Two-panel version of the mechanism figure for the meeting slides (2026-11-04):
(a) how close the answer's slow wave runs to the Alfven speed, by group;
(b) the seven-wave Newton from starts near the known answer (seed ceiling)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from feats import load                                   # noqa: E402

plt.rcParams.update({"font.size": 15})
COL = ("#999999", "#CC79A7", "#0072B2")
LAB = ("seven-wave, first try", "seven-wave, after a retry", "planar rescue")
d = load(os.path.join(HERE, "data", "sample.npz"))
g, w = d["group"], d["weight"]
LSs, RSs = d["f_LS_str"].astype(float), d["f_RS_str"].astype(float)
mL = np.where(LSs > 1e-3, np.abs(d["f_LS_alfven_margin"].astype(float)), np.inf)
mR = np.where(RSs > 1e-3, np.abs(d["f_RS_alfven_margin"].astype(float)), np.inf)
marg = np.minimum(mL, mR)
fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True)
a = ax[0]
bins = np.logspace(-8, 0, 49)
for gi in (0, 2):
    m = (g == gi) & np.isfinite(marg)
    h, e = np.histogram(np.maximum(marg[m], 1.1e-8), bins=bins, weights=w[m])
    a.step(e[:-1], h / w[g == gi].sum(), where="post", color=COL[gi], lw=2.6, label=LAB[gi])
a.axvline(1e-4, color="#666666", lw=1.2, ls="--")
a.set_xscale("log")
a.set_xlabel("slow wave's distance to the Alfvén speed, |s − λ_A|")
a.set_ylabel("fraction of the group")
a.set_title("(a) planar answers: the slow wave rides the Alfvén speed", fontsize=16, loc="left")
a.legend(frameon=False, fontsize=14, loc="upper left")
a.spines[["top", "right"]].set_visible(False)
b = ax[1]
S = json.load(open(os.path.join(HERE, "data", "seed_ceiling.json")))
r = np.array(S["radius"]); xr = np.where(r == 0, 1e-4, r)
for gi, gn in enumerate(("first", "retry", "planar")):
    if gi == 1:
        continue
    b.plot(xr, 100 * np.array(S["conv"][gn]), marker="o", ms=8, lw=2.6, color=COL[gi], label=LAB[gi],
           mec="white", mew=1.2)
    b.scatter([3.5e-5], [100 * S["net"][gn]], marker="D", s=70, color=COL[gi], edgecolor="white", zorder=3)
b.set_xscale("log"); b.set_xlim(2.2e-5, 0.16); b.set_ylim(-3, 103)
b.set_xticks([3.5e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
b.set_xticklabels(["net's\nseed", "0", "3e-4", "1e-3", "3e-3", "0.01", "0.03", "0.1"], fontsize=13)
b.minorticks_off()
b.set_xlabel("start: the known answer moved by a step of this size")
b.set_ylabel("% converged (one attempt)")
b.set_title("(b) not the seed: a hair from the answer it still fails", fontsize=16, loc="left")
b.legend(frameon=False, fontsize=14, loc="center right")
b.spines[["top", "right"]].set_visible(False)
fig.savefig(os.path.join(HERE, "meeting_2026-11-04", "c_mechanism.png"), dpi=150)
print("ok")
