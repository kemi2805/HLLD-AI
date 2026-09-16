"""Eval-set scores of the warm-start checkpoints, paired against the incumbent.

    python scripts/plot_ckpt_scores.py data/scores_all8.npz
"""
import sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rmhd.paths as _rp
f = sys.argv[1] if len(sys.argv) > 1 else _rp.resolve("data/scores_all8.npz")
d = np.load(f, allow_pickle=True)
names = [str(s) for s in d["names"]]
conv = d["converged"]          # (n_ckpt, n_lane) bool
Bx = np.abs(d["Bx"])
base = conv[0]                 # first checkpoint is the baseline

def mcnemar(b, o):
    r, k = int((~b & o).sum()), int((b & ~o).sum())
    n = r + k
    if n == 0: return 1.0, r, k
    t = sum(math.comb(n, j) for j in range(min(r, k) + 1)) / 2.0 ** n
    return min(1.0, 2 * t), r, k

rate = conv.mean(axis=1) * 100
p = [mcnemar(base, conv[i])[0] for i in range(len(names))]
new = [i for i, n in enumerate(names) if n.startswith("all_s")]
inc = [i for i, n in enumerate(names) if not n.startswith("all_s")]

fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8),
                       gridspec_kw={"width_ratios": [1.25, 1]})
# ---- panel 1: overall rate ----
a = ax[0]
order = list(range(len(names)))
cols = ["#1D9E75" if i in new else "#888780" for i in order]
cols[0] = "#D85A30"
bars = a.bar(range(len(order)), [rate[i] for i in order], color=cols, width=0.68)
a.axhline(rate[0], color="#D85A30", lw=1.0, ls="--", zorder=0)
for i, b in zip(order, bars):
    if i == 0: continue
    star = "*" if p[i] < 0.05 else ""
    a.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.18,
           f"{rate[i]-rate[0]:+.1f}{star}", ha="center", fontsize=8.5,
           color="#0F6E56" if p[i] < 0.05 else "0.45")
a.set_xticks(range(len(order)))
a.set_xticklabels([names[i] for i in order], rotation=45, ha="right", fontsize=9)
a.set_ylabel("converged on evalset_rotor_600 (%)")
a.set_ylim(min(rate) - 1.5, max(rate) + 1.6)
a.grid(axis="y", alpha=0.25)
a.set_title("Eval set, budget 3 — bars are new (green), incumbents (grey), baseline (orange)\n"
            "labels are $\\Delta$ vs baseline; * = McNemar $p<0.05$", fontsize=9.5)

# ---- panel 2: per-|Bx| bin, best new vs incumbent trio members ----
BINS = (0.0, 0.5, 1.5, 3.0, 6.0)
best = max(new, key=lambda i: rate[i])
show = [0, best] + [i for i in inc if i != 0]
w = 0.8 / len(show)
b2 = ax[1]
for j, i in enumerate(show):
    ys, ns = [], []
    for k in range(len(BINS) - 1):
        m = (Bx >= BINS[k]) & (Bx < BINS[k + 1])
        ys.append(100 * conv[i][m].mean() if m.sum() else np.nan)
        ns.append(int(m.sum()))
    c = "#D85A30" if i == 0 else ("#1D9E75" if i == best else "#B4B2A9")
    b2.bar(np.arange(len(ys)) + j * w, ys, width=w, label=names[i], color=c)
b2.set_xticks(np.arange(len(BINS) - 1) + 0.4 - w / 2)
b2.set_xticklabels([f"[{BINS[k]:.1f},{BINS[k+1]:.1f})\nn={ns[k]}" for k in range(len(BINS) - 1)], fontsize=8.5)
b2.set_xlabel(r"$|B_x|$ bin")
b2.set_ylabel("converged (%)")
b2.legend(fontsize=8.5, frameon=False)
b2.grid(axis="y", alpha=0.25)
b2.set_title(f"By field strength: best new ({names[best]}) vs incumbents", fontsize=9.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"docs/figs/ckpt_scores.{ext}", dpi=170 if ext == "png" else None)
print("wrote docs/figs/ckpt_scores.{pdf,png}")
print(f"{'checkpoint':<16}{'conv':>8}{'delta':>8}{'p':>10}")
for i in range(len(names)):
    print(f"{names[i]:<16}{rate[i]:7.1f}%{rate[i]-rate[0]:+8.1f}{p[i]:10.4g}")
