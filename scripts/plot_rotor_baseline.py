import glob, os, csv, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, "/Users/miler/Codes/HLLD/scripts")
import rotor_compare as RC

R = "/Users/miler/Codes/HLLD/results"
runs = {64: f"{R}/rotor_64_hlld", 128: f"{R}/rotor_128_hlld", 256: f"{R}/rotor_256_hlld"}
col = {64: "#2a78d6", 128: "#eb6834", 256: "#1baf7a"}          # categorical slots 1-3, validated
dash = {64: "-", 128: "--", 256: "-."}                          # secondary encoding
ink, ink2, grid = "#0b0b0b", "#52514e", "#e6e5e2"
seq = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
S = {n: RC.load_snaps(d) for n, d in runs.items()}

fig = plt.figure(figsize=(14, 12.5), facecolor="#fcfcfb")
gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.0, 1.15], hspace=0.55, wspace=0.32)
def style(ax, title, xl=None, yl=None):
    ax.set_facecolor("#fcfcfb"); ax.set_title(title, color=ink, fontsize=11, loc="left", pad=8)
    ax.grid(True, color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=9)
    if xl: ax.set_xlabel(xl, color=ink2, fontsize=9)
    if yl: ax.set_ylabel(yl, color=ink2, fontsize=9)

# ── row 1: relative L1 vs t, one panel per quantity, two resolution pairs ──
pairs = [((64, 128), "64² vs 128²", "#2a78d6", "-"), ((128, 256), "128² vs 256²", "#eb6834", "--")]
for j, (key, name) in enumerate((("rho", "ρ"), ("p", "p"), ("B", "|B|"))):
    ax = fig.add_subplot(gs[0, j]); style(ax, f"relative L1 in {name}, coarse vs finer", "t", "L1 / L1(finer)")
    for (a, b), lab, c, ls in pairs:
        ts, vals = [], []
        for t, u in RC.match_times(sorted(S[a]), sorted(S[b])):
            sa, sb = S[a][t], S[b][u]
            fa = sa[key] if key != "B" else RC.magB(sa); fb = sb[key] if key != "B" else RC.magB(sb)
            ca, cb = RC.to_common(fa, fb); ts.append(t); vals.append(RC.l1_rel(ca, cb))
        ax.plot(ts, vals, ls, color=c, lw=2, marker="o", ms=4)
        ax.annotate(f"{lab}  {vals[-1]*100:.0f}%", (ts[-1], vals[-1]), xytext=(-4, 6 if a == 64 else -12),
                    textcoords="offset points", ha="right", fontsize=8.5, color=ink)
    ax.set_ylim(0, None)

# ── row 2: front radius vs t (3 resolutions) + the two L1(rho) curves as ratio ──
ax = fig.add_subplot(gs[1, 0:2]); style(ax, "fast-front radius along y=0 vs t", "t", "r_front")
for n in (64, 128, 256):
    ts = sorted(S[n]); rs = [RC.front_radius(S[n][t]) for t in ts]
    ax.plot(ts, rs, dash[n], color=col[n], lw=2, marker="o", ms=3.5)
    ax.annotate(f"HLLD {n}²", (ts[-1], rs[-1]), xytext=(6, {64: 6, 128: 0, 256: -8}[n]), textcoords="offset points", fontsize=8.5, color=ink)
ax.set_xlim(0, 0.44)
ax = fig.add_subplot(gs[1, 2]); style(ax, "L1(ρ) at t=0.4 per pair", None, "relative L1")
labels, vals = [], []
for (a, b), lab, c, ls in pairs:
    t, u = RC.match_times(sorted(S[a]), sorted(S[b]))[-1]
    ca, cb = RC.to_common(S[a][t]["rho"], S[b][u]["rho"]); labels.append(lab); vals.append(RC.l1_rel(ca, cb))
bars = ax.bar(labels, vals, color=["#2a78d6", "#eb6834"], width=0.55)
for bar, v in zip(bars, vals): ax.text(bar.get_x() + bar.get_width()/2, v + 0.008, f"{v*100:.1f}%", ha="center", fontsize=9.5, color=ink)
ax.set_ylim(0, 0.42); ax.tick_params(axis="x", labelsize=9.5)

# ── row 3: cuts along y=0 at t=0.4 ──
for j, (key, name) in enumerate((("rho", "ρ"), ("p", "p"), ("B", "|B|"))):
    ax = fig.add_subplot(gs[2, j]); style(ax, f"{name} along y=0 at t=0.4", "x", name)
    for n in (64, 128, 256):
        t = max(S[n]); s = S[n][t]; f = s[key] if key != "B" else RC.magB(s); m = f.shape[0]
        ax.plot(s["x"], f[m // 2], dash[n], color=col[n], lw=1.6 if n != 256 else 2.0)
    if j == 0:
        for n in (64, 128, 256):
            t = max(S[n]); s = S[n][t]; m = s["rho"].shape[0]
            ax.annotate(f"{n}²", (s["x"][m//2 + m//6], s["rho"][m//2][m//2 + m//6]), xytext=(0, {64: 14, 128: 0, 256: -14}[n]), textcoords="offset points", fontsize=8.5, color=col[n] if n != 256 else ink)

# ── row 4: rho maps at t=0.4 ──
vmin = min(S[n][max(S[n])]["rho"].min() for n in runs); vmax = max(S[n][max(S[n])]["rho"].max() for n in runs)
for j, n in enumerate((64, 128, 256)):
    ax = fig.add_subplot(gs[3, j]); t = max(S[n]); s = S[n][t]
    im = ax.imshow(s["rho"], origin="lower", extent=[s["x"][0], s["x"][-1], s["y"][0], s["y"][-1]], cmap=seq, vmin=vmin, vmax=vmax)
    ax.set_title(f"ρ at t={t:.2f}, HLLD {n}²", color=ink, fontsize=11, loc="left", pad=8); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(grid)
cb = fig.colorbar(im, ax=fig.axes[-3:], fraction=0.025, pad=0.02); cb.ax.tick_params(colors=ink2, labelsize=8); cb.outline.set_edgecolor(grid)

fig.suptitle("Magnetic rotor, HLLD baseline: 64² / 128² / 256²  —  the bar the exact flux has to clear is 'exact at N ≈ HLLD at 2N'",
             color=ink, fontsize=12.5, x=0.02, ha="left", y=0.995)
fig.text(0.02, 0.965, "L1 between resolution levels is the reference's own convergence: 32.8% (64²→128²) and 26.8% (128²→256²) in ρ at t=0.4. "
         "Exact-64² vs HLLD-128² must come in below 32.8%; near 26.8% would support the claim.", color=ink2, fontsize=9.5)
out = "/Users/miler/Codes/HLLD/figs/rotor_hlld_baseline.png"; fig.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)
