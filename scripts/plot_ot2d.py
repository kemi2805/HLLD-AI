"""Orszag-Tang in 2D: the 512^2 reference field by field, its current sheets
over time, the 64^2 runs beside it, and where the exact solver fails.

    python scripts/plot_ot2d.py BASE OUTDIR

BASE holds ot_open/ot_512_hlld, ot_open/ot_64_hlld and canon/ot_64_exact_pool,
canon/ot_64_exact_canon2 (each a copy of results/<run>: snapshots, and for
the last its harvest/coverage_*.npz and diag.csv).
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import plot_rotor_output as PRO                                 # noqa: E402

S, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
GAM = 4.0 / 3.0
REF = S + "/ot_open/ot_512_hlld"
RUNS64 = [("ot_open/ot_64_hlld", "HLLD 64²"), ("canon/ot_64_exact_pool", "exact 64², plain"),
          ("canon/ot_64_exact_canon2", "exact 64², canonical")]


def snaps(run):
    return {round(float(np.load(f)["t"]), 2): dict(np.load(f))
            for f in sorted(glob.glob(os.path.join(S, run, "snap_0*.npz")))}


def derived(z):
    """Lorentz factor, plasma beta, magnetisation, current and vorticity."""
    x, y = z["x"], z["y"]
    dx, dy = x[1] - x[0], y[1] - y[0]
    vx, vy, Bx, By, rho, p = (z[k] for k in ("vx", "vy", "Bx", "By", "rho", "p"))
    W = 1.0 / np.sqrt(1.0 - vx ** 2 - vy ** 2)
    vB = vx * Bx + vy * By
    b2 = (Bx ** 2 + By ** 2) / W ** 2 + vB ** 2
    h = 1.0 + GAM / (GAM - 1.0) * p / rho
    ddx = lambda f: (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * dx)   # periodic
    ddy = lambda f: (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * dy)
    return dict(W=W, beta=2 * p / np.maximum(b2, 1e-30), sigma=b2 / (rho * h),
                Jz=ddx(By) - ddy(Bx), wz=ddx(vy) - ddy(vx))


def show(ax, z, f, title, cmap="viridis", log=False, sym=False, lim=None):
    if log:
        f = np.log10(np.maximum(f, 1e-30))
    if sym:
        L = lim if lim is not None else np.percentile(np.abs(f), 99.5)
        im = ax.pcolormesh(z["x"], z["y"], f.T, cmap="RdBu_r", vmin=-L, vmax=L,
                           shading="auto", rasterized=True)
    else:
        lo, hi = lim if lim is not None else (np.percentile(f, 0.5), np.percentile(f, 99.5))
        im = ax.pcolormesh(z["x"], z["y"], f.T, cmap=cmap, vmin=lo, vmax=hi,
                           shading="auto", rasterized=True)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=11)
    x, y = np.asarray(z["x"]), np.asarray(z["y"])
    ax.set_xlim(x[0] - 0.5 * (x[1] - x[0]), x[-1] + 0.5 * (x[1] - x[0]))
    ax.set_ylim(y[0] - 0.5 * (y[1] - y[0]), y[-1] + 0.5 * (y[1] - y[0]))
    ax.tick_params(labelsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return im


ref = snaps("ot_open/ot_512_hlld")

# ── 1. the reference, field by field, at t = 0.5 and 1 ────────────────────
for t in (0.5, 1.0):
    z = ref[t]; d = derived(z)
    fig, axs = plt.subplots(2, 4, figsize=(19, 9.2), constrained_layout=True)
    show(axs[0, 0], z, z["rho"], "log10 ρ", log=True)
    show(axs[0, 1], z, z["p"], "log10 p", cmap="magma", log=True)
    show(axs[0, 2], z, d["W"], "Lorentz factor W", cmap="inferno")
    show(axs[0, 3], z, d["beta"], "log10 plasma β", cmap="cividis", log=True)
    show(axs[1, 0], z, d["Jz"], "current density J_z = ∂xBy − ∂yBx", sym=True)
    show(axs[1, 1], z, d["wz"], "vorticity ω_z", sym=True)
    show(axs[1, 2], z, d["sigma"], "log10 magnetisation σ = b²/ρh", cmap="plasma", log=True)
    show(axs[1, 3], z, np.hypot(z["Bx"], z["By"]), "|B| (lab frame)", cmap="viridis")
    fig.suptitle("Relativistic Orszag–Tang, 512² HLLD reference, t = %.2f" % t, fontsize=14)
    fig.savefig(os.path.join(OUT, "ot2d_reference_t%.2f.png" % t), dpi=110); plt.close(fig)

# ── 2. current sheets over time, and the centre at t = 1 ─────────────────
ts = [0.25, 0.5, 0.75, 1.0]
fig, axs = plt.subplots(1, 5, figsize=(24, 5.0), constrained_layout=True)
for ax, t in zip(axs, ts):
    show(ax, ref[t], derived(ref[t])["Jz"], "J_z, t = %.2f" % t, sym=True)
z = ref[1.0]; J = derived(z)["Jz"]
sl = slice(320, 448)                                        # x, y in [0.625, 0.875]
zz = dict(x=z["x"][sl], y=z["y"][sl])
show(axs[4], zz, J[sl, sl], "J_z, t = 1, the vortex [0.625, 0.875]²", sym=True)
fig.suptitle("Current sheets of the 512² reference", fontsize=14)
fig.savefig(os.path.join(OUT, "ot2d_current_sheets.png"), dpi=110); plt.close(fig)

# ── 3. the 64² runs beside the reference averaged to 64² ─────────────────
def coarsen(a, n=64):
    k = a.shape[0] // n
    return a.reshape(n, k, n, k).mean(axis=(1, 3))


for t in (0.5, 1.0):
    zs = [snaps(r)[t] for r, _ in RUNS64]
    zr = {k: coarsen(ref[t][k]) for k in ("rho", "p", "vx", "vy", "Bx", "By")}
    zr.update(x=zs[0]["x"], y=zs[0]["y"])
    cols = list(zip(zs + [zr], [n for _, n in RUNS64] + ["512² reference, averaged to 64²"]))
    fig, axs = plt.subplots(4, 4, figsize=(19, 18), constrained_layout=True)
    lr = np.log10(np.concatenate([c[0]["rho"].ravel() for c in cols]))
    jl = np.percentile(np.abs(derived(zr)["Jz"]), 99.5)
    for j, (z, name) in enumerate(cols):
        d = derived(z)
        show(axs[0, j], z, z["rho"], "%s\nlog10 ρ" % name, log=True, lim=(lr.min(), lr.max()))
        show(axs[1, j], z, d["Jz"], "J_z", sym=True, lim=jl)
        show(axs[2, j], z, d["W"], "W", cmap="inferno", lim=(1.0, 1.8))
    a, b, c = zs
    L = np.percentile(np.abs(c["rho"] - a["rho"]), 99.5)
    show(axs[3, 0], a, c["rho"] - a["rho"], "ρ: canonical exact − HLLD", sym=True, lim=L)
    show(axs[3, 1], a, c["rho"] - b["rho"], "ρ: canonical − plain exact", sym=True, lim=L)
    asym = lambda r: r - r[::-1, ::-1]
    La = max(np.abs(asym(b["rho"])).max(), 1e-30)
    show(axs[3, 2], a, asym(b["rho"]), "ρ − rotated ρ: plain exact", sym=True, lim=La)
    show(axs[3, 3], a, asym(c["rho"]), "ρ − rotated ρ: canonical (same scale)", sym=True, lim=La)
    fig.suptitle("Orszag–Tang 64², t = %.2f: the fluxes beside the reference" % t, fontsize=14)
    fig.savefig(os.path.join(OUT, "ot2d_64_compare_t%.2f.png" % t), dpi=100); plt.close(fig)

# ── 4. where the exact solver fails, over the current and the field ──────
run = os.path.join(S, "canon/ot_64_exact_canon2")
sn = [dict(np.load(f)) for f in sorted(glob.glob(run + "/snap_0*.npz"))]
cov, order = PRO.coverage_maps(run, sn, 64, 64, 2)
fig, axs = plt.subplots(2, 3, figsize=(19, 12.5), constrained_layout=True)
for row, t in enumerate((0.5, 1.0)):
    k = [round(float(s["t"]), 2) for s in sn].index(t)
    z = sn[k]; d = derived(z)
    pairs = [(d["Jz"], "J_z", dict(sym=True)), (d["sigma"], "log10 σ", dict(cmap="plasma", log=True)),
             (z["rho"], "log10 ρ", dict(cmap="Greys", log=True))]
    for col, (f, name, kw) in enumerate(pairs):
        ax = axs[row, col]
        show(ax, z, f, "canonical exact 64², t = %.2f: %s\nfaces left to HLLD (o x-face, ^ y-face; size = sweeps)" % (t, name), **kw)
        for dd, mk in ((0, "o"), (1, "^")):
            fail = cov[dd]["failed"][k]
            ii, jj = np.nonzero(fail)
            if ii.size:
                fx, fy = PRO.face_xy(z["x"], z["y"], dd, ii, jj)
                ax.scatter(fx, fy, s=4 + 40 * fail[ii, jj] / max(fail.max(), 1), marker=mk,
                           facecolors="none", edgecolors="#000000" if col != 2 else "#d62728", linewidths=0.8)
fig.suptitle("Where the exact solver loses faces (canonical run, every sweep in the snapshot's window)", fontsize=14)
fig.savefig(os.path.join(OUT, "ot2d_failures.png"), dpi=100); plt.close(fig)
print("written to", OUT)
