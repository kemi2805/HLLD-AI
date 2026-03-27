"""
Solver comparison — 5 standalone figures (one per variable).
Solvers: HLLD, HLLD-AI, HLLE + exact solution. No HLLC.

Usage
-----
    python plot_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── file paths ────────────────────────────────────────────────────────────────
FILES = {
    "HLLD":    Path("/Users/miler/Codes/HLLD/results/output_st1_hlld/snap_000200.npz"),
    "HLLD-AI": Path("/Users/miler/Codes/HLLD/ken_output_new/snap_000200.npz"),
    "HLLE":    Path("/Users/miler/Codes/HLLD/results/output_st1_hlle/snap_000200.npz"),
}

EXACT_PATH = Path("/Users/miler/Promotion/Papers/HLLD/exact_solutions/Balsara1/solution.dat")

# ── Balsara 1 constant normal field ──────────────────────────────────────────
BX = 0.5

# ── style ─────────────────────────────────────────────────────────────────────
STYLE = {
    "HLLD":    dict(color="#185FA5", ls="-",  lw=1.8, zorder=4),
    "HLLD-AI": dict(color="#A32D2D", ls="--", lw=1.6, zorder=3),
    "HLLE":    dict(color="#3B6D11", ls="-.", lw=1.6, zorder=2),
}

# ── load exact solution ───────────────────────────────────────────────────────
# solution.dat columns (0-indexed): x  rho  p_tot  vx  vy  vz  By  Bz
# x runs -0.5 → 0.5; shift +0.5 to match simulation domain [0, 1]
exact   = np.loadtxt(EXACT_PATH)
x_exact = exact[:, 0] + 0.5

vx    = exact[:, 3]
vy    = exact[:, 4]
vz    = exact[:, 5]
By    = exact[:, 6]
Bz    = exact[:, 7]

v2    = vx**2 + vy**2 + vz**2
gamma = 1.0 / np.sqrt(1.0 - v2)
vdotB = vx*BX + vy*By + vz*Bz
B2    = BX**2 + By**2 + Bz**2
b2    = B2 / gamma**2 + vdotB**2
p_gas = exact[:, 2] - 0.5 * b2

exact_data = {
    "rho": exact[:, 1],
    "vx":  vx,
    "vy":  vy,
    "p":   p_gas,
    "By":  By,
}

# ── panels: (npz key, exact key, y-axis label, filename) ─────────────────────
PANELS = [
    ("prim_rho", "rho", r"$\rho$",  "Density",                "plot_rho.png"),
    ("prim_vx",  "vx",  r"$v_x$",   "Normal velocity",        "plot_vx.png"),
    ("prim_vy",  "vy",  r"$v_y$",   "Transverse velocity",    "plot_vy.png"),
    ("prim_p",   "p",   r"$p$",     "Pressure",               "plot_p.png"),
    ("By",       "By",  r"$B_y$",   "Transverse mag. field",  "plot_By.png"),
]

# ── load numerical data ───────────────────────────────────────────────────────
datasets = {label: np.load(path) for label, path in FILES.items()}
t = float(next(iter(datasets.values()))["t"])

# ── produce one figure per variable ──────────────────────────────────────────
for key, ekey, ylabel, title, fname in PANELS:

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # exact solution
    ax.plot(x_exact, exact_data[ekey],
            color="black", lw=1.4, ls="-", zorder=5, label="Exact")

    # numerical solvers
    for label, data in datasets.items():
        ax.plot(data["x"], data[key], label=label, **STYLE[label])

    ax.set_title(rf"{title}  ($t = {t:.4f}$)", fontsize=12)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(True, ls="--", alpha=0.35)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10, framealpha=0.9)

    fig.tight_layout()
    out = Path(fname)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)

