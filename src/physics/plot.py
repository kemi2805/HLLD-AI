"""
Quick-look plotter for 1D SR-MHD shocktube snapshots.

Usage
-----
    python -m srmhd_1d.plot output/snap_000050.npz
    python -m srmhd_1d.plot output/          # plots last snapshot found
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


_PANELS = [
    ("prim_rho", r"$\rho$"),
    ("prim_vx",  r"$v_x$"),
    ("prim_vy",  r"$v_y$"),
    ("prim_p",   r"$p$"),
    ("prim_eps", r"$\varepsilon$"),
    ("By",       r"$B_y$"),
    ("Bz",       r"$B_z$"),
    ("cons_D",   r"$D$"),
    ("cons_Sx",  r"$S_x$"),
    ("cons_Sy",  r"$S_y$"),
    ("cons_Sz",  r"$S_z$"),
    ("cons_tau", r"$\tau$"),
]


def plot_snapshot(path: Path, save: bool = True):
    data = np.load(path)
    t = float(data["t"])
    x = data["x"]
    print(list(data.keys()))

    keys_present = [p for p in _PANELS if p[0] in data]

    ncols = 4
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3 * nrows),
                             sharex=True)
    axes = axes.flatten()

    for ax, (key, label) in zip(axes, keys_present):
        ax.plot(x, data[key], lw=1.2, color="royalblue")
        ax.set_ylabel(label, fontsize=12)
        ax.set_xlabel(r"$x$", fontsize=10)
        ax.grid(True, ls="--", alpha=0.4)


    # Hide unused panels
    for ax in axes[len(keys_present):]:
        ax.set_visible(False)

    fig.suptitle(f"SR-MHD shocktube  —  $t = {t:.4f}$", fontsize=14)
    fig.tight_layout()

    if save:
        out = path.with_suffix(".png")
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
    else:
        plt.show()

    plt.close(fig)


def plot_all(directory: Path):
    snaps = sorted(directory.glob("snap_*.npz"))
    if not snaps:
        print(f"No snapshots found in {directory}")
        return
    print(f"Found {len(snaps)} snapshots, plotting last one: {snaps[-1].name}")
    plot_snapshot(snaps[-1], save=True)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    if target.is_dir():
        plot_all(target)
    else:
        plot_snapshot(target, save=True)