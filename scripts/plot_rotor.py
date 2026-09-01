"""Plot magnetic-rotor snapshots (density, pressure, |v|, magnetic pressure)."""
import argparse, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def panel(ax, x, y, z, title, cmap="viridis", log=False):
    if log:
        z = np.log10(np.maximum(z, 1e-30))
        title = f"log10({title})"
    im = ax.pcolormesh(x, y, z.T, cmap=cmap, shading="auto")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--snap", default="fin")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    f = os.path.join(a.rundir, f"snap_{a.snap}.npz")
    d = np.load(f)
    x, y, t = d["x"], d["y"], float(d["t"])
    rho, p = d["rho"], d["p"]
    vmag = np.sqrt(d["vx"] ** 2 + d["vy"] ** 2)
    pmag = 0.5 * (d["Bx"] ** 2 + d["By"] ** 2)

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    panel(axs[0, 0], x, y, rho, "rho", log=True)
    panel(axs[0, 1], x, y, p, "p", log=True)
    panel(axs[0, 2], x, y, vmag, "|v|", cmap="magma")
    panel(axs[1, 0], x, y, pmag, "magnetic pressure b^2/2", cmap="plasma")
    panel(axs[1, 1], x, y, d["By"], "By", cmap="RdBu_r")
    dv = np.abs(d["divB"])
    panel(axs[1, 2], x, y, np.maximum(dv, 1e-20), "|div B|", cmap="cividis", log=True)

    fig.suptitle(f"Magnetized rotor, t = {t:.3f}   ({rho.shape[0]}^2, "
                 f"constrained transport)", fontsize=13)
    fig.tight_layout()
    out = a.out or os.path.join(a.rundir, f"rotor_{a.snap}.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)

    # diagnostics over time, if present
    csv = os.path.join(a.rundir, "diag.csv")
    if os.path.exists(csv):
        import csv as _csv
        rows = list(_csv.DictReader(open(csv)))
        if rows:
            tt = [float(r["t"]) for r in rows]
            fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4))
            ax2[0].semilogy(tt, [float(r["divB_max"]) for r in rows], label="max")
            ax2[0].semilogy(tt, [float(r["divB_l2"]) for r in rows], label="L2")
            ax2[0].set_title("div B"); ax2[0].legend(); ax2[0].set_xlabel("t")
            ax2[1].semilogy(tt, [float(r["sym_err"]) for r in rows])
            ax2[1].set_title("pi-rotation symmetry error"); ax2[1].set_xlabel("t")
            ax2[2].plot(tt, [float(r["rho_max"]) for r in rows], label="rho_max")
            ax2[2].plot(tt, [float(r["W_max"]) for r in rows], label="W_max")
            ax2[2].set_title("extrema"); ax2[2].legend(); ax2[2].set_xlabel("t")
            for a_ in ax2: a_.grid(alpha=0.3)
            fig2.tight_layout()
            out2 = os.path.join(a.rundir, "rotor_diagnostics.png")
            fig2.savefig(out2, dpi=120)
            print("wrote", out2)


if __name__ == "__main__":
    main()
