"""Where the exact flux changes the error, against HLLD on the same grid
(2026-11-04, the four-quadrant 128^2 run with ft3; for Luciano's standing
request: WHERE, not only how much).

    python err_maps_flux.py OUT.png REF.npz HLLD.npz EXACT.npz [--lost RUNDIR]

Panels at t = 0.4: |rho - ref| for HLLD and for the exact flux (the 512^2
reference block-averaged onto the grid, r2d_compare64.py's measure), and
|rho_exact - ref| - |rho_HLLD - ref| (blue: the exact flux is closer).  With
--lost (a fetch_runs64.sh run directory), the exact run's lost faces over the
whole run are drawn on the difference panel as black dots.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from r2d_compare64 import contours, extent, l1, load_run


def final(path):
    z = np.load(path, allow_pickle=True)
    assert abs(float(z["t"]) - 0.4) < 1e-9, path
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("ref")
    ap.add_argument("hlld")
    ap.add_argument("exact")
    ap.add_argument("--lost", default=None)
    a = ap.parse_args()
    ref = final(a.ref)["rho"]
    h, e = final(a.hlld), final(a.exact)
    n = h["rho"].shape[0]
    Lh, r = l1(h["rho"], ref)
    Le, _ = l1(e["rho"], ref)
    eh, ee = np.abs(h["rho"] - r), np.abs(e["rho"] - r)
    D = ee - eh
    ext = extent(e)
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    em = max(eh.max(), ee.max())
    for j, (err, name, L) in enumerate(((eh, "HLLD", Lh), (ee, "exact flux", Le))):
        im = ax[j].imshow(err.T, origin="lower", extent=ext, cmap="Purples", vmin=0, vmax=em,
                          interpolation="nearest")
        contours(ax[j], h if j == 0 else e)
        ax[j].set_title("|ρ − ref|, %s, %d²\nL1 = %.4f" % (name, n, L), fontsize=10)
    fig.colorbar(im, ax=ax[:2].tolist(), shrink=0.8, pad=0.01).set_label("|ρ − ρ(512² HLLD, %d×%d mean)|"
                                                                        % (ref.shape[0] // n, ref.shape[0] // n))
    # the colour scale saturates at the 99th percentile of the changes, so the
    # pattern stays visible next to a few large cells
    dm = float(np.percentile(np.abs(D[D != 0]), 99)) if (D != 0).any() else 1.0
    imd = ax[2].imshow(D.T, origin="lower", extent=ext, cmap="RdBu_r", norm=TwoSlopeNorm(0, -dm, dm),
                       interpolation="nearest")
    contours(ax[2], e)
    better = (D < 0).sum()
    title = "|err exact| − |err HLLD|: exact closer in %d of %d cells\nnet change %+.1f%% of HLLD's L1" % (
        better, D.size, 100 * D.sum() / eh.sum())
    if a.lost:
        run = load_run(a.lost)
        L = run["lost"].sum(axis=0)                 # staggered (2n+1)^2, all windows
        fi, fj = np.nonzero(L)
        dx = (ext[1] - ext[0]) / n
        ax[2].scatter(ext[0] + fi * dx / 2, ext[2] + fj * dx / 2, s=0.6, c="k", alpha=0.5, lw=0)
        title += "; dots: %d faces lost at least once" % len(fi)
    ax[2].set_title(title, fontsize=9.5)
    fig.colorbar(imd, ax=ax[2], shrink=0.8, pad=0.01, extend="both").set_label(
        "exact worse (red) / closer (blue)")
    for x in ax:
        x.set_xticks([])
        x.set_yticks([])
    fig.suptitle("Four-quadrant problem, %d², t = 0.4: where the exact flux changes the density error" % n,
                 fontsize=11.5)
    fig.savefig(a.out, dpi=150)
    print("wrote %s: L1 HLLD %.4f, exact %.4f; exact closer in %d/%d cells" % (a.out, Lh, Le, better, D.size))


if __name__ == "__main__":
    main()
