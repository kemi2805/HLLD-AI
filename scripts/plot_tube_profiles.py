"""Look at a stubborn interface's shock tube: one page per lane.

``tube_features.py --pick <gidx,...> --dump profiles.npz`` runs the tubes on
calea and saves the fine profiles; this draws them, which is the part that
wants a laptop (the cluster venv has no matplotlib, and plotting is cheap).

Each page shows the profile against the similarity variable xi = x / t --
rho, total pressure, the tangential field's magnitude and angle, and the
three velocities -- with the detected features shaded and the seven
characteristic speeds of the LEFT and RIGHT states marked.  Where the exact
answer is known, its wave speeds are drawn too, so a feature that belongs to
no wave is visible rather than inferred.

    python scripts/plot_tube_profiles.py profiles.npz --out figs/tubes

What to look for: two distinct features inside one family's speed interval
(a compound wave, which the seven-wave formulation cannot represent), or a
feature sitting where no family's speeds reach.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

FAMILIES = ("F-", "A-", "S-", "CD", "S+", "A+", "F+")
GROUPS = ("control", "stub-nostart", "stub-lost", "stub-exact0")


def page(z, j, ax):
    xi = z["xi"]
    Bt = np.hypot(z["By"][:, j], z["Bz"][:, j])
    psi = np.arctan2(z["Bz"][:, j], z["By"][:, j])
    rows = (("rho", z["rho"][:, j]), ("P_tot", z["Ptot"][:, j]),
            ("|Bt|", Bt), ("psi", psi),
            ("vx", z["vx"][:, j]), ("vy", z["vy"][:, j]), ("vz", z["vz"][:, j]))
    for a, (name, q) in zip(ax, rows):
        a.plot(xi, q, lw=0.9)
        a.set_ylabel(name, fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(alpha=0.25, lw=0.4)
        for m in range(z["fspeed"].shape[1]):
            s, w, fam = z["fspeed"][j, m], z["fwidth"][j, m], z["ffam"][j, m]
            if not np.isfinite(s):
                continue
            a.axvspan(s - 0.5 * w, s + 0.5 * w, color="0.85", zorder=0)
            if a is ax[0]:
                a.text(s, a.get_ylim()[1], FAMILIES[fam] if fam >= 0 else "?",
                       fontsize=6, ha="center", va="bottom")
        if bool(z["known_ok"][j]):
            for k in range(7):
                a.axvline(z["known_spd"][j, k], color="C3", lw=0.6, ls=":")
    ax[-1].set_xlabel("xi = x / t", fontsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump")
    ap.add_argument("--out", default="figs/tubes")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(a.dump)
    os.makedirs(a.out, exist_ok=True)
    n = z["Bn"].size
    for j in range(n):
        fig, ax = plt.subplots(7, 1, figsize=(7.0, 9.0), sharex=True)
        page(z, j, ax)
        g = GROUPS[int(z["group"][j])]
        nf = int(np.isfinite(z["fspeed"][j]).sum())
        ax[0].set_title(
            "lane %d  (%s)   Bn = %+.3f   features %d   tube read residual %.1e"
            "   Newton from it: %s"
            % (int(z["gidx"][j]), g, float(z["Bn"][j]), nf,
               float(z["tube_read_nrm"][j]),
               "yes" if bool(z["tube_ok"][j]) else "no"), fontsize=9)
        fig.tight_layout()
        f = os.path.join(a.out, "tube_%s_%05d.png" % (g, int(z["gidx"][j])))
        fig.savefig(f, dpi=130)
        plt.close(fig)
        print("wrote", f)


if __name__ == "__main__":
    main()
