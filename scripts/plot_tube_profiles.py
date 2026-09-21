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


def pair(z, ja, jb, titles, out):
    """One control beside one stubborn interface, the same rows on both."""
    import matplotlib.pyplot as plt
    xi = z["xi"]
    m = (xi > -1.05) & (xi < 1.05)
    x = xi[m]
    fig, ax = plt.subplots(5, 2, figsize=(10.5, 9.5), sharex=True)
    for col, j in enumerate((ja, jb)):
        Bt = np.hypot(z["By"][m, j], z["Bz"][m, j])
        psi = np.arctan2(z["Bz"][m, j], z["By"][m, j])
        # the planar picture is one-dimensional in B_t: carry the reversal as a sign
        sgn = np.where(np.abs(np.abs(psi) - np.pi) < 1.0, -1.0, 1.0)
        rows = (("rho", z["rho"][m, j]), ("P_tot", z["Ptot"][m, j]),
                ("B_t (signed)", sgn * Bt), ("v_x", z["vx"][m, j]),
                ("v_y", z["vy"][m, j]))
        for r, (name, q) in enumerate(rows):
            a = ax[r, col]
            a.plot(x, q, lw=1.1, color="#2563eb")
            a.grid(alpha=0.25, lw=0.4)
            a.tick_params(labelsize=8)
            if col == 0:
                a.set_ylabel(name, fontsize=9)
            if name.startswith("B_t"):
                a.axhline(0.0, color="0.45", lw=0.8, ls=":")
            for k in range(z["fspeed"].shape[1]):
                sp, w = z["fspeed"][j, k], z["fwidth"][j, k]
                if np.isfinite(sp):
                    a.axvspan(sp - 0.5 * w, sp + 0.5 * w, color="0.85", zorder=0)
        # mark where the field reverses: the sign change is the whole point, and
        # on a scale set by the far side of the tube it is easy to miss
        sb = sgn * Bt
        flip = np.flatnonzero(np.diff(np.sign(sb)) != 0)
        span = sb.max() - sb.min()
        for n_f, f in enumerate(flip[:2]):
            for r in range(5):
                ax[r, col].axvline(x[f], color="#b91c1c", lw=1.0, ls="--")
            # alternate the label to the left-below / right-above of the crossing
            # so two reversals close together do not print on top of each other
            dx, dy = ((-0.45, -0.30) if n_f % 2 == 0 else (0.08, 0.30))
            ax[2, col].annotate("B_t reverses", xy=(x[f], 0.0),
                                xytext=(x[f] + dx, dy * span),
                                fontsize=9, color="#b91c1c",
                                arrowprops=dict(arrowstyle="->", color="#b91c1c",
                                                lw=0.9))
        ax[0, col].set_title(titles[col], fontsize=10)
        ax[-1, col].set_xlabel("xi = x / t", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump")
    ap.add_argument("--out", default="figs/tubes")
    ap.add_argument("--pair", help="two gidx values, control first: a single "
                                   "side-by-side figure instead of one page each")
    ap.add_argument("--fig", help="with --pair: write exactly this file (the "
                                  "extension picks the format, e.g. docs/figs/x.pdf)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(a.dump)
    os.makedirs(a.out, exist_ok=True)
    if a.pair:
        ga, gb = (int(v) for v in a.pair.split(","))
        ja = int(np.flatnonzero(z["gidx"] == ga)[0])
        jb = int(np.flatnonzero(z["gidx"] == gb)[0])
        lab = lambda j, g: "%s — lane %d, Bn = %+.2f" % (g, int(z["gidx"][j]),
                                                         float(z["Bn"][j]))
        pair(z, ja, jb, (lab(ja, "solved exactly: five planar waves"),
                         lab(jb, "stubborn: B_t reverses where rho and P jump")),
             a.fig or os.path.join(a.out, "pair_%05d_%05d.png" % (ga, gb)))
        return
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
