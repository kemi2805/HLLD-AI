#!/usr/bin/env python
"""Two interfaces the exact solver cannot solve (failure ledger, every rung on), run as 1D Riemann problems
with HLLE, HLLC and HLLD: the profiles at one time, side by side.

    python scripts/plot_two_faces.py results/weak_gate_2026-10/two_faces.npz \\
        --out docs/figs/two_faces

The npz is written by the calea job that evolves the two tubes (800 cells,
MC limiter, t = 0.3 on x in [-0.5, 0.5]); see docs/what_we_solve.md 5a.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARS = (("rho", r"density $\rho$"), ("p", r"gas pressure $p$"),
        ("vx", r"normal velocity $v_n$"), ("vt", r"tangential velocity $v_t$"),
        ("Bt", r"tangential field $B_t$ (signed)"))
# categorical, fixed order: HLLE, HLLC, HLLD
COLORS = {"hlle": "#7c3aed", "hllc": "#d97706", "hlld": "#1d4ed8"}
NAMES = {"hlle": "HLLE", "hllc": "HLLC", "hlld": "HLLD"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("npz")
    ap.add_argument("--out", default="docs/figs/two_faces")
    a = ap.parse_args()
    z = np.load(a.npz)
    for name, title in (("compound", "Compound interface (the tangential field reverses)"),
                        ("elementary", "Elementary interface (no field turn)")):
        t = float(z["%s_hlld_t" % name])
        L, R, Bn = z[name + "_L"], z[name + "_R"], float(z[name + "_Bn"])
        # the plane of the flow: the tangential field of the stronger side
        # points along +t, so a reversal shows as a sign change
        S = L if np.hypot(L[5], L[6]) >= np.hypot(R[5], R[6]) else R
        ang = np.arctan2(S[6], S[5])
        ca, sa = np.cos(ang), np.sin(ang)
        fig, axes = plt.subplots(2, len(VARS), figsize=(17, 6.6), constrained_layout=True)
        for row, (suf, nc) in enumerate((("", 800), ("_c100", 100))):
            if "%s_hlld%s_rho" % (name, suf) not in z.files:
                continue
            n = z["%s_hlld%s_rho" % (name, suf)].size
            x = (np.arange(n) + 0.5) / n - 0.5
            for ax, (k, lab) in zip(axes[row], VARS):
                for flux in ("hlle", "hllc", "hlld"):
                    g = lambda q: z["%s_%s%s_%s" % (name, flux, suf, q)]
                    y = (ca * g("vy") + sa * g("vz") if k == "vt" else
                         ca * g("By") + sa * g("Bz") if k == "Bt" else g(k))
                    ax.plot(x / t, y, color=COLORS[flux],
                            lw=1.8 if flux == "hlld" else 1.2, label=NAMES[flux],
                            ls="-" if flux != "hllc" else "--")
                ax.set_xlim(-1.0, 1.0)
                ax.set_xlabel(r"$x / t$")
                ax.set_title("%s, %d cells" % (lab, nc), fontsize=10)
                ax.grid(alpha=0.25)
        axes[0, 0].legend(frameon=False)
        fig.suptitle("%s: interface %d of the failure ledger, MC limiter, t = %.2f.  "
                     r"$B_n$ = %.3f;  left: $\rho$ %.3f, $P_{\rm tot}$ %.3f, $v_n$ %.3f;  "
                     r"right: $\rho$ %.3f, $P_{\rm tot}$ %.3f, $v_n$ %.3f"
                     % (title, int(z[name + "_lane"]), t, Bn, L[0], L[1], L[2],
                        R[0], R[1], R[2]), fontsize=10)
        for ext in ("png", "pdf"):
            fig.savefig("%s_%s.%s" % (a.out, name, ext), dpi=150)
        plt.close(fig)
        print("wrote %s_%s.png" % (a.out, name))


if __name__ == "__main__":
    main()
