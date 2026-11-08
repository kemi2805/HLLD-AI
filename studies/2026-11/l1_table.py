"""L1 of the four-quadrant runs at t = 0.4 against every 512^2 reference
(2026-11-04: the 128^2 exact run with ft3 against same-code HLLE/HLLC/HLLD
at 128^2, and the 64^2 runs for the trend).

    python l1_table.py OUTDIR --ref hlld=REF.npz [--ref hllc=...] FLUX@N=SNAP.npz [...]

The measure is r2d_compare64.py's (make_2d.py's): the reference block-averaged
onto the run's grid, mean |rho - ref| / mean ref.  FLUX@N names a run, e.g.
exact@128=runs128/ft3_128/snap_008.npz; a missing file is skipped.  Per
reference: L1(rho) and L1(p), the exact flux's change against HLLD on the same
grid, and the HLLD resolution with the same L1 (HLLD's own 64 -> 128 rate,
extrapolated beyond 128).  Figure l1_vs_n.png: L1(rho) against N for the
first reference, the whiskers spanning all references.
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

# the GR06 figures' colours (gr06_plot.py), checked with the dataviz validator
COLOR = {"hlle": "#E69F00", "hllc": "#009E73", "hlld": "#0072B2", "exact": "#D55E00"}
ORDER = ("hlle", "hllc", "hlld", "exact")


def l1(a, ref):
    k = ref.shape[0] // a.shape[0]
    r = ref.reshape(a.shape[0], k, a.shape[1], k).mean(axis=(1, 3))
    return float(np.abs(a - r).mean() / r.mean())


def final(path):
    z = np.load(path, allow_pickle=True)
    t = float(z["t"])
    if abs(t - 0.4) > 1e-9:
        raise SystemExit("%s is at t = %.6f, not 0.4" % (path, t))
    return {k: z[k] for k in ("rho", "p")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+", help="FLUX@N=SNAP.npz")
    ap.add_argument("--ref", action="append", required=True, help="name=SNAP.npz (512^2)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    refs = [(s.split("=", 1)[0], final(s.split("=", 1)[1])) for s in a.ref]
    runs = {}
    for s in a.runs:
        key, path = s.split("=", 1)
        flux, n = key.split("@")
        if not os.path.exists(path):
            print("skip %s: no %s" % (key, path))
            continue
        runs[(flux, int(n))] = final(path)
    res = {}
    for (flux, n), u in runs.items():
        res["%s@%d" % (flux, n)] = {rn: {v: l1(u[v], r[v]) for v in ("rho", "p")} for rn, r in refs}
    grids = sorted({n for _, n in runs})
    fluxes = [f for f in ORDER if any((f, n) in runs for n in grids)] + \
             sorted({f for f, _ in runs} - set(ORDER))
    for v in ("rho", "p"):
        print("\nL1(%s) at t = 0.4 against the 512^2 references" % v)
        print("| run | " + " | ".join("vs %s" % rn for rn, _ in refs) + " |")
        print("|---|" + "---|" * len(refs))
        for n in grids:
            for f in fluxes:
                k = "%s@%d" % (f, n)
                if k in res:
                    print("| %s %d² | " % (f, n) + " | ".join("%.4f" % res[k][rn][v] for rn, _ in refs) + " |")
    # exact against HLLD on the same grid; the equivalent HLLD resolution
    print()
    for rn, _ in refs:
        h = {n: res["hlld@%d" % n][rn]["rho"] for n in grids if "hlld@%d" % n in res}
        rate = math.log(h[64] / h[128]) / math.log(2) if 64 in h and 128 in h else None
        for n in grids:
            k = "exact@%d" % n
            if k not in res or n not in h:
                continue
            e = res[k][rn]["rho"]
            line = "vs %s, %d²: exact %.4f vs HLLD %.4f (%+.1f%%)" % (rn, n, e, h[n], 100 * (e / h[n] - 1))
            if rate:
                neq = n * 2 ** (math.log(h[n] / e) / math.log(2) / rate)
                line += "; HLLD converges at order %.2f (64->128), so exact %d² = HLLD at ~%.0f²" % (rate, n, neq)
            print(line)
            res[k].setdefault("vs_hlld", {})[rn] = dict(change=e / h[n] - 1, hlld_order=rate,
                                                       hlld_equiv_n=(n * 2 ** (math.log(h[n] / e) / math.log(2) / rate)
                                                                     if rate else None))
    json.dump(res, open(os.path.join(a.out, "l1.json"), "w"), indent=1)

    # L1(rho) against N, the first reference; whiskers over all references
    fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)
    r0 = refs[0][0]
    for f in [f for f in fluxes if f in ORDER]:          # extra arms stay in the table only
        ns = [n for n in grids if (f, n) in runs]
        if not ns:
            continue
        c = COLOR.get(f, "#444444")
        y = [res["%s@%d" % (f, n)][r0]["rho"] for n in ns]
        lo = [min(res["%s@%d" % (f, n)][rn]["rho"] for rn, _ in refs) for n in ns]
        hi = [max(res["%s@%d" % (f, n)][rn]["rho"] for rn, _ in refs) for n in ns]
        ax.plot(ns, y, color=c, lw=2, marker="o", ms=7, mec="white", mew=1.5, zorder=3)
        ax.vlines(ns, lo, hi, color=c, lw=1, alpha=0.6, zorder=2)
        ax.annotate("%s %.4f" % ("exact" if f == "exact" else f.upper(), y[-1]), (ns[-1], y[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color="0.15")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(grids)
    ax.set_xticklabels(["%d²" % n for n in grids])
    ax.set_xlim(grids[0] / 1.25, grids[-1] * 1.9)
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("grid")
    ax.set_ylabel("L1(ρ) at t = 0.4 vs 512² %s" % r0.upper())
    ax.set_title("Four-quadrant problem: error against resolution\n(whiskers: range over the %d 512² references)"
                 % len(refs), fontsize=10)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(a.out, "l1_vs_n.png"), dpi=150)
    plt.close(fig)
    print("\nwrote %s, %s" % (os.path.join(a.out, "l1.json"), os.path.join(a.out, "l1_vs_n.png")))


if __name__ == "__main__":
    main()
