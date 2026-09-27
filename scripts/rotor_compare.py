"""rotor_compare.py -- compare two rotor runs snapshot by snapshot.

    python scripts/rotor_compare.py results/rotor_64_exact results/rotor_128_hlld \
        [--ref results/rotor_256_hlld] [--out figs/rotor_64_vs_128]

The claim under test (plan 4.3): the exact flux at N matches HLLD at 2N.  So
the comparison must work ACROSS resolutions: the finer grid is coarsened by
cell averaging onto the coarser one, which is the finite-volume-consistent
projection (a coarse cell's value IS the mean of the fine cells it covers).

Per matched snapshot time:
  * L1(rho), L1(p), L1(|B|) between A and B on A's grid (relative to B's L1)
  * fast-front radius from the |B| profile along y = 0: the outermost cell
    where |B| departs from its ambient value by more than a threshold
  * symmetry_error of each run, from diag.csv when present
Plus per-sweep solved/attempted from diag.csv for an exact run.

shocktube_compare.py is 1D-only; this is its 2D counterpart.  Kept to numpy +
matplotlib so it runs on a cluster login node.
"""
import argparse, csv, glob, json, os, sys
import numpy as np


def load_snaps(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "snap_*.npz"))):
        z = np.load(f)
        out[float(z["t"])] = {k: z[k] for k in z.files}
    if not out:
        sys.exit(f"no snap_*.npz in {d}")
    return out


def coarsen(a, factor):
    """Cell-average a (n, n) field onto (n/factor, n/factor)."""
    n = a.shape[0]
    assert n % factor == 0, (a.shape, factor)
    m = n // factor
    return a.reshape(m, factor, m, factor).mean(axis=(1, 3))


def to_common(fa, fb):
    """Bring two fields to the coarser of the two grids."""
    na, nb = fa.shape[0], fb.shape[0]
    if na == nb:
        return fa, fb
    if na > nb:
        assert na % nb == 0; return coarsen(fa, na // nb), fb
    assert nb % na == 0; return fa, coarsen(fb, nb // na)


def magB(s):
    return np.sqrt(s["Bx"] ** 2 + s["By"] ** 2)


def l1_rel(a, b):
    return float(np.abs(a - b).mean() / max(np.abs(b).mean(), 1e-300))


# the annuli the regional norms use: the evacuated core, the dense shell
# where the structure lives, and the outer region -- which is also where the
# outflow boundary and the schemes' different ghost widths sit, so it is
# reported separately rather than folded into one number
REGIONS = (("core r<0.20", 0.0, 0.20), ("shell 0.20-0.40", 0.20, 0.40),
           ("outer r>0.40", 0.40, 9.9), ("inside r<0.45", 0.0, 0.45))


def region_masks(x, y):
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.hypot(X, Y)
    return {name: (r >= lo) & (r < hi) for name, lo, hi in REGIONS}


def front_radius(s, thresh=0.02, interpolate=True):
    """Outermost |x| along y=0 where |B| deviates from the ambient by > thresh.

    Interpolated across the crossing by default: taken at the cell centre it
    is quantised by dx and biased outward on a coarse grid, which is exactly
    the comparison this is used for (0.4766 / 0.4492 / 0.4395 at 64/128/256
    against 0.4882 / 0.4598 / 0.4450 interpolated).
    """
    B = magB(s); x = s["x"]; n = B.shape[0]
    row = B[n // 2]                       # y = 0 row
    amb = row[0]                          # ambient at the boundary
    dev = np.abs(row - amb) - thresh * max(abs(amb), 1e-300)
    idx = np.flatnonzero(dev > 0)
    if not idx.size:
        return 0.0
    if not interpolate:
        return float(np.abs(x[idx]).max())
    out = 0.0
    for i in (idx.min(), idx.max()):      # the two ends of the disturbed zone
        j = i - 1 if i == idx.min() else i + 1
        if 0 <= j < n and dev[j] * dev[i] < 0:
            w = dev[i] / (dev[i] - dev[j])          # linear crossing
            out = max(out, abs(x[i] + w * (x[j] - x[i])))
        else:
            out = max(out, abs(x[i]))
    return float(out)


def run_stamp(d):
    """`limiter/solver` of a run, so a comparison cannot silently mix two
    schemes.  Runs made before the stamp existed say so."""
    f = os.path.join(d, "run_meta.json")
    if os.path.exists(f):
        m = json.load(open(f))
        return "%s/%s" % (m.get("solver", "?"), m.get("limiter", "?"))
    for g in sorted(glob.glob(os.path.join(d, "snap_*.npz"))):
        z = np.load(g)
        if "limiter" in z.files:
            return "%s/%s" % (str(z.get("solver", "?")), str(z["limiter"]))
        break
    return "unstamped"


def read_diag(d):
    f = os.path.join(d, "diag.csv")
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    return rows


def match_times(ta, tb, tol=3e-3):
    """Pair snapshots by time.  run_2d saves a snapshot on the first step PAST
    each target time, so two runs at different resolutions land up to one
    coarse dt apart (dt ~ 1.5e-3 at 64^2).  An absolute tolerance of a few
    coarse dt pairs them; the earlier relative 1e-6 matched only t=0 and the
    exactly-hit final time."""
    pairs = []
    for t in ta:
        near = sorted(tb, key=lambda u: abs(u - t))
        if near and abs(near[0] - t) <= tol:
            pairs.append((t, near[0]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A"); ap.add_argument("B")
    ap.add_argument("--ref", default=None, help="a finer reference run for the front radius")
    ap.add_argument("--out", default=None, help="figure prefix; no figures if omitted")
    ap.add_argument("--regions", action="store_true",
                    help="L1 per annulus as well as globally: the global "
                         "number hides an inversion in the middle annulus, "
                         "and the outer one carries the boundary")
    ap.add_argument("--csv", help="append one row per (A, B, t, field, region)")
    a = ap.parse_args()

    SA, SB = load_snaps(a.A), load_snaps(a.B)
    SR = load_snaps(a.ref) if a.ref else None
    nA = next(iter(SA.values()))["rho"].shape[0]
    nB = next(iter(SB.values()))["rho"].shape[0]
    mA, mB = run_stamp(a.A), run_stamp(a.B)
    print(f"A = {a.A}  ({nA}^2, {mA})\nB = {a.B}  ({nB}^2, {mB})"
          + (f"\nref = {a.ref}" if a.ref else ""))
    if mA != mB:
        print("  NOTE: different schemes -- this is a difference between "
              "SCHEMES, not a resolution study")
    pairs = match_times(sorted(SA), sorted(SB))
    if not pairs:
        sys.exit("no matching snapshot times")

    print(f"\n{'t':>7} {'L1 rho':>9} {'L1 p':>9} {'L1 |B|':>9} {'front A':>8} {'front B':>8}"
          + (f" {'front ref':>10}" if SR else ""))
    rows = []
    for t, u in pairs:
        sa, sb = SA[t], SB[u]
        r = {}
        for k, fa, fb in (("rho", sa["rho"], sb["rho"]), ("p", sa["p"], sb["p"]),
                          ("B", magB(sa), magB(sb))):
            ca, cb = to_common(fa, fb)
            r[k] = l1_rel(ca, cb)
        r["fA"], r["fB"] = front_radius(sa), front_radius(sb)
        if a.regions:
            ca, cb = to_common(sa["rho"], sb["rho"])
            nx = ca.shape[0]
            xs = np.linspace(-0.5, 0.5, nx, endpoint=False) + 0.5 / nx
            for name, m in region_masks(xs, xs).items():
                r["reg_" + name] = float(np.abs(ca - cb)[m].mean()
                                         / max(np.abs(cb[m]).mean(), 1e-300))
        line = f"{t:7.4f} {r['rho']:9.3e} {r['p']:9.3e} {r['B']:9.3e} {r['fA']:8.4f} {r['fB']:8.4f}"
        if SR:
            tr = min(SR, key=lambda v: abs(v - t)); r["fR"] = front_radius(SR[tr])
            line += f" {r['fR']:10.4f}"
        print(line); rows.append((t, r))
        if a.regions:
            print("        regions, L1 rho: " + "  ".join(
                "%s %.3e" % (n.split()[0], r["reg_" + n])
                for n in [x[0] for x in REGIONS]))

    if a.csv:
        new = not os.path.exists(a.csv)
        with open(a.csv, "a") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["A", "B", "scheme_A", "scheme_B", "nA", "nB", "t",
                            "field", "region", "l1", "front_A", "front_B"])
            for t, r in rows:
                for k in ("rho", "p", "B"):
                    w.writerow([a.A, a.B, mA, mB, nA, nB, "%.4f" % t, k,
                                "global", "%.6e" % r[k], "%.5f" % r["fA"],
                                "%.5f" % r["fB"]])
                for n in [x[0] for x in REGIONS]:
                    if "reg_" + n in r:
                        w.writerow([a.A, a.B, mA, mB, nA, nB, "%.4f" % t,
                                    "rho", n, "%.6e" % r["reg_" + n],
                                    "%.5f" % r["fA"], "%.5f" % r["fB"]])
        print("\nappended %d rows to %s" % (len(rows), a.csv))

    for tag, d in (("A", a.A), ("B", a.B)):
        dg = read_diag(d)
        if dg:
            sym = [float(x["sym_err"]) for x in dg if x.get("sym_err") not in (None, "", "nan")]
            div = [float(x["divB_max"]) for x in dg]
            print(f"\n{tag}: {len(dg)} steps, |divB|max over run = {max(div):.2e}"
                  + (f", symmetry_error final = {sym[-1]:.3e} (max {max(sym):.3e})" if sym else ""))

    if a.out:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        t_last, _ = pairs[-1]
        sa, sb = SA[t_last], SB[pairs[-1][1]]
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for i, (k, fa, fb) in enumerate((("rho", sa["rho"], sb["rho"]),
                                          ("p", sa["p"], sb["p"]),
                                          ("|B|", magB(sa), magB(sb)))):
            n = fa.shape[0]; m = fb.shape[0]
            ax[i].plot(sa["x"], fa[n // 2], label=f"A {nA}^2")
            ax[i].plot(sb["x"], fb[m // 2], "--", label=f"B {nB}^2")
            ax[i].set_title(f"{k} along y=0, t={t_last:.3f}"); ax[i].legend()
        fig.tight_layout(); fig.savefig(a.out + "_cuts.png", dpi=130)
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ts = [t for t, _ in rows]
        for k in ("rho", "p", "B"):
            ax.semilogy(ts, [r[k] for _, r in rows], "o-", label=f"L1 {k}")
        ax.set_xlabel("t"); ax.set_ylabel("relative L1 (A vs B)"); ax.legend()
        fig.tight_layout(); fig.savefig(a.out + "_l1.png", dpi=130)
        print(f"\nfigures: {a.out}_cuts.png, {a.out}_l1.png")


if __name__ == "__main__":
    main()
