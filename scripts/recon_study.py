"""Does higher-order reconstruction close the rotor's resolution gap?

The measurement plan E was written for.  `docs/what_we_solve.md` section 7
records that switching on the exact seven-wave flux moves L1(rho) from 32.84%
to 32.75% -- the flux changes the solution by ~0.5% on the interfaces it owns,
against a 32.8% resolution gap.  The reconstruction was never varied, so this
varies it: PCM (first-order Godunov, no reconstruction), PLM, MP5, MP7 and
WENO5-Z, everything else fixed, against a converged reference.

Read this output in the order it prints, because the first section can
invalidate the rest:

1. **Does the metric discriminate?**  L1(rho) against the reference must
   separate ONE scheme's own 64, 128 and 256 runs.  If those three are nearly
   equal the metric is saturated at these resolutions and no statement about
   reconstruction order can be read from it -- that is a result about the
   measurement, not about the schemes, and it is printed first so it cannot
   be skipped past afterwards.
2. **Error against the reference**, global and per annulus.  The outer
   annulus carries the outflow boundary, where the schemes' different ghost
   widths sit, so it is reported apart.
3. **Observed order**, two independent ways: self-convergence within a scheme
   (L1 between its own N and 2N) and the slope of the error against the
   reference.  If they disagree the reference is not converged, and that is
   the finding.
4. **What the sharpness costs**: wall time per step, the pressure and density
   minima, how often the conservative-to-primitive inversion failed, and the
   violation of the rotor's pi-rotation symmetry -- which is exact in the
   continuum, so all of it is numerical.

    python scripts/recon_study.py --ref results/rotor_512_hlld_mp5 \\
        --runs results/rotor_{64,128,256}_hlld_{pcm,mc,mp5,mp7}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rotor_compare as RC                                      # noqa: E402

FIELDS = ("rho", "p", "B")


def snap_at(d, t=0.4, tol=3e-3):
    S = RC.load_snaps(d)
    k = min(S, key=lambda u: abs(u - t))
    if abs(k - t) > tol:
        raise SystemExit("%s has no snapshot near t = %.3f (nearest %.4f)"
                         % (d, t, k))
    return S[k]


def meta(d):
    f = os.path.join(d, "run_meta.json")
    m = json.load(open(f)) if os.path.exists(f) else {}
    dg = RC.read_diag(d) or []
    get = lambda k: [float(r[k]) for r in dg if r.get(k) not in (None, "", "nan")]
    sym, pmin = get("sym_err"), get("p_min")
    rmin, bad = get("rho_min"), get("c2p_bad")
    m.update(sym_final=sym[-1] if sym else np.nan,
             sym_max=max(sym) if sym else np.nan,
             p_min=min(pmin) if pmin else np.nan,
             rho_min=min(rmin) if rmin else np.nan,
             c2p_bad=int(sum(bad)) if bad else -1)
    return m


def errors(run, ref, regions=True):
    """L1 of a run against the reference, on the run's own grid."""
    a, b = snap_at(run), snap_at(ref)
    out = {}
    for k in FIELDS:
        fa = RC.magB(a) if k == "B" else a[k]
        fb = RC.magB(b) if k == "B" else b[k]
        ca, cb = RC.to_common(fa, fb)
        out[k] = RC.l1_rel(ca, cb)
    ca, cb = RC.to_common(a["rho"], b["rho"])
    n = ca.shape[0]
    xs = np.linspace(-0.5, 0.5, n, endpoint=False) + 0.5 / n
    if regions:
        for name, m in RC.region_masks(xs, xs).items():
            out[name] = float(np.abs(ca - cb)[m].mean()
                              / max(np.abs(cb[m]).mean(), 1e-300))
    out["front"] = RC.front_radius(a)
    out["front_ref"] = RC.front_radius(b)
    # the pi rotation is an exact symmetry of the rotor, so this is numerical
    r = a["rho"]
    d = np.abs(r - r[::-1, ::-1]) / np.maximum(r, 1e-30)
    out["asym_max"] = float(d.max())
    out["asym_cells"] = float((d > 0.05).mean())
    return out


# fixed categorical order, validated (dataviz validate_palette.js, light
# surface: CVD and normal-vision separation pass on adjacent pairs); marker
# shape is the secondary encoding, and every line is direct-labelled
SHORT = {"mc": "PLM", "mp5": "MP5", "weno5z": "WENO5-Z", "mp7": "MP7",
         "pcm": "PCM"}
STYLE = {"mc": ("PLM (2nd)", "#1d4ed8", "o"),
         "mp5": ("MP5 (5th)", "#d97706", "s"),
         "weno5z": ("WENO5-Z (5th)", "#0891b2", "D"),
         "mp7": ("MP7 (7th)", "#b91c1c", "^"),
         "pcm": ("PCM, no reconstruction (1st)", "#7c3aed", "v")}


def plot(runs, E, out, failed, ref_meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = (("rho", "L1(density) against the reference", False),
              ("p", "L1(gas pressure) against the reference", False),
              ("asym_max", "worst violation of the pi-rotation symmetry\n"
                           "(exact in the continuum: all of it is numerical)",
               True))
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.0))
    ns_all = sorted({n for _, n, _, _ in runs})
    for a_, (key, title, logy) in zip(ax, panels):
        ends = []                            # (x, y, text, colour) to label
        for lim in ("mc", "mp5", "weno5z", "mp7", "pcm"):
            pts = sorted((n, E[(lim, n)][key]) for _, n, l, _ in runs
                         if l == lim)
            if not pts:
                continue
            lab, col, mk = STYLE[lim]
            xs = [p[0] for p in pts]
            ys = [p[1] * (1 if logy else 100) for p in pts]
            a_.plot(xs, ys, color=col, marker=mk, lw=2, ms=8, label=lab,
                    markeredgecolor="white", markeredgewidth=1.0)
            ends.append((xs[-1], ys[-1], SHORT[lim]))
        # direct labels, pushed apart where lines end close together (the
        # three high-order-or-PLM lines finish within a point of each other
        # at 256^2): spacing in log space for the log panel, in % otherwise
        for x0 in sorted({e[0] for e in ends}):
            grp = sorted([e for e in ends if e[0] == x0], key=lambda e: e[1])
            tr = (np.log10 if logy else (lambda v: v))
            inv = ((lambda v: 10 ** v) if logy else (lambda v: v))
            gap = 0.16 if logy else 1.6
            placed = []
            for _, y0, txt in grp:
                v = tr(y0)
                if placed and v - placed[-1] < gap:
                    v = placed[-1] + gap
                placed.append(v)
                a_.annotate(txt, (x0, y0), xytext=(x0 * 1.12, inv(v)),
                            textcoords="data", fontsize=9, color="#374151",
                            va="center")
        a_.set_xscale("log", base=2)
        a_.set_xticks(ns_all + [512])
        a_.set_xticklabels(["%d$^2$" % n for n in ns_all + [512]])
        a_.set_xlim(ns_all[0] / 1.3, 512 * 1.3)
        if logy:
            a_.set_yscale("log")
        else:
            a_.set_ylabel("%")
            a_.set_ylim(bottom=0)
        a_.set_title(title, fontsize=10)
        a_.grid(alpha=0.2)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for i, f in enumerate(failed):
            lim, n = f.split(":")
            if lim in STYLE and key == "rho":
                lab, col, mk = STYLE[lim]
                # side by side, or the second hides the first
                a_.scatter([int(n) * (0.93 + 0.14 * i)],
                           [a_.get_ylim()[1] * 0.92], marker="x", s=70,
                           color=col, linewidths=2.2, zorder=5)
        if key == "rho" and failed:
            a_.annotate("x: MP5 and MP7 went\nnon-finite at 512$^2$\n"
                        "(t = 0.113 and 0.270)", (512, a_.get_ylim()[1] * 0.83),
                        fontsize=8, color="#374151", ha="center", va="top")
    ax[0].legend(fontsize=8, frameon=False, loc="lower left")
    # facts only in the title: a conclusion written here goes stale the
    # moment another run lands (this one said "second order is best" until
    # WENO5-Z reached 256^2)
    fig.suptitle("The rotor at t = 0.4, HLLD flux, %d reconstructions, "
                 "against a %s^2 %s reference"
                 % (len({l for _, _, l, _ in runs}), ref_meta.get("n", "?"),
                    STYLE.get(ref_meta.get("limiter"), ("?",))[0]),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    fig.savefig(os.path.splitext(out)[0] + ".pdf")
    print("wrote %s (+ .pdf)" % out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--t", type=float, default=0.4)
    ap.add_argument("--csv")
    ap.add_argument("--plot", help="write the summary figure here (.png; a "
                                   ".pdf is written beside it)")
    ap.add_argument("--failed", nargs="*", default=[],
                    help="scheme:N pairs that did not complete, marked on the "
                         "figure (e.g. mp5:512 mp7:512)")
    a = ap.parse_args()

    runs = []
    for d in a.runs:
        if not glob.glob(os.path.join(d, "snap_*.npz")):
            print("  ! skipping %s: no snapshots" % d)
            continue
        m = meta(d)
        runs.append((d, m.get("n", 0), m.get("limiter", "?"), m))
    runs.sort(key=lambda r: (r[2], r[1]))
    rm = meta(a.ref)
    print("reference: %s  (%s^2, %s)\n" % (a.ref, rm.get("n", "?"),
                                           rm.get("limiter", "?")))

    E = {}
    for d, n, lim, m in runs:
        E[(lim, n)] = errors(d, a.ref)

    # ── 1. does the metric discriminate? ─────────────────────────────────
    print("== 1. can L1(rho) against the reference separate resolutions?")
    for lim in sorted({k[0] for k in E}):
        ns = sorted(n for l, n in E if l == lim)
        if len(ns) < 2:
            continue
        v = [E[(lim, n)]["rho"] for n in ns]
        rates = [np.log2(v[i] / v[i + 1]) for i in range(len(v) - 1)]
        print("   %-7s " % lim + "  ".join("%d^2 %.2f%%" % (n, 100 * x)
                                           for n, x in zip(ns, v))
              + "   observed order " + ", ".join("%.2f" % r for r in rates))
    print("   (an order near zero means the metric is saturated: the error is\n"
          "    dominated by something refinement does not remove, and no\n"
          "    conclusion about reconstruction order can rest on it)")

    # ── 2. the error table ───────────────────────────────────────────────
    print("\n== 2. error against the reference at t = %.2f" % a.t)
    hdr = ("%-7s %5s %9s %9s %9s | %9s %9s %9s | %7s"
           % ("scheme", "N", "L1 rho", "L1 p", "L1 |B|", "core", "shell",
              "outer", "front"))
    print("   " + hdr); print("   " + "-" * (len(hdr)))
    for d, n, lim, m in runs:
        e = E[(lim, n)]
        print("   %-7s %5d %8.2f%% %8.2f%% %8.2f%% | %8.2f%% %8.2f%% %8.2f%% | %7.4f"
              % (lim, n, 100 * e["rho"], 100 * e["p"], 100 * e["B"],
                 100 * e["core r<0.20"], 100 * e["shell 0.20-0.40"],
                 100 * e["outer r>0.40"], e["front"]))
    print("   reference front radius: %.4f" % E[runs[0][2], runs[0][1]]["front_ref"])

    # ── 3. what it costs, and what it breaks ─────────────────────────────
    print("\n== 3. cost and robustness (the symmetry is exact in the continuum)")
    hdr = ("%-7s %5s %7s %9s %10s %10s %9s %9s"
           % ("scheme", "N", "steps", "s/step", "p_min", "rho_min",
              "sym max", "asym>5%"))
    print("   " + hdr); print("   " + "-" * len(hdr))
    for d, n, lim, m in runs:
        e = E[(lim, n)]
        print("   %-7s %5d %7d %9.3f %10.2e %10.3f %9.2e %8.2f%%"
              % (lim, n, m.get("steps", -1), m.get("s_per_step", np.nan),
                 m["p_min"], m["rho_min"], m["sym_max"],
                 100 * e["asym_cells"]))

    # ── 4. the trade ─────────────────────────────────────────────────────
    print("\n== 4. what a scheme at N is worth, in PLM resolutions and in seconds")
    plm = sorted([(n, E[("mc", n)]["rho"], m) for d, n, l, m in runs
                  if l == "mc"])
    for d, n, lim, m in runs:
        if lim == "mc" or not plm:
            continue
        e = E[(lim, n)]["rho"]
        cost = m.get("wall_s", np.nan)
        # the COARSEST PLM run that is already at least as accurate: that is
        # what this scheme has to beat to be worth its cost
        # the COARSEST PLM run that is at least as accurate; a 1% relative
        # margin counts as a tie, because differences below that are inside
        # the run-to-run noise measured across hosts (1.4e-4 absolute at
        # t = 0.4, and the schemes here differ by whole percent)
        match = [p for p in plm if p[1] <= e * 1.01]
        if match:
            p = match[0]
            c2 = p[2].get("wall_s", np.nan)
            rel = "the same cost" if 0.9 < cost / c2 < 1.1 else (
                "%.1fx cheaper" % (cost / c2) if c2 < cost
                else "%.1fx the cost" % (c2 / cost))
            verdict = ("matched by" if abs(p[1] - e) <= 0.01 * e
                       else "beaten by")
            print("   %-7s %4d^2 (%5.0f s, %.2f%%) %s PLM at %d^2 "
                  "(%5.0f s, %.2f%%) -- %s"
                  % (lim, n, cost, 100 * e, verdict, p[0], c2, 100 * p[1], rel))
        else:
            best = min(plm, key=lambda q: q[1])
            print("   %-7s %4d^2 (%5.0f s, %.2f%%) is better than EVERY PLM "
                  "run measured (best PLM %.2f%% at %d^2)"
                  % (lim, n, cost, 100 * e, 100 * best[1], best[0]))

    if a.plot:
        plot(runs, E, a.plot, a.failed, rm)

    if a.csv:
        import csv as _csv
        with open(a.csv, "w") as fh:
            w = _csv.writer(fh)
            keys = ["rho", "p", "B", "core r<0.20", "shell 0.20-0.40",
                    "outer r>0.40", "inside r<0.45", "front", "asym_max",
                    "asym_cells"]
            w.writerow(["scheme", "n", "steps", "wall_s", "s_per_step",
                        "p_min", "rho_min", "c2p_bad", "sym_max"] + keys)
            for d, n, lim, m in runs:
                e = E[(lim, n)]
                w.writerow([lim, n, m.get("steps"), m.get("wall_s"),
                            m.get("s_per_step"), m["p_min"], m["rho_min"],
                            m["c2p_bad"], m["sym_max"]]
                           + ["%.6e" % e[k] for k in keys])
        print("\nwrote %s" % a.csv)


if __name__ == "__main__":
    main()
