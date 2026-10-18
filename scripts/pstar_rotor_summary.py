"""Tables for calea_pstar_rotor.sh's runs: wall per step, face shares per
sweep axis, symmetry error, L1 against the reference and between runs.

    python scripts/pstar_rotor_summary.py RUNS_DIR [--l1 RUNS_DIR/l1.csv]
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

LABEL = {"hlld": "hlld", "ai": "hlld_ai one-shot", "ws": "hlld_ai warm start"}


def runs(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*_*"))):
        b = os.path.basename(p)
        m, _, n = b.rpartition("_")
        if m in LABEL and n.isdigit() and os.path.isdir(p):
            out.append((m, int(n), p))
    return sorted(out, key=lambda r: (r[1], list(LABEL).index(r[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--l1", default=None)
    a = ap.parse_args()
    R = runs(a.runs)

    print("| run | n | steps | wall s | s/step | t_end | finished | sym final | sym max | |divB| max | c2p_bad max |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for m, n, p in R:
        meta = json.load(open(os.path.join(p, "run_meta.json")))
        rows = list(csv.DictReader(open(os.path.join(p, "diag.csv"))))
        sym = [float(r["sym_err"]) for r in rows if r["sym_err"] not in ("", "nan")]
        div = [float(r["divB_max"]) for r in rows]
        c2p = [int(r["c2p_bad"]) for r in rows]
        fin = os.path.exists(os.path.join(p, "snap_fin.npz"))
        print(f"| {LABEL[m]} | {n} | {meta.get('steps')} | {meta.get('wall_s')} | "
              f"{meta.get('s_per_step')} | {float(rows[-1]['t']):.4f} | {fin} | "
              f"{sym[-1]:.3e} | {max(sym):.3e} | {max(div):.2e} | {max(c2p)} |")

    print("\nfaces per sweep axis (whole run): share answered by the solver itself, share handed to HLLE by reason")
    print("| run | n | axis | faces | own flux | HLLE: B_n=0 | HLLE: wave order | HLLE: other |")
    print("|---|---|---|---|---|---|---|---|")
    for m, n, p in R:
        f = os.path.join(p, "face_stats.json")
        if not os.path.exists(f):
            continue
        tot = json.load(open(f))["totals"]
        for ax in sorted(tot):
            t = tot[ax]
            print(f"| {LABEL[m]} | {n} | {'x' if ax.endswith('0') else 'y'} | {t['faces']} | "
                  f"{100*t['frac_used']:.3f}% | {100*t['frac_hlle_bn_zero']:.3f}% | "
                  f"{100*t['frac_hlle_wave']:.4f}% | {100*t['frac_hlle_other']:.4f}% |")

    # per-sweep share over time: first / median / last sweep of each axis
    print("\nper sweep, one-shot share of the y-sweeps over time (face_stats.csv)")
    for m, n, p in R:
        f = os.path.join(p, "face_stats.csv")
        if not os.path.exists(f) or m == "hlld":
            continue
        z = np.genfromtxt(f, delimiter=",", names=True)
        for ax in (0, 1):
            s = z[z["axis"] == ax]
            frac = s["used"] / np.maximum(s["faces"], 1)
            q = np.quantile(frac, [0.0, 0.5, 1.0])
            print(f"  {LABEL[m]} {n}^2 axis {'xy'[ax]}: {s.size} sweeps; own-flux share "
                  f"first {frac[0]:.4f}, min {q[0]:.4f}, median {q[1]:.4f}, last {frac[-1]:.4f}")

    if a.l1 and os.path.exists(a.l1):
        print("\nL1 at the final time (rotor_compare.py: finer grid cell-averaged onto the coarser; mean|A-B| / mean|B|)")
        print("| A | B | t | L1 rho | L1 p | L1 |B| |")
        print("|---|---|---|---|---|---|")
        rows = list(csv.DictReader(open(a.l1)))
        last = {}
        for r in rows:
            if r["region"] != "global":
                continue
            key = (r["A"], r["B"])
            t = float(r["t"])
            if key not in last or t > last[key][0]:
                last[key] = (t, {})
            if t == last[key][0]:
                last[key][1][r["field"]] = float(r["l1"])
        for (A, B), (t, v) in last.items():
            print(f"| {os.path.basename(A)} | {os.path.basename(B)} | {t:.4f} | "
                  f"{v.get('rho', float('nan')):.4e} | {v.get('p', float('nan')):.4e} | "
                  f"{v.get('B', float('nan')):.4e} |")


if __name__ == "__main__":
    main()
