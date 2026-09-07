"""Summarise an exact-solver harvest directory: coverage, solves, failures.

    python scripts/harvest_summary.py results/rotor_64_exact/harvest [--csv out.csv]

Three questions, one per shard kind:

* ``coverage_*.npz`` -- the COMPLETE map, one row per sweep: which interfaces
  the exact solver attempted (the weak-jump and validity gates let them
  through) and which came out exact.  Neither the stride nor the caps touch
  it, so this is the only trustworthy source for "what fraction of the run
  was actually exact"; ``diag.csv``'s ``hlle_frac_*`` columns are HLLD's own
  HLLE fallback fraction and answer a different question.
* ``solved_*.npz`` -- the seven-wave solutions with zones and speeds, i.e.
  ground truth for the flux-error harness and for training.
* ``unsolved_*.npz`` -- every lane that failed a gate, with its reason
  (not_converged / ray_failed / unphysical), attempt count and class.

Coverage is reported per direction and over the run (in tenths), because the
interesting failure mode is a fraction that DECAYS as the flow steepens --
which a single average would hide.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.physics.harvest import R_REASONS                    # noqa: E402

DIRS = {0: "x", 1: "y"}


def _unpack(bits, n):
    return np.unpackbits(bits, axis=-1)[..., :n].astype(bool)


def load_coverage(d):
    """All coverage rows, concatenated and sorted by (sweep, idir)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "coverage_*.npz"))):
        z = np.load(f)
        gen = int(os.path.basename(f).split("_r")[1][:3])     # restart generation
        for k in range(z["sweep"].shape[0]):
            n = int(z["n"][k])
            rows.append(dict(gen=gen, sweep=int(z["sweep"][k]),
                             idir=int(z["idir"][k]), n=n,
                             att=int(_unpack(z["attempted"][k], n).sum()),
                             ex=int(_unpack(z["exact"][k], n).sum())))
    rows.sort(key=lambda r: (r["gen"], r["sweep"], r["idir"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("harvest")
    ap.add_argument("--csv", default=None, help="write the per-sweep coverage here")
    a = ap.parse_args()

    rows = load_coverage(a.harvest)
    if not rows:
        print("no coverage shards in", a.harvest)
    else:
        tot_n = sum(r["n"] for r in rows)
        tot_a = sum(r["att"] for r in rows)
        tot_e = sum(r["ex"] for r in rows)
        print(f"coverage: {len(rows)} sweeps, {tot_n} interfaces")
        print(f"  attempted {tot_a} ({100 * tot_a / tot_n:.1f}% of interfaces)")
        print(f"  exact     {tot_e} ({100 * tot_e / tot_n:.1f}% of interfaces, "
              f"{100 * tot_e / max(tot_a, 1):.1f}% of attempted)")
        print(f"  the other {100 * (tot_n - tot_e) / tot_n:.1f}% took the fallback flux")

        print(f"\n  {'dir':>4}{'sweeps':>8}{'interf':>10}{'attempt%':>10}{'exact%':>9}"
              f"{'exact/att%':>12}")
        for d in sorted(DIRS):
            rr = [r for r in rows if r["idir"] == d]
            if not rr:
                continue
            n = sum(r["n"] for r in rr)
            at = sum(r["att"] for r in rr)
            ex = sum(r["ex"] for r in rr)
            print(f"  {DIRS[d]:>4}{len(rr):>8}{n:>10}{100 * at / n:>9.1f}%"
                  f"{100 * ex / n:>8.1f}%{100 * ex / max(at, 1):>11.1f}%")

        print(f"\n  over the run (tenths of the sweeps, in order)")
        print(f"  {'part':>6}{'sweeps':>8}{'attempt%':>10}{'exact%':>9}{'exact/att%':>12}")
        chunks = np.array_split(np.arange(len(rows)), min(10, len(rows)))
        for i, ch in enumerate(chunks):
            rr = [rows[j] for j in ch]
            n = sum(r["n"] for r in rr)
            at = sum(r["att"] for r in rr)
            ex = sum(r["ex"] for r in rr)
            print(f"  {i + 1:>6}{len(rr):>8}{100 * at / n:>9.1f}%{100 * ex / n:>8.1f}%"
                  f"{100 * ex / max(at, 1):>11.1f}%")

        if a.csv:
            with open(a.csv, "w") as fh:
                fh.write("gen,sweep,idir,n,attempted,exact\n")
                for r in rows:
                    fh.write(f"{r['gen']},{r['sweep']},{r['idir']},{r['n']},"
                             f"{r['att']},{r['ex']}\n")
            print("\nwrote", a.csv)

    for kind in ("solved", "unsolved"):
        files = sorted(glob.glob(os.path.join(a.harvest, f"{kind}_*.npz")))
        if not files:
            print(f"\nno {kind} shards")
            continue
        n = 0
        att = []
        reason = []
        cls = []
        for f in files:
            z = np.load(f)
            n += z["U_L"].shape[0]
            att.append(z["attempts"])
            if kind == "unsolved":
                reason.append(z["reason"])
                cls.append(z["cls"])
        att = np.concatenate(att)
        print(f"\n{kind}: {n} lanes in {len(files)} shards")
        h = np.bincount(np.clip(att, 0, 8))
        print("  attempts:", "  ".join(f"{k}:{v}" for k, v in enumerate(h) if v))
        if kind == "unsolved":
            reason = np.concatenate(reason)
            cls = np.concatenate(cls)
            for r, name in sorted(R_REASONS.items()):
                c = int((reason == r).sum())
                if c:
                    print(f"  {name:<16}{c:>8} ({100 * c / n:.1f}%)")
            hc = np.bincount(np.clip(cls, 0, 15))
            print("  class:", "  ".join(f"{k}:{v}" for k, v in enumerate(hc) if v))


if __name__ == "__main__":
    main()
