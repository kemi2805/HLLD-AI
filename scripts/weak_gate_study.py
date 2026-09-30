#!/usr/bin/env python
"""Below the weak-jump gate: what the exact solver does there, and what it
would change.

Production attempts a face only if its relative jump exceeds ``tau_weak``
(1e-2); the rest take HLLD.  The recordings of the failure ledger hold EVERY
face of each sweep, so the gate can be replayed at any threshold:

    python scripts/weak_gate_study.py <rec dir> <tags...> --tau 1e-3 0 \\
        --out <dir>

For each threshold, each recorded sweep is solved again with the gate at that
value and, per bin of the relative jump, the script reports: faces, attempted,
exact, the relative flux difference between the exact flux and HLLD's on the
faces that came out exact, and the wall time.  ``--linear`` adds the
linearised (Roe-type) flux on every face and its difference from the exact
flux -- the measure of whether it can stand in below the gate.  Per-sweep
fluxes are saved, so ``--linear`` can be evaluated later against saved exact
fluxes without solving again (``--from-saved``).

The flux difference is ``||F_a - F_b|| / ||F_HLLD||`` over the eight
conserved components, per face.
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.eos import hybrid_eos                        # noqa: E402
from src.physics.hlld import LAST_DIAG, hlld_flux             # noqa: E402
import src.physics.exact_flux as EF                           # noqa: E402

GAMMA = 5.0 / 3.0
KEYS = ("D", "Sx", "Sy", "Sz", "tau", "Bx", "By", "Bz")
BINS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, np.inf)
BIN_NAMES = ("< 1e-4", "1e-4 .. 1e-3", "1e-3 .. 1e-2", "1e-2 .. 1e-1",
             ">= 1e-1")


def flux_array(F):
    return np.stack([F[k].detach().cpu().numpy().astype(float).reshape(-1)
                     for k in KEYS], axis=0)


def rel_diff(Fa, Fb, Fref):
    num = np.sqrt(((Fa - Fb) ** 2).sum(axis=0))
    den = np.sqrt((Fref ** 2).sum(axis=0))
    return num / np.maximum(den, 1e-300)


def sweeps(rec, tags):
    for tag in tags:
        pat = re.compile(r"^sweep_%s_(\d+)\.pt$" % re.escape(tag))
        ks = sorted(int(pat.match(f).group(1)) for f in os.listdir(rec)
                    if pat.match(f))
        for k in ks:
            yield tag, k, torch.load(os.path.join(rec, "sweep_%s_%d.pt" % (tag, k)))


def q(x, p=(50, 90, 100)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "   -         -         -    "
    return " ".join("%9.2e" % v for v in np.percentile(x, p))


def study(a):
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    os.makedirs(a.out, exist_ok=True)
    lin = None
    if a.linear:
        from src.physics.linearised import linearised_flux
        lin = linearised_flux
    for tau in a.tau:
        rows = []
        wall = 0.0
        for tag, k, r in sweeps(a.rec, a.tags):
            sL, sR, idir = r["sL"], r["sR"], r["idir"]
            jump = EF.relative_jump(sL, sR).detach().cpu().numpy().reshape(-1)
            Fh = flux_array(hlld_flux(sL, sR, eos, idir=idir)[0])
            saved = os.path.join(a.out, "flux_%s_%d_tau%g.npz" % (tag, k, tau))
            if a.from_saved and os.path.exists(saved):
                z = np.load(saved)
                Fe, ex, att, w = z["Fe"], z["exact"], z["attempted"], float(z["wall"])
            else:
                t0 = time.time()
                F, U, ps = EF.exact_flux_batched(sL, sR, eos, idir=idir,
                                                 tau_weak=tau, tau_bt=1e-9,
                                                 n_retries=2, max_iter=40)
                w = time.time() - t0
                d = dict(LAST_DIAG)
                Fe = flux_array(F)
                ex = np.asarray(d["exact_mask"], bool).reshape(-1)
                att = np.zeros(ex.size, bool)
                att[np.asarray(d["sel"], int)] = True
                np.savez_compressed(saved, Fe=Fe, exact=ex, attempted=att,
                                    jump=jump, wall=w)
            wall += w
            Fl, lok = None, None
            if lin is not None:
                Fl_, _, _, lok = lin(sL, sR, eos, idir=idir, return_mask=True)
                Fl = flux_array(Fl_)
                lok = np.asarray(lok, bool).reshape(-1)
            rows.append((jump, att, ex, rel_diff(Fe, Fh, Fh),
                         None if Fl is None else rel_diff(Fl, Fe, Fh),
                         None if Fl is None else rel_diff(Fl, Fh, Fh), lok))
            print("  tau %g %s sweep %2d: attempted %5d exact %5d of %5d faces, %.1fs"
                  % (tau, tag, k, att.sum(), ex.sum(), ex.size, w), flush=True)
        jump = np.concatenate([r[0] for r in rows])
        att = np.concatenate([r[1] for r in rows])
        ex = np.concatenate([r[2] for r in rows])
        d_eh = np.concatenate([r[3] for r in rows])
        have_lin = rows[0][4] is not None
        if have_lin:
            d_le = np.concatenate([r[4] for r in rows])
            d_lh = np.concatenate([r[5] for r in rows])
            lok = np.concatenate([r[6] for r in rows])
        print("\n== tau_weak = %g: %d sweeps, %d faces, attempted %d (%.1f%%), "
              "exact %d (%.2f%% of attempted), wall %.0f s"
              % (tau, len(rows), jump.size, att.sum(), 100.0 * att.mean(),
                 ex.sum(), 100.0 * ex.sum() / max(att.sum(), 1), wall))
        print("   %-14s %7s %7s %7s %8s | %-29s" % (
            "jump", "faces", "attem.", "exact", "solved", "|F_exact - F_HLLD| / |F_HLLD|  p50 p90 max")
              + (" | %-29s | %-29s | %s" % ("|F_lin - F_exact| / |F_HLLD|", "|F_lin - F_HLLD| / |F_HLLD|", "lin ok") if have_lin else ""))
        for lo, hi, name in zip(BINS[:-1], BINS[1:], BIN_NAMES):
            m = (jump >= lo) & (jump < hi)
            e = m & ex
            line = "   %-14s %7d %7d %7d %7.1f%% | %s" % (
                name, m.sum(), (m & att).sum(), e.sum(),
                100.0 * e.sum() / max((m & att).sum(), 1), q(d_eh[e]))
            if have_lin:
                line += " | %s | %s | %5.1f%%" % (q(d_le[e & lok]), q(d_lh[e & lok]),
                                                 100.0 * lok[m].mean())
            print(line)
        np.savez_compressed(os.path.join(a.out, "summary_tau%g.npz" % tau),
                            jump=jump, attempted=att, exact=ex, d_exact_hlld=d_eh,
                            **({"d_lin_exact": d_le, "d_lin_hlld": d_lh, "lin_ok": lok}
                               if have_lin else {}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rec")
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--tau", type=float, nargs="+", default=[1e-3, 0.0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--from-saved", action="store_true",
                    help="reuse the exact fluxes saved by an earlier pass")
    a = ap.parse_args()
    study(a)


if __name__ == "__main__":
    main()
