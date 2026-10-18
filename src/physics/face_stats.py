"""
What the Riemann solver did with every face, tallied per sweep.

``FaceTally(flux_fn)`` wraps ``hlld_flux`` or ``hlld_ai_flux`` and reads the
per-row masks the solver already computes, through its ``row_diag`` hook:

    used         the solver's own flux (HLLD's star states, or the one-shot
                 network p*'s for hlld_ai)
    hlle_bn_zero handed to HLLE because B_n = 0 on both sides
    hlle_wave    handed to HLLE because the wave ordering failed
                 (and B_n != 0)
    hlle_other   handed to HLLE for anything else: the root-find (HLLD, or
                 hlld_ai's warm start) did not converge

The reasons are exclusive, in that order.  The wrapper only READS: the call
it makes is the solver's own, with ``row_diag`` (which both solvers fill
without touching their outputs), so the fluxes are identical with or
without it -- run_2d.py --face-stats is checked against a plain run to the
bit in tests/test_hlld_ai_2d.py.

Per-call rows go to an optional CSV (step, call, axis, faces, used, the
three HLLE reasons); ``totals()`` sums them per axis.
"""

from __future__ import annotations

import json

FIELDS = ("faces", "used", "hlle_bn_zero", "hlle_wave", "hlle_other")


class FaceTally:
    def __init__(self, flux_fn, csv_path=None, append=False):
        self.fn = flux_fn
        self.step = 0
        self.calls = 0
        self.tot = {}
        self.csv = None
        if csv_path is not None:
            self.csv = open(csv_path, "a" if append else "w")
            if not append:
                self.csv.write("step,call,axis," + ",".join(FIELDS) + "\n")

    def __call__(self, sL, sR, eos, idir=0, **kw):
        rd: dict = {}
        out = self.fn(sL, sR, eos, idir=idir, row_diag=rd, **kw)
        failed = rd["failed"]
        bn0 = rd["bn_zero"]
        wob = rd["wave_order_bad"]
        row = dict(faces=int(failed.numel()),
                   used=int((~failed).sum()),
                   hlle_bn_zero=int((failed & bn0).sum()),
                   hlle_wave=int((failed & ~bn0 & wob).sum()),
                   hlle_other=int((failed & ~bn0 & ~wob).sum()))
        self.calls += 1
        t = self.tot.setdefault(int(idir), {f: 0 for f in FIELDS})
        for f in FIELDS:
            t[f] += row[f]
        if self.csv is not None:
            self.csv.write(f"{self.step},{self.calls},{int(idir)},"
                           + ",".join(str(row[f]) for f in FIELDS) + "\n")
        return out

    def totals(self) -> dict:
        """Per axis: the counts and their shares of the faces."""
        out = {}
        for ax, t in sorted(self.tot.items()):
            n = max(t["faces"], 1)
            out[f"axis{ax}"] = dict(t, **{f"frac_{f}": t[f] / n
                                          for f in FIELDS if f != "faces"})
        return out

    def flush(self):
        if self.csv is not None:
            self.csv.flush()

    def close(self, json_path=None, **meta):
        if self.csv is not None:
            self.csv.close()
            self.csv = None
        tot = self.totals()
        if json_path is not None:
            with open(json_path, "w") as fh:
                json.dump(dict(totals=tot, calls=self.calls, **meta), fh,
                          indent=1)
        return tot
