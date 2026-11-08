#!/usr/bin/env python
"""Run a script with the slow-shock search window opened past the Alfven speed
(slow_shock_b.MARGIN, numpy path only -- RMHD_SLOWSHOCK must be unset): the
intermediate-shock branch, as compound_wave.wide_window does (2026-11-04,
user: "would intermediate shocks solve the problem of the 5 wave solver").

    WIDE_MARGIN=1.0 python run_wide.py SCRIPT.py [args ...]
"""
import os
import runpy
import sys

if os.environ.get("RMHD_SLOWSHOCK", "") == "njit":
    raise SystemExit("RMHD_SLOWSHOCK=njit ignores the window: unset it")
import rmhd.batched.slow_shock_b as SSB          # noqa: E402

SSB.MARGIN = float(os.environ.get("WIDE_MARGIN", "1.0"))
print("slow_shock_b.MARGIN = %g" % SSB.MARGIN, flush=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
