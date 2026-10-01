"""The exact flux's rungs are production defaults (2026-10-01): the planar
ladder, the crossed offer, the four-unknown solver, the linearised flux
below the gate, and (in rmhd_final) the fast-shock edge scan.  The tests
here were written against the base solver and test one feature at a time,
so the suite runs with every rung pinned OFF; a test that is about a rung
turns it on, and the tests of the defaults pop the pins, reload and restore
them.  Set before any test module imports the solver."""
import os

PINNED = {"RMHD_PLANAR5_LADDER": "0", "RMHD_PLANAR5_CROSSED": "0",
          "RMHD_PLANAR4": "0", "RMHD_WEAK_FLUX": "hlld",
          "RMHD_FAST_EDGE_SCAN": "0"}
for _k, _v in PINNED.items():
    os.environ.setdefault(_k, _v)
