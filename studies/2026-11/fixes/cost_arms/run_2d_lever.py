"""scripts/run_2d.py with tpatch's stage timers, and planar_njit.CHECK (the
lanes numpy and the compiled kernels disagree on, RMHD_PLANAR_NJIT=check)
written to $RUNG_TIMES's directory as njit_check.json at exit."""
import atexit
import json
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tpatch  # noqa: E402,F401
from rmhd.batched import planar_njit as _PNJ  # noqa: E402


def _dump_check():
    with open(os.path.join(os.path.dirname(os.environ["RUNG_TIMES"]), "njit_check.json"), "w") as f:
        json.dump({"mode": _PNJ.mode(), "check": _PNJ.CHECK}, f, indent=1)


atexit.register(_dump_check)
sys.argv = ["scripts/run_2d.py"] + sys.argv[1:]
runpy.run_path("scripts/run_2d.py", run_name="__main__")
