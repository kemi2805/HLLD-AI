"""scripts/run_2d.py with tpatch2.s timers and statistics; run from HLLD_DIR."""
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tpatch2  # noqa: E402,F401

sys.argv = ["scripts/run_2d.py"] + sys.argv[1:]
runpy.run_path("scripts/run_2d.py", run_name="__main__")
