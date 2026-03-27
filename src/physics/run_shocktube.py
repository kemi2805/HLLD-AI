#!/usr/bin/env python3
"""
Entry point for the 1D SR-MHD shocktube solver.

Usage
-----
    python run_shocktube.py                      # uses config.yaml
    python run_shocktube.py my_config.yaml
"""

import sys
import time
import yaml
from pathlib import Path

from src.physics.driver import run

if __name__ == "__main__":
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/shocktube.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    t_start = time.perf_counter()
    run(cfg)
    t_end = time.perf_counter()
    print(f"Run completed in {t_end - t_start:.3f} s")
