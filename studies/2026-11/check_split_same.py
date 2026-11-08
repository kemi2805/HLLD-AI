#!/usr/bin/env python
"""Are two split directories the same split?  (2026-11-03: split_proper.py
--bits 53 --tend 0.4 on the four-quadrant data against r2d_split2.py's
proper/, job 77245.)

    python check_split_same.py DIR_A DIR_B
"""
import sys

import numpy as np

a, b = sys.argv[1:3]
ok = True
for f in ("val.npz", "test.npz", "proxy_val.npz", "train_run.npz"):
    za, zb = np.load("%s/%s" % (a, f)), np.load("%s/%s" % (b, f))
    for k in ("U_L", "U_R"):
        same = za[k].shape == zb[k].shape and np.array_equal(za[k], zb[k])
        ok &= same
        print("  %-14s %s: %s vs %s  %s" % (f, k, za[k].shape, zb[k].shape, "identical" if same else "DIFFERENT"))
print("the same split:", ok)
sys.exit(0 if ok else 1)
