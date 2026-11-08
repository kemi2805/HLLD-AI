#!/bin/bash
# The 4Q L1 table at 64^2 and 128^2 against the three 512^2 references
# (2026-11-04).  Run from claude_scratch/r2d_retrain.
F=runs128/refs
exec /Users/miler/Codes/py-work/bin/python l1_table.py cmp128 \
    --ref hlld=$F/r2d_512_hlld__snap_008.npz --ref hllc=$F/r2d_512_hllc__snap_008.npz \
    --ref hlle=$F/r2d_512_hlle__snap_008.npz \
    hlle@64=$F/r2d_64_hlle__snap_008.npz hllc@64=$F/r2d_64_hllc__snap_008.npz \
    hlld@64=$F/r2d_64_hlld__snap_008.npz exact@64=runs64/it3/snap_008.npz \
    exact_prodnet@64=runs64/ctl/snap_008.npz \
    hlle@128=$F/r2d_128_hlle__snap_008.npz hllc@128=$F/r2d_128_hllc__snap_008.npz \
    hlld@128=$F/r2d_128_hlld__snap_008.npz exact@128=runs64/ft3_128/snap_008.npz
