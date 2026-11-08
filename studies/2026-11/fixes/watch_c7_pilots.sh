#!/bin/bash
# The three short planar-stage runs' pilots (switches off) against their controls.
for J in 79611 79612 79613; do
    bash /Users/miler/Codes/claude_scratch/r2d_retrain/wait_log_line.sh $J 'treatment' 2
done
