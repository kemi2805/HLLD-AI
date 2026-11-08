#!/bin/bash
# The six after-mode short tests' pilots (switches off) against their controls.
for J in 79643 79644 79645 79646 79647 79648; do
    bash /Users/miler/Codes/claude_scratch/r2d_retrain/wait_log_line.sh $J 'treatment' 1
done
