#!/bin/bash
# The three rotation runs' pilots (switches off) against their controls: wait for the
# "treatment" line of each, print the "==" lines.
for J in 79440 79441 79442; do
    bash /Users/miler/Codes/claude_scratch/r2d_retrain/wait_log_line.sh $J 'treatment' 2
done
