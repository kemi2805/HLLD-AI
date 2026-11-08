#!/bin/bash
# fetch the paired rotation replay (calea 79433) and compare its arms
set -u
F=/Users/miler/Codes/claude_scratch/r2d_retrain/fixes
mkdir -p $F/replay_rot
scp -q 'itp:/mnt/rafast/miler/codes/rotor2d/HLLD/results/fixes/replay_rot/*' $F/replay_rot/ < /dev/null
for a in prodgate prodgaterot finalgate finalgaterot; do echo "$a: $(grep -v -i warn $F/replay_rot/replay_$a.log | tail -n 1)"; done
cd $F && /Users/miler/Codes/py-work/bin/python compare_gate.py \
    /Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz \
    /Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/lanes_typed.npz \
    replay_rot prodgate,prodgaterot,finalgate,finalgaterot prodgate:prodgaterot,finalgate:finalgaterot
