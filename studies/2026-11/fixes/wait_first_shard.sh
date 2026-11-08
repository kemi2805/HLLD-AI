#!/bin/bash
# wait until the resumed 512^2 run (84516) has written its first harvest shard, then summarise it
SSH="ssh -o ConnectTimeout=20 -o BatchMode=yes itp"
H=/mnt/rafast/miler/codes/rotor2d/HLLD/results/r2d_512_exact_ml_r2d_it3_newchain/harvest
while true; do
    n=$($SSH "ls $H 2>/dev/null | grep -c '^coverage_'" < /dev/null 2>/dev/null)
    if [ -n "$n" ] && [ "$n" -ge 1 ] 2>/dev/null; then break; fi
    st=$($SSH "squeue -h -j 84516 -o %T" < /dev/null 2>/dev/null)
    [ -z "$st" ] && break
    sleep 120
done
echo "== $(date '+%a %H:%M') first shard written"
