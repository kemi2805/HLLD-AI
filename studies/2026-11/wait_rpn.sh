#!/bin/bash
# wait_rpn.sh: wait for the three replay arms (84575-84577) to leave the
# queue, then print their states and the paired comparison.
SSH="ssh -o ConnectTimeout=20 -o BatchMode=yes itp"
J=84575,84576,84577
while true; do
    q=$($SSH "squeue -h -j $J -o %i 2>&1" < /dev/null 2>/dev/null)
    rc=$?
    if [ $rc -ne 255 ]; then
        [ -z "$q" ] && break
        echo "$q" | grep -q "Invalid job id" && break
    fi
    sleep 300
done
echo "== $(date '+%a %H:%M') ended"
$SSH "sacct -j $J -X -n -o JobID,JobName%16,State,Elapsed; cd /mnt/rafast/miler/codes/rotor2d/HLLD/results/planar_ml/replay_v2 && ls *.npz && python3 /mnt/rafast/miler/scratch/fixes/planar_ml/compare_replay.py off.npz net.npz netfirst.npz" < /dev/null
