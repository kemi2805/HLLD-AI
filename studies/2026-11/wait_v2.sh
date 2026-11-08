#!/bin/bash
# wait_v2.sh: wait for hv-ot 81498, hv-rotor 81499 and the planar training
# 84559 (afterok on both) to leave the queue -- or for 84559 to be stuck on a
# dependency that can never be met -- then print their states and the
# training's summary lines.  An ssh failure (255) is retried; squeue's
# "Invalid job id" (every job purged) counts as ended.
SSH="ssh -o ConnectTimeout=20 -o BatchMode=yes itp"
J=81498,81499,84559
while true; do
    q=$($SSH "squeue -h -j $J -o '%i %r' 2>&1" < /dev/null 2>/dev/null)
    rc=$?
    if [ $rc -ne 255 ]; then
        echo "$q" | grep -q "Invalid job id" && break
        [ $rc -eq 0 ] && [ -z "$q" ] && break
        echo "$q" | grep -q "84559 DependencyNeverSatisfied" && { echo "== 84559 can never start"; break; }
    fi
    sleep 180
done
echo "== $(date '+%a %H:%M') ended"
$SSH "sacct -j $J -X -n -o JobID,JobName%16,State,Elapsed,End; tail -n 40 /mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_planar-train2_84559.out 2>/dev/null | cut -c1-300" < /dev/null
