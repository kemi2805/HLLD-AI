#!/bin/bash
# wait for the replay pilot 85217 to leave the queue, then show its parts' ends
SSH="ssh -o ConnectTimeout=20 -o BatchMode=yes itp"
for i in $(seq 1 120); do
    st=$($SSH "sacct -j 85217 -X -n -o State" < /dev/null 2>/dev/null | tr -d ' ')
    case "$st" in RUNNING|PENDING|"") sleep 30 ;; *) break ;; esac
done
echo "85217: $st"
$SSH 'O=/mnt/rafast/miler/codes/rotor2d/HLLD/results/planar_ml/replay_v3; for f in $O/netreg_pilot3_*.log; do echo "== $f"; tail -n 3 $f | cut -c1-200; done; tail -n 4 /mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_rpn-pilot_85217.out; ls $O; squeue -u miler -o "%.8i %.14j %.9T %.20R" | grep -E "rpn|verify"' < /dev/null
