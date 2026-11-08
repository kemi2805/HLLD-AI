#!/bin/bash
# wait_log_line.sh JOB PATTERN [N]: wait until the calea job's slurm log has a
# line matching PATTERN (grep -E), or the job has ended; then print the lines
# starting with "==" and the last N (default 10) lines.  ssh failures retried.
J=$1; P=$2; N=${3:-10}
SSH="ssh -o ConnectTimeout=20 -o BatchMode=yes itp"
LOGGLOB="/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_*_$J.out"
while true; do
    hit=$($SSH "grep -cE '$P' $LOGGLOB 2>/dev/null" < /dev/null 2>/dev/null)
    st=$($SSH "sacct -j $J -X -n -o State%30 | head -1 | awk '{print \$1}'" < /dev/null 2>/dev/null)
    if [ -n "$hit" ] && [ "$hit" -gt 0 ] 2>/dev/null; then break; fi
    case "$st" in
        ""|PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED) sleep 120 ;;
        *) break ;;
    esac
done
echo "== $(date '+%a %H:%M') job $J state ${st:-?}, pattern '$P' hits ${hit:-?}"
$SSH "grep -E '^== ' $LOGGLOB; echo ...; tail -n $N $LOGGLOB" < /dev/null
