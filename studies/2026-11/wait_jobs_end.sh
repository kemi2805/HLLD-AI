#!/bin/bash
# wait_jobs_end.sh JOB [JOB ...]: wait until every listed calea job has left
# the queue (ssh failures retried), then print their sacct states.
SSH="ssh -o ConnectTimeout=20 -o BatchMode=yes itp"
J=$(IFS=,; echo "$*")
while true; do
    left=$($SSH "squeue -h -j $J -o %i 2>/dev/null | wc -l" < /dev/null 2>/dev/null)
    if [ -n "$left" ] && [ "$left" -eq 0 ] 2>/dev/null; then break; fi
    sleep 120
done
echo "== $(date '+%a %H:%M') all of $J ended"
$SSH "sacct -j $J -X -n -o JobID,JobName%16,State,Elapsed,NodeList" < /dev/null
