#!/bin/bash
# wait_one.sh JOB LOG: wait until a calea job leaves the queue, then print
# its state and the last lines of its log (ssh failures retried).
SSH="ssh -o ConnectTimeout=20 -o BatchMode=yes itp"
while true; do
    q=$($SSH "squeue -h -j $1 -o %i 2>&1" < /dev/null 2>/dev/null)
    rc=$?
    if [ $rc -ne 255 ]; then
        [ -z "$q" ] && break
        echo "$q" | grep -q "Invalid job id" && break
    fi
    sleep 60
done
$SSH "sacct -j $1 -X -n -o JobID,JobName%16,State,Elapsed; tail -n 30 $2 | cut -c1-400" < /dev/null
