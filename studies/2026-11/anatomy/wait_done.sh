#!/bin/bash
# wait_done.sh JOB MAX_SECONDS: block until the calea job leaves RUNNING/PENDING or MAX_SECONDS pass
J=$1; T=${2:-300}; t=0
while [ $t -lt $T ]; do
    st=$(ssh -o ConnectTimeout=20 -o BatchMode=yes itp "sacct -j $J -X -n -o State | head -1" < /dev/null | tr -d ' ')
    case "$st" in RUNNING|PENDING|"") sleep 20; t=$((t + 20)) ;; *) echo "$J $st"; exit 0 ;; esac
done
echo "$J still running after $T s"
