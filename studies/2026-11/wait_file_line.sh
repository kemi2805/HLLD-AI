#!/bin/bash
# wait_file_line.sh FILE PATTERN [JOB]: wait until FILE on itp has a line
# matching PATTERN (grep -E), or calea job JOB has ended; print the file's
# last lines.  ssh failures retried.
F=$1; P=$2; J=${3:-}
SSH="ssh -o ConnectTimeout=20 -o BatchMode=yes itp"
while true; do
    hit=$($SSH "grep -cE '$P' $F 2>/dev/null" < /dev/null 2>/dev/null)
    if [ -n "$hit" ] && [ "$hit" -gt 0 ] 2>/dev/null; then break; fi
    if [ -n "$J" ]; then
        st=$($SSH "sacct -j $J -X -n -o State%30 | head -1 | awk '{print \$1}'" < /dev/null 2>/dev/null)
        case "$st" in ""|PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED) ;; *) break ;; esac
    fi
    sleep 60
done
echo "== $(date '+%a %H:%M') $F: pattern '$P' hits ${hit:-?} (job ${J:-none} ${st:-})"
$SSH "tail -n 12 $F" < /dev/null
