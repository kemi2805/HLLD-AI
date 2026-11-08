#!/bin/bash
# watch_job.sh JOBID MAXHOURS REMOTE_LOG [LINES]: wait (wait_state.sh: sacct
# every 5 min, ssh failures retried) until the calea job ends, then print the
# last LINES lines of its log.
J=$1; M=$2; L=$3; N=${4:-40}
bash /Users/miler/Codes/claude_scratch/r2d_retrain/planar_audit/wait_state.sh "$J" "$M"
ssh -o ConnectTimeout=30 itp "tail -n $N '$L'" < /dev/null
