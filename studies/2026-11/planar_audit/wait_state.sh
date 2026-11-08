#!/bin/bash
# wait_state.sh JOBID [MAXHOURS] : poll sacct on itp every 5 min until the job
# is no longer pending or running, then print its final state.  An ssh failure
# (laptop asleep, network) is retried, never taken for "done".
J=$1; M=${2:-24}
end=$(( $(date +%s) + M * 3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  s=$(ssh -o ConnectTimeout=30 itp "sacct -j $J -X -n -o State%20" < /dev/null 2>/dev/null)
  rc=$?
  s=$(echo $s | awk '{print $1}')
  if [ $rc -eq 0 ] && [ -n "$s" ]; then
    case "$s" in
      PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED|COMPLETING|CONFIGURING) ;;
      *) echo "job $J $s at $(date)"; exit 0 ;;
    esac
  fi
  sleep 300
done
echo "timeout waiting for $J after $M h"
exit 1
