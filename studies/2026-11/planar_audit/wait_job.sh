#!/bin/bash
# wait_job.sh JOBID : poll squeue on itp until the job leaves the queue
J=$1
for i in $(seq 1 360); do
  s=$(ssh itp "squeue -h -j $J -o %T" < /dev/null 2>/dev/null)
  if [ -z "$s" ]; then echo "job $J done"; exit 0; fi
  sleep 10
done
echo "timeout waiting for $J"
