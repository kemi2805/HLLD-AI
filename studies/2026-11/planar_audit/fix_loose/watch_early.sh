#!/bin/bash
# watch_early.sh: wait (<= 60 min) until the three admfix runs have printed
# their pilot verdict and the validation job has finished its replay step,
# then print those lines.  ssh failures are retried, never taken for done.
L=/mnt/rafast/miler/codes/rotor2d/HLLD/logs
end=$(( $(date +%s) + 3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  n=$(ssh -o ConnectTimeout=30 itp "grep -l 'PILOT' $L/slurm_admfix-r2d_78333.out $L/slurm_admfix-ot_78334.out $L/slurm_admfix-rotor_78335.out 2>/dev/null | wc -l; grep -c '(3) rmhd_final suite' $L/slurm_admfix-validate_78332.out 2>/dev/null" < /dev/null 2>/dev/null | tr '\n' ' ')
  set -- $n
  if [ "${1:-0}" = 3 ] && [ "${2:-0}" = 1 ]; then break; fi
  sleep 60
done
ssh -o ConnectTimeout=30 itp "grep -h -E '^== ' $L/slurm_admfix-r2d_78333.out $L/slurm_admfix-ot_78334.out $L/slurm_admfix-rotor_78335.out; echo ----; cat $L/slurm_admfix-validate_78332.out" < /dev/null
