#!/bin/bash
# progress_adm64.sh : treatment step count and cumulative wall vs the control at the same step
L=/mnt/rafast/miler/codes/rotor2d/HLLD/logs
for p in "r2d slurm_adm64-r2d_77965.out slurm_r2d-ctl64_75443.out" \
         "ot slurm_adm64-ot_77966.out slurm_orsza-ssr_60939.out" \
         "rotor slurm_adm64-rotor_77967.out slurm_rotor-ssr_60938.out"; do
  set -- $p
  last=$(grep -E "^\s*step\s+[0-9]+ t=" $L/$2 | tail -1)
  n=$(echo "$last" | awk '{print $2}')
  tw=$(echo "$last" | sed -E 's/.*\(([0-9]+)s\)\s*$/\1/')
  cw=$(grep -E "^\s*step\s+$n t=" $L/$3 | head -1 | sed -E 's/.*\(([0-9]+)s\)\s*$/\1/')
  tot=$(grep -E "^\s*step\s+[0-9]+ t=" $L/$3 | tail -1 | sed -E 's/.*\(([0-9]+)s\)\s*$/\1/')
  echo "$1: step ${n:-0}  treatment ${tw:-?} s  control ${cw:-?} s at the same step  (control total ${tot:-?} s)"
done
