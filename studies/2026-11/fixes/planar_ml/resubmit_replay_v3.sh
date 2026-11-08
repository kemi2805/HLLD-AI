#!/bin/bash
# 11-08: 84611 (netreg replay) lost two parts' results at the harvester's
# close (coverage maps of unequal calls do not stack; the save came after the
# close).  Fixed replay_net.py; rerun: a 3-sweep pilot, then netreg and net
# (both harvested, loose13) side by side, then the verification of both.
set -eu
cd /mnt/rafast/miler/scratch/fixes/planar_ml
R=/mnt/rafast/miler/codes/rotor2d/HLLD/results
scancel 84611 84612 || true
for i in $(seq 1 30); do
    [ -z "$(squeue -h -j 84611 2>/dev/null)" ] && break
    sleep 5
done
[ -n "$(squeue -h -j 84611 2>/dev/null)" ] && { echo "84611 still in the queue"; exit 1; }
[ -d $R/planar_ml/replay_v3 ] && mv $R/planar_ml/replay_v3 $R/planar_ml/replay_v3_crashed1108
ls -d $R/fixes/verify_pop 2>/dev/null && ls $R/fixes/verify_pop | head -3
jp=$(sbatch --parsable -p calea --job-name=rpn-pilot --time=01:00:00 --export=ALL,ARM=netreg,MAX_SWEEPS=3 calea_replay_net2.sh)
jr=$(sbatch --parsable -p calea --job-name=rpn-netreg --dependency=afterok:$jp --export=ALL,ARM=netreg calea_replay_net2.sh)
jn=$(sbatch --parsable -p calea --job-name=rpn-net --dependency=afterok:$jp --export=ALL,ARM=net calea_replay_net2.sh)
jv=$(sbatch --parsable -p calea --dependency=afterok:$jr,afterok:$jn calea_verify_pop.sh)
echo "pilot $jp, netreg $jr, net $jn, verify $jv"
