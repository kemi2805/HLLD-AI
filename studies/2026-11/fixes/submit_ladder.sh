#!/bin/bash
# The ladder's eight runs with their order (run on itp): 128^2 now, 256^2
# after the problem's 128^2 finished (afterok), 512^2 (rotor, OT) after 256^2.
set -eu
cd /mnt/rafast/miler/scratch/fixes
for p in riemann2d:r2d rotor:rotor orszag_tang:ot; do
    prob=${p%%:*}; s=${p#*:}
    j128=$(sbatch --parsable -p calea --job-name=lad-$s-128 --export=ALL,PROBLEM=$prob,N=128 calea_ladder.sh)
    j256=$(sbatch --parsable -p calea --job-name=lad-$s-256 --dependency=afterok:$j128 --export=ALL,PROBLEM=$prob,N=256 calea_ladder.sh)
    line="$s: 128 $j128, 256 $j256"
    if [ "$prob" != riemann2d ]; then
        j512=$(sbatch --parsable -p calea --job-name=lad-$s-512 --dependency=afterok:$j256 --export=ALL,PROBLEM=$prob,N=512 calea_ladder.sh)
        line="$line, 512 $j512"
    fi
    echo "$line"
done
