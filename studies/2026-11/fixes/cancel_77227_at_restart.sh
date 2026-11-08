#!/bin/bash
# Cancel 77227 right after its restart file reaches step 1380 (no step lost);
# the new run 512 (submitted with afterany:77227) then starts from that file.
SSH="ssh -o ConnectTimeout=20 -o BatchMode=yes itp"
F=/mnt/rafast/miler/codes/rotor2d/HLLD/results/r2d_512_exact_ml_r2d_it3/restart.npz
while true; do
    s=$($SSH "python3 -c \"import numpy as np; print(int(np.load('$F')['step']))\"" < /dev/null 2>/dev/null)
    if [ -n "$s" ] && [ "$s" -ge 1380 ] 2>/dev/null; then break; fi
    sleep 45
done
$SSH "scancel 77227; sleep 5; sacct -j 77227 -X -n -o JobID,State,Elapsed; python3 -c \"import numpy as np; z=np.load('$F'); print('restart step', int(z['step']), 't', float(z['t']))\"" < /dev/null
echo "== $(date '+%a %H:%M') cancelled 77227 at restart step $s"
