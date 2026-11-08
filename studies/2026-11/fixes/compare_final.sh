#!/bin/bash
# compare_final.sh PROBLEM JOBID : the paired 64^2 run of the FINAL loss-fix
# configuration (fixfinal2) of PROBLEM (r2d | ot | rotor) against production
# and the peer session's admissibility filter + fix, with that session's own
# tools (planar_audit: fetch_adm64.sh, adm_compare64.py, interim_where.py) so
# the columns line up: lost faces, rungs, wall, L1 vs 512^2, lost-face and
# error maps -> fixes/cmp_final_<p>/, per-face lost difference ctl vs final2
# -> fixes/where_final_<p>/.
set -u
p=$1; J=$2
A=/Users/miler/Codes/claude_scratch/r2d_retrain/planar_audit
F=/Users/miler/Codes/claude_scratch/r2d_retrain/fixes
PY=/Users/miler/Codes/py-work/bin/python
cd "$A" || exit 1
./fetch_adm64.sh "${p}_fin2:${p}_64_exact_fixfinal2:slurm_fixfin2-${p}_${J}.out"
case $p in
  r2d)   T=0.4; TI="Four-quadrant problem" ;;
  ot)    T=1.0; TI="Orszag-Tang" ;;
  rotor) T=0.4; TI="Magnetic rotor" ;;
esac
"$PY" adm_compare64.py "$F/cmp_final_$p" ctl=runs_adm64/${p}_ctl fix=runs_adm64/${p}_fix fin2=runs_adm64/${p}_fin2 \
    --ref runs_adm64/ref_$p.npz --base ctl --tend $T --title "$TI" \
    --label "ctl=production" "fix=admissibility filter + fix" "fin2=loss fixes (final)"
R=$F/runs_final_vs_ctl; mkdir -p "$R"
ln -sfn "$A/runs_adm64/${p}_ctl" "$R/${p}_ctl"
ln -sfn "$A/runs_adm64/${p}_fin2" "$R/${p}_adm"
"$PY" interim_where.py "$R" "$F/where_final_$p" "$p"
