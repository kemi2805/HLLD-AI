#!/bin/bash
# compare_fix.sh PROBLEM JOBID : fetch the fixed-filter rerun of PROBLEM (r2d | ot | rotor)
# into runs_adm64/<p>_fix, then (1) adm_compare64.py control / round 1 / rerun -> cmp_fix_<p>/
# (difference row: rerun - control), (2) interim_where.py per-face lost difference, control vs
# rerun -> where_fix_<p>/ and round 1 vs rerun -> where_fix_<p>_vs_adm/.
set -u
p=$1; J=$2
H=/Users/miler/Codes/claude_scratch/r2d_retrain/planar_audit
PY=/Users/miler/Codes/py-work/bin/python
cd "$H" || exit 1
./fetch_adm64.sh "${p}_fix:${p}_64_exact_admfix:slurm_admfix-${p}_${J}.out"
case $p in
  r2d)   T=0.4; TI="Four-quadrant problem" ;;
  ot)    T=1.0; TI="Orszag-Tang" ;;
  rotor) T=0.4; TI="Magnetic rotor" ;;
esac
"$PY" adm_compare64.py "cmp_fix_$p" ctl=runs_adm64/${p}_ctl adm=runs_adm64/${p}_adm fix=runs_adm64/${p}_fix \
    --ref runs_adm64/ref_$p.npz --base ctl --tend $T --title "$TI" \
    --label "ctl=production" "adm=filter, round 1" "fix=filter + fix"
for pair in "ctl:${p}_ctl" "adm:${p}_adm"; do
    k=${pair%%:*}; src=${pair##*:}
    R=runs_fix_vs_$k; mkdir -p $R
    ln -sfn "$H/runs_adm64/$src" "$R/${p}_ctl"
    ln -sfn "$H/runs_adm64/${p}_fix" "$R/${p}_adm"
    out=where_fix_$p; [ $k = adm ] && out=where_fix_${p}_vs_adm
    "$PY" interim_where.py "$R" "$out" "$p"
done
