#!/bin/bash
# compare_late.sh PROBLEM [late]: the 64^2 runs of PROBLEM (r2d | ot | rotor) with
# the planar seven-wave rung (finalrotc7lbgate, 81297-81299) and the rotation
# re-solve (finalrotgate, 79440-79442) against production and the loss fixes
# alone (fixfinal2), with planar_audit's tools: lost faces, rungs, wall, L1 vs
# 512^2, lost-face and error maps -> fixes/cmp_late_<p>/.  Without "late" the
# late run is left out (it is still running).
set -u
p=$1; WITH_LATE=${2:-}
A=/Users/miler/Codes/claude_scratch/r2d_retrain/planar_audit
F=/Users/miler/Codes/claude_scratch/r2d_retrain/fixes
PY=/Users/miler/Codes/py-work/bin/python
case $p in
  r2d)   T=0.4; TI="Four-quadrant problem"; J2=78636; JR=79440; JL=81299 ;;
  ot)    T=1.0; TI="Orszag-Tang"; J2=78637; JR=79441; JL=81297 ;;
  rotor) T=0.4; TI="Magnetic rotor"; J2=78638; JR=79442; JL=81298 ;;
esac
cd "$A" || exit 1
RUNS="${p}_rot:${p}_64_exact_finalrotgate:slurm_rot-fin-${p}_${JR}.out"
[ -d runs_adm64/${p}_fin2 ] || RUNS="$RUNS ${p}_fin2:${p}_64_exact_fixfinal2:slurm_fixfin2-${p}_${J2}.out"
[ -n "$WITH_LATE" ] && RUNS="$RUNS ${p}_late:${p}_64_exact_finalrotc7lbgate:slurm_c7lf-${p}_${JL}.out"
./fetch_adm64.sh $RUNS
ARGS="ctl=runs_adm64/${p}_ctl fin2=runs_adm64/${p}_fin2 rot=runs_adm64/${p}_rot"
LAB=("ctl=production" "fin2=loss fixes" "rot=fixes+gate+rotation")
if [ -n "$WITH_LATE" ]; then
    ARGS="$ARGS late=runs_adm64/${p}_late"
    LAB+=("late=+ late rung")
fi
"$PY" adm_compare64.py "$F/cmp_late_$p" $ARGS --ref runs_adm64/ref_$p.npz --base ctl --tend $T --title "$TI" \
    --label "${LAB[@]}"
