#!/bin/bash
# Submit the fixed-filter validation, the three paired reruns and their post
# audit (2026-11-04), after checking the calea worktree holds the laptop's files.
set -u
S=/mnt/rafast/miler/scratch/r2d_retrain/planar_audit
W=/mnt/rafast/miler/codes/rotor2d/rmhd_final-admiss
R=/mnt/rafast/miler/codes/rotor2d/HLLD/results
ok=1
for pair in "rmhd/batched/slow_shock_b.py:4d38ebb91533a5d64f45fdf5b15a3f41" "rmhd/batched/slow_shock_njit.py:395b5201729962036bfb9d46640e9ec6" "tests/batched/test_slow_shock_admissible.py:7f901217adb8939086dedd33bac036b9"; do
    f=${pair%%:*}; want=${pair##*:}
    got=$(md5sum "$W/$f" | awk '{print $1}')
    [ "$got" = "$want" ] && echo "same $f" || { echo "DIFFERS $f ($got vs $want)"; ok=0; }
done
for d in r2d_64_exact_admfix ot_64_exact_admfix rotor_64_exact_admfix; do
    [ -e "$R/$d" ] && { echo "EXISTS $R/$d"; ok=0; }
done
for c in r2d_64_exact_ctl1102 ot_64_exact_ssretry rotor_64_exact_ssretry; do
    [ -f "$R/$c/diag.csv" ] || { echo "NO CONTROL diag $c"; ok=0; }
done
[ $ok = 1 ] || { echo "NOT SUBMITTED"; exit 1; }
cd "$S" || exit 1
v=$(sbatch --parsable calea_admfix_validate.sh)
a=$(sbatch --parsable --job-name=admfix-r2d --export=ALL,PROBLEM=riemann2d,SHORT=r2d,CTRL=r2d_64_exact_ctl1102 calea_admfix64.sh)
b=$(sbatch --parsable --job-name=admfix-ot --export=ALL,PROBLEM=orszag_tang,SHORT=ot,CTRL=ot_64_exact_ssretry calea_admfix64.sh)
c=$(sbatch --parsable --job-name=admfix-rotor --export=ALL,PROBLEM=rotor,SHORT=rotor,CTRL=rotor_64_exact_ssretry calea_admfix64.sh)
p=$(sbatch --parsable --dependency=afterany:$a:$b:$c calea_admfix64_post.sh)
echo "validate $v  r2d $a  ot $b  rotor $c  post $p"
sleep 5
squeue -u miler -o "%.10i %.16j %.8T %.10M %R" | grep -E "admfix|JOBID"
