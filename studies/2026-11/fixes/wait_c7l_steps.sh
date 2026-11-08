#!/bin/bash
# wait until the late-rung short tests have reached the comparison steps
# (4Q 30, OT 36, rotor 21), then print the same-step cost table
SSH="ssh -o ConnectTimeout=20 -o BatchMode=yes itp"
L=/mnt/rafast/miler/codes/rotor2d/HLLD/logs
need() {   # need <short> <tag> <step>
    $SSH "f=\$(ls -t $L/$1_64_exact_$2_2*.log 2>/dev/null | head -1); [ -n \"\$f\" ] && grep -cE '^\s+step +$3 .*divB' \$f" < /dev/null 2>/dev/null
}
while true; do
    ok=1
    for spec in "r2d 30" "ot 36" "rotor 21"; do
        set -- $spec
        for t in finalrotc7lb_shortgate; do
            c=$(need $1 $t $2)
            [ "${c:-0}" -ge 1 ] 2>/dev/null || ok=0
        done
    done
    [ $ok = 1 ] && break
    sleep 120
done
$SSH "python3 /mnt/rafast/miler/scratch/fixes/same_step.py; python3 /mnt/rafast/miler/scratch/fixes/cov_steps.py r2d 30 1" < /dev/null
