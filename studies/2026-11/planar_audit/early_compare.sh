#!/bin/bash
# early_compare.sh : treatment vs control diag.csv so far (first differing step, counts)
R=/mnt/rafast/miler/codes/rotor2d/HLLD/results
for p in "r2d r2d_64_exact_ctl1102" "ot ot_64_exact_ssretry" "rotor rotor_64_exact_ssretry"; do
  set -- $p
  T=$R/${1}_64_exact_admiss/diag.csv; C=$R/$2/diag.csv
  n=$(($(wc -l < $T) - 1))
  first=$(diff <(head -$((n + 1)) $C) $T | grep -m1 '^>' | cut -d, -f1 | tr -d '> ')
  echo "$1: $n steps done; first step whose diag differs from $2: ${first:-none}"
  if [ -n "$first" ]; then
    echo "   control  : $(sed -n "$((first + 1))p" $C)"
    echo "   treatment: $(sed -n "$((first + 1))p" $T)"
  fi
done
