#!/bin/bash
# wait_grep.sh FILE PATTERN [MAXSEC] : poll a calea file until PATTERN appears
F=$1; P=$2; M=${3:-1800}
for i in $(seq 1 $((M / 15))); do
  if ssh itp "grep -q -E '$P' '$F'" < /dev/null 2>/dev/null; then echo "found"; exit 0; fi
  sleep 15
done
echo "timeout"
