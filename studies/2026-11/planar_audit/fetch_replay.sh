#!/bin/bash
# fetch_replay.sh CONFIG... : copy replay/<CONFIG>/{faces.npz,harvest/solved_*} from calea
set -u
R=/mnt/rafast/miler/codes/rotor2d/HLLD/results/r2d_retrain/planar_audit/replay
D=/Users/miler/Codes/claude_scratch/r2d_retrain/planar_audit/replay
for c in "$@"; do
  mkdir -p "$D/$c/harvest"
  scp -q "itp:$R/$c/faces.npz" "$D/$c/" < /dev/null
  scp -q "itp:$R/$c/harvest/solved_*.npz" "$D/$c/harvest/" < /dev/null
  echo "$c: $(ls $D/$c/harvest | wc -l) shards"
done
