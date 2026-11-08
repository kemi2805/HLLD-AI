#!/bin/bash
# fetch the paired replay of where the planar stage runs (calea 79680: finalrot / first / after / late) and compare its arms
set -u
F=/Users/miler/Codes/claude_scratch/r2d_retrain/fixes
mkdir -p $F/replay_c7l
scp -q 'itp:/mnt/rafast/miler/codes/rotor2d/HLLD/results/fixes/replay_c7l/*' $F/replay_c7l/ < /dev/null
for a in finalrot finalrotc7 finalrotc7a finalrotc7l; do echo "$a: $(grep -v -i warn $F/replay_c7l/replay_$a.log | tail -n 1)"; done
cd $F && /Users/miler/Codes/py-work/bin/python compare_gate.py \
    /Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz \
    /Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/lanes_typed.npz \
    replay_c7l finalrot,finalrotc7,finalrotc7a,finalrotc7l finalrot:finalrotc7l,finalrotc7:finalrotc7l,finalrotc7a:finalrotc7l 2>&1 | grep -v -i warn
/Users/miler/Codes/py-work/bin/python - <<'PY'
import numpy as np
from collections import Counter
F = "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes"
pop = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz", allow_pickle=True)
grp = pop["group"]
SRC = {0: "lost", 1: "seven", 2: "planar5", 3: "planar4", 4: "planar-seven"}
for arm in ("finalrot", "finalrotc7", "finalrotc7a", "finalrotc7l"):
    r = np.load(F + "/replay_c7l/replay_%s.npz" % arm)
    for g, name in ((1, "lost faces"), (0, "controls")):
        m = grp == g
        c = Counter(SRC[int(s)] for s in r["src"][m])
        extra = ""
        if "c7_stage" in r.files and r["c7"].any():
            st = Counter(int(s) for s in r["c7_stage"][m & r["c7"]])
            rot = int((r["c7"][m] & r["c7_rot"][:, m].any(axis=0)).sum())
            extra = "  planar-seven stages %s, with a rotation %d" % (dict(sorted(st.items())), rot)
        print("%-13s %-10s %s%s" % (arm, name, dict(c), extra))
PY
