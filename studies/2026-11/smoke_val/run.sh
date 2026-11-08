#!/bin/bash
# Smoke test of patch_val_split.py: the default path must give bit-identical
# weights to the unpatched trainer; --val-data must run and validate on its file.
cd /Users/miler/Codes/claude_scratch/r2d_retrain || exit 1
PY=/Users/miler/Codes/py-work/bin/python
D=smoke_audit/it3
COMMON=(--data $D/lost_it2.npz --init /Users/miler/Codes/rmhd_final/data/ml_guess_rotor_ft.pt
        --epochs 2 --batch 64 --lr 2e-4 --seed 42 --threads 2 --device cpu --val-frac 0.1)
PYTHONPATH=/Users/miler/Codes/rmhd_final $PY /Users/miler/Codes/rmhd_final/scripts/train_variant.py \
    "${COMMON[@]}" --out smoke_val/orig.pt > smoke_val/orig.log 2>&1
PYTHONPATH=$PWD/rmhd_valsplit $PY rmhd_valsplit/scripts/train_variant.py \
    "${COMMON[@]}" --out smoke_val/patched.pt > smoke_val/patched.log 2>&1
PYTHONPATH=$PWD/rmhd_valsplit $PY rmhd_valsplit/scripts/train_variant.py \
    "${COMMON[@]}" --val-data $D/newly_won.npz --out smoke_val/valdata.pt > smoke_val/valdata.log 2>&1
for f in orig patched valdata; do echo "== $f"; grep -E "variant:|epoch|Error|error" smoke_val/$f.log | head -4; done
$PY - <<'EOF'
import torch
def flat(d, p=''):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, p + k + '.')
        elif torch.is_tensor(v):
            yield p + k, v
a = dict(flat(torch.load('smoke_val/orig.pt', weights_only=False)))
b = dict(flat(torch.load('smoke_val/patched.pt', weights_only=False)))
c = dict(flat(torch.load('smoke_val/valdata.pt', weights_only=False)))
print('default path: tensors', len(a), 'bitwise identical', a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a))
print('--val-data differs from default (expected):', not all(torch.equal(a[k], c[k]) for k in a))
EOF
