#!/usr/bin/env python
"""Paired replay comparison: OFF.npz vs ON.npz from replay_pop.py on the same
population.  Lost lanes recovered per loss type (A / B weak / B strong), the
rung that answered, and what the switch did to lanes both arms solve.

    python compare_replay.py POP.npz TYPED.npz OFF.npz ON.npz
"""
import sys
import numpy as np

pop, typed, off, on = (np.load(f, allow_pickle=True) for f in sys.argv[1:5])
lanes = off["lane"]
assert (lanes == on["lane"]).all()
grp = pop["group"][lanes]
typ = typed["typ"][lanes] if "typ" in typed.files else np.full(len(lanes), "?")
UL, UR = pop["UL"][lanes], pop["UR"][lanes]


def beta(U):
    rho, Pt, vx, vy, vz, By, Bz, Bx = (U[:, j] for j in range(8))
    W2 = 1.0 / (1.0 - (vx * vx + vy * vy + vz * vz))
    eta = Bx * vx + By * vy + Bz * vz
    b2 = (Bx * Bx + By * By + Bz * Bz) / W2 + eta * eta
    return 2.0 * (Pt - 0.5 * b2) / np.maximum(b2, 1e-300)


bmin = np.minimum(beta(UL), beta(UR))
kind = np.where(typ == "A shear/degenerate", "A", np.where(bmin >= 1.0, "B weak", "B strong"))
kind = np.where(typ == "B near-normal", kind, np.where(typ == "A shear/degenerate", "A", "?"))
SRC = {0: "lost", 1: "seven", 2: "planar5", 3: "planar4"}
print("%-10s %6s %8s %8s %6s %6s | rungs of the recovered (ON)" % ("type", "lanes", "OFF", "ON", "gained", "lost"))
for k in ("A", "B weak", "B strong"):
    m = (grp == 1) & (kind == k)
    a, b = off["exact"][m], on["exact"][m]
    src = on["src"][m][b & ~a]
    rung = {SRC[s]: int((src == s).sum()) for s in (1, 2, 3)}
    print("%-10s %6d %8d %8d %6d %6d | %s" % (k, m.sum(), a.sum(), b.sum(), (b & ~a).sum(), (a & ~b).sum(), rung))
m = grp == 1
print("%-10s %6d %8d %8d %6d %6d" % ("all lost", m.sum(), off["exact"][m].sum(), on["exact"][m].sum(),
                                     (on["exact"][m] & ~off["exact"][m]).sum(), (off["exact"][m] & ~on["exact"][m]).sum()))
c = grp == 0
print("controls   %6d %8d %8d %6d %6d" % (c.sum(), off["exact"][c].sum(), on["exact"][c].sum(),
                                         (on["exact"][c] & ~off["exact"][c]).sum(), (off["exact"][c] & ~on["exact"][c]).sum()))
fk = [k for k in off.files if k.startswith("F_")]
F0 = np.stack([off[k] for k in fk]); F1 = np.stack([on[k] for k in fk])
sc = np.max(np.abs(F0), axis=0) + 1e-300
dF = np.max(np.abs(F1 - F0), axis=0) / sc
both = off["exact"] & on["exact"]
print("lanes exact in both arms: %d; flux bitwise equal on %d; max rel diff %.2e" % (
    both.sum(), (dF[both] == 0).sum(), dF[both].max() if both.any() else 0))
new = on["exact"] & ~off["exact"]
if new.any():
    q = np.percentile(dF[new], [50, 90, 100])
    print("recovered lanes: flux moved from HLLD by p50 %.1e p90 %.1e max %.1e" % tuple(q))
