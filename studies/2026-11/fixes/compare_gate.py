#!/usr/bin/env python
"""The admissibility gate, paired: replay_pop.py outputs of the same ledger
population with RMHD_ADMISSIBLE_GATE off and on, for production and for the
final loss fixes.

    python compare_gate.py POP.npz TYPED.npz DIR [ARMS [PAIRS]]
        (DIR/replay_<arm>.npz; ARMS default prod,prodgate,final,finalgate,
         PAIRS default prod:prodgate,final:finalgate)

Per pair (gate off -> on): the faces the gate turned away (exact before, not
after), the faces it sent to another rung that answered them (exact in both,
the answering rung or the flux changed), and how far the flux moved there;
the refusals by rung (seven-wave reason 6, planar reason 7, four-unknown: the
answering rung was planar4 and the face is lost or re-answered).
"""
import sys
import numpy as np

pop, typed = np.load(sys.argv[1], allow_pickle=True), np.load(sys.argv[2], allow_pickle=True)
D = sys.argv[3]
ARMS = (sys.argv[4] if len(sys.argv) > 4 else "prod,prodgate,final,finalgate").split(",")
PAIRS = [tuple(x.split(":")) for x in (sys.argv[5] if len(sys.argv) > 5 else "prod:prodgate,final:finalgate").split(",")]
arms = {a: np.load("%s/replay_%s.npz" % (D, a), allow_pickle=True) for a in ARMS}
lanes = arms[ARMS[0]]["lane"]
for a in arms.values():
    assert (a["lane"] == lanes).all()
grp = pop["group"][lanes]
typ = typed["typ"][lanes]
UL, UR = pop["UL"][lanes], pop["UR"][lanes]


def beta(U):
    rho, Pt, vx, vy, vz, By, Bz, Bx = (U[:, j] for j in range(8))
    W2 = 1.0 / (1.0 - (vx * vx + vy * vy + vz * vz))
    eta = Bx * vx + By * vy + Bz * vz
    b2 = (Bx * Bx + By * By + Bz * Bz) / W2 + eta * eta
    return 2.0 * (Pt - 0.5 * b2) / np.maximum(b2, 1e-300)


bmin = np.minimum(beta(UL), beta(UR))
kind = np.where(typ == "A shear/degenerate", "A",
                np.where(typ == "B near-normal", np.where(bmin >= 1.0, "B weak", "B strong"), "?"))
SRC = {0: "lost", 1: "seven", 2: "planar5", 3: "planar4"}


def flux(r):
    fk = sorted(k for k in r.files if k.startswith("F_"))
    return np.stack([r[k] for k in fk])


print("== exact faces per arm (population: %d lost faces of the 128^2 run, %d controls it answered)"
      % ((grp == 1).sum(), (grp == 0).sum()))
print("%-10s %6s %6s %6s %6s %6s %9s" % ("arm", "A", "Bweak", "Bstrg", "lost", "ctrl", "ctrl lost"))
for n, r in arms.items():
    e = r["exact"]
    print("%-10s %6d %6d %6d %6d %6d %9d" % (
        n, (e & (grp == 1) & (kind == "A")).sum(), (e & (grp == 1) & (kind == "B weak")).sum(),
        (e & (grp == 1) & (kind == "B strong")).sum(), (e & (grp == 1)).sum(), (e & (grp == 0)).sum(),
        (~e & (grp == 0)).sum()))

for off_n, on_n in PAIRS:
    off, on = arms[off_n], arms[on_n]
    e0, e1 = off["exact"], on["exact"]
    F0, F1 = flux(off), flux(on)
    sc = np.max(np.abs(F0), axis=0) + 1e-300
    dF = np.max(np.abs(F1 - F0), axis=0) / sc
    s0, s1 = off["src"], on["src"]
    moved = e0 & e1 & ((s0 != s1) | (dF > 0))
    away = e0 & ~e1
    gained = ~e0 & e1
    print("\n== %s -> %s" % (off_n, on_n))
    for g, gname in ((0, "controls"), (1, "lost faces")):
        m = grp == g
        print("  %-10s exact %4d -> %4d | turned away %3d, re-answered by another rung %3d, gained %d, unchanged bitwise %d"
              % (gname, (e0 & m).sum(), (e1 & m).sum(), (away & m).sum(), (moved & m).sum(), (gained & m).sum(),
                 (e0 & e1 & m & ~moved).sum()))
    hit = away | moved
    if hit.any():
        print("  refused answers by the rung that gave them (gate off):",
              {SRC[k]: int((s0[hit] == k).sum()) for k in (1, 2, 3)})
        print("  gate reasons (gate on): seven-wave refused %d, planar refused %d"
              % ((on["reason7"][hit] == 6).sum(), (on["reason5"][hit] == 7).sum()))
        print("  re-answered by:", {SRC[k]: int((s1[moved] == k).sum()) for k in (1, 2, 3)})
        if moved.any():
            q = np.percentile(dF[moved], [50, 90, 100])
            print("  flux moved where re-answered: p50 %.1e p90 %.1e max %.1e" % tuple(q))
        ty = {k: int(((kind == k) & hit).sum()) for k in ("A", "B weak", "B strong", "?")}
        print("  refused faces by type:", ty)
