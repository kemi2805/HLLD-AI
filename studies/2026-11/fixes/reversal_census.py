#!/usr/bin/env python
"""For every reversing slow jump the gate refuses: what admissible ways to
reverse B_t exist from the state ahead of it (behind its fast wave)?

Along the reversed slow-family Hugoniot from that state (b = reversed B_t,
|b| from 1e-4 to 12 x |B_t ahead|, the shock search window opened past the
Alfven speed as in compound_wave.py), look for
  ROTATION  a point where the jump runs at the Alfven speed of both sides,
            with rho continuous: an Alfven discontinuity reversing the field
            (isentropic) -- the regular structure is this rotation followed by
            a slow wave (fan or shock) to the contact's field;
  SONIC     a compressive (entropy-raising) intermediate shock whose flow
            behind is sonic: the Brio-Wu compound wave (shock + attached fan);
  NONE      neither: every reversing jump from that state expands.

    PYTHONPATH=rmhd_fix python reversal_census.py OUT.npz   (numpy slow shock)
"""
import sys
import numpy as np
sys.path.insert(0, "/Users/miler/Codes/HLLD/scripts")
import compound_wave as CW                                        # noqa: E402
import compound_refused as CR                                     # noqa: E402
from rmhd.batched import slow_shock_b as SSB                      # noqa: E402

g, S5, pop, ans = CR.g, CR.S5, CR.pop, CR.ans
W = CW.FB.make_waves(g)
shock = SSB.make_solver(g)
D = CR.D
fin = np.load(D + "/replay_gate/replay_final.npz")
fg = np.load(D + "/replay_gate/replay_finalgate.npz")
refused = sorted(set(np.nonzero(fin["exact"] & ~fg["exact"])[0]) & set(ans))
NB = 48
grid = np.exp(np.linspace(np.log(1e-4), np.log(12.0), NB))

cases = []                                    # (lane, side, A, target b*)
for lane in refused:
    UL, UR = pop["UL"][lane], pop["UR"][lane]
    Bn = np.array([UL[7]])
    L0, R0, alpha, _ = S5["to_planar"](CR.one(UL[:7]), CR.one(UR[:7]), Bn)
    Z, S = ans[lane]
    ca, sa = np.cos(alpha[0]), np.sin(alpha[0])
    bt = lambda z: ca * z[5] - sa * z[6]
    for side, ia, ib, P, X0, fsw in (("L", 2, 3, Z[1][1], L0, "LF"), ("R", 5, 4, Z[6][1], R0, "RF")):
        if bt(Z[ia]) * bt(Z[ib]) >= 0:
            continue
        A, _, eA = W["fast_wave"](np.array([P]), X0, fsw, Bn)
        cases.append((lane, side, [c[0] for c in A], float(bt(Z[ib])), float(Bn[0]), bool(eA[0])))
print("refused lanes with a capture: %d; reversing sides: %d" % (len(refused), len(cases)))

n = len(cases)
Ak = [np.repeat(np.array([c[2][j] for c in cases]), NB) for j in range(7)]
Bk = np.repeat(np.array([c[4] for c in cases]), NB)
sgn = -np.sign(Ak[5])
bk = sgn * np.abs(Ak[5]) * np.tile(grid, n)
out = {}
for side, sw in (("L", "LS"), ("R", "RS")):
    m = np.repeat(np.array([c[1] == side for c in cases]), NB)
    if not m.any():
        continue
    A = [c[m] for c in Ak]
    with CW.wide_window():
        V, Sx, e = shock(bk[m], np.zeros(m.sum()), A, sw, Bk[m], accuracy=1e-12)
    xs = W["xi"](Sx, sw, Bk[m])
    laS = W["alfven_speed"](Sx, Bk[m], side)
    laA = W["alfven_speed"](A, Bk[m], side)
    ent = CR.RI.entropy(Sx, Bk[m], g) / CR.RI.entropy(A, Bk[m], g) - 1.0
    out[side] = dict(idx=np.flatnonzero(m), V=V, err=np.asarray(e, bool), son=xs - V, aS=laS - V,
                     aA=laA - V, drho=Sx[0] / A[0] - 1.0, dS=ent, b=bk[m])

res = []
for k, (lane, side, A, bstar, Bn, eA) in enumerate(cases):
    o = out[side]
    j = np.flatnonzero((o["idx"] // NB) == k)
    ok = ~o["err"][j] & np.isfinite(o["V"][j])
    b, son, aS, aA, drho, dS = (o[q][j] for q in ("b", "son", "aS", "aA", "drho", "dS"))
    rot = son_ = None
    for t in range(len(j) - 1):
        if not (ok[t] and ok[t + 1]):
            continue
        # an Alfven discontinuity: rho continuous and the jump at the Alfven speed
        if rot is None and np.sign(drho[t]) != np.sign(drho[t + 1]) and max(abs(aA[t]), abs(aA[t + 1])) < 1e-4:
            rot = 0.5 * (b[t] + b[t + 1])
        if son_ is None and np.sign(son[t]) != np.sign(son[t + 1]) and dS[t] > 0 and dS[t + 1] > 0:
            son_ = 0.5 * (b[t] + b[t + 1])
    kind = "ROTATION" if rot is not None else ("SONIC" if son_ is not None else "NONE")
    reach = None
    if rot is not None:
        reach = abs(bstar) >= abs(rot)
    elif son_ is not None:
        reach = abs(bstar) >= abs(son_)
    res.append((lane, side, kind, rot, son_, bstar, A[5], reach, int(ok.sum())))

from collections import Counter
print("by kind:", Counter(r[2] for r in res))
print("rotation / sonic point inside |b*| (a fan can grow |B_t| from it to the contact's):",
      Counter((r[2], r[7]) for r in res))
lanes_ok = {}
for r in res:
    lanes_ok.setdefault(r[0], []).append(r[2] != "NONE")
print("lanes whose every reversing side has a rotation or sonic point: %d of %d" % (
    sum(all(v) for v in lanes_ok.values()), len(lanes_ok)))
for r in res[:40]:
    print("  lane %5d %s %-8s rot %s sonic %s  b* %+.4f  Bt ahead %+.4f  reach %s  valid pts %d" % (
        r[0], r[1], r[2], "%+.4f" % r[3] if r[3] is not None else "   -   ",
        "%+.4f" % r[4] if r[4] is not None else "   -   ", r[5], r[6], r[7], r[8]))
np.save(sys.argv[1] if len(sys.argv) > 1 else D + "/reversal_census.npy", np.array(res, dtype=object), allow_pickle=True)
