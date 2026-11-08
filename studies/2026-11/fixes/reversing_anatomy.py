#!/usr/bin/env python
"""What are the slow jumps the admissibility gate refuses?  For every answer of
the ledger capture (verify/ledger) on a lane the gate refused (final arm exact,
finalgate not): each slow jump's tangential field ahead and behind (signed,
along the plane of the field), its speed against the slow and Alfven speeds of
the states on both sides, the density and pressure jumps, the entropy change.

    python reversing_anatomy.py
"""
import glob
import numpy as np
import rmhd_ideal as RI
import sol_profiles as SP

D = "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes"
pop = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz", allow_pickle=True)
key = lambda A, B: [x.tobytes() + y.tobytes() for x, y in zip(np.ascontiguousarray(A), np.ascontiguousarray(B))]
lane_of = {k: i for i, k in enumerate(key(pop["UL"], pop["UR"]))}
fin = np.load(D + "/replay_gate/replay_final.npz")
fg = np.load(D + "/replay_gate/replay_finalgate.npz")
refused = set(np.nonzero(fin["exact"] & ~fg["exact"])[0])
g = float(pop["gamma"])

rows = []
for f in sorted(glob.glob(D + "/verify/ledger/chunk_*_verify.npz")):
    a = np.load(f.replace("_verify", ""), allow_pickle=True)
    r = np.load(f, allow_pickle=True)
    for i, k in enumerate(key(a["U_L"], a["U_R"])):
        L = lane_of.get(k, -1)
        rows.append((L, a["zones"][i], a["speeds"][i], a["U_L"][i, 7], r["kind"][i], r["dS"][i],
                     L in refused, str(a["source"][i])))

print("answers %d, on refused lanes %d" % (len(rows), sum(x[6] for x in rows)))


def st(Z, j):
    return [np.array([Z[j, c]]) for c in range(7)]


out = []
for L, Z, S, Bn, kind, dS, ref, src in rows:
    if not ref:
        continue
    # the plane of the tangential field: the larger outer tangential field
    bl, br = Z[0, 5:7], Z[7, 5:7]
    e = bl if np.hypot(*bl) >= np.hypot(*br) else br
    e = e / max(np.hypot(*e), 1e-300)
    for w in (2, 4):
        if kind[w] != "disc":
            continue
        left = w == 2
        ia, ib = (w, w + 1) if left else (w + 1, w)
        A, B = st(Z, ia), st(Z, ib)
        bta, btb = float(Z[ia, 5:7] @ e), float(Z[ib, 5:7] @ e)
        eigA, laA, raA, _ = SP.WB.xi_all(*A, np.array([Bn]), g)
        eigB, laB, raB, _ = SP.WB.xi_all(*B, np.array([Bn]), g)
        sA, sB = (eigA[0, 1], eigB[0, 1]) if left else (eigA[0, 2], eigB[0, 2])
        aA, aB = (laA[0], laB[0]) if left else (raA[0], raB[0])
        V = S[w]
        # Lax for the slow family: characteristics run into the jump from both sides
        lax = (sA > V > sB) if left else (sA < V < sB)
        inter = (V - aA) * (V - aB) < 0
        rhoA, pA = RI.prim(A, Bn, g)[:2]
        rhoB, pB = RI.prim(B, Bn, g)[:2]
        out.append(dict(lane=L, w="LS" if left else "RS", bta=bta, btb=btb, Bn=Bn, V=V, sA=sA, sB=sB,
                        aA=aA, aB=aB, lax=lax, inter=inter, drho=float(rhoB[0] / rhoA[0] - 1),
                        dp=float(pB[0] / pA[0] - 1), dS=float(dS[w]), src=src))

o = out
rev = [x for x in o if x["bta"] * x["btb"] < 0]
print("slow jumps on refused lanes: %d, reversing the field %d, entropy falls on %d (reversing %d)" % (
    len(o), len(rev), sum(x["dS"] < -1e-9 for x in o), sum(x["dS"] < -1e-9 for x in rev)))
fall = [x for x in o if x["dS"] < -1e-9]
print("of the entropy-falling jumps: reversing %d, Lax %d, crossing the Alfven speed %d, compressive (rho up) %d" % (
    sum(x["bta"] * x["btb"] < 0 for x in fall), sum(x["lax"] for x in fall), sum(x["inter"] for x in fall),
    sum(x["drho"] > 0 for x in fall)))
q = lambda v: np.percentile(v, [0, 50, 100])
print("|Bt| ahead / |Bn|: %s" % q([abs(x["bta"] / x["Bn"]) for x in fall]))
print("|Bt| behind / |Bn|: %s" % q([abs(x["btb"] / x["Bn"]) for x in fall]))
print("Bt behind / Bt ahead: %s" % q([x["btb"] / x["bta"] for x in fall]))
print("rho jump: %s" % q([x["drho"] for x in fall]))
print("p jump: %s" % q([x["dp"] for x in fall]))
print("(V - Alfven ahead): %s" % q([x["V"] - x["aA"] for x in fall]))
print("(V - Alfven behind): %s" % q([x["V"] - x["aB"] for x in fall]))
print("(V - slow ahead): %s" % q([x["V"] - x["sA"] for x in fall]))
print("(V - slow behind): %s" % q([x["V"] - x["sB"] for x in fall]))
print("(Alfven - slow) ahead: %s" % q([x["aA"] - x["sA"] for x in fall]))
print()
print("%5s %2s %10s %10s %9s %9s %10s %10s %10s %10s %5s %5s" % (
    "lane", "w", "Bt_a/Bn", "Bt_b/Bn", "drho", "dS", "V-sA", "V-sB", "V-aA", "V-aB", "lax", "inter"))
for x in sorted(fall, key=lambda x: x["dS"])[:25]:
    print("%5d %2s %10.3e %10.3e %9.2e %9.2e %10.2e %10.2e %10.2e %10.2e %5s %5s" % (
        x["lane"], x["w"], x["bta"] / x["Bn"], x["btb"] / x["Bn"], x["drho"], x["dS"], x["V"] - x["sA"],
        x["V"] - x["sB"], x["V"] - x["aA"], x["V"] - x["aB"], x["lax"], x["inter"]))
np.save(D + "/reversing_anatomy.npy", np.array(out, dtype=object), allow_pickle=True)
