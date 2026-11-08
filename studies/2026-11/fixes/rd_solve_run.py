#!/usr/bin/env python
"""Production's own answers outside the regular seven-wave family: in the
256,000-answer sample of the 4Q 128^2 run (verify/run), every answer with a
slow jump that crosses the Alfven speed is solved again as a regular one --
the coplanar rotation (closed form, either lab sign) in front of the slow
wave, signed-field slow waves, four unknowns, Newton from the answer itself,
combinations both / left / right -- and judged by the independent verifier.

    python rd_solve_run.py CHUNK.npz OUT.npy      (CHUNK: verify/run/chunk_k.npz, its _verify.npz beside it)
"""
import sys
import numpy as np
import rd_solve as RS
import verify_rmhd as VR

g = RS.g
W = RS.W


def main():
    f, out_path = sys.argv[1], sys.argv[2]
    a = np.load(f, allow_pickle=True)
    v = np.load(f.replace(".npz", "_verify.npz"), allow_pickle=True)
    idx = np.flatnonzero(v["inter"].any(axis=1))
    rows = []
    for k in idx:
        UL, UR = a["U_L"][k], a["U_R"][k]
        Bn = np.array([UL[7]])
        L0, R0, alpha, _ = RS.S5["to_planar"](RS.one(UL[:7]), RS.one(UR[:7]), Bn)
        Zold, Sold = a["zones"][k], a["speeds"][k]
        ca, sa = np.cos(alpha[0]), np.sin(alpha[0])
        bt = lambda z: ca * z[5] - sa * z[6]
        u0 = [np.array([np.log(Zold[1][1])]), np.array([bt(Zold[3])]), np.array([bt(Zold[4])]),
              np.array([np.log(Zold[6][1])])]
        # the original answer's face flux, from its own zones (planar frame)
        Zp = Zold.copy()
        for j in range(8):
            By, Bz, vy, vz = Zp[j, 5], Zp[j, 6], Zp[j, 3], Zp[j, 4]
            Zp[j, 5], Zp[j, 6] = ca * By - sa * Bz, sa * By + ca * Bz
            Zp[j, 3], Zp[j, 4] = ca * vy - sa * vz, sa * vy + ca * vz
        X0, where0 = RS.face_state(Zp, Sold, Bn[0], v["kind"][k])
        F0 = RS.flux(X0, Bn[0]) if X0 is not None else None
        res = dict(chunk=f, k=int(k), src=str(a["source"][k]), adm_old=bool(v["admissible"][k]),
                   dS_old=float(np.nanmin(np.where(v["inter"][k], v["dS"][k], np.nan))), combos=[])
        for rot in ((True, True), (True, False), (False, True)):
            u, nrm, st, it = RS.newton(L0, R0, u0, Bn, rot)
            if not np.isfinite(nrm) or nrm > 1e-9:
                res["combos"].append((rot, False, False, np.nan))
                continue
            Z, V = RS.zones_of(L0, R0, st)
            UL8 = np.concatenate([Z[0, 0], [Bn[0]]])[None]
            UR8 = np.concatenate([Z[0, 7], [Bn[0]]])[None]
            VR.FAILWHY.clear()
            r = VR.verify(UL8, UR8, Z, V, g)
            dF = np.nan
            if F0 is not None:
                X1, _ = RS.face_state(Z[0], V[0], Bn[0], r["kind"][0])
                if X1 is not None:
                    F1 = RS.flux(X1, Bn[0])
                    dF = float(np.max(np.abs(F1 - F0)) / np.max(np.abs(F0)))
            res["combos"].append((rot, bool(r["weak"][0]), bool(r["admissible"][0]), dF))
            res.setdefault("detail", []).append(dict(rot=rot, kind=r["kind"][0], dS=r["dS"][0], inter=r["inter"][0],
                                                     rh=r["rh"][0], order_ok=bool(r["order_ok"][0]),
                                                     failwhy=list(VR.FAILWHY), V=V[0], Z=Z[0]))
        rows.append(res)
    np.save(out_path, np.array(rows, dtype=object), allow_pickle=True)
    n_adm = sum(any(c[2] for c in r["combos"]) for r in rows)
    print("%s: %d answers with an Alfven-crossing slow jump; a regular admissible answer for %d" % (f, len(rows), n_adm))


if __name__ == "__main__":
    main()
