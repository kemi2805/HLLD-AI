#!/usr/bin/env python
"""The regular seven-wave answer for faces whose four-unknown answer the gate
refuses: on each side the fast wave, then an Alfven rotation (through pi in
the fluid frame; in the lab it may land on either sign of B_t), then a slow
wave steered by the signed tangential field (a fan carries B_t through zero
where it has to), meeting at the contact.  Four unknowns as in planar4
[ln P behind LF, B_t behind LS, B_t behind RS, ln P behind RF] against the
four contact jumps; the rotations carry no unknown.

Every combination of {no rotation, each rotation the Hugoniot scan finds} on
the two sides is solved by damped Newton (central differences) from our
refused answer; a converged answer is then judged by the independent
ideal-RMHD verifier (verify_rmhd.verify: jump conditions, fans re-integrated,
entropy, order) and its face flux compared with the refused answer's and, where
one exists, the face's own fine tube.

    PYTHONPATH=rmhd_fix python rd_solve.py OUT.npz LANE [LANE ...]   (numpy slow shock)
"""
import os
import sys
import itertools
import numpy as np
import refused_data as CR
import rd_helpers as RH
import sol_profiles as SP
import verify_rmhd as VR
import rmhd_ideal as RI

g, pop, ans, S5 = CR.g, CR.pop, CR.ans, CR.S5
from rmhd.batched import fullcontact_b as FB
W = FB.make_waves(g)
S4 = SP.S4
XI = SP.XI
KEYS = ("D", "Sx", "Sy", "Sz", "tau", "By", "Bz")


def one(x):
    return [np.array([float(c)]) for c in x]


def rotation(A, side, Bn):
    """The coplanar rotation from A in closed form (whichever lab sign it
    lands on), its speed, and err where none exists."""
    Vs = W["alfven_speed"](A, Bn, side)
    r = RH.coplanar_rotations(A, Bn, Vs, g)
    (Sr, okr), (Ss, oks) = r["rev"], r["same"]
    out = [np.where(okr, a, b) for a, b in zip(Sr, Ss)]
    return out, Vs, ~(okr | oks)


def structure(L0, R0, u, Bn, rot):
    """rot = (left, right): put the coplanar rotation in front of that slow wave."""
    A, VLF, eA = W["fast_wave"](np.exp(u[0]), L0, "LF", Bn)
    D, VRF, eD = W["fast_wave"](np.exp(u[3]), R0, "RF", Bn)
    bad = eA | eD
    A2, D2, VLA, VRA = A, D, W["alfven_speed"](A, Bn, "L"), W["alfven_speed"](D, Bn, "R")
    if rot[0] and not bad[0]:
        A2, VLA, e = rotation(A, "L", Bn)
        bad |= e
    if rot[1] and not bad[0]:
        D2, VRA, e = rotation(D, "R", Bn)
        bad |= e
    B, VLS, eB, shL = S4["slow_wave_s"](u[1], A2, "LS", Bn)
    C, VRS, eC, shR = S4["slow_wave_s"](u[2], D2, "RS", Bn)
    bad |= eB | eC
    Pbar = 0.5 * (L0[1] + R0[1])
    f = np.stack([B[2] - C[2], B[3] - C[3], (B[5] - C[5]) / np.sqrt(2.0 * Pbar), (B[1] - C[1]) / Pbar])
    f = np.where(bad[None], np.nan, f)
    return f, dict(A=A, A2=A2, B=B, C=C, D2=D2, D=D, V=[VLF, VLA, VLS, 0.5 * (B[2] + C[2]), VRS, VRA, VRF],
                   shocks=(shL, shR))


def newton(L0, R0, u0, Bn, rot, max_iter=40, tol=1e-11):
    u = [np.array(c, dtype=float, copy=True) for c in u0]
    f, st = structure(L0, R0, u, Bn, rot)
    nrm = np.nanmax(np.abs(f)) if np.isfinite(f).all() else np.inf
    for it in range(max_iter):
        if nrm <= tol:
            break
        J = np.zeros((4, 4))
        okJ = True
        for j in range(4):
            h = 1e-7 * max(1.0, abs(u[j][0]))
            up = [c.copy() for c in u]; um = [c.copy() for c in u]
            up[j] += h; um[j] -= h
            fp, _ = structure(L0, R0, up, Bn, rot)
            fm, _ = structure(L0, R0, um, Bn, rot)
            if not (np.isfinite(fp).all() and np.isfinite(fm).all()):
                okJ = False
                break
            J[:, j] = (fp[:, 0] - fm[:, 0]) / (2 * h)
        if not okJ:
            return u, np.inf, st, it
        try:
            p = np.linalg.lstsq(J, -f[:, 0], rcond=1e-14)[0]
        except np.linalg.LinAlgError:
            return u, np.inf, st, it
        cap = min(1.0, 0.5 / max(np.abs(p).max(), 1e-300))
        lam = cap
        for _ in range(30):
            cand = [u[j] + lam * p[j] for j in range(4)]
            fc, sc = structure(L0, R0, cand, Bn, rot)
            nc = np.nanmax(np.abs(fc)) if np.isfinite(fc).all() else np.inf
            if nc < nrm:
                u, f, st, nrm = cand, fc, sc, nc
                break
            lam *= 0.5
        else:
            return u, nrm, st, it
    return u, nrm, st, it


def zones_of(L0, R0, st):
    Z = np.zeros((1, 8, 7))
    for j, X in enumerate((L0, st["A"], st["A2"], st["B"], st["C"], st["D2"], st["D"], R0)):
        Z[0, j] = [float(c[0]) for c in X]
    V = np.array([[float(v[0]) for v in st["V"]]])
    return Z, V


def face_state(Z, V, Bn, kinds):
    """The state at xi = 0: the zone whose range holds 0, or inside a fan."""
    Bn1 = np.array([Bn])
    # wave ranges: discs at V, fans from ahead to behind characteristic speed
    edges = []
    for w in range(7):
        k = kinds[w]
        if k == "fan":
            left = w < 3
            ia, ib = (w, w + 1) if left else (w + 1, w)
            sw = {0: "LF", 2: "LS", 4: "RS", 6: "RF"}[w]
            la = float(XI([np.array([Z[ia, c]]) for c in range(7)], sw, Bn1)[0])
            lb = float(XI([np.array([Z[ib, c]]) for c in range(7)], sw, Bn1)[0])
            edges.append((min(la, lb), max(la, lb), w))
        else:
            edges.append((V[w], V[w], w))
    for j in range(8):
        lo = edges[j - 1][1] if j > 0 else -np.inf
        hi = edges[j][0] if j < 7 else np.inf
        if lo <= 0.0 <= hi:
            return [np.array([Z[j, c]]) for c in range(7)], "zone %d" % j
    for lo, hi, w in edges:
        if lo < 0.0 < hi:
            left = w < 3
            ia, ib = (w, w + 1) if left else (w + 1, w)
            sw = {0: "LF", 2: "LS", 4: "RS", 6: "RF"}[w]
            A = [np.array([Z[ia, c]]) for c in range(7)]
            Bz = [np.array([Z[ib, c]]) for c in range(7)]
            use_s = sw in ("LS", "RS")
            ys, _ = SP.fan_states(A, Bz, sw, 2 if use_s else 1, Bn1, k=4001)[:2] if False else (None, None)
            # sample along the fan in the parametrisation that reaches B
            k = 4001
            if use_s:
                tg = np.linspace(A[5][0], Bz[5][0], k)
                y, _, e = S4["fan_signed"](tg, [np.repeat(c, k) for c in A], sw, np.repeat(Bn1, k))
            else:
                tg = np.linspace(A[1][0], Bz[1][0], k)
                y, _, e = SP.FAN_P(tg, [np.repeat(c, k) for c in A], sw, np.repeat(Bn1, k))
            lam = XI(y, sw, np.repeat(Bn1, k))
            i = int(np.argmin(np.abs(lam)))
            return [np.array([y[c][i]]) for c in range(7)], "inside fan %d" % w
    return None, "none"


def flux(X, Bn):
    U, F = RI.cons_flux(X, np.array([Bn]), g)
    return F[:, 0]


def main():
    out_path = sys.argv[1]
    lanes = [int(a) for a in sys.argv[2:]]
    fin = np.load(CR.D + "/replay_gate/replay_final.npz")
    tubes = {}
    import glob
    for f in glob.glob(CR.D + "/tubes_hi/tube_*_16384.npz"):
        tubes[int(os.path.basename(f).split("_")[1])] = f
    rows = []
    for lane in lanes:
        UL, UR = pop["UL"][lane], pop["UR"][lane]
        Bn = np.array([UL[7]])
        L0, R0, alpha, _ = S5["to_planar"](one(UL[:7]), one(UR[:7]), Bn)
        Zold, Sold = ans[lane]
        ca, sa = np.cos(alpha[0]), np.sin(alpha[0])
        bt = lambda z: ca * z[5] - sa * z[6]
        u0 = [np.array([np.log(Zold[1][1])]), np.array([bt(Zold[3])]), np.array([bt(Zold[4])]),
              np.array([np.log(Zold[6][1])])]
        print("== lane %d" % lane, flush=True)
        for rot in ((True, True), (True, False), (False, True)):
            u, nrm, st, it = newton(L0, R0, u0, Bn, rot)
            tag = "rotation L %s R %s" % ("y" if rot[0] else "-", "y" if rot[1] else "-")
            if np.isfinite(nrm) and nrm <= 1e-9:
                tag += " (Bt %+.4f -> %+.4f | %+.4f -> %+.4f)" % (st["A"][5][0], st["A2"][5][0], st["D"][5][0], st["D2"][5][0])
            if not np.isfinite(nrm) or nrm > 1e-9:
                print("   %s: not converged (|f| %.1e after %d)" % (tag, nrm, it))
                continue
            Z, V = zones_of(L0, R0, st)
            UL8 = np.concatenate([Z[0, 0], [Bn[0]]])[None]
            UR8 = np.concatenate([Z[0, 7], [Bn[0]]])[None]
            VR.FAILWHY.clear()
            r = VR.verify(UL8, UR8, Z, V, g)
            kind = r["kind"][0]
            dS = r["dS"][0]
            ent = " ".join("%s:%s%s" % (VR.NAMES[w], kind[w][0], "(%+.1e)" % dS[w] if kind[w] == "disc" and w != 3 else "")
                           for w in range(7))
            Xf, where = face_state(Z[0], V[0], Bn[0], kind)
            msg = "   %s: |f| %.1e it %d  weak %s admissible %s  %s  face %s" % (
                tag, nrm, it, r["weak"][0], r["admissible"][0], ent, where)
            if Xf is not None:
                Fn = flux(Xf, Bn[0])
                # the refused answer's flux, rotated into the planar frame
                Fo = np.array([fin["F_" + k][np.flatnonzero(fin["lane"] == lane)[0]] for k in KEYS])
                rotm = np.array([1, 1, ca, ca, 1, ca, ca])     # alpha is 0 or pi on these 2D faces
                Fo_p = Fo * rotm
                sc = np.max(np.abs(Fn))
                msg += "  |F_new - F_refused| %.1e" % (np.max(np.abs(Fn - Fo_p)) / sc)
                if lane in tubes:
                    import tube_conv as TC
                    z = np.load(tubes[lane])
                    Ft = TC.tube_flux(z, 16384) * rotm
                    msg += "  vs tube16384: new %.1e refused %.1e" % (np.max(np.abs(Fn - Ft)) / sc, np.max(np.abs(Fo_p - Ft)) / sc)
            print(msg, flush=True)
            rows.append(dict(lane=lane, tag=tag, nrm=nrm, weak=bool(r["weak"][0]), adm=bool(r["admissible"][0]),
                             kind=kind, dS=dS, Z=Z[0], V=V[0], face=where))
    np.save(out_path, np.array(rows, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
