#!/usr/bin/env python
"""Every attempted face of an exact 2D run: where, when, what answered it, and
the physics of its two states and of its answer (2026-11-04; user: "an
analysis on what faces succeed and fail, and why they fail.  And what faces
use the 5 wave solver and why.  Be ambitious").

    python face_table.py RUN_DIR OUT_DIR [--procs 32]

The harvest's rows carry no grid position, but every shard holds whole
exact_flux calls and its coverage bitmaps say, per sweep and per face, what
answered it.  Within one sweep the rows come in face order:
  solved   popcount(seven) seven-wave answers (source 0), then
           popcount(planar5) planar answers (source 1)
  unsolved one per face the seven-wave path did not accept:
           attempted & ~seven & ~degenerate (degenerate-class lanes the
           three-wave solver answered are accepted and not written)
(checked on three shards 2026-11-04; checked again here per shard: the row
counts, the source sequence, and that a planar answer's states equal its
unsolved twin's bit for bit).  So every row gets its sweep, face, time and
final outcome, and every face its states.

Per face (all float32 unless said):
  where/when  sweep, idir, face (int), step, stage, t, x, y, phys (bool),
              xi = x / t, eta = y / t
  outcome     1 seven-wave, 2 planar five-wave, 3 four-unknown,
              4 continuation, 5 three-wave, 6 linear fallback, 8 lost to HLLD
              (int8); attempts (seven-wave attempts; >1 = a retry answered),
              reason7 (the seven-wave path's refusal, harvest codes 0 not
              converged, 1 ray, 2 unphysical, 3 self-crossing; -1 accepted)
  states      U_L, U_R (float64, solver frame [rho, P_tot, vx, vy, vz, By,
              Bz, Bx]) and their physics: rho, gas p, W, normal and
              tangential velocity, |B|, normal-field fraction, sigma, beta,
              the characteristic speeds, per side; across the face the
              production jump (exact_flux.relative_jump), shear, compression,
              field rotation and reversal, the compound-wave predictor
              (min |Bt| < 0.3 |Bn| and rotation > pi/2, docs 5)
  answer      (seven-wave and planar faces) per magnetosonic wave its kind
              (0 none, 1 shock, 2 fan) and strength; the rotations; the
              contact's tangential field against |B|; the smallest tangential
              field of the fan; whether a slow wave turns the field through
              zero; each slow wave's distance to the Alfven speed ahead; the
              admissibility flags (entropy fall > 1e-6, slow Lax < -1e-4)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "planar_audit"))

OUT_CODES = dict(seven=1, planar5=2, planar4=3, continuation=4, three_wave=5,
                 linear_fallback=6, lost=8)


def bits(b, n):
    return np.unpackbits(b)[:n].astype(bool)


def sweep_times(run):
    import csv
    rows = list(csv.DictReader(open(os.path.join(run, "diag.csv"))))
    t_after = np.array([float(r["t"]) for r in rows])
    dt = np.array([float(r["dt"]) for r in rows])
    return t_after, dt


def state_feats(U, gamma, WB):
    rho, P, vx, vy, vz, By, Bz, Bn = (U[:, k] for k in range(8))
    v2 = vx * vx + vy * vy + vz * vz
    W = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1e-300))
    vB = Bn * vx + By * vy + Bz * vz
    B2 = Bn * Bn + By * By + Bz * Bz
    b2 = B2 / (W * W) + vB * vB
    p = P - 0.5 * b2
    h = 1.0 + gamma / (gamma - 1.0) * p / rho
    with np.errstate(all="ignore"):
        eig, la, ra, ok = WB.xi_all(rho, P, vx, vy, vz, By, Bz, Bn, gamma)
    eig = np.asarray(eig)
    return dict(rho=rho, p=p, W=W, vn=vx, B=np.sqrt(B2), Bt=np.hypot(By, Bz),
                fn=np.abs(Bn) / np.sqrt(np.maximum(B2, 1e-300)),
                sigma=b2 / (rho * h), beta=2.0 * p / np.maximum(b2, 1e-300),
                lf=eig[:, 0], ls=eig[:, 1], rs=eig[:, 2], rf=eig[:, 3],
                la=np.asarray(la), ra=np.asarray(ra))


def face_feats(UL, UR, gamma, WB):
    L, R = state_feats(UL, gamma, WB), state_feats(UR, gamma, WB)
    # the plane: at most one solver-frame tangential component is non-zero
    # (docs 2), so take the dominant one as the signed tangential axis
    use_z = (np.abs(UL[:, 6]) + np.abs(UR[:, 6]) + np.abs(UL[:, 4]) + np.abs(UR[:, 4])
             > np.abs(UL[:, 5]) + np.abs(UR[:, 5]) + np.abs(UL[:, 3]) + np.abs(UR[:, 3]))
    vtL = np.where(use_z, UL[:, 4], UL[:, 3]); vtR = np.where(use_z, UR[:, 4], UR[:, 3])
    btL = np.where(use_z, UL[:, 6], UL[:, 5]); btR = np.where(use_z, UR[:, 6], UR[:, 5])
    Bn = UL[:, 7]
    psiL, psiR = np.arctan2(UL[:, 6], UL[:, 5]), np.arctan2(UR[:, 6], UR[:, 5])
    rot = np.abs(np.angle(np.exp(1j * (psiR - psiL))))
    # exact_flux.relative_jump, on the solver-frame vectors (rotation-invariant)
    tiny = 1e-30
    jump = np.maximum.reduce([
        np.abs(UL[:, 0] - UR[:, 0]) / (np.abs(UL[:, 0]) + np.abs(UR[:, 0]) + tiny),
        np.abs(L["p"] - R["p"]) / (np.abs(L["p"]) + np.abs(R["p"]) + tiny),
        np.sqrt(((UL[:, 2:5] - UR[:, 2:5]) ** 2).sum(axis=1)),
        np.sqrt(((UL[:, 5:7] - UR[:, 5:7]) ** 2).sum(axis=1))
        / (L["B"] + R["B"] + np.sqrt(np.abs(L["p"]) + np.abs(R["p"])) + tiny)])
    minBt = np.minimum(L["Bt"], R["Bt"])
    f = dict(
        jump=jump,
        log_rho_ratio=np.log10(UR[:, 0] / UL[:, 0]),
        log_p_ratio=np.log10(np.abs(R["p"]) / np.maximum(np.abs(L["p"]), 1e-300)),
        dvn=UR[:, 2] - UL[:, 2], shear=vtR - vtL,
        Wt=np.maximum(1 / np.sqrt(np.maximum(1 - vtL ** 2, 1e-300)),
                      1 / np.sqrt(np.maximum(1 - vtR ** 2, 1e-300))),
        Bn=Bn, btL=btL, btR=btR, rot=rot,
        reversal=(btL * btR < 0) & (minBt > 1e-12),
        compound_like=(minBt < 0.3 * np.abs(Bn)) & (rot > np.pi / 2),
        minBt_over_Bn=minBt / np.maximum(np.abs(Bn), 1e-300),
        vtL=vtL, vtR=vtR)
    for side, S in (("L", L), ("R", R)):
        for k in ("rho", "p", "W", "vn", "B", "Bt", "fn", "sigma", "beta"):
            f[k + side] = S[k]
    # the slow-Alfven gaps of the waves that leave each side: left-going from
    # the left state (lambda_2 - lambda_3 <= 0, |.|), right-going from the right
    f["gapAS_L"] = np.abs(L["ls"] - L["la"])
    f["gapAS_R"] = np.abs(R["ra"] - R["rs"])
    f["gapFA_L"] = np.abs(L["la"] - L["lf"])
    f["gapFA_R"] = np.abs(R["rf"] - R["ra"])
    f["sigma_max"] = np.maximum(L["sigma"], R["sigma"])
    f["beta_min"] = np.minimum(L["beta"], R["beta"])
    f["W_max"] = np.maximum(L["W"], R["W"])
    f["fn_min"] = np.minimum(L["fn"], R["fn"])
    f["fn_max"] = np.maximum(L["fn"], R["fn"])
    return f


def answer_feats(UL, UR, Z, S, gamma, WC):
    d = WC.check(UL, UR, Z, S, gamma)
    a = WC.admissibility(UL, UR, Z, S, gamma)
    f = {}
    for w in ("LF", "LS", "RS", "RF"):
        f[w + "_kind"] = d[w + "_kind"]
        f[w + "_str"] = d[w + "_strength"]
        f[w + "_laxmin"] = d[w + "_lax_min"]
    f["LA_str"], f["RA_str"] = d["LA_strength"], d["RA_strength"]
    Bt = d["Bt"]                                     # (N, 8) zones R1..R8
    Bmag = np.sqrt(Bt ** 2 + d["Bn"][:, None] ** 2)
    f["cd_Bt_rel"] = np.minimum(Bt[:, 3], Bt[:, 4]) / np.maximum(np.minimum(Bmag[:, 3], Bmag[:, 4]), 1e-300)
    f["fan_minBt_rel"] = (Bt[:, 1:7] / np.maximum(Bmag[:, 1:7], 1e-300)).min(axis=1)
    psi = d["psi"]
    dpsi = lambda i, j: np.abs(np.angle(np.exp(1j * (psi[:, j] - psi[:, i]))))
    tinyBt = lambda i, j: np.minimum(Bt[:, i], Bt[:, j]) > 1e-12
    f["LA_rot"] = np.where(tinyBt(1, 2), dpsi(1, 2), 0.0)
    f["RA_rot"] = np.where(tinyBt(5, 6), dpsi(5, 6), 0.0)
    # a slow wave that turns the field through zero (switch-off then on)
    f["LS_flips"] = (dpsi(2, 3) > np.pi / 2) & tinyBt(2, 3)
    f["RS_flips"] = (dpsi(4, 5) > np.pi / 2) & tinyBt(4, 5)
    # how close each slow wave runs to the Alfven speed ahead of it
    mLS, mRS = d["LS_lax_margins"], d["RS_lax_margins"]
    f["LS_alfven_margin"] = mLS[2]             # s - lambda_A-(ahead)
    f["RS_alfven_margin"] = mRS[3]             # lambda_A+(ahead) - s
    ent = np.zeros(UL.shape[0], bool)
    lax = np.zeros(UL.shape[0], bool)
    for w in ("LF", "LS", "RS", "RF"):
        sh = a[w + "_solver_kind"] == 1
        ent |= sh & (a[w + "_K"] < 1 - 1e-6)
        if w in ("LS", "RS"):
            lax |= sh & (a[w + "_lax_min"] < -1e-4)
    f["entropy_falls"], f["slow_lax_broken"] = ent, lax
    f["rh_max"] = np.maximum.reduce([d[w + "_rh"] for w in ("LF", "LS", "RS", "RF")])
    return f


def one_shard(args):
    run, out, k, ng, nx, ny, t_after, dt = args
    import wavecheck as WC
    from rmhd.batched import wave_speeds_b as WB
    c = np.load(os.path.join(run, "harvest", "coverage_r000_%05d.npz" % k))
    s = np.load(os.path.join(run, "harvest", "solved_r000_%05d.npz" % k))
    u = np.load(os.path.join(run, "harvest", "unsolved_r000_%05d.npz" % k))
    gamma = float(s["gamma"])
    nxt, nyt = nx + 2 * ng, ny + 2 * ng
    i7 = i5 = iu = 0
    rec = {k_: [] for k_ in ("sweep", "idir", "face", "outcome", "row_s", "row_u", "row_5")}
    bad = []
    for r in range(c["sweep"].shape[0]):
        n = int(c["n"][r])
        m = {key: bits(c[key][r], n) for key in ("attempted", "exact", "seven", "planar5", "planar4",
                                                 "continuation", "three_wave", "degenerate",
                                                 "linear_fallback")}
        f7 = np.flatnonzero(m["seven"])
        f5 = np.flatnonzero(m["planar5"])
        fu = np.flatnonzero(m["attempted"] & ~m["seven"] & ~m["degenerate"])
        # rows of this sweep: its seven-wave answers, then its planar answers
        rs7 = np.arange(i7, i7 + f7.size)
        rs5 = np.arange(i7 + f7.size, i7 + f7.size + f5.size)
        i7 += f7.size + f5.size
        ru = np.arange(iu, iu + fu.size); iu += fu.size
        sw, idr = int(c["sweep"][r]), int(c["idir"][r])
        # seven-wave faces
        rec["sweep"].append(np.full(f7.size, sw)); rec["idir"].append(np.full(f7.size, idr))
        rec["face"].append(f7); rec["outcome"].append(np.full(f7.size, 1, np.int8))
        rec["row_s"].append(rs7); rec["row_u"].append(np.full(f7.size, -1))
        rec["row_5"].append(np.full(f7.size, -1))
        # every face the seven-wave path refused: its unsolved row, its final outcome
        oc = np.full(fu.size, 8, np.int8)
        for key in ("linear_fallback", "three_wave", "continuation", "planar4", "planar5"):
            oc[m[key][fu]] = OUT_CODES[key]
        row5 = np.full(fu.size, -1)
        pos5 = np.searchsorted(fu, f5)
        if f5.size and not (fu[np.minimum(pos5, fu.size - 1)] == f5).all():
            bad.append("sweep %d: planar5 faces not in the unsolved set" % sw)
        row5[pos5] = rs5
        rec["sweep"].append(np.full(fu.size, sw)); rec["idir"].append(np.full(fu.size, idr))
        rec["face"].append(fu); rec["outcome"].append(oc)
        rec["row_s"].append(np.full(fu.size, -1)); rec["row_u"].append(ru); rec["row_5"].append(row5)
    src = s["source"]
    if i7 != src.size:
        bad.append("solved rows: %d vs %d" % (i7, src.size))
        return dict(shard=k, bad=bad)
    R = {k_: np.concatenate(v) for k_, v in rec.items()}
    seven = R["outcome"] == 1
    r5 = R["row_5"][R["row_5"] >= 0]
    if (src[R["row_s"][seven]] != 0).any() or (src[r5] != 1).any():
        bad.append("source sequence")
    nF = R["face"].size
    UL = np.full((nF, 8), np.nan); UR = np.full((nF, 8), np.nan)
    UL[seven] = s["U_L"][R["row_s"][seven]]; UR[seven] = s["U_R"][R["row_s"][seven]]
    has5 = R["row_5"] >= 0
    UL[has5] = s["U_L"][R["row_5"][has5]]; UR[has5] = s["U_R"][R["row_5"][has5]]
    # The refused faces, in shard order, against the unsolved rows.  A lane
    # the seven-wave path accepted and exact_flux then refused (a degenerate
    # answer of a reclassified lane, reason7 5) was accepted at harvest time:
    # it has no unsolved row, so the rows lag the faces by the number of such
    # lanes before them.  The planar answers are anchors: a planar face's
    # unsolved twin carries its states bit for bit.
    ref = np.flatnonzero(~seven)
    nU = u["reason"].size
    E = ref.size - nU
    row_u = np.full(nF, -1)
    aligned = np.zeros(nF, bool)
    aligned[seven] = True
    if E < 0:
        bad.append("more unsolved rows (%d) than refused faces (%d)" % (nU, ref.size))
        return dict(shard=k, bad=bad)
    keyu = np.ascontiguousarray(np.concatenate([u["U_L"], u["U_R"]], axis=1)).view(np.dtype((np.void, 128))).ravel()
    where = {}
    for j, kk in enumerate(keyu):
        where.setdefault(kk.tobytes(), []).append(j)
    anchors = []                     # (position in ref, row) of matched planar faces
    extra = np.zeros(ref.size, bool)
    last = -1
    for q in np.flatnonzero(has5[ref]):
        kk = np.ascontiguousarray(np.concatenate([UL[ref[q]], UR[ref[q]]])).view(np.dtype((np.void, 128)))[0].tobytes()
        cand = [j for j in where.get(kk, []) if last < j <= q and j >= q - E]
        if cand:
            anchors.append((q, cand[-1])); last = cand[-1]
        else:
            extra[q] = True          # a planar face with no unsolved row: one of the E
    n_amb = 0
    for (q0, j0), (q1, j1) in zip([(-1, -1)] + anchors, anchors + [(ref.size, nU)]):
        between = np.arange(q0 + 1, q1)
        free = between[~extra[between]]
        nr = j1 - j0 - 1
        if q0 >= 0:
            row_u[ref[q0]] = j0; aligned[ref[q0]] = True
        if free.size == nr:
            row_u[ref[free]] = np.arange(j0 + 1, j1); aligned[ref[free]] = True
        else:
            n_amb += free.size
    aligned[ref[extra]] = True       # planar faces without a row: states from their planar answer
    ok_u = row_u >= 0
    UL[ok_u] = u["U_L"][row_u[ok_u]]; UR[ok_u] = u["U_R"][row_u[ok_u]]
    twin = has5 & ok_u
    twin_ok = int(((s["U_L"][R["row_5"][twin]] == UL[twin]).all(axis=1)
                   & (s["U_R"][R["row_5"][twin]] == UR[twin]).all(axis=1)).sum())
    if twin_ok != int(twin.sum()):
        bad.append("planar twins: %d of %d bitwise" % (twin_ok, int(twin.sum())))
    uu = ok_u
    # answers: seven-wave rows and planar rows
    Z = np.full((nF, 8, 7), np.nan); S = np.full((nF, 7), np.nan)
    Z[seven] = s["zones"][R["row_s"][seven]]; S[seven] = s["speeds"][R["row_s"][seven]]
    Z[has5] = s["zones"][R["row_5"][has5]]; S[has5] = s["speeds"][R["row_5"][has5]]
    attempts = np.full(nF, -1, np.int16); reason7 = np.full(nF, -1, np.int8); cls = np.full(nF, -1, np.int8)
    attempts[seven] = s["attempts"][R["row_s"][seven]]
    attempts[uu] = u["attempts"][row_u[uu]]
    reason7[uu] = u["reason"][row_u[uu]]
    cls[uu] = u["cls"][row_u[uu]]
    # where and when
    sw = R["sweep"]; step = sw // 6; stage = (sw % 6) // 2
    t = t_after[step] - dt[step] + np.array([0.0, 1.0, 0.5])[stage] * dt[step]
    dx = 1.0 / nx
    f_ = R["face"]
    x = np.where(R["idir"] == 0, -0.5 + (f_ // nyt - ng) * dx, -0.5 + (f_ // (nyt + 1) - ng + 0.5) * dx)
    y = np.where(R["idir"] == 0, -0.5 + (f_ % nyt - ng + 0.5) * dx, -0.5 + (f_ % (nyt + 1) - ng) * dx)
    phys = (x >= -0.5 - 1e-12) & (x <= 0.5 + 1e-12) & (y >= -0.5 - 1e-12) & (y <= 0.5 + 1e-12)
    keep = aligned                   # faces whose states are known
    F = face_feats(np.where(keep[:, None], UL, 1.0), np.where(keep[:, None], UR, 1.0), gamma, WB)
    A = {}
    have = seven | has5
    if have.any():
        a = answer_feats(UL[have], UR[have], Z[have], S[have], gamma, WC)
        for k_, v in a.items():
            arr = np.full(nF, np.nan, np.float32) if v.dtype != bool else np.zeros(nF, bool)
            arr[have] = v
            A[k_] = arr
    out_d = dict(sweep=sw.astype(np.int32), idir=R["idir"].astype(np.int8), face=f_.astype(np.int32),
                 step=step.astype(np.int32), stage=stage.astype(np.int8), t=t.astype(np.float32),
                 x=x.astype(np.float32), y=y.astype(np.float32), phys=phys,
                 outcome=R["outcome"], attempts=attempts, reason7=reason7, cls=cls, aligned=aligned,
                 U_L=UL, U_R=UR, gamma=gamma)
    for k_, v in list(F.items()) + list(A.items()):
        out_d["f_" + k_] = v if v.dtype == bool else v.astype(np.float32)
    np.savez_compressed(os.path.join(out, "faces_%05d.npz" % k), **out_d)
    return dict(shard=k, bad=bad, faces=nF, twins=int(has5.sum()), extras=int(E),
                ambiguous=int(n_amb), unaligned=int((~aligned).sum()),
                counts={str(o): int((R["outcome"] == o).sum()) for o in np.unique(R["outcome"])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--shards", default=None, help="a:b, for a test")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.run, "run_meta.json")))
    ng, nx = int(meta.get("ng", 2)), int(meta["n"])
    t_after, dt = sweep_times(a.run)
    ks = sorted(int(os.path.basename(p)[-9:-4]) for p in glob.glob(os.path.join(a.run, "harvest", "coverage_r000_*.npz")))
    if a.shards:
        lo, hi = (int(v) for v in a.shards.split(":"))
        ks = [k for k in ks if lo <= k < hi]
    jobs = [(a.run, a.out, k, ng, nx, nx, t_after, dt) for k in ks]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(one_shard, jobs, chunksize=1)
    bad = [r for r in res if r["bad"]]
    tot = {}
    for r in res:
        for o, n in r.get("counts", {}).items():
            tot[o] = tot.get(o, 0) + n
    rep = dict(shards=len(res), bad=bad, outcome_counts=tot, faces=sum(r.get("faces", 0) for r in res),
               extras=sum(r.get("extras", 0) for r in res), unaligned=sum(r.get("unaligned", 0) for r in res),
               planar_twins=sum(r.get("twins", 0) for r in res))
    json.dump(rep, open(os.path.join(a.out, "face_table.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1)[:3000])


if __name__ == "__main__":
    main()
