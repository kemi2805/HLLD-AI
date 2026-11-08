#!/usr/bin/env python
"""Wave-by-wave check of stored seven-wave rows (2026-11-03, planar acceptance
study).  Independent of the solvers that produced the rows: every wave is
judged from its two stored zones and its stored speed alone.

Row layout (harvest schema): zones (N, 8, 7) = R1..R8, each
[rho, P_tot, vx, vy, vz, By, Bz]; speeds (N, 7) = LF, LA, LS, CD, RS, RA, RF
(a fan's speed is its HEAD, at the ahead state); B_n = U_L[:, 7].

Per wave:
  kind     shock / fan / none (no jump) -- from the zones: a magnetosonic wave
           is a shock if it COMPRESSES (rho behind > rho ahead), which is
           family-independent (the solvers' own predicates are P_tot for the
           fast family and |B_t| for the slow one; both are reported)
  rh       the jump conditions [[F]] = s [[U]] over the seven conserved
           components, each scaled by its own flux magnitudes with a floor of
           1e-3 of the lane's largest (planar4_b's certificate; jump.rh_resid
           without a floor divides out-of-plane round-off by 1e-12)
  lax      the evolutionary (Lax) conditions of the wave's family on both
           sides, with the neighbouring families' characteristics:
             LF: lam1(R1) > s > lam1(R2),  s < lamA-(R2)
             LS: lam3(R3) > s > lam3(R4),  lamA-(R3) < s < vx(R4)
             RS: lam5(R5) > s > lam5(R6),  vx(R5) < s < lamA+(R6)
             RF: lam7(R7) > s > lam7(R8),  lamA+(R7) < s
           (lam1..lam7 = LF, LA, LS, CD, RS, RA, RF eigenvalues).  Each
           inequality's margin is reported (positive = satisfied).
  fan      for a rarefaction: head < tail in the expanding direction.
Contact: [[vx]], [[vy]], [[vz]], [[P_tot]], [[By]], [[Bz]] across R4 -> R5.
Rotations: strength of R2 -> R3 and R6 -> R7 (zero for the planar solver).
"""
import numpy as np

from rmhd.batched import jump as J
from rmhd.batched import wave_speeds_b as WB

NAMES = ("LF", "LA", "LS", "CD", "RS", "RA", "RF")


def eigs(state, Bn, gamma):
    """[LF, LS, RS, RF] (N, 4), ok (N, 4), la, ra."""
    eig, la, ra, ok = WB.xi_all(*state, Bn, gamma)
    return np.asarray(eig), np.asarray(ok), np.asarray(la), np.asarray(ra)


def st(Z, k):
    return [Z[:, k, j] for j in range(7)]


def rh_cert(ahead, behind, Vs, Bn, gamma, floor=1e-3):
    with np.errstate(all="ignore"):
        Ua, Fa, bada = J.cons_flux(ahead, Bn, gamma)
        Ub, Fb, badb = J.cons_flux(behind, Bn, gamma)
        num = np.stack([(Fb[k] - Fa[k]) - Vs * (Ub[k] - Ua[k]) for k in range(7)])
        sc = np.stack([np.abs(Fa[k]) + np.abs(Fb[k]) + np.abs(Vs) * (np.abs(Ua[k]) + np.abs(Ub[k]))
                       for k in range(7)])
        sc = np.maximum(sc, floor * sc.max(axis=0, keepdims=True))
        r = np.abs(num) / sc
    r = np.where((bada | badb)[None, :], np.inf, r)
    return r.max(axis=0), r


def strength(a, b):
    """jump of one wave: rho and P_tot relative, |dv| absolute, |dB_t| vs sqrt(2 P_tot)."""
    sP = np.sqrt(2.0 * np.maximum(np.maximum(a[1], b[1]), 1e-300))
    return np.maximum.reduce([np.abs(b[0] - a[0]) / np.abs(a[0]),
                              np.abs(b[1] - a[1]) / np.abs(a[1]),
                              np.sqrt((b[2] - a[2]) ** 2 + (b[3] - a[3]) ** 2 + (b[4] - a[4]) ** 2),
                              np.hypot(b[5] - a[5], b[6] - a[6]) / sP])


def check(UL, UR, Z, S, gamma, tiny=1e-10):
    """Per-row diagnostics dict of (N,) arrays."""
    Bn = np.ascontiguousarray(UL[:, 7])
    R = [st(Z, k) for k in range(8)]
    E = [eigs(R[k], Bn, gamma) for k in range(8)]           # per zone
    out = {}
    lam = lambda k, fam: E[k][0][:, fam]                       # fam: 0 LF 1 LS 2 RS 3 RF
    la = lambda k: E[k][2]
    ra = lambda k: E[k][3]
    okq = lambda k, fam: E[k][1][:, fam]
    # wave definitions: (name, ahead zone, behind zone, speed column, family)
    waves = (("LF", 0, 1, 0, 0), ("LS", 2, 3, 2, 1), ("RS", 5, 4, 4, 2), ("RF", 7, 6, 6, 3))
    for name, ka, kb, ks, fam in waves:
        a, b, s = R[ka], R[kb], S[:, ks]
        stren = strength(a, b)
        comp = b[0] > a[0]
        kind = np.where(stren <= tiny, 0, np.where(comp, 1, 2))   # 0 none, 1 shock, 2 fan
        out[name + "_kind"] = kind
        out[name + "_strength"] = stren
        out[name + "_dP"] = (b[1] - a[1]) / a[1]
        out[name + "_dBt"] = np.hypot(b[5], b[6]) / np.maximum(np.hypot(a[5], a[6]), 1e-300)
        out[name + "_drho"] = b[0] / a[0]
        rh, rv = rh_cert(a, b, s, Bn, gamma)
        out[name + "_rh"] = rh
        out[name + "_rh_comp"] = np.argmax(rv, axis=0)
        # Lax margins (positive = satisfied), written for left waves with
        # U_l = ahead, U_r = behind and right waves with U_l = behind, U_r = ahead
        if name == "LF":
            m = [lam(ka, 0) - s, s - lam(kb, 0), la(kb) - s]
            tags = ["lam1(ahead)-s", "s-lam1(behind)", "lamA-(behind)-s"]
            okm = okq(ka, 0) & okq(kb, 0)
        elif name == "LS":
            m = [lam(ka, 1) - s, s - lam(kb, 1), s - la(ka), b[2] - s]
            tags = ["lam3(ahead)-s", "s-lam3(behind)", "s-lamA-(ahead)", "vx(behind)-s"]
            okm = okq(ka, 1) & okq(kb, 1)
        elif name == "RS":
            m = [lam(kb, 2) - s, s - lam(ka, 2), s - b[2], ra(ka) - s]
            tags = ["lam5(behind)-s", "s-lam5(ahead)", "s-vx(behind)", "lamA+(ahead)-s"]
            okm = okq(ka, 2) & okq(kb, 2)
        else:
            m = [lam(kb, 3) - s, s - lam(ka, 3), s - ra(kb)]
            tags = ["lam7(behind)-s", "s-lam7(ahead)", "s-lamA+(behind)"]
            okm = okq(ka, 3) & okq(kb, 3)
        M = np.stack(m)
        out[name + "_lax_margins"] = M
        out[name + "_lax_tags"] = tags
        out[name + "_lax_min"] = M.min(axis=0)
        out[name + "_quartic_ok"] = okm
        # fan: head (ahead) and tail (behind) speeds of the family; a left
        # fan expands if head < tail, a right one if tail < head
        fam_head, fam_tail = lam(ka, fam), lam(kb, fam)
        out[name + "_fan_head"] = fam_head
        out[name + "_fan_tail"] = fam_tail
        out[name + "_fan_expands"] = (fam_head <= fam_tail) if name in ("LF", "LS") else (fam_tail <= fam_head)
    # speeds the zones imply
    out["lam"] = np.stack([lam(0, 0), la(1), lam(2, 1), 0.5 * (R[3][2] + R[4][2]),
                           lam(5, 2), ra(6), lam(7, 3)], axis=1)
    # rotations and contact
    out["LA_strength"] = strength(R[1], R[2])
    out["RA_strength"] = strength(R[5], R[6])
    c = []
    sP = np.sqrt(2.0 * np.maximum(R[3][1], 1e-300))
    for j, sc in ((2, 1.0), (3, 1.0), (4, 1.0), (1, R[3][1]), (5, sP), (6, sP)):
        c.append(np.abs(R[3][j] - R[4][j]) / sc)
    out["CD_jump"] = np.stack(c)        # vx vy vz P(rel) By Bz (vs sqrt 2P)
    out["Bn"] = Bn
    out["Bt"] = np.stack([np.hypot(R[k][5], R[k][6]) for k in range(8)], axis=1)
    out["psi"] = np.stack([np.arctan2(R[k][6], R[k][5]) for k in range(8)], axis=1)
    return out


def summary_line(d, i):
    s = []
    for name in ("LF", "LS", "RS", "RF"):
        kind = ("-", "shock", "fan")[int(d[name + "_kind"][i])]
        s.append("%s %s str %.2e rh %.1e laxmin %+.2e" % (name, kind, d[name + "_strength"][i],
                                                         d[name + "_rh"][i], d[name + "_lax_min"][i]))
    return " | ".join(s)


def pgas(state, Bn):
    rho, P, vx, vy, vz, By, Bz = state
    v2 = vx * vx + vy * vy + vz * vz
    eta = Bn * vx + By * vy + Bz * vz
    b2 = (Bn * Bn + By * By + Bz * Bz) * (1.0 - v2) + eta * eta
    return P - 0.5 * b2


def admissibility(UL, UR, Z, S, gamma):
    """Per wave, by the SOLVERS' own shock/fan predicates (fast: P_tot rises;
    slow: |B_t| falls): entropy ratio K_b/K_a (K = p_gas / rho^gamma; >= 1 for
    an admissible shock, == 1 for a fan), density ratio, the RH residual at the
    stored speed, and the Lax margins (shocks).  Returns dict of (N,) arrays."""
    d = check(UL, UR, Z, S, gamma)
    Bn = d["Bn"]
    R = [st(Z, k) for k in range(8)]
    K = [pgas(R[k], Bn) / R[k][0] ** gamma for k in range(8)]
    out = dict(d)
    for name, ka, kb, fast in (("LF", 0, 1, True), ("LS", 2, 3, False), ("RS", 5, 4, False), ("RF", 7, 6, True)):
        a, b = R[ka], R[kb]
        if fast:
            sh = b[1] > a[1] * (1 + 1e-14)
        else:
            sh = np.hypot(b[5], b[6]) < np.hypot(a[5], a[6]) * (1 - 1e-12)
        nz = d[name + "_strength"] > 1e-10
        out[name + "_solver_kind"] = np.where(~nz, 0, np.where(sh, 1, 2))
        out[name + "_K"] = K[kb] / K[ka]
    return out
