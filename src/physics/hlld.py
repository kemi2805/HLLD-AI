"""
HLLD Riemann solver for SR-MHD in flat Minkowski spacetime.

All operations use PyTorch tensors so they run on MPS / CUDA / CPU.
Primitive state dicts hold individual component tensors — either scalars
(single state) or 1-D tensors of length N (batched).

Primitive variables  (Prim):
    rho, vx, vy, vz, p, eps, Bx, By, Bz

Conserved variables  (Cons):
    D, Sx, Sy, Sz, tau, Bx, By, Bz
"""

import torch
from typing import Dict, Tuple

from .eos import hybrid_eos
from .c2p import conservative_to_primitive


# ── type aliases ─────────────────────────────────────────────────────────────
Prim = Dict[str, torch.Tensor]
Cons = Dict[str, torch.Tensor]

# Smallest safe denominator magnitude (preserves sign, avoids NaN / Inf from 0-div)
_TINY = 1e-300


# ── helpers ──────────────────────────────────────────────────────────────────

def _t(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x)

def _delta(i: int, j: int) -> float:
    return 1.0 if i == j else 0.0


def _sdiv(num: torch.Tensor, den: torch.Tensor,
          tiny: float = _TINY) -> torch.Tensor:
    """
    Safe division.  Where |den| < tiny we return num / (±tiny),
    preserving sign — giving a large-but-finite value instead of NaN.
    """
    sgn = torch.sign(den)
    # treat exact zero as positive
    sgn = torch.where(sgn == 0.0, torch.ones_like(sgn), sgn)
    safe = torch.where(den.abs() < tiny, sgn * tiny, den)
    return num / safe


def lorentz(prim: Prim) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lorentz factor W and v² from a primitive state dict."""
    v2 = prim["vx"] ** 2 + prim["vy"] ** 2 + prim["vz"] ** 2
    # Clamp away from 1 to prevent sqrt(0) or sqrt(<0) after rounding
    v2_safe = torch.clamp(v2, max=1.0 - 1e-10)
    return 1.0 / torch.sqrt(1.0 - v2_safe), v2


def compute_smallb(v, B, W):
    """
    Comoving magnetic field 4-vector b^mu (Minkowski, alpha=1, beta=0).

        b^t = W (v · B)
        b^i = (B^i + b^t v^i) / W

    Returns (b^t, b^x, b^y, b^z).
    """
    vx, vy, vz = v
    Bx, By, Bz = B
    vdotB = vx * Bx + vy * By + vz * Bz
    bt = W * vdotB
    bx = (Bx / W + vdotB * vx * W)
    by = (By / W + vdotB * vy * W)
    bz = (Bz / W + vdotB * vz * W)
    return bt, bx, by, bz


def compute_b2(v, B, W) -> torch.Tensor:
    """b² = (B² / W² + (v·B)²) ."""
    vx, vy, vz = v
    Bx, By, Bz = B
    B2 = Bx ** 2 + By ** 2 + Bz ** 2
    vdotB = vx * Bx + vy * By + vz * Bz
    return (B2 / (W ** 2) + vdotB ** 2)  #Marie


def compute_v02_and_h(cs2, b2, rho, eps, p):
    """Fast magnetosonic speed² and specific enthalpy h."""
    h = 1.0 + eps + p / rho
    v02 = cs2 + (b2 / (rho * h + b2)) * (1.0 - cs2)
    return v02, h


def compute_cp_cm(v02, u0, vd, one_over_alp2=1.0, betad=0.0, gupdd=1.0):
    """Characteristic speeds c⁺, c⁻ from the dispersion relation."""
    u0_sq = u0 ** 2
    a = u0_sq * (1.0 - v02) + v02 * one_over_alp2
    b = 2.0 * (betad * one_over_alp2 * v02 - u0_sq * vd * (1.0 - v02))
    c = u0_sq * vd ** 2 * (1.0 - v02) - v02 * (gupdd - betad ** 2 * one_over_alp2)
    det = torch.sqrt(torch.clamp(b ** 2 - 4.0 * a * c, min=0.0))
    # Guard against a == 0 (degenerate non-relativistic limit)
    a_safe = torch.clamp(a, min=_TINY)
    c1 =  0.5 * (det - b) / a_safe
    c2 = -0.5 * (det + b) / a_safe
    cp = torch.max(c1, c2)
    cm = torch.min(c1, c2)
    return cp, cm

def compute_cp_cm_new(cs2, vtilde, b2, W, eps, rho, press): #Marie
    """
    Characteristic fast magnetosonic speeds for SRMHD.
    (flat spacetime: alp=1, beta=0, guu=1)
    """
    rh       = rho * (1.0 + eps) + press        # enthalpy density ρh
    total_e  = b2 + rh                           # b² + ρh

    alfven   =  b2  / total_e                    # b²/(b²+ρh)
    neg_hydro = alfven - 1.0                     # -ρh/(b²+ρh)
    sound    = -cs2 * neg_hydro                  # cs²·ρh/(b²+ρh)

    fast_num =  alfven + sound                   # x5
    mix      =  neg_hydro + sound                # x7
    fast_den =  W**2 * mix                       # x8

    # flat space: alp=1, beta=0, guu=1
    # quadratic: A*λ² + B*λ + C = 0
    A =  -(fast_num - fast_den)
    B =  -vtilde * fast_den                      # beta=0 kills first term
    C =   vtilde**2 * fast_den - fast_num        # guu=1, beta=0

    disc_raw = B**2 - A*C
    discriminant = torch.sqrt(torch.clamp(disc_raw, min=0.0))

    inv_A = 2.0 / A
    cp =  (-B + discriminant) * inv_A
    cm =  (-B - discriminant) * inv_A

    return torch.max(cp, cm), torch.min(cp, cm)


# ── HLL helpers ──────────────────────────────────────────────────────────────

def hlle_flux(fL, fR, uL, uR, cmin, cmax):
    """HLL numerical flux (scalar or dict values)."""
    return (cmax * fL + cmin * fR - cmax * cmin * (uR - uL)) / (cmax + cmin)


def hlle_state(fL, fR, uL, uR, cmin, cmax):
    """HLL intermediate state."""
    return (cmax * uR + cmin * uL + fL - fR) / (cmax + cmin)


def _hf(fL, fR, uL, uR, cm, cx):
    """Per-key HLL flux."""
    return (cx * fL + cm * fR - cx * cm * (uR - uL)) / (cx + cm)


def _hu(fL, fR, uL, uR, cm, cx):
    """Per-key HLL state."""
    return (cx * uR + cm * uL + fL - fR) / (cx + cm)


# ── primitive / conserved conversion ─────────────────────────────────────────

def primitive_state(rho, vx, vy, vz, p, eps, Bx, By, Bz) -> Prim:
    """Pack primitive variables into a typed dict."""
    return {k: _t(v) for k, v in zip(
        ["rho", "vx", "vy", "vz", "p", "eps", "Bx", "By", "Bz"],
        [rho, vx, vy, vz, p, eps, Bx, By, Bz])}


def primitive_to_conserved(prim: Prim) -> Cons:
    """
    Convert primitive → conserved for SR-MHD (flat Minkowski).

        D   = rho * W
        S_i = (rho*h*W² + B²) v_i − (v·B) B_i
        tau = rho*h*W² − p − D + ½(B² + B²v² − (v·B)²)
    """
    rho = prim["rho"]
    vx, vy, vz = prim["vx"], prim["vy"], prim["vz"]
    p = prim["p"]; eps = prim["eps"]
    Bx, By, Bz = prim["Bx"], prim["By"], prim["Bz"]

    v2 = vx ** 2 + vy ** 2 + vz ** 2
    W = 1.0 / torch.sqrt(torch.clamp(1.0 - v2, min=1e-10))
    vB = vx * Bx + vy * By + vz * Bz
    B2 = Bx ** 2 + By ** 2 + Bz ** 2

    h = 1.0 + eps + p / rho
    rHW2 = rho * h * W ** 2
    D = rho * W

    return {
        "D": D,
        "Sx": (rHW2 + B2) * vx - vB * Bx,
        "Sy": (rHW2 + B2) * vy - vB * By,
        "Sz": (rHW2 + B2) * vz - vB * Bz,
        "tau": rHW2 - p - D + 0.5 * (B2 + B2 * v2 - vB ** 2),
        "Bx": Bx, "By": By, "Bz": Bz,
    }


# ── physical fluxes ──────────────────────────────────────────────────────────

def compute_srmhd_fluxes(
    primL: Prim, primR: Prim, eos: hybrid_eos, idir: int = 0
) -> Tuple[Cons, Cons, Cons, Cons, torch.Tensor, torch.Tensor]:
    """
    SR-MHD conserved states and fluxes on both sides of an interface
    (flat Minkowski: alpha=1, beta=0).

    Returns  uL, uR, fL, fR, cmax, cmin.
    """
    vdir = ["vx", "vy", "vz"][idir]

    wl, _ = lorentz(primL)
    wr, _ = lorentz(primR)

    pl, cs2l = eos.press_and_cs2(primL["eps"], primL["rho"])
    pr, cs2r = eos.press_and_cs2(primR["eps"], primR["rho"])

    vl = (primL["vx"], primL["vy"], primL["vz"])
    vr = (primR["vx"], primR["vy"], primR["vz"])
    Bl = (primL["Bx"], primL["By"], primL["Bz"])
    Br = (primR["Bx"], primR["By"], primR["Bz"])

    b2l = compute_b2(vl, Bl, wl)
    b2r = compute_b2(vr, Br, wr)

    sbL = compute_smallb(vl, Bl, wl)
    sbR = compute_smallb(vr, Br, wr)

    def lower_b(sb):
        bt, bx, by, bz = sb
        return -bt, bx, by, bz

    sbDL = lower_b(sbL)
    sbDR = lower_b(sbR)

    v02l, hl = compute_v02_and_h(cs2l, b2l, primL["rho"], primL["eps"], pl)
    v02r, hr = compute_v02_and_h(cs2r, b2r, primR["rho"], primR["eps"], pr)

    #cp_l, cm_l = compute_cp_cm(cs2l, primL[vdir], b2l, wl, primL["eps"], primL["rho"], pl)
    #cp_r, cm_r = compute_cp_cm(cs2r, primR[vdir], b2r, wr, primR["eps"], primR["rho"], pr)

    cp_l, cm_l = compute_cp_cm(v02l, wl, primL[vdir])
    cp_r, cm_r = compute_cp_cm(v02r, wr, primR[vdir])

    cmax = torch.max(cp_l, cp_r)
    cmin = torch.max(-cm_l, -cm_r)

    cmax = torch.clamp(torch.max(cp_l, cp_r), min=0.0) # Marie hotfix
    cmin = torch.clamp(torch.max(-cm_l, -cm_r), min=0.0)


    both_small = (cmax < 1e-12) & (cmin < 1e-12)

    cmax = torch.where(both_small, torch.ones_like(cmax), cmax)
    cmin = torch.where(both_small, torch.ones_like(cmin), cmin)

    dl = primL["rho"] * wl
    dr = primR["rho"] * wr
    uDl = (wl * primL["vx"], wl * primL["vy"], wl * primL["vz"])
    uDr = (wr * primR["vx"], wr * primR["vy"], wr * primR["vz"])

    rhb2l = primL["rho"] * hl + b2l
    rhb2r = primR["rho"] * hr + b2r
    Pb2l = pl + 0.5 * b2l
    Pb2r = pr + 0.5 * b2r

    vdl = primL[vdir]
    vdr = primR[vdir]
    bdirl = sbL[1 + idir]
    bdirr = sbR[1 + idir]

    uL: Cons = {}; uR: Cons = {}
    fL: Cons = {}; fR: Cons = {}

    # D
    uL["D"] = dl; uR["D"] = dr
    fL["D"] = dl * vdl; fR["D"] = dr * vdr

    # tau
    Ttl = rhb2l * wl ** 2 + Pb2l * (-1.0) - sbL[0] ** 2
    Ttr = rhb2r * wr ** 2 + Pb2r * (-1.0) - sbR[0] ** 2
    uL["tau"] = Ttl - dl; uR["tau"] = Ttr - dr
    fL["tau"] = (rhb2l * wl ** 2 * vdl - sbL[0] * bdirl) - dl * vdl
    fR["tau"] = (rhb2r * wr ** 2 * vdr - sbR[0] * bdirr) - dr * vdr

    # S_i
    for ii, (key, ul_i, ur_i, sdl_i, sdr_i) in enumerate(zip(
            ["Sx", "Sy", "Sz"], uDl, uDr, sbDL[1:], sbDR[1:])):
        uL[key] = rhb2l * wl * ul_i - sbL[0] * sdl_i
        uR[key] = rhb2r * wr * ur_i - sbR[0] * sdr_i
        fL[key] = rhb2l * (wl * vdl) * ul_i + Pb2l * _delta(ii, idir) - bdirl * sdl_i
        fR[key] = rhb2r * (wr * vdr) * ur_i + Pb2r * _delta(ii, idir) - bdirr * sdr_i

    # B^i
    Bdirl = [primL["Bx"], primL["By"], primL["Bz"]][idir]
    Bdirr = [primR["Bx"], primR["By"], primR["Bz"]][idir]
    for key, vil, vir in zip(
            ["Bx", "By", "Bz"],
            [primL["vx"], primL["vy"], primL["vz"]],
            [primR["vx"], primR["vy"], primR["vz"]]):
        uL[key] = primL[key]; uR[key] = primR[key]
        fL[key] = vdl * primL[key] - vil * Bdirl
        fR[key] = vdr * primR[key] - vir * Bdirr

    return uL, uR, fL, fR, cmax, cmin


# ── HLLDComputation ──────────────────────────────────────────────────────────

class HLLDComputation:
    """
    Mirrors C++ HLLDComputation (advanced_riemann_solvers.hh).
    Batched over N interfaces; all arrays shape (N,) or scalar.
    """
    _S = ["Sx", "Sy", "Sz"]
    _B = ["Bx", "By", "Bz"]

    def __init__(
        self,
        fL: Cons, fR: Cons,
        uL: Cons, uR: Cons,
        primL: Prim, primR: Prim,
        cmin: torch.Tensor, cmax: torch.Tensor,
        idir: int = 0,
    ):
        self.fL = fL; self.fR = fR
        self.uL = uL; self.uR = uR
        self.primL = primL; self.primR = primR
        self.cmin = _t(cmin); self.cmax = _t(cmax)
        self.idir = idir
        self.t1 = (idir + 1) % 3
        self.t2 = (idir + 2) % 3
        self._build_R()

    # ---- pre-compute R vectors (depend only on wave speeds + fluxes) --------
    def _build_R(self):
        i, t1, t2 = self.idir, self.t1, self.t2
        fL, fR = self.fL, self.fR
        uL, uR = self.uL, self.uR
        cm, cx = self.cmin, self.cmax
        S, B = self._S, self._B

        self.RS_idir_L = -cm * uL[S[i]]  - fL[S[i]]
        self.RS_idir_R =  cx * uR[S[i]]  - fR[S[i]]
        self.RS_t1_L   = -cm * uL[S[t1]] - fL[S[t1]]
        self.RS_t1_R   =  cx * uR[S[t1]] - fR[S[t1]]
        self.RS_t2_L   = -cm * uL[S[t2]] - fL[S[t2]]
        self.RS_t2_R   =  cx * uR[S[t2]] - fR[S[t2]]

        self.RB_idir_L = -cm * uL[B[i]]  - fL[B[i]]
        self.RB_idir_R =  cx * uR[B[i]]  - fR[B[i]]
        self.RB_t1_L   = -cm * uL[B[t1]] - fL[B[t1]]
        self.RB_t1_R   =  cx * uR[B[t1]] - fR[B[t1]]
        self.RB_t2_L   = -cm * uL[B[t2]] - fL[B[t2]]
        self.RB_t2_R   =  cx * uR[B[t2]] - fR[B[t2]]

        self.RTau_L  = -cm * uL["tau"] - fL["tau"]
        self.RTau_R  =  cx * uR["tau"] - fR["tau"]
        self.RDens_L = -cm * uL["D"]   - fL["D"]
        self.RDens_R =  cx * uR["D"]   - fR["D"]

        self.S_L = torch.sign(_t(uL[B[i]]))
        self.S_R = torch.sign(_t(uR[B[i]]))

    # ---- solve all intermediate-state variables for a given p* --------------
    def compute_all_variables(self, p: torch.Tensor):
        """
        Given a trial total pressure p (the root-finder's current guess),
        compute all Alfvén-state and contact-state quantities.
        """
        p = _t(p)
        i, t1, t2 = self.idir, self.t1, self.t2
        S, B = self._S, self._B
        uL, uR = self.uL, self.uR
        cm, cx = self.cmin, self.cmax

        RSiL, RSiR = self.RS_idir_L, self.RS_idir_R
        RS1L, RS1R = self.RS_t1_L,   self.RS_t1_R
        RS2L, RS2R = self.RS_t2_L,   self.RS_t2_R
        RB1L, RB1R = self.RB_t1_L,   self.RB_t1_R
        RB2L, RB2R = self.RB_t2_L,   self.RB_t2_R
        RTL, RTR   = self.RTau_L,     self.RTau_R

        BnL = _t(uL[B[i]]); BnR = _t(uR[B[i]])

        AL = RSiL - (-cm) * RTL + p * (1 - cm * cm)
        AR = RSiR -   cx  * RTR + p * (1 - cx * cx)
        GL = RB1L ** 2 + RB2L ** 2
        GR = RB1R ** 2 + RB2R ** 2
        CL = RS1L * RB1L + RS2L * RB2L
        CR = RS1R * RB1R + RS2R * RB2R
        QL = -AL - GL + BnL ** 2 * (1 - cm * cm)
        QR = -AR - GR + BnR ** 2 * (1 - cx * cx)

        XL = BnL * (AL * (-cm) * BnL + CL) - (AL + GL) * ((-cm) * p + RTL)
        XR = BnR * (AR *   cx  * BnR + CR) - (AR + GR) * (  cx  * p + RTR)

        # Alfvén-state velocities — safe division against XL/XR = 0
        vaL  = _sdiv(BnL * (AL * BnL + (-cm) * CL) - (AL + GL) * (p + RSiL), XL)
        vaR  = _sdiv(BnR * (AR * BnR +   cx  * CR) - (AR + GR) * (p + RSiR), XR)

        vt1aL = _sdiv(QL * RS1L + RB1L * (CL + BnL * ((-cm) * RSiL - RTL)), XL)
        vt1aR = _sdiv(QR * RS1R + RB1R * (CR + BnR * (  cx  * RSiR - RTR)), XR)
        vt2aL = _sdiv(QL * RS2L + RB2L * (CL + BnL * ((-cm) * RSiL - RTL)), XL)
        vt2aR = _sdiv(QR * RS2R + RB2R * (CR + BnR * (  cx  * RSiR - RTR)), XR)

        # Alfvén-state B⊥ — safe division against (-cm - vaL) / (cx - vaR) = 0
        Bt1aL = _sdiv(RB1L - BnL * vt1aL, -cm - vaL)
        Bt1aR = _sdiv(RB1R - BnR * vt1aR,  cx - vaR)
        Bt2aL = _sdiv(RB2L - BnL * vt2aL, -cm - vaL)
        Bt2aR = _sdiv(RB2R - BnR * vt2aR,  cx - vaR)

        vRS_AL = vaL * RSiL + vt1aL * RS1L + vt2aL * RS2L
        vRS_AR = vaR * RSiR + vt1aR * RS1R + vt2aR * RS2R

        vdL = _t(self.primL[["vx", "vy", "vz"][i]])
        vdR = _t(self.primR[["vx", "vy", "vz"][i]])

        # Pseudo-enthalpies in the Alfvén states — safe division
        waL = p + _sdiv(RTL - vRS_AL, -cm - vaL)
        waR = p + _sdiv(RTR - vRS_AR,  cx - vaR)

        DaL = _sdiv(self.RDens_L, -cm - vaL)
        DaR = _sdiv(self.RDens_R,  cx - vaR)

        # v·B in each Alfvén state (uses outer L/R B fields, not Alfvén B)
        vBaL = vaL * BnL + vt1aL * Bt1aL + vt2aL * Bt2aL
        vBaR = vaR * BnR + vt1aR * Bt1aR + vt2aR * Bt2aR

        # Alfvén-state tau — note: uses p (current guess), NOT an outer p_star
        TauaL = _sdiv(RTL + p * vaL - vBaL * BnL, -cm - vaL) - DaL # Marie Hot fix
        TauaR = _sdiv(RTR + p * vaR - vBaR * BnR,  cx - vaR) - DaR

        SaL   = (TauaL + DaL + p) * vaL   - vBaL * BnL 
        SaR   = (TauaR + DaR + p) * vaR   - vBaR * BnR
        St1aL = (TauaL + DaL + p) * vt1aL - vBaL * Bt1aL
        St1aR = (TauaR + DaR + p) * vt1aR - vBaR * Bt1aR
        St2aL = (TauaL + DaL + p) * vt2aL - vBaL * Bt2aL
        St2aR = (TauaR + DaR + p) * vt2aR - vBaR * Bt2aR

        #sqL = torch.sqrt(torch.clamp(waL, min=0.0)) #Marie
        #sqR = torch.sqrt(torch.clamp(waR, min=0.0))

        sqL = torch.sqrt(torch.clamp(waL, min=0.0))
        sqR = torch.sqrt(torch.clamp(waR, min=0.0))

        # K-vector denominators — safe division
        dL = (-cm) * p + RTL - BnL * self.S_L * sqL 
        dR =   cx  * p + RTR + BnR * self.S_R * sqR

        # K-vector components (Mignone 2009 eq 37 discretised form)
        KaL   = _sdiv(RSiL + p - self.RB_idir_L * self.S_L * sqL, dL)
        KaR   = _sdiv(RSiR + p + self.RB_idir_R * self.S_R * sqR, dR)
        Kt1aL = _sdiv(RS1L - RB1L * self.S_L * sqL, dL)
        Kt1aR = _sdiv(RS1R + RB1R * self.S_R * sqR, dR)
        Kt2aL = _sdiv(RS2L - RB2L * self.S_L * sqL, dL)
        Kt2aR = _sdiv(RS2R + RB2R * self.S_R * sqR, dR)

        laL = KaL; laR = KaR

        # Contact B fields — safe division against (laR - laL) = 0
        denom_ll = laR - laL
        Bic  = BnR
        Bt1c = _sdiv(
            (Bt1aR * (laR - vaR) + BnR * vt1aR)
          - (Bt1aL * (laL - vaL) + BnL * vt1aL),
            denom_ll)
        Bt2c = _sdiv(
            (Bt2aR * (laR - vaR) + BnR * vt2aR)
          - (Bt2aL * (laL - vaL) + BnL * vt2aL),
            denom_ll)

        KaL2  = KaL  ** 2 + Kt1aL ** 2 + Kt2aL ** 2
        KaR2  = KaR  ** 2 + Kt1aR ** 2 + Kt2aR ** 2
        KaLBc = KaL  * Bic + Kt1aL * Bt1c + Kt2aL * Bt2c
        KaRBc = KaR  * Bic + Kt1aR * Bt1c + Kt2aR * Bt2c

        # ---- persist ---------------------------------------------------------
        self.vaL = vaL;     self.vaR = vaR
        self.vt1aL = vt1aL; self.vt2aL = vt2aL
        self.vt1aR = vt1aR; self.vt2aR = vt2aR
        self.Bt1aL = Bt1aL; self.Bt2aL = Bt2aL
        self.Bt1aR = Bt1aR; self.Bt2aR = Bt2aR
        self.waL = waL; self.waR = waR
        self.sqL = sqL; self.sqR = sqR
        self.DaL = DaL; self.DaR = DaR
        self.TauaL = TauaL; self.TauaR = TauaR
        self.SaL = SaL;     self.SaR = SaR
        self.St1aL = St1aL; self.St2aL = St2aL
        self.St1aR = St1aR; self.St2aR = St2aR
        self.laL = laL; self.laR = laR
        # K-vector components needed for the contact-velocity formula
        self.Kt1aL = Kt1aL; self.Kt2aL = Kt2aL
        self.Kt1aR = Kt1aR; self.Kt2aR = Kt2aR
        self.Bic = Bic; self.Bt1c = Bt1c; self.Bt2c = Bt2c
        self.KaL2 = KaL2; self.KaR2 = KaR2
        self.KaLBc = KaLBc; self.KaRBc = KaRBc

    # ---- residual for the root-finder (scalar in p*) ------------------------
    def residual(self, p: torch.Tensor) -> torch.Tensor:
        """
        Returns the pressure-balance residual (zero at p*).

        The L denominator (S_L·√w_L + K_L·B_c) is guaranteed nonzero at the
        true root (Mignone 2009 / C++ line 1257).  Safe division prevents NaN
        when evaluating away from the root.
        """
        i = self.idir
        BnL = _t(self.uL[self._B[i]])

        self.compute_all_variables(p)
        denom_R = self.S_R * self.sqR - self.KaRBc   # R-term denominator
        denom_L = self.S_L * self.sqL + self.KaLBc   # L-term denominator (+ sign)
        return (self.laL - self.laR) + BnL * (_sdiv(1.0 - self.KaR2, denom_R) + _sdiv(1.0 - self.KaL2, denom_L))

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        return self.residual(p)


# ── root-finder ──────────────────────────────────────────────────────────────

def safe_secant_bisection_old(
    func,
    x0: torch.Tensor,
    x1: torch.Tensor,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hybrid secant / bisection root-finder, batched over N independent
    problems.  Returns (root, err) where err>0 flags non-convergence.
    """
    x0 = _t(x0).clone(); x1 = _t(x1).clone()
    scalar = x0.ndim == 0
    if scalar:
        x0 = x0.reshape(1); x1 = x1.reshape(1)

    f0 = _t(func(x0)); f1 = _t(func(x1))
    err = torch.full(
        x0.shape,
        max_iter,
        dtype=torch.int32,
        device=x0.device,
    )
    iter=0
    for _ in range(max_iter):
        iter += 1

        conv = torch.abs(f1) < tol
        err = torch.where(conv, torch.zeros_like(err), err)
        if torch.all(conv):
            break

        df = f1 - f0
        safe = torch.abs(df) > 1e-30
        x2s = torch.where(safe,
                          x1 - f1 * (x1 - x0) / torch.where(safe, df, torch.ones_like(df)),
                          0.5 * (x0 + x1))
        inb = (x2s - x0) * (x2s - x1) < 0
        x2  = torch.where(inb, x2s, 0.5 * (x0 + x1))
        f2  = _t(func(x2))

        lft = (f0 * f2) <= 0
        x0  = torch.where(lft, x0, x1)
        f0  = torch.where(lft, f0, f1)
        x1  = x2; f1 = f2
    print("We needed this many iterations: ", iter)

    if scalar:
        return x1.reshape(()), err.reshape(())
    return x1, err

def safe_secant_bisection(
    func,
    x0: torch.Tensor,
    x1: torch.Tensor,
    tol: float = 1e-12,
    max_iter: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x0 = _t(x0).clone(); x1 = _t(x1).clone()
    scalar = x0.ndim == 0
    if scalar:
        x0 = x0.reshape(1); x1 = x1.reshape(1)

    f0  = _t(func(x0))
    f1  = _t(func(x1))
    err = torch.full(x0.shape, max_iter, dtype=torch.int32, device=x0.device)
    iter = 0

    for _ in range(max_iter):
        iter += 1

        conv = torch.abs(f1) < tol
        err  = torch.where(conv, torch.zeros_like(err), err)
        if torch.all(conv):
            break

        df   = f1 - f0
        safe = df.abs() > 1e-30 * torch.abs(f1)
        dx   = torch.where(safe,
                           f1 * (x1 - x0) / torch.where(safe, df, torch.ones_like(df)),
                           torch.zeros_like(f1))
        x2   = x1 - dx
        x2   = torch.clamp(x2, min=1e-30)

        # ── Key: freeze already-converged elements ──────────────────────────
        x2   = torch.where(conv, x1, x2)   # don't move converged points
        f2   = _t(func(x2))
        f2   = torch.where(conv, f1, f2)   # don't overwrite converged residuals

        x0, f0 = x1, f1
        x1, f1 = x2, f2

    if scalar:
        return x1.reshape(()), err.reshape(())
    print("We needed this many iterations: ", iter)
    return x1, err

def bisection(
    func,
    x_lo: torch.Tensor,
    x_hi: torch.Tensor,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_lo = _t(x_lo).clone(); x_hi = _t(x_hi).clone()
    scalar = x_lo.ndim == 0
    if scalar:
        x_lo = x_lo.reshape(1); x_hi = x_hi.reshape(1)

    f_lo = _t(func(x_lo))
    f_hi = _t(func(x_hi))

    err = torch.full(x_lo.shape, max_iter, dtype=torch.int32, device=x_lo.device)

    for _ in range(max_iter):
        x_mid = 0.5 * (x_lo + x_hi)
        f_mid = _t(func(x_mid))

        conv = (x_hi - x_lo).abs() < tol
        err  = torch.where(conv, torch.zeros_like(err), err)
        if torch.all(conv):
            break

        # Freeze converged cells
        x_mid = torch.where(conv, x_lo, x_mid)
        f_mid = torch.where(conv, f_lo, f_mid)

        # Standard bisection update
        left = (f_lo * f_mid) <= 0.0
        x_lo = torch.where(left, x_lo, x_mid)
        f_lo = torch.where(left, f_lo, f_mid)
        x_hi = torch.where(left, x_mid, x_hi)
        f_hi = torch.where(left, f_mid, f_hi)

    if scalar:
        return x_mid.reshape(()), err.reshape(())
    return x_mid, err

def secant(
    func,
    x0: torch.Tensor,
    x1: torch.Tensor,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x0 = _t(x0).clone(); x1 = _t(x1).clone()
    scalar = x0.ndim == 0
    if scalar:
        x0 = x0.reshape(1); x1 = x1.reshape(1)

    f0 = _t(func(x0))
    f1 = _t(func(x1))
    err = torch.full(x0.shape, max_iter, dtype=torch.int32, device=x0.device)

    for _ in range(max_iter):
        conv = torch.abs(f1) < tol
        err  = torch.where(conv, torch.zeros_like(err), err)
        if torch.all(conv):
            break

        df   = f1 - f0
        safe = df.abs() > 1e-30 * f1.abs()
        dx   = torch.where(safe,
                           f1 * (x1 - x0) / torch.where(safe, df, torch.ones_like(df)),
                           torch.zeros_like(f1))

        # Limit step to at most 50% change per iteration
        dx   = torch.clamp(dx, min=-0.5 * x1, max=0.5 * x1)

        x2   = x1 - dx
        x2   = torch.clamp(x2, min=1e-30)

        # Freeze converged cells
        x2   = torch.where(conv, x1, x2)
        f2   = _t(func(x2))
        f2   = torch.where(conv, f1, f2)

        x0, f0 = x1, f1
        x1, f1 = x2, f2

    if scalar:
        return x1.reshape(()), err.reshape(())
    return x1, err

# ── standalone HLLE flux ─────────────────────────────────────────────────────

def hlle_flux(
    sL: Prim,
    sR: Prim,
    eos: hybrid_eos,
    idir: int = 0,
) -> Tuple[Cons, Cons, torch.Tensor]:
    """
    HLLE numerical flux for SR-MHD in Minkowski spacetime.

    Drop-in replacement for hlld_flux — same signature and return layout:
        fHLLE : Cons   – numerical flux dict
        uHLLE : Cons   – resolved state dict
        p_hll : Tensor – HLL total-pressure estimate (replaces p_star)
    """
    uL, uR, fL, fR, cmax, cmin = compute_srmhd_fluxes(sL, sR, eos, idir)

    keys = list(uL.keys())

    fHLL: Cons = {k: _hf(fL[k], fR[k], uL[k], uR[k], cmin, cmax) for k in keys}
    uHLL: Cons = {k: _hu(fL[k], fR[k], uL[k], uR[k], cmin, cmax) for k in keys}
    
    fH: Cons = {k: torch.where(cmax <= 0, fR[k], 
                                torch.where(cmin <= 0, fL[k], 
                                fHLL[k])) for k in keys}
    uH: Cons = {k: torch.where(cmin <= 0, uL[k], 
                                torch.where(cmax <= 0, uR[k], 
                                uHLL[k])) for k in keys}

    # HLL total-pressure estimate (returned for interface parity with hlld_flux)
    pl, _ = eos.press_and_cs2(sL["eps"], sL["rho"])
    pr, _ = eos.press_and_cs2(sR["eps"], sR["rho"])
    wl, _ = lorentz(sL); wr, _ = lorentz(sR)
    b2l   = compute_b2((sL["vx"], sL["vy"], sL["vz"]),
                        (sL["Bx"], sL["By"], sL["Bz"]), wl)
    b2r   = compute_b2((sR["vx"], sR["vy"], sR["vz"]),
                        (sR["Bx"], sR["By"], sR["Bz"]), wr)
    p_hll = torch.clamp(0.5 * (pl + 0.5 * b2l + pr + 0.5 * b2r), min=1e-30)

    return fH, uH, p_hll


# ── main HLLD flux ───────────────────────────────────────────────────────────

def hlld_flux(
    sL: Prim,
    sR: Prim,
    eos: hybrid_eos,
    idir: int = 0,
    lo_fac: float = 0.1,
    hi_fac: float = 10,
) -> Tuple[Cons, Cons, torch.Tensor]:
    """
    HLLD numerical flux for SR-MHD in Minkowski spacetime.

    Parameters
    ----------
    sL, sR : Prim
        Left / right primitive states.
    eos : hybrid_eos
        Equation of state (must have ``press_and_cs2``).
    idir : int
        Flux direction (0=x, 1=y, 2=z).
    lo_fac, hi_fac : float
        Bracket factors around the HLL pressure estimate for the
        secant/bisection root-finder.

    Returns
    -------
    fHLLD : Cons   – numerical flux dict
    uHLLD : Cons   – resolved state dict
    p_star : Tensor – total pressure in the star region
    """
    uL, uR, fL, fR, cmax, cmin = compute_srmhd_fluxes(sL, sR, eos, idir)
    keys = list(uL.keys())

    # HLL fallback values (used when HLLD degenerates or the solver fails)
    fH: Cons = {k: _hf(fL[k], fR[k], uL[k], uR[k], cmin, cmax) for k in keys}
    uH: Cons = {k: _hu(fL[k], fR[k], uL[k], uR[k], cmin, cmax) for k in keys}


    # –– Calculate new set of conserved variables. ——————————————————
    lfacL, _ = lorentz(sL)
    lfacR, _ = lorentz(sR) 

    b2L = compute_b2((sL["vx"], sL["vy"], sL["vz"]),
                      (sL["Bx"], sL["By"], sL["Bz"]), lfacL)
    b2R = compute_b2((sR["vx"], sR["vy"], sR["vz"]),
                      (sR["Bx"], sR["By"], sR["Bz"]), lfacR)
    
    press_L, _ = eos.press_and_cs2(sL["eps"], sL["rho"])
    press_R, _ = eos.press_and_cs2(sR["eps"], sR["rho"])

    vl = (sL["vx"], sL["vy"], sL["vz"])
    vr = (sR["vx"], sR["vy"], sR["vz"])
    Bl = (sL["Bx"], sL["By"], sL["Bz"])
    Br = (sR["Bx"], sR["By"], sR["Bz"])

    sbL = compute_smallb(vl, Bl, lfacL)
    sbR = compute_smallb(vr, Br, lfacR)

    wL = sL["rho"] * (1.0 + sL["eps"]) + press_L + b2L
    wR = sR["rho"] * (1.0 + sR["eps"]) + press_R + b2R

    U_init_L : Cons = {}
    U_init_R : Cons = {}

    U_init_L["D"]   = uL["D"]
    U_init_R["D"]   = uR["D"]

    U_init_L["Sx"] = wL * lfacL**2 * sL["vx"] - sbL[0] * sbL[1] # its - (-sbL[0]) 
    U_init_R["Sx"] = wR * lfacR**2 * sR["vx"] - sbR[0] * sbR[1] # its - (-sbR[0])

    U_init_L["Sy"] = wL * lfacL**2 * sL["vy"] - sbL[0] * sbL[2] # Marie mögliches andes vorzeichen
    U_init_R["Sy"] = wR * lfacR**2 * sR["vy"] - sbR[0] * sbR[2]

    U_init_L["Sz"] = wL * lfacL**2 * sL["vz"] - sbL[0] * sbL[3] 
    U_init_R["Sz"] = wR * lfacR**2 * sR["vz"] - sbR[0] * sbR[3]

    U_init_L["tau"] = wL * lfacL**2 - press_L - 0.5 * b2L - sbL[0]**2
    U_init_R["tau"] = wR * lfacR**2 - press_R - 0.5 * b2R - sbR[0]**2

    U_init_L["Bx"]  = sL["Bx"]
    U_init_R["Bx"]  = sR["Bx"]
    U_init_L["By"]  = sL["By"]
    U_init_R["By"]  = sR["By"]
    U_init_L["Bz"]  = sL["Bz"]
    U_init_R["Bz"]  = sR["Bz"]

    F_init_L : Cons = {k: fL[k] for k in keys}
    F_init_R : Cons = {k: fR[k] for k in keys}

    F_init_L["Sx"] = wL * lfacL**2 * sL["vx"] * sL["vx"] + press_L + 0.5 * b2L - sbL[1] * sbL[1]
    F_init_R["Sx"] = wR * lfacR**2 * sR["vx"] * sR["vx"] + press_R + 0.5 * b2R - sbR[1] * sbR[1]
    
    F_init_L["Sy"] = wL * lfacL**2 * sL["vx"] * sL["vy"] - sbL[1] * sbL[2]
    F_init_R["Sy"] = wR * lfacR**2 * sR["vx"] * sR["vy"] - sbR[1] * sbR[2]

    F_init_L["Sz"] = wL * lfacL**2 * sL["vx"] * sL["vz"] - sbL[1] * sbL[3]
    F_init_R["Sz"] = wR * lfacR**2 * sR["vx"] * sR["vz"] - sbR[1] * sbR[3]

    F_init_L["tau"] = wL * lfacL**2 * sL["vx"] - sbL[1] * sbL[0] # its - (-sbL[0])
    F_init_R["tau"] = wR * lfacR**2 * sR["vx"] - sbR[1] * sbR[0] # its - (-sbR[0])

    # ── initial guess for p* from the HLL state (Mignone 2009 eq. 53) ─────────────────
    prim_hll = conservative_to_primitive(uH, eos)
    p_hll, _ = eos.press_and_cs2(prim_hll["eps"], prim_hll["rho"])

    # p0 from the Bx=0 limit (Mignone 2009 eq. 55)
    a = fH["tau"] + fH["D"]
    b = -fH["Sx"] - uH["tau"] - uH["D"]   
    c = uH["Sx"]

    disc = b**2 - 4.0 * a * c
    disc = torch.clamp(disc, min=0.0)

    # Guard a→0 (non-relativistic / degenerate limit)
    safe_a = torch.where(a.abs() > 1e-14, a, torch.ones_like(a))
    safe_b = torch.where(a.abs() > 1e-14, b, torch.ones_like(b))
    
    v_star = torch.where(
        a.abs() > 1e-14,
        (-b - torch.sqrt(disc)) / (2.0 * safe_a),   # negative root
        -c / safe_b,                                  # linear fallback
    )

    # Clamp v_star to subluminal range before using it
    v_star = torch.clamp(v_star, min=-(1.0 - 1e-10), max=(1.0 - 1e-10))

    p_star = fH["Sx"] - (fH["tau"] + fH["D"]) * v_star
    p_star = torch.clamp(p_star, min=1e-30) # p0


    # ── Pressure bracket ──────────────────────────────────────────────────────
    p0    = p_star  # rename for clarity

    # p_hll is pure hydro + b2/2 needed
    w_hll, _ = lorentz(prim_hll)
    b2_hll = compute_b2((prim_hll["vx"], prim_hll["vy"], prim_hll["vz"]),
                        (prim_hll["Bx"], prim_hll["By"], prim_hll["Bz"]), w_hll)
    p_hll = p_hll + 0.5 * b2_hll

    # Choose starting point per Mignone eq. (53)
    BnL   = _t(uL["Bx"])
    BnR   = _t(uR["Bx"])
    Bn2   = 0.5 * (BnL**2 + BnR**2)
    use_p0 = (Bn2 / p_hll) < 0.1

    # The two secant points: p0 and p_hll are naturally distinct
    # and physically motivated — use them directly
    p_lo = torch.where(use_p0, p0 * 0.99, p_hll * 0.99)
    p_hi = torch.where(use_p0, p0 * 1.01, p_hll * 1.01)

    B = HLLDComputation._B
    BnL = _t(uL[B[idir]]); BnR = _t(uR[B[idir]])
    bn_zero = (BnL.abs() < 1e-14) & (BnR.abs() < 1e-14)

    # ── Solve for p* ─────────────────────────────────────────────────────────
    #calc = HLLDComputation(fL, fR, uL, uR, sL, sR, cmin, cmax, idir) #Marie
    calc = HLLDComputation(F_init_L, F_init_R, U_init_L, U_init_R, sL, sR, cmin, cmax, idir)

    f_lo = _t(calc(p_lo))
    f_hi = _t(calc(p_hi))

    for _ in range(5):
        bad = (f_lo * f_hi) > 0.0
        if not torch.any(bad):
            break
        p_lo = torch.where(bad, p_lo / 2.0, p_lo)
        p_hi = torch.where(bad, p_hi * 2.0, p_hi)
        f_lo = torch.where(bad, _t(calc(p_lo)), f_lo)
        f_hi = torch.where(bad, _t(calc(p_hi)), f_hi)


    p_star, err = safe_secant_bisection(calc, p_lo, p_hi, tol=1e-6)
    #print("err", err)
    #print("p_star: ", p_star)

    #err = torch.ones_like(p_lo) * 0.0
    #p_star = p_hll
    calc.compute_all_variables(p_star)
    print("resi", calc(p_star))

    #p_star, err = secant(calc, p_lo, p_hi, tol=1e-12)

    #p_star, err = bisection(calc, p_lo, p_hi, tol=1e-15, max_iter=200)

    

    # ── Unpack intermediate-state quantities ──────────────────────────────────
    i  = idir; t1 = calc.t1; t2 = calc.t2
    S = calc._S; B = calc._B
    BnL = _t(uL[B[i]]); BnR = _t(uR[B[i]])

    # Fast-wave (a) intermediate states
    uaL: Cons = {
        B[i]: BnL, B[t1]: calc.Bt1aL, B[t2]: calc.Bt2aL,
        "D": calc.DaL, "tau": calc.TauaL,
        S[i]: calc.SaL, S[t1]: calc.St1aL, S[t2]: calc.St2aL,
    }
    uaR: Cons = {
        B[i]: BnR, B[t1]: calc.Bt1aR, B[t2]: calc.Bt2aR,
        "D": calc.DaR, "tau": calc.TauaR,
        S[i]: calc.SaR, S[t1]: calc.St1aR, S[t2]: calc.St2aR,
    }

    #uaL["tau"] = uaL["tau"] - uaL["D"] # Marie very important hot fix
    #uaR["tau"] = uaR["tau"] - uaR["D"]

    faL: Cons = {k: fL[k] + (-cmin) * (uaL[k] - uL[k]) for k in keys}
    faR: Cons = {k: fR[k] + ( cmax) * (uaR[k] - uR[k]) for k in keys}
    # ── Contact velocities (Mignone 2009, eq 44) ──────────────────────────────
    #
    # fac = (1 - |K_L|²) / (S_L √w_L − K_L·B_c)
    #
    # The denominator uses the SAME sign as the L-term in the residual function
    # (S_L·sqL − KaLBc).  Safe division via _sdiv prevents NaN if it hits zero.
    #
    # v_c^n    = K_n^{aL}   − Bc^n    · fac   (normal direction)
    # v_c^{t1} = K_{t1}^{aL} − Bc^{t1} · fac   (transverse — uses K-vector, not vaL)
    # v_c^{t2} = K_{t2}^{aL} − Bc^{t2} · fac
    fac = _sdiv(1.0 - calc.KaL2, - calc.S_L * calc.sqL - calc.KaLBc) #Marie changed  + calc.KaLBc  to  - calc.KaLBc

    vc  = calc.laL   - calc.Bic  * fac
    vt1 = calc.Kt1aL - calc.Bt1c * fac
    vt2 = calc.Kt2aL - calc.Bt2c * fac

    facR = _sdiv(1.0 - calc.KaR2, + calc.S_R * calc.sqR - calc.KaRBc)
    vc_R = calc.laR - calc.Bic * facR

    lC  = vc
    laL = calc.laL; laR = calc.laR

    # ── Contact (c) states ────────────────────────────────────────────────────
    ucL: Cons = {B[i]: calc.Bic, B[t1]: calc.Bt1c, B[t2]: calc.Bt2c}
    ucR: Cons = {B[i]: calc.Bic, B[t1]: calc.Bt1c, B[t2]: calc.Bt2c}

    # v·B with contact velocities and contact B fields
    vBc = vc * ucL[B[i]] + vt1 * ucL[B[t1]] + vt2 * ucL[B[t2]]

    #ucL["D"]   = _sdiv(laL * uaL["D"]   - faL["D"],   laL - vc)
    #ucR["D"]   = _sdiv(laR * uaR["D"]   - faR["D"],   laR - vc)
    #ucL["tau"] = _sdiv(laL * uaL["tau"] - faL["tau"] + p_star * vc - vBc * ucL[B[i]], laL - vc)
    #ucR["tau"] = _sdiv(laR * uaR["tau"] - faR["tau"] + p_star * vc - vBc * ucR[B[i]], laR - vc)

    # This uses the HLLC formulation from Mignone and
    # Marie TODO This is all unclear how to define it
    ucL["D"]   = uaL["D"] * _sdiv(laL - calc.vaL, laL - vc)
    ucR["D"]   = uaR["D"] * _sdiv(laR - calc.vaR, laR - vc)

    #ucL["tau"] = _sdiv(laL * uaL["tau"] - uaL["Sx"] + p_star * vc - vBc * ucL[B[i]], laL - vc)
    #ucR["tau"] = _sdiv(laR * uaR["tau"] - uaR["Sx"] + p_star * vc - vBc * ucR[B[i]], laR - vc)

    ucL["tau"] = _sdiv(laL * uaL["tau"] - faL["tau"] + p_star * vc - vBc * ucL[B[i]], laL - vc)
    ucR["tau"] = _sdiv(laR * uaR["tau"] - faR["tau"] + p_star * vc - vBc * ucR[B[i]], laR - vc)
    

    for key, vi, Bi in zip(
            [S[i], S[t1], S[t2]], [vc, vt1, vt2],
            [ucL[B[i]], ucL[B[t1]], ucL[B[t2]]]):
        ucL[key] = (ucL["tau"] + p_star + ucL["D"]) * vi - vBc * Bi
    for key, vi, Bi in zip(
            [S[i], S[t1], S[t2]], [vc, vt1, vt2],
            [ucR[B[i]], ucR[B[t1]], ucR[B[t2]]]):
        ucR[key] = (ucR["tau"] + p_star + ucR["D"]) * vi - vBc * Bi

    # ── Assemble flux based on wave-speed ordering ────────────────────────────
    vi = torch.zeros_like(cmin)   # Minkowski: shift = 0

    cL_mask  = vi <= -cmin
    caL_mask = (-cmin < vi) & (vi <= laL)
    ccL_mask = (laL < vi) & (vi < lC)
    ccR_mask = (lC <= vi) & (vi < laR)
    caR_mask = (laR <= vi) & (vi < cmax)
    cR_mask  = vi >= cmax

    fD: Cons = {k: fH[k].clone() for k in keys}
    uD: Cons = {k: uH[k].clone() for k in keys}


    for k in keys:
        fcL = faL[k] + laL * (ucL[k] - uaL[k])
        fcR = faR[k] + laR * (ucR[k] - uaR[k])

        f = fD[k]
        f = torch.where(cL_mask,  fL[k],   f)
        f = torch.where(caL_mask, faL[k],  f)
        f = torch.where(ccL_mask, fcL,     f)
        f = torch.where(ccR_mask, fcR,     f)
        f = torch.where(caR_mask, faR[k],  f)
        f = torch.where(cR_mask,  fR[k],   f)
        fD[k] = f

        u = uD[k]
        u = torch.where(cL_mask,  uL[k],  u)
        u = torch.where(caL_mask, uaL[k], u)
        u = torch.where(ccL_mask, ucL[k], u)
        u = torch.where(ccR_mask, ucR[k], u)
        u = torch.where(caR_mask, uaR[k], u)
        u = torch.where(cR_mask,  uR[k],  u)
        uD[k] = u

    # ── Detect degenerate HLLD states ─────────────────────────────────────────
    # 1. Negative pseudo-enthalpy in an Alfvén state (√waL/R is imaginary).
    # 2. Superluminal Alfvén K-vector (|K|² > 1+ε), meaning the Alfvén and fast
    #    waves merge — the 5-wave structure is ill-defined in that limit.
    # Both conditions require falling back to HLLE.
    wave_order_bad = (
      (calc.vaR >  cmax)
      | (calc.vaL < -cmin)
      | (vc < calc.laL)
      | (vc > calc.laR)
    )

    #print(wave_order_bad)
    
    #degen = ( torch.tesnorfalse |
    #    (calc.waL < 0.0) | (calc.waR < 0.0) |
    #     (calc.KaL2 > 1.0 + 1e-4) |
    #     (calc.KaR2 > 1.0 + 1e-4) |
    #     wave_order_bad                    # ← add this
    #)
    #print(degen)


    # ── Fall back to HLLE where solver failed, B_n = 0, or state degenerate ──
    failed = (err > 0)  | wave_order_bad 
    if torch.any(failed):
        p_star = torch.where(failed, -torch.abs(p_hll) - 1e-10, p_star)
        for k in keys:
            fD[k] = torch.where(failed, fH[k], fD[k])
            uD[k] = torch.where(failed, uH[k], uD[k])

    # ── Final safety: replace any residual NaN / Inf with HLLE ───────────────
    #for k in keys:
    #    bad = ~torch.isfinite(fD[k]) | ~torch.isfinite(uD[k])
    #    if torch.any(bad):
    #        fD[k] = torch.where(bad, fH[k], fD[k])
    #        uD[k] = torch.where(bad, uH[k], uD[k])

    return fD, uD, p_star

# ── HLLC flux ────────────────────────────────────────────────────────────────

def hllc_flux(
    sL: Prim,
    sR: Prim,
    eos: hybrid_eos,
    idir: int = 0,
) -> Tuple[Cons, Cons, torch.Tensor]:
    """
    HLLC numerical flux for SR-MHD in Minkowski spacetime (no GLM/Phi).
    Follows the GRACE C++ hllc_riemann_solver_t implementation exactly.
    Reference: Mignone https://arxiv.org/pdf/2111.09369
    """
    uL, uR, fL, fR, cmax, cmin = compute_srmhd_fluxes(sL, sR, eos, idir)
    keys = list(uL.keys())

    t1 = (idir + 1) % 3
    t2 = (idir + 2) % 3
    Si = ["Sx", "Sy", "Sz"]
    Bi = ["Bx", "By", "Bz"]

    S_L = -cmin   # left fast speed  (negative, stored as positive cmin)
    S_R =  cmax   # right fast speed (positive)

    # ── Step 1: HLLE intermediate state ──────────────────────────────────────
    fH: Cons = {k: _hf(fL[k], fR[k], uL[k], uR[k], cmin, cmax) for k in keys}
    uH: Cons = {k: _hu(fL[k], fR[k], uL[k], uR[k], cmin, cmax) for k in keys}

    # Normal B from left state (constant across face; use HLLE if GLM)
    BG_idir = uL[Bi[idir]]
    has_B   = BG_idir.abs() > 1e-15

    # ── Step 2: Contact wave speed (C++ get_contact_wave_speed) ──────────────
    #
    # With magnetic field the quadratic to solve is:
    #   a*vc² + b*vc + c = 0
    # where cross_term, bg_tan_2, f_bg_tan_2 are magnetic cross terms.
    # Without B this reduces to the standard linear HD formula.
    # C++ always takes the MINUS root.

    cons_bg_t1 = uH[Bi[t1]]
    cons_bg_t2 = uH[Bi[t2]]
    f_bg_t1    = fH[Bi[t1]]
    f_bg_t2    = fH[Bi[t2]]

    cross_term  = torch.where(has_B, cons_bg_t1*f_bg_t1 + cons_bg_t2*f_bg_t2,
                              torch.zeros_like(BG_idir))
    bg_tan_2    = torch.where(has_B, cons_bg_t1**2 + cons_bg_t2**2,
                              torch.zeros_like(BG_idir))
    f_bg_tan_2  = torch.where(has_B, f_bg_t1**2 + f_bg_t2**2,
                              torch.zeros_like(BG_idir))

    E_hll  = uH["tau"] + uH["D"]
    fE_hll = fH["tau"] + fH["D"]

    qa = fE_hll - cross_term
    qb = -fH[Si[idir]] - E_hll + bg_tan_2 + f_bg_tan_2
    qc = uH[Si[idir]] - cross_term

    # Solve quadratic; fall back to linear when |qa| is tiny
    disc_vc   = torch.sqrt(torch.clamp(qb**2 - 4.0*qa*qc, min=0.0))
    vc_quad   = _sdiv(-0.5*(qb + disc_vc), qa)          # minus root (causal)
    vc_linear = _sdiv(-qc, qb)                           # a→0 limit
    vc = torch.where(qa.abs() < 1e-15,
                     torch.where(qb.abs() < 1e-15,
                                 torch.zeros_like(qa),
                                 vc_linear),
                     vc_quad)

    # ── Step 3: Star-region transverse velocities and gamma* ─────────────────
    # Only meaningful when B is present (C++ guards with has_magnetic_field)
    vt1 = torch.where(has_B,
                      _sdiv(uH[Bi[t1]]*vc - fH[Bi[t1]], BG_idir),
                      torch.zeros_like(vc))
    vt2 = torch.where(has_B,
                      _sdiv(uH[Bi[t2]]*vc - fH[Bi[t2]], BG_idir),
                      torch.zeros_like(vc))

    v2         = vc**2 + vt1**2 + vt2**2
    v2_safe    = torch.clamp(v2, max=1.0 - 1e-10)
    gamma_star = 1.0 / torch.sqrt(1.0 - v2_safe)

    v_star_B_star = torch.where(
        has_B,
        vc*uH[Bi[idir]] + vt1*uH[Bi[t1]] + vt2*uH[Bi[t2]],
        torch.zeros_like(vc),
    )

    # ── Step 4: p* ────────────────────────────────────────────────────────────
    # C++ (with B):
    #   p* = fHLLE[Si] + (BG/γ*)² - vc*(fHLLE[τ]+fHLLE[D] - BG*v*B*)
    # C++ (without B):
    #   p* = fHLLE[Si] - vc*(fHLLE[τ]+fHLLE[D])
    p_star_B  = (fH[Si[idir]]
                 + (BG_idir / gamma_star)**2
                 - vc * (fH["tau"] + fH["D"] - BG_idir*v_star_B_star))
    p_star_0  = fH[Si[idir]] - vc * (fH["tau"] + fH["D"])
    p_star    = torch.where(has_B, p_star_B, p_star_0)

    # ── Step 5: Star states ucL, ucR ─────────────────────────────────────────

    def _star(uK, fK, SK, vdK):
        """HLLC star state for one side (C++ formulas, both B branches)."""
        denom = SK - vc

        ucK = {}

        # Magnetic field: normal unchanged, transverse from HLLE
        ucK[Bi[idir]] = BG_idir
        ucK[Bi[t1]]   = uH[Bi[t1]]
        ucK[Bi[t2]]   = uH[Bi[t2]]

        # Density (same formula both branches)
        ucK["D"] = _sdiv(uK["D"] * (SK - vdK), denom)

        # Energy (same formula both branches — p* and v*B* already account for B)
        ucK["tau"] = _sdiv(
            SK * uK["tau"] - fK["tau"] + p_star*vc - v_star_B_star*BG_idir,
            denom,
        )

        # Normal momentum
        ucK[Si[idir]] = (ucK["tau"] + ucK["D"] + p_star)*vc - v_star_B_star*BG_idir

        # Transverse momenta: two branches
        # With B (C++):
        #   St1* = (-BG*(Bt1/γ*² + v*B* vt1) - fK[St1] + SK*uK[St1]) / (SK - vc)
        St1_B = _sdiv(
            -BG_idir*(uH[Bi[t1]]/gamma_star**2 + v_star_B_star*vt1)
            - fK[Si[t1]] + SK*uK[Si[t1]],
            denom,
        )
        St2_B = _sdiv(
            -BG_idir*(uH[Bi[t2]]/gamma_star**2 + v_star_B_star*vt2)
            - fK[Si[t2]] + SK*uK[Si[t2]],
            denom,
        )
        # Without B (pure advection):
        St1_0 = _sdiv(uK[Si[t1]] * (SK - vdK), denom)
        St2_0 = _sdiv(uK[Si[t2]] * (SK - vdK), denom)

        ucK[Si[t1]] = torch.where(has_B, St1_B, St1_0)
        ucK[Si[t2]] = torch.where(has_B, St2_B, St2_0)

        return ucK

    vd_L = sL[["vx","vy","vz"][idir]]
    vd_R = sR[["vx","vy","vz"][idir]]

    ucL = _star(uL, fL, S_L, vd_L)
    ucR = _star(uR, fR, S_R, vd_R)

    # ── Step 6: HLLC fluxes via Rankine-Hugoniot ─────────────────────────────
    fStarL: Cons = {k: fL[k] + S_L*(ucL[k] - uL[k]) for k in ucL}
    fStarR: Cons = {k: fR[k] + S_R*(ucR[k] - uR[k]) for k in ucR}

    # ── Step 7: Region selection (vi=0 in Minkowski) ─────────────────────────
    vi = torch.zeros_like(cmin)

    fHLLC: Cons = {}
    uHLLC: Cons = {}
    for k in keys:
        f = fH[k].clone()
        u = uH[k].clone()

        fL_k  = fL[k];  uL_k  = uL[k]
        fR_k  = fR[k];  uR_k  = uR[k]
        fsL_k = fStarL.get(k, fH[k]);  usL_k = ucL.get(k, uH[k])
        fsR_k = fStarR.get(k, fH[k]);  usR_k = ucR.get(k, uH[k])

        # C++ ordering: vi <= -cmin → L;  -cmin < vi < vc → *L;
        #               vc <= vi < cmax → *R;  vi >= cmax → R
        f = torch.where(vi <= S_L,               fL_k,  f)
        f = torch.where((S_L < vi) & (vi <  vc), fsL_k, f)
        f = torch.where((vc  <= vi) & (vi < S_R), fsR_k, f)
        f = torch.where(vi >= S_R,               fR_k,  f)

        u = torch.where(vi <= S_L,               uL_k,  u)
        u = torch.where((S_L < vi) & (vi <  vc), usL_k, u)
        u = torch.where((vc  <= vi) & (vi < S_R), usR_k, u)
        u = torch.where(vi >= S_R,               uR_k,  u)

        fHLLC[k] = f
        uHLLC[k] = u

    # ── Step 8: Degenerate / superluminal fallback (C++ explicit guard) ───────
    # C++: if (lambdaC <= -cmin || lambdaC >= cmax || v2 >= 1.0) → HLLE
    degen = (vc <= S_L) | (vc >= S_R) | (v2 >= 1.0)
    if torch.any(degen):
        for k in keys:
            fHLLC[k] = torch.where(degen, fH[k], fHLLC[k])
            uHLLC[k] = torch.where(degen, uH[k], uHLLC[k])

    # ── Step 9: NaN/Inf safety ────────────────────────────────────────────────
    for k in keys:
        bad = ~torch.isfinite(fHLLC[k]) | ~torch.isfinite(uHLLC[k])
        if torch.any(bad):
            fHLLC[k] = torch.where(bad, fH[k], fHLLC[k])
            uHLLC[k] = torch.where(bad, uH[k], uHLLC[k])

    # p_hll for interface parity with hlld_flux
    pl, _ = eos.press_and_cs2(sL["eps"], sL["rho"])
    pr, _ = eos.press_and_cs2(sR["eps"], sR["rho"])
    wl, _ = lorentz(sL); wr, _ = lorentz(sR)
    b2l   = compute_b2((sL["vx"],sL["vy"],sL["vz"]),(sL["Bx"],sL["By"],sL["Bz"]),wl)
    b2r   = compute_b2((sR["vx"],sR["vy"],sR["vz"]),(sR["Bx"],sR["By"],sR["Bz"]),wr)
    p_hll = torch.clamp(0.5*(pl + 0.5*b2l + pr + 0.5*b2r), min=1e-30)

    return fHLLC, uHLLC, p_hll

# ── ML-assisted HLLD flux ─────────────────────────────────────────────────────

def hlld_ai_flux(
    sL: Prim,
    sR: Prim,
    eos: hybrid_eos,
    model: torch.nn.Module,
    norm_stats: dict,
    idir: int = 0,
) -> Tuple[Cons, Cons, torch.Tensor]:
    """
    HLLD flux where p* is predicted by a neural network instead of
    iterative root-finding. Everything else is identical to hlld_flux.

    Parameters
    ----------
    model      : PressureNet — loaded and in eval mode
    norm_stats : dict with keys mu, sigma, y_mu, y_sigma (from norm_stats.npz)
    """
    uL, uR, fL, fR, cmax, cmin = compute_srmhd_fluxes(sL, sR, eos, idir)
    keys = list(uL.keys())

    fH: Cons = {k: _hf(fL[k], fR[k], uL[k], uR[k], cmin, cmax) for k in keys}
    uH: Cons = {k: _hu(fL[k], fR[k], uL[k], uR[k], cmin, cmax) for k in keys}

    # ── Reconstruct the same conserved/flux quantities as hlld_flux ───────────
    lfacL, _ = lorentz(sL)
    lfacR, _ = lorentz(sR)

    b2L = compute_b2((sL["vx"], sL["vy"], sL["vz"]),
                      (sL["Bx"], sL["By"], sL["Bz"]), lfacL)
    b2R = compute_b2((sR["vx"], sR["vy"], sR["vz"]),
                      (sR["Bx"], sR["By"], sR["Bz"]), lfacR)

    press_L, _ = eos.press_and_cs2(sL["eps"], sL["rho"])
    press_R, _ = eos.press_and_cs2(sR["eps"], sR["rho"])

    vl = (sL["vx"], sL["vy"], sL["vz"])
    vr = (sR["vx"], sR["vy"], sR["vz"])
    Bl = (sL["Bx"], sL["By"], sL["Bz"])
    Br = (sR["Bx"], sR["By"], sR["Bz"])

    sbL = compute_smallb(vl, Bl, lfacL)
    sbR = compute_smallb(vr, Br, lfacR)

    wL = sL["rho"] * (1.0 + sL["eps"]) + press_L + b2L
    wR = sR["rho"] * (1.0 + sR["eps"]) + press_R + b2R

    U_init_L: Cons = {
        "D":   uL["D"],
        "Sx":  wL * lfacL**2 * sL["vx"] - sbL[0] * sbL[1],
        "Sy":  wL * lfacL**2 * sL["vy"] - sbL[0] * sbL[2],
        "Sz":  wL * lfacL**2 * sL["vz"] - sbL[0] * sbL[3],
        "tau": wL * lfacL**2 - press_L - 0.5 * b2L - sbL[0]**2,
        "Bx":  sL["Bx"], "By": sL["By"], "Bz": sL["Bz"],
    }
    U_init_R: Cons = {
        "D":   uR["D"],
        "Sx":  wR * lfacR**2 * sR["vx"] - sbR[0] * sbR[1],
        "Sy":  wR * lfacR**2 * sR["vy"] - sbR[0] * sbR[2],
        "Sz":  wR * lfacR**2 * sR["vz"] - sbR[0] * sbR[3],
        "tau": wR * lfacR**2 - press_R - 0.5 * b2R - sbR[0]**2,
        "Bx":  sR["Bx"], "By": sR["By"], "Bz": sR["Bz"],
    }

    F_init_L: Cons = {k: fL[k] for k in keys}
    F_init_R: Cons = {k: fR[k] for k in keys}
    F_init_L.update({
        "Sx":  wL * lfacL**2 * sL["vx"]**2 + press_L + 0.5*b2L - sbL[1]*sbL[1],
        "Sy":  wL * lfacL**2 * sL["vx"] * sL["vy"] - sbL[1]*sbL[2],
        "Sz":  wL * lfacL**2 * sL["vx"] * sL["vz"] - sbL[1]*sbL[3],
        "tau": wL * lfacL**2 * sL["vx"] - sbL[1]*sbL[0],
    })
    F_init_R.update({
        "Sx":  wR * lfacR**2 * sR["vx"]**2 + press_R + 0.5*b2R - sbR[1]*sbR[1],
        "Sy":  wR * lfacR**2 * sR["vx"] * sR["vy"] - sbR[1]*sbR[2],
        "Sz":  wR * lfacR**2 * sR["vx"] * sR["vz"] - sbR[1]*sbR[3],
        "tau": wR * lfacR**2 * sR["vx"] - sbR[1]*sbR[0],
    })

    # ── ML prediction of p* ───────────────────────────────────────────────────
    # Build the 15-feature input vector:
    # [rho_L, vx_L, vy_L, vz_L, p_L, By_L, Bz_L,
    #  rho_R, vx_R, vy_R, vz_R, p_R, By_R, Bz_R, Bx]
    def _stack(*tensors):
        return torch.stack([t.flatten() for t in tensors], dim=1)

    X_raw = _stack(
        sL["rho"], sL["vx"], sL["vy"], sL["vz"], press_L, sL["By"], sL["Bz"],
        sL["rho"], sR["vx"], sR["vy"], sR["vz"], press_R, sR["By"], sR["Bz"],
        sL["Bx"],
    )  # shape (N, 15)

    # Normalize using training stats
    mu    = torch.tensor(norm_stats["mu"],    dtype=X_raw.dtype, device=X_raw.device)
    sigma = torch.tensor(norm_stats["sigma"], dtype=X_raw.dtype, device=X_raw.device)
    X_norm = (X_raw - mu) / sigma
    X_norm = X_norm.float()

    with torch.no_grad():
        y_norm_pred = model(X_norm).squeeze(-1)   # (N,)

    # Inverse-transform: y_norm → log10(p*) → p*
    y_mu    = float(norm_stats["y_mu"])
    y_sigma = float(norm_stats["y_sigma"])
    p_star  = 10.0 ** (y_norm_pred * y_sigma + y_mu)
    p_star  = torch.clamp(p_star, min=1e-30)

    # ── Everything below is identical to hlld_flux ────────────────────────────
    BnL = _t(uL["Bx"]); BnR = _t(uR["Bx"])
    Bn2 = 0.5 * (BnL**2 + BnR**2)

    calc = HLLDComputation(F_init_L, F_init_R, U_init_L, U_init_R, sL, sR, cmin, cmax, idir)
    calc.compute_all_variables(p_star)

    i  = idir; t1 = calc.t1; t2 = calc.t2
    S  = calc._S; B  = calc._B
    BnL = _t(uL[B[i]]); BnR = _t(uR[B[i]])

    uaL: Cons = {
        B[i]: BnL, B[t1]: calc.Bt1aL, B[t2]: calc.Bt2aL,
        "D": calc.DaL, "tau": calc.TauaL,
        S[i]: calc.SaL, S[t1]: calc.St1aL, S[t2]: calc.St2aL,
    }
    uaR: Cons = {
        B[i]: BnR, B[t1]: calc.Bt1aR, B[t2]: calc.Bt2aR,
        "D": calc.DaR, "tau": calc.TauaR,
        S[i]: calc.SaR, S[t1]: calc.St1aR, S[t2]: calc.St2aR,
    }

    faL: Cons = {k: fL[k] + (-cmin) * (uaL[k] - uL[k]) for k in keys}
    faR: Cons = {k: fR[k] + ( cmax) * (uaR[k] - uR[k]) for k in keys}

    fac  = _sdiv(1.0 - calc.KaL2, -calc.S_L * calc.sqL - calc.KaLBc)
    vc   = calc.laL   - calc.Bic  * fac
    vt1  = calc.Kt1aL - calc.Bt1c * fac
    vt2  = calc.Kt2aL - calc.Bt2c * fac

    lC  = vc
    laL = calc.laL; laR = calc.laR

    ucL: Cons = {B[i]: calc.Bic, B[t1]: calc.Bt1c, B[t2]: calc.Bt2c}
    ucR: Cons = {B[i]: calc.Bic, B[t1]: calc.Bt1c, B[t2]: calc.Bt2c}

    vBc = vc * ucL[B[i]] + vt1 * ucL[B[t1]] + vt2 * ucL[B[t2]]

    ucL["D"]   = uaL["D"] * _sdiv(laL - calc.vaL, laL - vc)
    ucR["D"]   = uaR["D"] * _sdiv(laR - calc.vaR, laR - vc)
    ucL["tau"] = _sdiv(laL * uaL["tau"] - faL["tau"] + p_star * vc - vBc * ucL[B[i]], laL - vc)
    ucR["tau"] = _sdiv(laR * uaR["tau"] - faR["tau"] + p_star * vc - vBc * ucR[B[i]], laR - vc)

    for key, vi, Bi in zip([S[i], S[t1], S[t2]], [vc, vt1, vt2],
                            [ucL[B[i]], ucL[B[t1]], ucL[B[t2]]]):
        ucL[key] = (ucL["tau"] + p_star + ucL["D"]) * vi - vBc * Bi
    for key, vi, Bi in zip([S[i], S[t1], S[t2]], [vc, vt1, vt2],
                            [ucR[B[i]], ucR[B[t1]], ucR[B[t2]]]):
        ucR[key] = (ucR["tau"] + p_star + ucR["D"]) * vi - vBc * Bi

    vi = torch.zeros_like(cmin)
    cL_mask  = vi <= -cmin
    caL_mask = (-cmin < vi) & (vi <= laL)
    ccL_mask = (laL < vi) & (vi < lC)
    ccR_mask = (lC <= vi) & (vi < laR)
    caR_mask = (laR <= vi) & (vi < cmax)
    cR_mask  = vi >= cmax

    fD: Cons = {k: fH[k].clone() for k in keys}
    uD: Cons = {k: uH[k].clone() for k in keys}

    for k in keys:
        fcL = faL[k] + laL * (ucL[k] - uaL[k])
        fcR = faR[k] + laR * (ucR[k] - uaR[k])
        f = fD[k]
        f = torch.where(cL_mask,  fL[k],  f)
        f = torch.where(caL_mask, faL[k], f)
        f = torch.where(ccL_mask, fcL,    f)
        f = torch.where(ccR_mask, fcR,    f)
        f = torch.where(caR_mask, faR[k], f)
        f = torch.where(cR_mask,  fR[k],  f)
        fD[k] = f

        u = uD[k]
        u = torch.where(cL_mask,  uL[k],  u)
        u = torch.where(caL_mask, uaL[k], u)
        u = torch.where(ccL_mask, ucL[k], u)
        u = torch.where(ccR_mask, ucR[k], u)
        u = torch.where(caR_mask, uaR[k], u)
        u = torch.where(cR_mask,  uR[k],  u)
        uD[k] = u

    wave_order_bad = (
        (calc.vaR > cmax) | (calc.vaL < -cmin) |
        (vc < calc.laL)   | (vc > calc.laR)
    )
    failed = wave_order_bad
    if torch.any(failed):
        p_hll_fb = torch.clamp(0.5 * (press_L + 0.5*b2L + press_R + 0.5*b2R), min=1e-30)
        p_star = torch.where(failed, -torch.abs(p_hll_fb) - 1e-10, p_star)
        for k in keys:
            fD[k] = torch.where(failed, fH[k], fD[k])
            uD[k] = torch.where(failed, uH[k], uD[k])

    return fD, uD, p_star