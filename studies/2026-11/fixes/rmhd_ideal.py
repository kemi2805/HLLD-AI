"""Ideal special-relativistic MHD in one dimension (x), written from the
equations and nothing else -- deliberately independent of the solver code it
is used to check (rmhd_final / HLLD).

Primitive state, as the solver frame stores it: (rho, P_tot, vx, vy, vz, By, Bz)
with the normal field Bx constant and the gas pressure p = P_tot - b^2/2.
Ideal gas, p = (gamma - 1) rho eps, specific enthalpy h = 1 + gamma p / ((gamma - 1) rho).

    W   = 1 / sqrt(1 - v^2)
    b^0 = W (v.B),  b^i = B^i / W + b^0 v^i,  b^2 = B^2 / W^2 + (v.B)^2
    D   = rho W
    S   = (rho h W^2 + B^2) v - (v.B) B
    tau = rho h W^2 - p + B^2/2 + (v^2 B^2 - (v.B)^2)/2 - D
    F(D)   = D vx
    F(S_j) = S_j vx - Bx (B_j / W^2 + (v.B) v_j) + p_tot delta_jx
    F(tau) = S_x - D vx
    F(By)  = By vx - Bx vy,   F(Bz) = Bz vx - Bx vz
"""
import numpy as np


def prim(state, Bx, gamma):
    rho, Pt, vx, vy, vz, By, Bz = (np.asarray(c, dtype=float) for c in state)
    v2 = vx * vx + vy * vy + vz * vz
    W2 = 1.0 / (1.0 - v2)
    vB = vx * Bx + vy * By + vz * Bz
    B2 = Bx * Bx + By * By + Bz * Bz
    b2 = B2 / W2 + vB * vB
    p = Pt - 0.5 * b2
    return rho, p, vx, vy, vz, By, Bz, v2, W2, vB, B2, b2


def cons_flux(state, Bx, gamma):
    """(U, F), each (7, N): D, Sx, Sy, Sz, tau, By, Bz."""
    rho, p, vx, vy, vz, By, Bz, v2, W2, vB, B2, b2 = prim(state, Bx, gamma)
    h = 1.0 + gamma * p / ((gamma - 1.0) * rho)
    W = np.sqrt(W2)
    D = rho * W
    e = rho * h * W2 + B2
    Sx, Sy, Sz = e * vx - vB * Bx, e * vy - vB * By, e * vz - vB * Bz
    tau = rho * h * W2 - p + 0.5 * B2 + 0.5 * (v2 * B2 - vB * vB) - D
    pt = p + 0.5 * b2
    U = np.stack([D, Sx, Sy, Sz, tau, By, Bz])
    F = np.stack([D * vx,
                  Sx * vx - Bx * (Bx / W2 + vB * vx) + pt,
                  Sy * vx - Bx * (By / W2 + vB * vy),
                  Sz * vx - Bx * (Bz / W2 + vB * vz),
                  Sx - D * vx,
                  By * vx - Bx * vy,
                  Bz * vx - Bx * vz])
    return U, F


def entropy(state, Bx, gamma):
    """p / rho^gamma: rest-frame quantities, monotone in the specific entropy."""
    rho, p = prim(state, Bx, gamma)[:2]
    return p / rho ** gamma


def rh_residual(ahead, behind, V, Bx, gamma):
    """max_k |[[F_k]] - V [[U_k]]| / scale_k, the scale of each component its
    own flux and density magnitudes with a floor of 1e-3 x the largest scale
    (an out-of-plane component whose fluxes vanish is measured against the
    others, not against its own round-off)."""
    Ua, Fa = cons_flux(ahead, Bx, gamma)
    Ub, Fb = cons_flux(behind, Bx, gamma)
    num = np.abs((Fb - Fa) - V * (Ub - Ua))
    sc = np.abs(Fa) + np.abs(Fb) + np.abs(V) * (np.abs(Ua) + np.abs(Ub))
    floor = 1e-3 * sc.max(axis=0)
    return np.max(num / (sc + floor + 1e-300), axis=0)
