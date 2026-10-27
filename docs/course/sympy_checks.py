"""Symbolic and numerical checks of the identities used in the course
"The Riemann Problem in Special-Relativistic Magnetohydrodynamics".

Run with any Python that has sympy and mpmath, e.g.

    /Users/miler/Codes/py-work/bin/python sympy_checks.py

Every check prints OK or raises AssertionError.  Conventions as in the
course: c = 1, signature (-,+,+,+), magnetic field with sqrt(4 pi) absorbed,
ideal gas p = (Gamma - 1) rho eps.
"""
import sympy as sp
import mpmath as mp

mp.mp.dps = 40

# ---------------------------------------------------------------- helpers
vx, vy, vz = sp.symbols('v_x v_y v_z', real=True)
Bx, By, Bz = sp.symbols('B_x B_y B_z', real=True)
rho, p, eps, Gam = sp.symbols('rho p epsilon Gamma', positive=True)
v = sp.Matrix([vx, vy, vz])
B = sp.Matrix([Bx, By, Bz])
v2 = (v.T * v)[0]
W = 1 / sp.sqrt(1 - v2)
vB = (v.T * B)[0]
eta_m = sp.diag(-1, 1, 1, 1)          # Minkowski metric, also its inverse


def ok(name, expr, subs=None):
    e = sp.simplify(expr if subs is None else expr.subs(subs))
    assert e == 0, (name, e)
    print(f"OK  {name}")


def lower(vec):
    return eta_m * vec


def dot(a, b):
    return (a.T * eta_m * b)[0]


# ------------------------------------------------- 1. b^mu from the dual tensor
# *F^{0i} = B^i, *F^{ij} = -eps^{ijk} E_k with E = -v x B (ideal MHD)
E = -v.cross(B)
Fd = sp.zeros(4, 4)
for i in range(3):
    Fd[0, i + 1] = B[i]
    Fd[i + 1, 0] = -B[i]
for i in range(3):
    for j in range(3):
        Fd[i + 1, j + 1] = -sum(sp.LeviCivita(i, j, k) * E[k] for k in range(3))
u = sp.Matrix([W, W * vx, W * vy, W * vz])
b = Fd * lower(u)                      # b^mu = *F^{mu nu} u_nu
b0_expect = W * vB
bi_expect = B / W + b0_expect * v
ok("b^0 = W (v.B)", b[0] - b0_expect)
for i in range(3):
    ok(f"b^{i+1} = B^i/W + b^0 v^i", b[i + 1] - bi_expect[i])
ok("u.b = 0", dot(u, b))
ok("u.u = -1", dot(u, u) + 1)
b2 = dot(b, b)
ok("b^2 = B^2/W^2 + (v.B)^2", b2 - ((B.T * B)[0] / W**2 + vB**2))
# *F^{mu nu} = u^mu b^nu - u^nu b^mu
for m in range(4):
    for n in range(4):
        ok(f"*F^{m}{n} = u^m b^n - u^n b^m",
           Fd[m, n] - (u[m] * b[n] - u[n] * b[m])) if (m, n) in [(0, 1), (1, 2), (2, 3)] else None
# induction flux: u^i b^j - u^j b^i = v^i B^j - v^j B^i
ok("x-flux of B^y = v_x B_y - v_y B_x", (u[1] * b[2] - u[2] * b[1]) - (vx * By - vy * Bx))

# -------------------------------------------- 2. conserved variables and fluxes
h = 1 + eps + p / rho                  # gas specific enthalpy
w = rho * h + b2                       # total enthalpy density
Ptot = p + b2 / 2
T = sp.zeros(4, 4)
for m in range(4):
    for n in range(4):
        T[m, n] = w * u[m] * u[n] + Ptot * eta_m[m, n] - b[m] * b[n]
D = rho * W
S_code = (rho * h * W**2 + (B.T * B)[0]) * v - vB * B
tau_code = rho * h * W**2 + (B.T * B)[0] - p - ((B.T * B)[0] / W**2 + vB**2) / 2 - D
tau_hlld = rho * h * W**2 - p - D + sp.Rational(1, 2) * ((B.T * B)[0]
                                                        + (B.T * B)[0] * v2 - vB**2)
for i in range(3):
    ok(f"T^0{i+1} = S_{i+1} = (rho h W^2 + B^2) v - (v.B) B", T[0, i + 1] - S_code[i])
ok("T^00 - D = tau (course form)", T[0, 0] - D - tau_code)
ok("tau (course form) = tau (hlld.py primitive_to_conserved)", tau_code - tau_hlld)
ok("tau = (rho h + b^2) W^2 - (p + b^2/2) - (b^0)^2 - D  (hlld.py fluxes)",
   tau_code - (w * W**2 - Ptot - b[0]**2 - D))
# x-fluxes as in hlld.py compute_srmhd_fluxes / jump.cons_flux
for i in range(3):
    F_code = w * W * vx * u[i + 1] + Ptot * (1 if i == 0 else 0) - b[1] * b[i + 1]
    ok(f"T^x{i+1} = momentum flux (code)", T[1, i + 1] - F_code)
ok("T^0x - D v_x = tau flux (code)", (T[0, 1] - D * vx) - (w * W**2 * vx - b[0] * b[1] - D * vx))
# Mignone & Bodo (2006) eq. (13) form of the momentum flux
mx = S_code[0]
ok("T^xx = m_x v_x - B_x b^x/W + p_tot  (MB06 eq. 13)", T[1, 1] - (mx * vx - Bx * b[1] / W + Ptot))

# ------------------------------- 2b. the remaining identities of Appendix A (app:bmu)
B2s = (B.T * B)[0]
ok("b^0 = v.b", b[0] - sum(v[i] * b[i + 1] for i in range(3)))
ok("W^2 b^2 - (b^0)^2 = B^2", W**2 * b2 - b[0]**2 - B2s)
for i in range(3):
    ok(f"B^{i+1} = u^0 b^{i+1} - u^{i+1} b^0", B[i] - (u[0] * b[i + 1] - u[i + 1] * b[0]))
    ok(f"B^{i+1} = W (b^{i+1} - b^0 v^{i+1})", B[i] - W * (b[i + 1] - b[0] * v[i]))
ExB = E.cross(B)
E2s = (E.T * E)[0]
ok("|E|^2 = v^2 B^2 - (v.B)^2", E2s - (v2 * B2s - vB**2))
ok("T^00 = rho h W^2 + B^2 - p - b^2/2", T[0, 0] - (rho * h * W**2 + B2s - p - b2 / 2))
ok("T^00 = rho h W^2 - p + (|E|^2 + B^2)/2", T[0, 0] - (rho * h * W**2 - p + (E2s + B2s) / 2))
for i in range(3):
    ok(f"T^0{i+1} = rho h W^2 v^{i+1} + (E x B)^{i+1}", T[0, i + 1] - (rho * h * W**2 * v[i] + ExB[i]))
for i in range(3):
    for j in range(3):
        dij = 1 if i == j else 0
        ok(f"T^{i+1}{j+1} = (rho h W^2 + B^2) v^i v^j - (v.B)(B^i v^j + B^j v^i) - B^i B^j/W^2 + p_tot delta",
           T[i + 1, j + 1] - ((rho * h * W**2 + B2s) * v[i] * v[j] - vB * (B[i] * v[j] + B[j] * v[i])
                              - B[i] * B[j] / W**2 + Ptot * dij))
        ok(f"T^{i+1}{j+1} = S^j v^i + p_tot delta - B^i b^j/W",
           T[i + 1, j + 1] - (S_code[j] * v[i] + Ptot * dij - B[i] * b[j + 1] / W))
rest = {vx: 0, vy: 0, vz: 0}
ok("rest frame: b^0 = 0", b[0].subs(rest))
for i in range(3):
    ok(f"rest frame: b^{i+1} = B^{i+1}", (b[i + 1] - B[i]).subs(rest))
    ok(f"rest frame: T^0{i+1} = 0", T[0, i + 1].subs(rest))
ok("rest frame: T^00 = e + b^2/2 with e = rho (1 + eps)", (T[0, 0] - (rho * (1 + eps) + b2 / 2)).subs(rest))
for i in range(3):
    for j in range(3):
        ok(f"rest frame: T^{i+1}{j+1} = (p + b^2/2) delta - b^i b^j",
           (T[i + 1, j + 1] - ((p + b2 / 2) * (1 if i == j else 0) - b[i + 1] * b[j + 1])).subs(rest))
ok("sigma = b^2/(rho h) = c_a^2/(1 - c_a^2) with c_a^2 = b^2/w", b2 / (rho * h) - (b2 / w) / (1 - b2 / w))

# --------------------------------------------------------- 3. Newtonian limit
d = sp.symbols('delta', positive=True)
sc = {vx: d * vx, vy: d * vy, vz: d * vz, Bx: d * Bx, By: d * By, Bz: d * Bz,
      p: d**2 * p, eps: d**2 * eps}
tau_N = sp.series(tau_code.subs(sc, simultaneous=True), d, 0, 3).removeO()
ok("tau -> rho eps + rho v^2/2 + B^2/2 (order delta^2)",
   tau_N - d**2 * (rho * eps + rho * v2 / 2 + (B.T * B)[0] / 2))
Sx_N = sp.series(S_code[0].subs(sc, simultaneous=True), d, 0, 2).removeO()
ok("S_x -> rho v_x (order delta)", Sx_N - d * rho * vx)
Txx_N = sp.series(T[1, 1].subs(sc, simultaneous=True), d, 0, 3).removeO()
ok("T^xx -> rho v_x^2 + p + B^2/2 - B_x^2 (order delta^2)",
   Txx_N - d**2 * (rho * vx**2 + p + (B.T * B)[0] / 2 - Bx**2))
Fe = (T[0, 1] - D * vx).subs(sc, simultaneous=True)
Fe_N = sp.series(Fe, d, 0, 4).removeO()
ok("energy flux -> (rho v^2/2 + rho eps + p + B^2) v_x - (v.B) B_x (order delta^3)",
   Fe_N - d**3 * ((rho * v2 / 2 + rho * eps + p + (B.T * B)[0]) * vx - vB * Bx))

# --------------------------------------------- 4. characteristic speeds (lab)
lam = sp.symbols('lambda', real=True)
cs2 = sp.symbols('c_s2', positive=True)       # c_s^2, kept independent
wt = sp.symbols('w', positive=True)           # rho h + b^2 (total enthalpy)
rh = wt - b2                                  # rho h
a_ = W * (lam - vx)
calB = b[1] - lam * b[0]                      # b^x - lambda b^0
quartic_course = rh * (1 - cs2) * a_**4 - (1 - lam**2) * ((b2 + rh * cs2) * a_**2 - cs2 * calB**2)
# the code's normalised coefficients (rmhd/wave_speeds.py, GR06 eq. 4.50)
bt0 = W * vB / sp.sqrt(wt)
bt1 = (Bx / W + W * vB * vx) / sp.sqrt(wt)
bt2 = b2 / wt
e2 = cs2 + bt2 - cs2 * bt2
W2, W4 = W**2, W**4
a0 = bt1**2 * cs2 - e2 * W2 * vx**2 - (e2 - 1) * W4 * vx**4
a1 = -2 * bt0 * bt1 * cs2 + 2 * e2 * W2 * vx + 4 * (e2 - 1) * W4 * vx**3
a2 = (bt0**2 - bt1**2) * cs2 + W2 * (6 * W2 * vx**2 + e2 * (-1 + (1 - 6 * W2) * vx**2))
a3 = 2 * bt0 * bt1 * cs2 + 2 * W2 * (-e2 + 2 * (-1 + e2) * W2) * vx
a4 = -bt0**2 * cs2 + W4 + e2 * (W2 - W4)
code_poly = a0 + a1 * lam + a2 * lam**2 + a3 * lam**3 + a4 * lam**4
ok("code quartic == course quartic / w", sp.expand(code_poly - quartic_course / wt))
# DZBL03 eq. (16) form
dz = (1 - e2) * (W * lam - W * vx)**4 + (1 - lam**2) * (cs2 * (bt0 * lam - bt1)**2 - e2 * (W * lam - W * vx)**2)
ok("DZBL03 eq. 16 == course quartic / w", sp.expand(dz - quartic_course / wt))

# Alfven speeds: (b^x +- sqrt(w) u^x)/(b^0 +- sqrt(w) u^0) = v_x + B_x (1-v^2)/(eta +- sqrt(w))
sw = sp.sqrt(wt)
for sgn in (+1, -1):
    lamA = (b[1] + sgn * sw * u[1]) / (b[0] + sgn * sw * u[0])
    ok(f"Alfven speed closed form, sign {sgn:+d}",
       lamA - (vx + Bx * (1 - v2) / (vB + sgn * sw)))
    # it is a root of A = w a^2 - calB^2 (Anton et al. 2010, eq. 34)
    ok(f"Alfven speed solves w a^2 = B^2, sign {sgn:+d}",
       (wt * a_**2 - calB**2).subs(lam, lamA))
    # sign-resolved label: the (+/-) of (3.14) is calB +/- sqrt(w) W (v_x - lambda) = 0
    ok(f"Alfven label: calB + sgn sqrt(w) W(v_x - lam) = 0, sign {sgn:+d}",
       (calB - sgn * sw * a_).subs(lam, lamA))

# rest frame, v = 0:  lambda^4 - Q lambda^2 + cs2 ca2 cos^2 = 0
th = sp.symbols('theta', real=True)
ca2 = sp.symbols('c_a2', positive=True)
Bmag = sp.symbols('B', positive=True)
rest = quartic_course.subs({vx: 0, vy: 0, vz: 0}).subs(
    {Bx: Bmag * sp.cos(th), By: Bmag * sp.sin(th), Bz: 0})
rest = sp.simplify(rest / wt).subs(Bmag**2, ca2 * wt)
Q = ca2 + cs2 - cs2 * ca2 * sp.sin(th)**2
ok("rest-frame quartic", sp.simplify(sp.expand(rest - (lam**4 - Q * lam**2 + cs2 * ca2 * sp.cos(th)**2))))
X = sp.symbols('X')
P = X**2 - Q * X + cs2 * ca2 * sp.cos(th)**2
ok("P(c_a^2 cos^2) = -c_a^4 cos^2 sin^2 (1-c_s^2)",
   P.subs(X, ca2 * sp.cos(th)**2) + ca2**2 * sp.cos(th)**2 * sp.sin(th)**2 * (1 - cs2))
ok("P(1) = (1-c_a^2)(1-c_s^2)", P.subs(X, 1) - (1 - ca2) * (1 - cs2))
eps2 = ca2 + cs2 - ca2 * cs2
ok("P(eps^2) = c_s^2 c_a^2 cos^2 (1-eps^2)",
   P.subs(X, eps2) - cs2 * ca2 * sp.cos(th)**2 * (1 - eps2))
ok("Q - eps^2 = c_s^2 c_a^2 cos^2", Q - eps2 - cs2 * ca2 * sp.cos(th)**2)

# linearised equations in the fluid rest frame -> dispersion relation
V_, hh_, bb_ = sp.symbols('V h b', positive=True)
w_r = rho * hh_ + bb_**2
bvec = sp.Matrix([bb_ * sp.cos(th), bb_ * sp.sin(th), 0])
drho, dvx, dvy, dvz, dby, dbz, dp_ = sp.symbols('drho dvx dvy dvz dby dbz dp')
dv_ = sp.Matrix([dvx, dvy, dvz])
db_ = sp.Matrix([0, dby, dbz])                    # k.db = 0 (and the induction x-row gives db_x = 0)
bdv = (bvec.T * dv_)[0]
dP_ = dp_ + (bvec.T * db_)[0]
lin = [-V_ * drho + rho * dvx]                                            # continuity
lin += list(-V_ * (w_r * dv_ - bdv * bvec) + sp.Matrix([dP_, 0, 0]) - bvec[0] * db_)   # momentum
ind_ = -V_ * db_ - (dv_ * bvec[0] - bvec * dvx)                           # induction
lin += [ind_[1], ind_[2], dp_ - hh_ * cs2 * drho]                        # + adiabatic closure
Mlin = sp.Matrix([[sp.diff(e_, x_) for x_ in (drho, dvx, dvy, dvz, dby, dbz, dp_)] for e_ in lin])
ca2_r = bb_**2 / w_r
Q_r = ca2_r + cs2 - cs2 * ca2_r * sp.sin(th)**2
ok("rest-frame linear system: det = rho h w^2 (V^2 - c_a^2 cos^2)(V^4 - Q V^2 + c_s^2 c_a^2 cos^2)",
   sp.simplify(Mlin.det() - rho * hh_ * w_r**2 * (V_**2 - ca2_r * sp.cos(th)**2)
               * (V_**4 - Q_r * V_**2 + cs2 * ca2_r * sp.cos(th)**2)))

# the 2x2 in-plane system written out in the notes (chapter 3)
rh_s = sp.symbols('rh', positive=True)
sn, cn = sp.sin(th), sp.cos(th)
det2 = (((rh_s * cs2 + bb_**2 * sn**2) - V_**2 * (rh_s + bb_**2 * sn**2))
        * (bb_**2 * cn**2 - V_**2 * (rh_s + bb_**2 * cn**2)) - (1 - V_**2)**2 * bb_**4 * sn**2 * cn**2)
ok("in-plane 2x2 determinant = rho h [w V^4 - (b^2 + rho h c_s^2 + c_s^2 b^2 cos^2) V^2 + c_s^2 b^2 cos^2]",
   sp.expand(det2 - rh_s * ((rh_s + bb_**2) * V_**4 - (bb_**2 + rh_s * cs2 + cs2 * bb_**2 * cn**2) * V_**2
                            + cs2 * bb_**2 * cn**2)))

# covariantisation: V'^2 = a^2/(G + a^2), cos^2 theta' = calB^2/(b^2 (G + a^2))
aa, GG, cBB, bsq = sp.symbols('a G calB b2', positive=True)
ca2_s = sp.symbols('c_a2s', positive=True)
Vp2 = aa**2 / (GG + aa**2)
cth2 = cBB**2 / (bsq * (GG + aa**2))
restq = Vp2**2 - (ca2_s + cs2 - cs2 * ca2_s * (1 - cth2)) * Vp2 + cs2 * ca2_s * cth2
eps2_s = ca2_s + cs2 - ca2_s * cs2
labq = (1 - eps2_s) * aa**4 - GG * (eps2_s * aa**2 - cs2 * ca2_s * cBB**2 / bsq)
ok("rest-frame quartic x (G + a^2)^2 == lab quartic / w  (with c_a^2/b^2 = 1/w)",
   sp.simplify(restq * (GG + aa**2)**2 - labq))

# fast-speed bound: W^2 (1-z^2)(lambda - v_x)^2 = z^2 (1 - lambda^2), closed-form roots
z, vxs, vts = sp.symbols('zeta v_x v_t', positive=True)
vsq = vxs**2 + vts**2
Wb2 = 1 / (1 - vsq)
for sgn in (+1, -1):
    lam_b = (vxs * (1 - z**2) + sgn * z * sp.sqrt((1 - vsq) * (1 - vsq * z**2 - vxs**2 * (1 - z**2)))) / (1 - vsq * z**2)
    ok(f"isotropic-bound root, sign {sgn:+d}",
       sp.simplify(sp.expand(Wb2 * (1 - z**2) * (lam_b - vxs)**2 - z**2 * (1 - lam_b**2))))
# monotonicity of the rest-frame phase speed along lambda: d/dlambda [G/a^2]
Rfun = (1 - lam**2) / (Wb2 * (vxs - lam)**2)
ok("d/dlambda (G/a^2) = 2 (1 - lambda v_x)/(W^2 (v_x - lambda)^3)",
   sp.simplify(sp.diff(Rfun, lam) - 2 * (1 - lam * vxs) / (Wb2 * (vxs - lam)**3)))
# B_x = 0: the quartic factorises as a^2 x quadratic (Mignone & Bodo 2006, eq. 58)
qa = sp.expand(quartic_course.subs(Bx, 0))
Qmb = b2.subs(Bx, 0) - cs2 * vB.subs(Bx, 0)**2
quad = rh.subs(Bx, 0) * (1 - cs2) * a_**2 - (1 - lam**2) * (rh.subs(Bx, 0) * cs2 + Qmb)
ok("B_x = 0: quartic = a^2 x [rho h (1-c_s^2) a^2 - (1-lambda^2)(rho h c_s^2 + Q)]",
   sp.simplify(qa - a_**2 * quad))

# ---------------------------------------- primitive recovery: one equation in Z
# (identities checked exactly at a rational state; symbolic simplification of the
#  general 3-vector expressions is slow and adds nothing)
num_state = {rho: sp.Rational(13, 10), p: sp.Rational(7, 10), eps: sp.Rational(7, 10) / (sp.Rational(2, 3) * sp.Rational(13, 10)),
             vx: sp.Rational(31, 100), vy: sp.Rational(-22, 100), vz: sp.Rational(17, 100),
             Bx: sp.Rational(9, 10), By: sp.Rational(6, 10), Bz: sp.Rational(-45, 100), Gam: sp.Rational(5, 3)}
Z = rho * h * W**2
Bsq = (B.T * B)[0]
Etot = tau_code + D
SB = (S_code.T * B)[0]
S2 = (S_code.T * S_code)[0]
ok("S.B = rho h W^2 (v.B)", (SB - Z * vB).subs(num_state))
ok("v^2 = [S^2 Z^2 + (S.B)^2 (2Z + B^2)] / [Z^2 (Z + B^2)^2]",
   (v2 - (S2 * Z**2 + SB**2 * (2 * Z + Bsq)) / (Z**2 * (Z + Bsq)**2)).subs(num_state))
ok("E = Z - p + (1 - 1/(2W^2)) B^2 - (S.B)^2/(2 Z^2)  (Mignone & Bodo 2006, eq. 16)",
   (Etot - (Z - p + (1 - 1 / (2 * W**2)) * Bsq - SB**2 / (2 * Z**2))).subs(num_state))
ok("ideal gas (p = (Gamma-1) rho eps): p = (Z - D W)/(Gamma/(Gamma-1) W^2)",
   (p - (Z - rho * W * W) / (Gam / (Gam - 1) * W**2)).subs(num_state))

# --------------------- 1D relativistic hydro: eigenstructure in primitive variables
r_, vv_, pp_, vt_ = sp.symbols('rho v p v_t', positive=True)
hh_g = 1 + Gam * pp_ / ((Gam - 1) * r_)
csh = sp.sqrt(Gam * pp_ / (r_ * hh_g))


def jac_eig(Uv, Fv, Vv, subs):
    """Numerical eigen-decomposition of A = (dU/dV)^-1 dF/dV at a state."""
    JU = Uv.jacobian(Vv).subs(subs)
    JF = Fv.jacobian(Vv).subs(subs)
    A_ = mp.matrix([[mp.mpf(sp.N(JU[i, j], 40)) for j in range(JU.shape[1])] for i in range(JU.shape[0])])
    B_ = mp.matrix([[mp.mpf(sp.N(JF[i, j], 40)) for j in range(JF.shape[1])] for i in range(JF.shape[0])])
    return mp.eig(mp.inverse(A_) * B_)


Ww = 1 / sp.sqrt(1 - vv_**2)
Uh = sp.Matrix([r_ * Ww, r_ * hh_g * Ww**2 * vv_, r_ * hh_g * Ww**2 - pp_])
Fh = sp.Matrix([r_ * Ww * vv_, r_ * hh_g * Ww**2 * vv_**2 + pp_, r_ * hh_g * Ww**2 * vv_])
numer = {r_: sp.Rational(7, 5), vv_: sp.Rational(3, 10), pp_: sp.Rational(9, 10), Gam: sp.Rational(5, 3)}
E_, ER_ = jac_eig(Uh, Fh, sp.Matrix([r_, vv_, pp_]), numer)
cs_n = mp.mpf(sp.N(csh.subs(numer), 40))
v_n = mp.mpf(3) / 10
W2_n = 1 / (1 - v_n**2)
rh_n = mp.mpf(sp.N((r_ * hh_g).subs(numer), 40))
for k in range(3):
    lamk = E_[k]
    if abs(lamk - v_n) < 1e-30:
        continue
    sgn = 1 if lamk > v_n else -1
    assert abs(lamk - (v_n + sgn * cs_n) / (1 + sgn * v_n * cs_n)) < 1e-30
    dvdp = ER_[1, k] / ER_[2, k]
    assert abs(dvdp - sgn / (W2_n * rh_n * cs_n)) < 1e-30
    dpdrho = ER_[2, k] / ER_[0, k]                      # dp = h c_s^2 drho
    assert abs(dpdrho - (rh_n / mp.mpf(sp.N(r_.subs(numer), 40))) * cs_n**2) < 1e-30
print("OK  1D hydro: eigenvalues v, (v -/+ c_s)/(1 -/+ v c_s); along the -/+ family dv/dp = -/+ 1/(W^2 rho h c_s) "
      "and dp = h c_s^2 drho")

# ------ tangential velocity (Pons et al. 2000): rarefaction ODE in the normal velocity
Wt = 1 / sp.sqrt(1 - vv_**2 - vt_**2)
Ut = sp.Matrix([r_ * Wt, r_ * hh_g * Wt**2 * vv_, r_ * hh_g * Wt**2 * vt_, r_ * hh_g * Wt**2 - pp_])
Ft = sp.Matrix([r_ * Wt * vv_, r_ * hh_g * Wt**2 * vv_**2 + pp_, r_ * hh_g * Wt**2 * vv_ * vt_,
                r_ * hh_g * Wt**2 * vv_])
numt = {r_: sp.Rational(7, 5), vv_: sp.Rational(3, 10), vt_: sp.Rational(2, 5), pp_: sp.Rational(9, 10),
        Gam: sp.Rational(5, 3)}
Vt = sp.Matrix([r_, vv_, vt_, pp_])
E_, ER_ = jac_eig(Ut, Ft, Vt, numt)
hval = mp.mpf(sp.N(hh_g.subs(numt), 40))
cval = mp.mpf(sp.N(csh.subs(numt), 40))
Aexpr = hh_g * Wt * vt_
Aval = mp.mpf(sp.N(Aexpr.subs(numt), 40))
gradA = [mp.mpf(sp.N(sp.diff(Aexpr, x_).subs(numt), 40)) for x_ in (r_, vv_, vt_, pp_)]
n_acoustic = 0
for k in range(4):
    lamk = E_[k]
    if abs(lamk - v_n) < 1e-30:
        continue
    sgn = 1 if lamk > v_n else -1
    dvx_dp = ER_[1, k] / ER_[3, k]
    pred = sgn * (1 - v_n**2) * mp.sqrt(hval**2 + Aval**2 * (1 - cval**2)) / ((hval**2 + Aval**2) * mp.mpf(7) / 5 * cval)
    assert abs(dvx_dp - pred) < 1e-28, (dvx_dp, pred)
    assert abs(sum(gradA[i] * ER_[i, k] for i in range(4))) < 1e-28
    n_acoustic += 1
assert n_acoustic == 2
print("OK  tangential velocity: along the -/+ acoustic family dv_x/dp = -/+ (1-v_x^2) sqrt(h^2 + A^2(1-c_s^2))"
      " / ((h^2+A^2) rho c_s), and A = h W v_t is constant along both")

# ------- rarefaction ODEs (GR06 App. B): the B_y denominator vanishes at the Alfven speeds
xi_ = sp.symbols('xi', real=True)
wt2 = sp.symbols('w', positive=True)
den_By = Bx**2 + 2 * Bx * vB * W**2 * (vx - xi_) + W**4 * (vB**2 - wt2) * (vx - xi_)**2
lAp = vx + Bx * (1 - v2) / (vB + sp.sqrt(wt2))
lAm = vx + Bx * (1 - v2) / (vB - sp.sqrt(wt2))
ok("B_x^2 + 2 B_x eta W^2 (v_x - xi) + W^4 (eta^2 - w)(v_x - xi)^2 = W^4 (eta^2 - w)(lambda_A+ - xi)(lambda_A- - xi)",
   sp.simplify(den_By - W**4 * (vB**2 - wt2) * (lAp - xi_) * (lAm - xi_)))

# ------------------------------------------- 5. HLL flux, two equivalent forms
FL, FR, UL, UR, SL, SR, cmn, cmx = sp.symbols('F_L F_R U_L U_R S_L S_R c_min c_max')
Fhll = (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)
Uhll = (SR * UR - SL * UL + FL - FR) / (SR - SL)
ok("HLL flux, code form (c_min = -S_L, c_max = S_R)",
   Fhll.subs({SL: -cmn, SR: cmx}) - (cmx * FL + cmn * FR - cmx * cmn * (UR - UL)) / (cmx + cmn))
ok("HLL state satisfies F_hll = F_L + S_L (U_hll - U_L)", Fhll - (FL + SL * (Uhll - UL)))
ok("HLL state satisfies F_hll = F_R + S_R (U_hll - U_R)", Fhll - (FR + SR * (Uhll - UR)))

# ----------------------------------- 6. Romero et al. (2005) effective sound speed
hh, Bh = sp.symbols('h Bhat', positive=True)
hhat = hh + Bh**2 * rho                      # h + b^2/rho with b = Bhat rho
chat2 = (hh * cs2 + Bh**2 * rho) / hhat
vA2 = Bh**2 * rho / hhat                     # b^2/(rho hhat)
ok("Romero: chat^2 = c_s^2 + v_A^2 - c_s^2 v_A^2", chat2 - (cs2 + vA2 - cs2 * vA2))

# -------------------------------- 7. relativistic Riemann invariant, ideal gas
c = sp.symbols('c', positive=True)
g1 = Gam - 1
G = sp.log((sp.sqrt(g1) + c) / (sp.sqrt(g1) - c)) / sp.sqrt(g1)
ok("d/dc G(c) = 2/(Gamma-1-c^2)", sp.diff(G, c) - 2 / (g1 - c**2))
K = sp.symbols('K', positive=True)
th_ = K * rho**(Gam - 1)                    # p/rho on the isentrope p = K rho^Gamma
cs2_iso = Gam * th_ / (1 + Gam * th_ / g1)  # Gamma p /(rho h)
c_iso = sp.sqrt(cs2_iso)
ok("c_s drho/rho = 2 dc/(Gamma-1-c^2) on an isentrope",
   sp.simplify(c_iso / rho - 2 * sp.diff(c_iso, rho) / (g1 - cs2_iso)))
ok("c_s^2 = dp/drho / h on the isentrope (definition)",
   sp.simplify(cs2_iso - sp.diff(K * rho**Gam, rho) / (1 + Gam * th_ / g1)))

# ----------------------- 8. numerical hydro shock from the closed-form relations
def hydro_shock(rho_a, p_a, v_a, p_b, Gm, sign):
    """sign=+1 right-moving shock (state a ahead on the right), -1 left-moving."""
    rho_a, p_a, v_a, p_b, Gm = map(mp.mpf, (rho_a, p_a, v_a, p_b, Gm))
    h_a = 1 + Gm / (Gm - 1) * p_a / rho_a
    # Taub adiabat h_b^2 - h_a^2 = (h_a/rho_a + h_b/rho_b)(p_b - p_a), ideal gas
    dp = p_b - p_a
    A = 1 + (Gm - 1) * (p_a - p_b) / (Gm * p_b)
    Bq = -(Gm - 1) * (p_a - p_b) / (Gm * p_b)
    C = h_a * (p_a - p_b) / rho_a - h_a**2
    h_b = (-Bq + mp.sqrt(Bq**2 - 4 * A * C)) / (2 * A)
    rho_b = Gm * p_b / ((Gm - 1) * (h_b - 1))
    j2 = -(p_a - p_b) / (h_a / rho_a - h_b / rho_b)
    j = sign * mp.sqrt(j2)
    W_a = 1 / mp.sqrt(1 - v_a**2)
    Vs = (rho_a**2 * W_a**2 * v_a + j * mp.sqrt(j2 + rho_a**2 * W_a**2 * (1 - v_a**2))) / (rho_a**2 * W_a**2 + j2)
    W_s = 1 / mp.sqrt(1 - Vs**2)
    v_b = (h_a * W_a * v_a + W_s * dp / j) / (h_a * W_a + dp * (W_s * v_a / j + 1 / (rho_a * W_a)))
    return (rho_a, p_a, v_a, h_a), (rho_b, p_b, v_b, h_b), Vs, j


def hydro_UF(rho_, p_, v_, h_):
    W_ = 1 / mp.sqrt(1 - v_**2)
    D_ = rho_ * W_
    S_ = rho_ * h_ * W_**2 * v_
    E_ = rho_ * h_ * W_**2 - p_
    return [D_, S_, E_], [D_ * v_, S_ * v_ + p_, S_]


for sign, args in ((+1, (1.0, 1.0, 0.2, 7.0, 5 / 3.0)), (-1, (0.3, 0.05, -0.5, 2.0, 4 / 3.0))):
    a, bb, Vs, j = hydro_shock(*args[:4], args[4], sign)
    Ua, Fa = hydro_UF(*a)
    Ub, Fb = hydro_UF(*bb)
    res = max(abs((Fa[k] - Fb[k]) - Vs * (Ua[k] - Ub[k])) for k in range(3))
    assert res < mp.mpf('1e-30'), res
    # Lax: fluid on both sides compared with Vs (compressive: flow decelerates through shock)
    print(f"OK  hydro shock from Taub/j^2/V_s/v_b formulas satisfies all jump conditions "
          f"(max residual {mp.nstr(res, 3)}), V_s = {mp.nstr(Vs, 8)}")

# independent check: solve the hydro jump conditions directly (no Taub relation used)
# and evaluate the Taub adiabat on the result
Gh = mp.mpf(4) / 3
ra_, pa_, va_ = mp.mpf('1.0'), mp.mpf('0.5'), mp.mpf('0.1')
pb_ = mp.mpf('2.0')
ha_ = 1 + Gh / (Gh - 1) * pa_ / ra_
Ua_h, Fa_h = hydro_UF(ra_, pa_, va_, ha_)


def hydro_rh(rb, vb, Vs_):
    hb = 1 + Gh / (Gh - 1) * pb_ / rb
    Ub_h, Fb_h = hydro_UF(rb, pb_, vb, hb)
    return [(Fa_h[k] - Fb_h[k]) - Vs_ * (Ua_h[k] - Ub_h[k]) for k in range(3)]


rb_s, vb_s, Vs_s = mp.findroot(hydro_rh, [mp.mpf('2.0'), mp.mpf('0.5'), mp.mpf('0.9')])
hb_s = 1 + Gh / (Gh - 1) * pb_ / rb_s
taub_res = (ha_**2 - hb_s**2) - (ha_ / ra_ + hb_s / rb_s) * (pa_ - pb_)
assert abs(taub_res) < mp.mpf('1e-30') and rb_s > ra_, (taub_res, rb_s)
print(f"OK  Taub adiabat [[h^2]] = (h_a/rho_a + h_b/rho_b)[[p]] on a shock solved from the jump "
      f"conditions alone (|residual| = {mp.nstr(abs(taub_res), 3)}, V_s = {mp.nstr(Vs_s, 8)})")

# -------------------- 9. MHD shock invariants on a numerically built fast shock
def mhd_UF(st, Bn, Gm):
    rho_, P_, vx_, vy_, vz_, By_, Bz_ = st
    vv = mp.matrix([vx_, vy_, vz_])
    BB = mp.matrix([Bn, By_, Bz_])
    v2_ = vx_**2 + vy_**2 + vz_**2
    W_ = 1 / mp.sqrt(1 - v2_)
    vB_ = vx_ * Bn + vy_ * By_ + vz_ * Bz_
    B2_ = Bn**2 + By_**2 + Bz_**2
    b2_ = B2_ / W_**2 + vB_**2
    pg = P_ - b2_ / 2
    h_ = 1 + Gm / (Gm - 1) * pg / rho_
    w_ = rho_ * h_ + b2_
    b0 = W_ * vB_
    bv = [BB[i] / W_ + b0 * vv[i] for i in range(3)]
    D_ = rho_ * W_
    U = [D_] + [w_ * W_**2 * vv[i] - b0 * bv[i] for i in range(3)] + [w_ * W_**2 - P_ - b0**2, By_, Bz_]
    F = [D_ * vx_] + [w_ * W_**2 * vx_ * vv[i] + (P_ if i == 0 else 0) - bv[0] * bv[i] for i in range(3)] \
        + [w_ * W_**2 * vx_ - b0 * bv[0], By_ * vx_ - Bn * vy_, Bz_ * vx_ - Bn * vz_]
    aux = dict(W=W_, b0=b0, bv=bv, b2=b2_, h=h_, w=w_, pg=pg, rho=rho_, P=P_, vx=vx_)
    return U, F, aux


Gm = mp.mpf(5) / 3
Bn = mp.mpf('1.2')
ahead = [mp.mpf('1.0'), mp.mpf('2.0'), mp.mpf('0.1'), mp.mpf('0.2'), mp.mpf('0.05'),
         mp.mpf('0.9'), mp.mpf('0.3')]
Pb = mp.mpf('3.0')      # total pressure behind; gas pressure ahead is 0.84
Ua, Fa, auxa = mhd_UF(ahead, Bn, Gm)


def rh(*x):
    rb, vxb, vyb, vzb, Byb, Bzb, Vs = x
    st = [rb, Pb, vxb, vyb, vzb, Byb, Bzb]
    Ub, Fb, _ = mhd_UF(st, Bn, Gm)
    return [(Fa[k] - Fb[k]) - Vs * (Ua[k] - Ub[k]) for k in range(7)]


# start from a point found by continuation in P_b from the ahead state
guess = [1.30, 0.29, 0.13, 0.03, 1.35, 0.44, 0.87]
sol = mp.findroot(rh, guess)
rb, vxb, vyb, vzb, Byb, Bzb, Vs = sol
behind = [rb, Pb, vxb, vyb, vzb, Byb, Bzb]
assert max(abs(r) for r in rh(*sol)) < mp.mpf('1e-30')
Ub, Fb, auxb = mhd_UF(behind, Bn, Gm)
Ws = 1 / mp.sqrt(1 - Vs**2)
n_low = [-Vs * Ws, Ws, 0, 0]              # n_mu = W_s(-V_s, 1, 0, 0)


def J_and_Bn(aux, st):
    W_ = aux['W']
    uvec = [W_, W_ * st[2], W_ * st[3], W_ * st[4]]
    bvec = [aux['b0']] + aux['bv']
    J_ = st[0] * sum(uvec[k] * n_low[k] for k in range(4))
    Bn_ = sum(bvec[k] * n_low[k] for k in range(4))
    return J_, Bn_


Ja, Bna = J_and_Bn(auxa, ahead)
Jb, Bnb = J_and_Bn(auxb, behind)
assert abs(Ja - Jb) < 1e-30
J = Ja
Ha = Bna**2 / J**2 - auxa['b2'] / auxa['rho']**2
Hb = Bnb**2 / J**2 - auxb['b2'] / auxb['rho']**2
hga, hgb = auxa['h'], auxb['h']
ra, rb_ = auxa['rho'], auxb['rho']
checks = {
    "[[H]] = 0, H = B_n^2/J^2 - b^2/rho^2": Ha - Hb,
    "[[h_g B_n]] = 0": hga * Bna - hgb * Bnb,
    "[[P]] + J^2 [[h_g/rho]] = 0  (GR06 eq. 4.26)": (auxa['P'] - auxb['P']) + J**2 * (hga / ra - hgb / rb_),
    "Lichnerowicz adiabat (GR06 eq. 4.27)":
        (hga**2 - hgb**2) - (hga / ra + hgb / rb_) * (auxa['P'] - auxb['P'])
        - Ha * (auxa['b2'] - auxb['b2']) + 2 * (auxa['b2'] * hga / ra - auxb['b2'] * hgb / rb_)
        - 2 * J**2 * Ha * (hga / ra - hgb / rb_),
}
for k_, val in checks.items():
    assert abs(val) < mp.mpf('1e-25'), (k_, val)
    print(f"OK  {k_} on a numerical fast shock (|residual| = {mp.nstr(abs(val), 3)})")
print(f"    (shock: V_s = {mp.nstr(Vs, 10)}, rho_b = {mp.nstr(rb, 8)}, compression {mp.nstr(rb/ahead[0], 6)})")


def speeds(st, Bn_, Gm_):
    """Four magnetosonic speeds (sorted) and the two Alfven speeds (sorted)."""
    _, _, aux = mhd_UF(st, Bn_, Gm_)
    W_, b0, bv, w_, pg = aux['W'], aux['b0'], aux['bv'], aux['w'], aux['pg']
    cs2_ = Gm_ * pg / (aux['rho'] * aux['h'])
    rh_ = w_ - aux['b2']
    vx_ = st[2]
    # quartic in lambda: rh(1-cs2) a^4 - (1-l^2)[(b2 + rh cs2) a^2 - cs2 (b^x - l b^0)^2]
    # polynomial arithmetic on coefficient lists (lowest order first), all in mpmath
    def pmul(p1, p2):
        out = [mp.mpf(0)] * (len(p1) + len(p2) - 1)
        for i1, c1 in enumerate(p1):
            for i2, c2 in enumerate(p2):
                out[i1 + i2] += c1 * c2
        return out

    def padd(p1, p2, s=1):
        n = max(len(p1), len(p2))
        p1 = p1 + [mp.mpf(0)] * (n - len(p1))
        p2 = p2 + [mp.mpf(0)] * (n - len(p2))
        return [p1[k] + s * p2[k] for k in range(n)]

    a1p = [-W_ * vx_, W_]                       # a = W (lambda - v_x)
    a2p = pmul(a1p, a1p)
    a4p = pmul(a2p, a2p)
    Bp = [bv[0], -b0]                           # b^x - lambda b^0
    inner = padd([c_ * (aux['b2'] + rh_ * cs2_) for c_ in a2p], [c_ * cs2_ for c_ in pmul(Bp, Bp)], -1)
    poly = padd([c_ * rh_ * (1 - cs2_) for c_ in a4p], pmul([1, 0, -1], inner), -1)
    roots = sorted(mp.re(r) for r in mp.polyroots(list(reversed(poly)), maxsteps=200, extraprec=200))
    sw_ = mp.sqrt(w_)
    vB_ = Bn_ * st[2] + st[5] * st[3] + st[6] * st[4]
    v2_ = st[2]**2 + st[3]**2 + st[4]**2
    lA = sorted([vx_ + Bn_ * (1 - v2_) / (vB_ + sw_), vx_ + Bn_ * (1 - v2_) / (vB_ - sw_)])
    return roots, lA


Bn_shock = mp.mpf('1.2')
ms_a, lA_a = speeds(ahead, Bn_shock, Gm)
ms_b, lA_b = speeds(behind, Bn_shock, Gm)
assert ms_a[3] < Vs < ms_b[3], (ms_a, ms_b, Vs)       # Lax, right-going fast family
assert lA_b[1] < Vs                                    # behind: super-Alfvenic (evolutionary)
assert ms_a[0] < lA_a[0] < ms_a[1] < ahead[2] < ms_a[2] < lA_a[1] < ms_a[3]   # ordering
print(f"OK  the shock is an evolutionary right-going fast shock: lambda_f+(a) = {mp.nstr(ms_a[3], 8)}"
      f" < V_s < lambda_f+(b) = {mp.nstr(ms_b[3], 8)}, lambda_A+(b) = {mp.nstr(lA_b[1], 8)} < V_s;"
      f" ahead state's seven speeds ordered")

# ---------------- 10. a numerical Alfven (rotational) discontinuity, built from sigma
st_a = [mp.mpf('1.0'), mp.mpf('2.0'), mp.mpf('0.2'), mp.mpf('0.3'), mp.mpf('-0.1'),
        mp.mpf('0.7'), mp.mpf('0.4')]
Bn = mp.mpf('0.8')
Ua, Fa, auxa = mhd_UF(st_a, Bn, Gm)
W_a = auxa['W']
u_a = [W_a, W_a * st_a[2], W_a * st_a[3], W_a * st_a[4]]
b_a = [auxa['b0']] + auxa['bv']
for eta_s in (+1, -1):
    eta = eta_s * mp.sqrt(auxa['w'])
    sig = [b_a[k] + eta * u_a[k] for k in range(4)]   # sigma^mu = b^mu + eta u^mu
    lamA = sig[1] / sig[0]

    def eqs(ux, uy, uz_fixed):
        u0 = mp.sqrt(1 + ux**2 + uy**2 + uz_fixed**2)
        ub = [u0, ux, uy, uz_fixed]
        bb = [sig[k] - eta * ub[k] for k in range(4)]
        udots = -ub[0] * sig[0] + ub[1] * sig[1] + ub[2] * sig[2] + ub[3] * sig[3]
        Bx_b = ub[0] * bb[1] - ub[1] * bb[0]          # B^x = u^0 b^x - u^x b^0
        return [udots + eta, Bx_b - Bn]

    # the family is one-parametric: fix u^z behind and solve for (u^x, u^y)
    for uz in (u_a[3] + mp.mpf('0.25'), u_a[3] + mp.mpf('0.1'), u_a[3] - mp.mpf('0.1')):
        try:
            ux, uy = mp.findroot(lambda x1, x2: eqs(x1, x2, uz), [u_a[1], u_a[2]])
            break
        except ValueError:
            continue
    else:
        raise AssertionError("no rotational discontinuity found")
    u0 = mp.sqrt(1 + ux**2 + uy**2 + uz**2)
    ub = [u0, ux, uy, uz]
    bb = [sig[k] - eta * ub[k] for k in range(4)]
    Bb = [ub[0] * bb[i] - ub[i] * bb[0] for i in (1, 2, 3)]
    st_b = [st_a[0], st_a[1], ub[1] / u0, ub[2] / u0, ub[3] / u0, Bb[1], Bb[2]]
    Ub, Fb, auxb = mhd_UF(st_b, Bn, Gm)
    res = max(abs((Fa[k] - Fb[k]) - lamA * (Ua[k] - Ub[k])) for k in range(7))
    assert res < mp.mpf('1e-28'), res
    assert abs(auxb['b2'] - auxa['b2']) < 1e-28 and abs(auxb['pg'] - auxa['pg']) < 1e-28
    lam_formula = st_a[2] + Bn * (1 - (st_a[2]**2 + st_a[3]**2 + st_a[4]**2)) / (
        (Bn * st_a[2] + st_a[5] * st_a[3] + st_a[6] * st_a[4]) + eta)
    assert abs(lam_formula - lamA) < 1e-28
    dvx = st_b[2] - st_a[2]
    Bt_a = mp.sqrt(st_a[5]**2 + st_a[6]**2)
    Bt_b = mp.sqrt(st_b[5]**2 + st_b[6]**2)
    print(f"OK  Alfven discontinuity (eta = {'+' if eta_s > 0 else '-'}sqrt(w)): RH residual {mp.nstr(res, 3)}, "
          f"speed {mp.nstr(lamA, 8)}; rho, p, b^2 continuous; [[v_x]] = {mp.nstr(dvx, 4)}, "
          f"|B_t| {mp.nstr(Bt_a, 5)} -> {mp.nstr(Bt_b, 5)}")

# ------------- 11. covariant characteristic matrix and eigenvectors (Anton et al. 2010)
# Variables (delta u^nu, delta b^nu, delta p, delta s); phi_mu = (-lambda, 1, 0, 0).
# The (b,b) block is taken as -a delta^mu_nu (their eq. 22; their printed eq. 32 shows +a),
# and l^mu = phi^mu + (rho h - b^2/c_s^2) a u^mu/(rho h) + calB b^mu/(rho h) (text below
# their eq. 32).  With these, det = E a^2 A^2 N4 holds identically and eqs. (46)-(50) are
# exact null vectors.
gdiag = [-1, 1, 1, 1]


def levi(*idx):
    if len(set(idx)) < 4:
        return 0
    s_ = 1
    for i in range(4):
        for j in range(i + 1, 4):
            if idx[i] > idx[j]:
                s_ = -s_
    return s_


rho_c, pg_c = mp.mpf('1.3'), mp.mpf('0.7')
v_c = [mp.mpf('0.31'), mp.mpf('-0.22'), mp.mpf('0.17')]
B_c = [mp.mpf('0.9'), mp.mpf('0.6'), mp.mpf('-0.45')]
W_c = 1 / mp.sqrt(1 - sum(x * x for x in v_c))
vB_c = sum(v_c[i] * B_c[i] for i in range(3))
u_c = [W_c] + [W_c * x for x in v_c]
b_c = [W_c * vB_c] + [B_c[i] / W_c + W_c * vB_c * v_c[i] for i in range(3)]
b2_c = sum(gdiag[i] * b_c[i]**2 for i in range(4))
rh_c = rho_c + Gm / (Gm - 1) * pg_c
cs2_c = Gm * pg_c / rh_c
E_c = rh_c + b2_c


def charmat(lam_, bb_sign=-1, with_calB=True):
    """The contracted matrix; bb_sign and with_calB select the (wrong) printed variants."""
    phl = [-lam_, 1, 0, 0]
    phu = [lam_, 1, 0, 0]
    a_c = sum(u_c[i] * phl[i] for i in range(4))
    cB = sum(b_c[i] * phl[i] for i in range(4))
    G_c = sum(phu[i] * phl[i] for i in range(4))
    bl = [gdiag[i] * b_c[i] for i in range(4)]
    l_c = [phu[m] + (rh_c - b2_c / cs2_c) * a_c * u_c[m] / rh_c
           + (cB * b_c[m] / rh_c if with_calB else 0) for m in range(4)]
    f_c = [(a_c * b_c[m] / cs2_c - cB * u_c[m]) / rh_c for m in range(4)]
    M = mp.zeros(10, 10)
    for m in range(4):
        for n in range(4):
            dl = 1 if m == n else 0
            M[m, n] = E_c * a_c * dl
            M[m, 4 + n] = (phu[m] + 2 * a_c * u_c[m]) * bl[n] - cB * dl
            M[4 + m, n] = cB * dl
            M[4 + m, 4 + n] = bb_sign * a_c * dl
        M[m, 8] = l_c[m]
        M[4 + m, 8] = f_c[m]
        M[8, m] = rh_c * phl[m]
    M[8, 8] = a_c / cs2_c
    M[9, 9] = a_c
    A_c = E_c * a_c**2 - cB**2
    N4 = rh_c * (1 / cs2_c - 1) * a_c**4 - (rh_c + b2_c / cs2_c) * a_c**2 * G_c + cB**2 * G_c
    return M, a_c, cB, G_c, A_c, N4, phu, phl, f_c


for lam_t in (mp.mpf('0.13'), mp.mpf('-0.4'), mp.mpf('0.77')):
    M, a_c, cB, G_c, A_c, N4, *_ = charmat(lam_t)
    assert abs(mp.det(M) / (E_c * a_c**2 * A_c**2 * N4) - 1) < mp.mpf('1e-35')
print("OK  det(A^alpha phi_alpha) = E a^2 A^2 N_4 (Anton et al. 2010, eq. 33), three lambdas")
uL_ = [gdiag[i] * u_c[i] for i in range(4)]
bL_ = [gdiag[i] * b_c[i] for i in range(4)]
for s_ in (+1, -1):
    lamA = (b_c[1] + s_ * mp.sqrt(E_c) * u_c[1]) / (b_c[0] + s_ * mp.sqrt(E_c) * u_c[0])
    M, a_c, cB, G_c, A_c, N4, phu, phl, f_c = charmat(lamA)
    epsv = [sum(levi(al, be, ga, de) * uL_[be] * phl[ga] * bL_[de]
                for be in range(4) for ga in range(4) for de in range(4)) for al in range(4)]
    r = mp.matrix([a_c * x for x in epsv] + [cB * x for x in epsv] + [0, 0])
    assert mp.norm(M * r) / mp.norm(r) < mp.mpf('1e-35')
    assert abs(cB / a_c + s_ * mp.sqrt(E_c)) < mp.mpf('1e-35')      # calB/a = -/+ sqrt(E)
print("OK  Alfven eigenvectors (eq. 47) are null vectors; calB/a = -/+ sqrt(E) (eq. 51)")
for guess_l in (-0.9, -0.3, 0.4, 0.9):
    lam_m = mp.findroot(lambda x_: charmat(x_)[5], mp.mpf(guess_l))
    M, a_c, cB, G_c, A_c, N4, phu, phl, f_c = charmat(lam_m)
    d_c = [((cs2_c - 1) / cs2_c) * ((a_c**4 / G_c) * (phu[m] + a_c * u_c[m]) - (a_c**2 * cB / rh_c) * b_c[m])
           for m in range(4)]
    r = mp.matrix([a_c * x for x in d_c] + [cB * d_c[m] + a_c * A_c * f_c[m] for m in range(4)]
                  + [a_c**2 * A_c, 0])
    assert mp.norm(M * r) / mp.norm(r) < mp.mpf('1e-35')
print("OK  magnetosonic eigenvectors (eqs. 48-50) are null vectors at all four roots of N_4")


def ms_vector(lam_m, M_ok):
    """The eq. (48)-(50) eigenvector at a root of N_4 (independent of the matrix variant)."""
    _, a_c, cB, G_c, A_c, N4, phu, phl, f_c = M_ok
    d_c = [((cs2_c - 1) / cs2_c) * ((a_c**4 / G_c) * (phu[m] + a_c * u_c[m]) - (a_c**2 * cB / rh_c) * b_c[m])
           for m in range(4)]
    return mp.matrix([a_c * x for x in d_c] + [cB * d_c[m] + a_c * A_c * f_c[m] for m in range(4)]
                     + [a_c**2 * A_c, 0])


# the two printed variants must FAIL both tests: +a delta in the (b,b) block (Anton et al. eq. 32),
# and -a delta without the calB b^mu/(rho h) term in l^mu (their eq. 26)
for label, kw in (("+a delta^mu_nu in the (b,b) block", dict(bb_sign=+1)),
                  ("l^mu without calB b^mu/(rho h)", dict(with_calB=False))):
    for lam_t in (mp.mpf('0.13'), mp.mpf('-0.4'), mp.mpf('0.77')):
        M, a_c, cB, G_c, A_c, N4, *_ = charmat(lam_t, **kw)
        assert abs(mp.det(M) / (E_c * a_c**2 * A_c**2 * N4) - 1) > mp.mpf('1e-6'), label
    for guess_l in (-0.9, -0.3, 0.4, 0.9):
        lam_m = mp.findroot(lambda x_: charmat(x_)[5], mp.mpf(guess_l))
        r = ms_vector(lam_m, charmat(lam_m))
        M_bad = charmat(lam_m, **kw)[0]
        assert mp.norm(M_bad * r) / mp.norm(r) > mp.mpf('1e-6'), label
    print(f"OK  with {label}: the determinant identity fails at three lambdas and the "
          f"magnetosonic eigenvectors are not null vectors (negative test)")
print("all checks passed")
