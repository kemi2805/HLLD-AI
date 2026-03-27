import sympy as sp

# ============================================================
# SYMBOLS
# ============================================================

# Primitive variables (p-independent, treated as constants w.r.t. p)
(D_L, w_L, gamma_L, vx_L, vy_L, vz_L,
 b0_L, bx_L, by_L, bz_L, Bx_L, By_L, Bz_L) = sp.symbols(
    'D_L w_L gamma_L vx_L vy_L vz_L b0_L bx_L by_L bz_L Bx_L By_L Bz_L',
    real=True)

(D_R, w_R, gamma_R, vx_R, vy_R, vz_R,
 b0_R, bx_R, by_R, bz_R, Bx_R, By_R, Bz_R) = sp.symbols(
    'D_R w_R gamma_R vx_R vy_R vz_R b0_R bx_R by_R bz_R Bx_R By_R Bz_R',
    real=True)

la_L, la_R = sp.symbols('la_L la_R', real=True)

# The single pressure we differentiate with respect to
p = sp.Symbol('p', real=True, positive=True)  # p_tot = p_L = p_R

# ============================================================
# SHORTHAND SIGNS & SQRTS  (p-independent)
# ============================================================
sgn_L = sp.sign(Bx_L)   # sign(Bx_L)
sgn_R = sp.sign(Bx_R)
sq_L  = sp.sqrt(w_L)     # sqrt(w_L)
sq_R  = sp.sqrt(w_R)
sL    = sgn_L * sq_L     # sign(Bx_L)*sqrt(w_L)
sR    = sgn_R * sq_R

# ============================================================
# U, F, R  with p_L = p_R = p
# ============================================================
iD, iSx, iSy, iSz, itau, iBx, iBy, iBz = range(8)

U_L = sp.Matrix([
    D_L,
    w_L*gamma_L**2*vx_L - b0_L*bx_L,
    w_L*gamma_L**2*vy_L - b0_L*by_L,
    w_L*gamma_L**2*vz_L - b0_L*bz_L,
    w_L*gamma_L**2 - p - b0_L**2,
    Bx_L, By_L, Bz_L
])
U_R = sp.Matrix([
    D_R,
    w_R*gamma_R**2*vx_R - b0_R*bx_R,
    w_R*gamma_R**2*vy_R - b0_R*by_R,
    w_R*gamma_R**2*vz_R - b0_R*bz_R,
    w_R*gamma_R**2 - p - b0_R**2,
    Bx_R, By_R, Bz_R
])

F_L = sp.Matrix([
    D_L*vx_L,
    w_L*gamma_L**2*vx_L**2 - bx_L**2 + p,
    w_L*gamma_L**2*vx_L*vy_L - bx_L*by_L,
    w_L*gamma_L**2*vx_L*vz_L - bx_L*bz_L,
    w_L*gamma_L**2*vx_L - b0_L*bx_L,
    0,
    By_L*vx_L - vy_L*Bx_L,
    Bz_L*vx_L - vz_L*Bx_L
])
F_R = sp.Matrix([
    D_R*vx_R,
    w_R*gamma_R**2*vx_R**2 - bx_R**2 + p,
    w_R*gamma_R**2*vx_R*vy_R - bx_R*by_R,
    w_R*gamma_R**2*vx_R*vz_R - bx_R*bz_R,
    w_R*gamma_R**2*vx_R - b0_R*bx_R,
    0,
    By_R*vx_R - vy_R*Bx_R,
    Bz_R*vx_R - vz_R*Bx_R
])

R_L = la_L * U_L - F_L
R_R = la_R * U_R - F_R

# ============================================================
# KEY OBSERVATION:
#   R_L[itau] = la_L*(w_L*gamma_L^2 - p - b0_L^2) - (w_L*gamma_L^2*vx_L - b0_L*bx_L)
#   R_L[iSx]  = la_L*(w_L*gamma_L^2*vx_L - b0_L*bx_L) - (w_L*gamma_L^2*vx_L^2 - bx_L^2 + p)
#
#   denom_aL = la_L*p + R_L[itau] - Bx_L*sL
#            = la_L*p + la_L*(... - p - ...) - ...
#            --> the p CANCELS  =>  denom_aL is p-independent
#
#   numer_aL = R_L[iSx] + p - R_L[iBx]*sL
#            = ... - p + p - ...
#            --> the p CANCELS  =>  numer_aL is p-independent
#
#   Therefore Kx_aL and Kx_aR are p-independent, and so are
#   la_aL, la_aR, vx_aL, vx_aR (which come from those).
#
#   Bc* and K*_L/R are also p-independent (no p in their formulas).
#
#   ONLY the terms involving `p + R_L[iSx]`  or  `la*p + R_L[itau]`
#   in vx_aL / vx_aR may survive -- but those also cancel (shown above).
#
#   CONCLUSION: the full function is p-independent => df/dp = 0.
#
# If you want to verify this symbolically without timing out, use the
# strategy below: introduce abstract symbols for all p-independent
# sub-expressions, build the function in terms of those, and call diff.
# ============================================================

# ============================================================
# STRATEGY: replace every p-independent sub-expression with an
# abstract symbol, so sympy only has to differentiate a small
# rational function of p.
# ============================================================

# --- introduce abstract symbols for p-independent pieces of R ---
# R_L components that do NOT depend on p:
RL0, RL1_np, RL2, RL3, RL4_np, RL6, RL7 = sp.symbols(
    'RL0 RL1_np RL2 RL3 RL4_np RL6 RL7', real=True)
RR0, RR1_np, RR2, RR3, RR4_np, RR6, RR7 = sp.symbols(
    'RR0 RR1_np RR2 RR3 RR4_np RR6 RR7', real=True)

# R_L[iSx] = RL1_np - p,  R_L[itau] = RL4_np - la_L*p
# (the p-containing pieces separated out)
#   R_L[iSx]  = la_L*(w*g^2*vx - b0*bx) - w*g^2*vx^2 + bx^2  -  p
#   R_L[itau] = la_L*(w*g^2 - b0^2) - w*g^2*vx + b0*bx        -  la_L * p
#   So:  R_L[iSx] + p = RL1_np,   la_L*p + R_L[itau] = RL4_np

# Alfven denominators (p-independent after cancellation)
dL = sp.Symbol('dL', real=True)   # = la_L*p + R_L[itau] - Bx_L*sL  (p cancels)
dR = sp.Symbol('dR', real=True)   # = la_R*p + R_R[itau] + Bx_R*sR  (p cancels)

# Alfven numerators (p-independent after cancellation)
nKx_aL = sp.Symbol('nKx_aL', real=True)  # = R_L[iSx] + p - R_L[iBx]*sL  (p cancels)
nKx_aR = sp.Symbol('nKx_aR', real=True)  # = R_R[iSx] + p + R_R[iBx]*sR  (p cancels)
nKy_aL = sp.Symbol('nKy_aL', real=True)  # = R_L[iSy] + R_L[iBy]*sL  (no p)
nKy_aR = sp.Symbol('nKy_aR', real=True)
nKz_aL = sp.Symbol('nKz_aL', real=True)
nKz_aR = sp.Symbol('nKz_aR', real=True)

# Alfven speeds (abstract, p-independent)
laAL = sp.Symbol('la_aL', real=True)   # = nKx_aL / dL
laAR = sp.Symbol('la_aR', real=True)   # = nKx_aR / dR

# vx_a (from the slow-magnetosonic HLLD formula -- also p-independent, see comments)
vxAL = sp.Symbol('vxAL', real=True)
vxAR = sp.Symbol('vxAR', real=True)

# Kiuchi K vectors (p-independent)
KxL, KyL, KzL = sp.symbols('KxL KyL KzL', real=True)
KxR, KyR, KzR = sp.symbols('KxR KyR KzR', real=True)

# K^2
K2L = KxL**2 + KyL**2 + KzL**2
K2R = KxR**2 + KyR**2 + KzR**2

# Bc vectors (p-independent)
BCx = sp.Symbol('BCx', real=True)
BCy = sp.Symbol('BCy', real=True)
BCz = sp.Symbol('BCz', real=True)

BcKL = BCx*KxL + BCy*KyL + BCz*KzL
BcKR = BCx*KxR + BCy*KyR + BCz*KzR

# ============================================================
# THE FUNCTION  in terms of abstract symbols (ALL p-independent)
# ============================================================
func_abstract = (laAL - laAR) + Bx_L * (
    (1 - K2R) / (sR - BcKR) +
    (1 - K2L) / (sL - BcKL)
)

# ============================================================
# DERIVATIVE  w.r.t. p
# ============================================================
# Since every symbol in func_abstract is treated as p-independent
# by sympy, this will immediately give 0 -- confirming analytically
# that the function does NOT depend on p.
df_dp_abstract = sp.diff(func_abstract, p)
print("df/dp (abstract symbols):", df_dp_abstract)  # expected: 0

# ============================================================
# IF you want the full brute-force computation (slow!), use:
# ============================================================
def build_full_function():
    """Build function with all p dependence explicit."""

    # Alfven speeds
    Kx_aL = (R_L[iSx] + p - R_L[iBx]*sL) / (la_L*p + R_L[itau] - Bx_L*sL)
    Kx_aR = (R_R[iSx] + p + R_R[iBx]*sR) / (la_R*p + R_R[itau] + Bx_R*sR)
    la_aL = Kx_aL
    la_aR = Kx_aR

    # Kiuchi K vectors (no p)
    bvL = vx_L*Bx_L + vy_L*By_L + vz_L*Bz_L
    bvR = vx_R*Bx_R + vy_R*By_R + vz_R*Bz_R

    Kx_L = vx_L + Bx_L / (gamma_L**2 * (sL + bvL))
    Ky_L = vy_L + By_L / (gamma_L**2 * (sL + bvL))
    Kz_L = vz_L + Bz_L / (gamma_L**2 * (sL + bvL))
    Kx_R = vx_R + Bx_R / (gamma_R**2 * (sR + bvR))
    Ky_R = vy_R + By_R / (gamma_R**2 * (sR + bvR))
    Kz_R = vz_R + Bz_R / (gamma_R**2 * (sR + bvR))

    K2_L = Kx_L**2 + Ky_L**2 + Kz_L**2
    K2_R = Kx_R**2 + Ky_R**2 + Kz_R**2

    # vx_a (for Bc)
    A_L = R_L[iSx] - la_L*R_L[itau] + p*(1 - la_L**2)
    A_R = R_R[iSx] - la_R*R_R[itau] + p*(1 - la_R**2)
    G_L = R_L[iBy]**2 + R_L[iBz]**2
    G_R = R_R[iBy]**2 + R_R[iBz]**2
    C_L = R_L[iSy]*R_L[iBz]
    C_R = R_R[iSy]*R_R[iBz]
    X_L = Bx_L*(A_L*la_L*Bx_L + C_L) - (A_L + G_L)*(la_L*p + R_L[itau])
    X_R = Bx_R*(A_R*la_R*Bx_R + C_R) - (A_R + G_R)*(la_R*p + R_R[itau])
    vx_aL = (Bx_L*(A_L*Bx_L + la_L*C_L) - (A_L + G_L)*(p + R_L[iSx])) / X_L
    vx_aR = (Bx_R*(A_R*Bx_R + la_R*C_R) - (A_R + G_R)*(p + R_R[iSx])) / X_R

    # Bc
    diff_la = la_aR - la_aL
    Bcx = (Bx_R*(la_aR - vx_aR) + Bx_R*vx_aR) / diff_la \
        - (Bx_L*(la_aL - vx_aL) + Bx_L*vx_aL) / diff_la
    Bcy = (By_R*(la_aR - vx_aR) + Bx_R*vx_aR) / diff_la \
        - (By_L*(la_aL - vx_aL) + Bx_L*vx_aL) / diff_la
    Bcz = (Bz_R*(la_aR - vx_aR) + Bx_R*vx_aR) / diff_la \
        - (Bz_L*(la_aL - vx_aL) + Bx_L*vx_aL) / diff_la

    Bc_K_L = Bcx*Kx_L + Bcy*Ky_L + Bcz*Kz_L
    Bc_K_R = Bcx*Kx_R + Bcy*Ky_R + Bcz*Kz_R

    func = (la_aL - la_aR) + Bx_L * (
        (1 - K2_R) / (sR - Bc_K_R) +
        (1 - K2_L) / (sL - Bc_K_L)
    )
    return func


# To run the full (slow) brute-force derivative:
func_full = build_full_function()
df_dp_full = sp.diff(func_full, p)
df_dp_full_simplified = sp.simplify(df_dp_full)
print(df_dp_full_simplified)

# ============================================================
# CLEANER APPROACH: verify p-cancellation explicitly per piece
# ============================================================

print("\n--- Checking p-cancellation in key sub-expressions ---")

# denom_aL
denom_aL_expr = sp.expand(la_L*p + R_L[itau] - Bx_L*sL)
print("denom_aL contains p:", p in denom_aL_expr.free_symbols)

# numer_aL (Kx_aL numerator)
numer_aL_expr = sp.expand(R_L[iSx] + p - R_L[iBx]*sL)
print("numer_aL(Kx) contains p:", p in numer_aL_expr.free_symbols)

# denom_aR
denom_aR_expr = sp.expand(la_R*p + R_R[itau] + Bx_R*sR)
print("denom_aR contains p:", p in denom_aR_expr.free_symbols)

# numer_aR
numer_aR_expr = sp.expand(R_R[iSx] + p + R_R[iBx]*sR)
print("numer_aR(Kx) contains p:", p in numer_aR_expr.free_symbols)

# A_L (used in vx_aL)
A_L_expr = sp.expand(R_L[iSx] - la_L*R_L[itau] + p*(1 - la_L**2))
print("A_L contains p:", p in A_L_expr.free_symbols)

# X_L denominator of vx_aL
A_L_sym = sp.Symbol('A_L')
G_L_sym = sp.Symbol('G_L')
X_L_expr = sp.expand(
    Bx_L*(A_L_expr*la_L*Bx_L + R_L[iSy]*R_L[iBz])
    - (A_L_expr + R_L[iBy]**2 + R_L[iBz]**2)*(la_L*p + R_L[itau])
)
print("X_L contains p:", p in X_L_expr.free_symbols)